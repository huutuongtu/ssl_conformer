import torch
import json
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
import einops
import numpy as np
import math
from typing import Dict, Optional, Tuple, List

class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(-2, -1)

class Fp32LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float = 0.0,
            mode: str = "default",
            conv_bias: bool = False,
            conv_type: str = "default"
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
                n_in,
                n_out,
                k,
                stride,
                is_layer_norm=False,
                is_group_norm=False,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                           is_layer_norm and is_group_norm
                   ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.conv_type = conv_type
        if self.conv_type == "default":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl

                self.conv_layers.append(
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                    )
                )
                in_d = dim
        elif self.conv_type == "conv2d":
            in_d = 1
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl

                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride)
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
        elif self.conv_type == "custom":
            in_d = 1
            idim = 80
            self.conv_layers = nn.ModuleList()
            for i, cl in enumerate(conv_layers):
                assert len(cl) == 3
                (dim, k, stride) = cl
                self.conv_layers.append(
                    torch.nn.Conv2d(in_d, dim, k, stride, padding=1)
                )
                self.conv_layers.append(
                    torch.nn.LayerNorm([dim, idim])
                )
                self.conv_layers.append(torch.nn.ReLU())
                in_d = dim
                if (i + 1) % 2 == 0:
                    self.conv_layers.append(
                        torch.nn.MaxPool2d(2, stride=2, ceil_mode=True)
                    )
                    idim = int(math.ceil(idim / 2))
        else:
            pass

    def forward(self, x, mask=None):

        # BxT -> BxCxT
        x = x.unsqueeze(1)
        if self.conv_type == "custom":
            for conv in self.conv_layers:
                if isinstance(conv, nn.LayerNorm):
                    x = x.transpose(1, 2)
                    x = conv(x).transpose(1, 2)
                else:
                    x = conv(x)
            x = x.transpose(2, 3).contiguous()
            x = x.view(x.size(0), -1, x.size(-1))
        else:
            for conv in self.conv_layers:
                x = conv(x)
            if self.conv_type == "conv2d":
                b, c, t, f = x.size()
                x = x.transpose(2, 3).contiguous().view(b, c * f, t)
        return x


class ASP(nn.Module): 
  
    def __init__(self, input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def attention(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        if (len(attention_weights.shape)==1):
            attention_weights = attention_weights.unsqueeze(0)
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized
      
    def weighted_sd(self,inputs,attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance
    
    # from input, calculate self-attention weights (after softmax) (shape leng x length), 
    # then matmul inputs with aw (shape length x features)
    # then get mean and variances of this features and concat, (shape 1 x features x 2)
    # this features use for classifier
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling
      
    def forward(self, inputs): #batch x length x features => batch x (features x 2)
        attn_weights = self.attention(inputs)
        stat_pool_out = self.stat_attn_pool(inputs, attn_weights)
        return stat_pool_out


"""
From pytorch implementation
"""

__all__ = ["Conformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)
        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:

        residual = input
        x = self.ffn1(input)
        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)

        residual = x
        x = self.ffn2(x)
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        return x


class Conformer(torch.nn.Module):


    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, key_padding_mask=None)
        return x.transpose(0, 1)


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())