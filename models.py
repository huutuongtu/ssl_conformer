import torch
import json
import torch.nn as nn
from torch.nn import functional as F
from modules import ASP, Conformer, ConvFeatureExtractionModel
from torch.nn import LayerNorm
import logging


logger = logging.getLogger(__name__)


class SSL_SER_Config:
    def __init__(self, cfg=None):
        self.extractor_mode: str = "default"     # mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with normalize=True)
        self.encoder_layers: int = 12     # num encoder layers in the transformer

        self.encoder_embed_dim: int = 768     # encoder embedding dimension
        self.encoder_ffn_embed_dim: int = 3072     # encoder embedding dimension for FFN
        self.encoder_attention_heads: int = 12     # num encoder attention heads
        self.activation_fn: str = "gelu"     # activation function to use

        self.layer_norm_first: bool = False     # apply layernorm first in the transformer
        self.conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2"     # string describing convolutional feature extraction layers in form of a python list that contains [(dim, kernel_size, stride), ...]
        self.conv_bias: bool = False     # include bias in conv encoder
        self.feature_grad_mult: float = 1.0     # multiply feature extractor var grads by this

        self.normalize: bool = False  # normalize input to have 0 mean and unit variance during training

        # dropouts
        self.dropout: float = 0.1     # dropout probability for the transformer
        self.attention_dropout: float = 0.1     # dropout probability for attention weights
        self.activation_dropout: float = 0.0     # dropout probability after activation in FFN
        self.encoder_layerdrop: float = 0.0     # probability of dropping a tarnsformer layer
        self.dropout_input: float = 0.0     # dropout to apply to the input (after feat extr)
        self.dropout_features: float = 0.0     # dropout to apply to the features (after feat extr)

        # masking
        self.mask_length: int = 10     # mask length
        self.mask_prob: float = 0.65     # probability of replacing a token with mask
        self.mask_selection: str = "static"     # how to choose mask length
        self.mask_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indicesh
        self.no_mask_overlap: bool = False     # whether to allow masks to overlap
        self.mask_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # channel masking
        self.mask_channel_length: int = 10     # length of the mask for features (channels)
        self.mask_channel_prob: float = 0.0     # probability of replacing a feature with 0
        self.mask_channel_selection: str = "static"     # how to choose mask length for channel masking
        self.mask_channel_other: float = 0     # secondary mask argument (used for more complex distributions), see help in compute_mask_indices
        self.no_mask_channel_overlap: bool = False     # whether to allow channel masks to overlap
        self.mask_channel_min_space: int = 1     # min space between spans (if no overlap is enabled)

        # positional embeddings
        self.conv_pos: int = 128     # number of filters for convolutional positional embeddings
        self.conv_pos_groups: int = 16     # number of groups for convolutional positional embedding

        # relative position embedding
        self.relative_position_embedding: bool = False     # apply relative position embedding
        self.num_buckets: int = 320     # number of buckets for relative position embedding
        self.max_distance: int = 1280     # maximum distance for relative position embedding
        self.gru_rel_pos: bool = False     # apply gated relative position embedding

        if cfg is not None:
            self.update(cfg)

    def update(self, cfg: dict):
        self.__dict__.update(cfg)

class conformer_d(nn.Module):
    def __init__(
        self,
        cfg: SSL_SER_Config,
    ) -> None:
        super().__init__()
        logger.info(f"Conformer Config: {cfg.__dict__}")

        self.cfg = cfg
        feature_enc_layers = eval(cfg.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        )

        self.encoder = Conformer(input_dim = 256,
        num_heads = 8,
        ffn_dim = 256,
        num_layers = 12,
        depthwise_conv_kernel_size = 31,
        dropout = 0.1,
        use_group_norm = False,
        convolution_first = False,
        )

        self.layer_norm = LayerNorm(256)
        self.pre_conformer_proj = nn.Linear(512, 256)
        self.pre_conformer_dropout = nn.Dropout(0.1)
        self.post_proj = nn.Linear(256, 1024)
        self.asp = ASP(input_dim=cfg.encoder_embed_dim, embed_dim=cfg.encoder_embed_dim, attn_dropout=cfg.attention_dropout)

    def forward(
        self,
        source: torch.Tensor,
    ):
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.pre_conformer_proj(features)
        features = self.pre_conformer_dropout(features)
        features = self.layer_norm(features)
        x = self.encoder(features)
        x = self.post_proj(x)
        
        return x


class Conformer_SER_SSL(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.conformer = conformer_d(config)
        self.asp = ASP(input_dim=config.encoder_embed_dim, embed_dim=config.encoder_embed_dim, attn_dropout=config.attention_dropout)

    def forward(self, x, y, temperature):

        vx = self.conformer(x)
        vy = self.conformer(y)
        vx = self.asp(vx)
        vy = self.asp(vy)

        norm_vx = F.normalize(vx, p=2, dim=1, eps=1e-12)
        norm_vy = F.normalize(vy, p=2, dim=1, eps=1e-12)
        logits = torch.mm(norm_vx, norm_vy.t()) * torch.exp(temperature)

        return logits

    def get_embedding(self, x):
        vx = self.conformer(x)
        vx = self.asp(vx)
        norm_vx = F.normalize(vx, p=2, dim=1, eps=1e-12)

        return norm_vx


class Downstream_model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.linear = nn.Linear(2048, 512)
        self.classifier = nn.Linear(512, 143)
    def forward(self, x):
        x = self.linear(x)
        x = self.classifier(x)
        return x
    def get_embedding(self, x):
        x = self.linear(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        return x