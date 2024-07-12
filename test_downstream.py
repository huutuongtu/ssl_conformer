from dataloader_ds import AudioDataset, load_wav
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Conformer_SER_SSL, SSL_SER_Config, Downstream_model
from torch.nn import functional as F
import torch.nn as nn
import torch
import json
import argparse
import os
import pandas as pd

dev = pd.read_csv("test.csv")

device = 'cuda'


batches = [16,32,64,128,256,512]
    # Initialize configuration and model

for xyz in batches:
    load_dir = "./logs/log_" + str(xyz) + "/" + "checkpoint.pth"
    save_dir = "./logs/log_" + str(xyz) + "/" + "checkpoint_ds.pth"

    xx = open("result_linear_classifier.txt", "a+", encoding="utf8")
    # Load configuration
    with open("config.json", "r", encoding='utf8') as f:
        cfg_dict = json.load(f)
    cfg = SSL_SER_Config(cfg_dict)
    ssl = Conformer_SER_SSL(cfg).to(device)
    ssl.load_state_dict(torch.load(load_dir))
    for param in ssl.parameters():
        param.requires_grad = False
    ssl.eval().to(device)
    model = Downstream_model()
    model.load_state_dict(torch.load(save_dir))
    with torch.no_grad():
        model.eval().to(device)
        ss = 600
        cnt = 0
        for point in tqdm(range(len(dev))):
            path = dev['Path'][point]
            label = dev['Label'][point]
            audio, sampling_rate = load_wav(path)
            input = ssl.get_embedding(torch.tensor(audio, device=device).unsqueeze(0))
            output = model(input)
            output = torch.argmax(output, dim=1).detach().cpu().numpy()[0]
            if str(output)==str(label):
                cnt = cnt + 1
        epoch_WA = cnt/ss #unweighted Accuracy
        xx.write(str(save_dir) + "|" + str(epoch_WA) + "%\n")
