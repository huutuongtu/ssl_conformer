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

data = pd.read_csv("train.csv")
training_files = data['Path']
training_labels = data['Label']

dev = pd.read_csv("dev.csv")

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int,
                        help='batch size')
args = parser.parse_args()

xyz = args.batch
load_dir = "./logs/log_" + str(xyz) + "/" + "checkpoint.pth"
save_dir = "./logs/log_" + str(xyz) + "/" + "checkpoint_ds.pth"

# Load configuration
with open("config.json", "r", encoding='utf8') as f:
    cfg_dict = json.load(f)

# Initialize configuration and model
cfg = SSL_SER_Config(cfg_dict)
ssl = Conformer_SER_SSL(cfg).to(device)
ssl.load_state_dict(torch.load(load_dir))
for param in ssl.parameters():
    param.requires_grad = False
ssl.eval().to(device)

model = Downstream_model()
model.train().to(device)

nll_loss = nn.NLLLoss(ignore_index = 2)

def collate_fn(batch):
    
    with torch.no_grad():    
        sr = 16000
        cols = {'features':[], 'labels':[]}
        
        for row in batch:
            with torch.no_grad():
                f = ssl.get_embedding(torch.tensor(row[0], device=device).unsqueeze(0)).squeeze(0).detach().cpu().numpy()
                cols['features'].append(f)
            cols['labels'].append(row[1])
        return torch.tensor(cols['features'], device=device), torch.tensor(cols['labels'], dtype=torch.long, device=device)
        
    
trainset = AudioDataset(training_audio=training_files, training_label=training_labels, device=device)
batch_size = 16
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

max_acc = 0
# Loop training
for epoch in range(25):
    for i, batch in tqdm(enumerate(train_loader)):
        features, labels = batch
        logits = model(features)
        logits = F.log_softmax(logits, dim=1)
        loss = nll_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%500==0:
            print(loss)
        # break
    with torch.no_grad():
        model.eval().to(device)
        ss = 307
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
        epoch_WA = cnt/ss
        if (epoch_WA > max_acc):
            print("save_checkpoint...")
            max_acc = epoch_WA
            torch.save(model.state_dict(), save_dir)
        print("Acc " + str(epoch) + ": " + str(epoch_WA))
        print("Max Acc: " + str(max_acc))

# pretraining 32 64 128 256 done
#32 64 training downstream       