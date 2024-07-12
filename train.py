from dataloader import AudioDataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Conformer_SER_SSL, SSL_SER_Config
from torch.nn import functional as F
import torch
import json
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
# training_files = pd.read_csv("/home/jovyan/voice-chung/tuht/emoVDB/RAVDESS/Ravdess.csv")['Path']

training_files = []
f = open("/home/jovyan/voice-chung/tuht/mfreevc/FreeVC/filelists/vi_train.txt", "r", encoding="utf8").readlines()
for line in f:
    training_files.append(line.strip())

device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch', type=int,
                        help='batch size')
args = parser.parse_args()

batch_size = args.batch
log_dir = "./logs/log_" + str(args.batch)
load_dir = "./logs/log_" + str(args.batch) + "/" + "checkpoint.pth"
save_dir = "./logs/log_" + str(args.batch) + "/" + "checkpoint_200.pth"

try:
    os.mkdir(log_dir)
except:
    print(log_dir)

writer = SummaryWriter(log_dir=log_dir)

trainset = AudioDataset(training_audio=training_files, device=device)
train_loader = DataLoader(trainset, num_workers=32, batch_size=batch_size, shuffle=True)

# Load configuration
with open("config.json", "r", encoding='utf8') as f:
    cfg_dict = json.load(f)

# Initialize configuration and model
cfg = SSL_SER_Config(cfg_dict)
model = Conformer_SER_SSL(cfg).to(device)
model.load_state_dict(torch.load(load_dir))

temperature = torch.LongTensor([2]).to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

loss = 30
# Loop training
for epoch in range(100):
    for i, batch in tqdm(enumerate(train_loader)):
        audio1, audio2, length1, length2 = batch
        # try:
        audio1, audio2 = audio1.to(device), audio2.to(device)
        logits = model(audio1, audio2, temperature)
        labels = torch.arange(audio1.shape[0], dtype=torch.long).to(device)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i%100==0:
            print(loss)
    writer.add_scalar("Loss/train", loss, epoch+100)
    torch.save(model.state_dict(), save_dir)
        # except:
        #     continue
writer.flush()
writer.close()
    