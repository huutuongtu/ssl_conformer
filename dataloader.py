import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from glob import glob
import librosa

def load_wav(full_path):
    data, sampling_rate = librosa.load(full_path, sr=16000)
    return data, sampling_rate

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, training_audio=[], segment_size=8192, sampling_rate=16000, hop_size=256, device=None):
        self.audio_files = training_audio
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_files[index]
        audio, sampling_rate = load_wav(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)
        length_audio = torch.LongTensor([0])

        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start_1 = random.randint(0, max_audio_start)
            audio_start_2 = random.randint(0, max_audio_start)
            audio1 = audio[:, audio_start_1:audio_start_1+self.segment_size]
            audio2 = audio[:, audio_start_2:audio_start_2+self.segment_size]
            length_audio = torch.LongTensor([self.segment_size//self.hop_size])

            return (audio1.squeeze(0), audio2.squeeze(0), length_audio.squeeze(0), length_audio.squeeze(0))

    def __len__(self):
        return len(self.audio_files)
