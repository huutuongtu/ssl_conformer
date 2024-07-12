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
    def __init__(self, training_audio=[], training_label=[], sampling_rate=16000, device='cuda'):
        self.audio_files = training_audio
        self.label = training_label
        self.sampling_rate = sampling_rate
        self.device = device

    def __getitem__(self, index):
        filename = self.audio_files[index]
        label = self.label[index]
        audio, sampling_rate = load_wav(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))

        return audio, int(label)

    def __len__(self):
        return len(self.audio_files)
