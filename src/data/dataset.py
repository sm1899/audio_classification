# src/data/dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np

class SpectrogramDataset(Dataset):
    def __init__(self, spectrograms, labels):
        self.spectrograms = torch.FloatTensor(spectrograms).unsqueeze(1)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.spectrograms[idx], self.labels[idx]