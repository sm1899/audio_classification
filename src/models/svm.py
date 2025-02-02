# src/models/svm.py
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from tqdm import tqdm
import numpy as np
import torch

class AudioSVM:
    def __init__(self, config):
        self.model = SVC(**config.SVM_CONFIG)
    
    def train(self, X, y):
        # Move data to CPU if it's on GPU
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        print("Training SVM...")
        with tqdm(total=1, desc="SVM Training") as pbar:
            self.model.fit(X.reshape(len(X), -1), y)
            pbar.update(1)
    
    def evaluate(self, X, y):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
            
        predictions = self.model.predict(X.reshape(len(X), -1))
        accuracy = accuracy_score(y, predictions)
        return accuracy
    
    def save(self, path):
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path, config):
        instance = cls(config)
        instance.model = joblib.load(path)
        return instance