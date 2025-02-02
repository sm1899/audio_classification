# main.py
import os
import sys
import torch
from tqdm import tqdm
from src.config import Config
from src.training.train_utils import load_data, train_resnet, train_svm


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    results = {}
    
    for window_type in Config.WINDOW_TYPES:
        results[window_type] = {}
        print(f"\nProcessing {window_type} window spectrograms...")
        
        X, y = load_data(Config.DATA_DIR, window_type)
        print(f"Data shape: {X.shape}")
        
        if 'resnet' in Config.MODELS:
            print("\nTraining ResNet...")
            results[window_type]['resnet'] = train_resnet(X, y, window_type, Config, device)
            
        if 'svm' in Config.MODELS:
            print("\nTraining SVM...")
            results[window_type]['svm'] = train_svm(X, y, window_type, Config)
    
    print("\nFinal Results:")
    for window_type, models in results.items():
        print(f"\n{window_type.capitalize()} Window:")
        if 'resnet' in models:
            print(f"ResNet Best Validation Accuracy: {models['resnet']:.4f}")
        if 'svm' in models:
            print(f"SVM Train Accuracy: {models['svm']['train_acc']:.4f}")
            print(f"SVM Validation Accuracy: {models['svm']['val_acc']:.4f}")

if __name__ == '__main__':
    main()