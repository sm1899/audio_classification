# src/training/train_utils.py
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from ..data.dataset import SpectrogramDataset
from ..models.resnet import ModifiedResNet
from ..models.svm import AudioSVM
from ..utils.visualization import plot_confusion_matrix, plot_training_metrics, save_training_summary

def load_data(data_dir, window_type):
   specs = np.load(os.path.join(data_dir, f'{window_type}_specs.npy'))
   labels = np.load(os.path.join(data_dir, f'{window_type}_labels.npy'))
   return specs, labels

def train_resnet(X, y, window_type, config, device):
   results_dir = os.path.join(config.RESULTS_DIR, 'models', window_type)
   os.makedirs(results_dir, exist_ok=True)

   dataset = SpectrogramDataset(X, y)
   train_size = int(config.TRAIN_SPLIT * len(dataset))
   val_size = len(dataset) - train_size
   train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
   
   train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
   
   model = ModifiedResNet(len(np.unique(y)), config)
   model, history, best_acc = model.train_model(
       model, train_loader, val_loader, config, device, window_type
   )
   
   # Generate predictions for confusion matrix
   if config.SAVE_METRICS:
       model.eval()
       all_preds = []
       all_labels = []
       with torch.no_grad():
           for inputs, labels in val_loader:
               inputs = inputs.to(device)
               outputs = model(inputs)
               _, preds = outputs.max(1)
               all_preds.extend(preds.cpu().numpy())
               all_labels.extend(labels.numpy())
       
       plot_confusion_matrix(
           all_labels, all_preds,
           [str(i) for i in range(len(np.unique(y)))],
           'ResNet Confusion Matrix',
           os.path.join(results_dir, 'resnet_confusion.png')
       )
   
   if config.SAVE_MODELS:
       torch.save(model.state_dict(), os.path.join(results_dir, 'resnet.pt'))
   
   return best_acc

def train_svm(X, y, window_type, config):
   results_dir = os.path.join(config.RESULTS_DIR, 'models', window_type)
   os.makedirs(results_dir, exist_ok=True)

   total_size = len(y)
   train_size = int(config.TRAIN_SPLIT * total_size)
   indices = torch.randperm(total_size)
   train_indices = indices[:train_size]
   val_indices = indices[train_size:]
   
   X_train, y_train = X[train_indices], y[train_indices]
   X_val, y_val = X[val_indices], y[val_indices]
   
   svm = AudioSVM(config)
   svm.train(X_train, y_train)
   
   train_preds = svm.model.predict(X_train.reshape(len(X_train), -1))
   val_preds = svm.model.predict(X_val.reshape(len(X_val), -1))
   
   train_acc = svm.evaluate(X_train, y_train)
   val_acc = svm.evaluate(X_val, y_val)
   
   if config.SAVE_METRICS:
       plot_confusion_matrix(
           y_val, val_preds,
           [str(i) for i in range(len(np.unique(y)))],
           'SVM Confusion Matrix',
           os.path.join(results_dir, 'svm_confusion.png')
       )
   
   if config.SAVE_MODELS:
       svm.save(os.path.join(results_dir, 'svm.joblib'))
   
   return {'train_acc': train_acc, 'val_acc': val_acc}