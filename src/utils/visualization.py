# src/utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
from src.config import Config


def plot_spectrogram(spec, title, save_path, sr=Config.SAMPLE_RATE, hop_length=Config.HOP_LENGTH):
    """Plot single spectrogram with time in seconds."""
    plt.figure(figsize=(10, 6))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    
    # Calculate time axis
    time_bins = spec.shape[1]
    time_sec = np.linspace(0, time_bins * hop_length / sr, num=time_bins)
    
    # Fixing tick labels to match tick locations
    tick_locations = np.linspace(0, time_bins - 1, 5, dtype=int)
    tick_labels = [f'{time_sec[tick]:.1f}' for tick in tick_locations]

    plt.xticks(tick_locations, tick_labels)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Frequency Bin')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compare_spectrograms(specs_dict, save_path, sr=Config.SAMPLE_RATE, hop_length=Config.HOP_LENGTH):
    """Plot multiple spectrograms for comparison with time in seconds
    
    Args:
        specs_dict (dict): Dictionary of spectrograms with window type as key
        save_path (str): Path to save the comparison plot
        sr (int): Sample rate
        hop_length (int): Hop length for STFT
    """
    plt.figure(figsize=(15, 5))
    
    # Calculate time axis
    time_bins = list(specs_dict.values())[0].shape[1]
    time_sec = np.linspace(0, time_bins * hop_length / sr, num=time_bins)
    
    # Calculate number of ticks and their positions
    num_ticks = 5
    tick_positions = np.linspace(0, time_bins-1, num_ticks)
    tick_labels = [f'{time_sec[int(pos)]:.1f}' for pos in tick_positions]
    
    for i, (window_type, spec) in enumerate(specs_dict.items(), 1):
        plt.subplot(1, len(specs_dict), i)
        plt.imshow(spec, aspect='auto', origin='lower')
        plt.title(f'{window_type.capitalize()} Window')
        plt.colorbar(format='%+2.0f dB')
        
        # Set x-axis ticks and labels in seconds
        plt.xticks(tick_positions, tick_labels)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency Bin')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_metrics(history, save_dir, model_name):
    """Plot training and validation metrics"""
    # Loss and Accuracy plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title(f'{model_name} Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title(f'{model_name} Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_metrics.png'))
    plt.close()

    # Save metrics as CSV
    df = pd.DataFrame({
        'epoch': range(len(history['train_loss'])),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'train_acc': history['train_acc'],
        'val_acc': history['val_acc']
    })
    df.to_csv(os.path.join(save_dir, f'{model_name}_metrics.csv'), index=False)

def save_training_summary(results, save_path):
    """Save comparison of all models and window types"""
    df = pd.DataFrame(results)
    df.to_csv(save_path)
    
    # Plot summary
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.xlabel('Window Type')
    plt.ylabel('Accuracy')
    plt.legend(title='Model Type')
    plt.tight_layout()
    plt.savefig(save_path.replace('.csv', '.png'))
    plt.close()