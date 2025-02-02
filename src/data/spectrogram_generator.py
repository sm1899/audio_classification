# src/data/spectrogram_generator.py

# import os
# import numpy as np
# import librosa
# from tqdm.notebook import tqdm
# from ..utils.audio_processing import apply_window, create_spectrogram

# class SpectrogramGenerator:
#     def __init__(self, data_path, save_dir, sr=22050, duration=10):
#         self.data_path = data_path
#         self.save_dir = save_dir
#         self.sr = sr
#         self.target_length = sr * duration
#         os.makedirs(save_dir, exist_ok=True)
    
#     def process_file(self, file_path, window_type):
#         audio, _ = librosa.load(file_path, sr=self.sr)
#         return create_spectrogram(audio, window_type, self.target_length)
    
#     def generate(self, window_type):
#         spectrograms = []
#         labels = []
#         file_paths = []
        
#         save_path = os.path.join(self.save_dir, f'{window_type}_specs.npy')
#         label_path = os.path.join(self.save_dir, f'{window_type}_labels.npy')
        
#         if os.path.exists(save_path) and os.path.exists(label_path):
#             print(f"Loading saved {window_type} spectrograms...")
#             return np.load(save_path), np.load(label_path)
        
#         print(f"Generating {window_type} spectrograms...")
        
#         for folder in tqdm(os.listdir(self.data_path), desc="Processing folders"):
#             folder_path = os.path.join(self.data_path, folder)
#             if not os.path.isdir(folder_path):
#                 continue
                
#             for file in tqdm(os.listdir(folder_path), desc=f"Folder {folder}", leave=False):
#                 if not file.endswith('.wav'):
#                     continue
                    
#                 file_path = os.path.join(folder_path, file)
#                 class_id = int(file.split('-')[1])
                
#                 try:
#                     spectrogram = self.process_file(file_path, window_type)
#                     spectrograms.append(spectrogram)
#                     labels.append(class_id)
#                     file_paths.append(file_path)
#                 except Exception as e:
#                     print(f"Error processing {file_path}: {str(e)}")
        
#         X = np.array(spectrograms)
#         y = np.array(labels)
        
#         np.save(save_path, X)
#         np.save(label_path, y)
        
#         return X, y

# src/data/spectrogram_generator.py
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Optional
from ..utils.audio_processing import load_audio, create_spectrogram

class SpectrogramGenerator:
    def __init__(
        self, 
        data_path: str, 
        save_dir: str, 
        sr: int = 22050, 
        duration: int = 10, 
        n_fft: int = 2048, 
        hop_length: int = 512
    ):
        """
        Initialize Spectrogram Generator
        
        Args:
            data_path (str): Path to input audio files
            save_dir (str): Directory to save generated spectrograms
            sr (int): Sample rate
            duration (int): Target audio duration in seconds
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
        """
        self.data_path = data_path
        self.save_dir = save_dir
        self.sr = sr
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
    
    def process_file(
        self, 
        file_path: str, 
        window_type: str
    ) -> np.ndarray:
        """
        Generate spectrogram for a single audio file
        
        Args:
            file_path (str): Path to audio file
            window_type (str): Type of window function to use
        
        Returns:
            Spectrogram as numpy array
        """
        # Load audio
        audio = load_audio(
            file_path, 
            target_sr=self.sr, 
            target_duration=self.duration,
            device=self.device
        )
        
        # Create spectrogram
        return create_spectrogram(
            audio, 
            window_type, 
            target_length=self.sr * self.duration,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            device=self.device
        )
    
    def generate(
        self, 
        window_type: str,
        batch_size: int = 32,
        limit: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spectrograms for all audio files
        
        Args:
            window_type (str): Type of window function to use
            batch_size (int): Number of files to process in memory at once
            limit (int, optional): Limit the number of files to process
        
        Returns:
            Tuple of spectrograms and labels
        """
        spectrograms = []
        labels = []
        
        # Process files by walking through directories
        for folder in tqdm(os.listdir(self.data_path), desc="Processing folders"):
            folder_path = os.path.join(self.data_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            for file in tqdm(os.listdir(folder_path), desc=f"Folder {folder}", leave=False):
                if not file.endswith('.wav'):
                    continue
                
                file_path = os.path.join(folder_path, file)
                
                # Extract class ID from filename (UrbanSound8K naming convention)
                try:
                    class_id = int(file.split('-')[1])
                except (IndexError, ValueError):
                    print(f"Skipping file {file} - cannot extract class ID")
                    continue
                
                try:
                    spectrogram = self.process_file(file_path, window_type)
                    spectrograms.append(spectrogram)
                    labels.append(class_id)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                
                # Optional: limit processing if specified
                if limit and len(spectrograms) >= limit:
                    break
            
            if limit and len(spectrograms) >= limit:
                break
        
        # Convert to numpy arrays
        return np.array(spectrograms), np.array(labels)