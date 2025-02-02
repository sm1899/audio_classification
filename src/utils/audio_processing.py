# # src/utils/audio_processing.py
# import numpy as np
# from scipy.signal import windows
# import librosa

# def apply_window(signal, window_type):
#     if window_type == 'hann':
#         window = windows.hann(len(signal))
#     elif window_type == 'hamming':
#         window = windows.hamming(len(signal))
#     else:  # rectangular
#         window = windows.boxcar(len(signal))
#     return signal * window

# def create_spectrogram(audio, window_type, target_length):
#     # Pad or truncate
#     if len(audio) > target_length:
#         audio = audio[:target_length]
#     else:
#         audio = np.pad(audio, (0, target_length - len(audio)))
    
#     # Apply window
#     windowed_signal = apply_window(audio, window_type)
    
#     # Generate spectrogram
#     spectrogram = librosa.feature.melspectrogram(y=windowed_signal, sr=22050)
#     spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
#     return spectrogram_db

# src/utils/audio_processing.py
import torch
import torchaudio
import math
import numpy as np

from .stft import stft

def hann_window(window_length, device):
    """Create a Hann window tensor"""
    n = torch.arange(window_length, device=device)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (window_length - 1))

def hamming_window(window_length, device):
    """Create a Hamming window tensor"""
    n = torch.arange(window_length, device=device)
    return 0.54 - 0.46 * torch.cos(2 * math.pi * n / (window_length - 1))

def rectangular_window(window_length, device):
    """Create a rectangular window tensor"""
    return torch.ones(window_length, device=device)

def load_audio(
    file_path, 
    target_sr=22050, 
    target_duration=10, 
    device='cuda'
):
    """
    Load and preprocess audio file using TorchAudio
    
    Args:
        file_path (str): Path to audio file
        target_sr (int): Target sample rate
        target_duration (int): Target duration in seconds
        device (str): Device to load tensor on
    
    Returns:
        torch.Tensor: Preprocessed audio signal
    """
    # Load audio
    waveform, original_sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Resample if necessary
    if original_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    
    # Calculate target length in samples
    target_length = int(target_sr * target_duration)
    
    # Trim or pad to target length
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        pad_length = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_length))
    
    return waveform.to(device)

def create_spectrogram(
    audio: torch.Tensor,
    window_type: str,
    target_length: int = 220500,
    n_fft: int = 2048,
    hop_length: int = 512,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Create spectrogram from audio signal using custom STFT implementation
    
    Args:
        audio: Input audio signal
        window_type: Type of window function to use
        target_length: Target length of audio signal
        n_fft: FFT window size
        hop_length: Number of samples between successive frames
        device: Device to perform computation on
    
    Returns:
        Spectrogram magnitude in decibels as numpy array
    """
    # Ensure audio is on the correct device
    audio = audio.to(device)
    
    # Select window function
    window_functions = {
        'hann': torch.hann_window,
        'hamming': torch.hamming_window,
        'rectangular': lambda n, device: torch.ones(n, device=device)
    }
    
    # Create window
    window = window_functions[window_type](n_fft, device=device)
    
    # Compute spectrogram using our custom STFT
    # We explicitly handle the complex tensor case
    spectrogram_result = stft(
        audio.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        return_complex=True  # This ensures we get a single complex tensor
    )
    assert isinstance(spectrogram_result, torch.Tensor)  # Type assertion for type checker
    spectrogram_complex: torch.Tensor = spectrogram_result
    
    # Now we're guaranteed to have a complex tensor
    spectrogram_mag = torch.abs(spectrogram_complex)
    spectrogram_db = 20 * torch.log10(spectrogram_mag + 1e-10)
    
    return spectrogram_db.cpu().numpy()