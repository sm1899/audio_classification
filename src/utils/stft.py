import torch
import numpy as np
from typing import Union, Tuple, Optional

def stft(
    signal: torch.Tensor,
    n_fft: int,
    hop_length: int,
    window: torch.Tensor,
    center: bool = True,
    return_complex: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Implement Short-Time Fourier Transform from scratch
    
    Args:
        signal: Input signal [1D tensor]
        n_fft: FFT size
        hop_length: Number of samples between successive frames
        window: Window function
        center: Whether to pad signal on both sides
        return_complex: Whether to return complex tensor
    
    Returns:
        Either a complex tensor or a tuple of (magnitude, phase) tensors
    """
    # Convert to numpy for easier manipulation if needed
    device = signal.device
    signal_np = signal.cpu().numpy() if isinstance(signal, torch.Tensor) else signal
    window_np = window.cpu().numpy() if isinstance(window, torch.Tensor) else window
    
    # Center padding
    if center:
        padding = [(n_fft // 2, n_fft // 2)]
        signal_np = np.pad(signal_np, padding, mode='reflect')
    
    # Calculate number of frames
    n_samples = len(signal_np)
    n_frames = 1 + (n_samples - n_fft) // hop_length
    
    # Initialize output matrix
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    
    # Process each frame
    for frame in range(n_frames):
        # Extract frame
        start = frame * hop_length
        end = start + n_fft
        frame_signal = signal_np[start:end]
        
        # Apply window
        windowed_frame = frame_signal * window_np
        
        # Compute FFT
        # Note: We only keep positive frequencies (n_fft//2 + 1) samples
        fft_frame = np.fft.rfft(windowed_frame)
        
        # Store in output matrix
        stft_matrix[:, frame] = fft_frame
    
    # Convert back to torch tensor
    real_part = torch.from_numpy(stft_matrix.real).to(device)
    imag_part = torch.from_numpy(stft_matrix.imag).to(device)
    
    if return_complex:
        return torch.complex(real_part, imag_part)
    else:
        magnitude = torch.sqrt(real_part.pow(2) + imag_part.pow(2))
        phase = torch.atan2(imag_part, real_part)
        return magnitude, phase