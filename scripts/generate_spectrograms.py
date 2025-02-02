# # scripts/generate_spectrograms.py
# import os
# import sys
# import argparse
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.config import Config
# from src.data.spectrogram_generator import SpectrogramGenerator

# def parse_args():
#    parser = argparse.ArgumentParser(description='Generate spectrograms with different window types')
#    parser.add_argument('--data_path', type=str, default=None,
#                       help='Path to UrbanSound8K audio files')
#    parser.add_argument('--save_dir', type=str, default=None,
#                       help='Directory to save processed spectrograms')
#    return parser.parse_args()

# def main():
#    args = parse_args()
   
#    # Use arguments if provided, else use config
#    data_path = args.data_path or Config.DATA_PATH
#    save_dir = args.save_dir or Config.DATA_DIR
   
#    os.makedirs(save_dir, exist_ok=True)
#    generator = SpectrogramGenerator(data_path, save_dir, 
#                                   sr=Config.SAMPLE_RATE, 
   
#    for window_type in Config.WINDOW_TYPES:
#        X, y = generator.generate(window_type)
#        print(f"Generated {window_type} spectrograms: {X.shape}")

# if __name__ == '__main__':
#    main()


# scripts/generate_spectrograms.py
import os
import sys
import json
import argparse
import numpy as np
import time
import torch
from tqdm import tqdm
import math
import gc

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.data.spectrogram_generator import SpectrogramGenerator

def parse_args():
    """
    Parse command-line arguments for spectrogram generation
    """
    parser = argparse.ArgumentParser(description='Generate spectrograms with different window types')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to audio files')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save processed spectrograms')
    parser.add_argument('--window-types', type=str, nargs='+', default=None,
                        help='Specific window types to use (e.g., hann hamming rectangular)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of files to process')
    return parser.parse_args()

def main():
    """
    Main function to generate spectrograms
    """
    # Parse arguments
    args = parse_args()
    
    # Use arguments if provided, else use config
    data_path = args.data_path or Config.DATA_PATH
    save_dir = args.save_dir or Config.DATA_DIR
    
    # Determine window types
    window_types = args.window_types or Config.WINDOW_TYPES
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize generator
    generator = SpectrogramGenerator(
        data_path, 
        save_dir, 
        sr=Config.SAMPLE_RATE, 
        duration=Config.DURATION,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH
    )
    
    # Store results
    results = {}
    
    # Process each window type
    for window_type in window_types:
        try:
            # Clear GPU cache before each window type
            torch.cuda.empty_cache()
            gc.collect()
            
            # Progress bar for batch processing
            with tqdm(total=1, desc=f'Processing {window_type} spectrograms') as proc_pbar:
                # Generate spectrograms and labels
                X, y = generator.generate(
                    window_type, 
                    batch_size=Config.BATCH_SIZE,
                    limit=args.limit
                )
                proc_pbar.update(1)
            
            # Create filenames using the specified convention
            spectrogram_file = os.path.join(save_dir, f'{window_type}_specs.npy')
            labels_file = os.path.join(save_dir, f'{window_type}_labels.npy')
            
            # Progress bar for saving spectrograms
            with tqdm(total=1, desc=f'Saving {window_type} spectrograms') as specs_pbar:
                np.save(spectrogram_file, X)
                specs_pbar.update(1)
            
            # Progress bar for saving labels
            with tqdm(total=1, desc=f'Saving {window_type} labels') as labels_pbar:
                np.save(labels_file, y)
                labels_pbar.update(1)
            
            # Track results
            results[window_type] = {
                'spectrograms_shape': X.shape,
                'labels_shape': y.shape,
                'spectrogram_file': spectrogram_file,
                'labels_file': labels_file,
                'unique_classes': np.unique(y).tolist()
            }
            
            # Explicitly delete large variables
            del X, y
            gc.collect()
            torch.cuda.empty_cache()
        
        except Exception as e:
            print(f"Error processing {window_type} window: {e}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
    
    # Save results summary
    results_file = os.path.join(save_dir, f'spectrogram_generation_results_{int(time.time())}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results summary saved to {results_file}")

if __name__ == '__main__':
    main()