import os
import numpy as np
import sys
# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import Config
from src.utils.visualization import plot_spectrogram, compare_spectrograms

def load_subset_specs(file_path, num_samples=5):
    """Load only a subset of spectrograms to conserve memory
    
    Args:
        file_path (str): Path to the .npy file containing spectrograms
        num_samples (int): Number of spectrograms to load
        
    Returns:
        np.ndarray: Subset of spectrograms
    """
    # Memory-mapped array allows reading without loading entire file
    specs = np.load(file_path, mmap_mode='r')
    # Get total number of spectrograms
    total_specs = specs.shape[0]
    
    # Calculate indices to select evenly spaced samples
    if total_specs <= num_samples:
        indices = np.arange(total_specs)
    else:
        indices = np.linspace(0, total_specs-1, num_samples, dtype=int)
    
    # Load only the selected spectrograms
    return specs[indices].copy()

def process_window_type(window_type, results_dir, num_samples=5):
    """Process and plot spectrograms for a single window type
    
    Args:
        window_type (str): Type of window function used
        results_dir (str): Directory to save results
        num_samples (int): Number of spectrograms to process
        
    Returns:
        np.ndarray: First spectrogram for comparison plot
    """
    file_path = os.path.join(Config.DATA_DIR, f'{window_type}_specs.npy')
    specs = load_subset_specs(file_path, num_samples)
    
    # Plot each spectrogram in the subset
    for i, spec in enumerate(specs):
        plot_spectrogram(
            spec,
            f'{window_type.capitalize()} Window - Sample {i+1}',
            os.path.join(results_dir, f'{window_type}_spectrogram_{i+1}.png')
        )
    
    # Return the first spectrogram for comparison plot
    return specs[0]

def main():
    """Main function to process and visualize spectrograms"""
    # Create results directory if it doesn't exist
    results_dir = os.path.join(Config.RESULTS_DIR, 'spectrograms')
    os.makedirs(results_dir, exist_ok=True)

    # Dictionary to store one spectrogram per window type for comparison
    comparison_specs = {}

    try:
        # Process each window type independently
        for window_type in Config.WINDOW_TYPES:
            print(f"Processing {window_type} window type...")
            # Process and get one spectrogram for comparison
            comparison_specs[window_type] = process_window_type(window_type, results_dir)

        print("Creating comparison plot...")
        # Create comparison plot
        compare_spectrograms(comparison_specs, 
                           os.path.join(results_dir, 'window_comparison.png'))
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    
    finally:
        # Clear memory
        if 'comparison_specs' in locals():
            del comparison_specs
        
    print("Processing complete!")

if __name__ == '__main__':
    main()