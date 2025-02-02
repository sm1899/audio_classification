# Audio Classification with UrbanSound8K Dataset

## Setup

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)

```bash
git clone https://github.com/sm1899/audio-classification.git
cd audio-classification
pip install -r requirements.txt
```

### Dataset Setup
Option 1: Direct Download
```bash
# Download: https://goo.gl/8hY5ER
# Extract to: PROJECT_ROOT/data/UrbanSound8K/audio/
```

Option 2: Using Soundata
```python
pip install soundata
import soundata
dataset = soundata.initialize('urbansound8k')
dataset.download()
```

## Project Structure
```
audio_classification/
├── data/
│   └── processed/
├── results/
│   ├── spectrograms/
│   └── models/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   └── spectrogram_generator.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── resnet.py
│   │   └── svm.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── train_utils.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_processing.py
│   │   └── visualization.py
│   └── config.py
├── scripts/
│   ├── generate_spectrograms.py
│   └── visualize.py
├── main.py
└── requirements.txt
```

## Usage

1. Generate spectrograms:
```bash
python scripts/generate_spectrograms.py --data_path /path/to/UrbanSound8K/audio --save_dir data/processed
```

2. Visualize spectrograms:
```bash
python scripts/visualize.py
```

3. Train models:
```bash
python main.py
```

## Configuration
`src/config.py`:
```python
class Config:
    # Data Processing
    SAMPLE_RATE = 22050
    DURATION = 10
    BATCH_SIZE = 32
    
    # Paths
    DATA_PATH = 'data/UrbanSound8K/audio'  # Default dataset path
    DATA_DIR = 'data/processed'            # Processed data
    MODEL_DIR = 'models'
    RESULTS_DIR = 'results'
    
    # Training
    TRAIN_SPLIT = 0.8
    MODELS = ['resnet', 'svm']
    WINDOW_TYPES = ['hann', 'hamming', 'rectangular']

    SVM_CONFIG = {
        'kernel': 'rbf',
        'C': 1.0,
    }

    RESNET_CONFIG = {
        'pretrained': True,
        'num_epochs': 10,
        'learning_rate': 0.001,
    }
```

## Requirements
```
numpy>=1.21.0
librosa>=0.9.0
torch>=1.9.0
tqdm>=4.62.0
scikit-learn>=0.24.2
matplotlib>=3.4.3
scipy>=1.7.0
pandas>=1.3.0
seaborn>=0.11.2
soundata>=0.1.0  # Optional
```

## Citation
```
J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research", 
22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
```

## License
MIT
```
