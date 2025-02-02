class Config:
   # Data Processing
   SAMPLE_RATE = 22050
   DURATION = 4
   BATCH_SIZE = 16
   TRAIN_SPLIT = 0.8

   # Spectrogram Generation Parameters
   N_FFT = 1024  # FFT window size
   HOP_LENGTH = 256  # Number of samples between successive frames

   # Paths
   DATA_PATH = '../UrbanSound8K/audio'
   DATA_DIR = 'data/processed'  
   MODEL_DIR = 'models'
   RESULTS_DIR = 'results'

   # Windowing
   WINDOW_TYPES = ['hann', 'hamming', 'rectangular']

   # Models
   MODELS = ['resnet']

   SVM_CONFIG = {
       'kernel': 'rbf',
       'C': 1.0,
       'gamma': 'scale'
   }

   RESNET_CONFIG = {
       'pretrained': True,
       'freeze_backbone': False,
       'num_epochs': 10,
       'learning_rate': 0.0005,
       'optimizer': 'adam',
       'scheduler': 'cosine'
   }

   # Results
   SAVE_PLOTS = True
   SAVE_MODELS = True
   SAVE_METRICS = True