import os

class PathConfig:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_RAW = os.path.join(self.BASE_DIR, 'data', 'raw')
        self.DATA_PROCESSED = os.path.join(self.BASE_DIR, 'data', 'processed')
        self.MODELS_DIR = os.path.join(self.BASE_DIR, 'models')
        self.MLFLOW_DIR = os.path.join(self.BASE_DIR, 'mlruns')

config = PathConfig()
