import os
import yaml
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from picnic_topic_prediction.config import LABEL_MAPPING

PIPELINE_MODE = os.environ.get('MODE', 'prod')

def get_root_directory():
    return os.getcwd()

def load_yaml(dir: str):
    with open(get_root_directory() + dir, 'r') as file:
        return yaml.safe_load(file)

def load_data(split: str = 'train'):
    splits = {'train': 'data/train.parquet', 'test': 'data/test.parquet'}
    df = pd.read_parquet(splits.get(split))
    
    if PIPELINE_MODE == 'prod':
        return df
    elif PIPELINE_MODE == 'test':
        n_samples = {'train': 500, 'test': 100}
        return df.sample(n=n_samples.get(split), random_state=42)

def create_confusion_matrix(y_true, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABEL_MAPPING.values())
    return disp.figure_