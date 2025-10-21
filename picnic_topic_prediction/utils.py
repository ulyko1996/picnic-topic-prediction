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

def load_training_data():
    df = pd.read_parquet(get_root_directory() + '/data/train.parquet')
    if PIPELINE_MODE == 'prod':
        return df
    elif PIPELINE_MODE == 'test':
        return df.sample(n=500, random_state=42)
        

def load_test_data():
    df = pd.read_parquet(get_root_directory() + '/data/test.parquet')
    if PIPELINE_MODE == 'prod':
        return df
    elif PIPELINE_MODE == 'test':
        return df.sample(n=100, random_state=42)

def create_confusion_matrix(y_true, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABEL_MAPPING.values())
    return disp.figure_