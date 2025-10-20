import os
import yaml
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay
from picnic_topic_prediction.config import LABEL_MAPPING

def get_root_directory():
    return os.getcwd()

def load_yaml(dir: str):
    with open(get_root_directory() + dir, 'r') as file:
        return yaml.safe_load(file)

def load_training_data():
    return pd.read_parquet(get_root_directory() + '/data/train.parquet')[:500]

def load_test_data():
    return pd.read_parquet(get_root_directory() + '/data/train.parquet')

def create_confusion_matrix(y_true, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABEL_MAPPING.values())
    return disp.figure_