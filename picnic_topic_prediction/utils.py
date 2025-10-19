import os
import yaml
import pandas as pd

def get_root_directory():
    return os.getcwd()

def load_training_data():
    return pd.read_parquet(get_root_directory() + '/data/train.parquet')

def load_test_data():
    return pd.read_parquet(get_root_directory() + '/data/train.parquet')

def get_param_grid(model_type: str):
    with open(get_root_directory() + '/picnic_topic_prediction/param_grid.yaml', 'r') as file:
        param_grid = yaml.safe_load(file).get(model_type, None)
        
    return param_grid