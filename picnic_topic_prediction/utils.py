import os
import yaml
import pandas as pd

def get_root_directory():
    return os.getcwd()

def load_yaml(dir: str):
    with open(get_root_directory() + dir, 'r') as file:
        return yaml.safe_load(file)

def load_training_data():
    return pd.read_parquet(get_root_directory() + '/data/train.parquet')

def load_test_data():
    return pd.read_parquet(get_root_directory() + '/data/train.parquet')
