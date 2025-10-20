from typing import Any, Dict
from picnic_topic_prediction.utils import load_training_data
from picnic_topic_prediction.config import OptunaSearchCVConfig, PARAMETER_GRID

from optuna_integration import OptunaSearchCV
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

import mlflow

MODELS = {
    'tfidf_lgb': Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), 
                           ('lightgbm', LGBMClassifier(objective='multiclass', random_state=42, verbose=-1))])
    }

def train_model(model_type:str = 'tfidf_lgb'):
    training_data = load_training_data()
    output = OptunaSearchCV(estimator=MODELS.get(model_type), 
                            param_distributions=PARAMETER_GRID.get(model_type), 
                            n_jobs=-1,
                            **OptunaSearchCVConfig().model_dump(),
                            ).fit(training_data['text'], training_data['label'])
    
    mlflow.log_params(output.best_params_)
    mlflow.log_metric('Training accuracy', output.best_score_)
    return output.best_estimator_