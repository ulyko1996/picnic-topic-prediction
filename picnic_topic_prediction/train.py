from typing import Any, Dict
from picnic_topic_prediction.utils import load_training_data
from picnic_topic_prediction.hp_tuning import get_param_grid

from optuna_integration import OptunaSearchCV
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

def train_model(model_type:str = 'tfidf_lgb'):
    models = {'tfidf_lgb': Pipeline([('tfidf_vectorizer', TfidfVectorizer(stop_words='english')), 
                                     ('lightgbm', LGBMClassifier(objective='multiclass', random_state=42, verbosity=0))])}
    
    output = OptunaSearchCV(estimator=models.get(model_type), 
                            param_distributions=get_param_grid(model_type), 
                            cv=5, 
                            scoring='accuracy', 
                            n_trials=1,
                            n_jobs=-1).fit(load_training_data()['text'], load_training_data()['label'])
    
    print(output.cv_results_)
    print(output.best_score_)
    return output.best_estimator_