from typing import Any, Dict
from picnic_topic_prediction.utils import load_training_data, get_param_grid

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier

def train_model(model_type:str = 'tfidf_lgb'):
    models = {'tfidf_lgb': Pipeline([('tfidf_vectorizer', TfidfVectorizer(stop_words='english')), 
                                     ('lightgbm', LGBMClassifier(objective='multiclass', random_state=42, verbosity=0))])}
    
    output = RandomizedSearchCV(models.get(model_type), 
                                get_param_grid(model_type), 
                                cv=5, 
                                scoring='accuracy', 
                                n_jobs=-1).fit(load_training_data()['text'], load_training_data()['label'])
    
    print(output.cv_results_)
    print(output.best_score_)
    return output.best_estimator_