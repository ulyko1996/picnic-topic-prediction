import mlflow

from picnic_topic_prediction.utils import load_data, create_confusion_matrix
from picnic_topic_prediction.config import OptunaSearchCVConfig, PARAMETER_GRID

from optuna_integration import OptunaSearchCV
from sklearn.pipeline import Pipeline

# import re
# from nltk.stem import WordNetLemmatizer
# from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
from lightgbm import LGBMClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# The lemmatizer can be passed to the tokenizer argument of TfidfVectorizer
# LEMMATIZER = lambda text: [WordNetLemmatizer().lemmatize(token) 
#                            for token in re.compile('(?u)\\b\\w\\w+\\b').findall(text) 
#                            if token not in ENGLISH_STOP_WORDS]

MODELS = {
    'tfidf_lgb': Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                           ('lightgbm', LGBMClassifier(objective='multiclass', random_state=42, verbose=-1))])
    }

def train_model(model_type:str = 'tfidf_lgb'):
    training_data = load_data('train')
    X_train = training_data['text']
    y_train = training_data['label']
    output = OptunaSearchCV(estimator=MODELS.get(model_type), 
                            param_distributions=PARAMETER_GRID.get(model_type), 
                            n_jobs=-1,
                            **OptunaSearchCVConfig().model_dump(),
                            ).fit(X_train, y_train)
    
    y_pred = output.best_estimator_.predict(X_train)
    
    mlflow.log_params(output.best_params_)
    mlflow.log_metrics({
        'Training accuracy': output.best_score_, 
        'Training F1 Score': f1_score(y_train, y_pred, average='macro'),
    })
    mlflow.log_figure(create_confusion_matrix(y_train, y_pred), "Training Confusion Matrix.png")
    
    if model_type == 'tfidf_lgb':
        lookup_dict = {value: key for key, value in output.best_estimator_['tfidf'].vocabulary_.items()}
        feature_importance_fig, ax = plt.subplots(figsize=(10, 6))
        ax = lgb.plot_importance(output.best_estimator_['lightgbm'], ax=ax, importance_type="gain", title="LightGBM Feature Importance (Gain)", max_num_features=20)
        ax.set_yticklabels([lookup_dict.get(int(label.get_text().split('_')[-1])) for label in ax.get_yticklabels()])
        mlflow.log_figure(feature_importance_fig, "LightGBM Feature Importance (Gain).png")
        plt.close(feature_importance_fig)
    
    return output.best_estimator_