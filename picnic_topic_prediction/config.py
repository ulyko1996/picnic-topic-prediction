from pydantic import BaseModel
from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

class MLFlowExperimentConfig(BaseModel):
    experiment_name: str = 'tfidf-lightgbm'

class OptunaSearchCVConfig(BaseModel):
    cv: int = 5
    scoring: str = 'accuracy'
    n_trials: int = 1
    
LABEL_MAPPING = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Science/Technology"
}
    
PARAMETER_GRID = {
    'tfidf_lgb': {
        'tfidf__ngram_range': CategoricalDistribution([(1,1), (1,2)]),
        'lightgbm__n_estimators': IntDistribution(low=10, high=200, step=10),
        'lightgbm__num_leaves': IntDistribution(low=4, high=64, step=4),
        'lightgbm__max_depth': CategoricalDistribution([-1, 4, 8, 16, 32]),
        'lightgbm__learning_rate': FloatDistribution(low=0.01, high=0.1, step=0.01),
        'lightgbm__subsample': FloatDistribution(low=0.8, high=1, step=0.05),
        'lightgbm__subsample_freq': IntDistribution(low=0, high=1, step=1),
        'lightgbm__colsample_bytree': FloatDistribution(low=0.8, high=1, step=0.05),
    }
}