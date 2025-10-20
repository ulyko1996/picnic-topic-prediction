from picnic_topic_prediction.utils import load_yaml

from optuna.distributions import IntDistribution, FloatDistribution, CategoricalDistribution

def get_param_grid(model_type: str):
    param_grid = load_yaml('/picnic_topic_prediction/param_grid.yaml').get(model_type, None)
    
    # Raise error if parameter grid config is not found for the model type 
    if param_grid is None:
        raise ValueError(f'Model type {model_type} not found')
    
    for param in param_grid:
        distributions = {'int': IntDistribution, 'float': FloatDistribution, 'categorical': CategoricalDistribution}
        param_grid[param] = distributions.get(param_grid[param]['type'])(**param_grid[param]['kwargs'])

    return param_grid