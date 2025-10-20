from pydantic import BaseModel

class OptunaSearchCVConfig(BaseModel):
    cv: int = 5
    scoring: str = 'accuracy'
    n_trials: int = 1