import mlflow

from picnic_topic_prediction.train import train_model
from picnic_topic_prediction.eval import evaluate_model
from picnic_topic_prediction.config import MLFlowExperimentConfig

def main():
    mlflow.set_experiment(**MLFlowExperimentConfig().model_dump())
    
    final_model = train_model()
    print(evaluate_model(final_model))


if __name__ == "__main__":
    main()
