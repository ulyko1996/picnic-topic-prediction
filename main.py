from picnic_topic_prediction.train import train_model
from picnic_topic_prediction.eval import evaluate_model

def main():
    final_model = train_model()
    print(evaluate_model(final_model))


if __name__ == "__main__":
    main()
