from picnic_topic_prediction.utils import load_test_data

from sklearn.metrics import accuracy_score

def evaluate_model(model):
    return {'accuracy': accuracy_score(load_test_data()['label'], model.predict(load_test_data()['text']))}