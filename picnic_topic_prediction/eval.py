import mlflow
from picnic_topic_prediction.utils import load_data, create_confusion_matrix

from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model):
    test_data = load_data('test')
    y_true = test_data['label']
    y_pred = model.predict(test_data['text'])
    
    # Calculate metrics
    mlflow.log_metrics({
        'Test accuracy': accuracy_score(y_true, y_pred), 
        'Test F1 Score': f1_score(y_true, y_pred, average='macro'),
    })
    mlflow.log_figure(create_confusion_matrix(y_true, y_pred), "Test Confusion Matrix.png")