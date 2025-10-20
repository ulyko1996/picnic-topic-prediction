import mlflow
from picnic_topic_prediction.utils import load_test_data
from picnic_topic_prediction.config import LABEL_MAPPING

from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

def evaluate_model(model):
    test_data = load_test_data()
    y_true = test_data['label']
    y_pred = model.predict(test_data['text'])
    
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=LABEL_MAPPING.values())
    
    # Calculate metrics
    mlflow.log_metric('Test accuracy', accuracy_score(y_true, y_pred))
    mlflow.log_figure(disp.figure_, "Confusion Matrix.png")