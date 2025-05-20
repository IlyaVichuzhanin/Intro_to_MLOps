from sklearn.datasets import load_iris
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import json

def test(model_path: str, test_csv: str):
    loaded_model = joblib.load(model_path)
    df = pd.read_csv(test_csv, header=0, sep=';', index_col='id')
    X_test = df.drop('target', axis=1) 
    Y_test = df['target']
    predicted_result=loaded_model.predict(X_test)
    accuracy = accuracy_score(Y_test, predicted_result)
    matrix=confusion_matrix(Y_test, predicted_result)
    report=classification_report(Y_test, predicted_result)
    metrics = {'accuracy': accuracy, 'report': report}
    print(accuracy)
    print(matrix)
    print(metrics)


