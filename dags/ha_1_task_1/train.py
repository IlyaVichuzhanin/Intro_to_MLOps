from sklearn.datasets import load_iris
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

def train(train_csv: str) -> str:
    """Обучение модели логистической регрессии на тренировочной выборке и сохранение модели."""
    df = pd.read_csv(train_csv, header=0, sep=';', index_col='id')
    X_train = df.drop('target', axis=1) 
    Y_train = df['target']
    model = LogisticRegression()
    model.fit(X_train,Y_train)
    filename = 'model.pkl'
    joblib.dump(model, filename=filename)