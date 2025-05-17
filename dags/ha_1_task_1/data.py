from sklearn.datasets import load_iris
import csv 
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data() -> str:
    data = load_iris()
    file_path = 'dataset/iris.csv'
    if os.path.exists(file_path):
        os.remove(file_path)
    delim=';'

    header=np.array(["id", "target"]+data.feature_names)

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=delim)
        writer.writerow(header)
        for row in range(np.array(data.data).shape[0]):
            target = np.concatenate((np.array([row, data.target[row]]), data.data[row]))
            writer.writerow(target)

    return file_path


def prepare_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=0, sep=';', index_col='id')
    X = df.drop('target', axis=1) 
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 80% train, 20% test
    
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_file_path = 'dataset/iris_train.csv'
    test_file_path = 'dataset/iris_test.csv'

    if os.path.exists(train_file_path):
        os.remove(train_file_path)

    if os.path.exists(test_file_path):
        os.remove(test_file_path)

    train_df.to_csv(train_file_path, index=True, sep=';')
    test_df.to_csv(test_file_path, index=True, sep=';')

    return df
