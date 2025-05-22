from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from config import config
from data import get_data
import mlflow 
from mlflow.models import infer_signature
import joblib

mlflow.set_tracking_uri("http://127.0.0.1:8085")
experiment = mlflow.set_experiment("logistic_regression_model")

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер

    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    r2=r2_score(y_true=y_test, y_pred=y_pred)
    mae=mean_absolute_error(y_true=y_test, y_pred=y_pred)

    joblib.dump(model, 'logistic_regression_model.plk', compress=True)

    return y_pred, accuracy, r2, mae




if __name__ == "__main__":
    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
    )

    data = get_data()

    with mlflow.start_run():
        train(logistic_regression_model, data["x_train"], data["y_train"])
        y_pred, accuracy, r2, mae = test(logistic_regression_model, data["x_test"], data["y_test"])

        signature = infer_signature(data["x_test"], y_pred)

        for param_name, param_value in config.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


        mlflow.sklearn.log_model(
            sk_model=logistic_regression_model,
            artifact_path="logistic_regression",
            signature=signature,
            registered_model_name="logistic_regression"
        )

    

