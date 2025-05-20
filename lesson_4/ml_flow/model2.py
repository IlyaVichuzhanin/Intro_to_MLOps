from sklearn.tree import DecisionTreeClassifier
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
experiment = mlflow.set_experiment("decision_tree_classifier")


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    r2=r2_score(y_true=y_test, y_pred=y_pred)
    mae=mean_absolute_error(y_true=y_test, y_pred=y_pred)

    joblib.dump(model, 'decision_tree_classifier.plk', compress=True)

    return y_pred, accuracy, r2, mae


if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )

    data = get_data()
    
    with mlflow.start_run():
        train(decision_tree_model, data["x_train"], data["y_train"])
        y_pred, accuracy, r2, mae = test(decision_tree_model, data["x_test"], data["y_test"])

        signature = infer_signature(data["x_test"], y_pred)

        for param_name, param_value in config.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)


        mlflow.sklearn.log_model(
            sk_model=decision_tree_model,
            artifact_path="decision_tree_model",
            signature=signature,
            registered_model_name="decision_tree"
        )
