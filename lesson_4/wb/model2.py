from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from config import config
from data import get_data
import wandb
import joblib


wandb.init(project="wb_project")

def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    r2=r2_score(y_true=y_test, y_pred=y_pred)
    mae=mean_absolute_error(y_true=y_test, y_pred=y_pred)

    wandb.summary['accuracy']=accuracy
    wandb.summary['r2']=r2
    wandb.summary['mae']=mae

    joblib.dump(model, 'decision_tree_classifier.plk', compress=True)

    artifact=wandb.Artifact('decision_tree_classifier', type='model')
    artifact.add_file('decision_tree_classifier.plk')
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
