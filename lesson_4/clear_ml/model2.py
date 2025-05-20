from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from config import config
from data import get_data
from clearml import Task
import joblib



task = Task.init(
    project_name='clear_ml_project', 
    task_name='decision_tree_classifier',
    tags=['decision_tree_classifier']
    )


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    # Здесь необходимо получить метрики и логировать их в трекер
    
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    r2=r2_score(y_true=y_test, y_pred=y_pred)
    mae=mean_absolute_error(y_true=y_test, y_pred=y_pred)

    logger = task.get_logger()
    logger.report_single_value(name='accuracy', value=accuracy)
    logger.report_single_value(name='r2', value=r2)
    logger.report_single_value(name='mae', value=mae)

    task.close()

    joblib.dump(model, 'decision_tree_classifier.plk', compress=True)


if __name__ == "__main__":
    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"]
    )

    data = get_data()
    train(decision_tree_model, data["x_train"], data["y_train"])
    test(decision_tree_model, data["x_test"], data["y_test"])
