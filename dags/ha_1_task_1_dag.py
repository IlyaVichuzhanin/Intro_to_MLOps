# import sys
# sys.path.append("C:/Users/user/MyProjects/Intro_to_MLOps")
from ha_1_task_1.data import load_data, prepare_data
from ha_1_task_1.train import train
from ha_1_task_1.test import test
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator



default_dag_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 5, 15),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

file_path = 'dataset/iris.csv'
train_file_path = 'dataset/iris_train.csv'
test_file_path = 'dataset/iris_test.csv'
model_file_path = 'ha_1_task_1/model.plk'

dag = DAG(
    'logistic_regression_on_iris_dataset',
    default_args=default_dag_args,
    schedule='@daily'
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable= load_data,
    dag=dag,
)

prepare_data_task = PythonOperator(
    task_id='prepare_data',
    python_callable= prepare_data,
    op_kwargs={'csv_path': file_path},
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable= train,
    op_kwargs={'train_csv': train_file_path},
    dag=dag,
)

test_model_task = PythonOperator(
    task_id='test_model',
    python_callable= test,
    op_kwargs={'model_path': model_file_path, 'test_csv': test_file_path},
    dag=dag,
)

load_data_task >> prepare_data_task >> train_model_task >> test_model_task



