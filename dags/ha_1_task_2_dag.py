# import sys
# sys.path.append("C:/Users/user/MyProjects/Intro_to_MLOps")
from ha_1_task_2.data import fetch_weather
from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import requests
import json 
import numpy as np
import csv
from datetime import datetime



lat=44.21
lon=10.50


with DAG(
    'get_weather_info',
    # default_args=default_dag_args,
    description="A DAG that runs one task every minute",
    start_date=datetime(2025, 5, 16),
    schedule="*/1 * * * *",
    catchup=False
) as dag:
    
    fetch_weather_task = PythonOperator(
    task_id='fetch_weather',
    python_callable= fetch_weather,
    op_kwargs={'lat': lat, 'lon': lon}
    )








            




