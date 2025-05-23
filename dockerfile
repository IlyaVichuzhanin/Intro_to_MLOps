FROM apache/airflow:3.0.0-python3.13

COPY .env /app/.env
COPY airflow_setup_script.py /app/airflow_setup_script.py

RUN envsubst < /tmp/.env > /opt/airflow/.env && rm /tmp/.env


RUN airflow db init


RUN airflow users create --username ${AIRFLOW_USER_NAME} --password ${AIRFLOW_PASSWORD} --ffirstnmae ${AIRFLOW_FIRST_NAME} --lastname ${AIRFLOW_LAST_NAME} --email ${AIRFLOW_EMAIL}
RUN pip install numpy pandas scikit-learn apache-airflow

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY --chown=airflow:root dags/train_decision_tree_dag.py /opt/airflow/dags
ENTRYPOINT [ "airflow", "standalone" ]