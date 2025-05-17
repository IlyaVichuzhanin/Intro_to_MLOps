FROM apache/airflow:3.0.0-python3.13
RUN airflow db init
RUN airflow users create --username admin --password password --ffirstnmae admin --lastname admin --email admin@admin.com
RUN pip install numpy pandas scikit-learn apache-airflow

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY --chown=airflow:root dags/train_decision_tree_dag.py /opt/airflow/dags
ENTRYPOINT [ "airflow", "standalone" ]