import os
from dotenv import load_dotenv

load_dotenv()

airflow_user = os.getenv("AIRFLOW_USER_NAME")
airflow_password = os.getenv("AIRFLOW_PASSWORD")
airflow_email = os.getenv("AIRFLOW_EMAIL")
airflow_first_name = os.getenv("AIRFLOW_FIRST_NAME")
airflow_last_name = os.getenv("AIRFLOW_LAST_NAME")

if airflow_user and airflow_password:
    print(f"AIRFLOW_USER_NAME={airflow_user}")
    print(f"AIRFLOW_PASSWORD={airflow_password}")
    print(f"AIRFLOW_EMAIL={airflow_email}")
    print(f"AIRFLOW_FIRST_NAME={airflow_first_name}")
    print(f"AIRFLOW_LAST_NAME={airflow_last_name}")
else:
    print("AIRFLOW_WWW_USER_USERNAME or AIRFLOW_WWW_USER_PASSWORD not found in .env")

