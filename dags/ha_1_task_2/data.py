import os
import requests
import json 
import numpy as np
import csv
from datetime import datetime
from dotenv import load_dotenv


def fetch_weather(lat:str, lon:str):
        
        ENV_PATH = "/opt/airflow/dags/ha_1_task_2/.env"
        load_dotenv(ENV_PATH)
        WEATHER_API_KEY=os.getenv("WEATHER_API_KEY")


        if not WEATHER_API_KEY:
            raise ValueError("Missing required environment variables")

        res = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}')
        response = json.loads(res.text) 

        weather={
            'date_time':datetime.now(),
            'id':response['weather'][0]['id'],
            'city':response['name'],
            'lon':response['coord']['lon'],
            'lat':response['coord']['lat'],
            'description':response['weather'][0]['description'],
            'temp':response['main']['temp'],
            'feels_like':response['main']['feels_like'],
            'pressure':response['main']['pressure'],
            'wind_speed':response['wind']['speed']
        }

        file_path='dataset/weather.csv'
        delim=';'
        header = np.array(["date_time", "id", "city", "lon", "lat", "description", "temp", "feels_like", "pressure", "wind_speed"])

        if os.path.exists(file_path)==False:
            with open(file_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delim)
                writer.writerow(header)
            csvfile.close
               
        if os.path.exists(file_path):
            with open(file_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=delim)
                weather_data = np.array(
                    [
                        weather['date_time'],
                        weather['id'], 
                        weather['city'],
                        weather['lon'],
                        weather['lat'],
                        weather['description'],
                        weather['temp'],
                        weather['feels_like'],
                        weather['pressure'],
                        weather['wind_speed']
                        ]
                    )
                writer.writerow(weather_data)
            