from fastapi import APIRouter
import pandas as pd
import joblib


prediction_router = APIRouter(tags=['Prediction'])

@prediction_router.post('/get_prediction')
def get_prediction(
    fixed_acidity: float, 
    volatile_acidity: float, 
    citric_acid: float, 
    residual_sugar: float,
    chlorides: float,
    free_sulfur_dioxide: float,
    total_sulfur_dioxide: float,
    density: float,
    pH: float,
    sulphates: float,
    alcohol: float):

    data = {
        'fixed_acidity': [fixed_acidity],
        'volatile_acidity': [volatile_acidity],
        'citric_acid': [citric_acid],
        'residual_sugar':[residual_sugar],
        'chlorides':[chlorides],
        'free_sulfur_dioxide':[free_sulfur_dioxide],
        'total_sulfur_dioxide':[total_sulfur_dioxide],
        'density':[density],
        'pH':[pH],
        'sulphates':[sulphates],
        'alcohol':[alcohol]
    }

    df = pd.DataFrame(data)

    with open('model.pkl', 'rb') as file:
        model = joblib.load(file)

    prediction = model.predict(df)

    print(prediction)


    return {"predicted_quality": prediction[0]}