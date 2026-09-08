from fastapi import FastAPI
from pydantic import BaseModel,Field
import joblib
from pathlib import Path
import pandas as pd


BASE_DIR=Path(__file__).resolve().parent.parent
MODEL_PATH=BASE_DIR/'model.pkl'

model = joblib.load(MODEL_PATH)

app=FastAPI()

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.post("/predict")
def predict(data: CustomerData):

    
    input_data = pd.DataFrame([data.model_dump()])

    
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4)
    }