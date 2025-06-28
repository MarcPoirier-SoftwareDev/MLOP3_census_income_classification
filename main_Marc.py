# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List
from data import process_data
from model import inference

app = FastAPI(title="Machine Learning Inference API", description="API for model inference and greetings")

# Load the trained model, encoder, and label binarizer
model = joblib.load("model.pkl")
encoder = joblib.load("encoder.pkl")
lb = joblib.load("lb.pkl")

# Define categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Pydantic model for POST request body
class InferenceInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }

# GET endpoint at root
@app.get("/")
async def root():
    return {"message": "Welcome to the Machine Learning Inference API!"}

# POST endpoint for model inference
@app.post("/predict")
async def predict(input_data: InferenceInput):
    # Convert Pydantic model to DataFrame
    data_dict = input_data.dict()
    data_df = pd.DataFrame([data_dict])

    # Process the data
    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Run inference
    preds = inference(model, X)

    # Transform prediction back to label
    pred_label = lb.inverse_transform(preds)[0]

    return {"prediction": pred_label}