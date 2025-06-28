import pytest
import httpx
from httpx import AsyncClient

# Base URL for the API
BASE_URL = "http://127.0.0.1:8000"

# Sample input data for <=50K prediction
INPUT_LE50K = {
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
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States"
}

# Sample input data for >50K prediction
INPUT_GT50K = {
    "age": 50,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 83311,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 10000,
    "capital_loss": 0,
    "hours_per_week": 60,
    "native_country": "United-States"
}

@pytest.mark.asyncio
async def test_get_root():
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "Welcome to the Machine Learning Inference API!"}

@pytest.mark.asyncio
async def test_predict_le50k():
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=INPUT_LE50K)
        assert response.status_code == 200
        assert response.json() == {"prediction": "<=50K"}

@pytest.mark.asyncio
async def test_predict_gt50k():
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.post("/predict", json=INPUT_GT50K)
        assert response.status_code == 200
        assert response.json() == {"prediction": ">50K"}