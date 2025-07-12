"""
Test if the API GET and POST requests work properly.
"""

import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


# example to change to have another prediction
data = {
    "age": 52,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Divorced",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 7000,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

url = 'https://mlop3-census-income-classification-3.onrender.com'

# url = "http://127.0.0.1:8000/"
# when run with uvicorn works fine
response1 = requests.get(url)
print(f"status get request: {response1.status_code}")
print(response1.text)

url = url + "predict"
r = requests.post(
    url=url,
    json=data,
)
status = r.status_code
print(f"status post request: {status}")

if status == 200:
    print(r.json())