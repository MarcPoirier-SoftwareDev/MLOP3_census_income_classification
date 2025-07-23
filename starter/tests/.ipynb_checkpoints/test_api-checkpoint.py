import pytest
import json 
import pandas as pd
from fastapi.testclient import TestClient
from main import CensusItem, app
from starter.ml.data import get_cat_features, process_data
from starter.ml.model import get_trained_mlp, inference


@pytest.fixture(scope='module')
def client():
    mock_client = TestClient(app)
    return mock_client


@pytest.fixture(scope='module')
def census_item(negative_example):
    return CensusItem(**negative_example)


def test_census_item(census_item):
    """
    Test that CensusItem is defined properly
    """
    # When creating an instance of CensusItem using a row of our data, no error should be raised
    _ = census_item


def test_api_get_root(client):
    r = client.get("/")
    assert r.status_code == 200
    output = r.json()
    expected_output = {'Greetings': 'Welcome to This API predicting income category using Census data.'}
    assert output == expected_output


# Combined parametrized test for inferences
@pytest.mark.parametrize("example_fixture, type_ex, expected_class", [
    ("positive_example", "positive", 1),  # 1 for >50K
    ("negative_example", "negative", 0),  # 0 for <=50K
])
def test_api_post_inference(example_fixture, type_ex, expected_class, client, request):
    """
    Test that the prediction from the API matches direct model inference for each possible outcome.
    Also checks status code and output type.
    """
    example = request.getfixturevalue(example_fixture)
    response = client.post("/predict", json=example)  # Standardize to json= for all
    assert response.status_code == 200
    output = response.json()['predicted_salary_class']
    assert isinstance(output, int), "Output should be an integer"
    # Direct inference check
    data = pd.DataFrame([example])
    cat_features = get_cat_features()
    model = get_trained_mlp()
    x, _, _, _, _ = process_data(data, categorical_features=cat_features, label=None,
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    predicted = inference(model, x)[0]
    assert output == predicted, f"API prediction ({output}) != direct inference ({predicted}) for {type_ex} example"
    assert predicted == expected_class, f"Unexpected class {predicted} (expected {expected_class}) for {type_ex} example"


# Simple test for invalid input (error handling)
def test_api_post_invalid(client, positive_example):
    """
    Test API handles invalid input (e.g., missing field) with proper error.
    """
    invalid_example = positive_example.copy()
    del invalid_example['age']  # Remove a required field
    response = client.post("/predict", json=invalid_example)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation)
    assert "field required" in response.json()['detail'][0]['msg'].lower()  # Check error message content