import pytest
from ..ml.data import get_raw_data, get_clean_data, get_cat_features, process_data
from ..ml.model import get_trained_mlp, inference


@pytest.fixture(scope='module')
def raw_data():
    """
    Get raw data and return a pd.DataFrame
    """
    df = get_raw_data()
    return df


@pytest.fixture(scope='module')
def clean_data():
    """
    Get clean data and return a pd.DataFrame
    """
    df = get_clean_data()
    return df


@pytest.fixture(scope='module')
def get_examples():
    """
    This fixture creates two dictionaries of features, one that is predicted as positive and one
    that is predicted as negative by our trained model
    """
    model = get_trained_mlp()
    df = get_clean_data()
    cat_features = get_cat_features()
    x, _, _, _, _ = process_data(df, categorical_features=cat_features, label="salary",
                                 training=False, encoder=model.encoder, lb=model.lb, scaler=model.scaler)
    predicted = inference(model, x)  # Change to inference (handles torch eval, tensor conversion, thresholding)
    num_positive = (predicted == 1).sum()
    num_negative = (predicted == 0).sum()
    assert num_positive > 0, f"No positive predictions found (all {num_negative} negative) – verify model training or inference logic"
    assert num_negative > 0, f"No negative predictions found (all {num_positive} positive)"
    positive = df[predicted == 1].iloc[0].to_dict()
    negative = df[predicted == 0].iloc[0].to_dict()
    return positive, negative


@pytest.fixture(scope='module')
def negative_example():
    # From your data head: <=50K example, drop 'salary'
    return {
        'age': 39, 'workclass': 'State-gov', 'fnlgt': 77516, 'education': 'Bachelors', 'education-num': 13,
        'marital-status': 'Never-married', 'occupation': 'Adm-clerical', 'relationship': 'Not-in-family',
        'race': 'White', 'sex': 'Male', 'capital-gain': 2174, 'capital-loss': 0, 'hours-per-week': 40,
        'native-country': 'United-States'
    }


@pytest.fixture(scope='module')
def positive_example():
    # Hardcoded from a known >50K row (extract via df[df['salary'] == '>50K'].iloc[0].drop('salary').to_dict() locally)
    return {
        'age': 52, 'workclass': 'Self-emp-inc', 'fnlgt': 287927, 'education': 'HS-grad', 'education-num': 9,
        'marital-status': 'Married-civ-spouse', 'occupation': 'Exec-managerial', 'relationship': 'Wife',
        'race': 'White', 'sex': 'Female', 'capital-gain': 15024, 'capital-loss': 0, 'hours-per-week': 40,
        'native-country': 'United-States'
    }

