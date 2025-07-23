import pytest
from ..ml.data import get_raw_data, get_clean_data, get_cat_features, process_data
from ..ml.model import get_trained_mlp, inference


@pytest.fixture(scope='module')
def raw_data():
    """Load raw census data from data/census.csv as a pandas DataFrame."""
    df = get_raw_data()
    return df


@pytest.fixture(scope='module')
def clean_data():
    """Load cleaned census data from data/census_clean.csv as a pandas DataFrame."""
    df = get_clean_data()
    return df


@pytest.fixture(scope='module')
def negative_example():
    # <=50K example, drop 'salary'
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

