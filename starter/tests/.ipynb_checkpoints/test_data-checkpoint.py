from ..ml.data import get_path_root, get_hyperparameters, get_cat_features


def test_get_path_root():
    """Verify get_path_root returns the absolute path to the project root directory."""
    path_root = get_path_root()
    assert path_root.name == 'MLOP3_census_income_classification', "Wrong project directory name"
    assert path_root.is_absolute(), "Path is not absolute"


def test_get_raw_data(raw_data):
    """Verify raw census data (census.csv) has expected structure and columns."""
    assert not raw_data.empty, "Raw data is empty"
    assert len(raw_data.columns) == 15, "Unexpected number of columns"
    expected_columns = [
        'age', 'workclass', 'fnlgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary'
    ]
    assert list(raw_data.columns) == expected_columns, "Raw data columns do not match expected"


def test_raw_data_integrity(raw_data):
    """Ensure raw data meets basic expectations."""
    assert len(raw_data) >= 30000, "Raw data too small"
    assert set(raw_data['salary'].unique()) == {' >50K', ' <=50K'}, "Unexpected salary values in raw data"
    assert raw_data.isnull().sum().sum() == 0, "Null values in raw data"


def test_get_hyperparameters():
    """Verify hyperparameters dictionary contains expected keys with valid values."""
    params = get_hyperparameters()['parameters']
    assert set(params.keys()) == {'batch_size', 'dropout_rate', 'hidden_dim', 'learning_rate', 'n_layers'}
    assert params['batch_size'] > 0, "Batch size must be positive"
    assert 0 <= params['dropout_rate'] <= 1, "Dropout rate must be between 0 and 1"
    assert params['hidden_dim'] > 0, "Hidden dimension must be positive"
    assert params['learning_rate'] > 0, "Learning rate must be positive"
    assert params['n_layers'] >= 1, "Number of layers must be at least 1"


def test_get_cat_features():
    """Verify get_cat_features returns the correct list of categorical feature names."""
    exp_cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    cat_features = get_cat_features()
    assert cat_features == exp_cat_features



def test_clean_data_integrity(clean_data):
    """Verify clean data (census_clean.csv) has expected size, salary values, and no nulls."""
    assert len(clean_data) >= 30000, "Clean data too small"
    assert set(clean_data['salary'].unique()) == {'>50K', '<=50K'}, "Unexpected salary values"
    assert clean_data.isnull().sum().sum() == 0, "Null values in clean data"




    