"""
Methods related to data manipulation.
This is a modification of the file provided in the original Udacity repository.

"""
import yaml
import os
import csv
import pathlib
from pathlib import Path, PurePath
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import logging
import pickle
import torch 


logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def process_data(
        X, categorical_features=[], label=None, training=True, encoder=None, lb=None, scaler=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using one hot encoding for the categorical features and a
    label binarizer for the labels. As we use a neural network we scale the continuous data.
    This can be used in either training or inference/validation.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    scaler : sklearn.preprocessing.StandardScaler
        Trained sklearn scaler, only used if training=False

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the encoder passed
        in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the binarizer
        passed in.
    scaler : sklearn.preprocessing.StandardScaler
        Trained sklearn scalerif training is True, otherwise returns the scaler
        passed in.
    """

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1).values

    if training is True:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
        # we're using neural networks so we need to scale the data
        scaler = StandardScaler()
        X_continuous = scaler.fit_transform(X_continuous)
    else:
        X_categorical = encoder.transform(X_categorical)
        X_continuous = scaler.transform(X_continuous)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)

    return X, y, encoder, lb, scaler


def get_path_root() -> Path:
    """
    Determine the project root directory.

    First, checks if the PROJECT_ROOT environment variable is set (e.g., for deployed environments like Render).
    If not, traverses up from the current working directory until a directory containing 'Procfile' is found (e.g., for local use).
    
    Returns:
        Path: Absolute path to the project root directory.
    
    Raises:
        Exception: If the project root cannot be found.
    """
    # Check for environment variable (e.g., set in Render)
    if 'PROJECT_ROOT' in os.environ:
        root = Path(os.environ['PROJECT_ROOT'])
        logger.info(f"Using PROJECT_ROOT from environment: {root}")
        return root
    
    # Fallback to searching for a marker file (Procfile) locally
    current_path = Path.cwd()
    for _ in range(10):  # Limit traversal to 10 levels
        if (current_path / 'Procfile').exists():
            logger.info(f"Found Procfile at: {current_path}")
            return current_path
        if current_path.parent == current_path:  # Reached filesystem root
            break
        current_path = current_path.parent
    
    raise Exception('Cannot find project path')


def get_path_file(file_local_path):
    """
    Return the full path of a file given its local path
    :param file_local_path: local path of the file in the project (ex: "data/census.csv")
    """
    project_dir = get_path_root()
    raw_path = PurePath.joinpath(project_dir, file_local_path)
    return raw_path


def get_raw_data():
    """
    Get the raw data as a DataFrame
    :return:
    pd.DataFrame containing raw data as read from the csv file
    """
    raw_path = get_path_file("data/census.csv")
    raw_data = pd.read_csv(raw_path)
    return raw_data


def get_clean_data():
    """
    Get the pre-processed data as a DataFrame
    :return:
    pd.DataFrame containing pre-processed data as read from the csv file
    """
    raw_path = get_path_file("data/census_clean.csv")
    raw_data = pd.read_csv(raw_path)
    return raw_data


def save_clean_data():
    """
    Remove white spaces from "census.csv" and save data processed as such to "census_clean.csv".
    """
    raw_path = get_path_file("data/census.csv")
    clean_path = get_path_file("data/census_clean.csv")
    with open(raw_path, 'r') as f_raw, open(clean_path, 'w') as f_clean:
        reader = csv.reader(f_raw, skipinitialspace=False, delimiter=',', quoting=csv.QUOTE_NONE)
        writer = csv.writer(f_clean)
        for row in reader:
            clean_row = [item.strip() for item in row]
            writer.writerow(clean_row)


def save_hyperparameters(params: dict, random_state):
    params = {'name': 'MLP hyper parameters', 'parameters': params, 'random_state': random_state}
    file_name = get_path_file('parameters/hyperparams.yml')
    if file_name.is_file():
        os.remove(file_name)  # if the file already exists, delete it
    with open(file_name, 'w') as outfile:
        yaml.dump(params, outfile, default_flow_style=False)


def get_hyperparameters():
    file_name = get_path_file('parameters/hyperparams.yml')
    with open(file_name, "r") as stream:
        params = yaml.safe_load(stream)

    return params


def get_cat_features(for_api=False) -> list:
    """
    Get the categorical features. When using the api, the hyphens are replaced by underscores
    :param for_api:
    :return:
    """
    file_name = get_path_file('parameters/cat_features.yml')
    with open(file_name, "r") as stream:
        features = yaml.safe_load(stream)['features']
    # if used by api hyphens are replaced by underscore
    if for_api:
        for i, item in enumerate(features):
            features[i] = item.replace('-', '_')
    return features


def get_processed_test_data(encoder, lb, scaler, data=None):
    if data is None:
        data = get_clean_data()

    random_state = get_hyperparameters()['random_state']
    _, test = train_test_split(data, test_size=0.20, random_state=random_state)
    cat_features = get_cat_features()
    x_test, y_test, _, _, _ = process_data(test, categorical_features=cat_features, label="salary",
                                           training=False, encoder=encoder, lb=lb, scaler=scaler)
    return x_test, y_test


def get_data_slices(selected_feature: str, encoder: OneHotEncoder, lb: LabelBinarizer, scaler: StandardScaler) -> dict:
    """
    Return slices of data corresponding to the different possible values that can take the categorical variable
    selected_feature
    :param selected_feature: categorical feature used for slicing
    :param encoder: sklearn one hot encoder
    :param lb:  sklearn label binarizer
    :param scaler: sklearn scaler
    :return:
    dictionary which keys are the possible values that can take selected_feature. The values of that dictionary are
    dictionaries {'x': features, 'y': labels} for the slice of the data corresponding to selected_feature taking the
    value in the key of the dictionary.
    """

    data = get_clean_data()
    random_state = get_hyperparameters()['random_state']
    _, test = train_test_split(data, test_size=0.20, random_state=random_state)
    grouped = test.groupby(selected_feature)
    cat_features = get_cat_features()
    output = {}
    for value, sliced in grouped:
        x_sliced, y_sliced, _, _, _ = process_data(sliced, categorical_features=cat_features, label="salary",
                                                   training=False, encoder=encoder, lb=lb, scaler=scaler)
        output[value] = {'x': x_sliced, 'y': y_sliced}
    return output


if __name__ == '__main__':
    
    # Get the root and model directory
    root = get_path_root()
    model_dir = root / 'model'
    
    # Load the required preprocessor objects
    with open(model_dir / 'encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open(model_dir / 'lb.pkl', 'rb') as f:
        lb = pickle.load(f)
    with open(model_dir / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Get the data slices for 'education'
    get_data_slices('education', encoder, lb, scaler)
    
