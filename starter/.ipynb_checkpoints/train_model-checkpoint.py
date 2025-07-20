"""
Script to train and save machine learning model and to perform model evaluation on data slices.

"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from .ml.data import process_data, get_clean_data, get_cat_features, get_data_slices, get_processed_test_data
from .ml.model import train_model, compute_model_metrics, inference, Mlp, get_trained_mlp
import argparse


def train_and_save_model(tuning: bool = True, random_state: int = 42,
                         use_saved_model: bool = False) -> Tuple[float, float, float]:
    """
    Train and save a model. The tools used for processing the data is saved as well, both the data processing tools
    and the model
    are saved in the 'model' folder.
    :param tuning: Set to true if hyperparameters are to be optimised. If false hyperparameters are loaded from a
    yaml file.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output.
    :param use_saved_model: If True, starts from the model already saved.
    :return: tuple (precision, recall, F1)
    """

    # Load the cleaned dataset for training and testing
    data = get_clean_data()

    # Split the data into training and testing sets (80/20 split)
    # Note: For enhancement, consider using K-fold cross-validation in future iterations
    train, test = train_test_split(data, test_size=0.20, random_state=random_state)

    # Retrieve categorical features for data processing
    cat_features = get_cat_features()

    # Process the training data: encode categoricals, scale features, and binarize labels
    X_train, y_train, encoder, lb, scaler = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data using the same encoder, label binarizer, and scaler from training
    X_test, y_test, _, _, _ = process_data(
        test, categorical_features=cat_features, label="salary",
        training=False, encoder=encoder, lb=lb, scaler=scaler
    )

    # Train the model (with optional hyperparameter tuning)
    model = train_model(X_train, y_train, tuning, random_state, use_saved_model)

    # Save the trained model along with preprocessing tools (encoder, lb, scaler)
    model.save_model(encoder, lb, scaler)

    # Make predictions on the test set
    y_pred = inference(model, X_test)

    # Compute and return evaluation metrics: precision, recall, F1 score
    evaluation = compute_model_metrics(y_test, y_pred)

    return evaluation


def model_metrics_slices(model: Mlp, selected_feature: str) -> pd.DataFrame:
    """
    Computes performance on model slices for selected_feature. The print output is saved in screenshots/slice_output.txt
    :param model: trained model which performance is measured on slices of data.
    :param selected_feature: categorical feature used for slicing
    :return:
    Dataframe whith the different values of selected_feature as rows and columns precision, recall and f1 score.
    """
    # Get data slices based on the selected categorical feature
    slices = get_data_slices(selected_feature, model.encoder, model.lb, model.scaler)

    # Dictionary to store metrics for each slice
    slices_metrics = {}

    # Compute metrics for each slice
    for key, data_dict in slices.items():
        x = data_dict['x']
        y = data_dict['y']
        slices_metrics[key] = model_metrics(model, x, y)

    # Convert metrics dictionary to a DataFrame for easy viewing and saving
    slices_metrics_df = pd.DataFrame(slices_metrics).T

    # Save the slice metrics to a text file
    with open('screenshots/slice_output.txt', 'w') as f:
        print(f'Performance on model slices for {selected_feature}:\n\n', slices_metrics_df, file=f)

    return slices_metrics_df


def model_metrics(model: Mlp, x_test: np.array = None, y_test: np.array = None) -> dict:
    """
    Compute the performance of model on test data x_test, y_test
    :param model: trained model which performance is measured
    :param x_test: test data features
    :param y_test: test data labels
    :return:
    a dictionary with the precision, recall and f1 metrics calculated on test data
    """
    # If test data is not provided, load and process the default test data
    if x_test is None or y_test is None:
        x_test, y_test = get_processed_test_data(model.encoder, model.lb, model.scaler)

    # Make predictions on the test data
    y_pred = inference(model, x_test)

    # Compute precision, recall, and F1 score
    precision, recall, f1 = compute_model_metrics(y_test, y_pred)

    # Return metrics as a dictionary
    evaluation = {'precision': precision, 'recall': recall, 'f1': f1}
    return evaluation


if __name__ == '__main__':
    # Parse command-line arguments to allow flexible configuration (e.g., for CI/CD pipelines)
    parser = argparse.ArgumentParser(description="Train model and compute slice metrics.")
    parser.add_argument('--tuning', action='store_true', default=True, help="Enable hyperparameter tuning.")
    parser.add_argument('--random_state', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--use_saved_model', action='store_true', default=False, help="Use pre-saved model for training.")
    parser.add_argument('--slice_feature', type=str, default='education', help="Categorical feature for slicing (e.g., 'education', 'race').")
    args = parser.parse_args()

    # Train the model and evaluate overall performance
    print("Training and saving model...")
    evaluation = train_and_save_model(tuning=args.tuning, random_state=args.random_state, use_saved_model=args.use_saved_model)
    print(f"Overall test metrics: Precision={evaluation[0]:.4f}, Recall={evaluation[1]:.4f}, F1={evaluation[2]:.4f}")

    # Load the trained model for slice evaluation
    model = get_trained_mlp()

    # Compute metrics on data slices for the selected feature
    print(f"Computing metrics on slices for feature: {args.slice_feature}")
    slices_df = model_metrics_slices(model, args.slice_feature)
    print("Slice metrics saved to screenshots/slice_output.txt")
    print(slices_df)  # Print to console for immediate feedback or logging