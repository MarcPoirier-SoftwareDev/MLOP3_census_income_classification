# Script to train machine learning model.

# Necessary imports
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from data import process_data
from model import train_model

# Load the dataset
data = pd.read_csv('data.csv')  # Adjust the filename as needed

# Split the data into training and testing sets
train, test = train_test_split(data, test_size=0.20, random_state=42)

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

# Process the training data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train the model
model = train_model(X_train, y_train)

# Save the trained model, encoder, and label binarizer
joblib.dump(model, 'model.pkl')
joblib.dump(encoder, 'encoder.pkl')
joblib.dump(lb, 'lb.pkl')
