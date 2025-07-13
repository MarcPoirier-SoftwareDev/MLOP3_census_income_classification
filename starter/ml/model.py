"""
Merged implementation of MLP model, training, and evaluation functions.
This combines the contents of model.py and nn_model.py while preserving functionality.
Refactored slightly for consolidation: unified imports, logging, and removed redundant relative imports.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
import optuna
from pickle import dump, load
from .data import save_hyperparameters, get_hyperparameters, get_path_file
from typing import Callable, Tuple
import logging

# Consolidated logging setup
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """
    Multi-layer perceptron (MLP) neural network implementation in PyTorch.

    Parameters
    ----------
    n_layers : int, default=2
        Number of hidden layers.
    hidden_dim : int, default=50
        Size of each hidden layer.
    n_classes : int, default=2
        Number of output classes.
    input_dim : int, default=108
        Dimension of the input features.
    batch_size : int, default=1028
        Number of examples per batch.
    epochs : int, default=200
        Number of training iterations over the dataset.
    learning_rate : float, default=0.001
        Learning rate for the optimizer.
    dropout_rate : float, default=0.5
        Dropout probability for regularization.
    hyper_tuning : bool, default=False
        If True, suppresses training logs for hyperparameter tuning.
    use_saved_hyper_params : bool, default=False
        If True, loads hyperparameters from a YAML file.
    """
    def __init__(
        self,
        n_layers: int = 2,
        hidden_dim: int = 50,
        n_classes: int = 2,
        input_dim: int = 108,
        batch_size: int = 1028,
        epochs: int = 200,
        learning_rate: float = 0.001,
        dropout_rate: float = 0.5,
        hyper_tuning: bool = False,
        use_saved_hyper_params: bool = False
    ):
        logger.info("Initializing MLP")
        super(Mlp, self).__init__()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.hyper_tuning = hyper_tuning
        
        # Load hyperparameters if specified
        if use_saved_hyper_params:
            params = get_hyperparameters()['parameters']
            self.batch_size = params['batch_size']
            self.dropout_rate = params['dropout_rate']
            hidden_dim = params['hidden_dim']
            self.learning_rate = params['learning_rate']
            n_layers = params['n_layers']
        else:
            self.batch_size = batch_size
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate

        # Define network and optimizer
        self.loss_fn = nn.CrossEntropyLoss()  # Expects raw logits
        self.network = build_mlp(
            input_size=input_dim,
            output_size=n_classes,
            n_layers=n_layers,
            hidden_size=hidden_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)

        # Preprocessing tools (initialized as None, populated via load_model or training)
        self.encoder = None  # OneHotEncoder
        self.lb = None  # LabelBinarizer
        self.scaler = None  # StandardScaler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args
        ----
        x : torch.Tensor
            Input features [batch_size, input_dim].

        Returns
        -------
        torch.Tensor
            Logits [batch_size, n_classes].
        """
        return self.network(x)

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train the MLP on the provided data.

        Args
        ----
        x_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels (integers in {0, 1, ..., n_classes-1}).

        Returns
        -------
        float
            Average loss of the last epoch.
        """
        dataset = CensusDataset(x_train, y_train, self.device)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.train()  # Set model to training mode
        last_loss = 0.0

        for epoch in range(self.epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches = len(train_loader)
                if (epoch % 20 == 0 and i == n_batches - 1 and not self.hyper_tuning):
                    last_loss = running_loss / n_batches
                    logger.info(f"Epoch {epoch}, Loss: {last_loss:.4f}")
                    running_loss = 0.0

        return last_loss

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Perform inference on input data.

        Args
        ----
        x : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Predicted class indices.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).to(self.device).float()
            logits = self.network(x_tensor)
            return logits.argmax(dim=1).cpu().numpy().reshape(-1)

    def save_model(self, encoder: OneHotEncoder, lb: LabelBinarizer, scaler: StandardScaler) -> None:
        """
        Save the model state and preprocessing tools.

        Args
        ----
        encoder : OneHotEncoder
            Fitted one-hot encoder.
        lb : LabelBinarizer
            Fitted label binarizer.
        scaler : StandardScaler
            Fitted standard scaler.
        """
        model_path = get_path_file('model/mlp.pt')
        torch.save(self.state_dict(), model_path)
        for obj, filename in [(encoder, 'encoder.pkl'), (lb, 'lb.pkl'), (scaler, 'scaler.pkl')]:
            dump(obj, open(get_path_file(f'model/{filename}'), 'wb'))

    def load_model(self) -> None:
        """
        Load the model state and preprocessing tools.
        """
        model_path = get_path_file('model/mlp.pt')
        self.load_state_dict(torch.load(model_path, map_location=self.device))
        for attr, filename in [('encoder', 'encoder.pkl'), ('lb', 'lb.pkl'), ('scaler', 'scaler.pkl')]:
            path = get_path_file(f'model/{filename}')
            setattr(self, attr, load(open(path, 'rb')))


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    hidden_size: int,
    dropout_rate: float
) -> nn.Module:
    """
    Construct an MLP network.

    Args
    ----
    input_size : int
        Dimension of input features.
    output_size : int
        Number of output classes.
    n_layers : int
        Number of hidden layers.
    hidden_size : int
        Size of each hidden layer.
    dropout_rate : float
        Dropout probability.

    Returns
    -------
    nn.Module
        The constructed MLP network.
    """
    layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
    for _ in range(n_layers):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.Dropout(p=dropout_rate), nn.ReLU()])
    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


class CensusDataset(Dataset):
    """
    Dataset class for census data.

    Parameters
    ----------
    features : np.ndarray
        Processed features.
    labels : np.ndarray
        Labels (integers in {0, 1, ..., n_classes-1}).
    device : str
        Device to store tensors ('cuda' or 'cpu').
    """
    def __init__(self, features: np.ndarray, labels: np.ndarray, device: str):
        self.features = torch.from_numpy(features).to(device).float()
        self.labels = torch.from_numpy(labels).to(device).long()  # LongTensor for CrossEntropyLoss

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple:
        return self.features[idx], self.labels[idx]


def train_model(x_train: np.array, y_train: np.array, tuning: bool = True, random_state: int = 42,
                use_saved_model: bool = False) -> Mlp:
    """
    Trains a machine learning model and returns it.
    :param x_train: Training features
    :param y_train: Training labels
    :param tuning: indicates if optuna will be used for hyperparameters tuning.
    :param random_state: random seed used for splitting data
    :param use_saved_model: indicates if the model already trained will be used as a starting point for training
    :return: Trained machine learning model.
    """

    # If we use the saved model, we use the hyperparameters saved in the yaml file
    if use_saved_model:
        assert tuning is False
    n_classes = len(set(y_train))

    # split between training and eval
    x_train2, x_val, y_train2, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42)

    if tuning:
        # Use optuna to estimate the best hyper-parameters
        print('Tuning hyper parameters...')
        params = hyperparameters_tuning(x_train2, y_train2, x_val, y_val, random_state)
    else:
        # read the hyper-parameters saved
        params = get_hyperparameters()['parameters']
    print('Hyper parameters selected for training:')
    print(params)

    print('training the model...')
    # Now that we've estimated the optimal value for hyper-parameters, we can train the model
    # on all the training data available.
    model = training_session(x_train, y_train, n_classes, **params, epochs=500, hyper_tuning=False,
                             use_saved_model=use_saved_model)
    return model


def training_session(x_train: np.array, y_train: np.array, n_classes: int, epochs: int, hyper_tuning: bool,
                     use_saved_model: bool, **params) -> Mlp:
    """
    Trains the model for a given set of hyperparameters given in params.
    :param x_train: features used for training
    :param y_train: labels used for training
    :param n_classes: number of different labels
    :param epochs: number of epochs used to train the model
    :param hyper_tuning: This boolean indicates if the method is used for hyperparameters optimization. In that case the
    model is set up so that it doesn't print losses during training
    :param use_saved_model: indicates if the model already trained will be used as a starting point for training
    :param params: dictionary with hyperparameters which keys are 'batch_size', 'dropout_rate', 'hidden_dim',
    'learning_rate' and 'n_layers'.
    :return: Trained model.
    """
    model = Mlp(epochs=epochs,
                input_dim=x_train.shape[1],
                n_classes=n_classes,
                hyper_tuning=hyper_tuning,
                **params)
    if use_saved_model:
        model.load_model()
    model.train_model(x_train, y_train)
    return model


def hyperparameters_tuning(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array,
                           random_state: int) -> dict:
    """
    Select optimal hyperparameters using optuna.
    :param x_train: features used for training
    :param y_train: labels for training
    :param x_val: features used for validation
    :param y_val: labels for validation
    :param random_state: random seed used for splitting data
    :return:
    Dictionary with selected hyperparameters
    """
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    objective_used = get_objective(x_train, y_train, x_val, y_val)
    study.optimize(objective_used, n_trials=10)
    best_params = study.best_params
    save_hyperparameters(best_params, random_state)
    return best_params


def objective(trial: optuna.Trial, x_train: np.array, y_train: np.array, x_val: np.array,
              y_val: np.array) -> float:
    """
    Objective function used by Optuna. The criteria used is the f1 score
    :param trial: Optuna Trial object
    :param x_train: features used for training
    :param y_train: labels for training
    :param x_val: features used for validation
    :param y_val: labels for validation
    :return: f1 score
    """
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [5, 10, 25, 50]),
        'n_layers': trial.suggest_categorical('n_layers', [1, 2, 3, 4, 5]),
        'dropout_rate': trial.suggest_categorical('dropout_rate', [0.3, 0.4, 0.5, 0.6, 0.7])
    }
    model = training_session(x_train, y_train, use_saved_model=False, **params, epochs=150, n_classes=2, hyper_tuning=True)
    preds = model.predict(x_val)
    f1 = fbeta_score(y_val, preds, beta=1, zero_division=1)
    return f1


def get_objective(x_train: np.array, y_train: np.array, x_val: np.array, y_val: np.array) -> Callable:
    """
    Returns the optimization function as used by optuna given our dataset. The function returned is a function of trial.
    :param x_train: features used for training
    :param y_train: labels for training
    :param x_val: features used for validation
    :param y_val: labels for validation
    :return:
    optimization function
    """
    return lambda trial: objective(trial, x_train, y_train, x_val, y_val)


def compute_model_metrics(y: np.array, preds: np.array) -> Tuple[float, float, float]:
    """
    Validates the trained machine learning model using precision, recall, and F1.
    :param y: Known labels, binarized.
    :param preds: Predicted labels, binarized.
    :return: tuple (precision, recall, F1)
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model: Mlp, x: np.array) -> np.array:
    """
    Run model inferences and return the predictions.
    :param model: Trained machine learning model.
    :param x: Data used for prediction.
    :return: Predictions from the model
    """

    y_pred = model.predict(x)
    return y_pred


def get_trained_mlp() -> Mlp:
    """
    Return trained model for inference
    :return: Mlp model

    """
    logger.info('Starting get_trained_mlp')
    model = Mlp(use_saved_hyper_params=True)
    model.load_model()
    return model