"""
Implementation of a feed-forward network (multi-layer perceptron, or MLP).
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from pickle import dump, load
from .data import get_path_file, get_hyperparameters
import logging

# Configure logging
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