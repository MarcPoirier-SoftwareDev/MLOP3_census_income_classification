import pytest
import numpy as np
from torch import nn, Tensor
import torch
from ..ml.model import build_mlp, Mlp, CensusDataset, train_model, training_session, hyperparameters_tuning, objective, get_objective, compute_model_metrics, inference, get_trained_mlp
from ..ml.data import save_hyperparameters, get_hyperparameters  # Assuming these exist
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer, StandardScaler
from sklearn.metrics import fbeta_score
import optuna
import logging
from pathlib import Path

# Fixture for small dummy data
@pytest.fixture
def dummy_data():
    np.random.seed(42)
    x = np.random.rand(20, 10).astype(np.float32)
    y = np.random.randint(0, 2, 20)
    return x, y

# Fixture for temp path mocking
@pytest.fixture
def mock_paths(tmp_path, monkeypatch):
    def mock_get_path_file(filename):
        return str(tmp_path / Path(filename).name)
    monkeypatch.setattr('..ml.model.get_path_file', mock_get_path_file)
    return tmp_path


# Improvement: Already suggested in improvements section, but add this for dropout
@pytest.mark.parametrize('n_layers, hidden_size, dropout_rate', [(1, 5, 0.5), (2, 10, 0.0), (3, 20, 0.7)])
def test_build_mlp_extended(n_layers, hidden_size, dropout_rate):
    model = build_mlp(10, 2, n_layers, hidden_size, dropout_rate)
    assert len(model) == 3 * n_layers + 3  # Fixed formula
    # Check dropout p in hidden layers
    for i in range(2, len(model), 3):  # Dropout positions
        assert isinstance(model[i], nn.Dropout)
        assert model[i].p == dropout_rate
    # Edge: n_layers=0 (should have input + output)
    if n_layers == 0:  # But function assumes >0; test separately if needed


def test_mlp_inference():
    """
    Test if the model returns expected output when used for inference
    """
    model = Mlp(n_layers=2, hidden_dim=5, n_classes=2, input_dim=10)
    n_examples = 300
    data = np.random.rand(n_examples, 10)
    output = model.predict(data)
    assert output.shape[0] == n_examples

# New: Test Mlp init and device
def test_mlp_init(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)  # Force CPU
    model = Mlp(n_layers=1, hidden_dim=5, n_classes=2, input_dim=10, dropout_rate=0.3)
    assert model.device == torch.device('cpu')
    assert model.learning_rate == 0.001  # Default
    assert isinstance(model.loss_fn, nn.CrossEntropyLoss)
    assert isinstance(model.optimizer, torch.optim.Adam)
    # With saved hyperparams (mock get_hyperparameters)
    with patch('..ml.model.get_hyperparameters') as mock_get:
        mock_get.return_value = {'parameters': {'batch_size': 64, 'dropout_rate': 0.4, 'hidden_dim': 10, 'learning_rate': 0.01, 'n_layers': 3}}
        model_saved = Mlp(use_saved_hyper_params=True)
        assert model_saved.batch_size == 64
        assert model_saved.dropout_rate == 0.4

# New: Test forward
def test_mlp_forward():
    model = Mlp(n_layers=1, hidden_dim=5, n_classes=2, input_dim=10)
    x = torch.rand(3, 10)
    output = model.forward(x)
    assert output.shape == (3, 2)  # Logits

# New: Test train_model
def test_mlp_train_model(dummy_data):
    x, y = dummy_data
    model = Mlp(n_layers=1, hidden_dim=5, n_classes=2, input_dim=10, epochs=2, batch_size=10, hyper_tuning=True)
    last_loss = model.train_model(x, y)
    assert last_loss > 0  # Loss should be positive
    # Check no logging in hyper_tuning=True (caplog fixture)
    # With hyper_tuning=False, add caplog to assert logs

# New: Test save and load model
def test_mlp_save_load(mock_paths):
    model = Mlp(n_layers=1, hidden_dim=5, n_classes=2, input_dim=10)
    encoder = OneHotEncoder()
    lb = LabelBinarizer()
    scaler = StandardScaler()
    model.save_model(encoder, lb, scaler)
    # Load into new model
    model2 = Mlp()
    model2.load_model()
    assert model2.encoder is not None
    assert model2.scaler is not None
    # Compare state dicts
    assert torch.equal(model.network[0].weight, model2.network[0].weight)

# New: Test CensusDataset
def test_census_dataset(dummy_data):
    x, y = dummy_data
    dataset = CensusDataset(x, y, 'cpu')
    assert len(dataset) == 20
    features, label = dataset[0]
    assert features.shape == (10,)
    assert label.item() in [0, 1]
    assert isinstance(features, Tensor)
    assert features.device == torch.device('cpu')

# New: Test train_model (full function)
def test_train_model(dummy_data, monkeypatch):
    x, y = dummy_data
    # Mock tuning=False
    with patch('..ml.model.get_hyperparameters') as mock_get:
        mock_get.return_value = {'parameters': {'batch_size': 10, 'dropout_rate': 0.5, 'hidden_dim': 5, 'learning_rate': 0.001, 'n_layers': 1}}
        model = train_model(x, y, tuning=False, use_saved_model=False)
    assert isinstance(model, Mlp)
    # With tuning=True (mock optuna to 1 trial)
    with patch('optuna.create_study') as mock_study:
        mock_trial = MagicMock()
        mock_trial.best_params = {'batch_size': 10, 'dropout_rate': 0.5, 'hidden_dim': 5, 'learning_rate': 0.001, 'n_layers': 1}
        mock_study.return_value = mock_trial
        model_tuned = train_model(x, y, tuning=True)
    # Test use_saved_model=True (assumes load works)

# New: Test training_session
def test_training_session(dummy_data):
    x, y = dummy_data
    params = {'batch_size': 10, 'dropout_rate': 0.5, 'hidden_dim': 5, 'learning_rate': 0.001, 'n_layers': 1}
    model = training_session(x, y, n_classes=2, epochs=2, hyper_tuning=True, use_saved_model=False, **params)
    assert isinstance(model, Mlp)
    # With use_saved_model=True (after saving a mock model)

# New: Test hyperparameters_tuning (mock optuna)
def test_hyperparameters_tuning(dummy_data, monkeypatch):
    x_train, y_train = dummy_data[0][:15], dummy_data[1][:15]
    x_val, y_val = dummy_data[0][15:], dummy_data[1][15:]
    with patch('optuna.create_study') as mock_create:
        mock_study = MagicMock()
        mock_create.return_value = mock_study
        params = hyperparameters_tuning(x_train, y_train, x_val, y_val, random_state=42)
    assert isinstance(params, dict)
    mock_study.optimize.assert_called()

# New: Test objective
def test_objective(dummy_data):
    x_train, y_train = dummy_data[0][:15], dummy_data[1][:15]
    x_val, y_val = dummy_data[0][15:], dummy_data[1][15:]
    trial = optuna.trial.FixedTrial({'learning_rate': 0.001, 'batch_size': 10, 'hidden_dim': 5, 'n_layers': 1, 'dropout_rate': 0.5})
    f1 = objective(trial, x_train, y_train, x_val, y_val)
    assert 0 <= f1 <= 1

# New: Test get_objective
def test_get_objective(dummy_data):
    x_train, y_train = dummy_data[0][:15], dummy_data[1][:15]
    x_val, y_val = dummy_data[0][15:], dummy_data[1][15:]
    obj_func = get_objective(x_train, y_train, x_val, y_val)
    trial = optuna.trial.FixedTrial({'learning_rate': 0.001, 'batch_size': 10, 'hidden_dim': 5, 'n_layers': 1, 'dropout_rate': 0.5})
    f1 = obj_func(trial)
    assert 0 <= f1 <= 1

# New: Test compute_model_metrics
def test_compute_model_metrics():
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0  # 1 true positive / 1 predicted positive
    assert recall == 0.5     # 1 TP / 2 actual positives
    assert fbeta == pytest.approx(0.666, 0.001)
    # Edge: All zeros (zero_division)
    y_zero = np.array([0, 0])
    preds_zero = np.array([0, 0])
    p, r, f = compute_model_metrics(y_zero, preds_zero)
    assert p == 1.0  # As per zero_division=1

# New: Test inference
def test_inference(dummy_data):
    model = Mlp(n_layers=1, hidden_dim=5, n_classes=2, input_dim=10)
    preds = inference(model, dummy_data[0])
    assert preds.shape == (20,)
    assert np.all(np.isin(preds, [0, 1]))

# New: Test get_trained_mlp (mock paths and files)
def test_get_trained_mlp(mock_paths):
    # Create dummy saved files
    dummy_model = Mlp(use_saved_hyper_params=True)
    dummy_model.save_model(OneHotEncoder(), LabelBinarizer(), StandardScaler())
    model = get_trained_mlp()
    assert isinstance(model, Mlp)
    assert model.encoder is not None

# Bonus: Test logging (using caplog)
def test_logging(caplog):
    caplog.set_level(logging.INFO)
    model = Mlp()  # Triggers logger.info
    assert "Initializing MLP" in caplog.text