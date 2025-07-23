import pytest
import numpy as np
from torch import nn
from ..ml.model import build_mlp
from ..ml.model import Mlp


@pytest.mark.parametrize('n_layers, hidden_size, dropout_rate', [(1, 5, 0.5), (2, 10, 0.0), (3, 20, 0.7)])
def test_build_mlp(n_layers, hidden_size, dropout_rate):
    model = build_mlp(10, 2, n_layers, hidden_size, dropout_rate)
    assert len(model) == 3 * n_layers + 3 
    # Check dropout p in hidden layers
    for i in range(2, len(model), 3):  # Dropout positions
        assert isinstance(model[i], nn.Dropout)
        assert model[i].p == dropout_rate


def test_mlp_inference():
    """
    Test if the model returns expected output when used for inference
    """
    model = Mlp(n_layers=2, hidden_dim=5, n_classes=2, input_dim=10)
    n_examples = 300
    data = np.random.rand(n_examples, 10)
    output = model.predict(data)
    assert output.shape[0] == n_examples

