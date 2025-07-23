import pytest
import numpy as np
import torch
from torch import nn
from ..ml.model import build_mlp
from ..ml.model import Mlp


@pytest.mark.parametrize('n_layers, dropout_rate', [
    (1, 0.5),  # Standard with dropout
    (2, 0.5),
    (1, 0.0),  # No dropout
    (2, 0.0)
])
def test_build_mlp(n_layers, dropout_rate):
    """
    Test MLP architecture: layer count, types (Linear, ReLU, optional Dropout), and dimensions match inputs.
    """
    input_size = 10
    output_size = 2
    hidden_size = 5
    model = build_mlp(input_size, output_size, n_layers, hidden_size, dropout_rate=dropout_rate)
    # Adjust layer count: For each hidden layer: Linear + ReLU + Dropout (if rate > 0), plus output Linear
    expected_layers = 2 * n_layers + 1 + n_layers * (1 if dropout_rate > 0 else 0) + 1  # +1 for output
    assert len(model) == expected_layers, f"Unexpected number of layers for n_layers={n_layers}, dropout_rate={dropout_rate}"

    assert isinstance(model[0], nn.Linear)
    assert model[0].in_features == input_size
    assert model[0].out_features == hidden_size
    assert isinstance(model[1], nn.ReLU)
    if dropout_rate > 0:
        assert isinstance(model[2], nn.Dropout)
        assert model[2].p == dropout_rate  # Check dropout probability
    assert model[-1].out_features == output_size


def test_mlp_forward_pass():
    """
    Test direct forward pass with tensors to ensure model handles inputs correctly
    """
    model = Mlp(n_layers=2, hidden_dim=5, n_classes=2, input_dim=10)
    model.eval()  # Set to eval mode for consistency
    batch_size = 5
    input_tensor = torch.randn(batch_size, 10)  # Random inputs mimicking processed features
    output = model(input_tensor)
    assert output.shape == (batch_size, 2), "Unexpected output shape"  # Logits for 2 classes
    assert torch.all(output >= -10) and torch.all(output <= 10), "Output logits out of reasonable range (possible uninitialized weights)"



def test_mlp_inference():
    """
    Test if the model returns expected output when used for inference
    """
    input_dim = 105  # Realistic for census: 6 num + OHE cats (~99)
    model = Mlp(n_layers=2, hidden_dim=5, n_classes=2, input_dim=input_dim)
    n_examples = 300
    data = np.random.rand(n_examples, input_dim)
    output = model.predict(data)
    assert output.shape[0] == n_examples
    assert np.all((output >= 0) & (output < 2))  # Classes 0 or 1
    assert len(np.unique(output)) > 1, "All predictions are the same (possible model bias or uninitialized)"  # Ensure variety
