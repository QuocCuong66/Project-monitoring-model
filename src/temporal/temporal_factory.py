"""Temporal Model Factory and Registry for action classification architectures."""

from typing import Callable, Dict, Any

try:
    import torch
    import torch.nn as nn
    from src.temporal.temporal_model import SpatialTemporalTransformer, TCNSequenceClassifier
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    SpatialTemporalTransformer = None
    TCNSequenceClassifier = None
    TORCH_AVAILABLE = False

_TEMPORAL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_temporal_model(name: str) -> Callable:
    def decorator(cls: Callable[..., Any]) -> Callable[..., Any]:
        _TEMPORAL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


if TORCH_AVAILABLE and SpatialTemporalTransformer is not None:
    register_temporal_model("st_transformer")(SpatialTemporalTransformer)
if TORCH_AVAILABLE and TCNSequenceClassifier is not None:
    register_temporal_model("tcn")(TCNSequenceClassifier)


if TORCH_AVAILABLE and nn is not None:
    class BiLSTMTemporalModel(nn.Module):
        """Bidirectional LSTM alternative for temporal sequence modeling."""

        def __init__(self, input_dim: int = 51, num_classes: int = 10, hidden_dim: int = 64, num_layers: int = 2, **kwargs) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, x):
            if x.dim() == 4:
                b, t, k, c = x.shape
                x = x.view(b, t, k * c)
            out, _ = self.lstm(x)
            # Last time step
            last_out = out[:, -1, :]
            return self.fc(last_out)

    register_temporal_model("bilstm")(BiLSTMTemporalModel)


def create_temporal_model(architecture: str = "st_transformer", **kwargs: Any) -> nn.Module:
    """Instantiate a temporal neural network matching architecture name in configs/model.yaml."""
    arch_key = architecture.lower()
    if arch_key not in _TEMPORAL_REGISTRY:
        available = list(_TEMPORAL_REGISTRY.keys())
        raise ValueError(f"Unknown temporal architecture '{architecture}'. Available: {available}")

    return _TEMPORAL_REGISTRY[arch_key](**kwargs)
