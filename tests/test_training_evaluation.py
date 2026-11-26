"""Regression checks for real-data evaluation and hard-negative coverage."""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from training.evaluate import load_split
from training.prepare_ntu import NTU_SIT_DOWN, NTU_SKIP
from training.train_action import evaluate


def test_sitting_down_is_not_discarded() -> None:
    assert NTU_SIT_DOWN == 7
    assert NTU_SIT_DOWN not in NTU_SKIP


def test_real_split_keeps_source_and_never_fabricates_test(tmp_path: Path) -> None:
    path = tmp_path / "windows.npz"
    np.savez(path, x=np.zeros((2, 30, 17, 3)), y=np.array([2, 5]),
             split=np.array([0, 1]), src=np.array([7, 42]))
    x, y, src = load_split(path, 1)
    assert x.shape == (1, 30, 17, 3)
    assert y.tolist() == [5]
    assert src.tolist() == [42]
    with pytest.raises(ValueError, match="No real samples"):
        load_split(path, 2)


def test_invalid_labels_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    np.savez(path, x=np.zeros((1, 30, 17, 3)), y=np.array([10]), split=np.array([1]))
    with pytest.raises(ValueError, match="Invalid action labels"):
        load_split(path, 1)


def test_false_positive_metric_counts_normal_windows() -> None:
    logits = torch.zeros(4, 10)
    logits[range(4), [5, 2, 5, 1]] = 10
    loader = DataLoader(TensorDataset(logits, torch.tensor([2, 2, 5, 5])), batch_size=4)
    metrics = evaluate(torch.nn.Identity(), loader, torch.device("cpu"))
    assert metrics["normal_emergency_fpr"] == 0.5
    assert metrics["emergency_recall"] == 0.5
    assert metrics["emergency_precision"] == 0.5
