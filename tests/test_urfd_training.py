"""URFD ingestion must preserve source semantics and prevent recording leakage."""

import zipfile
from pathlib import Path
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from training.download_urfd import discover, transfer, validate
from training.prepare_urfd import allowed_classes, make_windows, recording_splits
from training.train_urfd import binary_metrics, partial_label_loss
from src.action.action_classifier import ACTION_CLASSES


def test_discovery_handles_multiline_links_and_excludes_other_hosts() -> None:
    html = '<a href=\n "data/fall-01-cam0-rgb.zip">RGB</a><a href="data/fall-01-cam1-d.zip">depth</a>'
    html += '<a href="https://evil.example/fall-02-cam0-rgb.zip">no</a>'
    page = "https://example.org/uf.html"
    assert len(discover(html, page)) == 2
    assert len(discover(html, page, depth=False)) == 1


def test_archives_must_pass_integrity(tmp_path: Path) -> None:
    path = tmp_path / "test.zip"
    path.write_bytes(b"not a zip")
    with pytest.raises(zipfile.BadZipFile):
        validate(path)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("frame.png", b"example")
    assert len(validate(path)) == 64


def test_range_resume_appends_without_duplicate_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "file.part"
    path.write_bytes(b"abc")
    replies = iter([(b"de", "bytes 3-4/7"), (b"fg", "bytes 5-6/7")])
    def get(url: str, **kwargs: object) -> object:
        body, content_range = next(replies)
        return nullcontext(SimpleNamespace(status_code=206,
            headers={"Content-Range": content_range, "Content-Length": str(len(body))},
            raise_for_status=lambda: None, iter_content=lambda size: [body]))
    monkeypatch.setattr("training.download_urfd.requests.Session",
                        lambda: nullcontext(SimpleNamespace(get=get)))
    transfer("https://example.org/file.zip", path, 10)
    assert path.read_bytes() == b"abcdefg"


def test_server_ignoring_range_restarts_instead_of_corrupting_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "file.part"
    path.write_bytes(b"stale")
    response = SimpleNamespace(status_code=200, headers={"Content-Length": "3"},
                               raise_for_status=lambda: None, iter_content=lambda size: [b"new"])
    monkeypatch.setattr("training.download_urfd.requests.Session",
                        lambda: nullcontext(SimpleNamespace(get=lambda *a, **k: nullcontext(response))))
    transfer("https://example.org/file.zip", path, 10)
    assert path.read_bytes() == b"new"


def test_deliberate_lying_is_not_labeled_falling_or_immobile() -> None:
    falling = ACTION_CLASSES.index("falling")
    assert falling not in allowed_classes("adl-01", 0)
    assert allowed_classes("adl-01", 1) == [ACTION_CLASSES.index("lying")]
    assert allowed_classes("fall-01", 1) == [ACTION_CLASSES.index("lying")]
    assert allowed_classes("fall-01", 0) == [falling]
    assert len(allowed_classes("adl-01", -1)) > 1


def test_recordings_stay_in_one_split_and_split_is_repeatable() -> None:
    names = [f"{kind}-{i:02d}" for kind, n in (("adl", 40), ("fall", 30)) for i in range(1, n + 1)]
    split = recording_splits(names, 42, 0.6, 0.2)
    assert split == recording_splits(names, 42, 0.6, 0.2)
    assert {s: list(split.values()).count(s) for s in (0, 1, 2)} == {0: 42, 1: 14, 2: 14}


def test_partial_loss_matches_cross_entropy_for_exact_labels() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 2.0]], requires_grad=True)
    target = torch.tensor([1, 2])
    allowed = torch.nn.functional.one_hot(target, 3).bool()
    torch.testing.assert_close(partial_label_loss(logits, allowed), torch.nn.functional.cross_entropy(logits, target))
    with pytest.raises(ValueError):
        partial_label_loss(logits, torch.zeros_like(allowed))


def test_binary_metrics_do_not_hide_false_alarms() -> None:
    result = binary_metrics(np.array([0.8, 0.7, 0.1, 0.2]), np.array([1, 0, 1, 0]), 0.5)
    assert result["tp"] == result["fp"] == result["fn"] == result["tn"] == 1


def test_windows_use_transition_history_but_not_settled_lying(tmp_path: Path) -> None:
    x = np.zeros((8, 17, 3), dtype=np.float32)
    x[..., 2] = 1
    for name in ("fall-01", "adl-01"):
        np.savez(tmp_path / f"{name}.npz", x=x, frame=np.arange(1, 9))
    labels = {name: {i: 0 if i == 3 else 1 for i in range(1, 9)} for name in ("fall-01", "adl-01")}
    cfg = {"window_size": 4, "window_stride": 1, "min_visible_joints": 6,
           "keypoint_confidence": 0.3, "min_valid_fraction": 0.7}
    out = tmp_path / "windows.npz"
    make_windows(tmp_path, labels, cfg, {"fall-01": 0, "adl-01": 2}, out)
    with np.load(out) as data:
        assert data["event"][data["group"] == "adl-01"].sum() == 0
        assert data["event"][data["group"] == "fall-01"].tolist() == [1, 1, 1, 0, 0]
