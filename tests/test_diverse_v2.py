"""Data provenance, holdouts and false-alarm accounting for the new phase labels."""

import io
import copy
import zipfile

import numpy as np
import pytest

from training.prepare_diverse_v2 import canonical_rows, parse_annotation, subject_split, upsample_timeline, phase_interval
from training.train_diverse_v2 import native_metrics
from training.promote_diverse_v2 import heldout_safe


SOURCE = {"train_subjects": [1], "val_subjects": [2], "test_subjects": [3]}


def test_subject_holdout_covers_every_camera_and_phase() -> None:
    a = parse_annotation(r"Subject_2\Subject_2_ADL_01_5_0.500_2.800_2_1_GMDCSA24.avi 5", SOURCE)
    b = parse_annotation(r"Subject_2\Subject_2_ADL_01_6_2.800_5.000_2_2_GMDCSA24.avi 6", SOURCE)
    assert a["split"] == b["split"] == 1
    assert a["group"] == b["group"]
    assert a["recording"] == "Subject_2_ADL_01"
    with pytest.raises(ValueError):
        subject_split(9, SOURCE)
    with pytest.raises(ValueError):
        subject_split(1, {**SOURCE, "val_subjects": [1, 2]})


def test_oops_cannot_enter_training() -> None:
    assert subject_split(-1, {"test_only": True}) == 2


def test_low_fps_video_keeps_physical_duration() -> None:
    times = np.arange(11)/5  # Two seconds measured at 5 Hz.
    grid, index = upsample_timeline(times, 15)
    assert len(grid) == 31 and grid[-1] == times[-1]
    assert np.max(np.abs(times[index]-grid)) <= 1/10
    assert np.all(np.diff(index) >= 0)


def test_audited_tail_export_is_cropped_to_labeled_phase() -> None:
    row = {"start": 25.943, "end": 27.350, "overlong_tail_export": True}
    assert phase_interval(row, 27.333, .15) == (0., 27.350-25.943)
    assert phase_interval(row, 1.4, .15) == (0., 27.350-25.943)
    with pytest.raises(ValueError, match="disagrees"):
        phase_interval({**row, "overlong_tail_export": False}, 10., .15)
    with pytest.raises(ValueError, match="disagrees"):
        phase_interval(row, .3, .15)


@pytest.mark.parametrize("line", [
    "../clip_1_0.0_1.0_1_1_test.avi 1",
    "C:/clip_1_0.0_1.0_1_1_test.avi 1",
    "clip_1_0.0_1.0_1_1_test.avi 6",
    "clip_1_2.0_1.0_1_1_test.avi 1",
])
def test_reject_unsafe_or_inconsistent_annotations(line: str) -> None:
    with pytest.raises(ValueError):
        parse_annotation(line, SOURCE)


def archive(rows: list[tuple[str, int, bytes]]) -> zipfile.ZipFile:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        z.writestr("release/mmaction2_annotations.txt", "\n".join(f"{path} {label}" for path, label, _ in rows))
        for path, _, content in rows:
            z.writestr("release/clips/"+path, content)
        z.writestr("release/clips/unreferenced.avi", b"alternate export")
    stream.seek(0)
    return zipfile.ZipFile(stream)


def test_canonical_only_and_postfall_context() -> None:
    with archive([("record_1_0.0_1.0_1_1_test.avi", 1, b"impact"),
                  ("record_6_1.0_2.0_1_1_test.avi", 6, b"lying"),
                  ("other_6_0.0_2.0_1_1_test.avi", 6, b"voluntary")]) as z:
        rows, report = canonical_rows(z, SOURCE)
    assert len(rows) == 3 and report["unreferenced_exports"] == 1
    assert not rows[1]["normal_recording"]
    assert rows[2]["normal_recording"]


def test_duplicate_content_cannot_leak_into_heldout_subject() -> None:
    with archive([("one_6_0.0_2.0_1_1_test.avi", 6, b"same video"),
                  ("two_6_0.0_2.0_3_1_test.avi", 6, b"same video")]) as z:
        with pytest.raises(ValueError, match="conflicting split"):
            canonical_rows(z, SOURCE)


def test_conflicting_duplicate_labels_are_both_quarantined() -> None:
    with archive([("one_1_0.0_2.0_1_1_test.avi", 1, b"same video"),
                  ("two_9_0.0_2.0_1_1_test.avi", 9, b"same video"),
                  ("three_6_0.0_2.0_1_1_test.avi", 6, b"different")]) as z:
        rows, report = canonical_rows(z, SOURCE)
    assert len(rows) == 1 and rows[0]["label"] == 6
    assert len(report["quarantined_conflicting_members"]) == 2


def test_fallen_and_postfall_lying_excluded_from_benign_fpr() -> None:
    # Three emergency-looking postfall windows are never counted as false alarms.
    labels = np.array([1, 2, 6, 5, 6, 8])
    normal_recording = np.array([False, False, False, True, True, True])
    p = np.eye(10)[[5, 9, 9, 3, 3, 1]]
    result = native_metrics({"example": p}, {"example": {"source_label": labels, "normal_recording": normal_recording}})
    source = result["sources"]["example"]
    assert source["fall_recall"] == 1
    assert source["normal_count"] == 3
    assert source["benign_down_count"] == 2
    assert source["normal_fpr"] == source["benign_down_fpr"] == 0


def test_release_rejects_rehearsal_false_positive_or_fall_regression() -> None:
    baseline = {"old": {"new_normal_fpr": .1, "upfall_impact_recall": .7},
                "native": {"sources": {"gmdcsa24": {"normal_fpr": .3, "fall_recall": .6},
                                       "oops": {"normal_fpr": None, "fall_recall": .2}}}}
    candidate = copy.deepcopy(baseline)
    candidate["native"]["sources"]["gmdcsa24"]["normal_fpr"] = .1
    assert heldout_safe(candidate, baseline)
    candidate["old"]["new_normal_fpr"] = .11
    assert not heldout_safe(candidate, baseline)
    candidate = copy.deepcopy(baseline)
    candidate["native"]["sources"]["gmdcsa24"]["fall_recall"] = .59
    assert not heldout_safe(candidate, baseline)
