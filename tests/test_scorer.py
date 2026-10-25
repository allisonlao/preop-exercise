"""Unit tests for the offline scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from models import Decision, IssueCategory
from scorer import (
    INVALID,
    LabeledCase,
    Prediction,
    build_report,
    load_predictions,
    main,
    parse_prediction_row,
    score_all,
)

DOC = IssueCategory.REQUIRED_DOCUMENTATION
TEST = IssueCategory.REQUIRED_TESTING
ACUTE = IssueCategory.ACUTE_SAFETY_EXCLUSION


def labeled(case_id: str, decision: Decision, categories: list[IssueCategory]) -> LabeledCase:
    return LabeledCase.model_validate(
        {"case_id": case_id, "submission": {}, "label": {"decision": decision, "categories": categories}}
    )


def output(decision: str, categories: list[IssueCategory]) -> dict[str, Any]:
    return {"decision": decision, "issues": [{"category": c.value, "evidence": "free text"} for c in categories]}


@pytest.fixture
def labels() -> list[LabeledCase]:
    return [
        labeled("a", Decision.READY, []),
        labeled("b", Decision.NEEDS_FOLLOW_UP, [DOC]),
        labeled("c", Decision.NOT_CLEARED, [ACUTE, TEST]),
        labeled("d", Decision.NEEDS_FOLLOW_UP, [TEST]),
    ]


def test_confusion_matrix_false_ready_and_invalid(labels: list[LabeledCase]) -> None:
    predictions = {
        "a": Prediction("a", output("READY", [])),
        "b": Prediction("b", output("READY", [])),  # false READY
        "c": Prediction("c", output("NEEDS_FOLLOW_UP", [TEST])),
        # "d" missing -> INVALID
    }
    summary = score_all(labels, predictions)
    assert summary.confusion["READY"]["READY"] == 1
    assert summary.confusion["NEEDS_FOLLOW_UP"]["READY"] == 1
    assert summary.confusion["NOT_CLEARED"]["NEEDS_FOLLOW_UP"] == 1
    assert summary.confusion["NEEDS_FOLLOW_UP"][INVALID] == 1
    assert summary.false_ready_case_ids == ["b"]
    assert summary.invalid_case_ids == ["d"]
    assert summary.decision_accuracy_pct == 25.0


def test_category_precision_and_recall(labels: list[LabeledCase]) -> None:
    predictions = {
        "a": Prediction("a", output("NEEDS_FOLLOW_UP", [DOC])),  # FP for DOC
        "b": Prediction("b", output("NEEDS_FOLLOW_UP", [DOC])),  # TP for DOC
        "c": Prediction("c", output("NOT_CLEARED", [ACUTE])),  # TP ACUTE, FN TEST
        "d": Prediction("d", output("NEEDS_FOLLOW_UP", [TEST])),  # TP TEST
    }
    metrics = score_all(labels, predictions).categories
    assert (metrics[DOC].precision, metrics[DOC].recall) == (0.5, 1.0)
    assert (metrics[TEST].precision, metrics[TEST].recall) == (1.0, 0.5)
    assert (metrics[ACUTE].precision, metrics[ACUTE].recall) == (1.0, 1.0)
    missing = metrics[IssueCategory.MISSING_REQUIRED_DATA]
    assert (missing.precision, missing.recall, missing.support) == (None, None, 0)


def test_invalid_enums_make_output_unscorable(labels: list[LabeledCase]) -> None:
    predictions = {"a": Prediction("a", {"decision": "MAYBE", "issues": []})}
    (score,) = score_all(labels, predictions, only_predicted=True).scores
    assert score.predicted_decision == INVALID
    assert not score.scorable
    assert not score.is_false_ready


def test_lenient_output_is_scorable_but_not_strict(labels: list[LabeledCase]) -> None:
    (score,) = score_all(labels, {"b": Prediction("b", output("NEEDS_FOLLOW_UP", [DOC]))}, only_predicted=True).scores
    assert score.scorable and score.categories_match
    assert not score.strict_schema_valid


def test_unknown_case_ids_are_rejected(labels: list[LabeledCase]) -> None:
    with pytest.raises(ValueError, match="unknown case_ids"):
        score_all(labels, {"zzz": Prediction("zzz", output("READY", []))})


def test_parses_baseline_rows_and_bare_outputs() -> None:
    baseline = parse_prediction_row({"case_id": "a", "output": None, "error": "timeout"})
    assert baseline.output is None and baseline.error == "timeout"
    bare = parse_prediction_row({"case_id": "a", **output("READY", [])})
    assert bare.output is not None and bare.output["decision"] == "READY"


def test_duplicate_predictions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "out.jsonl"
    row = json.dumps({"case_id": "a", "output": output("READY", [])})
    path.write_text(f"{row}\n{row}\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(path)


def test_report_is_viewer_compatible(labels: list[LabeledCase]) -> None:
    predictions = {"a": Prediction("a", output("READY", []))}
    summary = score_all(labels, predictions)
    report = build_report(summary, labels, predictions)
    assert report["primary_score"]["name"] == "aggregate_local_score_pct"
    record = report["records"][0]
    assert set(record["metrics"]) == {"json_schema_valid", "decision_match_oracle", "issue_categories_match_oracle"}
    assert {"record_index", "case_id", "submission", "label", "parsed_output", "aggregate_local_score"} <= set(record)
    assert report["summary"]["false_ready_count"] == 0


def test_cli_exit_code_gates_on_false_ready(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.jsonl"
    outputs_path = tmp_path / "outputs.jsonl"
    labels_path.write_text(
        json.dumps({"case_id": "a", "submission": {}, "label": {"decision": "NOT_CLEARED", "categories": ["ACUTE_SAFETY_EXCLUSION"]}})
        + "\n"
    )
    outputs_path.write_text(json.dumps({"case_id": "a", "output": output("READY", [])}) + "\n")
    args = ["--labels", str(labels_path), "--outputs", str(outputs_path), "--report", str(tmp_path / "r.json")]
    assert main(args) == 0
    assert main([*args, "--fail-on-false-ready"]) == 2
