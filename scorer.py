#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pydantic>=2.8.0",
# ]
# ///

"""Offline scorer: compare triage outputs against labeled cases, no network required.

Replaces the OpenAI Evals upload previously used by ``make evals``. Reports:

* a decision confusion matrix (rows = label, columns = prediction, plus INVALID),
* the false-READY count (predicted READY when the label is not READY; target 0),
* precision / recall per issue category (set-based per case),
* strict-schema validity against ``models.TriageResult``.

It also writes a JSON report that ``view_report.py`` can open: per-record
``metrics`` keep the legacy names (``json_schema_valid``,
``decision_match_oracle``, ``issue_categories_match_oracle``) so the viewer and
``make score`` keep working.

Prediction rows may be either the ``run_baseline.py`` format
(``{"case_id", "output", "error", ...}``) or a bare output object carrying a
``case_id``. Predictions are joined to labels by ``case_id``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from models import Decision, IssueCategory, TriageResult

ROOT = Path(__file__).resolve().parent

INVALID: Final = "INVALID"
DECISIONS: Final[tuple[str, ...]] = tuple(d.value for d in Decision)
PREDICTED_COLUMNS: Final[tuple[str, ...]] = (*DECISIONS, INVALID)
CATEGORIES: Final[tuple[IssueCategory, ...]] = tuple(IssueCategory)

# Legacy per-record metrics, equally weighted, kept for view_report.py / `make score`.
LEGACY_METRICS: Final[tuple[str, ...]] = (
    "json_schema_valid",
    "decision_match_oracle",
    "issue_categories_match_oracle",
)
PRIMARY_SCORE_NAME: Final = "aggregate_local_score_pct"


# =============================================================================
# Input models
# =============================================================================


class CaseLabel(BaseModel):
    decision: Decision
    categories: list[IssueCategory]
    rationale: str | None = None


class LabeledCase(BaseModel):
    model_config = ConfigDict(extra="ignore")

    case_id: str
    submission: dict[str, Any]
    label: CaseLabel


class _LenientIssue(BaseModel):
    model_config = ConfigDict(extra="allow")
    category: IssueCategory


class _LenientOutput(BaseModel):
    """The scored part of any output; evidence format is not constrained here."""

    model_config = ConfigDict(extra="allow")
    decision: Decision
    issues: list[_LenientIssue]


@dataclass(frozen=True)
class Prediction:
    case_id: str
    output: Mapping[str, Any] | None
    error: str | None = None


# =============================================================================
# Loading
# =============================================================================


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def load_labels(path: Path) -> list[LabeledCase]:
    cases = [LabeledCase.model_validate(row) for row in _read_jsonl(path)]
    ids = [case.case_id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate case_ids in labels: {duplicates}")
    return cases


def parse_prediction_row(row: Mapping[str, Any]) -> Prediction:
    case_id = row.get("case_id")
    if not isinstance(case_id, str):
        raise ValueError(f"prediction row is missing a string case_id: {sorted(row)}")
    error = row.get("error")
    error_text = None if error is None else str(error)
    if "output" in row:
        output = row["output"]
        return Prediction(case_id, output if isinstance(output, dict) else None, error_text)
    return Prediction(case_id, row, error_text)


def load_predictions(path: Path) -> dict[str, Prediction]:
    predictions: dict[str, Prediction] = {}
    for row in _read_jsonl(path):
        prediction = parse_prediction_row(row)
        if prediction.case_id in predictions:
            raise ValueError(f"duplicate prediction for case_id {prediction.case_id}")
        predictions[prediction.case_id] = prediction
    return predictions


# =============================================================================
# Per-case scoring
# =============================================================================


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    expected_decision: str
    expected_categories: frozenset[IssueCategory]
    predicted_decision: str  # a Decision value, or INVALID
    predicted_categories: frozenset[IssueCategory]
    scorable: bool  # decision + category enums parse (legacy "json_schema_valid")
    strict_schema_valid: bool  # validates against models.TriageResult
    error: str | None = None

    @property
    def decision_match(self) -> bool:
        return self.predicted_decision == self.expected_decision

    @property
    def categories_match(self) -> bool:
        return self.scorable and self.predicted_categories == self.expected_categories

    @property
    def is_false_ready(self) -> bool:
        return self.predicted_decision == Decision.READY and self.expected_decision != Decision.READY

    @property
    def legacy_metrics(self) -> dict[str, bool]:
        return {
            "json_schema_valid": self.scorable,
            "decision_match_oracle": self.decision_match,
            "issue_categories_match_oracle": self.categories_match,
        }

    @property
    def legacy_score(self) -> float:
        values = self.legacy_metrics.values()
        return 100.0 * sum(values) / len(LEGACY_METRICS)


def score_case(label: CaseLabel, case_id: str, prediction: Prediction | None) -> CaseScore:
    expected_decision = label.decision.value
    expected_categories = frozenset(label.categories)

    def invalid(error: str) -> CaseScore:
        return CaseScore(
            case_id, expected_decision, expected_categories, INVALID, frozenset(), False, False, error
        )

    if prediction is None:
        return invalid("no prediction for this case")
    if prediction.output is None:
        return invalid(prediction.error or "prediction has no output")

    try:
        lenient = _LenientOutput.model_validate(prediction.output)
    except ValidationError as exc:
        return invalid(f"unscorable output: {exc.error_count()} validation error(s)")

    try:
        TriageResult.model_validate(prediction.output)
        strict_ok = True
    except ValidationError:
        strict_ok = False

    return CaseScore(
        case_id=case_id,
        expected_decision=expected_decision,
        expected_categories=expected_categories,
        predicted_decision=lenient.decision.value,
        predicted_categories=frozenset(issue.category for issue in lenient.issues),
        scorable=True,
        strict_schema_valid=strict_ok,
        error=prediction.error,
    )


# =============================================================================
# Aggregate metrics
# =============================================================================


@dataclass(frozen=True)
class CategoryMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def support(self) -> int:
        return self.true_positives + self.false_negatives

    @property
    def precision(self) -> float | None:
        predicted = self.true_positives + self.false_positives
        return None if predicted == 0 else self.true_positives / predicted

    @property
    def recall(self) -> float | None:
        return None if self.support == 0 else self.true_positives / self.support


def confusion_matrix(scores: Iterable[CaseScore]) -> dict[str, dict[str, int]]:
    matrix = {expected: {predicted: 0 for predicted in PREDICTED_COLUMNS} for expected in DECISIONS}
    for score in scores:
        matrix[score.expected_decision][score.predicted_decision] += 1
    return matrix


def category_metrics(scores: Iterable[CaseScore]) -> dict[IssueCategory, CategoryMetrics]:
    """Set-based per case: each (case, category) pair is one TP, FP, or FN."""

    counts = {category: [0, 0, 0] for category in CATEGORIES}
    for score in scores:
        for category in CATEGORIES:
            expected = category in score.expected_categories
            predicted = category in score.predicted_categories
            if expected and predicted:
                counts[category][0] += 1
            elif predicted:
                counts[category][1] += 1
            elif expected:
                counts[category][2] += 1
    return {category: CategoryMetrics(*values) for category, values in counts.items()}


@dataclass(frozen=True)
class ScoreSummary:
    scores: tuple[CaseScore, ...]
    confusion: dict[str, dict[str, int]]
    categories: dict[IssueCategory, CategoryMetrics]

    @classmethod
    def from_scores(cls, scores: Iterable[CaseScore]) -> ScoreSummary:
        frozen = tuple(scores)
        return cls(frozen, confusion_matrix(frozen), category_metrics(frozen))

    @property
    def records(self) -> int:
        return len(self.scores)

    @property
    def false_ready_case_ids(self) -> list[str]:
        return [score.case_id for score in self.scores if score.is_false_ready]

    @property
    def invalid_case_ids(self) -> list[str]:
        return [score.case_id for score in self.scores if score.predicted_decision == INVALID]

    def rate(self, predicate: Callable[[CaseScore], bool]) -> float:
        """Percentage of scored cases for which ``predicate`` holds."""

        if not self.scores:
            return 0.0
        return 100.0 * sum(1 for score in self.scores if predicate(score)) / len(self.scores)

    @property
    def decision_accuracy_pct(self) -> float:
        return self.rate(lambda s: s.decision_match)

    @property
    def category_set_match_pct(self) -> float:
        return self.rate(lambda s: s.categories_match)

    @property
    def scorable_pct(self) -> float:
        return self.rate(lambda s: s.scorable)

    @property
    def strict_schema_valid_pct(self) -> float:
        return self.rate(lambda s: s.strict_schema_valid)

    @property
    def legacy_aggregate_pct(self) -> float:
        if not self.scores:
            return 0.0
        return sum(score.legacy_score for score in self.scores) / len(self.scores)


def score_all(
    labels: Sequence[LabeledCase],
    predictions: Mapping[str, Prediction],
    *,
    only_predicted: bool = False,
) -> ScoreSummary:
    """Score every labeled case. Missing predictions count as INVALID unless ``only_predicted``."""

    unknown = sorted(set(predictions) - {case.case_id for case in labels})
    if unknown:
        raise ValueError(f"predictions reference unknown case_ids: {unknown}")
    selected = [c for c in labels if not only_predicted or c.case_id in predictions]
    return ScoreSummary.from_scores(
        score_case(c.label, c.case_id, predictions.get(c.case_id)) for c in selected
    )


# =============================================================================
# Rendering
# =============================================================================


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:5.1f}%"


def render_text(summary: ScoreSummary) -> str:
    lines: list[str] = [f"Records scored: {summary.records}", ""]

    lines.append("Decision confusion matrix (rows = label, columns = prediction)")
    width = max(len(name) for name in PREDICTED_COLUMNS) + 2
    lines.append("".ljust(width) + "".join(name.rjust(width) for name in PREDICTED_COLUMNS))
    for expected in DECISIONS:
        row = summary.confusion[expected]
        lines.append(expected.ljust(width) + "".join(str(row[p]).rjust(width) for p in PREDICTED_COLUMNS))
    lines.append("")

    false_ready = summary.false_ready_case_ids
    status = "OK" if not false_ready else "FAIL"
    lines.append(f"False READY (target 0): {len(false_ready)} [{status}]")
    if false_ready:
        lines.append("  cases: " + ", ".join(false_ready))
    if summary.invalid_case_ids:
        lines.append(f"Invalid / missing outputs: {len(summary.invalid_case_ids)}")
        lines.append("  cases: " + ", ".join(summary.invalid_case_ids))
    lines.append("")

    lines.append("Issue categories (set-based per case)")
    name_width = max(len(c.value) for c in CATEGORIES) + 2
    lines.append(
        "category".ljust(name_width)
        + "precision".rjust(11) + "recall".rjust(9)
        + "TP".rjust(5) + "FP".rjust(5) + "FN".rjust(5) + "support".rjust(9)
    )
    for category, metrics in summary.categories.items():
        lines.append(
            category.value.ljust(name_width)
            + _pct(metrics.precision).rjust(11) + _pct(metrics.recall).rjust(9)
            + str(metrics.true_positives).rjust(5) + str(metrics.false_positives).rjust(5)
            + str(metrics.false_negatives).rjust(5) + str(metrics.support).rjust(9)
        )
    lines.append("")

    lines.append(f"Decision accuracy:        {summary.decision_accuracy_pct:5.1f}%")
    lines.append(f"Exact category-set match: {summary.category_set_match_pct:5.1f}%")
    lines.append(f"Scorable outputs:         {summary.scorable_pct:5.1f}%")
    lines.append(f"Strict schema valid:      {summary.strict_schema_valid_pct:5.1f}%")
    lines.append(f"Legacy aggregate score:   {summary.legacy_aggregate_pct:5.1f}%")
    return "\n".join(lines)


def build_report(
    summary: ScoreSummary,
    labels: Sequence[LabeledCase],
    predictions: Mapping[str, Prediction],
) -> dict[str, Any]:
    """JSON report; ``records`` follows the shape ``view_report.py`` expects."""

    by_id = {case.case_id: case for case in labels}
    index_of = {case.case_id: index for index, case in enumerate(labels)}
    records: list[dict[str, Any]] = []
    for score in summary.scores:
        case = by_id[score.case_id]
        prediction = predictions.get(score.case_id)
        records.append(
            {
                "record_index": index_of[score.case_id],
                "case_id": score.case_id,
                "submission": case.submission,
                "label": case.label.model_dump(mode="json"),
                "parsed_output": dict(prediction.output) if prediction and prediction.output else None,
                "baseline_error": score.error,
                "parse_error": score.error if not score.scorable else None,
                "metrics": score.legacy_metrics,
                "strict_schema_valid": score.strict_schema_valid,
                "false_ready": score.is_false_ready,
                "aggregate_local_score": score.legacy_score,
            }
        )

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "mode": "eval",
        "scorer": "offline",
        "primary_score": {
            "name": PRIMARY_SCORE_NAME,
            "value_pct": round(summary.legacy_aggregate_pct, 2),
            "goal": "maximize",
        },
        "summary": {
            "records": summary.records,
            "false_ready_count": len(summary.false_ready_case_ids),
            "false_ready_case_ids": summary.false_ready_case_ids,
            "invalid_case_ids": summary.invalid_case_ids,
            "decision_accuracy_pct": round(summary.decision_accuracy_pct, 2),
            "category_set_match_pct": round(summary.category_set_match_pct, 2),
            "strict_schema_valid_pct": round(summary.strict_schema_valid_pct, 2),
            "confusion_matrix": summary.confusion,
            "categories": {
                category.value: {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "tp": metrics.true_positives,
                    "fp": metrics.false_positives,
                    "fn": metrics.false_negatives,
                    "support": metrics.support,
                }
                for category, metrics in summary.categories.items()
            },
        },
        "local_metrics_summary": {
            "records": summary.records,
            PRIMARY_SCORE_NAME: round(summary.legacy_aggregate_pct, 2),
        },
        "records": records,
    }


# =============================================================================
# CLI
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", default=str(ROOT / "data" / "patients_sample_50.jsonl"))
    parser.add_argument("--outputs", default=str(ROOT / "data" / "baseline_outputs.jsonl"))
    parser.add_argument("--report", default=str(ROOT / "data" / "eval_report.json"))
    parser.add_argument(
        "--only-predicted",
        action="store_true",
        help="Score only cases that have a prediction (default: missing predictions are INVALID).",
    )
    parser.add_argument(
        "--fail-on-false-ready",
        action="store_true",
        help="Exit with status 2 when any false READY is found.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    labels = load_labels(Path(args.labels))
    predictions = load_predictions(Path(args.outputs))
    summary = score_all(labels, predictions, only_predicted=args.only_predicted)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(build_report(summary, labels, predictions), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(render_text(summary))
    print(f"\nWrote report -> {report_path}")
    if args.fail_on_false_ready and summary.false_ready_case_ids:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
