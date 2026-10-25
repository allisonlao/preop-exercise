"""Hand-reconciled facts for sample cases, checked against the human labels.

Each fixture is what a perfect JSON+text merger should produce for that case
(times from note text are taken as UTC). If the engine reproduces the label from
perfect facts, any later mismatch is an extraction/reconciliation problem, not a
policy problem. Cases were chosen to cover each edge pattern in the sample.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from engine import evaluate
from fact_builders import (
    antiplatelet,
    bp,
    cbc,
    cmp,
    consent,
    hp,
    med,
    other_lab,
    plan,
    procedure,
    temp,
    temp_c,
)
from models import LabStatus, MedicationStatus, ReconciledPatientFacts
from scorer import LabeledCase, Prediction, load_labels, score_all

DATA = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"
EIE = LabStatus.ENTERED_IN_ERROR
INACTIVE = MedicationStatus.INACTIVE
UNKNOWN = MedicationStatus.UNKNOWN


def _facts(case_id: str, date: str | None, risk: str | None, **fields: object) -> ReconciledPatientFacts:
    return ReconciledPatientFacts(case_id=case_id, procedure=procedure(date, risk, case_id), **fields)  # type: ignore[arg-type]


GOLD: dict[str, Callable[[], ReconciledPatientFacts]] = {
    # Xarelto only in the nursing med reconciliation; cardiology plan covers it -> READY.
    "case_00000": lambda: _facts(
        "case_00000", "2026-03-01", "LOW",
        history_and_physicals=(hp("2025-07-20", doc="wellness"), hp("2026-02-21")),
        consents=(consent(signed_at="2026-02-27T08:05"),),
        labs=(cbc("2026-02-11T11:29"), cbc("2026-01-06T10:14"), other_lab("2026-02-05T09:15")),
        medications=(med("rivaroxaban"),),
        anticoagulation_plans=(plan(("rivaroxaban",), written_on="2026-02-20"),),
        blood_pressures=(bp(130, 79, "2026-02-17T09:05"), bp(119, 73, "2026-02-24T10:12"), bp(127, 74, "2026-02-21T10:05")),
        temperatures=(temp("97.9", "2026-02-24T10:15"), temp("98.0", "2026-02-21T10:05")),
    ),
    # Fever in Celsius in a nursing note + portal consent not completed.
    "case_00002": lambda: _facts(
        "case_00002", "2026-03-03", "LOW",
        history_and_physicals=(hp("2026-02-07"),),
        consents=(consent(signed=False, on="2026-02-24"),),
        labs=(cbc("2026-01-28T08:00"), cbc("2026-02-26T09:56")),
        medications=(antiplatelet(),),
        blood_pressures=(bp(125, 79, "2026-02-19T09:05"), bp(126, 80, "2026-02-26T10:12"), bp(127, 75, "2026-03-01T08:30")),
        temperatures=(temp("99.1", "2026-02-26T10:15"), temp_c("38.6", "2026-03-01T08:30")),
    ),
    # Newest CBC entered-in-error; fall back to prior CBC. Warfarin plan complete.
    "case_00004": lambda: _facts(
        "case_00004", "2026-03-05", "MODERATE",
        history_and_physicals=(hp("2026-02-24"),),
        consents=(consent(signed_at="2026-03-03T11:30"),),
        labs=(cbc("2026-02-10T07:19", EIE), cbc("2026-02-09T11:35"), cbc("2026-01-06T07:27"), other_lab("2026-01-25T08:59")),
        medications=(med("warfarin"),),
        anticoagulation_plans=(plan(("warfarin",), written_on="2026-02-24"),),
        blood_pressures=(bp(133, 84, "2026-02-28T10:12"), bp(138, 76, "2026-02-21T09:05"), bp(140, 78, "2026-03-02T08:53")),
        temperatures=(temp("98.6", "2026-02-28T10:15"), temp("98.1", "2026-03-02T08:45")),
    ),
    # Pending consent superseded by a later signed one; HIGH-risk labs pass.
    "case_00007": lambda: _facts(
        "case_00007", "2026-03-08", "HIGH",
        history_and_physicals=(hp("2026-02-12"), hp("2025-07-09", doc="wellness")),
        consents=(consent(signed=False, doc="old", on="2026-02-18"), consent(doc="new", signed_at="2026-03-03T16:44")),
        labs=(cmp("2026-02-15T09:35"), cmp("2026-03-06T08:19"), cbc("2026-02-05T08:19"), cbc("2026-03-04T10:46")),
        blood_pressures=(bp(123, 79, "2026-03-03T10:12"), bp(118, 71, "2026-02-24T09:05"), bp(121, 80, "2026-03-06T08:54")),
        temperatures=(temp("97.7", "2026-03-03T10:15"), temp_c("37.2", "2026-03-06T08:45")),
    ),
    # CBC 15 days out for HIGH risk; no temperature anywhere.
    "case_00009": lambda: _facts(
        "case_00009", "2026-03-10", "HIGH",
        history_and_physicals=(hp("2026-02-26"),),
        consents=(consent(signed_at="2026-03-01T11:33"),),
        labs=(cbc("2026-02-23T10:04"), cmp("2026-03-04T11:25")),
        blood_pressures=(bp(116, 78, "2026-02-26T09:05"), bp(131, 78, "2026-03-05T10:12"), bp(128, 76, "2026-03-08T08:52")),
    ),
    # Null procedure date (H&P says "to be determined") and no temperature.
    "case_00011": lambda: _facts(
        "case_00011", None, "MODERATE",
        history_and_physicals=(hp("2026-02-18"),),
        consents=(consent(signed_at="2026-03-07T10:02"),),
        labs=(cbc("2026-02-04T07:53"), cbc("2026-03-08T09:41")),
        medications=(antiplatelet(),),
        blood_pressures=(bp(132, 74, "2026-03-07T10:12"), bp(117, 75, "2026-02-28T09:05"), bp(117, 71, "2026-02-18T10:05")),
    ),
    # H&P and CBC exactly 30 days out.
    "case_00012": lambda: _facts(
        "case_00012", "2026-03-13", "LOW",
        history_and_physicals=(hp("2026-02-11"), hp("2025-06-18", doc="external")),
        consents=(consent(signed_at="2026-03-08T12:21"),),
        labs=(cbc("2026-02-11T10:31"),),
        blood_pressures=(bp(131, 70, "2026-03-01T09:05"), bp(116, 77, "2026-03-08T10:12"), bp(125, 73, "2026-03-10T08:55")),
        temperatures=(temp("97.7", "2026-03-08T10:15"), temp("97.5", "2026-03-10T08:45")),
    ),
    # Only H&P is 31 days out.
    "case_00014": lambda: _facts(
        "case_00014", "2026-03-15", "LOW",
        history_and_physicals=(hp("2026-02-12"),),
        consents=(consent(signed_at="2026-03-11T08:53"),),
        labs=(cbc("2026-02-18T10:00"),),
        blood_pressures=(bp(127, 79, "2026-03-03T09:05"), bp(128, 72, "2026-03-10T10:12")),
        temperatures=(temp("98.5", "2026-03-10T10:15"),),
    ),
    # 189/112 at 08:06, re-measured 137/78 at 08:30 -> recheck governs.
    "case_00016": lambda: _facts(
        "case_00016", "2026-03-17", "LOW",
        history_and_physicals=(hp("2025-06-22", doc="wellness"), hp("2026-02-21")),
        consents=(consent(signed_at="2026-03-13T12:06"),),
        labs=(cbc("2026-03-02T10:55"), cbc("2026-02-07T11:47")),
        medications=(antiplatelet(),),
        blood_pressures=(bp(117, 72, "2026-03-05T09:05"), bp(123, 75, "2026-03-12T10:12"), bp(189, 112, "2026-03-15T08:06"), bp(137, 78, "2026-03-15T08:30")),
        temperatures=(temp("98.8", "2026-03-12T10:15"), temp_c("37.1", "2026-03-15T08:06")),
    ),
    # HIGH risk with a BMP (not a CMP) -> CMP missing.
    "case_00017": lambda: _facts(
        "case_00017", "2026-03-18", "HIGH",
        history_and_physicals=(hp("2026-03-10"), hp("2025-08-18", doc="old")),
        consents=(consent(signed_at="2026-03-10T16:53"),),
        labs=(cbc("2026-02-09T10:15"), other_lab("2026-03-12T11:26"), cbc("2026-03-06T10:41")),
        blood_pressures=(bp(127, 79, "2026-03-13T10:12"), bp(120, 73, "2026-03-06T09:05")),
        temperatures=(temp("97.8", "2026-03-13T10:15"),),
    ),
    # Switched warfarin -> apixaban; only plan on file is for warfarin.
    "case_00018": lambda: _facts(
        "case_00018", "2026-03-19", "HIGH",
        history_and_physicals=(hp("2026-03-05"), hp("2025-10-16", doc="wellness")),
        consents=(consent(signed_at="2026-03-17T11:04"),),
        labs=(cmp("2026-03-13T09:10"), cbc("2026-03-17T09:40"), cbc("2026-02-16T07:42")),
        medications=(med("apixaban"), med("warfarin", INACTIVE)),
        anticoagulation_plans=(plan(("warfarin",), written_on="2026-02-06"),),
        blood_pressures=(bp(117, 78, "2026-03-14T10:12"), bp(119, 73, "2026-03-07T09:05")),
        temperatures=(temp("97.6", "2026-03-14T10:15"),),
    ),
    # Warfarin plan covers pre-op only; post-op "will be addressed after surgery".
    "case_00020": lambda: _facts(
        "case_00020", "2026-03-21", "MODERATE",
        history_and_physicals=(hp("2026-03-09"), hp("2025-05-13", doc="old")),
        consents=(consent(signed_at="2026-03-10T14:48"),),
        labs=(cbc("2026-02-25T07:58"), cbc("2026-01-18T08:38"), other_lab("2026-03-12T11:04")),
        medications=(med("warfarin"), antiplatelet()),
        anticoagulation_plans=(plan(("warfarin",), postop=None, written_on="2026-03-15"),),
        blood_pressures=(bp(129, 76, "2026-03-16T10:12"), bp(127, 78, "2026-03-09T09:05")),
        temperatures=(temp("98.9", "2026-03-16T10:15"),),
    ),
    # Signed consent is for a different procedure (CABG vs thyroidectomy).
    "case_00024": lambda: _facts(
        "case_00024", "2026-03-25", "MODERATE",
        history_and_physicals=(hp("2026-03-08"), hp("2025-05-28", doc="old")),
        consents=(consent(procedure_matches=False, signed_at="2026-03-15T08:27"),),
        labs=(cbc("2026-03-16T07:01"), cbc("2026-02-07T09:24")),
        medications=(antiplatelet(),),
        blood_pressures=(bp(127, 77, "2026-03-20T10:12"), bp(127, 71, "2026-03-13T09:05")),
        temperatures=(temp("98.8", "2026-03-20T10:15"),),
    ),
    # Diastolic exactly 110 + portal consent not completed.
    "case_00028": lambda: _facts(
        "case_00028", "2026-03-29", "MODERATE",
        history_and_physicals=(hp("2026-03-15"),),
        consents=(consent(signed=False, on="2026-03-26"),),
        labs=(cbc("2026-03-08T08:31"),),
        blood_pressures=(bp(148, 110, "2026-03-24T10:12"), bp(142, 79, "2026-03-17T09:05")),
        temperatures=(temp("99.1", "2026-03-24T10:15"),),
    ),
    # Risk level not assigned.
    "case_00029": lambda: _facts(
        "case_00029", "2026-03-30", None,
        history_and_physicals=(hp("2026-03-10"), hp("2025-05-29", doc="old")),
        consents=(consent(signed_at="2026-03-25T15:26"),),
        labs=(cbc("2026-03-25T07:10"), cmp("2026-03-28T10:43")),
        blood_pressures=(bp(123, 74, "2026-03-25T10:12"), bp(127, 70, "2026-03-18T09:05")),
        temperatures=(temp("97.4", "2026-03-25T10:15"),),
    ),
    # 37.9 C = 100.22 F, under the threshold.
    "case_00030": lambda: _facts(
        "case_00030", "2026-03-31", "MODERATE",
        history_and_physicals=(hp("2026-03-09"),),
        consents=(consent(signed_at="2026-03-26T14:14"),),
        labs=(cbc("2026-03-29T11:26"),),
        blood_pressures=(bp(129, 72, "2026-03-19T09:05"), bp(117, 77, "2026-03-26T10:12"), bp(127, 73, "2026-03-29T08:50")),
        temperatures=(temp("98.8", "2026-03-26T10:15"), temp_c("37.9", "2026-03-29T08:50")),
    ),
    # Recheck still 181/98 -> systolic exclusion; no CBC at all.
    "case_00032": lambda: _facts(
        "case_00032", "2026-04-02", "LOW",
        history_and_physicals=(hp("2026-03-25"), hp("2025-06-13", doc="external")),
        consents=(consent(signed_at="2026-03-31T14:56"),),
        labs=(other_lab("2026-03-07T09:11"),),
        medications=(antiplatelet(),),
        blood_pressures=(bp(131, 76, "2026-03-28T10:12"), bp(196, 110, "2026-03-31T09:06"), bp(181, 98, "2026-03-31T09:38")),
        temperatures=(temp("98.5", "2026-03-28T10:15"), temp("98.2", "2026-03-31T09:06")),
    ),
    # Coumadin active:null and patient cannot confirm -> status unknown.
    "case_00033": lambda: _facts(
        "case_00033", "2026-04-03", "MODERATE",
        history_and_physicals=(hp("2026-03-09"),),
        consents=(consent(signed_at="2026-03-24T08:12"),),
        labs=(cbc("2026-03-12T11:47"),),
        medications=(med("warfarin", UNKNOWN),),
        blood_pressures=(bp(118, 76, "2026-03-29T10:12"), bp(121, 71, "2026-03-22T09:05")),
        temperatures=(temp("98.4", "2026-03-29T10:15"),),
    ),
    # Null procedure date; everything else complete.
    "case_00044": lambda: _facts(
        "case_00044", None, "LOW",
        history_and_physicals=(hp("2026-03-20"),),
        consents=(consent(signed_at="2026-04-07T08:27"),),
        labs=(cbc("2026-04-03T07:24"), cbc("2026-03-02T08:11")),
        blood_pressures=(bp(131, 80, "2026-04-02T09:05"), bp(120, 71, "2026-04-09T10:12")),
        temperatures=(temp("98.3", "2026-04-09T10:15"),),
    ),
}


@pytest.fixture(scope="module")
def labels() -> dict[str, LabeledCase]:
    return {case.case_id: case for case in load_labels(DATA)}


@pytest.mark.parametrize("case_id", sorted(GOLD))
def test_engine_reproduces_label_from_perfect_facts(case_id: str, labels: dict[str, LabeledCase]) -> None:
    result = evaluate(GOLD[case_id]())
    label = labels[case_id].label
    assert result.decision == label.decision, label.rationale
    assert {issue.category for issue in result.issues} == set(label.categories), label.rationale


def test_gold_outputs_score_perfectly_with_zero_false_ready(labels: dict[str, LabeledCase]) -> None:
    predictions = {
        case_id: Prediction(case_id, json.loads(evaluate(build()).model_dump_json()))
        for case_id, build in GOLD.items()
    }
    summary = score_all(list(labels.values()), predictions, only_predicted=True)
    assert summary.records == len(GOLD)
    assert summary.false_ready_case_ids == []
    assert summary.decision_accuracy_pct == 100.0
    assert summary.category_set_match_pct == 100.0
    assert summary.strict_schema_valid_pct == 100.0
