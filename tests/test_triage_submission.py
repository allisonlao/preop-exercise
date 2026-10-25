"""End-to-end tests for the staged pipeline in ``core.triage_submission``.

The OpenAI client is faked: it answers each document with a canned extraction
keyed by the document text, so every stage after the LLM call (verification,
reconciliation, engine, evidence packet) runs for real. No network.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from openai import OpenAI

import extract
import reconcile
from core import PatientSubmission, TriageOutput, triage_submission
from extract import ExtractionCache
from llm_fixtures import case_00020, patient_reported_anticoagulants
from models import Decision, IssueCategory, ReasonCode
from packet import CheckId

HP_TEXT = """HISTORY & PHYSICAL - PRE-OP
Date of service: 01/20/2026
Planned procedure: carpal tunnel release
PHYSICAL EXAM:
  Vitals 10:05  BP: 118/80, 97.8 F
Electronically signed by Dr. Lee on 01/20/2026."""

CONSENT_TEXT = """INFORMED CONSENT FOR SURGICAL PROCEDURE
Procedure: carpal tunnel release
Patient signature: /s/ Grace Lopez (electronically signed 01/22/2026 08:05)"""


def _fact(value: Any, quote: str) -> dict[str, Any]:
    return {"value": value, "quote": quote}


def _response(**sections: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "patient_identifiers": [], "hnp_details": None, "consent_details": None, "vitals_mentions": [],
        "medication_mentions": [], "anticoagulation_plans": [], "lab_mentions": [], "date_risk_mentions": [],
    }
    return {**base, **sections}


RESPONSES: dict[str, dict[str, Any]] = {
    HP_TEXT: _response(
        doc_metadata={"document_type": _fact("HISTORY_AND_PHYSICAL", "HISTORY & PHYSICAL - PRE-OP")},
        hnp_details={
            "completion": _fact("COMPLETE", "Electronically signed by Dr. Lee on 01/20/2026."),
            "date_of_service": _fact("01/20/2026", "Date of service: 01/20/2026"),
        },
        vitals_mentions=[
            _fact({"vital_type": "BLOOD_PRESSURE", "raw_value": "118/80", "raw_unit": None, "time_raw": "10:05"},
                  "Vitals 10:05  BP: 118/80"),
        ],
    ),
    CONSENT_TEXT: _response(
        doc_metadata={"document_type": _fact("SURGICAL_CONSENT", "INFORMED CONSENT FOR SURGICAL PROCEDURE")},
        consent_details={
            "signature_states": [_fact("SIGNED", "Patient signature: /s/ Grace Lopez (electronically signed 01/22/2026 08:05)")],
            "procedure": _fact("carpal tunnel release", "Procedure: carpal tunnel release"),
        },
    ),
}


class FakeCompletions:
    """``client.chat.completions`` stand-in answering by document text (RESPONSES by default)."""

    def __init__(self, responses: dict[str, dict[str, Any]] | None = None) -> None:
        self.calls = 0
        self.responses = RESPONSES if responses is None else responses

    def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls += 1
        wrapped: str = kwargs["messages"][1]["content"]
        text = wrapped.removeprefix("<untrusted_document>\n").removesuffix("\n</untrusted_document>")
        content = json.dumps(self.responses[text])
        message = SimpleNamespace(content=content, parsed=object(), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def fake_client(completions: FakeCompletions) -> OpenAI:
    return cast(OpenAI, SimpleNamespace(chat=SimpleNamespace(completions=completions)))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # No real API key and no access to the real data/llm_cache.json.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(extract, "_default_cache", ExtractionCache(tmp_path / "default_cache.json"))


@pytest.fixture
def cache(tmp_path: Path) -> ExtractionCache:
    return ExtractionCache(tmp_path / "llm_cache.json")


@pytest.fixture
def submission() -> dict[str, Any]:
    return {
        "patient": {"id": "patient-1", "mrn": "MRN-1", "name": {"given": "Grace", "family": "Lopez"}},
        "procedure": {
            "case_id": "case-1",
            "procedure_type": "Elective carpal tunnel release",
            "procedure_risk": "LOW",
            "procedure_date": "2026-02-01",
        },
        "vitals": [
            {"type": "blood_pressure", "systolic": 124, "diastolic": 78, "date": "2026-01-25T09:00:00Z"},
            {"type": "temperature", "value_f": 98.1, "date": "2026-01-25T09:05:00Z"},
        ],
        "labs": [{"id": "l1", "code": "CBC", "display": "Complete blood count",
                  "effective_at": "2026-01-20T08:00:00Z", "status": "final"}],
        "medications": [{"name": "ASA 81 mg PO daily", "active": True}],
        "documents": [
            {"doc_id": "hp-1", "type": "H&P", "date": "2026-01-20", "text": HP_TEXT},
            {"doc_id": "consent-1", "type": "Consent", "date": "2026-01-22", "text": CONSENT_TEXT},
        ],
    }


def canonical(output: TriageOutput) -> str:
    return json.dumps(output.model_dump(), sort_keys=True, separators=(",", ":"))


# =============================================================================
# READY path and the passed-checks packet
# =============================================================================


def test_ready_case_lists_why_every_check_passed(submission: dict[str, Any], cache: ExtractionCache) -> None:
    completions = FakeCompletions()
    output = triage_submission(submission, model="gpt-4o", client=fake_client(completions), cache=cache)

    assert output.decision == Decision.READY, output.explanation
    assert output.issues == []
    assert completions.calls == 2

    checks = {check.check: check for check in output.passed_checks}
    assert list(checks) == sorted(checks)
    assert set(checks) == {
        "R0_PROCEDURE_DATA", "R0_SOURCES_READABLE", "R1_CONSENT", "R1_HISTORY_AND_PHYSICAL",
        "R2_CBC", "R3_ANTICOAGULATION", "R4_BLOOD_PRESSURE", "R4_TEMPERATURE",
    }
    assert "Completed H&P dated 2026-01-20 (12 days before" in checks[CheckId.HISTORY_AND_PHYSICAL].description
    assert "Most recent BP: 124/78 mmHg on 2026-01-25 09:00 UTC" in checks[CheckId.BLOOD_PRESSURE].description
    assert "98.1 F on 2026-01-25 09:05 UTC" in checks[CheckId.TEMPERATURE].description
    assert "CBC collected 2026-01-20 (12 days before" in checks[CheckId.CBC].description
    assert "aspirin do not require a plan" in checks[CheckId.ANTICOAGULATION].description

    hp_evidence = checks[CheckId.HISTORY_AND_PHYSICAL].evidence
    assert {e.source for e in hp_evidence} == {"doc:hp-1"}
    assert "Date of service: 01/20/2026" in {e.quote for e in hp_evidence}
    (bp_evidence,) = checks[CheckId.BLOOD_PRESSURE].evidence
    assert bp_evidence.source == "json:vitals[0]"
    assert bp_evidence.date == "2026-01-25"
    assert bp_evidence.raw_value is not None and '"systolic": 124' in bp_evidence.raw_value

    assert output.explanation.startswith("READY: every policy check passed.")
    assert "Completed H&P dated 2026-01-20" in output.explanation


def test_output_is_json_native_and_round_trips(submission: dict[str, Any], cache: ExtractionCache) -> None:
    output = triage_submission(submission, client=fake_client(FakeCompletions()), cache=cache)
    reparsed = TriageOutput.model_validate(json.loads(canonical(output)))
    assert reparsed == output


def test_patient_submission_model_is_accepted(submission: dict[str, Any], cache: ExtractionCache) -> None:
    output = triage_submission(PatientSubmission.model_validate(submission), client=fake_client(FakeCompletions()), cache=cache)
    assert output.decision == Decision.READY


# =============================================================================
# Issues and evidence formatting
# =============================================================================


def test_issues_are_sorted_and_evidence_is_chronological(submission: dict[str, Any], cache: ExtractionCache) -> None:
    submission["vitals"].append({"type": "blood_pressure", "systolic": 190, "diastolic": 112, "date": "2026-01-30T08:00:00Z"})
    submission["labs"] = []
    submission["medications"].append({"name": "apixaban 5 mg PO BID", "active": True})
    output = triage_submission(submission, client=fake_client(FakeCompletions()), cache=cache)

    assert output.decision == Decision.NOT_CLEARED
    keys = [(issue.category, issue.reason_code) for issue in output.issues]
    assert keys == sorted(keys)
    assert {issue.reason_code for issue in output.issues} == {
        ReasonCode.SYSTOLIC_BP_HIGH, ReasonCode.DIASTOLIC_BP_HIGH, ReasonCode.CBC_MISSING, ReasonCode.ANTICOAG_PLAN_MISSING,
    }
    for issue in output.issues:
        assert issue.evidence and issue.resolution and issue.description
        dated = [e.date for e in issue.evidence if e.date is not None]
        assert dated == sorted(dated)
        assert all(e.date is not None for e in issue.evidence[: len(dated)])  # undated evidence last
    high = next(i for i in output.issues if i.reason_code == ReasonCode.SYSTOLIC_BP_HIGH)
    assert [e.source for e in high.evidence] == ["json:vitals[2]"]
    assert "R4_BLOOD_PRESSURE" not in {c.check for c in output.passed_checks}
    assert "R4_TEMPERATURE" in {c.check for c in output.passed_checks}


def test_no_api_key_and_no_cache_fails_closed(submission: dict[str, Any]) -> None:
    output = triage_submission(submission)  # default client: None (no key); empty cache

    assert output.decision == Decision.NEEDS_FOLLOW_UP
    failed = next(i for i in output.issues if i.reason_code == ReasonCode.EXTRACTION_FAILED)
    assert failed.category == IssueCategory.MISSING_REQUIRED_DATA
    assert {e.source for e in failed.evidence} == {"doc:hp-1", "doc:consent-1"}
    assert all("extraction failed" in (e.detail or "") for e in failed.evidence)
    assert "R0_SOURCES_READABLE" not in {c.check for c in output.passed_checks}


def test_document_without_text_or_id_fails_closed(submission: dict[str, Any], cache: ExtractionCache) -> None:
    submission["documents"].append({"doc_id": "blank", "type": "Note", "date": "2026-01-23", "text": "  "})
    submission["documents"].append({"type": "Orphan note", "date": "2026-01-24", "text": "No id."})
    output = triage_submission(submission, client=fake_client(FakeCompletions()), cache=cache)
    (failed,) = [i for i in output.issues if i.reason_code == ReasonCode.EXTRACTION_FAILED]
    assert {e.source for e in failed.evidence} == {"doc:blank", "json:documents[3]"}


# =============================================================================
# case_00020: post-op deferral extracted as RESUME
# =============================================================================


def test_case_00020_deferral_is_caught_even_when_extracted_as_resume(
    cache: ExtractionCache, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    case = case_00020()
    replay = case["replay_responses_resume_on_deferral"]
    by_text = {doc["text"]: replay[doc["doc_id"]] for doc in case["submission"]["documents"]}
    client = fake_client(FakeCompletions(by_text))

    with caplog.at_level(logging.WARNING, logger="reconcile"):
        output = triage_submission(case["submission"], client=client, cache=cache)

    assert output.decision == case["label"]["decision"] == Decision.NEEDS_FOLLOW_UP
    assert {issue.category for issue in output.issues} == set(case["label"]["categories"])
    (issue,) = output.issues
    assert issue.reason_code == ReasonCode.ANTICOAG_PLAN_POSTOP_MISSING
    (plan_evidence,) = [e for e in issue.evidence if (e.detail or "").startswith("plan for warfarin")]
    assert plan_evidence.detail is not None and "post-op UNCLEAR [extracted as RESUME]" in plan_evidence.detail
    assert case["deferral_sentence"] in (plan_evidence.quote or "")
    assert "post-op action for warfarin downgraded from RESUME to UNCLEAR" in caplog.text

    # The same extraction without the quote check is the original false READY.
    monkeypatch.setattr(reconcile, "check_instruction", lambda action, quote: action)
    assert triage_submission(case["submission"], client=client, cache=cache).decision == Decision.READY


# =============================================================================
# Patient-reported anticoagulant missing from the medication list
# =============================================================================

REPORTED_CASES = patient_reported_anticoagulants()


def run_reported_case(case_id: str, status: str, cache: ExtractionCache) -> TriageOutput:
    """Replay the case's real extractions with the reported-use sentence set to ``status``."""

    case = copy.deepcopy(REPORTED_CASES[case_id])
    for response in case["responses"].values():
        for mention in response["medication_mentions"]:
            if mention["quote"] == case["reported_quote"]:
                mention["value"]["status"] = status
    by_text = {doc["text"]: case["responses"][doc["doc_id"]] for doc in case["submission"]["documents"]}
    return triage_submission(case["submission"], client=fake_client(FakeCompletions(by_text)), cache=cache)


@pytest.mark.parametrize("case_id", sorted(REPORTED_CASES))
def test_patient_reported_anticoagulant_as_current_matches_label(case_id: str, cache: ExtractionCache) -> None:
    label = REPORTED_CASES[case_id]["label"]
    output = run_reported_case(case_id, "CURRENT", cache)
    assert output.decision == label["decision"], output.explanation
    assert {issue.category for issue in output.issues} == set(label["categories"])


@pytest.mark.parametrize("case_id", ["case_00000", "case_00027"])
def test_patient_reported_anticoagulant_as_unclear_fails_closed(case_id: str, cache: ExtractionCache) -> None:
    # What the model returned before the prompt rule: never READY, but the wrong category.
    assert REPORTED_CASES[case_id]["extracted_status"] == "UNCLEAR"
    output = run_reported_case(case_id, "UNCLEAR", cache)
    assert output.decision == Decision.NEEDS_FOLLOW_UP
    assert {issue.reason_code for issue in output.issues} == {ReasonCode.ANTICOAG_STATUS_UNKNOWN}


# =============================================================================
# Determinism
# =============================================================================


def test_ten_runs_are_byte_identical_and_cache_replays_without_a_client(
    submission: dict[str, Any], cache: ExtractionCache
) -> None:
    completions = FakeCompletions()
    first = canonical(triage_submission(submission, client=fake_client(completions), cache=cache))
    replays = {canonical(triage_submission(copy.deepcopy(submission), client=None, cache=cache)) for _ in range(10)}
    assert replays == {first}
    assert completions.calls == 2  # one call per document; every later run is a cache hit
