"""Tests for Stage 4 reconciliation (reconcile.py).

Extracted documents are built in code and run through the real quote verifier
against their document text, so "verified" / "unverified" arise exactly as in
the pipeline: a quote present in the text verifies, an invented one does not.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

import pytest

from engine import evaluate, plan_gaps
from llm_fixtures import NURSING_DOC_ID, SAMPLE, case_00020, raw_response
from models import (
    Decision,
    IssueCategory,
    MedicationStatus,
    ReasonCode,
    ReconciledPatientFacts,
    TimePrecision,
)
from preprocess import StructuredSubmission, preprocess_submission
from reconcile import (
    parse_raw_clock,
    parse_raw_date,
    procedure_matches,
    reconcile_patient_facts,
    reconcile_with_notes,
)
from schemas import (
    POSTOP_INSTRUCTIONS,
    PREOP_INSTRUCTIONS,
    AnticoagulationPlanMention,
    ConsentDetails,
    ConsentState,
    DateRiskKind,
    DateRiskMention,
    DocMetadata,
    DocumentType,
    ExtractedDocument,
    HnPDetails,
    HnPState,
    MedicationMention,
    MedStatus,
    PostopAction,
    PreopAction,
    QuotedFact,
    VitalMention,
    VitalType,
)
from verify import verify_extracted_document

# =============================================================================
# Builders
# =============================================================================

HP_TEXT = """HISTORY & PHYSICAL - PRE-OP
Date of service: 03/01/2026
Planned procedure: carpal tunnel release
Electronically signed by Dr. Lee on 03/01/2026."""

CONSENT_TEXT = """INFORMED CONSENT FOR SURGICAL PROCEDURE
Procedure: carpal tunnel release
Patient signature: /s/ Grace Lopez (electronically signed 03/02/2026 08:05)"""


def q(value: Any, quote: str) -> QuotedFact[Any]:
    return QuotedFact(value=value, quote=quote)


def extracted(doc_id: str, text: str, doc_type: DocumentType, type_quote: str, **sections: Any) -> ExtractedDocument:
    """An ExtractedDocument run through the real verifier against ``text``."""

    fields: dict[str, Any] = {
        "doc_metadata": DocMetadata(document_type=q(doc_type, type_quote)),
        "patient_identifiers": [],
        "hnp_details": None,
        "consent_details": None,
        "vitals_mentions": [],
        "medication_mentions": [],
        "anticoagulation_plans": [],
        "lab_mentions": [],
        "date_risk_mentions": [],
        "doc_id": doc_id,
        **sections,
    }
    return verify_extracted_document(ExtractedDocument(**fields), text)


def hp_doc(doc_id: str = "hp", date_quote: str = "Date of service: 03/01/2026", date_raw: str = "03/01/2026") -> ExtractedDocument:
    return extracted(
        doc_id, HP_TEXT, DocumentType.HISTORY_AND_PHYSICAL, "HISTORY & PHYSICAL - PRE-OP",
        hnp_details=HnPDetails(
            completion=q(HnPState.COMPLETE, "Electronically signed by Dr. Lee on 03/01/2026."),
            date_of_service=q(date_raw, date_quote),
        ),
    )


def consent_doc(
    doc_id: str = "consent",
    state: ConsentState = ConsentState.SIGNED,
    state_quote: str = "Patient signature: /s/ Grace Lopez (electronically signed 03/02/2026 08:05)",
    type_quote: str = "INFORMED CONSENT FOR SURGICAL PROCEDURE",
    procedure_quote: str = "Procedure: carpal tunnel release",
    text: str = CONSENT_TEXT,
) -> ExtractedDocument:
    return extracted(
        doc_id, text, DocumentType.SURGICAL_CONSENT, type_quote,
        consent_details=ConsentDetails(
            signature_states=[q(state, state_quote)],
            procedure=q("carpal tunnel release", procedure_quote),
        ),
    )


def note_doc(doc_id: str, text: str, **sections: Any) -> ExtractedDocument:
    return extracted(doc_id, text, DocumentType.NURSING_ASSESSMENT, text.splitlines()[0], **sections)


def bp_mention(value: str, time_raw: str | None, quote: str) -> QuotedFact[Any]:
    return q(VitalMention(vital_type=VitalType.BLOOD_PRESSURE, raw_value=value, raw_unit="mmHg", time_raw=time_raw), quote)


def med_mention(name: str, status: MedStatus, quote: str) -> QuotedFact[Any]:
    return q(MedicationMention(drug_name_raw=name, status=status), quote)


def plan_mention(
    drug: str, preop: str | None, postop: str | None, quote: str, *,
    preop_action: PreopAction = PreopAction.HOLD, postop_action: PostopAction = PostopAction.RESUME,
    timing: str | None = "2 days before", postop_timing: str | None = None,
) -> QuotedFact[Any]:
    """A plan whose ``preop`` / ``postop`` texts are the action quotes; None means NOT_ADDRESSED."""

    return q(
        AnticoagulationPlanMention(
            drug_name_raw=drug,
            preop_action=q(preop_action if preop else PreopAction.NOT_ADDRESSED, preop or ""),
            preop_timing_raw=timing if preop else None,
            postop_action=q(postop_action if postop else PostopAction.NOT_ADDRESSED, postop or ""),
            postop_timing_raw=postop_timing,
        ),
        quote,
    )


def submission(
    *,
    documents: list[tuple[str, str]],
    medications: list[dict[str, Any]] | None = None,
    procedure_date: str | None = "2026-03-10",
    vitals: list[dict[str, Any]] | None = None,
) -> StructuredSubmission:
    """A READY-by-default structured submission; ``documents`` is ``(doc_id, date)`` pairs."""

    return preprocess_submission(
        {
            "patient": {"id": "p1", "mrn": "MRN-1", "name": {"given": "Grace", "family": "Lopez"}, "dob": "1950-01-01"},
            "procedure": {
                "case_id": "case_t",
                "procedure_type": "Elective carpal tunnel release",
                "procedure_risk": "LOW",
                "procedure_date": procedure_date,
            },
            "vitals": vitals if vitals is not None else [
                {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "2026-03-01T09:00:00Z"},
                {"type": "temperature", "value_f": 98.2, "date": "2026-03-01T09:05:00Z"},
            ],
            "labs": [{"id": "l1", "code": "CBC", "display": "CBC", "effective_at": "2026-03-02T08:00:00Z", "status": "final"}],
            "medications": medications or [],
            "documents": [{"doc_id": doc_id, "type": doc_id, "date": on, "text": ""} for doc_id, on in documents],
        }
    )


BASE_DOCS = [("hp", "2026-03-01"), ("consent", "2026-03-02")]


def base_extractions() -> list[ExtractedDocument]:
    return [hp_doc(), consent_doc()]


def codes(structured: StructuredSubmission, docs: list[ExtractedDocument]) -> set[ReasonCode]:
    return {issue.reason_code for issue in evaluate(reconcile_patient_facts(structured, docs)).issues}


def test_baseline_fixture_is_ready() -> None:
    result = evaluate(reconcile_patient_facts(submission(documents=BASE_DOCS), base_extractions()))
    assert result.decision is Decision.READY, result.explanation


# =============================================================================
# Procedure date
# =============================================================================


def test_null_json_procedure_date_is_never_filled_from_text() -> None:
    text = "SCHEDULING NOTE\nTarget surgery date: 03/12/2026 (tentative)"
    note = extracted(
        "sched", text, DocumentType.OTHER, "SCHEDULING NOTE",
        date_risk_mentions=[q(DateRiskMention(kind=DateRiskKind.PROCEDURE_DATE, raw_text="03/12/2026"),
                              "Target surgery date: 03/12/2026 (tentative)")],
    )
    structured = submission(documents=[*BASE_DOCS, ("sched", "2026-03-01")], procedure_date=None)
    facts = reconcile_patient_facts(structured, [*base_extractions(), note])
    assert facts.procedure.procedure_date is None
    assert ReasonCode.PROCEDURE_DATE_MISSING in codes(structured, [*base_extractions(), note])


# =============================================================================
# Document resolution
# =============================================================================


def test_hps_are_ordered_most_recent_first() -> None:
    old_text = "HISTORY AND PHYSICAL\nDate of service: 06/22/2025\nSigned."
    old = extracted("old-hp", old_text, DocumentType.HISTORY_AND_PHYSICAL, "HISTORY AND PHYSICAL",
                    hnp_details=HnPDetails(completion=q(HnPState.COMPLETE, "HISTORY AND PHYSICAL\nDate"),
                                           date_of_service=q("06/22/2025", "Date of service: 06/22/2025")))
    facts = reconcile_patient_facts(submission(documents=[*BASE_DOCS, ("old-hp", "2025-06-22")]), [old, *base_extractions()])
    assert [(hp.document_id, hp.performed_on) for hp in facts.history_and_physicals] == [
        ("hp", date(2026, 3, 1)), ("old-hp", date(2025, 6, 22)),
    ]


def test_unverified_hp_date_of_service_is_unknown_not_the_document_date() -> None:
    hp = hp_doc(date_quote="Date of service: 03/05/2026", date_raw="03/05/2026")  # not in the text
    structured = submission(documents=BASE_DOCS)
    (fact,) = reconcile_patient_facts(structured, [hp, consent_doc()]).history_and_physicals
    assert fact.performed_on is None
    assert ReasonCode.HP_DATE_UNKNOWN in codes(structured, [hp, consent_doc()])


def test_unverified_hp_completion_is_not_complete() -> None:
    hp = extracted("hp", HP_TEXT, DocumentType.HISTORY_AND_PHYSICAL, "HISTORY & PHYSICAL - PRE-OP",
                   hnp_details=HnPDetails(completion=q(HnPState.COMPLETE, "Attending attestation complete"),
                                          date_of_service=q("03/01/2026", "Date of service: 03/01/2026")))
    assert ReasonCode.HP_NOT_COMPLETED in codes(submission(documents=BASE_DOCS), [hp, consent_doc()])


def test_unverified_consent_is_ignored_resulting_in_missing_consent() -> None:
    # Every consent quote is invented: the document does not legally exist as a consent.
    hallucinated = consent_doc(
        state_quote="Patient signed the consent form in clinic",
        type_quote="SURGICAL CONSENT - SIGNED",
        procedure_quote="Procedure: carpal tunnel release (left)",
    )
    structured = submission(documents=BASE_DOCS)
    facts = reconcile_patient_facts(structured, [hp_doc(), hallucinated])
    assert facts.consents == ()
    result = evaluate(facts)
    assert result.decision is Decision.NEEDS_FOLLOW_UP
    assert {i.reason_code for i in result.issues} == {ReasonCode.CONSENT_MISSING}


def test_unverified_signature_on_a_real_consent_is_not_signed() -> None:
    unsigned = consent_doc(state_quote="Patient signature: /s/ Grace Lopez (signed in clinic)")
    structured = submission(documents=BASE_DOCS)
    (consent,) = reconcile_patient_facts(structured, [hp_doc(), unsigned]).consents
    assert consent.signed is False
    assert codes(structured, [hp_doc(), unsigned]) == {ReasonCode.CONSENT_UNSIGNED}


def test_most_recent_consent_wins() -> None:
    pending_text = "INFORMED CONSENT FOR SURGICAL PROCEDURE\nProcedure: carpal tunnel release\nPatient signature: pending"
    pending = consent_doc("pending", ConsentState.PENDING, "Patient signature: pending", text=pending_text)
    revoked_text = "INFORMED CONSENT FOR SURGICAL PROCEDURE\nProcedure: carpal tunnel release\nConsent revoked by patient 03/05/2026."
    revoked = consent_doc("revoked", ConsentState.REVOKED, "Consent revoked by patient 03/05/2026.", text=revoked_text)

    # Older pending, newer signed: the signed one counts.
    structured = submission(documents=[*BASE_DOCS, ("pending", "2026-02-20")])
    result = reconcile_with_notes(structured, [*base_extractions(), pending])
    assert [c.document_id for c in result.facts.consents] == ["consent"]
    assert any("superseded" in note for note in result.notes)
    assert evaluate(result.facts).decision is Decision.READY

    # Older signed, newer revoked: the revocation is not masked.
    structured = submission(documents=[*BASE_DOCS, ("revoked", "2026-03-05")])
    assert codes(structured, [*base_extractions(), revoked]) == {ReasonCode.CONSENT_REVOKED}


@pytest.mark.parametrize(
    ("consented", "scheduled", "expected"),
    [
        ("carpal tunnel release", "Elective carpal tunnel release", True),
        ("Carpal Tunnel Release", "carpal tunnel release", True),
        ("coronary artery bypass grafting", "Elective thyroidectomy", False),
        ("knee arthroscopy", "Elective total knee arthroplasty", False),
        ("hernia repair", "Elective inguinal hernia repair", None),
        (None, "Elective thyroidectomy", None),
        ("thyroidectomy", None, None),
    ],
)
def test_procedure_matching(consented: str | None, scheduled: str | None, expected: bool | None) -> None:
    assert procedure_matches(consented, scheduled) is expected


# =============================================================================
# Vitals
# =============================================================================

NURSING_TEXT = """PRE-OP NURSING ASSESSMENT
VITAL SIGNS:
  08:06  NIBP 189/112, T 37.1 C, HR 74"""


def test_newer_note_vital_overrides_older_structured_vital() -> None:
    note = note_doc("nursing", NURSING_TEXT, vitals_mentions=[bp_mention("189/112", "08:06", "08:06  NIBP 189/112")])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-08")])
    facts = reconcile_patient_facts(structured, [*base_extractions(), note])

    newest = facts.blood_pressures[0]
    assert (newest.systolic, newest.diastolic) == (Decimal(189), Decimal(112))
    assert newest.observed_at == datetime(2026, 3, 8, 8, 6, tzinfo=UTC)
    assert newest.precision is TimePrecision.MINUTE
    assert [r.observed_at for r in facts.blood_pressures] == sorted((r.observed_at for r in facts.blood_pressures), reverse=True)

    result = evaluate(facts)
    assert result.decision is Decision.NOT_CLEARED
    assert {i.reason_code for i in result.issues} == {ReasonCode.SYSTOLIC_BP_HIGH, ReasonCode.DIASTOLIC_BP_HIGH}


def test_older_note_vital_does_not_override_newer_structured_vital() -> None:
    note = note_doc("nursing", NURSING_TEXT, vitals_mentions=[bp_mention("189/112", "08:06", "08:06  NIBP 189/112")])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-02-20")])
    assert evaluate(reconcile_patient_facts(structured, [*base_extractions(), note])).decision is Decision.READY


def test_date_only_note_vital_tying_with_structured_vital_is_worst_case() -> None:
    note = note_doc("nursing", NURSING_TEXT, vitals_mentions=[bp_mention("189/112", None, "08:06  NIBP 189/112")])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-01")])  # same day as structured 09:00 BP
    result = reconcile_with_notes(structured, [*base_extractions(), note])

    day_only = [r for r in result.facts.blood_pressures if r.precision is TimePrecision.DAY]
    assert len(day_only) == 1
    assert any("unresolved time tie" in note for note in result.notes)
    assert evaluate(result.facts).decision is Decision.NOT_CLEARED


def test_unverified_newer_vital_fails_closed_but_older_one_is_ignored() -> None:
    fake = note_doc("nursing", NURSING_TEXT, vitals_mentions=[bp_mention("120/70", "08:06", "08:06 BP 120/70")])

    newer = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-08")])
    facts = reconcile_patient_facts(newer, [*base_extractions(), fake])
    assert all(r.systolic != Decimal(120) or r.diastolic != Decimal(70) for r in facts.blood_pressures)
    assert codes(newer, [*base_extractions(), fake]) == {ReasonCode.EXTRACTION_FAILED}

    older = submission(documents=[*BASE_DOCS, ("nursing", "2026-02-01")])
    assert evaluate(reconcile_patient_facts(older, [*base_extractions(), fake])).decision is Decision.READY


def test_celsius_note_temperature_is_converted_exactly() -> None:
    text = "PRE-OP NURSING ASSESSMENT\n09:00 Temp 38.0 °C oral"
    mention = q(VitalMention(vital_type=VitalType.TEMPERATURE, raw_value="38.0", raw_unit="°C", time_raw="09:00"), "09:00 Temp 38.0 °C")
    note = note_doc("nursing", text, vitals_mentions=[mention])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-08")])
    facts = reconcile_patient_facts(structured, [*base_extractions(), note])
    assert facts.temperatures[0].value_f == Decimal("100.4")
    assert evaluate(facts).decision is Decision.READY  # 100.4 is not > 100.4


# =============================================================================
# Medications
# =============================================================================

APIXABAN_JSON_INACTIVE = [{"name": "apixaban 5 mg PO BID", "active": False}]


def test_json_inactive_but_text_current_is_conflict() -> None:
    text = "PRE-OP NURSING ASSESSMENT\n  - Eliquis 5 mg PO BID - taking as prescribed"
    note = note_doc("nursing", text, medication_mentions=[med_mention("Eliquis", MedStatus.CURRENT, "Eliquis 5 mg PO BID - taking as prescribed")])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-08")], medications=APIXABAN_JSON_INACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), note])

    (apixaban,) = facts.medications
    assert apixaban.generic_name == "apixaban"
    assert apixaban.status is MedicationStatus.CONFLICT
    result = evaluate(facts)
    assert result.decision is Decision.NEEDS_FOLLOW_UP
    assert {(i.category, i.reason_code) for i in result.issues} == {
        (IssueCategory.MISSING_REQUIRED_DATA, ReasonCode.ANTICOAG_STATUS_CONFLICT)
    }


def test_latest_note_wins_over_older_history_case_18_pattern() -> None:
    old_text = "PERIOPERATIVE MEDICATION PLAN\nHPI: maintained on warfarin 5 mg PO daily."
    new_text = "PRE-OP NURSING ASSESSMENT\nNote: patient states warfarin was stopped last month."
    old = note_doc("plan", old_text, medication_mentions=[med_mention("warfarin", MedStatus.CURRENT, "maintained on warfarin 5 mg PO daily")])
    new = note_doc("nursing", new_text, medication_mentions=[med_mention("warfarin", MedStatus.STOPPED, "warfarin was stopped last month")])
    structured = submission(
        documents=[*BASE_DOCS, ("plan", "2026-02-06"), ("nursing", "2026-03-08")],
        medications=[{"name": "warfarin 5 mg PO daily", "active": False}],
    )
    (warfarin,) = reconcile_patient_facts(structured, [*base_extractions(), old, new]).medications
    assert warfarin.status is MedicationStatus.INACTIVE


def test_brand_name_only_in_text_is_an_active_anticoagulant() -> None:
    text = "PRE-OP NURSING ASSESSMENT\nPatient also reports taking Xarelto 20 mg daily."
    note = note_doc("nursing", text, medication_mentions=[med_mention("Xarelto", MedStatus.CURRENT, "taking Xarelto 20 mg daily")])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-08")])
    facts = reconcile_patient_facts(structured, [*base_extractions(), note])
    assert [(m.generic_name, m.status) for m in facts.medications] == [("rivaroxaban", MedicationStatus.ACTIVE)]
    assert codes(structured, [*base_extractions(), note]) == {ReasonCode.ANTICOAG_PLAN_MISSING}


def test_unverified_anticoagulant_mention_is_unknown_not_dropped() -> None:
    text = "PRE-OP NURSING ASSESSMENT\nMedications reviewed."
    note = note_doc("nursing", text, medication_mentions=[med_mention("Eliquis", MedStatus.CURRENT, "Eliquis 5 mg BID - taking")])
    structured = submission(documents=[*BASE_DOCS, ("nursing", "2026-03-08")])
    (apixaban,) = reconcile_patient_facts(structured, [*base_extractions(), note]).medications
    assert apixaban.status is MedicationStatus.UNKNOWN


def test_antiplatelets_are_resolved_but_never_need_a_plan() -> None:
    structured = submission(documents=BASE_DOCS, medications=[{"name": "ASA 81 mg PO daily", "active": True}])
    facts = reconcile_patient_facts(structured, base_extractions())
    assert [m.generic_name for m in facts.medications] == ["aspirin"]
    assert evaluate(facts).decision is Decision.READY


# =============================================================================
# Anticoagulation plans
# =============================================================================

APIXABAN_ACTIVE = [{"name": "apixaban 5 mg PO BID", "active": True}]
PLAN_A_TEXT = "HEMATOLOGY CONSULT\n1. Hold Eliquis for the 2 days before surgery. Resume Eliquis 48 hours post-op."
PLAN_B_TEXT = "CARDIOLOGY NOTE\n1. Hold apixaban 5 days before surgery. Resume apixaban on POD 2."


def plan_doc(doc_id: str, text: str, drug: str, preop: str, postop: str, quote: str, **kw: Any) -> ExtractedDocument:
    return extracted(doc_id, text, DocumentType.CONSULT_NOTE, text.splitlines()[0],
                     anticoagulation_plans=[plan_mention(drug, preop, postop, quote, **kw)])


def test_complete_plan_satisfies_active_anticoagulant() -> None:
    plan = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                    "Resume Eliquis 48 hours post-op", "Hold Eliquis for the 2 days before surgery. Resume Eliquis 48 hours post-op.")
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), plan])
    (resolved,) = facts.anticoagulation_plans
    assert resolved.drugs_addressed == ("apixaban",)
    assert resolved.preop_timing_specified and not resolved.has_conflicting_plans
    assert evaluate(facts).decision is Decision.READY


def test_contradictory_plans_for_same_drug_are_flagged() -> None:
    a = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                 "Resume Eliquis 48 hours post-op", "Hold Eliquis for the 2 days before surgery.")
    b = plan_doc("b", PLAN_B_TEXT, "apixaban", "Hold apixaban 5 days before surgery",
                 "Resume apixaban on POD 2", "Hold apixaban 5 days before surgery.", timing="5 days before")
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03"), ("b", "2026-03-04")], medications=APIXABAN_ACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), a, b])
    assert all(plan.has_conflicting_plans for plan in facts.anticoagulation_plans)
    assert codes(structured, [*base_extractions(), a, b]) == {ReasonCode.ANTICOAG_PLAN_CONFLICT}


def test_same_plan_restated_in_two_documents_is_not_a_conflict() -> None:
    restated_text = "NURSING NOTE\nPer hematology: hold Eliquis 48 hours before surgery; resume 48 hours post-op."
    a = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                 "Resume Eliquis 48 hours post-op", "Hold Eliquis for the 2 days before surgery.")
    b = plan_doc("b", restated_text, "Eliquis", "hold Eliquis 48 hours before surgery", "resume 48 hours post-op",
                 "hold Eliquis 48 hours before surgery; resume 48 hours post-op.", timing="48 hours before")
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03"), ("b", "2026-03-04")], medications=APIXABAN_ACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), a, b])
    assert not any(plan.has_conflicting_plans for plan in facts.anticoagulation_plans)
    assert evaluate(facts).decision is Decision.READY


def test_unverified_plan_is_ignored() -> None:
    fake = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis 3 days before surgery", "Resume on POD 1",
                    "Hold Eliquis 3 days before surgery. Resume on POD 1.")
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), fake])
    assert facts.anticoagulation_plans == ()
    assert codes(structured, [*base_extractions(), fake]) == {ReasonCode.ANTICOAG_PLAN_MISSING}


# =============================================================================
# Plan actions: only firm actions count as an instruction (case_00020)
# =============================================================================

CASE_20 = case_00020()
CARDIOLOGY_TEXT = next(
    d["text"] for d in CASE_20["submission"]["documents"] if d["doc_id"] == CASE_20["deferral_doc_id"]
)
CARDIOLOGY_PLAN_QUOTE = CASE_20["observed_v1_anticoagulation_plans"][0]["quote"]
WARFARIN_PREOP = "Stop warfarin 5 days before surgery (last dose 03/15/2026)"
DEFERRAL = CASE_20["deferral_sentence"]
WARFARIN_ACTIVE = [{"name": "Coumadin 5 mg PO daily (INR goal 2-3)", "active": True}]
NON_INSTRUCTION_POSTOP = [PostopAction.DEFERRED, PostopAction.NOT_ADDRESSED, PostopAction.UNCLEAR]
NON_INSTRUCTION_PREOP = [PreopAction.DEFERRED, PreopAction.NOT_ADDRESSED, PreopAction.UNCLEAR]


def warfarin_plan_facts(postop_action: PostopAction, postop_quote: str = DEFERRAL, **kw: Any) -> ReconciledPatientFacts:
    """The real case_00020 cardiology note, with the given post-op action, reconciled on a READY baseline."""

    note = extracted(
        "cardio", CARDIOLOGY_TEXT, DocumentType.CONSULT_NOTE, "CARDIOLOGY PROGRESS NOTE",
        anticoagulation_plans=[plan_mention(
            "warfarin", WARFARIN_PREOP, postop_quote, CARDIOLOGY_PLAN_QUOTE,
            postop_action=postop_action, timing="5 days before surgery", **kw,
        )],
    )
    structured = submission(documents=[*BASE_DOCS, ("cardio", "2026-03-05")], medications=WARFARIN_ACTIVE)
    return reconcile_patient_facts(structured, [*base_extractions(), note])


def assert_postop_missing(facts: ReconciledPatientFacts) -> None:
    (plan,) = facts.anticoagulation_plans
    assert plan.postop_instruction is None
    assert ReasonCode.ANTICOAG_PLAN_POSTOP_MISSING in plan_gaps(plan)
    result = evaluate(facts)
    assert result.decision is Decision.NEEDS_FOLLOW_UP
    assert {issue.category for issue in result.issues} == {IssueCategory.ANTICOAGULATION_MANAGEMENT}


def test_case_00020_deferred_postop_is_a_missing_instruction() -> None:
    facts = warfarin_plan_facts(PostopAction.DEFERRED)
    assert_postop_missing(facts)
    (plan,) = facts.anticoagulation_plans
    assert plan.hedged
    assert plan.preop_instruction == WARFARIN_PREOP and plan.preop_timing_specified
    assert plan.evidence[0].detail == "plan for warfarin: pre-op HOLD (5 days before surgery); post-op DEFERRED"

    (issue,) = evaluate(facts).issues
    assert any(DEFERRAL in (evidence.quote or "") for evidence in issue.evidence)
    assert {issue.category.value} == set(CASE_20["label"]["categories"])


@pytest.mark.parametrize("action", NON_INSTRUCTION_POSTOP)
def test_non_instruction_postop_actions_count_as_missing(action: PostopAction) -> None:
    assert_postop_missing(warfarin_plan_facts(action))


@pytest.mark.parametrize("action", sorted(POSTOP_INSTRUCTIONS))
def test_instruction_postop_actions_count_as_present(action: PostopAction) -> None:
    plan = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                    "Resume Eliquis 48 hours post-op", PLAN_A_TEXT.splitlines()[1], postop_action=action)
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
    assert codes(structured, [*base_extractions(), plan]) == set()


@pytest.mark.parametrize("action", NON_INSTRUCTION_PREOP)
def test_non_instruction_preop_actions_count_as_missing(action: PreopAction) -> None:
    plan = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                    "Resume Eliquis 48 hours post-op", PLAN_A_TEXT.splitlines()[1], preop_action=action)
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), plan])
    (resolved,) = facts.anticoagulation_plans
    assert resolved.preop_instruction is None
    assert ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE in plan_gaps(resolved)
    assert evaluate(facts).decision is Decision.NEEDS_FOLLOW_UP


@pytest.mark.parametrize("action", sorted(PREOP_INSTRUCTIONS))
def test_instruction_preop_actions_still_require_timing(action: PreopAction) -> None:
    def plan_codes(timing: str | None) -> set[ReasonCode]:
        plan = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                        "Resume Eliquis 48 hours post-op", PLAN_A_TEXT.splitlines()[1],
                        preop_action=action, timing=timing)
        structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
        return codes(structured, [*base_extractions(), plan])

    assert plan_codes("2 days before") == set()
    assert ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE in plan_codes(None)


def test_unverified_action_quote_downgrades_resume_to_missing() -> None:
    assert_postop_missing(warfarin_plan_facts(PostopAction.RESUME, "Resume warfarin on postoperative day 1."))


def test_postop_timing_never_gates_the_decision() -> None:
    no_timing = plan_doc("a", PLAN_A_TEXT, "Eliquis", "Hold Eliquis for the 2 days before surgery",
                         "Resume Eliquis 48 hours post-op", PLAN_A_TEXT.splitlines()[1], postop_timing=None)
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
    assert codes(structured, [*base_extractions(), no_timing]) == set()
    assert_postop_missing(warfarin_plan_facts(PostopAction.DEFERRED, postop_timing="after surgery"))


def test_resume_label_on_a_deferral_quote_is_downgraded_by_the_quote_check() -> None:
    # The pre-enum extraction's mistake carried into the new schema: a verified
    # deferral sentence labeled as a firm RESUME.
    facts = warfarin_plan_facts(PostopAction.RESUME)
    assert_postop_missing(facts)
    (plan,) = facts.anticoagulation_plans
    assert plan.evidence[0].detail == (
        "plan for warfarin: pre-op HOLD (5 days before surgery); post-op UNCLEAR [extracted as RESUME]"
    )


def test_resume_label_on_a_bare_hand_off_becomes_deferred() -> None:
    text = "CARDIOLOGY NOTE\n1. Hold apixaban for the 2 days before surgery. Resume apixaban per cardiology."
    plan = plan_doc("a", text, "apixaban", "Hold apixaban for the 2 days before surgery",
                    "Resume apixaban per cardiology", text.splitlines()[1])
    structured = submission(documents=[*BASE_DOCS, ("a", "2026-03-03")], medications=APIXABAN_ACTIVE)
    facts = reconcile_patient_facts(structured, [*base_extractions(), plan])
    (resolved,) = facts.anticoagulation_plans
    assert resolved.postop_instruction is None and resolved.hedged
    assert evaluate(facts).decision is Decision.NEEDS_FOLLOW_UP


# =============================================================================
# Failed and missing extractions
# =============================================================================


def test_failed_or_missing_extraction_fails_closed() -> None:
    structured = submission(documents=[*BASE_DOCS, ("broken", "2026-03-05"), ("never-extracted", "2026-03-06")])
    facts = reconcile_patient_facts(structured, [*base_extractions(), ExtractedDocument.failed("broken", "timeout")])
    details = {ev.locator: ev.detail for ev in facts.unreadable_sources}
    assert details["documents[broken]"] == "document extraction failed: timeout"
    assert details["documents[never-extracted]"] == "document was not extracted"
    result = evaluate(facts)
    assert result.decision is Decision.NEEDS_FOLLOW_UP
    assert {i.reason_code for i in result.issues} == {ReasonCode.EXTRACTION_FAILED}


# =============================================================================
# Raw date / time parsing
# =============================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("03/15/2026", date(2026, 3, 15)),
        ("3/5/26 14:00", date(2026, 3, 5)),
        ("2026-03-15T08:00", date(2026, 3, 15)),
        ("Mar 15, 2026", date(2026, 3, 15)),
        ("03/15", date(2026, 3, 15)),
        ("08:06", None),
        ("02/30/2026", None),
    ],
)
def test_parse_raw_date(text: str, expected: date | None) -> None:
    assert parse_raw_date(text, reference=date(2026, 1, 1)) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("08:06", time(8, 6)),
        ("2:15 PM", time(14, 15)),
        ("12:05 am", time(0, 5)),
        ("2026-03-15T08:00", time(8, 0)),
        ("25:00", None),
        ("03/15/2026", None),
    ],
)
def test_parse_raw_clock(text: str, expected: time | None) -> None:
    assert parse_raw_clock(text) == expected


# =============================================================================
# Real sample data
# =============================================================================


def test_case_16_nursing_recheck_supersedes_high_bp() -> None:
    """Case 16: 08:06 BP 189/112, re-measured 137/78 at 08:30 (cached LLM response)."""

    row = next(json.loads(line) for line in SAMPLE.read_text().splitlines() if '"case_00016"' in line)
    structured = preprocess_submission(row["submission"])
    structured = structured.model_copy(update={"documents": tuple(d for d in structured.documents if d.doc_id == NURSING_DOC_ID)})
    text = next(d["text"] for d in row["submission"]["documents"] if d["doc_id"] == NURSING_DOC_ID)
    doc = verify_extracted_document(
        ExtractedDocument.model_validate_json(raw_response(NURSING_DOC_ID)).model_copy(update={"doc_id": NURSING_DOC_ID}), text
    )
    facts = reconcile_patient_facts(structured, [doc])

    newest_bp, newest_temp = facts.blood_pressures[0], facts.temperatures[0]
    assert (newest_bp.systolic, newest_bp.diastolic) == (Decimal(137), Decimal(78))
    assert newest_bp.observed_at == datetime(2026, 3, 15, 8, 30, tzinfo=UTC)
    assert (newest_temp.value, newest_temp.unit.value) == (Decimal("37.1"), "C")
    assert [m.generic_name for m in facts.medications] == ["aspirin"]
    assert facts.unreadable_sources == ()
    issues = {i.reason_code for i in evaluate(facts).issues}
    assert not issues & {ReasonCode.SYSTOLIC_BP_HIGH, ReasonCode.DIASTOLIC_BP_HIGH, ReasonCode.TEMPERATURE_HIGH}
