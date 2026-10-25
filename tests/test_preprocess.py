"""Tests for Stage 2 structured-data preprocessing."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from engine import check_acute_safety
from fact_builders import ready_facts
from models import DrugClass, LabTest, ReasonCode, RiskLevel, TimePrecision, celsius_to_fahrenheit
from preprocess import (
    ANTICOAGULANT_LOOKUP,
    ANTIPLATELET_LOOKUP,
    ExclusionReason,
    classify_drug,
    extract_patient_demographics,
    extract_structured_labs,
    extract_structured_medications,
    extract_structured_vitals,
    normalize_lab,
    parse_timestamp,
    preprocess_submission,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "patients_sample_50.jsonl"


def lab_record(code: str | None, status: str = "final", display: str | None = None, at: str = "2026-02-20T08:00:00Z") -> dict[str, Any]:
    return {"id": "x", "code": code, "display": display, "effective_at": at, "status": status, "source": "Lab"}


def temp_record(**fields: Any) -> dict[str, Any]:
    return {"type": "temperature", "date": "2026-02-28T09:00:00Z", "source": "Clinic", **fields}


# =============================================================================
# Labs
# =============================================================================


@pytest.mark.parametrize("status", ["entered-in-error", "cancelled", "preliminary", "amended", None, ""])
def test_non_final_labs_are_dropped(status: str | None) -> None:
    result = extract_structured_labs({"labs": [lab_record("CBC", status=status)]})  # type: ignore[arg-type]
    assert result.labs == ()
    assert [e.reason for e in result.excluded] == [ExclusionReason.LAB_STATUS_NOT_FINAL]


def test_status_match_is_case_insensitive() -> None:
    assert len(extract_structured_labs({"labs": [lab_record("CBC", status=" Final ")]}).labs) == 1


def test_entered_in_error_is_dropped_but_prior_final_is_kept() -> None:
    labs = [lab_record("CBC", "entered-in-error", at="2026-02-25T08:00:00Z"), lab_record("CBC", at="2026-02-10T08:00:00Z")]
    result = extract_structured_labs({"labs": labs})
    assert [lab.collected_at for lab in result.labs] == [datetime(2026, 2, 10, 8, tzinfo=UTC)]


@pytest.mark.parametrize(
    ("code", "display", "expected"),
    [
        ("CBC", "Hemogram/CBC", LabTest.CBC),
        ("LAB-CBC", "Complete Blood Count w/ Differential", LabTest.CBC),
        ("lab-cbc", None, LabTest.CBC),
        ("CMP", None, LabTest.CMP),
        ("LAB-CMP", "Comprehensive Metabolic Panel w/ Magnesium", LabTest.CMP),
        ("BMP", "Basic Metabolic Panel", LabTest.IGNORE),
        ("LAB-BMP", "Basic Metabolic Panel w/ eGFR", LabTest.IGNORE),
        ("HBA1C", "Hemoglobin A1c", LabTest.IGNORE),
        (None, "Complete Blood Count", LabTest.CBC),  # display fallback
        ("XYZ-999", "Comprehensive Metabolic Panel", LabTest.CMP),  # unknown code, known display
        (None, "Basic Metabolic Panel", LabTest.IGNORE),  # never substring-matched to CMP
        ("UNKNOWN", "Something else", LabTest.IGNORE),
    ],
)
def test_lab_normalization(code: str | None, display: str | None, expected: LabTest) -> None:
    assert normalize_lab(code, display) is expected


def test_lab_timestamps_are_aware_utc() -> None:
    (lab,) = extract_structured_labs({"labs": [lab_record("CBC", at="2026-02-20T20:00:00-05:00")]}).labs
    assert lab.collected_at == datetime(2026, 2, 21, 1, tzinfo=UTC)
    assert lab.collected_at.utcoffset() is not None and lab.collected_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


def test_final_lab_without_timestamp_is_excluded_as_malformed() -> None:
    result = extract_structured_labs({"labs": [lab_record("CBC", at="")]})
    assert result.labs == ()
    assert [e.reason for e in result.excluded] == [ExclusionReason.MALFORMED]


# =============================================================================
# Vitals
# =============================================================================


@pytest.mark.parametrize(
    ("fields", "expected_f"),
    [
        ({"value_c": 38.0}, Decimal("100.4")),
        ({"value_c": 37.9}, Decimal("100.22")),  # Decimal(37.9) would be 37.8999...
        ({"value_c": 38.1}, Decimal("100.58")),
        ({"value": "36.6", "unit": "°C"}, Decimal("97.88")),
        ({"value": 99.5, "unit": "F"}, Decimal("99.5")),
        ({"value_f": 100.4}, Decimal("100.4")),
    ],
)
def test_temperature_conversion_is_exact(fields: dict[str, Any], expected_f: Decimal) -> None:
    (reading,) = extract_structured_vitals({"vitals": [temp_record(**fields)]}).temperatures
    assert reading.value_f == expected_f


def test_parsing_via_str_avoids_binary_float_error() -> None:
    # Decimal(float) keeps the binary error; the parser goes through str() instead.
    assert celsius_to_fahrenheit(Decimal(37.9)) != Decimal("100.22")
    (reading,) = extract_structured_vitals({"vitals": [temp_record(value_c=37.9)]}).temperatures
    assert reading.value_f == Decimal("100.22")


@pytest.mark.parametrize(("celsius", "flagged"), [(38.0, False), (38.1, True)])
def test_celsius_reading_against_the_rule(celsius: float, flagged: bool) -> None:
    temps = extract_structured_vitals({"vitals": [temp_record(value_c=celsius)]}).temperatures
    issues = check_acute_safety(ready_facts(temperatures=temps))
    assert [i.reason_code for i in issues] == ([ReasonCode.TEMPERATURE_HIGH] if flagged else [])


def test_conversion_is_recorded_in_evidence() -> None:
    (reading,) = extract_structured_vitals({"vitals": [temp_record(value_c=38.6)]}).temperatures
    assert reading.evidence[0].detail == "recorded as 38.6 C"
    assert reading.evidence[0].locator == "vitals[0]"


def test_conflicting_temperature_representations_use_worst_case() -> None:
    (reading,) = extract_structured_vitals({"vitals": [temp_record(value_f=98.6, value_c=38.6)]}).temperatures
    assert reading.value_f == Decimal("101.48")


def test_blood_pressure_parsing() -> None:
    record = {"type": "blood_pressure", "systolic": 148, "diastolic": 110, "date": "2026-03-24T10:12:00Z", "source": "Pre-op clinic"}
    (reading,) = extract_structured_vitals({"vitals": [record]}).blood_pressures
    assert (reading.systolic, reading.diastolic) == (Decimal(148), Decimal(110))
    assert reading.observed_at == datetime(2026, 3, 24, 10, 12, tzinfo=UTC)
    assert reading.evidence[0].label == "Pre-op clinic"


@pytest.mark.parametrize(
    ("stamp", "expected", "precision"),
    [
        ("2026-02-28T09:00:00Z", datetime(2026, 2, 28, 9, tzinfo=UTC), TimePrecision.MINUTE),
        ("2026-02-28T09:00:00", datetime(2026, 2, 28, 9, tzinfo=UTC), TimePrecision.MINUTE),  # naive -> UTC
        ("2026-02-28T04:00:00-05:00", datetime(2026, 2, 28, 9, tzinfo=UTC), TimePrecision.MINUTE),
        ("2026-02-28", datetime(2026, 2, 28, tzinfo=UTC), TimePrecision.DAY),
    ],
)
def test_timestamps_become_aware_utc(stamp: str, expected: datetime, precision: TimePrecision) -> None:
    parsed, parsed_precision = parse_timestamp(stamp)
    assert parsed == expected and parsed.tzinfo is UTC and parsed_precision is precision


@pytest.mark.parametrize(
    "record",
    [
        {"type": "blood_pressure", "systolic": 120, "date": "2026-02-28T09:00:00Z"},  # no diastolic
        {"type": "blood_pressure", "systolic": True, "diastolic": 80, "date": "2026-02-28T09:00:00Z"},
        {"type": "blood_pressure", "systolic": 120, "diastolic": 80, "date": "yesterday"},
        temp_record(value=38.2),  # no unit
        temp_record(),  # no value
    ],
)
def test_malformed_vitals_are_excluded_not_silently_dropped(record: dict[str, Any]) -> None:
    result = extract_structured_vitals({"vitals": [record]})
    assert result.blood_pressures == () and result.temperatures == ()
    assert [e.reason for e in result.excluded] == [ExclusionReason.MALFORMED]


def test_unused_vital_types_are_recorded_as_not_applicable() -> None:
    result = extract_structured_vitals({"vitals": [{"type": "heart_rate", "value": 70, "date": "2026-02-28T09:00:00Z"}]})
    assert [e.reason for e in result.excluded] == [ExclusionReason.NOT_APPLICABLE]


# =============================================================================
# Medications
# =============================================================================


@pytest.mark.parametrize(
    ("name", "drug_class", "generic"),
    [
        ("warfarin (Coumadin) 5 mg PO daily", DrugClass.ANTICOAGULANT, "warfarin"),
        ("Coumadin 5 mg PO daily (INR goal 2-3)", DrugClass.ANTICOAGULANT, "warfarin"),
        ("Xarelto 20 mg PO daily with evening meal", DrugClass.ANTICOAGULANT, "rivaroxaban"),
        ("Eliquis 5 mg PO BID", DrugClass.ANTICOAGULANT, "apixaban"),
        ("dabigatran (Pradaxa) 150 mg PO BID", DrugClass.ANTICOAGULANT, "dabigatran"),
        ("edoxaban (Savaysa) 60 mg PO daily", DrugClass.ANTICOAGULANT, "edoxaban"),
        ("Lovenox 40 mg SC daily", DrugClass.ANTICOAGULANT, "enoxaparin"),
        ("Arixtra 2.5 mg SC daily", DrugClass.ANTICOAGULANT, "fondaparinux"),
        ("Fragmin 5000 units SC daily", DrugClass.ANTICOAGULANT, "dalteparin"),
        ("ASA 81 mg PO daily", DrugClass.ANTIPLATELET, "aspirin"),
        ("Aspirin EC 81 mg PO daily", DrugClass.ANTIPLATELET, "aspirin"),
        ("clopidogrel 75 mg PO daily", DrugClass.ANTIPLATELET, "clopidogrel"),
        ("Plavix 75 mg PO daily", DrugClass.ANTIPLATELET, "clopidogrel"),
        ("Brilinta 90 mg PO BID", DrugClass.ANTIPLATELET, "ticagrelor"),
        ("atorvastatin 40 mg PO nightly", DrugClass.OTHER, None),
    ],
)
def test_drug_classification(name: str, drug_class: DrugClass, generic: str | None) -> None:
    assert classify_drug(name) == (drug_class, generic)


def test_antiplatelets_are_never_anticoagulants() -> None:
    for name in ANTIPLATELET_LOOKUP:
        assert classify_drug(f"{name} 81 mg PO daily")[0] is DrugClass.ANTIPLATELET


def test_lookups_are_disjoint_and_generics_map_to_themselves() -> None:
    assert not set(ANTICOAGULANT_LOOKUP) & set(ANTIPLATELET_LOOKUP)
    for generic in set(ANTICOAGULANT_LOOKUP.values()):
        assert ANTICOAGULANT_LOOKUP[generic] == generic


def test_whole_word_matching_only() -> None:
    # "heparin" must not match inside "enoxaparin"; "asa" must not match inside other words.
    assert classify_drug("enoxaparin 40 mg") == (DrugClass.ANTICOAGULANT, "enoxaparin")
    assert classify_drug("Asacol 800 mg PO TID") == (DrugClass.OTHER, None)


def test_name_matching_both_lists_is_treated_as_anticoagulant() -> None:
    assert classify_drug("aspirin + warfarin") == (DrugClass.ANTICOAGULANT, "warfarin")


@pytest.mark.parametrize(("raw", "expected"), [(True, True), (False, False), (None, None), ("yes", None)])
def test_active_flag_is_retained(raw: object, expected: bool | None) -> None:
    (med,) = extract_structured_medications({"medications": [{"name": "Coumadin 5 mg PO daily", "active": raw}]}).medications
    assert med.active is expected
    assert med.generic_name == "warfarin"


def test_missing_active_key_is_null() -> None:
    (med,) = extract_structured_medications({"medications": [{"name": "apixaban 5 mg"}]}).medications
    assert med.active is None


def test_medication_without_name_is_excluded() -> None:
    result = extract_structured_medications({"medications": [{"name": " ", "active": True}]})
    assert result.medications == () and [e.reason for e in result.excluded] == [ExclusionReason.MALFORMED]


# =============================================================================
# Demographics and procedure
# =============================================================================


def test_demographics_and_procedure() -> None:
    submission = {
        "patient": {"id": "p1", "mrn": "MRN-1", "name": {"given": "Grace", "family": "Lopez"}, "dob": "1941-01-01", "sex": "F"},
        "procedure": {"case_id": "case_1", "procedure_type": "Elective X", "procedure_risk": "LOW", "procedure_date": "2026-03-01"},
    }
    demo = extract_patient_demographics(submission).demographics
    assert (demo.mrn, demo.given_name, demo.family_name, demo.dob) == ("MRN-1", "Grace", "Lopez", date(1941, 1, 1))
    assert demo.procedure.case_id == "case_1"
    assert demo.procedure.procedure_date == date(2026, 3, 1)
    assert demo.procedure.risk_level is RiskLevel.LOW


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("LOW", RiskLevel.LOW), (" moderate ", RiskLevel.MODERATE), ("High", RiskLevel.HIGH), ("SEVERE", None), (None, None), (3, None)],
)
def test_risk_normalization_keeps_raw_value(raw: object, expected: RiskLevel | None) -> None:
    procedure = extract_patient_demographics({"procedure": {"procedure_risk": raw}}).demographics.procedure
    assert procedure.risk_level is expected
    assert procedure.risk_level_raw == (None if raw is None else str(raw))


def test_null_and_unparseable_procedure_dates() -> None:
    assert extract_patient_demographics({"procedure": {"procedure_date": None}}).demographics.procedure.procedure_date is None
    bad = extract_patient_demographics({"procedure": {"procedure_date": "TBD"}})
    assert bad.demographics.procedure.procedure_date is None
    assert [e.locator for e in bad.excluded] == ["procedure.procedure_date"]


def test_empty_submission_does_not_crash() -> None:
    result = preprocess_submission({})
    assert result.labs == () and result.medications == () and result.excluded == ()
    assert result.demographics.procedure.procedure_date is None


def test_non_list_sections_are_reported() -> None:
    result = preprocess_submission({"labs": "oops", "vitals": [42]})
    assert {(e.locator, e.reason) for e in result.excluded} == {
        ("labs", ExclusionReason.MALFORMED),
        ("vitals[0]", ExclusionReason.MALFORMED),
    }


# =============================================================================
# Sample dataset regression
# =============================================================================


@pytest.fixture(scope="module")
def sample() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


def test_sample_preprocesses_with_only_entered_in_error_labs_excluded(sample: list[dict[str, Any]]) -> None:
    excluded = [(row["case_id"], e.reason) for row in sample for e in preprocess_submission(row["submission"]).excluded]
    assert excluded == [
        ("case_00004", ExclusionReason.LAB_STATUS_NOT_FINAL),
        ("case_00006", ExclusionReason.LAB_STATUS_NOT_FINAL),
        ("case_00041", ExclusionReason.LAB_STATUS_NOT_FINAL),
    ]


def test_sample_drug_classes(sample: list[dict[str, Any]]) -> None:
    meds = [m for row in sample for m in preprocess_submission(row["submission"]).medications]
    anticoagulants = {m.generic_name for m in meds if m.drug_class is DrugClass.ANTICOAGULANT}
    antiplatelets = {m.generic_name for m in meds if m.drug_class is DrugClass.ANTIPLATELET}
    assert anticoagulants == {"warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban"}
    assert antiplatelets == {"aspirin", "clopidogrel"}


def test_sample_null_date_and_risk_cases(sample: list[dict[str, Any]]) -> None:
    by_id = {row["case_id"]: preprocess_submission(row["submission"]).demographics.procedure for row in sample}
    assert {cid for cid, p in by_id.items() if p.procedure_date is None} == {"case_00011", "case_00044"}
    assert {cid for cid, p in by_id.items() if p.risk_level is None} == {"case_00029", "case_00039"}
