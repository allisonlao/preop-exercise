"""Stage 2: deterministic preprocessing of the structured parts of a submission.

Pure functions that turn the raw submission JSON into typed objects. Nothing
here reads document text, calls an LLM, or merges sources; that happens in
later stages. Where a structured record maps one-to-one onto an engine fact
(labs, vitals, procedure), the fact type from ``models.py`` is emitted
directly. Medications get an intermediate type, because their final status
depends on merging with document text.

Every record that is read but not emitted goes into an ``excluded`` list with a
reason, so nothing disappears silently. That matters most for vitals: dropping
an unreadable newer reading would let an older normal one become "most recent".

Numbers are read with ``Decimal(str(value))``. ``Decimal(37.9)`` would capture
the binary float error (37.899999...); ``str`` gives the shortest repr, "37.9".
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel, ConfigDict

from models import (
    BloodPressureReading,
    DrugClass,
    Evidence,
    EvidenceSource,
    LabResult,
    LabStatus,
    LabTest,
    ProcedureFacts,
    RiskLevel,
    TemperatureReading,
    TemperatureUnit,
    TimePrecision,
    celsius_to_fahrenheit,
)

# =============================================================================
# Reference data
# =============================================================================
#
# Lookups are exact (after case/whitespace normalization), never substring
# matches: "Basic Metabolic Panel" must not match "Metabolic Panel" and become a
# CMP. A lab not found by code or display is IGNORE, which can only make a
# required test look missing (NEEDS_FOLLOW_UP), never satisfy it.

LAB_CODE_LOOKUP: Final[Mapping[str, LabTest]] = {
    "CBC": LabTest.CBC,
    "LAB-CBC": LabTest.CBC,
    "CBC W/ DIFF": LabTest.CBC,
    "CBC WITH DIFF": LabTest.CBC,
    "CMP": LabTest.CMP,
    "LAB-CMP": LabTest.CMP,
    "BMP": LabTest.IGNORE,
    "LAB-BMP": LabTest.IGNORE,
    "HBA1C": LabTest.IGNORE,
    "PT-INR": LabTest.IGNORE,
    "INR": LabTest.IGNORE,
    "PTT": LabTest.IGNORE,
    "TSH": LabTest.IGNORE,
    "UA": LabTest.IGNORE,
    "T&S": LabTest.IGNORE,
}

# Fallback when the code is missing or unknown.
LAB_DISPLAY_LOOKUP: Final[Mapping[str, LabTest]] = {
    "COMPLETE BLOOD COUNT": LabTest.CBC,
    "COMPLETE BLOOD COUNT W/ DIFFERENTIAL": LabTest.CBC,
    "COMPLETE BLOOD COUNT WITH DIFFERENTIAL": LabTest.CBC,
    "CBC (COMPLETE BLOOD COUNT)": LabTest.CBC,
    "HEMOGRAM/CBC": LabTest.CBC,
    "COMPREHENSIVE METABOLIC PANEL": LabTest.CMP,
    "COMPREHENSIVE METABOLIC PANEL W/ MAGNESIUM": LabTest.CMP,
    "CMP (COMPREHENSIVE METABOLIC PANEL)": LabTest.CMP,
    "BASIC METABOLIC PANEL": LabTest.IGNORE,
    "BASIC METABOLIC PANEL W/ EGFR": LabTest.IGNORE,
}

# Statuses that count as a completed result (FHIR semantics: amended and corrected
# are revised final results). Entered-in-error, preliminary, cancelled, and
# anything else are excluded.
ACCEPTED_LAB_STATUSES: Final[Mapping[str, LabStatus]] = {
    "final": LabStatus.FINAL,
    "amended": LabStatus.AMENDED,
    "corrected": LabStatus.CORRECTED,
}

# Brand and generic names (lowercase) -> base generic.
ANTICOAGULANT_LOOKUP: Final[Mapping[str, str]] = {
    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "jantoven": "warfarin",
    "apixaban": "apixaban",
    "eliquis": "apixaban",
    "rivaroxaban": "rivaroxaban",
    "xarelto": "rivaroxaban",
    "dabigatran": "dabigatran",
    "pradaxa": "dabigatran",
    "edoxaban": "edoxaban",
    "savaysa": "edoxaban",
    "lixiana": "edoxaban",
    "betrixaban": "betrixaban",
    "bevyxxa": "betrixaban",
    "enoxaparin": "enoxaparin",
    "lovenox": "enoxaparin",
    "dalteparin": "dalteparin",
    "fragmin": "dalteparin",
    "tinzaparin": "tinzaparin",
    "innohep": "tinzaparin",
    "fondaparinux": "fondaparinux",
    "arixtra": "fondaparinux",
    "heparin": "heparin",
    "argatroban": "argatroban",
    "bivalirudin": "bivalirudin",
    "angiomax": "bivalirudin",
}

# Antiplatelets are NOT anticoagulants; Rule 3 ignores them.
ANTIPLATELET_LOOKUP: Final[Mapping[str, str]] = {
    "aspirin": "aspirin",
    "asa": "aspirin",
    "ecotrin": "aspirin",
    "clopidogrel": "clopidogrel",
    "plavix": "clopidogrel",
    "prasugrel": "prasugrel",
    "effient": "prasugrel",
    "ticagrelor": "ticagrelor",
    "brilinta": "ticagrelor",
    "ticlopidine": "ticlopidine",
    "dipyridamole": "dipyridamole",
    "persantine": "dipyridamole",
    "aggrenox": "aspirin/dipyridamole",
    "cilostazol": "cilostazol",
    "pletal": "cilostazol",
    "vorapaxar": "vorapaxar",
    "zontivity": "vorapaxar",
}

assert not set(ANTICOAGULANT_LOOKUP) & set(ANTIPLATELET_LOOKUP), "drug lookups must be disjoint"

_RISK_ALIASES: Final[Mapping[str, RiskLevel]] = {level.value: level for level in RiskLevel}

_BP_TYPES: Final = frozenset({"blood_pressure", "bp"})
_TEMPERATURE_TYPES: Final = frozenset({"temperature", "temp", "body_temperature"})
_FAHRENHEIT_UNITS: Final = frozenset({"f", "°f", "degf", "fahrenheit"})
_CELSIUS_UNITS: Final = frozenset({"c", "°c", "degc", "cel", "celsius"})

# =============================================================================
# Output types
# =============================================================================


class ExclusionReason(StrEnum):
    LAB_STATUS_EXCLUDED = "LAB_STATUS_EXCLUDED"
    MALFORMED = "MALFORMED"  # a field the policy needs is missing or unparseable
    NOT_APPLICABLE = "NOT_APPLICABLE"  # well-formed, but not used by any rule (e.g. heart rate)


class ExcludedRecord(BaseModel):
    """A structured record that was read but deliberately not emitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locator: str
    reason: ExclusionReason
    detail: str


class StructuredMedication(BaseModel):
    """One JSON medication entry, classified; status is not merged with notes yet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name_raw: str
    # Base generic when the drug is a known anticoagulant or antiplatelet, else None.
    generic_name: str | None
    drug_class: DrugClass
    # Exactly as submitted: True, False, or None when the field is null or absent.
    active: bool | None
    evidence: Evidence


class PatientDemographics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    patient_id: str | None
    mrn: str | None
    given_name: str | None
    family_name: str | None
    dob: date | None
    sex: str | None
    procedure: ProcedureFacts


class LabExtraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    labs: tuple[LabResult, ...]
    excluded: tuple[ExcludedRecord, ...]


class VitalsExtraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    blood_pressures: tuple[BloodPressureReading, ...]
    temperatures: tuple[TemperatureReading, ...]
    excluded: tuple[ExcludedRecord, ...]


class MedicationExtraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    medications: tuple[StructuredMedication, ...]
    excluded: tuple[ExcludedRecord, ...]


class DemographicsExtraction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    demographics: PatientDemographics
    excluded: tuple[ExcludedRecord, ...]


class StructuredSubmission(BaseModel):
    """Everything Stage 2 extracts from one submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    demographics: PatientDemographics
    labs: tuple[LabResult, ...]
    blood_pressures: tuple[BloodPressureReading, ...]
    temperatures: tuple[TemperatureReading, ...]
    medications: tuple[StructuredMedication, ...]
    excluded: tuple[ExcludedRecord, ...]


class _Malformed(ValueError):
    """Raised inside per-record parsers; converted into an ExcludedRecord."""


# =============================================================================
# Primitive parsers
# =============================================================================


def _records(submission: Mapping[str, Any], key: str) -> list[tuple[str, Mapping[str, Any]]]:
    """``(locator, record)`` pairs for the object entries of a top-level array.

    Non-object entries are skipped here and reported by ``_non_objects``.
    """

    value = submission.get(key)
    if not isinstance(value, list):
        return []
    return [(f"{key}[{index}]", item) for index, item in enumerate(value) if isinstance(item, Mapping)]


def _non_objects(submission: Mapping[str, Any], key: str) -> list[ExcludedRecord]:
    value = submission.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        return [ExcludedRecord(locator=key, reason=ExclusionReason.MALFORMED, detail="expected a list")]
    return [
        ExcludedRecord(locator=f"{key}[{index}]", reason=ExclusionReason.MALFORMED, detail="expected an object")
        for index, item in enumerate(value)
        if not isinstance(item, Mapping)
    ]


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _decimal(value: object, field: str) -> Decimal:
    """Parse a JSON number (or numeric string) exactly. Booleans are rejected."""

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _Malformed(f"{field} is not a number: {value!r}")
    try:
        parsed = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise _Malformed(f"{field} is not a number: {value!r}") from exc
    if not parsed.is_finite():
        raise _Malformed(f"{field} is not finite: {value!r}")
    return parsed


def parse_timestamp(value: object) -> tuple[datetime, TimePrecision]:
    """Parse an ISO timestamp as timezone-aware UTC.

    A naive timestamp is assumed to be UTC (the intake API emits ``Z``). A bare
    date becomes midnight UTC with ``DAY`` precision.
    """

    text = _optional_str(value)
    if text is None:
        raise _Malformed(f"timestamp missing or not a string: {value!r}")
    try:
        if len(text) == 10:
            day = date.fromisoformat(text)
            return datetime(day.year, day.month, day.day, tzinfo=UTC), TimePrecision.DAY
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise _Malformed(f"unparseable timestamp: {text!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC), TimePrecision.MINUTE


def parse_calendar_date(value: object) -> date | None:
    """Parse a calendar date ('2026-03-01'); a datetime string keeps its written date."""

    text = _optional_str(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _structured_evidence(locator: str, record: Mapping[str, Any], label: str | None, on: date | None) -> Evidence:
    return Evidence(
        source=EvidenceSource.STRUCTURED,
        locator=locator,
        label=label,
        recorded_on=on,
        quote=json.dumps(record, sort_keys=True, default=str),
    )


# =============================================================================
# Labs
# =============================================================================


def normalize_lab(code: object, display: object) -> LabTest:
    """Map a lab to CBC, CMP, or IGNORE by exact code, then exact display."""

    code_key = _optional_str(code)
    if code_key is not None and code_key.upper() in LAB_CODE_LOOKUP:
        return LAB_CODE_LOOKUP[code_key.upper()]
    display_key = _optional_str(display)
    if display_key is not None and display_key.upper() in LAB_DISPLAY_LOOKUP:
        return LAB_DISPLAY_LOOKUP[display_key.upper()]
    return LabTest.IGNORE


def _parse_lab(locator: str, record: Mapping[str, Any], status: LabStatus) -> LabResult:
    collected_at, _ = parse_timestamp(record.get("effective_at"))
    display = _optional_str(record.get("display"))
    return LabResult(
        test=normalize_lab(record.get("code"), display),
        code_raw=_optional_str(record.get("code")),
        collected_at=collected_at,
        status=status,
        evidence=(_structured_evidence(locator, record, display, collected_at.date()),),
    )


def extract_structured_labs(submission: Mapping[str, Any]) -> LabExtraction:
    """Keep final / amended / corrected results, normalized to CBC / CMP / IGNORE."""

    labs: list[LabResult] = []
    excluded = _non_objects(submission, "labs")
    for locator, record in _records(submission, "labs"):
        raw_status = _optional_str(record.get("status"))
        status = ACCEPTED_LAB_STATUSES.get(raw_status.lower()) if raw_status else None
        if status is None:
            excluded.append(
                ExcludedRecord(
                    locator=locator,
                    reason=ExclusionReason.LAB_STATUS_EXCLUDED,
                    detail=f"status {raw_status!r} is not one of {sorted(ACCEPTED_LAB_STATUSES)}",
                )
            )
            continue
        try:
            labs.append(_parse_lab(locator, record, status))
        except _Malformed as exc:
            excluded.append(ExcludedRecord(locator=locator, reason=ExclusionReason.MALFORMED, detail=str(exc)))
    return LabExtraction(labs=tuple(labs), excluded=tuple(excluded))


# =============================================================================
# Vitals
# =============================================================================


def _temperature_fahrenheit(record: Mapping[str, Any]) -> tuple[Decimal, str]:
    """Return the temperature in F plus a note on how it was recorded.

    Accepts ``value_f``, ``value_c``, or ``value`` + ``unit``. If several
    representations are present, the highest Fahrenheit value is used (worst
    case), consistent with how the engine resolves ambiguous vitals.
    """

    candidates: list[tuple[Decimal, str]] = []
    if record.get("value_f") is not None:
        value = _decimal(record["value_f"], "value_f")
        candidates.append((value, f"recorded as {value} F"))
    if record.get("value_c") is not None:
        value = _decimal(record["value_c"], "value_c")
        candidates.append((celsius_to_fahrenheit(value), f"recorded as {value} C"))
    if record.get("value") is not None:
        value = _decimal(record["value"], "value")
        unit = _optional_str(record.get("unit"))
        unit_key = unit.lower() if unit else None
        if unit_key in _FAHRENHEIT_UNITS:
            candidates.append((value, f"recorded as {value} F"))
        elif unit_key in _CELSIUS_UNITS:
            candidates.append((celsius_to_fahrenheit(value), f"recorded as {value} C"))
        else:
            raise _Malformed(f"temperature unit missing or unknown: {unit!r}")
    if not candidates:
        raise _Malformed("temperature has no value_f, value_c, or value+unit")

    fahrenheit, note = max(candidates, key=lambda pair: pair[0])
    if len(candidates) > 1:
        note += f"; {len(candidates)} representations present, highest used"
    return fahrenheit, note


def _parse_blood_pressure(locator: str, record: Mapping[str, Any]) -> BloodPressureReading:
    observed_at, precision = parse_timestamp(record.get("date"))
    return BloodPressureReading(
        observed_at=observed_at,
        precision=precision,
        systolic=_decimal(record.get("systolic"), "systolic"),
        diastolic=_decimal(record.get("diastolic"), "diastolic"),
        evidence=(_structured_evidence(locator, record, _optional_str(record.get("source")), observed_at.date()),),
    )


def _parse_temperature(locator: str, record: Mapping[str, Any]) -> TemperatureReading:
    observed_at, precision = parse_timestamp(record.get("date"))
    fahrenheit, note = _temperature_fahrenheit(record)
    evidence = _structured_evidence(locator, record, _optional_str(record.get("source")), observed_at.date())
    return TemperatureReading(
        observed_at=observed_at,
        precision=precision,
        value=fahrenheit,
        unit=TemperatureUnit.FAHRENHEIT,
        evidence=(evidence.model_copy(update={"detail": note}),),
    )


def extract_structured_vitals(submission: Mapping[str, Any]) -> VitalsExtraction:
    """Parse BP and temperature readings; temperatures are emitted in Fahrenheit."""

    pressures: list[BloodPressureReading] = []
    temperatures: list[TemperatureReading] = []
    excluded = _non_objects(submission, "vitals")
    for locator, record in _records(submission, "vitals"):
        kind = (_optional_str(record.get("type")) or "").lower()
        try:
            if kind in _BP_TYPES:
                pressures.append(_parse_blood_pressure(locator, record))
            elif kind in _TEMPERATURE_TYPES:
                temperatures.append(_parse_temperature(locator, record))
            else:
                excluded.append(
                    ExcludedRecord(
                        locator=locator,
                        reason=ExclusionReason.NOT_APPLICABLE,
                        detail=f"vital type {kind or None!r} is not used by any rule",
                    )
                )
        except _Malformed as exc:
            excluded.append(ExcludedRecord(locator=locator, reason=ExclusionReason.MALFORMED, detail=str(exc)))
    return VitalsExtraction(blood_pressures=tuple(pressures), temperatures=tuple(temperatures), excluded=tuple(excluded))


# =============================================================================
# Medications
# =============================================================================

_TOKEN = re.compile(r"[a-z]+")


def classify_drug(name: str) -> tuple[DrugClass, str | None]:
    """Classify a free-text medication name by whole-word lookup.

    ``"warfarin (Coumadin) 5 mg PO daily"`` -> ``(ANTICOAGULANT, "warfarin")``.
    If a name matches both lists, the anticoagulant wins: treating it as an
    anticoagulant can only require a plan, never skip one.
    """

    tokens = _TOKEN.findall(name.lower())
    for token in tokens:
        if token in ANTICOAGULANT_LOOKUP:
            return DrugClass.ANTICOAGULANT, ANTICOAGULANT_LOOKUP[token]
    for token in tokens:
        if token in ANTIPLATELET_LOOKUP:
            return DrugClass.ANTIPLATELET, ANTIPLATELET_LOOKUP[token]
    return DrugClass.OTHER, None


def extract_structured_medications(submission: Mapping[str, Any]) -> MedicationExtraction:
    """Classify each JSON medication; the ``active`` flag is kept exactly as submitted."""

    medications: list[StructuredMedication] = []
    excluded = _non_objects(submission, "medications")
    for locator, record in _records(submission, "medications"):
        name = _optional_str(record.get("name"))
        if name is None:
            excluded.append(ExcludedRecord(locator=locator, reason=ExclusionReason.MALFORMED, detail="name is missing"))
            continue
        active_raw = record.get("active")
        active = active_raw if isinstance(active_raw, bool) else None
        drug_class, generic = classify_drug(name)
        evidence = _structured_evidence(locator, record, name, None)
        if active_raw is not None and not isinstance(active_raw, bool):
            evidence = evidence.model_copy(update={"detail": f"active value {active_raw!r} is not a boolean; treated as null"})
        medications.append(
            StructuredMedication(name_raw=name, generic_name=generic, drug_class=drug_class, active=active, evidence=evidence)
        )
    return MedicationExtraction(medications=tuple(medications), excluded=tuple(excluded))


# =============================================================================
# Demographics and procedure
# =============================================================================


def normalize_risk(value: object) -> RiskLevel | None:
    """'LOW' / 'Moderate' / ' high ' -> RiskLevel; anything else -> None."""

    text = _optional_str(value)
    return None if text is None else _RISK_ALIASES.get(text.upper())


def extract_patient_demographics(submission: Mapping[str, Any]) -> DemographicsExtraction:
    """MRN, name, DOB, and the procedure facts used by rule R0.

    Only the structured ``procedure_date`` is used; dates in note text never are.
    """

    patient = submission.get("patient")
    patient = patient if isinstance(patient, Mapping) else {}
    name = patient.get("name")
    name = name if isinstance(name, Mapping) else {}
    procedure = submission.get("procedure")
    procedure = procedure if isinstance(procedure, Mapping) else {}
    excluded: list[ExcludedRecord] = []

    dob = parse_calendar_date(patient.get("dob"))
    if patient.get("dob") is not None and dob is None:
        excluded.append(ExcludedRecord(locator="patient.dob", reason=ExclusionReason.MALFORMED, detail=f"unparseable date {patient.get('dob')!r}"))

    raw_date = procedure.get("procedure_date")
    procedure_date = parse_calendar_date(raw_date)
    if raw_date is not None and procedure_date is None:
        excluded.append(
            ExcludedRecord(locator="procedure.procedure_date", reason=ExclusionReason.MALFORMED, detail=f"unparseable date {raw_date!r}")
        )

    raw_risk = procedure.get("procedure_risk")
    risk_raw_text = None if raw_risk is None else str(raw_risk)

    facts = ProcedureFacts(
        case_id=_optional_str(procedure.get("case_id")),
        procedure_type=_optional_str(procedure.get("procedure_type")),
        procedure_date=procedure_date,
        risk_level=normalize_risk(raw_risk),
        risk_level_raw=risk_raw_text,
        date_evidence=Evidence(
            source=EvidenceSource.STRUCTURED,
            locator="procedure.procedure_date",
            quote=json.dumps(raw_date, default=str),
        ),
        risk_evidence=Evidence(
            source=EvidenceSource.STRUCTURED,
            locator="procedure.procedure_risk",
            quote=json.dumps(raw_risk, default=str),
        ),
    )
    demographics = PatientDemographics(
        patient_id=_optional_str(patient.get("id")),
        mrn=_optional_str(patient.get("mrn")),
        given_name=_optional_str(name.get("given")),
        family_name=_optional_str(name.get("family")),
        dob=dob,
        sex=_optional_str(patient.get("sex")),
        procedure=facts,
    )
    return DemographicsExtraction(demographics=demographics, excluded=tuple(excluded))


# =============================================================================
# Entry point
# =============================================================================


def preprocess_submission(submission: Mapping[str, Any]) -> StructuredSubmission:
    """Run every structured extractor and collect all exclusions."""

    demographics = extract_patient_demographics(submission)
    labs = extract_structured_labs(submission)
    vitals = extract_structured_vitals(submission)
    medications = extract_structured_medications(submission)
    excluded: Sequence[ExcludedRecord] = (
        *demographics.excluded,
        *labs.excluded,
        *vitals.excluded,
        *medications.excluded,
    )
    return StructuredSubmission(
        demographics=demographics.demographics,
        labs=labs.labs,
        blood_pressures=vitals.blood_pressures,
        temperatures=vitals.temperatures,
        medications=medications.medications,
        excluded=tuple(excluded),
    )
