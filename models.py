"""Typed contracts for the pre-op triage pipeline.

Two groups of models live here:

* Output models (``TriageResult`` and friends): the strict JSON contract returned
  to callers and reviewed by pre-op nurses.
* Fact models (``ReconciledPatientFacts`` and friends): the fully merged view of
  one patient after structured JSON and document text have been reconciled. The
  rule engine (``engine.py``) consumes only these, never raw submissions.

Fact models record *what the chart says*; they never encode policy outcomes.
Policy thresholds and windows live in ``engine.py`` so they can be audited in
one place.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Shared enums
# =============================================================================


class Decision(StrEnum):
    READY = "READY"
    NEEDS_FOLLOW_UP = "NEEDS_FOLLOW_UP"
    NOT_CLEARED = "NOT_CLEARED"


class IssueCategory(StrEnum):
    REQUIRED_DOCUMENTATION = "REQUIRED_DOCUMENTATION"
    REQUIRED_TESTING = "REQUIRED_TESTING"
    ANTICOAGULATION_MANAGEMENT = "ANTICOAGULATION_MANAGEMENT"
    ACUTE_SAFETY_EXCLUSION = "ACUTE_SAFETY_EXCLUSION"
    MISSING_REQUIRED_DATA = "MISSING_REQUIRED_DATA"


class RuleId(StrEnum):
    R0_PRECONDITIONS = "R0"
    R1_DOCUMENTATION = "R1"
    R2_TESTING = "R2"
    R3_ANTICOAGULATION = "R3"
    R4_ACUTE_SAFETY = "R4"


class ReasonCode(StrEnum):
    """Granular, machine-readable reason for an issue.

    Every code belongs to exactly one category (see ``REASON_CATEGORY``); the
    ``Issue`` validator rejects any other pairing.
    """

    # R0 - preconditions
    PROCEDURE_DATE_MISSING = "PROCEDURE_DATE_MISSING"
    RISK_LEVEL_UNKNOWN = "RISK_LEVEL_UNKNOWN"

    # R1 - documentation
    HP_MISSING = "HP_MISSING"
    HP_NOT_COMPLETED = "HP_NOT_COMPLETED"
    HP_DATE_UNKNOWN = "HP_DATE_UNKNOWN"
    HP_OUTSIDE_WINDOW = "HP_OUTSIDE_WINDOW"
    CONSENT_MISSING = "CONSENT_MISSING"
    CONSENT_UNSIGNED = "CONSENT_UNSIGNED"
    CONSENT_REVOKED = "CONSENT_REVOKED"
    CONSENT_PROCEDURE_MISMATCH = "CONSENT_PROCEDURE_MISMATCH"
    CONSENT_PROCEDURE_UNVERIFIED = "CONSENT_PROCEDURE_UNVERIFIED"

    # R2 - testing
    CBC_MISSING = "CBC_MISSING"
    CBC_OUTSIDE_WINDOW = "CBC_OUTSIDE_WINDOW"
    CMP_MISSING = "CMP_MISSING"
    CMP_OUTSIDE_WINDOW = "CMP_OUTSIDE_WINDOW"

    # R3 - anticoagulation
    ANTICOAG_PLAN_MISSING = "ANTICOAG_PLAN_MISSING"
    ANTICOAG_PLAN_DRUG_MISMATCH = "ANTICOAG_PLAN_DRUG_MISMATCH"
    ANTICOAG_PLAN_HEDGED = "ANTICOAG_PLAN_HEDGED"
    ANTICOAG_PLAN_PREOP_INCOMPLETE = "ANTICOAG_PLAN_PREOP_INCOMPLETE"
    ANTICOAG_PLAN_POSTOP_MISSING = "ANTICOAG_PLAN_POSTOP_MISSING"
    ANTICOAG_PLAN_CONFLICT = "ANTICOAG_PLAN_CONFLICT"
    ANTICOAG_STATUS_CONFLICT = "ANTICOAG_STATUS_CONFLICT"
    ANTICOAG_STATUS_UNKNOWN = "ANTICOAG_STATUS_UNKNOWN"

    # R4 - acute safety
    SYSTOLIC_BP_HIGH = "SYSTOLIC_BP_HIGH"
    DIASTOLIC_BP_HIGH = "DIASTOLIC_BP_HIGH"
    TEMPERATURE_HIGH = "TEMPERATURE_HIGH"
    BP_MISSING = "BP_MISSING"
    TEMPERATURE_MISSING = "TEMPERATURE_MISSING"

    # Pipeline failure (used by later stages when extraction fails; fails closed)
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


_DOC = IssueCategory.REQUIRED_DOCUMENTATION
_TEST = IssueCategory.REQUIRED_TESTING
_ANTICOAG = IssueCategory.ANTICOAGULATION_MANAGEMENT
_ACUTE = IssueCategory.ACUTE_SAFETY_EXCLUSION
_MISSING = IssueCategory.MISSING_REQUIRED_DATA

REASON_CATEGORY: dict[ReasonCode, IssueCategory] = {
    ReasonCode.PROCEDURE_DATE_MISSING: _MISSING,
    ReasonCode.RISK_LEVEL_UNKNOWN: _MISSING,
    ReasonCode.HP_MISSING: _DOC,
    ReasonCode.HP_NOT_COMPLETED: _DOC,
    ReasonCode.HP_DATE_UNKNOWN: _DOC,
    ReasonCode.HP_OUTSIDE_WINDOW: _DOC,
    ReasonCode.CONSENT_MISSING: _DOC,
    ReasonCode.CONSENT_UNSIGNED: _DOC,
    ReasonCode.CONSENT_REVOKED: _DOC,
    ReasonCode.CONSENT_PROCEDURE_MISMATCH: _DOC,
    ReasonCode.CONSENT_PROCEDURE_UNVERIFIED: _DOC,
    ReasonCode.CBC_MISSING: _TEST,
    ReasonCode.CBC_OUTSIDE_WINDOW: _TEST,
    ReasonCode.CMP_MISSING: _TEST,
    ReasonCode.CMP_OUTSIDE_WINDOW: _TEST,
    ReasonCode.ANTICOAG_PLAN_MISSING: _ANTICOAG,
    ReasonCode.ANTICOAG_PLAN_DRUG_MISMATCH: _ANTICOAG,
    ReasonCode.ANTICOAG_PLAN_HEDGED: _ANTICOAG,
    ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE: _ANTICOAG,
    ReasonCode.ANTICOAG_PLAN_POSTOP_MISSING: _ANTICOAG,
    ReasonCode.ANTICOAG_PLAN_CONFLICT: _ANTICOAG,
    ReasonCode.ANTICOAG_STATUS_CONFLICT: _MISSING,
    ReasonCode.ANTICOAG_STATUS_UNKNOWN: _MISSING,
    ReasonCode.SYSTOLIC_BP_HIGH: _ACUTE,
    ReasonCode.DIASTOLIC_BP_HIGH: _ACUTE,
    ReasonCode.TEMPERATURE_HIGH: _ACUTE,
    ReasonCode.BP_MISSING: _MISSING,
    ReasonCode.TEMPERATURE_MISSING: _MISSING,
    ReasonCode.EXTRACTION_FAILED: _MISSING,
}

# Fail loudly at import time if a reason code was added without a category.
assert set(REASON_CATEGORY) == set(ReasonCode), "every ReasonCode needs a category"


def decision_for(categories: set[IssueCategory]) -> Decision:
    """Policy precedence: NOT_CLEARED > NEEDS_FOLLOW_UP > READY."""

    if IssueCategory.ACUTE_SAFETY_EXCLUSION in categories:
        return Decision.NOT_CLEARED
    if categories:
        return Decision.NEEDS_FOLLOW_UP
    return Decision.READY


# =============================================================================
# Evidence (shared by facts and output)
# =============================================================================


class EvidenceSource(StrEnum):
    STRUCTURED = "STRUCTURED"  # a field in the submission JSON
    DOCUMENT = "DOCUMENT"  # free text in a clinical document
    ABSENCE = "ABSENCE"  # records a search that found nothing


class Evidence(BaseModel):
    """One verifiable pointer into the submission.

    ``locator`` is what a reviewer uses to find the source: a JSON path such as
    ``labs[2]`` / ``procedure.procedure_date``, or ``documents[<doc_id>]``.
    ``quote`` is verbatim source text when available; ``detail`` explains how
    the engine used it (e.g. "18 days before procedure; window is 30").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: EvidenceSource
    locator: str = Field(min_length=1)
    label: str | None = None
    recorded_on: date | None = None
    quote: str | None = None
    detail: str | None = None


# =============================================================================
# Output models
# =============================================================================


class Issue(BaseModel):
    """A single blocking finding, designed to be verified without re-reading the chart."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: IssueCategory
    reason_code: ReasonCode
    rule: RuleId
    description: str = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    resolution: str = Field(min_length=1)

    @model_validator(mode="after")
    def _reason_matches_category(self) -> Self:
        expected = REASON_CATEGORY[self.reason_code]
        if self.category is not expected:
            raise ValueError(
                f"reason_code {self.reason_code} belongs to {expected}, not {self.category}"
            )
        return self


class TriageResult(BaseModel):
    """The final JSON contract. ``decision`` must agree with the issue categories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: Decision
    issues: list[Issue]
    explanation: str

    @model_validator(mode="after")
    def _decision_matches_issues(self) -> Self:
        expected = decision_for({issue.category for issue in self.issues})
        if self.decision is not expected:
            raise ValueError(f"decision {self.decision} is inconsistent with issues ({expected})")
        return self


# =============================================================================
# Reconciled fact models (rule-engine input)
# =============================================================================


class _Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RiskLevel(StrEnum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class TimePrecision(StrEnum):
    """How precisely ``observed_at`` is known.

    ``DAY`` means only the calendar date is known (e.g. a note vital without a
    clock time); ``observed_at`` is then midnight UTC of that date.
    """

    MINUTE = "MINUTE"
    DAY = "DAY"


class ProcedureFacts(_Fact):
    case_id: str | None = None
    procedure_type: str | None = None
    # Only the structured JSON date; provisional/"target" dates from text are never used.
    procedure_date: date | None
    # None when the raw value is missing or not one of LOW/MODERATE/HIGH.
    risk_level: RiskLevel | None
    risk_level_raw: str | None = None
    date_evidence: Evidence
    risk_evidence: Evidence


class HistoryAndPhysical(_Fact):
    """Any document whose content is a History and Physical, regardless of its title."""

    document_id: str
    performed_on: date | None
    completed: bool
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class SurgicalConsent(_Fact):
    document_id: str
    documented_on: date | None = None
    signed: bool
    signed_at: AwareDatetime | None = None
    revoked: bool = False
    # None when the consent's procedure could not be compared to the scheduled one.
    procedure_matches: bool | None
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class LabTest(StrEnum):
    CBC = "CBC"
    CMP = "CMP"
    IGNORE = "IGNORE"  # anything the policy does not require (BMP, TSH, UA, ...)


class LabStatus(StrEnum):
    FINAL = "FINAL"
    AMENDED = "AMENDED"
    CORRECTED = "CORRECTED"
    PRELIMINARY = "PRELIMINARY"
    CANCELLED = "CANCELLED"
    ENTERED_IN_ERROR = "ENTERED_IN_ERROR"
    UNKNOWN = "UNKNOWN"


class LabResult(_Fact):
    test: LabTest
    code_raw: str | None = None
    collected_at: AwareDatetime
    status: LabStatus
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class DrugClass(StrEnum):
    ANTICOAGULANT = "ANTICOAGULANT"
    ANTIPLATELET = "ANTIPLATELET"
    OTHER = "OTHER"


class MedicationStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    # Sources disagree (e.g. JSON active:false but a nursing note reports current use).
    CONFLICT = "CONFLICT"
    # No source establishes whether the patient is currently taking it.
    UNKNOWN = "UNKNOWN"


class ResolvedMedication(_Fact):
    """One drug after merging the med list with notes; brand names resolved to generic."""

    generic_name: str = Field(min_length=1)
    display_name: str
    drug_class: DrugClass
    status: MedicationStatus
    evidence: tuple[Evidence, ...] = Field(min_length=1)

    @field_validator("generic_name")
    @classmethod
    def _normalize_generic(cls, value: str) -> str:
        return value.strip().lower()


class AnticoagulationPlan(_Fact):
    """Structured reading of one documented perioperative anticoagulation plan."""

    plan_id: str = Field(min_length=1)
    document_id: str
    written_on: date | None = None
    # Generic names of the drugs the plan explicitly manages.
    drugs_addressed: tuple[str, ...]
    preop_instruction: str | None = None
    # True when the pre-op instruction states timing (hold N days, last dose on X, ...).
    preop_timing_specified: bool = False
    postop_instruction: str | None = None
    # True when management is deferred or non-committal ("to be determined", "will follow up").
    hedged: bool = False
    # True when a later document explicitly replaces this plan.
    superseded: bool = False
    # plan_ids of other plans that give contradictory instructions for the same drug.
    conflicts_with: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = Field(min_length=1)

    @property
    def has_conflicting_plans(self) -> bool:
        return bool(self.conflicts_with)

    @field_validator("drugs_addressed")
    @classmethod
    def _normalize_drugs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(sorted({drug.strip().lower() for drug in value}))


class BloodPressureReading(_Fact):
    observed_at: AwareDatetime
    precision: TimePrecision = TimePrecision.MINUTE
    systolic: Decimal
    diastolic: Decimal
    evidence: tuple[Evidence, ...] = Field(min_length=1)


class TemperatureUnit(StrEnum):
    FAHRENHEIT = "F"
    CELSIUS = "C"


def celsius_to_fahrenheit(celsius: Decimal) -> Decimal:
    """Exact for any finite decimal input, because 9/5 = 1.8 is a terminating decimal."""

    return celsius * Decimal(9) / Decimal(5) + Decimal(32)


class TemperatureReading(_Fact):
    observed_at: AwareDatetime
    precision: TimePrecision = TimePrecision.MINUTE
    # Decimal keeps conversions exact: 38.0 C must compare as exactly 100.4 F.
    value: Decimal
    unit: TemperatureUnit
    evidence: tuple[Evidence, ...] = Field(min_length=1)

    @property
    def value_f(self) -> Decimal:
        if self.unit is TemperatureUnit.FAHRENHEIT:
            return self.value
        return celsius_to_fahrenheit(self.value)


class ReconciledPatientFacts(_Fact):
    """Everything the rule engine needs, already merged from JSON and documents."""

    case_id: str
    procedure: ProcedureFacts
    history_and_physicals: tuple[HistoryAndPhysical, ...] = ()
    consents: tuple[SurgicalConsent, ...] = ()
    labs: tuple[LabResult, ...] = ()
    medications: tuple[ResolvedMedication, ...] = ()
    anticoagulation_plans: tuple[AnticoagulationPlan, ...] = ()
    blood_pressures: tuple[BloodPressureReading, ...] = ()
    temperatures: tuple[TemperatureReading, ...] = ()
    # Sources whose content is unknown (failed extraction, unreadable record,
    # unverifiable recent vital). Any entry fails closed as MISSING_REQUIRED_DATA,
    # because the unknown content could hide a blocking fact.
    unreadable_sources: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _medications_are_merged(self) -> Self:
        names = [med.generic_name for med in self.medications]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"medications must be merged per generic drug; duplicates: {duplicates}")
        return self

    @model_validator(mode="after")
    def _plan_ids_unique(self) -> Self:
        ids = [plan.plan_id for plan in self.anticoagulation_plans]
        if len(ids) != len(set(ids)):
            raise ValueError("anticoagulation plan_ids must be unique")
        return self
