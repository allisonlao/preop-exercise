"""Terse constructors for ReconciledPatientFacts used by unit and gold-case tests.

``ready_facts()`` returns a baseline that passes every rule; tests override only
the fields they exercise. Note-derived clock times are treated as UTC.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from models import (
    AnticoagulationPlan,
    BloodPressureReading,
    DrugClass,
    Evidence,
    EvidenceSource,
    HistoryAndPhysical,
    LabResult,
    LabStatus,
    LabTest,
    MedicationStatus,
    ProcedureFacts,
    ReconciledPatientFacts,
    ResolvedMedication,
    RiskLevel,
    SurgicalConsent,
    TemperatureReading,
    TemperatureUnit,
    TimePrecision,
)


def at(stamp: str) -> datetime:
    """Parse '2026-03-01T10:15' or '...Z' as an aware UTC datetime; a bare date is midnight UTC."""

    parsed = datetime.fromisoformat(stamp)
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def d(day: str) -> date:
    return date.fromisoformat(day)


def doc_ev(doc_id: str, quote: str | None = None) -> Evidence:
    return Evidence(source=EvidenceSource.DOCUMENT, locator=f"documents[{doc_id}]", quote=quote)


def json_ev(locator: str) -> Evidence:
    return Evidence(source=EvidenceSource.STRUCTURED, locator=locator)


def procedure(
    procedure_date: str | None = "2026-03-01",
    risk: str | None = "LOW",
    case_id: str = "case_test",
) -> ProcedureFacts:
    return ProcedureFacts(
        case_id=case_id,
        procedure_type="Elective test procedure",
        procedure_date=d(procedure_date) if procedure_date else None,
        risk_level=RiskLevel(risk) if risk is not None and risk in RiskLevel.__members__ else None,
        risk_level_raw=risk,
        date_evidence=json_ev("procedure.procedure_date"),
        risk_evidence=json_ev("procedure.procedure_risk"),
    )


def hp(performed_on: str | None, completed: bool = True, doc: str = "hp") -> HistoryAndPhysical:
    return HistoryAndPhysical(
        document_id=doc,
        performed_on=d(performed_on) if performed_on else None,
        completed=completed,
        evidence=(doc_ev(doc),),
    )


def consent(
    signed: bool = True,
    procedure_matches: bool | None = True,
    revoked: bool = False,
    doc: str = "consent",
    on: str | None = None,
    signed_at: str | None = None,
) -> SurgicalConsent:
    return SurgicalConsent(
        document_id=doc,
        documented_on=d(on) if on else None,
        signed=signed,
        signed_at=at(signed_at) if signed_at else None,
        revoked=revoked,
        procedure_matches=procedure_matches,
        evidence=(doc_ev(doc),),
    )


def lab(test: LabTest, collected: str, status: LabStatus = LabStatus.FINAL, ref: str = "labs[0]") -> LabResult:
    return LabResult(test=test, code_raw=test.value, collected_at=at(collected), status=status, evidence=(json_ev(ref),))


def cbc(collected: str, status: LabStatus = LabStatus.FINAL) -> LabResult:
    return lab(LabTest.CBC, collected, status)


def cmp(collected: str, status: LabStatus = LabStatus.FINAL) -> LabResult:
    return lab(LabTest.CMP, collected, status)


def other_lab(collected: str) -> LabResult:
    return lab(LabTest.IGNORE, collected)


def bp(
    systolic: int | str,
    diastolic: int | str,
    observed: str,
    precision: TimePrecision = TimePrecision.MINUTE,
    ref: str = "vitals[0]",
) -> BloodPressureReading:
    return BloodPressureReading(
        observed_at=at(observed),
        precision=precision,
        systolic=Decimal(str(systolic)),
        diastolic=Decimal(str(diastolic)),
        evidence=(json_ev(ref),),
    )


def temp(
    value: str,
    observed: str,
    unit: TemperatureUnit = TemperatureUnit.FAHRENHEIT,
    precision: TimePrecision = TimePrecision.MINUTE,
    ref: str = "vitals[1]",
) -> TemperatureReading:
    return TemperatureReading(
        observed_at=at(observed), precision=precision, value=Decimal(value), unit=unit, evidence=(json_ev(ref),)
    )


def temp_c(value: str, observed: str) -> TemperatureReading:
    return temp(value, observed, unit=TemperatureUnit.CELSIUS)


def med(
    generic: str,
    status: MedicationStatus = MedicationStatus.ACTIVE,
    drug_class: DrugClass = DrugClass.ANTICOAGULANT,
) -> ResolvedMedication:
    return ResolvedMedication(
        generic_name=generic, display_name=generic, drug_class=drug_class, status=status, evidence=(json_ev("medications[0]"),)
    )


def antiplatelet(generic: str = "aspirin") -> ResolvedMedication:
    return med(generic, drug_class=DrugClass.ANTIPLATELET)


def plan(
    drugs: tuple[str, ...],
    plan_id: str = "plan",
    preop: str | None = "Hold 2 days before surgery",
    timing: bool = True,
    postop: str | None = "Resume 48 hours post-op",
    hedged: bool = False,
    superseded: bool = False,
    conflicts_with: tuple[str, ...] = (),
    written_on: str | None = None,
) -> AnticoagulationPlan:
    return AnticoagulationPlan(
        plan_id=plan_id,
        document_id=plan_id,
        written_on=d(written_on) if written_on else None,
        drugs_addressed=drugs,
        preop_instruction=preop,
        preop_timing_specified=timing,
        postop_instruction=postop,
        hedged=hedged,
        superseded=superseded,
        conflicts_with=conflicts_with,
        evidence=(doc_ev(plan_id),),
    )


def ready_facts(**overrides: Any) -> ReconciledPatientFacts:
    """A LOW-risk case on 2026-03-01 that passes every rule, with optional field overrides."""

    fields: dict[str, Any] = {
        "case_id": "case_test",
        "procedure": procedure(),
        "history_and_physicals": (hp("2026-02-20"),),
        "consents": (consent(),),
        "labs": (cbc("2026-02-20T08:00"),),
        "medications": (),
        "anticoagulation_plans": (),
        "blood_pressures": (bp(120, 80, "2026-02-25T10:00"),),
        "temperatures": (temp("98.6", "2026-02-25T10:00"),),
    }
    fields.update(overrides)
    return ReconciledPatientFacts(**fields)
