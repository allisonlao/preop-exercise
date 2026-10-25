"""Stage 5 evidence packet: the final, deterministic triage output.

Turns the engine's ``TriageResult`` plus the ``ReconciledPatientFacts`` it was
computed from into ``TriageOutput``, the JSON a pre-op nurse reviews:

* ``issues``: every blocking finding with category, reason code, rule,
  description, evidence, and resolution.
* ``passed_checks``: every check that passed, with the facts that satisfied it
  ("Completed H&P dated 2026-02-21 ..."), so a READY can be spot-checked
  without re-reading the chart.
* ``explanation``: one deterministic summary sentence.

Evidence items are flat JSON objects: ``source`` (``json:vitals[1]``,
``doc:<doc_id>``, or ``absent:<where>``), ``date``, and either the verbatim
``quote`` (document text) or the ``raw_value`` (the JSON record as submitted).

Determinism: nothing here reads a clock or depends on input order. Issues are
sorted by category, reason code, then content; evidence chronologically, then
by content; passed checks by check id. Duplicate evidence is removed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from engine import (
    HP_WINDOW_DAYS,
    REQUIRED_TESTS,
    SYSTOLIC_LIMIT,
    DIASTOLIC_LIMIT,
    TEMPERATURE_LIMIT_F,
    VALID_LAB_STATUSES,
    consent_failure,
    days_before,
    most_recent_readings,
    plan_gaps,
    utc_date,
    within_window,
)
from models import (
    BloodPressureReading,
    Decision,
    DrugClass,
    Evidence,
    EvidenceSource,
    Issue,
    IssueCategory,
    LabTest,
    MedicationStatus,
    ReasonCode,
    ReconciledPatientFacts,
    RuleId,
    TemperatureReading,
    TemperatureUnit,
    TimePrecision,
    TriageResult,
    decision_for,
)

# =============================================================================
# Output contract
# =============================================================================


class _Out(BaseModel):
    # Enum fields are stored as plain strings so model_dump() is JSON-native.
    model_config = ConfigDict(frozen=True, extra="forbid", use_enum_values=True)


class EvidenceItem(_Out):
    source: str = Field(min_length=1)
    """``json:<path>`` for submission fields, ``doc:<doc_id>`` for document text,
    ``absent:<where>`` for a search that found nothing."""
    date: str | None
    """ISO date the evidence is from (collection, measurement, or document date)."""
    quote: str | None
    """Verbatim document text (document evidence only)."""
    raw_value: str | None
    """The JSON record exactly as submitted (structured evidence only)."""
    label: str | None
    detail: str | None


class TriageIssue(_Out):
    category: IssueCategory
    reason_code: ReasonCode
    rule: RuleId
    description: str = Field(min_length=1)
    evidence: list[EvidenceItem] = Field(min_length=1)
    resolution: str = Field(min_length=1)


class CheckId(StrEnum):
    PROCEDURE_DATA = "R0_PROCEDURE_DATA"
    SOURCES_READABLE = "R0_SOURCES_READABLE"
    HISTORY_AND_PHYSICAL = "R1_HISTORY_AND_PHYSICAL"
    CONSENT = "R1_CONSENT"
    CBC = "R2_CBC"
    CMP = "R2_CMP"
    ANTICOAGULATION = "R3_ANTICOAGULATION"
    BLOOD_PRESSURE = "R4_BLOOD_PRESSURE"
    TEMPERATURE = "R4_TEMPERATURE"


class PassedCheck(_Out):
    check: CheckId
    rule: RuleId
    description: str = Field(min_length=1)
    evidence: list[EvidenceItem]


class TriageOutput(_Out):
    """The final JSON returned for one submission."""

    decision: Decision
    issues: list[TriageIssue]
    passed_checks: list[PassedCheck]
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _decision_matches_issues(self) -> Self:
        expected = decision_for({IssueCategory(issue.category) for issue in self.issues})
        if self.decision != expected:
            raise ValueError(f"decision {self.decision} is inconsistent with issues ({expected})")
        return self


# =============================================================================
# Evidence formatting and ordering
# =============================================================================

_DOC_LOCATOR = re.compile(r"^documents\[(?P<doc_id>.+)\]$")


def evidence_item(evidence: Evidence) -> EvidenceItem:
    """Flatten engine evidence into the reviewer-facing shape."""

    on = evidence.recorded_on.isoformat() if evidence.recorded_on else None
    if evidence.source is EvidenceSource.STRUCTURED:
        return EvidenceItem(source=f"json:{evidence.locator}", date=on, quote=None,
                            raw_value=evidence.quote, label=evidence.label, detail=evidence.detail)
    if evidence.source is EvidenceSource.DOCUMENT:
        match = _DOC_LOCATOR.match(evidence.locator)
        doc_id = match["doc_id"] if match else evidence.locator
        return EvidenceItem(source=f"doc:{doc_id}", date=on, quote=evidence.quote,
                            raw_value=None, label=evidence.label, detail=evidence.detail)
    return EvidenceItem(source=f"absent:{evidence.locator}", date=on, quote=evidence.quote,
                        raw_value=None, label=evidence.label, detail=evidence.detail)


def _evidence_key(item: EvidenceItem) -> tuple[bool, str, str, str, str, str, str]:
    # Chronological; undated evidence (e.g. "absent:") last; then fully by content.
    return (item.date is None, item.date or "", item.source, item.quote or "",
            item.raw_value or "", item.label or "", item.detail or "")


def evidence_list(evidence: Iterable[Evidence]) -> list[EvidenceItem]:
    """Converted, de-duplicated, chronologically sorted evidence."""

    unique = {evidence_item(ev) for ev in evidence}
    return sorted(unique, key=_evidence_key)


def _issue_key(issue: TriageIssue) -> tuple[str, str, str, str]:
    first = issue.evidence[0]
    return (str(issue.category), str(issue.reason_code), issue.description, first.source)


def format_issue(issue: Issue) -> TriageIssue:
    return TriageIssue(
        category=issue.category,
        reason_code=issue.reason_code,
        rule=issue.rule,
        description=issue.description,
        evidence=evidence_list(issue.evidence),
        resolution=issue.resolution,
    )


# =============================================================================
# Passed checks (why a check passed, with the facts that satisfied it)
# =============================================================================

_PROCEDURE_CODES = frozenset({ReasonCode.PROCEDURE_DATE_MISSING, ReasonCode.RISK_LEVEL_UNKNOWN})
_HP_CODES = frozenset({ReasonCode.HP_MISSING, ReasonCode.HP_NOT_COMPLETED, ReasonCode.HP_DATE_UNKNOWN, ReasonCode.HP_OUTSIDE_WINDOW})
_CONSENT_CODES = frozenset({
    ReasonCode.CONSENT_MISSING, ReasonCode.CONSENT_UNSIGNED, ReasonCode.CONSENT_REVOKED,
    ReasonCode.CONSENT_PROCEDURE_MISMATCH, ReasonCode.CONSENT_PROCEDURE_UNVERIFIED,
})
_TEST_CODES = {
    LabTest.CBC: frozenset({ReasonCode.CBC_MISSING, ReasonCode.CBC_OUTSIDE_WINDOW}),
    LabTest.CMP: frozenset({ReasonCode.CMP_MISSING, ReasonCode.CMP_OUTSIDE_WINDOW}),
}
_TEST_CHECK = {LabTest.CBC: CheckId.CBC, LabTest.CMP: CheckId.CMP}
_BP_CODES = frozenset({ReasonCode.BP_MISSING, ReasonCode.SYSTOLIC_BP_HIGH, ReasonCode.DIASTOLIC_BP_HIGH})
_TEMP_CODES = frozenset({ReasonCode.TEMPERATURE_MISSING, ReasonCode.TEMPERATURE_HIGH})


def _passed(check: CheckId, rule: RuleId, description: str, evidence: Iterable[Evidence]) -> PassedCheck:
    return PassedCheck(check=check, rule=rule, description=description, evidence=evidence_list(evidence))


def _procedure_check(facts: ReconciledPatientFacts, failed: frozenset[ReasonCode]) -> list[PassedCheck]:
    procedure = facts.procedure
    if failed & _PROCEDURE_CODES:
        return []
    return [_passed(CheckId.PROCEDURE_DATA, RuleId.R0_PRECONDITIONS,
                    f"Procedure date {procedure.procedure_date}; risk level {procedure.risk_level}.",
                    [procedure.date_evidence, procedure.risk_evidence])]


def _sources_check(facts: ReconciledPatientFacts, failed: frozenset[ReasonCode]) -> list[PassedCheck]:
    if ReasonCode.EXTRACTION_FAILED in failed or facts.unreadable_sources:
        return []
    return [_passed(CheckId.SOURCES_READABLE, RuleId.R0_PRECONDITIONS,
                    "Every document was extracted and every structured vital/medication record was readable.", [])]


def _hp_check(facts: ReconciledPatientFacts, failed: frozenset[ReasonCode]) -> list[PassedCheck]:
    procedure_date = facts.procedure.procedure_date
    if procedure_date is None or failed & _HP_CODES:
        return []
    qualifying = [
        (hp, hp.performed_on) for hp in facts.history_and_physicals
        if hp.completed and hp.performed_on is not None and within_window(procedure_date, hp.performed_on, HP_WINDOW_DAYS)
    ]
    if not qualifying:
        return []
    hp, performed = max(qualifying, key=lambda pair: (pair[1], pair[0].document_id))
    return [_passed(CheckId.HISTORY_AND_PHYSICAL, RuleId.R1_DOCUMENTATION,
                    f"Completed H&P dated {performed} ({days_before(procedure_date, performed)} days before the "
                    f"{procedure_date} procedure; window is {HP_WINDOW_DAYS} days).", hp.evidence)]


def _consent_check(facts: ReconciledPatientFacts, failed: frozenset[ReasonCode]) -> list[PassedCheck]:
    if failed & _CONSENT_CODES:
        return []
    valid = [c for c in facts.consents if consent_failure(c) is None]
    if not valid:
        return []
    consent = max(valid, key=lambda c: (c.documented_on or date.min, c.document_id))
    return [_passed(CheckId.CONSENT, RuleId.R1_DOCUMENTATION,
                    f"Patient-signed, unrevoked consent for the scheduled procedure "
                    f"({facts.procedure.procedure_type or 'procedure'}), dated {consent.documented_on or 'undated'}.",
                    consent.evidence)]


def _testing_checks(facts: ReconciledPatientFacts, failed: frozenset[ReasonCode]) -> list[PassedCheck]:
    procedure_date, risk = facts.procedure.procedure_date, facts.procedure.risk_level
    if procedure_date is None or risk is None:
        return []
    checks: list[PassedCheck] = []
    for test, window in REQUIRED_TESTS[risk].items():
        if failed & _TEST_CODES[test]:
            continue
        valid = [lab for lab in facts.labs if lab.test is test and lab.status in VALID_LAB_STATUSES]
        if not valid:
            continue
        latest = max(valid, key=lambda lab: lab.collected_at)
        collected = utc_date(latest.collected_at)
        checks.append(_passed(_TEST_CHECK[test], RuleId.R2_TESTING,
                              f"{test} collected {collected} ({days_before(procedure_date, collected)} days before "
                              f"the procedure; window is {window} days for {risk} risk).", latest.evidence))
    return checks


def _anticoagulation_check(facts: ReconciledPatientFacts, failed_rules: frozenset[RuleId]) -> list[PassedCheck]:
    if RuleId.R3_ANTICOAGULATION in failed_rules:
        return []
    parts: list[str] = []
    evidence: list[Evidence] = []
    for med in sorted(facts.medications, key=lambda m: m.generic_name):
        if med.drug_class is not DrugClass.ANTICOAGULANT:
            continue
        evidence.extend(med.evidence)
        if med.status is MedicationStatus.INACTIVE:
            parts.append(f"{med.generic_name} is not currently taken")
            continue
        plans = sorted(
            (p for p in facts.anticoagulation_plans
             if not p.superseded and med.generic_name in p.drugs_addressed and not plan_gaps(p)),
            key=lambda p: (p.written_on or date.min, p.plan_id),
        )
        if med.status is not MedicationStatus.ACTIVE or not plans:
            # No R3 issue but no satisfying plan either (e.g. the plan-missing finding was
            # withheld because a document is unreadable): the check has not passed.
            return []
        plan = plans[-1]
        evidence.extend(plan.evidence)
        parts.append(f"{med.generic_name} is active with a complete perioperative plan "
                     f"(written {plan.written_on or 'undated'})")
    if not parts:
        parts.append("no anticoagulant found on the medication list or in the documents")
        evidence.append(Evidence(source=EvidenceSource.ABSENCE, locator="medications",
                                 detail="No anticoagulant in structured medications or document text."))
    antiplatelets = sorted(m.generic_name for m in facts.medications if m.drug_class is DrugClass.ANTIPLATELET)
    if antiplatelets:
        parts.append(f"antiplatelet(s) {', '.join(antiplatelets)} do not require a plan under Rule 3")
    summary = "; ".join(parts)
    return [_passed(CheckId.ANTICOAGULATION, RuleId.R3_ANTICOAGULATION, summary[:1].upper() + summary[1:] + ".", evidence)]


def _when(observed_at: datetime, precision: TimePrecision) -> str:
    if precision is TimePrecision.DAY:
        return f"{observed_at.strftime('%Y-%m-%d')} (time not recorded)"
    return f"{observed_at.strftime('%Y-%m-%d %H:%M')} UTC"


def _format_bp(reading: BloodPressureReading) -> str:
    return f"{reading.systolic}/{reading.diastolic} mmHg on {_when(reading.observed_at, reading.precision)}"


def _format_temp(reading: TemperatureReading) -> str:
    if reading.unit is TemperatureUnit.FAHRENHEIT:
        value = f"{reading.value} F"
    else:
        value = f"{reading.value} C ({reading.value_f.normalize()} F)"
    return f"{value} on {_when(reading.observed_at, reading.precision)}"


def _ordered(readings: Sequence[BloodPressureReading]) -> list[BloodPressureReading]:
    return sorted(readings, key=lambda r: (r.observed_at, r.systolic, r.diastolic))


def _vital_checks(facts: ReconciledPatientFacts, failed: frozenset[ReasonCode]) -> list[PassedCheck]:
    checks: list[PassedCheck] = []
    bps = _ordered(most_recent_readings(facts.blood_pressures))
    if bps and not failed & _BP_CODES:
        lead = "Most recent BP" if len(bps) == 1 else "Most recent BP candidates (time tie; all evaluated)"
        checks.append(_passed(CheckId.BLOOD_PRESSURE, RuleId.R4_ACUTE_SAFETY,
                              f"{lead}: {'; '.join(_format_bp(r) for r in bps)} "
                              f"(limits: systolic < {SYSTOLIC_LIMIT}, diastolic < {DIASTOLIC_LIMIT}).",
                              [ev for r in bps for ev in r.evidence]))
    temps = sorted(most_recent_readings(facts.temperatures), key=lambda r: (r.observed_at, r.value_f))
    if temps and not failed & _TEMP_CODES:
        lead = "Most recent temperature" if len(temps) == 1 else "Most recent temperature candidates (time tie; all evaluated)"
        checks.append(_passed(CheckId.TEMPERATURE, RuleId.R4_ACUTE_SAFETY,
                              f"{lead}: {'; '.join(_format_temp(r) for r in temps)} "
                              f"(limit: not above {TEMPERATURE_LIMIT_F} F).",
                              [ev for r in temps for ev in r.evidence]))
    return checks


def passed_checks(facts: ReconciledPatientFacts, issues: Sequence[Issue]) -> list[PassedCheck]:
    """Every check with no issue, described with the facts that satisfied it."""

    failed = frozenset(issue.reason_code for issue in issues)
    failed_rules = frozenset(issue.rule for issue in issues)
    checks = [
        *_procedure_check(facts, failed),
        *_sources_check(facts, failed),
        *_hp_check(facts, failed),
        *_consent_check(facts, failed),
        *_testing_checks(facts, failed),
        *_anticoagulation_check(facts, failed_rules),
        *_vital_checks(facts, failed),
    ]
    return sorted(checks, key=lambda c: str(c.check))


# =============================================================================
# Assembly
# =============================================================================


def _explain(decision: Decision, issues: Sequence[TriageIssue], checks: Sequence[PassedCheck]) -> str:
    if not issues:
        return "READY: every policy check passed. " + " ".join(check.description for check in checks)
    reasons = ", ".join(f"{issue.category} ({issue.reason_code})" for issue in issues)
    return f"{decision}: {len(issues)} issue(s): {reasons}. {len(checks)} check(s) passed; see passed_checks."


def build_output(facts: ReconciledPatientFacts, result: TriageResult) -> TriageOutput:
    """The deterministic evidence packet for one submission."""

    issues = sorted((format_issue(issue) for issue in result.issues), key=_issue_key)
    checks = passed_checks(facts, result.issues)
    return TriageOutput(
        decision=result.decision,
        issues=issues,
        passed_checks=checks,
        explanation=_explain(result.decision, issues, checks),
    )
