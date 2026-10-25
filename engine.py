"""Pure, deterministic rule engine for the Cadence pre-op scheduling policy.

Each rule is a pure function ``ReconciledPatientFacts -> list[Issue]``: no I/O,
no LLM, no clock. Rules never short-circuit each other, so a NOT_CLEARED case
still reports its documentation or testing gaps.

Policy constants are defined once at the top of this module. Date arithmetic
uses calendar days between UTC dates: an event ``N`` days before the procedure
passes a window of ``W`` days when ``0 <= N <= W``. Events dated after the
procedure date never satisfy a pre-op window.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from typing import TypeVar

from models import (
    REASON_CATEGORY,
    AnticoagulationPlan,
    BloodPressureReading,
    Decision,
    DrugClass,
    Evidence,
    EvidenceSource,
    HistoryAndPhysical,
    Issue,
    LabResult,
    LabStatus,
    LabTest,
    MedicationStatus,
    ReasonCode,
    ReconciledPatientFacts,
    ResolvedMedication,
    RiskLevel,
    RuleId,
    SurgicalConsent,
    TemperatureReading,
    TemperatureUnit,
    TimePrecision,
    TriageResult,
    decision_for,
)

# =============================================================================
# Policy constants
# =============================================================================

HP_WINDOW_DAYS = 30

# Required tests and their windows (days before the procedure), per risk level.
REQUIRED_TESTS: dict[RiskLevel, dict[LabTest, int]] = {
    RiskLevel.LOW: {LabTest.CBC: 30},
    RiskLevel.MODERATE: {LabTest.CBC: 30},
    RiskLevel.HIGH: {LabTest.CBC: 14, LabTest.CMP: 14},
}

# Results with any other status (entered-in-error, cancelled, preliminary, unknown)
# are skipped; the most recent *valid* result is the one considered.
VALID_LAB_STATUSES = frozenset({LabStatus.FINAL, LabStatus.AMENDED, LabStatus.CORRECTED})

SYSTOLIC_LIMIT = Decimal(180)  # NOT_CLEARED when systolic >= limit
DIASTOLIC_LIMIT = Decimal(110)  # NOT_CLEARED when diastolic >= limit
TEMPERATURE_LIMIT_F = Decimal("100.4")  # NOT_CLEARED when temperature > limit

_MISSING_TEST_CODE = {LabTest.CBC: ReasonCode.CBC_MISSING, LabTest.CMP: ReasonCode.CMP_MISSING}
_STALE_TEST_CODE = {
    LabTest.CBC: ReasonCode.CBC_OUTSIDE_WINDOW,
    LabTest.CMP: ReasonCode.CMP_OUTSIDE_WINDOW,
}

# =============================================================================
# Small pure helpers
# =============================================================================


def utc_date(moment: datetime) -> date:
    return moment.astimezone(UTC).date()


def days_before(procedure_date: date, event_date: date) -> int:
    """Calendar days from ``event_date`` to ``procedure_date`` (negative if after)."""

    return (procedure_date - event_date).days


def within_window(procedure_date: date, event_date: date, window_days: int) -> bool:
    return 0 <= days_before(procedure_date, event_date) <= window_days


def _describe_offset(procedure_date: date, event_date: date, window_days: int) -> str:
    offset = days_before(procedure_date, event_date)
    if offset < 0:
        return f"{-offset} day(s) after the procedure date; window is {window_days} days before"
    return f"{offset} day(s) before the procedure; window is {window_days} days"


def _issue(
    reason: ReasonCode,
    rule: RuleId,
    description: str,
    evidence: Iterable[Evidence],
    resolution: str,
) -> Issue:
    return Issue(
        category=REASON_CATEGORY[reason],
        reason_code=reason,
        rule=rule,
        description=description,
        evidence=list(evidence),
        resolution=resolution,
    )


def _absence(locator: str, detail: str) -> Evidence:
    return Evidence(source=EvidenceSource.ABSENCE, locator=locator, detail=detail)


def _annotate(evidence: Iterable[Evidence], detail: str) -> list[Evidence]:
    return [item.model_copy(update={"detail": detail}) for item in evidence]


_T = TypeVar("_T")


def _latest(items: Iterable[_T], key: Callable[[_T], datetime | date]) -> _T | None:
    """Most recent item by ``key``; ties broken by input order (first wins)."""

    best: _T | None = None
    for item in items:
        if best is None or key(item) > key(best):
            best = item
    return best


# =============================================================================
# R0 - preconditions
# =============================================================================


def check_preconditions(facts: ReconciledPatientFacts) -> list[Issue]:
    """Procedure date and risk level must be known; otherwise dependent checks are blocked."""

    procedure = facts.procedure
    issues: list[Issue] = []

    if procedure.procedure_date is None:
        issues.append(
            _issue(
                ReasonCode.PROCEDURE_DATE_MISSING,
                RuleId.R0_PRECONDITIONS,
                "Procedure date is not set in the submission; date-window checks cannot run.",
                _annotate(
                    [procedure.date_evidence],
                    "Blocked checks: R1 H&P 30-day window; R2 lab windows. "
                    "Provisional or target dates in notes are not used.",
                ),
                "Confirm the scheduled procedure date, then re-run triage.",
            )
        )

    if procedure.risk_level is None:
        raw = procedure.risk_level_raw
        shown = "missing" if raw is None else f"invalid ({raw!r})"
        issues.append(
            _issue(
                ReasonCode.RISK_LEVEL_UNKNOWN,
                RuleId.R0_PRECONDITIONS,
                f"Procedure risk level is {shown}; required testing cannot be determined.",
                _annotate([procedure.risk_evidence], "Blocked checks: R2 required testing."),
                "Have the surgical team assign a LOW, MODERATE, or HIGH risk level.",
            )
        )

    if facts.unreadable_sources:
        blocked = (
            " Until they are read, the H&P, consent, and anticoagulation-plan checks cannot pass."
            if documents_unreadable(facts)
            else ""
        )
        issues.append(
            _issue(
                ReasonCode.EXTRACTION_FAILED,
                RuleId.R0_PRECONDITIONS,
                f"{len(facts.unreadable_sources)} source item(s) could not be read reliably; "
                f"any facts they contain are unknown.{blocked}",
                facts.unreadable_sources,
                "Review the listed documents or records manually, then re-run triage.",
            )
        )

    return issues


# Findings that only mean "not found in the readable documents". If any document is
# unreadable the item may be inside it, so "missing" would be a false statement; the
# EXTRACTION_FAILED issue (MISSING_REQUIRED_DATA) already blocks READY instead.
ABSENCE_IN_DOCUMENTS = frozenset(
    {ReasonCode.HP_MISSING, ReasonCode.CONSENT_MISSING, ReasonCode.ANTICOAG_PLAN_MISSING}
)


def documents_unreadable(facts: ReconciledPatientFacts) -> bool:
    return any(
        ev.source is EvidenceSource.DOCUMENT or ev.locator.startswith("documents")
        for ev in facts.unreadable_sources
    )


# =============================================================================
# R1 - required documentation
# =============================================================================


def check_documentation(facts: ReconciledPatientFacts) -> list[Issue]:
    """H&P within 30 days (needs a procedure date) and a valid signed consent (always)."""

    issues: list[Issue] = []
    procedure_date = facts.procedure.procedure_date
    if procedure_date is not None:
        hp_issue = _check_history_and_physical(facts.history_and_physicals, procedure_date)
        if hp_issue is not None:
            issues.append(hp_issue)
    consent_issue = _check_consent(facts.consents)
    if consent_issue is not None:
        issues.append(consent_issue)
    return issues


def _check_history_and_physical(
    hps: Sequence[HistoryAndPhysical], procedure_date: date
) -> Issue | None:
    completed = [hp for hp in hps if hp.completed]
    dated = [(hp, hp.performed_on) for hp in completed if hp.performed_on is not None]

    # Any completed H&P inside the window satisfies the rule.
    if any(within_window(procedure_date, performed, HP_WINDOW_DAYS) for _, performed in dated):
        return None

    rule = RuleId.R1_DOCUMENTATION
    resolution = f"Obtain a completed H&P dated within {HP_WINDOW_DAYS} days of {procedure_date}."

    newest = _latest(dated, key=lambda pair: pair[1])
    if newest is not None:
        newest_hp, performed = newest
        return _issue(
            ReasonCode.HP_OUTSIDE_WINDOW,
            rule,
            f"Most recent completed H&P ({performed}) is outside the {HP_WINDOW_DAYS}-day window.",
            _annotate(newest_hp.evidence, _describe_offset(procedure_date, performed, HP_WINDOW_DAYS)),
            resolution,
        )
    if completed:
        return _issue(
            ReasonCode.HP_DATE_UNKNOWN,
            rule,
            "A completed H&P exists but its date of service cannot be determined.",
            [ev for hp in completed for ev in hp.evidence],
            resolution,
        )
    if hps:
        return _issue(
            ReasonCode.HP_NOT_COMPLETED,
            rule,
            "An H&P is on file but is not completed.",
            [ev for hp in hps for ev in hp.evidence],
            resolution,
        )
    return _issue(
        ReasonCode.HP_MISSING,
        rule,
        "No History and Physical found in the submission.",
        [_absence("documents", "No document identified as a History and Physical.")],
        resolution,
    )


def consent_failure(consent: SurgicalConsent) -> ReasonCode | None:
    """Why a consent does not qualify, checked in order of severity."""

    if consent.procedure_matches is False:
        return ReasonCode.CONSENT_PROCEDURE_MISMATCH
    if consent.revoked:
        return ReasonCode.CONSENT_REVOKED
    if not consent.signed:
        return ReasonCode.CONSENT_UNSIGNED
    if consent.procedure_matches is None:
        return ReasonCode.CONSENT_PROCEDURE_UNVERIFIED
    return None


_CONSENT_DESCRIPTIONS = {
    ReasonCode.CONSENT_PROCEDURE_MISMATCH: "The only consent(s) on file are for a different procedure.",
    ReasonCode.CONSENT_REVOKED: "The surgical consent on file has been revoked.",
    ReasonCode.CONSENT_UNSIGNED: "The surgical consent on file is not signed by the patient.",
    ReasonCode.CONSENT_PROCEDURE_UNVERIFIED: (
        "A signed consent exists but it cannot be confirmed to cover the scheduled procedure."
    ),
}


def _consent_recency(consent: SurgicalConsent) -> date:
    if consent.signed_at is not None:
        return utc_date(consent.signed_at)
    return consent.documented_on or date.min


def _check_consent(consents: Sequence[SurgicalConsent]) -> Issue | None:
    # Any one valid consent satisfies the rule (a later signed form supersedes a pending one).
    if any(consent_failure(consent) is None for consent in consents):
        return None

    rule = RuleId.R1_DOCUMENTATION
    resolution = "Obtain a patient-signed surgical consent for the scheduled procedure."
    if not consents:
        return _issue(
            ReasonCode.CONSENT_MISSING,
            rule,
            "No surgical consent found in the submission.",
            [_absence("documents", "No document identified as a surgical consent.")],
            resolution,
        )

    # Report the failure of the most recent consent; cite every consent so the reviewer sees all.
    newest = _latest(consents, key=_consent_recency)
    assert newest is not None
    reason = consent_failure(newest)
    assert reason is not None
    evidence = [ev for consent in consents for ev in consent.evidence]
    return _issue(reason, rule, _CONSENT_DESCRIPTIONS[reason], evidence, resolution)


# =============================================================================
# R2 - required testing
# =============================================================================


def check_testing(facts: ReconciledPatientFacts) -> list[Issue]:
    """One issue per required test that is missing or outside its window.

    Skipped entirely when the procedure date or risk level is unknown (R0 reports it).
    """

    procedure_date = facts.procedure.procedure_date
    risk = facts.procedure.risk_level
    if procedure_date is None or risk is None:
        return []

    issues: list[Issue] = []
    for test, window in REQUIRED_TESTS[risk].items():
        issue = _check_single_test(facts.labs, test, window, procedure_date, risk)
        if issue is not None:
            issues.append(issue)
    return issues


def _check_single_test(
    labs: Sequence[LabResult],
    test: LabTest,
    window: int,
    procedure_date: date,
    risk: RiskLevel,
) -> Issue | None:
    candidates = [lab for lab in labs if lab.test is test]
    valid = [lab for lab in candidates if lab.status in VALID_LAB_STATUSES]
    latest = _latest(valid, key=lambda lab: lab.collected_at)
    resolution = f"Obtain a {test} dated within {window} days of {procedure_date} ({risk} risk)."

    if latest is None:
        skipped = [lab for lab in candidates if lab.status not in VALID_LAB_STATUSES]
        evidence: list[Evidence] = [
            _absence("labs", f"No valid {test} result found among {len(labs)} lab entries.")
        ]
        evidence += _annotate(
            [ev for lab in skipped for ev in lab.evidence], "Not counted: result status is not valid."
        )
        return _issue(
            _MISSING_TEST_CODE[test], RuleId.R2_TESTING, f"Required {test} is missing.", evidence, resolution
        )

    collected = utc_date(latest.collected_at)
    if within_window(procedure_date, collected, window):
        return None
    return _issue(
        _STALE_TEST_CODE[test],
        RuleId.R2_TESTING,
        f"Most recent valid {test} ({collected}) is outside the {window}-day window.",
        _annotate(latest.evidence, _describe_offset(procedure_date, collected, window)),
        resolution,
    )


# =============================================================================
# R3 - anticoagulation management
# =============================================================================


def check_anticoagulation(facts: ReconciledPatientFacts) -> list[Issue]:
    """Every active anticoagulant needs a complete, unhedged, non-conflicting plan."""

    issues: list[Issue] = []
    anticoagulants = [m for m in facts.medications if m.drug_class is DrugClass.ANTICOAGULANT]
    for med in sorted(anticoagulants, key=lambda m: m.generic_name):
        if med.status is MedicationStatus.INACTIVE:
            continue
        if med.status is MedicationStatus.CONFLICT:
            issues.append(
                _issue(
                    ReasonCode.ANTICOAG_STATUS_CONFLICT,
                    RuleId.R3_ANTICOAGULATION,
                    f"Sources disagree on whether the patient is currently taking {med.generic_name}.",
                    med.evidence,
                    f"Confirm current {med.generic_name} use; if active, document a perioperative plan.",
                )
            )
        elif med.status is MedicationStatus.UNKNOWN:
            issues.append(
                _issue(
                    ReasonCode.ANTICOAG_STATUS_UNKNOWN,
                    RuleId.R3_ANTICOAGULATION,
                    f"Current use of {med.generic_name} is not established.",
                    med.evidence,
                    f"Confirm current {med.generic_name} use; if active, document a perioperative plan.",
                )
            )
        else:
            plan_issue = _check_plan_for(med, facts.anticoagulation_plans)
            if plan_issue is not None:
                issues.append(plan_issue)
    return issues


def plan_gaps(plan: AnticoagulationPlan) -> list[ReasonCode]:
    """Completeness gaps for a plan, most informative first."""

    gaps: list[ReasonCode] = []
    if plan.hedged:
        gaps.append(ReasonCode.ANTICOAG_PLAN_HEDGED)
    if plan.preop_instruction is None or not plan.preop_timing_specified:
        gaps.append(ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE)
    if plan.postop_instruction is None:
        gaps.append(ReasonCode.ANTICOAG_PLAN_POSTOP_MISSING)
    return gaps


_GAP_TEXT = {
    ReasonCode.ANTICOAG_PLAN_HEDGED: "management is deferred or non-committal",
    ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE: "no pre-op instruction with timing",
    ReasonCode.ANTICOAG_PLAN_POSTOP_MISSING: "no post-op restart instruction",
}


def _check_plan_for(
    med: ResolvedMedication, plans: Sequence[AnticoagulationPlan]
) -> Issue | None:
    rule = RuleId.R3_ANTICOAGULATION
    drug = med.generic_name
    resolution = f"Document a perioperative plan stating when to stop and when to restart {drug}."
    live = [plan for plan in plans if not plan.superseded]
    matching = [plan for plan in live if drug in plan.drugs_addressed]

    if not matching:
        if live:
            others = sorted({d for plan in live for d in plan.drugs_addressed})
            return _issue(
                ReasonCode.ANTICOAG_PLAN_DRUG_MISMATCH,
                rule,
                f"Patient is on {drug}, but the plan(s) on file address {', '.join(others) or 'no named drug'}.",
                [*med.evidence, *(ev for plan in live for ev in plan.evidence)],
                resolution,
            )
        return _issue(
            ReasonCode.ANTICOAG_PLAN_MISSING,
            rule,
            f"Patient is on {drug} and no perioperative anticoagulation plan is documented.",
            [*med.evidence, _absence("documents", "No perioperative anticoagulation plan found.")],
            resolution,
        )

    matching_ids = {plan.plan_id for plan in matching}
    conflicted = [plan for plan in matching if matching_ids.intersection(plan.conflicts_with)]
    if conflicted:
        return _issue(
            ReasonCode.ANTICOAG_PLAN_CONFLICT,
            rule,
            f"Documented plans for {drug} give contradictory instructions.",
            [ev for plan in conflicted for ev in plan.evidence],
            f"Reconcile the conflicting {drug} plans into one documented plan.",
        )

    if any(not plan_gaps(plan) for plan in matching):
        return None

    newest = _latest(matching, key=lambda plan: plan.written_on or date.min)
    assert newest is not None
    gaps = plan_gaps(newest)
    return _issue(
        gaps[0],
        rule,
        f"Plan for {drug} is incomplete: {'; '.join(_GAP_TEXT[g] for g in gaps)}.",
        [*med.evidence, *newest.evidence],
        resolution,
    )


# =============================================================================
# R4 - acute safety exclusions
# =============================================================================

_Reading = TypeVar("_Reading", BloodPressureReading, TemperatureReading)


def _time_bounds(observed_at: datetime, precision: TimePrecision) -> tuple[datetime, datetime]:
    """Earliest and latest instant a reading could have been taken."""

    if precision is TimePrecision.MINUTE:
        return observed_at, observed_at
    start = datetime.combine(utc_date(observed_at), time.min, tzinfo=UTC)
    return start, start + timedelta(days=1) - timedelta(microseconds=1)


def most_recent_readings(readings: Sequence[_Reading]) -> list[_Reading]:
    """All readings that could be the most recent one.

    Normally a single reading. When timestamps tie, or a date-only reading
    overlaps the latest timed one, every candidate is returned so the caller
    can apply worst-case evaluation.
    """

    if not readings:
        return []
    bounds = [_time_bounds(r.observed_at, r.precision) for r in readings]
    latest_start = max(start for start, _ in bounds)
    return [r for r, (_, end) in zip(readings, bounds, strict=True) if end >= latest_start]


def check_acute_safety(facts: ReconciledPatientFacts) -> list[Issue]:
    """Evaluate the most recent BP and temperature; one issue per violated threshold."""

    rule = RuleId.R4_ACUTE_SAFETY
    issues: list[Issue] = []

    bp_candidates = most_recent_readings(facts.blood_pressures)
    if not bp_candidates:
        issues.append(
            _issue(
                ReasonCode.BP_MISSING,
                rule,
                "No blood pressure reading found in structured vitals or documents.",
                [_absence("vitals", "No blood pressure reading in vitals or document text.")],
                "Record a current blood pressure.",
            )
        )
    else:
        high_systolic = [r for r in bp_candidates if r.systolic >= SYSTOLIC_LIMIT]
        if high_systolic:
            issues.append(
                _issue(
                    ReasonCode.SYSTOLIC_BP_HIGH,
                    rule,
                    f"Most recent systolic BP is >= {SYSTOLIC_LIMIT} mmHg.",
                    [ev for r in high_systolic for ev in _annotate(r.evidence, f"{r.systolic}/{r.diastolic} mmHg")],
                    "Hold scheduling; refer for blood pressure management and recheck.",
                )
            )
        high_diastolic = [r for r in bp_candidates if r.diastolic >= DIASTOLIC_LIMIT]
        if high_diastolic:
            issues.append(
                _issue(
                    ReasonCode.DIASTOLIC_BP_HIGH,
                    rule,
                    f"Most recent diastolic BP is >= {DIASTOLIC_LIMIT} mmHg.",
                    [ev for r in high_diastolic for ev in _annotate(r.evidence, f"{r.systolic}/{r.diastolic} mmHg")],
                    "Hold scheduling; refer for blood pressure management and recheck.",
                )
            )

    temp_candidates = most_recent_readings(facts.temperatures)
    if not temp_candidates:
        issues.append(
            _issue(
                ReasonCode.TEMPERATURE_MISSING,
                rule,
                "No temperature reading found in structured vitals or documents.",
                [_absence("vitals", "No temperature reading in vitals or document text.")],
                "Record a current temperature.",
            )
        )
    else:
        febrile = [r for r in temp_candidates if r.value_f > TEMPERATURE_LIMIT_F]
        if febrile:
            issues.append(
                _issue(
                    ReasonCode.TEMPERATURE_HIGH,
                    rule,
                    f"Most recent temperature is > {TEMPERATURE_LIMIT_F} F.",
                    [ev for r in febrile for ev in _annotate(r.evidence, _format_temperature(r))],
                    "Hold scheduling; evaluate for fever before rescheduling.",
                )
            )

    return issues


def _format_temperature(reading: TemperatureReading) -> str:
    if reading.unit is TemperatureUnit.FAHRENHEIT:
        return f"{reading.value} F"
    return f"{reading.value} C = {reading.value_f.normalize()} F"


# =============================================================================
# Orchestration
# =============================================================================

Rule = Callable[[ReconciledPatientFacts], list[Issue]]

RULES: tuple[Rule, ...] = (
    check_preconditions,
    check_documentation,
    check_testing,
    check_anticoagulation,
    check_acute_safety,
)


def evaluate(facts: ReconciledPatientFacts) -> TriageResult:
    """Run every rule and derive the decision from the combined issues."""

    issues = [issue for rule in RULES for issue in rule(facts)]
    if documents_unreadable(facts):
        issues = [issue for issue in issues if issue.reason_code not in ABSENCE_IN_DOCUMENTS]
    decision = decision_for({issue.category for issue in issues})
    return TriageResult(decision=decision, issues=issues, explanation=_explain(decision, issues))


def _explain(decision: Decision, issues: Sequence[Issue]) -> str:
    if not issues:
        return "READY: all documentation, testing, anticoagulation, and safety checks passed."
    reasons = ", ".join(f"{issue.category} ({issue.reason_code})" for issue in issues)
    return f"{decision}: {len(issues)} issue(s): {reasons}."
