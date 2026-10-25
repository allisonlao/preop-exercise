"""Unit tests for the pure rule engine: boundaries, branches, and invariants."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from engine import (
    check_acute_safety,
    check_anticoagulation,
    check_documentation,
    check_preconditions,
    check_testing,
    evaluate,
    most_recent_readings,
)
from fact_builders import (
    antiplatelet,
    at,
    bp,
    cbc,
    cmp,
    consent,
    doc_ev,
    hp,
    med,
    plan,
    procedure,
    ready_facts,
    temp,
    temp_c,
)
from models import (
    Decision,
    Issue,
    IssueCategory,
    LabStatus,
    MedicationStatus,
    ReasonCode,
    ReconciledPatientFacts,
    RuleId,
    TimePrecision,
    TriageResult,
)


def codes(issues: list[Issue]) -> list[ReasonCode]:
    return [issue.reason_code for issue in issues]


# =============================================================================
# Baseline and decision precedence
# =============================================================================


def test_ready_baseline_has_no_issues() -> None:
    result = evaluate(ready_facts())
    assert result.decision is Decision.READY
    assert result.issues == []


def test_acute_exclusion_takes_precedence_and_other_issues_are_still_reported() -> None:
    facts = ready_facts(consents=(consent(signed=False),), temperatures=(temp("101.0", "2026-02-28T09:00"),))
    result = evaluate(facts)
    assert result.decision is Decision.NOT_CLEARED
    assert {i.category for i in result.issues} == {
        IssueCategory.ACUTE_SAFETY_EXCLUSION,
        IssueCategory.REQUIRED_DOCUMENTATION,
    }


def test_issue_rejects_reason_code_from_another_category() -> None:
    with pytest.raises(ValidationError):
        Issue(
            category=IssueCategory.REQUIRED_TESTING,
            reason_code=ReasonCode.HP_MISSING,
            rule=RuleId.R1_DOCUMENTATION,
            description="x",
            evidence=[doc_ev("d")],
            resolution="y",
        )


def test_triage_result_rejects_inconsistent_decision() -> None:
    issue = check_testing(ready_facts(labs=()))[0]
    with pytest.raises(ValidationError):
        TriageResult(decision=Decision.READY, issues=[issue], explanation="")


def test_output_is_independent_of_input_list_order() -> None:
    bps = (bp(120, 80, "2026-02-20T09:00"), bp(185, 90, "2026-02-27T09:00"), bp(130, 85, "2026-02-25T09:00"))
    labs = (cbc("2026-01-01T08:00"), cbc("2026-02-25T08:00", LabStatus.ENTERED_IN_ERROR), cbc("2026-01-15T08:00"))
    results = {
        evaluate(ready_facts(blood_pressures=b, labs=lab_order)).model_dump_json()
        for b in itertools.permutations(bps)
        for lab_order in itertools.permutations(labs)
    }
    assert len(results) == 1


def test_evaluation_is_deterministic() -> None:
    facts = ready_facts(labs=(), consents=())
    assert evaluate(facts).model_dump_json() == evaluate(facts).model_dump_json()


# =============================================================================
# R0 - preconditions
# =============================================================================


def test_missing_procedure_date_blocks_window_checks_only() -> None:
    # Labels (cases 11, 44) show date-independent rules still run when the date is null.
    facts = ready_facts(
        procedure=procedure(procedure_date=None),
        history_and_physicals=(),  # would fail R1 if the window check ran
        labs=(),  # would fail R2 if it ran
        consents=(consent(signed=False),),
        temperatures=(),
    )
    assert codes(evaluate(facts).issues) == [
        ReasonCode.PROCEDURE_DATE_MISSING,
        ReasonCode.CONSENT_UNSIGNED,
        ReasonCode.TEMPERATURE_MISSING,
    ]


def test_missing_date_issue_lists_blocked_checks() -> None:
    issue = check_preconditions(ready_facts(procedure=procedure(procedure_date=None)))[0]
    assert issue.category is IssueCategory.MISSING_REQUIRED_DATA
    assert issue.evidence[0].detail is not None and "Blocked checks" in issue.evidence[0].detail


@pytest.mark.parametrize("raw", [None, "SEVERE", "high"])
def test_missing_or_invalid_risk_skips_testing(raw: str | None) -> None:
    facts = ready_facts(procedure=procedure(risk=raw), labs=())
    assert codes(evaluate(facts).issues) == [ReasonCode.RISK_LEVEL_UNKNOWN]


# =============================================================================
# R1 - documentation
# =============================================================================


@pytest.mark.parametrize(
    ("performed_on", "passes"),
    [
        ("2026-01-30", True),  # exactly 30 days before 2026-03-01
        ("2026-01-29", False),  # 31 days
        ("2026-03-01", True),  # same day
        ("2026-03-02", False),  # after the procedure
    ],
)
def test_hp_window_boundaries(performed_on: str, passes: bool) -> None:
    issues = check_documentation(ready_facts(history_and_physicals=(hp(performed_on),)))
    assert codes(issues) == ([] if passes else [ReasonCode.HP_OUTSIDE_WINDOW])


def test_any_completed_hp_in_window_counts_even_if_an_older_one_exists() -> None:
    hps = (hp("2025-06-01", doc="wellness"), hp("2026-02-15", doc="preop"))
    assert check_documentation(ready_facts(history_and_physicals=hps)) == []


@pytest.mark.parametrize(
    ("hps", "expected"),
    [
        ((), ReasonCode.HP_MISSING),
        ((hp("2026-02-20", completed=False),), ReasonCode.HP_NOT_COMPLETED),
        ((hp(None),), ReasonCode.HP_DATE_UNKNOWN),
    ],
)
def test_hp_failure_reasons(hps: tuple[object, ...], expected: ReasonCode) -> None:
    assert codes(check_documentation(ready_facts(history_and_physicals=hps))) == [expected]


@pytest.mark.parametrize(
    ("consents", "expected"),
    [
        ((), ReasonCode.CONSENT_MISSING),
        ((consent(signed=False),), ReasonCode.CONSENT_UNSIGNED),
        ((consent(revoked=True),), ReasonCode.CONSENT_REVOKED),
        ((consent(procedure_matches=False),), ReasonCode.CONSENT_PROCEDURE_MISMATCH),
        ((consent(procedure_matches=None),), ReasonCode.CONSENT_PROCEDURE_UNVERIFIED),
    ],
)
def test_consent_failure_reasons(consents: tuple[object, ...], expected: ReasonCode) -> None:
    assert codes(check_documentation(ready_facts(consents=consents))) == [expected]


def test_later_signed_consent_supersedes_pending_one() -> None:
    consents = (
        consent(signed=False, doc="old", on="2026-02-10"),
        consent(signed=True, doc="new", on="2026-02-25", signed_at="2026-02-25T10:00"),
    )
    assert check_documentation(ready_facts(consents=consents)) == []


def test_consent_failure_reports_most_recent_consent_and_cites_all() -> None:
    consents = (
        consent(procedure_matches=False, doc="other-procedure", on="2026-02-01", signed_at="2026-02-01T10:00"),
        consent(signed=False, doc="pending", on="2026-02-20"),
    )
    (issue,) = check_documentation(ready_facts(consents=consents))
    assert issue.reason_code is ReasonCode.CONSENT_UNSIGNED
    assert len(issue.evidence) == 2


# =============================================================================
# R2 - testing
# =============================================================================


@pytest.mark.parametrize(
    ("risk", "collected", "passes"),
    [
        ("LOW", "2026-01-30T08:00", True),  # 30 days
        ("LOW", "2026-01-29T23:59", False),  # 31 days
        ("MODERATE", "2026-01-30T08:00", True),
        ("MODERATE", "2026-01-29T08:00", False),
        ("HIGH", "2026-02-15T08:00", True),  # 14 days
        ("HIGH", "2026-02-14T08:00", False),  # 15 days
    ],
)
def test_cbc_window_boundaries(risk: str, collected: str, passes: bool) -> None:
    labs = (cbc(collected), cmp("2026-02-25T08:00"))
    issues = check_testing(ready_facts(procedure=procedure(risk=risk), labs=labs))
    assert codes(issues) == ([] if passes else [ReasonCode.CBC_OUTSIDE_WINDOW])


def test_window_uses_utc_calendar_date() -> None:
    # 2026-01-29T20:00-05:00 is 2026-01-30T01:00Z: exactly 30 days, so it passes.
    assert check_testing(ready_facts(labs=(cbc("2026-01-29T20:00-05:00"),))) == []


def test_high_risk_reports_one_issue_per_failing_test() -> None:
    facts = ready_facts(procedure=procedure(risk="HIGH"), labs=(cbc("2026-01-01T08:00"),))
    assert codes(check_testing(facts)) == [ReasonCode.CBC_OUTSIDE_WINDOW, ReasonCode.CMP_MISSING]


def test_low_risk_ignores_cmp() -> None:
    facts = ready_facts(labs=(cbc("2026-02-20T08:00"), cmp("2025-01-01T08:00")))
    assert check_testing(facts) == []


def test_invalid_status_is_skipped_and_previous_result_used() -> None:
    labs = (cbc("2026-02-25T08:00", LabStatus.ENTERED_IN_ERROR), cbc("2026-02-10T08:00"))
    assert check_testing(ready_facts(labs=labs)) == []


def test_only_invalid_results_means_missing() -> None:
    labs = (cbc("2026-02-25T08:00", LabStatus.ENTERED_IN_ERROR),)
    assert codes(check_testing(ready_facts(labs=labs))) == [ReasonCode.CBC_MISSING]


def test_only_most_recent_valid_result_is_considered() -> None:
    # The newest valid CBC is dated after the procedure; an older in-window CBC does not rescue it.
    labs = (cbc("2026-02-20T08:00"), cbc("2026-03-02T08:00"))
    assert codes(check_testing(ready_facts(labs=labs))) == [ReasonCode.CBC_OUTSIDE_WINDOW]


# =============================================================================
# R3 - anticoagulation
# =============================================================================


def test_complete_plan_for_active_anticoagulant_passes() -> None:
    facts = ready_facts(medications=(med("apixaban"),), anticoagulation_plans=(plan(("Apixaban",)),))
    assert check_anticoagulation(facts) == []


def test_antiplatelet_and_inactive_anticoagulant_need_no_plan() -> None:
    meds = (antiplatelet("aspirin"), antiplatelet("clopidogrel"), med("warfarin", MedicationStatus.INACTIVE))
    assert check_anticoagulation(ready_facts(medications=meds)) == []


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (MedicationStatus.CONFLICT, ReasonCode.ANTICOAG_STATUS_CONFLICT),
        (MedicationStatus.UNKNOWN, ReasonCode.ANTICOAG_STATUS_UNKNOWN),
    ],
)
def test_uncertain_status_is_missing_data(status: MedicationStatus, expected: ReasonCode) -> None:
    (issue,) = check_anticoagulation(ready_facts(medications=(med("warfarin", status),)))
    assert issue.reason_code is expected
    assert issue.category is IssueCategory.MISSING_REQUIRED_DATA


@pytest.mark.parametrize(
    ("plan_kwargs", "expected"),
    [
        ({"hedged": True, "preop": None, "postop": None}, ReasonCode.ANTICOAG_PLAN_HEDGED),
        ({"timing": False}, ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE),
        ({"preop": None}, ReasonCode.ANTICOAG_PLAN_PREOP_INCOMPLETE),
        ({"postop": None}, ReasonCode.ANTICOAG_PLAN_POSTOP_MISSING),
    ],
)
def test_incomplete_plan_reasons(plan_kwargs: dict[str, object], expected: ReasonCode) -> None:
    facts = ready_facts(
        medications=(med("warfarin"),),
        anticoagulation_plans=(plan(("warfarin",), **plan_kwargs),),  # type: ignore[arg-type]
    )
    (issue,) = check_anticoagulation(facts)
    assert issue.reason_code is expected
    assert issue.category is IssueCategory.ANTICOAGULATION_MANAGEMENT


def test_missing_plan() -> None:
    assert codes(check_anticoagulation(ready_facts(medications=(med("rivaroxaban"),)))) == [
        ReasonCode.ANTICOAG_PLAN_MISSING
    ]


def test_plan_for_a_different_drug_is_a_mismatch() -> None:
    facts = ready_facts(
        medications=(med("apixaban"), med("warfarin", MedicationStatus.INACTIVE)),
        anticoagulation_plans=(plan(("warfarin",)),),
    )
    assert codes(check_anticoagulation(facts)) == [ReasonCode.ANTICOAG_PLAN_DRUG_MISMATCH]


def test_superseded_plan_is_ignored() -> None:
    facts = ready_facts(medications=(med("warfarin"),), anticoagulation_plans=(plan(("warfarin",), superseded=True),))
    assert codes(check_anticoagulation(facts)) == [ReasonCode.ANTICOAG_PLAN_MISSING]


def test_conflicting_plans_fail() -> None:
    plans = (
        plan(("warfarin",), plan_id="a", conflicts_with=("b",)),
        plan(("warfarin",), plan_id="b", preop="Hold 3 days before surgery"),
    )
    facts = ready_facts(medications=(med("warfarin"),), anticoagulation_plans=plans)
    assert codes(check_anticoagulation(facts)) == [ReasonCode.ANTICOAG_PLAN_CONFLICT]


def test_one_complete_plan_is_enough_when_another_is_hedged() -> None:
    plans = (plan(("warfarin",), plan_id="a", hedged=True), plan(("warfarin",), plan_id="b"))
    facts = ready_facts(medications=(med("warfarin"),), anticoagulation_plans=plans)
    assert check_anticoagulation(facts) == []


def test_each_active_anticoagulant_is_checked_separately() -> None:
    facts = ready_facts(
        medications=(med("warfarin"), med("apixaban")),
        anticoagulation_plans=(plan(("warfarin",)),),
    )
    assert codes(check_anticoagulation(facts)) == [ReasonCode.ANTICOAG_PLAN_DRUG_MISMATCH]


def test_duplicate_generic_medications_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ready_facts(medications=(med("warfarin"), med("Warfarin")))


# =============================================================================
# R4 - acute safety
# =============================================================================


@pytest.mark.parametrize(
    ("systolic", "diastolic", "expected"),
    [
        (179, 109, []),
        (180, 80, [ReasonCode.SYSTOLIC_BP_HIGH]),
        (120, 110, [ReasonCode.DIASTOLIC_BP_HIGH]),
        (180, 110, [ReasonCode.SYSTOLIC_BP_HIGH, ReasonCode.DIASTOLIC_BP_HIGH]),
    ],
)
def test_bp_thresholds(systolic: int, diastolic: int, expected: list[ReasonCode]) -> None:
    facts = ready_facts(blood_pressures=(bp(systolic, diastolic, "2026-02-28T09:00"),))
    assert codes(check_acute_safety(facts)) == expected


@pytest.mark.parametrize(
    ("reading", "febrile"),
    [
        (temp("100.4", "2026-02-28T09:00"), False),
        (temp("100.5", "2026-02-28T09:00"), True),
        (temp_c("38.0", "2026-02-28T09:00"), False),  # exactly 100.4 F
        (temp_c("38.1", "2026-02-28T09:00"), True),  # 100.58 F
        (temp_c("37.9", "2026-02-28T09:00"), False),  # 100.22 F
    ],
)
def test_temperature_threshold_is_strict_and_conversion_exact(reading: object, febrile: bool) -> None:
    facts = ready_facts(temperatures=(reading,))
    assert codes(check_acute_safety(facts)) == ([ReasonCode.TEMPERATURE_HIGH] if febrile else [])


def test_all_violations_are_reported() -> None:
    facts = ready_facts(
        blood_pressures=(bp(190, 115, "2026-02-28T09:00"),),
        temperatures=(temp("101.2", "2026-02-28T09:00"),),
    )
    assert codes(check_acute_safety(facts)) == [
        ReasonCode.SYSTOLIC_BP_HIGH,
        ReasonCode.DIASTOLIC_BP_HIGH,
        ReasonCode.TEMPERATURE_HIGH,
    ]


def test_later_normal_recheck_clears_earlier_high_reading() -> None:
    bps = (bp(189, 112, "2026-02-28T08:06"), bp(137, 78, "2026-02-28T08:30"))
    assert check_acute_safety(ready_facts(blood_pressures=bps)) == []


def test_later_high_recheck_is_not_cleared_by_earlier_normal() -> None:
    bps = (bp(120, 80, "2026-02-28T08:00"), bp(181, 98, "2026-02-28T09:38"))
    assert codes(check_acute_safety(ready_facts(blood_pressures=bps))) == [ReasonCode.SYSTOLIC_BP_HIGH]


def test_bp_and_temperature_are_evaluated_independently() -> None:
    # The latest visit has BP only; the latest temperature comes from an earlier visit.
    facts = ready_facts(
        blood_pressures=(bp(120, 80, "2026-02-28T09:00"),),
        temperatures=(temp("101.8", "2026-02-25T09:00"),),
    )
    assert codes(check_acute_safety(facts)) == [ReasonCode.TEMPERATURE_HIGH]


def test_missing_vitals_are_missing_data() -> None:
    facts = ready_facts(blood_pressures=(), temperatures=())
    issues = check_acute_safety(facts)
    assert codes(issues) == [ReasonCode.BP_MISSING, ReasonCode.TEMPERATURE_MISSING]
    assert {i.category for i in issues} == {IssueCategory.MISSING_REQUIRED_DATA}


def test_identical_timestamps_use_worst_case() -> None:
    bps = (bp(120, 80, "2026-02-28T09:00"), bp(182, 80, "2026-02-28T09:00"))
    assert codes(check_acute_safety(ready_facts(blood_pressures=bps))) == [ReasonCode.SYSTOLIC_BP_HIGH]


def test_date_only_reading_overlapping_latest_timed_reading_is_a_candidate() -> None:
    timed = bp(120, 80, "2026-02-28T09:00")
    untimed = bp(185, 80, "2026-02-28", precision=TimePrecision.DAY)
    older = bp(190, 80, "2026-02-20T09:00")
    assert most_recent_readings([older, timed, untimed]) == [timed, untimed]


def test_date_only_reading_on_earlier_day_is_not_a_candidate() -> None:
    timed = bp(120, 80, "2026-02-28T09:00")
    untimed = bp(185, 80, "2026-02-27", precision=TimePrecision.DAY)
    assert most_recent_readings([timed, untimed]) == [timed]


def test_aware_timestamps_compare_across_offsets() -> None:
    # 08:00-05:00 is 13:00Z, later than 12:00Z.
    early = bp(185, 80, "2026-02-28T12:00Z")
    late = bp(120, 80, "2026-02-28T08:00-05:00")
    assert at("2026-02-28T08:00-05:00") > at("2026-02-28T12:00Z")
    assert check_acute_safety(ready_facts(blood_pressures=(early, late))) == []


def test_facts_reject_naive_timestamps() -> None:
    with pytest.raises(ValidationError):
        ReconciledPatientFacts.model_validate(
            {
                **ready_facts().model_dump(),
                "blood_pressures": [
                    {"observed_at": "2026-02-28T09:00", "systolic": 120, "diastolic": 80, "evidence": [doc_ev("x").model_dump()]}
                ],
            }
        )
