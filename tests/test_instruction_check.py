"""Tests for the Layer 3 deterministic quote check (instruction_check.py)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from instruction_check import (
    ACTION_TERMS,
    CLINICAL_ANCHORS,
    DEFERRAL_PHRASES,
    check_instruction,
    is_instruction,
)
from schemas import PostopAction, PreopAction

ALL_ACTIONS: list[PreopAction | PostopAction] = [*PreopAction, *PostopAction]

# Building blocks for generated quotes: every list term plus anchors, decoys and filler.
FRAGMENTS = sorted(
    {*ACTION_TERMS, *DEFERRAL_PHRASES, *CLINICAL_ANCHORS}
    | {"24 hours", "48-72 hrs", "POD 2", "3 days", "1 week", "after surgery", "warfarin", "apixaban",
       "resumed", "restarted", "holding", "the surgeon's", "discretion", "at", "per", "will", "",
       "RESUME", "Per Cardiology", "TBD", "…", "\u00a0", "\n", ";", ","}
)
quotes = st.lists(st.sampled_from(FRAGMENTS), max_size=8).map(" ".join) | st.text(max_size=80)
deterministic = settings(derandomize=True, database=None, max_examples=1000)


@deterministic
@given(action=st.sampled_from(ALL_ACTIONS), quote=quotes)
def test_check_never_returns_a_more_permissive_action(action: PreopAction | PostopAction, quote: str) -> None:
    effective = check_instruction(action, quote)
    assert type(effective) is type(action)
    if not is_instruction(action):
        assert effective is action
    else:
        assert effective is action or effective in {type(action).DEFERRED, type(action).UNCLEAR}
    assert not is_instruction(effective) or effective is action


@deterministic
@given(action=st.sampled_from(ALL_ACTIONS), quote=quotes)
def test_check_is_idempotent(action: PreopAction | PostopAction, quote: str) -> None:
    once = check_instruction(action, quote)
    assert check_instruction(once, quote) is once


@pytest.mark.parametrize("quote", [
    "resume per cardiology",
    "to be determined",
    "will discuss postop",
    "restart at the surgeon's discretion",
])
def test_bare_hand_offs_downgrade_to_deferred(quote: str) -> None:
    assert check_instruction(PostopAction.RESUME, quote) is PostopAction.DEFERRED


@pytest.mark.parametrize("quote", [
    "resume 24 hours after surgery",
    "restart on POD 2",
    "resume when hemostasis is secure, per surgical team",
    "Resume Xarelto 24-48 hours after surgery once hemostasis is secure, per surgical team.",
])
def test_anchored_instructions_stay_resume(quote: str) -> None:
    assert check_instruction(PostopAction.RESUME, quote) is PostopAction.RESUME


def test_quote_without_an_action_term_downgrades_to_unclear() -> None:
    deferral = "Post-operative anticoagulation will be addressed after surgery."
    assert check_instruction(PostopAction.RESUME, deferral) is PostopAction.UNCLEAR
    assert check_instruction(PreopAction.HOLD, "Anticoagulation reviewed.") is PreopAction.UNCLEAR


def test_action_terms_match_whole_words_only() -> None:
    assert check_instruction(PostopAction.RESUME, "bridging restarted 48-72 hours post-op") is PostopAction.UNCLEAR
    assert check_instruction(PostopAction.RESUME, "Restart dabigatran on POD1") is PostopAction.RESUME


def test_deferral_phrase_with_anchor_but_no_action_term_is_still_unclear() -> None:
    assert check_instruction(PostopAction.RESUME, "per cardiology, 48 hours after surgery") is PostopAction.UNCLEAR


@pytest.mark.parametrize("action", [a for a in ALL_ACTIONS if not is_instruction(a)])
def test_missing_actions_are_left_alone(action: PreopAction | PostopAction) -> None:
    for quote in ("Resume apixaban 48 hours post-op.", "to be determined", "", "Hold warfarin per cardiology."):
        assert check_instruction(action, quote) is action


@pytest.mark.parametrize(("action", "quote"), [
    (PreopAction.HOLD, "Hold Xarelto for the 2 days before surgery (last dose 02/26/2026)."),
    (PreopAction.HOLD, "Stop warfarin 5 days before surgery; check INR the day before surgery."),
    (PreopAction.BRIDGE, "Bridge with enoxaparin 1 mg/kg SC q12h starting 3 days prior."),
    (PreopAction.CONTINUE, "Continue aspirin through surgery without interruption."),
])
def test_firm_preop_instructions_are_kept(action: PreopAction, quote: str) -> None:
    assert check_instruction(action, quote) is action


def test_preop_hand_off_downgrades_to_deferred() -> None:
    quote = "Periprocedural management of dabigatran to be determined by the surgical team."
    assert check_instruction(PreopAction.HOLD, quote) is PreopAction.DEFERRED
