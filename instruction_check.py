"""Layer 3: deterministic check of a pre-op / post-op action against its quote.

The extraction model labels each side of an anticoagulation plan with an action
enum. This module re-reads the verified quote and may only make that label
more restrictive: an instruction action (HOLD/CONTINUE/BRIDGE, RESUME/
CONTINUE_HOLD/DISCONTINUE) can become DEFERRED or UNCLEAR, and every other
action is returned unchanged. It never turns a missing action into an
instruction.

Order of checks for an instruction action:
  1. Deny-list: a deferral phrase with no concrete anchor -> DEFERRED.
  2. Allow-list: no action term at all -> UNCLEAR.
The deny-list runs first so that a bare hand-off with no action verb
("to be determined", "will discuss postop") is reported as DEFERRED.
"""

from __future__ import annotations

import re
from typing import Final, overload

from schemas import POSTOP_INSTRUCTIONS, PREOP_INSTRUCTIONS, PostopAction, PreopAction
from verify import normalize_for_match

ACTION_TERMS: Final = frozenset(
    {"resume", "restart", "reinitiate", "continue", "hold", "withhold", "stop", "discontinue", "bridge"}
)
DEFERRAL_PHRASES: Final = frozenset({
    "per cardiology", "per surgeon", "per anesthesia", "per surgical team", "to be determined", "tbd",
    "to be decided", "will discuss", "at discretion", "pending", "if appropriate", "defer", "deferred",
})
# Phrases whose words may be separated by up to three other words, so that
# "at the surgeon's discretion" matches "at discretion".
GAPPED_PHRASES: Final = frozenset({"at discretion"})
CLINICAL_ANCHORS: Final = frozenset({"hemostasis", "bleeding", "inr"})


def _phrase_pattern(phrase: str) -> str:
    gap = r"(?:\s+\S+){0,3}\s+" if phrase in GAPPED_PHRASES else r"\s+"
    return r"\b" + gap.join(re.escape(word) for word in phrase.split()) + r"\b"


def _alternation(patterns: list[str]) -> re.Pattern[str]:
    # Sorted so the compiled pattern (and so behavior) never depends on set order.
    return re.compile("|".join(f"(?:{p})" for p in sorted(patterns)))


_ACTION: Final = _alternation([rf"\b{re.escape(term)}\b" for term in ACTION_TERMS])
_DEFERRAL: Final = _alternation([_phrase_pattern(phrase) for phrase in DEFERRAL_PHRASES])
_ANCHOR: Final = _alternation([
    r"\d+\s*(?:hour|hr|day|week|month)s?\b",
    r"\bpod(?:\b|\d)",
    *(rf"\b{re.escape(term)}\b" for term in CLINICAL_ANCHORS),
])


def normalize_quote(quote: str) -> str:
    """NFKC, collapsed whitespace, case-folded."""

    return normalize_for_match(quote).casefold()


def is_instruction(action: PreopAction | PostopAction) -> bool:
    if isinstance(action, PreopAction):
        return action in PREOP_INSTRUCTIONS
    return action in POSTOP_INSTRUCTIONS


def has_deferral(text: str) -> bool:
    return _DEFERRAL.search(text) is not None


def has_anchor(text: str) -> bool:
    return _ANCHOR.search(text) is not None


def has_action_term(text: str) -> bool:
    return _ACTION.search(text) is not None


@overload
def check_instruction(action: PreopAction, quote: str) -> PreopAction: ...
@overload
def check_instruction(action: PostopAction, quote: str) -> PostopAction: ...
def check_instruction(action: PreopAction | PostopAction, quote: str) -> PreopAction | PostopAction:
    """The effective action for ``action`` given its verbatim ``quote``.

    Returns ``action`` itself or, for an instruction action only, the same
    enum's DEFERRED or UNCLEAR. Never returns a more permissive action.
    """

    if not is_instruction(action):
        return action
    text = normalize_quote(quote)
    effective: PreopAction | PostopAction = action
    if has_deferral(text) and not has_anchor(text):
        effective = type(action).DEFERRED
    elif not has_action_term(text):
        effective = type(action).UNCLEAR
    if effective is not action and is_instruction(effective):
        raise AssertionError(f"check_instruction upgraded {action} to {effective}")
    return effective
