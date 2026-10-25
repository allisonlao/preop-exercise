"""Stage 3 deterministic quote verification.

Checks every ``QuotedFact`` in an ``ExtractedDocument`` against the original
document text. No LLM involved.

Matching: both sides are Unicode-normalized (NFKC) and every whitespace run
(spaces, tabs, newlines) is collapsed to a single space, then the quote must be
a substring of the source. Matching is case-sensitive: a quote whose case was
changed counts as altered.

Rejected outright: quotes shorter than ``MIN_QUOTE_LENGTH`` characters after
normalization (trivially matchable) and quotes containing an ellipsis (a sign
of stitched or elided text).

FAIL CLOSED: a fact whose quote fails is never deleted. It is kept with
``verified = False`` and a warning is logged, so the reconciler can treat it as
unconfirmed (e.g. an unverified "consent revoked" still blocks READY).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterator
from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel

from schemas import ExtractedDocument, ExtractionStatus, QuotedFact

logger = logging.getLogger(__name__)

MIN_QUOTE_LENGTH: Final = 4
ELLIPSIS: Final = "..."  # NFKC maps the single-character "…" to "...".
_WHITESPACE: Final = re.compile(r"\s+")


class QuoteCheck(StrEnum):
    MATCH = "MATCH"
    EMPTY = "EMPTY"
    TOO_SHORT = "TOO_SHORT"
    ELLIPSIS = "ELLIPSIS"
    NOT_FOUND = "NOT_FOUND"


def normalize_for_match(text: str) -> str:
    """NFKC, then collapse every whitespace run to one space, then strip."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text)).strip()


def check_quote(quote: str, normalized_source: str) -> QuoteCheck:
    """Classify one quote against an already-normalized source text."""

    normalized = normalize_for_match(quote)
    if not normalized:
        return QuoteCheck.EMPTY
    if len(normalized) < MIN_QUOTE_LENGTH:
        return QuoteCheck.TOO_SHORT
    if ELLIPSIS in normalized:
        return QuoteCheck.ELLIPSIS
    if normalized not in normalized_source:
        return QuoteCheck.NOT_FOUND
    return QuoteCheck.MATCH


def iter_quoted_facts(node: Any, path: str = "$") -> Iterator[tuple[str, QuotedFact[Any]]]:
    """Yield ``(json_path, fact)`` for every QuotedFact reachable from ``node``."""

    if isinstance(node, QuotedFact):
        yield path, node
    if isinstance(node, BaseModel):
        for name in type(node).model_fields:
            yield from iter_quoted_facts(getattr(node, name), f"{path}.{name}")
    elif isinstance(node, (list, tuple)):
        for index, item in enumerate(node):
            yield from iter_quoted_facts(item, f"{path}[{index}]")


def _verify_node(node: Any, source: str, path: str, doc_label: str) -> Any:
    """Return a copy of ``node`` with every nested QuotedFact's ``verified`` set.

    Rebuilds via ``model_copy`` so the input is never mutated and nothing is
    removed; only ``verified`` flags change.
    """

    if isinstance(node, BaseModel):
        updates: dict[str, Any] = {
            name: _verify_node(getattr(node, name), source, f"{path}.{name}", doc_label)
            for name in type(node).model_fields
        }
        if isinstance(node, QuotedFact):
            result = check_quote(node.quote, source)
            updates["verified"] = result is QuoteCheck.MATCH
            if result is not QuoteCheck.MATCH:
                logger.warning(
                    "doc %s: unverified quote at %s (%s); keeping fact with verified=False: %r",
                    doc_label,
                    path,
                    result.value,
                    node.quote,
                )
        return node.model_copy(update=updates)
    if isinstance(node, list):
        return [_verify_node(item, source, f"{path}[{index}]", doc_label) for index, item in enumerate(node)]
    if isinstance(node, tuple):
        return tuple(_verify_node(item, source, f"{path}[{index}]", doc_label) for index, item in enumerate(node))
    return node


def verify_extracted_document(doc: ExtractedDocument, original_text: str) -> ExtractedDocument:
    """Set ``verified`` on every QuotedFact by checking its quote against the source.

    Verification is recomputed from scratch on every call; incoming ``verified``
    values are ignored. Facts are never removed. FAILED documents carry no
    facts and are returned unchanged.
    """

    if doc.extraction_status is ExtractionStatus.FAILED:
        return doc
    source = normalize_for_match(original_text)
    verified: ExtractedDocument = _verify_node(doc, source, "$", doc.doc_id or "<unknown>")
    return verified
