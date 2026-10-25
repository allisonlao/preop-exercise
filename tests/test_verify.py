"""Tests for Stage 3 deterministic quote verification (verify.py)."""

from __future__ import annotations

import logging

import pytest

from llm_fixtures import CONSENT_DOC_ID, NURSING_DOC_ID, document_text, raw_response
from schemas import (
    ConsentDetails,
    ConsentState,
    DocMetadata,
    DocumentType,
    ExtractedDocument,
    ExtractionStatus,
    MedicationMention,
    MedStatus,
    QuotedFact,
)
from verify import QuoteCheck, check_quote, iter_quoted_facts, normalize_for_match, verify_extracted_document

SOURCE = """\
INFORMED CONSENT FOR SURGICAL PROCEDURE
Procedure: laparoscopic cholecystectomy

Patient signature on file - signed in clinic 03/13/2026 12:06.
ADDENDUM: Patient   revoked consent
  by phone on 03/14/2026.
MEDICATIONS:
  - Eliquis 5 mg PO BID
"""


def consent_doc(*states: QuotedFact[ConsentState]) -> ExtractedDocument:
    return ExtractedDocument(
        doc_metadata=DocMetadata(
            document_type=QuotedFact(value=DocumentType.SURGICAL_CONSENT, quote="INFORMED CONSENT FOR SURGICAL PROCEDURE")
        ),
        patient_identifiers=[],
        hnp_details=None,
        consent_details=ConsentDetails(signature_states=list(states), procedure=None),
        vitals_mentions=[],
        medication_mentions=[],
        anticoagulation_plans=[],
        lab_mentions=[],
        date_risk_mentions=[],
        doc_id="doc-1",
    )


def single_state(doc: ExtractedDocument) -> QuotedFact[ConsentState]:
    assert doc.consent_details is not None
    (state,) = doc.consent_details.signature_states
    return state


# =============================================================================
# Normalization and single-quote checks
# =============================================================================


def test_normalize_collapses_all_whitespace_and_applies_nfkc() -> None:
    assert normalize_for_match("  a\t\tb\n\n c\u00a0d  ") == "a b c d"
    assert normalize_for_match("ＢＰ １８９") == "BP 189"  # fullwidth -> ASCII
    assert normalize_for_match("wait…") == "wait..."


@pytest.mark.parametrize(
    ("quote", "expected"),
    [
        ("Procedure: laparoscopic cholecystectomy", QuoteCheck.MATCH),
        ("Patient revoked consent by phone", QuoteCheck.MATCH),
        ("", QuoteCheck.EMPTY),
        ("   \n ", QuoteCheck.EMPTY),
        ("BID", QuoteCheck.TOO_SHORT),
        ("Patient signature ... 03/13/2026", QuoteCheck.ELLIPSIS),
        ("Patient signature on file…", QuoteCheck.ELLIPSIS),
        ("Patient declined consent", QuoteCheck.NOT_FOUND),
        ("patient signature on file", QuoteCheck.NOT_FOUND),  # case is part of the quote
    ],
)
def test_check_quote(quote: str, expected: QuoteCheck) -> None:
    assert check_quote(quote, normalize_for_match(SOURCE)) is expected


def test_four_character_quote_is_long_enough() -> None:
    assert check_quote("BID\n", normalize_for_match(SOURCE)) is QuoteCheck.TOO_SHORT
    assert check_quote("5 mg", normalize_for_match(SOURCE)) is QuoteCheck.MATCH


# =============================================================================
# Document verification
# =============================================================================


def test_valid_quote_sets_verified_true() -> None:
    doc = consent_doc(QuotedFact(value=ConsentState.SIGNED, quote="Patient signature on file - signed in clinic 03/13/2026"))
    result = verify_extracted_document(doc, SOURCE)
    assert single_state(result).verified is True
    assert result.doc_metadata.document_type.verified is True


def test_quote_with_extra_or_different_whitespace_still_matches() -> None:
    # Source has "Patient   revoked consent\n  by phone"; quote has single spaces / tabs.
    for quote in ("Patient revoked consent by phone on 03/14/2026", "Patient\trevoked  consent\n by phone"):
        doc = consent_doc(QuotedFact(value=ConsentState.REVOKED, quote=quote))
        assert single_state(verify_extracted_document(doc, SOURCE)).verified is True


@pytest.mark.parametrize(
    "quote",
    [
        "Patient revoked consent in writing",  # hallucinated
        "Patient revoked consent ... 03/14/2026",  # elided
        "Patient revoked consent…",  # unicode ellipsis
        "",  # missing
    ],
)
def test_bad_quote_is_unverified_but_fact_is_preserved(quote: str, caplog: pytest.LogCaptureFixture) -> None:
    doc = consent_doc(QuotedFact(value=ConsentState.REVOKED, quote=quote))
    with caplog.at_level(logging.WARNING, logger="verify"):
        result = verify_extracted_document(doc, SOURCE)
    state = single_state(result)
    assert state.verified is False
    assert state.value is ConsentState.REVOKED
    assert state.quote == quote
    assert "unverified quote at $.consent_details.signature_states[0]" in caplog.text


def test_verification_never_removes_facts_and_flags_each_independently() -> None:
    doc = consent_doc(
        QuotedFact(value=ConsentState.SIGNED, quote="signed in clinic 03/13/2026 12:06"),
        QuotedFact(value=ConsentState.REVOKED, quote="Consent revoked by patient"),
    ).model_copy(
        update={
            "medication_mentions": [
                QuotedFact(value=MedicationMention(drug_name_raw="Eliquis", status=MedStatus.CURRENT), quote="Eliquis 5 mg PO BID"),
            ]
        }
    )
    result = verify_extracted_document(doc, SOURCE)
    assert len(list(iter_quoted_facts(result))) == len(list(iter_quoted_facts(doc))) == 4
    assert [fact.verified for _, fact in iter_quoted_facts(result)] == [True, True, False, True]
    assert [fact.value for _, fact in iter_quoted_facts(result)] == [fact.value for _, fact in iter_quoted_facts(doc)]


def test_incoming_verified_flag_is_recomputed_not_trusted() -> None:
    forged = QuotedFact(value=ConsentState.SIGNED, quote="Patient signed electronically").model_copy(update={"verified": True})
    result = verify_extracted_document(consent_doc(forged), SOURCE)
    assert single_state(result).verified is False


def test_verified_cannot_be_loaded_from_json() -> None:
    fact = QuotedFact[ConsentState].model_validate({"value": "SIGNED", "quote": "x", "verified": True})
    assert fact.verified is False


def test_input_document_is_not_mutated() -> None:
    doc = consent_doc(QuotedFact(value=ConsentState.SIGNED, quote="Patient signature on file"))
    verify_extracted_document(doc, SOURCE)
    assert single_state(doc).verified is False


def test_failed_document_passes_through_unchanged(caplog: pytest.LogCaptureFixture) -> None:
    failed = ExtractedDocument.failed("doc-9", "timeout")
    with caplog.at_level(logging.WARNING, logger="verify"):
        assert verify_extracted_document(failed, SOURCE) is failed
    assert caplog.text == ""


# =============================================================================
# Hardcoded model responses against real sample documents
# =============================================================================


@pytest.mark.parametrize("doc_id", [NURSING_DOC_ID, CONSENT_DOC_ID])
def test_cached_sample_responses_verify_against_real_documents(doc_id: str) -> None:
    doc = ExtractedDocument.model_validate_json(raw_response(doc_id))
    result = verify_extracted_document(doc, document_text(doc_id))
    facts = list(iter_quoted_facts(result))
    assert facts
    assert all(fact.verified for _, fact in facts), [path for path, fact in facts if not fact.verified]


def test_quote_against_the_wrong_document_fails() -> None:
    doc = ExtractedDocument.model_validate_json(raw_response(CONSENT_DOC_ID))
    result = verify_extracted_document(doc, document_text(NURSING_DOC_ID))
    assert not any(fact.verified for _, fact in iter_quoted_facts(result))
    assert result.extraction_status is ExtractionStatus.SUCCESS
