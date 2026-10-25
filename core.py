"""Pre-op triage pipeline entry point and submission schema.

``triage_submission`` runs the staged pipeline for one submission:

1. ``preprocess.preprocess_submission``: structured JSON -> typed facts (Stage 2).
2. For every document: ``extract.extract_document`` (cached LLM call, Stage 3)
   then ``verify.verify_extracted_document`` (deterministic quote check).
3. ``reconcile.reconcile_patient_facts``: merge into engine input (Stage 4).
4. ``engine.evaluate``: the pure policy rules (Stage 1).
5. ``packet.build_output``: the sorted, reviewer-facing evidence packet.

Without an API key the extractor serves cached responses from
``data/llm_cache.json``; any document with no cached response is marked
FAILED and the case fails closed (MISSING_REQUIRED_DATA / EXTRACTION_FAILED).
Nothing in the output depends on the clock or on run order.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from engine import evaluate
from extract import DEFAULT_MODEL, ExtractionCache, extract_document
from packet import EvidenceItem, PassedCheck, TriageIssue, TriageOutput, build_output
from preprocess import preprocess_submission
from reconcile import reconcile_patient_facts
from schemas import ExtractedDocument
from verify import verify_extracted_document

__all__ = [
    "DEFAULT_MODEL",
    "EvidenceItem",
    "PassedCheck",
    "PatientSubmission",
    "TriageIssue",
    "TriageOutput",
    "default_client",
    "extract_documents",
    "triage_submission",
]

Decision = Literal["READY", "NEEDS_FOLLOW_UP", "NOT_CLEARED"]
ProcedureRisk = Literal["LOW", "MODERATE", "HIGH"]
IssueCategory = Literal[
    "REQUIRED_DOCUMENTATION",
    "REQUIRED_TESTING",
    "ANTICOAGULATION_MANAGEMENT",
    "ACUTE_SAFETY_EXCLUSION",
    "MISSING_REQUIRED_DATA",
]

# -------------------------
# Schemas
# -------------------------

class PatientName(BaseModel):

    given: str | None = None
    family: str | None = None

class PatientInfo(BaseModel):

    id: str | None = None
    mrn: str | None = None
    name: PatientName | None = None
    dob: str | None = None
    sex: str | None = None

class ProcedureInfo(BaseModel):

    case_id: str | None = None
    procedure_type: str | None = None
    procedure_risk: ProcedureRisk | None = None
    procedure_date: str | None = None
    is_elective: bool | None = None
    location: str | None = None

class BloodPressureVital(BaseModel):

    type: str | None = None
    systolic: float | int | None = None
    diastolic: float | int | None = None
    date: str | None = None
    source: str | None = None

class TemperatureVital(BaseModel):

    type: str | None = None
    value_f: float | int | None = None
    date: str | None = None
    source: str | None = None

class GenericVital(BaseModel):

    type: str | None = None
    date: str | None = None
    source: str | None = None

Vital = BloodPressureVital | TemperatureVital | GenericVital

class LabResult(BaseModel):

    id: str | None = None
    code: str | None = None
    display: str | None = None
    effective_at: str | None = None
    status: str | None = None
    source: str | None = None

class Medication(BaseModel):

    name: str | None = None
    active: bool | None = None

class Condition(BaseModel):

    name: str | None = None
    active: bool | None = None

class Document(BaseModel):

    doc_id: str | None = None
    type: str | None = None
    date: str | None = None
    author: str | None = None
    text: str | None = None

class SubmissionMetadata(BaseModel):

    submission_received_at: str | None = None
    source_system: str | None = None

class PatientSubmission(BaseModel):
    """Single submission package shape from the take-home prompt."""

    patient: PatientInfo | None = None
    procedure: ProcedureInfo | None = None
    vitals: list[Vital] = Field(default_factory=list)
    labs: list[LabResult] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    metadata: SubmissionMetadata | None = None


# -------------------------
# Pipeline
# -------------------------


def default_client() -> OpenAI | None:
    """An OpenAI client when ``OPENAI_API_KEY`` is set, else None (cache-only mode)."""

    return OpenAI() if os.environ.get("OPENAI_API_KEY") else None


def extract_documents(
    submission: Mapping[str, Any],
    client: OpenAI | None,
    *,
    model: str,
    cache: ExtractionCache | None = None,
) -> list[ExtractedDocument]:
    """Extract and quote-verify every document, one LLM call (or cache hit) each.

    Documents without a ``doc_id`` are skipped here; Stage 2 reports them as
    malformed and reconciliation fails closed on them. A document with no text
    is returned as FAILED.
    """

    raw_documents = submission.get("documents")
    extracted: list[ExtractedDocument] = []
    for raw in raw_documents if isinstance(raw_documents, list) else []:
        if not isinstance(raw, Mapping):
            continue
        doc_id = raw.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id.strip():
            continue
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            extracted.append(ExtractedDocument.failed(doc_id, "document has no text"))
            continue
        document = extract_document(text, doc_id, client, model=model, cache=cache)
        extracted.append(verify_extracted_document(document, text))
    return extracted


def triage_submission(
    submission: Mapping[str, Any] | PatientSubmission,
    *,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
    cache: ExtractionCache | None = None,
) -> TriageOutput:
    """Run the full pipeline for one submission. Never calls the LLM for policy logic.

    ``client`` defaults to ``default_client()``. The raw mapping is used as-is
    (no lossy re-validation), so malformed fields reach Stage 2 and are reported.
    """

    payload: Mapping[str, Any] = (
        submission.model_dump() if isinstance(submission, PatientSubmission) else submission
    )
    structured = preprocess_submission(payload)
    documents = extract_documents(payload, client if client is not None else default_client(), model=model, cache=cache)
    facts = reconcile_patient_facts(structured, documents)
    return build_output(facts, evaluate(facts))
