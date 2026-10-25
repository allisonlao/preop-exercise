"""Stage 3 LLM output schemas (OpenAI Structured Outputs).

These models describe what the extraction model returns for ONE unstructured
document. They are deliberately separate from ``models.py``: values here are raw
strings exactly as written in the document (no brand-to-generic mapping, no date
or unit parsing). Normalization and reconciliation happen downstream in code.

Structured Outputs constraints honored here:
  * every field the model sees is required (optionality is ``T | None``, never a
    default), and objects forbid extra keys;
  * no string ``format`` / length keywords (dates and numbers stay raw strings).

Trust fields are hidden from the JSON schema with ``SkipJsonSchema`` so the model
cannot set them: ``QuotedFact.verified`` is owned by ``verify.py`` and
``ExtractedDocument.extraction_status`` / ``doc_id`` / ``failure_reason`` by
``extract.py``.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Final, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, field_validator
from pydantic.json_schema import SkipJsonSchema

T = TypeVar("T")


class _Strict(BaseModel):
    # Attribute docstrings become JSON-schema descriptions the model can read.
    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)


# =============================================================================
# Enums (each has an UNCLEAR fallback so the model never has to guess)
# =============================================================================


class ConsentState(StrEnum):
    """Patient-signature state. SIGNED only when the patient's signature is recorded
    as done. A signature that is promised or awaited ("to be signed", "signature:
    pending", "will return the signed consent") is PENDING. A blank or explicitly
    absent signature is UNSIGNED."""

    SIGNED = "SIGNED"
    UNSIGNED = "UNSIGNED"
    REVOKED = "REVOKED"
    VERBAL_ONLY = "VERBAL_ONLY"
    PENDING = "PENDING"
    UNCLEAR = "UNCLEAR"


class HnPState(StrEnum):
    """H&P completion. COMPLETE only when the note is finished and signed. A note
    that is still to be finished ("to be completed", draft, unsigned) is INCOMPLETE."""

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    UNCLEAR = "UNCLEAR"


class PreopAction(StrEnum):
    """What the plan tells the patient to do with this drug before surgery.
    HOLD: stop or hold it ("Hold Xarelto for the 2 days before surgery").
    CONTINUE: keep taking it through surgery ("continue without interruption").
    BRIDGE: give a bridging agent in its place ("Bridge with enoxaparin ... starting 3 days prior").
    DEFERRED: the decision is postponed or handed to someone else ("to be determined by
    the surgical team", "per cardiology", "will discuss").
    NOT_ADDRESSED: the plan says nothing about pre-op management.
    UNCLEAR: ambiguous or contradictory wording."""

    HOLD = "HOLD"
    CONTINUE = "CONTINUE"
    BRIDGE = "BRIDGE"
    DEFERRED = "DEFERRED"
    NOT_ADDRESSED = "NOT_ADDRESSED"
    UNCLEAR = "UNCLEAR"


class PostopAction(StrEnum):
    """What the plan tells the patient to do with this drug after surgery.
    RESUME: restart it ("Resume Eliquis 48 hours post-op", "Resume warfarin after surgery").
    CONTINUE_HOLD: keep it held after surgery until a stated point ("remain off until POD 3").
    DISCONTINUE: do not restart it.
    DEFERRED: the decision is postponed or handed to someone else instead of being made
    here ("will be addressed after surgery", "resume per cardiology", "to be determined").
    A restart that states its own timing or condition is RESUME even if it also names
    who oversees it ("Resume Xarelto 24-48 hours after surgery, per surgical team").
    NOT_ADDRESSED: the plan says nothing about post-op management.
    UNCLEAR: ambiguous or contradictory wording."""

    RESUME = "RESUME"
    CONTINUE_HOLD = "CONTINUE_HOLD"
    DISCONTINUE = "DISCONTINUE"
    DEFERRED = "DEFERRED"
    NOT_ADDRESSED = "NOT_ADDRESSED"
    UNCLEAR = "UNCLEAR"


# Only these actions count as an instruction. DEFERRED, NOT_ADDRESSED and UNCLEAR
# count as a missing instruction (fail closed).
PREOP_INSTRUCTIONS: Final = frozenset({PreopAction.HOLD, PreopAction.CONTINUE, PreopAction.BRIDGE})
POSTOP_INSTRUCTIONS: Final = frozenset({PostopAction.RESUME, PostopAction.CONTINUE_HOLD, PostopAction.DISCONTINUE})


class MedStatus(StrEnum):
    CURRENT = "CURRENT"
    STOPPED = "STOPPED"
    HELD = "HELD"
    UNCLEAR = "UNCLEAR"


class ExtractionStatus(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class DocumentType(StrEnum):
    """Classification by content, not by the (often inconsistent) title."""

    HISTORY_AND_PHYSICAL = "HISTORY_AND_PHYSICAL"
    SURGICAL_CONSENT = "SURGICAL_CONSENT"
    NURSING_ASSESSMENT = "NURSING_ASSESSMENT"
    ANESTHESIA_ASSESSMENT = "ANESTHESIA_ASSESSMENT"
    MEDICATION_MANAGEMENT = "MEDICATION_MANAGEMENT"
    CONSULT_NOTE = "CONSULT_NOTE"
    OTHER = "OTHER"
    UNCLEAR = "UNCLEAR"


class IdentifierKind(StrEnum):
    NAME = "NAME"
    MRN = "MRN"
    DOB = "DOB"


class VitalType(StrEnum):
    BLOOD_PRESSURE = "BLOOD_PRESSURE"
    TEMPERATURE = "TEMPERATURE"


class DateRiskKind(StrEnum):
    PROCEDURE_DATE = "PROCEDURE_DATE"
    RISK_LEVEL = "RISK_LEVEL"


# =============================================================================
# QuotedFact wrapper
# =============================================================================


class QuotedFact(_Strict, Generic[T]):
    """One extracted fact plus the verbatim source span that supports it.

    ``verified`` is never produced by the model; it is always False until
    ``verify.verify_extracted_document`` checks ``quote`` against the source.
    Unverified facts are kept (fail closed), never dropped.
    """

    value: T
    quote: str
    verified: SkipJsonSchema[bool] = False

    @field_validator("verified", mode="after")
    @classmethod
    def _never_trust_input(cls, _: bool) -> bool:
        # Verification status cannot be loaded from JSON (cache, model output);
        # only verify.py sets it, via model_copy, after checking the source.
        return False


# =============================================================================
# Fact payloads (the ``value`` of each QuotedFact)
# =============================================================================


class PatientIdentifier(_Strict):
    kind: IdentifierKind
    raw_value: str


class VitalMention(_Strict):
    vital_type: VitalType
    raw_value: str
    """As written, e.g. ``"189/112"`` or ``"38.6"``."""
    raw_unit: str | None
    """As written, e.g. ``"mmHg"``, ``"°C"``, ``"C"``; null when absent."""
    time_raw: str | None
    """Time/date of the measurement as written, e.g. ``"08:06"``; null when absent."""


class MedicationMention(_Strict):
    drug_name_raw: str
    """Exactly as written (``"Eliquis"`` stays ``"Eliquis"``)."""
    status: MedStatus


class AnticoagulationPlanMention(_Strict):
    drug_name_raw: str
    preop_action: QuotedFact[PreopAction]
    """The pre-op action; the quote is the sentence that states it (empty when NOT_ADDRESSED)."""
    preop_timing_raw: str | None
    """The timing phrase of the pre-op action (``"2 days before"``, ``"last dose 03/01"``);
    null when none is given."""
    postop_action: QuotedFact[PostopAction]
    """The post-op action; the quote is the sentence that states it (empty when NOT_ADDRESSED)."""
    postop_timing_raw: str | None
    """The timing phrase of the post-op action (``"48 hours post-op"``); null when none is given."""


class LabMention(_Strict):
    test_name_raw: str
    date_raw: str | None
    result_raw: str | None


class DateRiskMention(_Strict):
    kind: DateRiskKind
    raw_text: str


# =============================================================================
# Sections
# =============================================================================


class DocMetadata(_Strict):
    document_type: QuotedFact[DocumentType]


class HnPDetails(_Strict):
    completion: QuotedFact[HnPState]
    date_of_service: QuotedFact[str] | None


class ConsentDetails(_Strict):
    signature_states: list[QuotedFact[ConsentState]]
    """Every patient-signature state statement (e.g. PENDING then SIGNED)."""
    procedure: QuotedFact[str] | None


class ExtractedDocument(_Strict):
    """Raw, quote-backed facts extracted from one unstructured document."""

    doc_metadata: DocMetadata
    patient_identifiers: list[QuotedFact[PatientIdentifier]]
    hnp_details: HnPDetails | None
    consent_details: ConsentDetails | None
    vitals_mentions: list[QuotedFact[VitalMention]]
    medication_mentions: list[QuotedFact[MedicationMention]]
    anticoagulation_plans: list[QuotedFact[AnticoagulationPlanMention]]
    lab_mentions: list[QuotedFact[LabMention]]
    date_risk_mentions: list[QuotedFact[DateRiskMention]]

    extraction_status: SkipJsonSchema[ExtractionStatus] = ExtractionStatus.SUCCESS
    doc_id: SkipJsonSchema[str | None] = None
    failure_reason: SkipJsonSchema[str | None] = None

    @classmethod
    def failed(cls, doc_id: str, reason: str) -> ExtractedDocument:
        """An empty, FAILED document. Downstream treats it as missing data."""

        return cls(
            doc_metadata=DocMetadata(document_type=QuotedFact(value=DocumentType.UNCLEAR, quote="")),
            patient_identifiers=[],
            hnp_details=None,
            consent_details=None,
            vitals_mentions=[],
            medication_mentions=[],
            anticoagulation_plans=[],
            lab_mentions=[],
            date_risk_mentions=[],
            extraction_status=ExtractionStatus.FAILED,
            doc_id=doc_id,
            failure_reason=reason,
        )


def _schema_fingerprint() -> str:
    canonical = json.dumps(ExtractedDocument.model_json_schema(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# Embeds a fingerprint of the JSON schema, so any schema change gets a new name
# and therefore new cache keys (stale cached JSON is never replayed).
SCHEMA_NAME: Final = f"extracted_document_{_schema_fingerprint()}"
