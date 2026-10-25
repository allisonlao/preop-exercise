"""Stage 4 reconciliation: structured JSON facts + verified document facts -> engine input.

Pure and deterministic: no LLM, no I/O, no clock. Pipeline:

1. ``filter_unverified`` turns every ``ExtractedDocument`` into a
   ``TrustedDocument`` *before* any merging. A ``QuotedFact`` whose quote did
   not verify is treated as if the document never said it (``None``) or as
   ``UNCLEAR``/unknown, never as its claimed value. A hallucinated "consent
   signed" therefore cannot satisfy a rule. Where silently dropping the fact
   could itself cause a false READY (an unverified newer vital, an unverified
   anticoagulant mention), it becomes an *unknown* that fails closed.
2. One helper per fact family merges the two sources:
   ``resolve_procedure``, ``resolve_history_and_physicals``,
   ``resolve_consents``, ``merge_vitals``, ``resolve_medications``,
   ``resolve_anticoagulation_plans``.
3. Anything whose content is unknown (failed extraction, unextracted document,
   malformed vital/medication record, unreadable or unverifiable recent vital)
   is collected in ``ReconciledPatientFacts.unreadable_sources``; the engine
   turns it into a MISSING_REQUIRED_DATA issue.

Time convention: clock times written in notes ("08:06") are read on the same
clock as the structured vitals (UTC), the convention the intake data uses.
A note vital with no clock time gets ``DAY`` precision on its date; the engine
then evaluates every reading that could be the most recent (worst case).
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from operator import attrgetter
from typing import Final, Literal, TypeVar

from models import (
    AnticoagulationPlan,
    BloodPressureReading,
    DrugClass,
    Evidence,
    EvidenceSource,
    HistoryAndPhysical,
    MedicationStatus,
    ProcedureFacts,
    ReconciledPatientFacts,
    ResolvedMedication,
    SurgicalConsent,
    TemperatureReading,
    TemperatureUnit,
    TimePrecision,
)
from preprocess import DocumentHeader, ExclusionReason, StructuredMedication, StructuredSubmission, classify_drug
from schemas import (
    AnticoagulationPlanMention,
    ConsentState,
    DocumentType,
    ExtractedDocument,
    ExtractionStatus,
    HnPState,
    MedStatus,
    QuotedFact,
    VitalMention,
    VitalType,
)

T = TypeVar("T")


class Unknown(Enum):
    """A fact the document asserted, but whose quote failed verification."""

    TOKEN = "UNKNOWN"


UNKNOWN: Final = Unknown.TOKEN


# =============================================================================
# Step 1: strict filtering of LLM output
# =============================================================================


@dataclass(frozen=True)
class TrustedHnP:
    completion: HnPState  # UNCLEAR when the completion quote did not verify
    completion_quote: str | None
    # None = the document states no date of service; UNKNOWN = it does, unverified.
    date_of_service: str | None | Literal[Unknown.TOKEN]
    date_quote: str | None


@dataclass(frozen=True)
class TrustedConsent:
    states: tuple[ConsentState, ...]  # unverified states become UNCLEAR
    procedure: str | None  # None when absent or unverified
    quotes: tuple[str, ...]


@dataclass(frozen=True)
class TrustedVital:
    mention: VitalMention
    quote: str
    verified: bool  # unverified vitals are kept only to decide if they could hide a newer reading


@dataclass(frozen=True)
class TrustedMedication:
    drug_name_raw: str
    status: MedStatus  # UNCLEAR when the quote did not verify
    quote: str
    verified: bool


@dataclass(frozen=True)
class TrustedPlan:
    index: int  # position in the extracted list; keeps plan ids stable
    plan: AnticoagulationPlanMention
    quote: str


@dataclass(frozen=True)
class TrustedDocument:
    """One document's facts after strict filtering. Only this is merged."""

    doc_id: str
    title: str | None
    documented_on: date | None
    extraction_failed: bool
    failure_reason: str | None
    document_type: DocumentType  # UNCLEAR when the classification quote did not verify
    type_quote: str | None
    hnp: TrustedHnP | None
    consent: TrustedConsent | None
    vitals: tuple[TrustedVital, ...]
    medications: tuple[TrustedMedication, ...]
    plans: tuple[TrustedPlan, ...]  # unverified plans are dropped (treated as None)


def trusted_value(fact: QuotedFact[T] | None) -> T | None:
    """The fact's value if its quote verified, else None."""

    return fact.value if fact is not None and fact.verified else None


def filter_unverified(doc: ExtractedDocument, header: DocumentHeader | None, fallback_id: str) -> TrustedDocument:
    """Map one extracted document to its verified-only view."""

    doc_id = doc.doc_id or (header.doc_id if header else fallback_id)
    type_fact = doc.doc_metadata.document_type

    hnp: TrustedHnP | None = None
    if doc.hnp_details is not None:
        details = doc.hnp_details
        date_fact = details.date_of_service
        hnp = TrustedHnP(
            completion=trusted_value(details.completion) or HnPState.UNCLEAR,
            completion_quote=details.completion.quote if details.completion.verified else None,
            date_of_service=None if date_fact is None else (date_fact.value if date_fact.verified else UNKNOWN),
            date_quote=date_fact.quote if date_fact is not None and date_fact.verified else None,
        )

    consent: TrustedConsent | None = None
    if doc.consent_details is not None:
        details_c = doc.consent_details
        consent = TrustedConsent(
            states=tuple(trusted_value(state) or ConsentState.UNCLEAR for state in details_c.signature_states),
            procedure=trusted_value(details_c.procedure),
            quotes=tuple(
                fact.quote
                for fact in (*details_c.signature_states, details_c.procedure)
                if fact is not None and fact.verified
            ),
        )

    return TrustedDocument(
        doc_id=doc_id,
        title=header.title if header else None,
        documented_on=header.documented_on if header else None,
        extraction_failed=doc.extraction_status is ExtractionStatus.FAILED,
        failure_reason=doc.failure_reason,
        document_type=trusted_value(type_fact) or DocumentType.UNCLEAR,
        type_quote=type_fact.quote if type_fact.verified else None,
        hnp=hnp,
        consent=consent,
        vitals=tuple(TrustedVital(v.value, v.quote, v.verified) for v in doc.vitals_mentions),
        medications=tuple(
            TrustedMedication(
                drug_name_raw=m.value.drug_name_raw,
                status=m.value.status if m.verified else MedStatus.UNCLEAR,
                quote=m.quote,
                verified=m.verified,
            )
            for m in doc.medication_mentions
        ),
        plans=tuple(
            TrustedPlan(index, p.value, p.quote) for index, p in enumerate(doc.anticoagulation_plans) if p.verified
        ),
    )


# =============================================================================
# Shared helpers: evidence and raw date/time parsing
# =============================================================================


def doc_evidence(doc: TrustedDocument, quote: str | None, detail: str | None = None) -> Evidence:
    return Evidence(
        source=EvidenceSource.DOCUMENT,
        locator=f"documents[{doc.doc_id}]",
        label=doc.title,
        recorded_on=doc.documented_on,
        quote=quote or None,
        detail=detail,
    )


_MONTHS: Final = {name: index for index, name in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), start=1
)}
# Digit lookarounds instead of \b so "2026-03-15T08:00" splits into date and time.
_ISO_DATE: Final = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")
_US_DATE: Final = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])")
_US_DATE_NO_YEAR: Final = re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})(?![\d/])")
_NAMED_DATE: Final = re.compile(r"\b([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b")
_CLOCK: Final = re.compile(r"(?<![\d:])(\d{1,2}):(\d{2})(?::\d{2})?(?:\s*([AaPp])\.?\s*[Mm]\.?)?")


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_raw_date(text: str, reference: date | None = None) -> date | None:
    """First calendar date written in ``text``; None if there is none or it is invalid.

    Accepts ``2026-03-15``, ``03/15/2026``, ``3/15/26``, ``Mar 15, 2026``, and
    ``03/15`` (year taken from ``reference``, the document's own date).
    """

    if match := _ISO_DATE.search(text):
        return _safe_date(int(match[1]), int(match[2]), int(match[3]))
    if match := _US_DATE.search(text):
        year = int(match[3])
        return _safe_date(year + 2000 if year < 100 else year, int(match[1]), int(match[2]))
    if (match := _NAMED_DATE.search(text)) and match[1].lower() in _MONTHS:
        return _safe_date(int(match[3]), _MONTHS[match[1].lower()], int(match[2]))
    if reference is not None and (match := _US_DATE_NO_YEAR.search(text)):
        return _safe_date(reference.year, int(match[1]), int(match[2]))
    return None


def parse_raw_clock(text: str) -> time | None:
    """First clock time written in ``text`` (``08:06``, ``2:15 PM``); None if absent or invalid."""

    match = _CLOCK.search(text)
    if match is None:
        return None
    hour, minute, meridiem = int(match[1]), int(match[2]), (match[3] or "").lower()
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if meridiem == "p" else 0)
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def vital_timestamp(time_raw: str | None, documented_on: date | None) -> tuple[datetime, TimePrecision] | None:
    """When a note vital was taken: its own date (or the document's) plus clock time if written."""

    day = parse_raw_date(time_raw, documented_on) if time_raw else None
    day = day or documented_on
    if day is None:
        return None
    clock = parse_raw_clock(time_raw) if time_raw else None
    if clock is None:
        return datetime.combine(day, time.min, tzinfo=UTC), TimePrecision.DAY
    return datetime.combine(day, clock, tzinfo=UTC), TimePrecision.MINUTE


# =============================================================================
# Step 2.1: procedure date (structured only)
# =============================================================================


def resolve_procedure(structured: StructuredSubmission) -> ProcedureFacts:
    """Strict precedence for the structured procedure date and risk level.

    ``date_risk_mentions`` from documents are intentionally never consulted: a
    null JSON date stays null (R0 reports it) rather than being filled from a
    "target" or "tentative" date in a note.
    """

    return structured.demographics.procedure


# =============================================================================
# Step 2.2: document resolution (H&P, consent)
# =============================================================================


def _hnp_date(doc: TrustedDocument, hnp: TrustedHnP | None) -> date | None:
    """Date of service; falls back to the document date only when no date is written.

    An unverified or unparseable date of service gives None (HP_DATE_UNKNOWN), never the
    document date, which could be later than the real date of service.
    """

    if hnp is None or hnp.date_of_service is None:
        return doc.documented_on
    if hnp.date_of_service is UNKNOWN:
        return None
    return parse_raw_date(hnp.date_of_service, doc.documented_on)


def resolve_history_and_physicals(docs: Sequence[TrustedDocument]) -> tuple[HistoryAndPhysical, ...]:
    """Every document verified as an H&P, most recent date of service first.

    All H&Ps are passed on: the engine accepts any completed H&P within the window
    and, when none qualifies, reports on the most recent one.
    """

    hps: list[HistoryAndPhysical] = []
    for doc in docs:
        if doc.document_type is not DocumentType.HISTORY_AND_PHYSICAL:
            continue
        hnp = doc.hnp
        completion = hnp.completion if hnp is not None else HnPState.UNCLEAR
        performed_on = _hnp_date(doc, hnp)
        evidence = [doc_evidence(doc, doc.type_quote, f"H&P state: {completion}")]
        if hnp is not None and hnp.completion_quote and hnp.completion_quote != doc.type_quote:
            evidence.append(doc_evidence(doc, hnp.completion_quote, f"H&P state: {completion}"))
        if hnp is not None and hnp.date_quote:
            evidence.append(doc_evidence(doc, hnp.date_quote, f"date of service: {performed_on}"))
        hps.append(
            HistoryAndPhysical(
                document_id=doc.doc_id,
                performed_on=performed_on,
                completed=completion is HnPState.COMPLETE,
                evidence=tuple(evidence),
            )
        )
    return tuple(sorted(hps, key=lambda hp: hp.performed_on or date.min, reverse=True))


_PROCEDURE_FILLER: Final = frozenset({"elective", "planned", "procedure", "surgery", "surgical", "the", "of"})


def _procedure_tokens(text: str) -> tuple[str, ...]:
    return tuple(t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _PROCEDURE_FILLER)


def _contains_run(longer: tuple[str, ...], shorter: tuple[str, ...]) -> bool:
    size = len(shorter)
    return any(longer[i : i + size] == shorter for i in range(len(longer) - size + 1))


def procedure_matches(consented: str | None, scheduled: str | None) -> bool | None:
    """True on an exact match (ignoring filler words), False when unrelated.

    None when either side is unknown or one is only part of the other
    ("hernia repair" vs "inguinal hernia repair"): a human must confirm.
    """

    if consented is None or scheduled is None:
        return None
    a, b = _procedure_tokens(consented), _procedure_tokens(scheduled)
    if not a or not b:
        return None
    if a == b:
        return True
    if _contains_run(a, b) or _contains_run(b, a):
        return None
    return False


def _consent_from(doc: TrustedDocument, scheduled: str | None) -> SurgicalConsent:
    states = doc.consent.states if doc.consent else ()
    consented = doc.consent.procedure if doc.consent else None
    # Any unclear (including unverified) signature statement blocks "signed".
    signed = ConsentState.SIGNED in states and ConsentState.UNCLEAR not in states
    detail = "signature states: " + (", ".join(states) if states else "none verified")
    evidence = [doc_evidence(doc, doc.type_quote, detail)]
    evidence += [doc_evidence(doc, quote) for quote in (doc.consent.quotes if doc.consent else ())]
    return SurgicalConsent(
        document_id=doc.doc_id,
        documented_on=doc.documented_on,
        signed=signed,
        revoked=ConsentState.REVOKED in states,
        procedure_matches=procedure_matches(consented, scheduled),
        evidence=tuple(evidence),
    )


def _consent_is_valid(consent: SurgicalConsent) -> bool:
    return consent.signed and not consent.revoked and consent.procedure_matches is True


def resolve_consents(
    docs: Sequence[TrustedDocument], scheduled_procedure: str | None, notes: list[str]
) -> tuple[SurgicalConsent, ...]:
    """The most recently dated consent document only (zero or one consent).

    Older consents are superseded, so a later revocation or pending re-consent
    is never masked by an earlier signature. Same-date ties pick the worst
    consent. Documents not verified as consents are ignored, so an extraction
    whose quotes all failed yields "consent missing".
    """

    consents = [_consent_from(doc, scheduled_procedure) for doc in docs if doc.document_type is DocumentType.SURGICAL_CONSENT]
    if not consents:
        return ()
    newest = max(consents, key=lambda c: (c.documented_on or date.min, not _consent_is_valid(c)))
    for other in consents:
        if other is not newest:
            notes.append(f"consent {other.document_id} superseded by more recent consent {newest.document_id}")
    return (newest,)


# =============================================================================
# Step 2.3: vitals (chronological merge)
# =============================================================================

_BP_VALUE: Final = re.compile(r"(\d{2,3})\s*/\s*(\d{2,3})")
_NUMBER: Final = re.compile(r"\d{2,3}(?:\.\d+)?")


def _temperature_unit(mention: VitalMention, value: Decimal) -> TemperatureUnit | None:
    """Unit as written; if absent, inferred only from non-overlapping physiological ranges."""

    written = " ".join(filter(None, (mention.raw_unit, re.sub(r"[\d.\s/]+", " ", mention.raw_value))))
    letters = re.sub(r"[^a-z]", "", written.lower().replace("deg", ""))
    if letters in {"c", "celsius", "cel"}:
        return TemperatureUnit.CELSIUS
    if letters in {"f", "fahrenheit"}:
        return TemperatureUnit.FAHRENHEIT
    if letters:
        return None
    if Decimal(30) <= value <= Decimal(45):
        return TemperatureUnit.CELSIUS
    if Decimal(85) <= value <= Decimal(115):
        return TemperatureUnit.FAHRENHEIT
    return None


def _parse_note_vital(
    vital: TrustedVital, doc: TrustedDocument
) -> BloodPressureReading | TemperatureReading | str:
    """A typed reading, or a reason string when the mention cannot be read."""

    stamp = vital_timestamp(vital.mention.time_raw, doc.documented_on)
    if stamp is None:
        return "no date: neither the vital nor its document is dated"
    observed_at, precision = stamp
    detail = f"note vital ({precision} precision)" + ("" if vital.verified else "; quote unverified")
    evidence = (doc_evidence(doc, vital.quote, detail),)
    if vital.mention.vital_type is VitalType.BLOOD_PRESSURE:
        match = _BP_VALUE.search(vital.mention.raw_value)
        if match is None:
            return f"unreadable blood pressure {vital.mention.raw_value!r}"
        return BloodPressureReading(
            observed_at=observed_at, precision=precision,
            systolic=Decimal(match[1]), diastolic=Decimal(match[2]), evidence=evidence,
        )
    number = _NUMBER.search(vital.mention.raw_value)
    try:
        value = Decimal(number[0]) if number else None
    except InvalidOperation:
        value = None
    unit = _temperature_unit(vital.mention, value) if value is not None else None
    if value is None or unit is None:
        return f"unreadable temperature {vital.mention.raw_value!r} {vital.mention.raw_unit or ''}".rstrip()
    return TemperatureReading(observed_at=observed_at, precision=precision, value=value, unit=unit, evidence=evidence)


def _bounds(reading: BloodPressureReading | TemperatureReading) -> tuple[datetime, datetime]:
    if reading.precision is TimePrecision.MINUTE:
        return reading.observed_at, reading.observed_at
    return reading.observed_at, reading.observed_at + timedelta(days=1) - timedelta(microseconds=1)


def _could_be_newest(candidate: BloodPressureReading | TemperatureReading,
                     trusted: Sequence[BloodPressureReading | TemperatureReading]) -> bool:
    if not trusted:
        return True
    latest_start = max(_bounds(r)[0] for r in trusted)
    return _bounds(candidate)[1] >= latest_start


@dataclass
class _VitalsMerge:
    blood_pressures: list[BloodPressureReading] = field(default_factory=list)
    temperatures: list[TemperatureReading] = field(default_factory=list)
    unreadable: list[Evidence] = field(default_factory=list)


def merge_vitals(
    structured: StructuredSubmission, docs: Sequence[TrustedDocument], notes: list[str]
) -> tuple[tuple[BloodPressureReading, ...], tuple[TemperatureReading, ...], tuple[Evidence, ...]]:
    """Structured + verified note vitals, newest first, plus vitals that fail closed.

    * A verified note vital that cannot be parsed or dated is unreadable (it
      might be the newest reading).
    * An unverified note vital is ignored when older than the newest trusted
      reading of its type; if it could be the newest, it is unreadable.
    * Date-only note vitals keep DAY precision; when one shares a date with a
      timed reading, the tie is noted and the engine evaluates both (worst case).
    """

    merged = _VitalsMerge(list(structured.blood_pressures), list(structured.temperatures))
    unverified: list[tuple[TrustedDocument, TrustedVital, BloodPressureReading | TemperatureReading | str]] = []

    for doc in docs:
        for vital in doc.vitals:
            parsed = _parse_note_vital(vital, doc)
            if not vital.verified:
                unverified.append((doc, vital, parsed))
            elif isinstance(parsed, str):
                merged.unreadable.append(doc_evidence(doc, vital.quote, f"note vital not usable: {parsed}"))
            elif isinstance(parsed, BloodPressureReading):
                merged.blood_pressures.append(parsed)
            else:
                merged.temperatures.append(parsed)

    for doc, vital, parsed in unverified:
        if isinstance(parsed, str):
            merged.unreadable.append(doc_evidence(doc, vital.quote, f"unverified note vital, {parsed}"))
            continue
        same_type: Sequence[BloodPressureReading | TemperatureReading] = (
            merged.blood_pressures if isinstance(parsed, BloodPressureReading) else merged.temperatures
        )
        if _could_be_newest(parsed, same_type):
            merged.unreadable.append(
                doc_evidence(doc, vital.quote, "unverified note vital that could be the most recent reading")
            )
        else:
            notes.append(f"ignored unverified older vital in {doc.doc_id}: {vital.quote!r}")

    for readings in (merged.blood_pressures, merged.temperatures):
        _note_day_ties(readings, notes)

    # sorted() is stable with reverse=True, so equal timestamps keep source order.
    return (
        tuple(sorted(merged.blood_pressures, key=attrgetter("observed_at"), reverse=True)),
        tuple(sorted(merged.temperatures, key=attrgetter("observed_at"), reverse=True)),
        tuple(merged.unreadable),
    )


def _note_day_ties(readings: Sequence[BloodPressureReading | TemperatureReading], notes: list[str]) -> None:
    timed_days = {r.observed_at.date() for r in readings if r.precision is TimePrecision.MINUTE}
    for reading in readings:
        if reading.precision is TimePrecision.DAY and reading.observed_at.date() in timed_days:
            notes.append(
                f"unresolved time tie on {reading.observed_at.date()}: date-only note vital vs timed reading; "
                "engine evaluates both (worst case)"
            )


# =============================================================================
# Step 2.4: medications and anticoagulant status
# =============================================================================


class _Signal(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    UNKNOWN = "UNKNOWN"
    CONFLICT = "CONFLICT"


_TEXT_SIGNAL: Final = {
    MedStatus.CURRENT: _Signal.ACTIVE,
    MedStatus.HELD: _Signal.ACTIVE,  # a temporary hold means the patient is on the drug
    MedStatus.STOPPED: _Signal.INACTIVE,
    MedStatus.UNCLEAR: _Signal.UNKNOWN,
}


def combine_signals(signals: Iterable[_Signal]) -> MedicationStatus:
    """Merge status signals, failing closed.

    ACTIVE + INACTIVE is a CONFLICT. ACTIVE wins over UNKNOWN (it already
    requires a plan); INACTIVE does not (an unknown might mean still taking it).
    No signal at all is UNKNOWN.
    """

    present = set(signals)
    if _Signal.CONFLICT in present or {_Signal.ACTIVE, _Signal.INACTIVE} <= present:
        return MedicationStatus.CONFLICT
    if _Signal.ACTIVE in present:
        return MedicationStatus.ACTIVE
    if _Signal.UNKNOWN in present or not present:
        return MedicationStatus.UNKNOWN
    return MedicationStatus.INACTIVE


def _structured_signal(entries: Sequence[StructuredMedication]) -> _Signal | None:
    """The JSON list's view; ``active: null`` entries add no signal."""

    values = {entry.active for entry in entries if entry.active is not None}
    if values == {True}:
        return _Signal.ACTIVE
    if values == {False}:
        return _Signal.INACTIVE
    return _Signal.CONFLICT if values else None


def _text_signal(mentions: Sequence[tuple[TrustedDocument, TrustedMedication]]) -> _Signal | None:
    """Status from the most recently dated document(s) that mention the drug.

    Older notes are history ("maintained on warfarin" written before a switch);
    only the latest date counts. Disagreement on that date is a conflict.
    """

    if not mentions:
        return None
    latest = max(doc.documented_on or date.min for doc, _ in mentions)
    on_latest = [_TEXT_SIGNAL[med.status] for doc, med in mentions if (doc.documented_on or date.min) == latest]
    status = combine_signals(on_latest)
    return _Signal(status.value)


def resolve_medications(
    structured: StructuredSubmission, docs: Sequence[TrustedDocument]
) -> tuple[ResolvedMedication, ...]:
    """Anticoagulants and antiplatelets, merged per generic drug.

    Final status = combine(JSON signal, latest-text signal). JSON ``active:
    false`` with a latest note saying CURRENT (or the reverse) is a CONFLICT.
    Unverified mentions count as UNCLEAR, so an unconfirmed report of an
    anticoagulant still blocks READY rather than vanishing.
    """

    json_entries: dict[str, list[StructuredMedication]] = defaultdict(list)
    for entry in structured.medications:
        if entry.drug_class is not DrugClass.OTHER and entry.generic_name is not None:
            json_entries[entry.generic_name].append(entry)

    text_mentions: dict[str, list[tuple[TrustedDocument, TrustedMedication]]] = defaultdict(list)
    drug_classes: dict[str, DrugClass] = {e.generic_name: e.drug_class for es in json_entries.values() for e in es if e.generic_name}
    for doc in docs:
        for med in doc.medications:
            drug_class, generic = classify_drug(med.drug_name_raw)
            if generic is None or drug_class is DrugClass.OTHER:
                continue
            text_mentions[generic].append((doc, med))
            drug_classes.setdefault(generic, drug_class)

    resolved: list[ResolvedMedication] = []
    for generic in sorted(set(json_entries) | set(text_mentions)):
        entries, mentions = json_entries.get(generic, []), text_mentions.get(generic, [])
        signals = [s for s in (_structured_signal(entries), _text_signal(mentions)) if s is not None]
        evidence = [e.evidence.model_copy(update={"detail": f"JSON active={e.active}"}) for e in entries]
        evidence += [
            doc_evidence(doc, med.quote, f"note status {med.status}" + ("" if med.verified else " (quote unverified)"))
            for doc, med in mentions
        ]
        resolved.append(
            ResolvedMedication(
                generic_name=generic,
                display_name=entries[0].name_raw if entries else mentions[0][1].drug_name_raw,
                drug_class=drug_classes[generic],
                status=combine_signals(signals),
                evidence=tuple(evidence),
            )
        )
    return tuple(resolved)


# =============================================================================
# Step 2.5: anticoagulation plans
# =============================================================================

_DURATION: Final = re.compile(r"\b(\d+)\s*(days?|hours?|hrs?|h)\b", re.IGNORECASE)
_LAST_DOSE: Final = re.compile(r"last dose\s*(?:on\s*)?([0-9/\-]+)", re.IGNORECASE)
_CONTINUE: Final = re.compile(r"\b(continue|do not (?:hold|stop)|without interruption)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PreopDirective:
    """The comparable core of a pre-op instruction."""

    continues: bool
    hold_hours: int | None
    last_dose: date | None


def preop_directive(plan: AnticoagulationPlan) -> PreopDirective | None:
    """``"Hold Xarelto for the 2 days before surgery (last dose 02/26/2026)"`` -> hold 48 h, last dose 2026-02-26."""

    text = plan.preop_instruction
    if not text:
        return None
    hours: int | None = None
    if duration := _DURATION.search(text):
        hours = int(duration[1]) * (24 if duration[2].lower().startswith("day") else 1)
    last_dose: date | None = None
    if dose := _LAST_DOSE.search(text):
        last_dose = parse_raw_date(dose[1], plan.written_on)
    return PreopDirective(continues=bool(_CONTINUE.search(text)), hold_hours=hours, last_dose=last_dose)


def plans_contradict(a: AnticoagulationPlan, b: AnticoagulationPlan) -> bool:
    """Contradictory pre-op instructions for the same drug.

    Continue vs hold, different hold durations, or different last-dose dates.
    Different wording of the same instruction is not a contradiction.
    """

    da, db = preop_directive(a), preop_directive(b)
    if da is None or db is None:
        return False
    if da.continues != db.continues:
        return True
    if da.hold_hours is not None and db.hold_hours is not None and da.hold_hours != db.hold_hours:
        return True
    return da.last_dose is not None and db.last_dose is not None and da.last_dose != db.last_dose


def _plan_from(doc: TrustedDocument, trusted: TrustedPlan) -> AnticoagulationPlan:
    mention = trusted.plan
    _, generic = classify_drug(mention.drug_name_raw)
    preop = mention.preop_instruction or None
    return AnticoagulationPlan(
        plan_id=f"{doc.doc_id}:plan[{trusted.index}]",
        document_id=doc.doc_id,
        written_on=doc.documented_on,
        drugs_addressed=(generic or mention.drug_name_raw,),
        preop_instruction=preop,
        preop_timing_specified=preop is not None and bool(mention.preop_timing_raw),
        postop_instruction=mention.postop_instruction or None,
        hedged=mention.is_hedged,
        evidence=(doc_evidence(doc, trusted.quote, f"plan for {mention.drug_name_raw}"),),
    )


def resolve_anticoagulation_plans(docs: Sequence[TrustedDocument]) -> tuple[AnticoagulationPlan, ...]:
    """Every verified plan; contradictory plans for the same drug cross-reference each other.

    ``conflicts_with`` (exposed as ``has_conflicting_plans``) makes the engine
    raise ANTICOAG_PLAN_CONFLICT for that drug instead of accepting either plan.
    """

    plans = [_plan_from(doc, trusted) for doc in docs for trusted in doc.plans]
    conflicts: dict[str, set[str]] = defaultdict(set)
    for i, a in enumerate(plans):
        for b in plans[i + 1 :]:
            if set(a.drugs_addressed) & set(b.drugs_addressed) and plans_contradict(a, b):
                conflicts[a.plan_id].add(b.plan_id)
                conflicts[b.plan_id].add(a.plan_id)
    return tuple(plan.model_copy(update={"conflicts_with": tuple(sorted(conflicts[plan.plan_id]))}) for plan in plans)


# =============================================================================
# Unreadable sources
# =============================================================================

_FAIL_CLOSED_RECORDS: Final = ("vitals", "medications")


def collect_unreadable_sources(
    structured: StructuredSubmission, docs: Sequence[TrustedDocument]
) -> tuple[Evidence, ...]:
    """Failed or missing extractions, and malformed vital / medication records."""

    found: list[Evidence] = []
    extracted_ids = {doc.doc_id for doc in docs}
    for doc in docs:
        if doc.extraction_failed:
            found.append(doc_evidence(doc, None, f"document extraction failed: {doc.failure_reason or 'unknown error'}"))
    for header in structured.documents:
        if header.doc_id not in extracted_ids:
            found.append(
                Evidence(
                    source=EvidenceSource.DOCUMENT, locator=f"documents[{header.doc_id}]",
                    label=header.title, recorded_on=header.documented_on, detail="document was not extracted",
                )
            )
    for record in structured.excluded:
        if record.reason is ExclusionReason.MALFORMED and record.locator.startswith(_FAIL_CLOSED_RECORDS):
            found.append(Evidence(source=EvidenceSource.STRUCTURED, locator=record.locator, detail=record.detail))
    return tuple(found)


# =============================================================================
# Entry points
# =============================================================================


@dataclass(frozen=True)
class ReconciliationResult:
    facts: ReconciledPatientFacts
    # Human-readable audit trail of merge decisions (superseded consents, ties, ignored facts).
    notes: tuple[str, ...]


def reconcile_with_notes(
    structured_facts: StructuredSubmission, extracted_docs: Sequence[ExtractedDocument]
) -> ReconciliationResult:
    headers = {header.doc_id: header for header in structured_facts.documents}
    docs = [
        filter_unverified(doc, headers.get(doc.doc_id or ""), fallback_id=f"extracted[{index}]")
        for index, doc in enumerate(extracted_docs)
    ]
    usable = [doc for doc in docs if not doc.extraction_failed]
    notes: list[str] = []

    procedure = resolve_procedure(structured_facts)
    blood_pressures, temperatures, unreadable_vitals = merge_vitals(structured_facts, usable, notes)
    facts = ReconciledPatientFacts(
        case_id=procedure.case_id or structured_facts.demographics.patient_id or "UNKNOWN",
        procedure=procedure,
        history_and_physicals=resolve_history_and_physicals(usable),
        consents=resolve_consents(usable, procedure.procedure_type, notes),
        labs=structured_facts.labs,
        medications=resolve_medications(structured_facts, usable),
        anticoagulation_plans=resolve_anticoagulation_plans(usable),
        blood_pressures=blood_pressures,
        temperatures=temperatures,
        unreadable_sources=(*collect_unreadable_sources(structured_facts, docs), *unreadable_vitals),
    )
    return ReconciliationResult(facts=facts, notes=tuple(notes))


def reconcile_patient_facts(
    structured_facts: StructuredSubmission, extracted_docs: Sequence[ExtractedDocument]
) -> ReconciledPatientFacts:
    """Merge Stage 2 structured facts with verified Stage 3 document facts.

    ``extracted_docs`` must already have been through ``verify.verify_extracted_document``;
    facts that were never verified are treated as unverified.
    """

    return reconcile_with_notes(structured_facts, extracted_docs).facts
