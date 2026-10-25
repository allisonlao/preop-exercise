"""Shared Stage 3 test helpers: sample-document lookup and hardcoded LLM responses."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "data" / "patients_sample_50.jsonl"
RESPONSES = Path(__file__).resolve().parent / "fixtures" / "llm_responses.json"
CASE_00020 = Path(__file__).resolve().parent / "fixtures" / "case_00020_postop_deferral.json"
PATIENT_REPORTED = Path(__file__).resolve().parent / "fixtures" / "patient_reported_anticoagulants.json"

NURSING_DOC_ID = "1f5be5f7-1a56-5e66-9c45-2e94e55b0271"
CONSENT_DOC_ID = "fdce8743-2288-5749-b5df-f9f191788315"


@cache
def _documents() -> dict[str, str]:
    texts: dict[str, str] = {}
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        if line.strip():
            for doc in json.loads(line)["submission"].get("documents", []):
                texts[doc["doc_id"]] = doc["text"]
    return texts


def document_text(doc_id: str) -> str:
    return _documents()[doc_id]


def raw_response(doc_id: str) -> str:
    """The hardcoded model output for ``doc_id`` as a raw JSON string."""

    responses: dict[str, Any] = json.loads(RESPONSES.read_text(encoding="utf-8"))
    return json.dumps(responses[doc_id])


@cache
def case_00020() -> dict[str, Any]:
    """The case_00020 false-READY record: submission, label, and the pre-enum model output."""

    loaded: dict[str, Any] = json.loads(CASE_00020.read_text(encoding="utf-8"))
    return loaded


@cache
def patient_reported_anticoagulants() -> dict[str, Any]:
    """Cases 00000, 00027, 00038: submission, label, real extractions, and the reported-use sentence."""

    loaded: dict[str, Any] = json.loads(PATIENT_REPORTED.read_text(encoding="utf-8"))
    return {case_id: case for case_id, case in loaded.items() if not case_id.startswith("_")}
