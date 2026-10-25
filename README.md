# Pre-Op Triage Take-Home

## Objective

Implement `triage_submission(...)` in `core.py` - it is a pre-op triage function for a single submission package. It is currently a naive LLM-based solution that makes a real model API call. The starter implementation intentionally does not follow some best practices in using the OpenAI API. You may use whatever file structure makes sense for your solution.

Your output must match this schema:

- `decision`: `READY | NEEDS_FOLLOW_UP | NOT_CLEARED`
- `issues[]`: `category` + evidence (how you design this is up to you)
- `explanation`

## What Is Provided

- `data/patients_sample_50.jsonl` includes:
  - `case_id`
  - `submission`
  - `label` (what a human labeled this case as)
- `run_baseline.py` runs your `triage_submission` implementation and writes outputs.
- `run_evals.py` scores outputs against provided `label` and can run determinism checks.

## Completion

Note: this exercise is evaluated on engineering judgment. You may not reach a 100% score, and that is OK! We are looking to understand how you approached the problem and designed a working solution.

## Setup

1. Confirm `uv` is installed.

```bash
uv --version
```

**(If you don't have uv installed already, run:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Then after cloning the repo, run:
```bash
uv sync
```
This installs the dependencies.

2. Set your OpenAI API key.

```bash
export OPENAI_API_KEY="<your_api_key>"
```

## Recommended Workflow

1. Implement `triage_submission` in `core.py`.
2. Run baseline outputs:

```bash
make baseline
```

3. Run eval scoring (offline, no API key needed):

```bash
make evals
# gate on false READYs (exit status 2 if any):
make evals SCORER_FLAGS=--fail-on-false-ready
```

`make evals` runs `scorer.py`, which replaced the OpenAI Evals upload in
`run_evals.py`. It prints a decision confusion matrix, the false-READY count
(target 0), and precision/recall per issue category, and writes
`data/eval_report.json` in the format `make report` / `make score` read.
`run_evals.py` is still used for `make determinism` only.

4. Run determinism check:

```bash
make determinism
```

5. Print score:

```bash
make score
```

6. View the interactive report (TUI):

```bash
make report
```

This opens a terminal UI (`view_report.py`) that shows per-case results side-by-side with oracle expectations. You can browse records, see metric pass/fail status, and inspect submission data. Press `f` on a metric row to filter the case list to failures. Press `q` to quit.

## Tests

Run the unit tests with:

```bash
make test
```

## Architecture (in progress)

- `models.py`: strict output contract (`TriageResult`, `Issue`, `Evidence`) and
  `ReconciledPatientFacts`, the merged JSON + document view the engine consumes.
- `engine.py`: pure rule functions R0-R4 and `evaluate(facts) -> TriageResult`.
  All policy thresholds and windows are constants at the top of the file.
- `preprocess.py`: Stage 2 structured-data parsing (labs, vitals, medications,
  demographics) with lab and drug lookup tables. Records that are read but not
   used are returned in `excluded` with a reason, never dropped silently.
- `schemas.py`: Stage 3 LLM output schema (`ExtractedDocument`). Every fact is a
  `QuotedFact[T]` (`value`, verbatim `quote`, `verified`). Values are raw text as
  written; enums fall back to `UNCLEAR`. `verified` and `extraction_status` are
  hidden from the model's JSON schema.
- `extract.py`: `extract_document(doc_text, doc_id, client)` makes one GPT-4o
  Structured Outputs call per document, with the document wrapped in
  `<untrusted_document>` tags. Raw responses are cached in `data/llm_cache.json`
  under SHA-256(document text + system prompt + schema name). Cache hits need no
  API key. Any failure returns `extraction_status = FAILED` and never raises.
- `verify.py`: `verify_extracted_document(doc, original_text)` checks every quote
  against the source after NFKC and whitespace normalization. Quotes under 4
  characters or containing an ellipsis are rejected. Facts that fail are kept with
  `verified = False` (fail closed), never deleted.
- `reconcile.py`: Stage 4. `reconcile_patient_facts(structured, extracted_docs)`
  merges the Stage 2 structured facts with the verified Stage 3 document facts into
  `ReconciledPatientFacts`. It is pure Python with no LLM calls.
  - Unverified quotes are first mapped to "absent" or `UNCLEAR`.
  - The procedure date and risk level come only from the JSON.
  - All H&Ps are kept, newest first. Only the most recent consent is kept.
  - Note vitals are merged with the structured vitals, newest first. Date-only
    readings are evaluated worst-case.
  - Each drug's status comes from the JSON flag and the latest dated note mention.
    A disagreement between them is a `CONFLICT`.
  - Plans for the same drug with contradictory pre-op instructions are flagged as
    conflicting.
  - Failed or missing extractions, malformed vital or medication records, and
    unreadable recent vitals become `unreadable_sources`. The engine reports these
    as `EXTRACTION_FAILED` (`MISSING_REQUIRED_DATA`).
- `scorer.py`: offline scoring harness.
- `tests/test_gold_cases.py`: hand-reconciled facts for 19 sample cases, checked
  against their labels.

## Outputs

- Baseline outputs: `data/baseline_outputs.jsonl`
- Eval report: `data/eval_report.json`
- Determinism report: `data/determinism_report.json`

## Configurable Variables

- `MODEL` (default `gpt-4o`)
- `INPUT` (default `data/patients_sample_50.jsonl`)
- `OUTPUT` (default `data/baseline_outputs.jsonl`)
- `REPORT` (default `data/eval_report.json`)
- `DETERMINISM_REPORT` (default `data/determinism_report.json`)
