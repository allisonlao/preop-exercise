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
make typecheck   # mypy --strict
```

## Architecture (in progress)

- `models.py`: strict output contract (`TriageResult`, `Issue`, `Evidence`) and
  `ReconciledPatientFacts`, the merged JSON + document view the engine consumes.
- `engine.py`: pure rule functions R0-R4 and `evaluate(facts) -> TriageResult`.
  All policy thresholds and windows are constants at the top of the file.
- `preprocess.py`: Stage 2 structured-data parsing (labs, vitals, medications,
  demographics) with lab and drug lookup tables. Records that are read but not
  used are returned in `excluded` with a reason, never dropped silently.
- `scorer.py`: offline scoring harness.
- `tests/test_gold_cases.py`: hand-reconciled facts for 19 sample cases, checked
  against their labels.

## Outputs

- Baseline outputs: `data/baseline_outputs.jsonl`
- Eval report: `data/eval_report.json`
- Determinism report: `data/determinism_report.json`

## Configurable Variables

- `MODEL` (default `gpt-4.1-mini`)
- `INPUT` (default `data/patients_sample_50.jsonl`)
- `OUTPUT` (default `data/baseline_outputs.jsonl`)
- `REPORT` (default `data/eval_report.json`)
- `DETERMINISM_REPORT` (default `data/determinism_report.json`)
