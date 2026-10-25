INPUT ?= data/patients_sample_50.jsonl
OUTPUT ?= data/baseline_outputs.jsonl
REPORT ?= data/eval_report.json
DETERMINISM_REPORT ?= data/determinism_report.json
MODEL ?= gpt-4o
SCORER_FLAGS ?=

.PHONY: baseline evals determinism score report test typecheck all clean

baseline:
	uv run run_baseline.py \
		--input $(INPUT) \
		--output $(OUTPUT) \
		--model $(MODEL)

# Offline scorer (no network). Add SCORER_FLAGS=--fail-on-false-ready to gate on false READYs.
evals:
	uv run scorer.py \
		--labels $(INPUT) \
		--outputs $(OUTPUT) \
		--report $(REPORT) \
		$(SCORER_FLAGS)

determinism:
	uv run run_evals.py \
		--determinism \
		--input $(INPUT) \
		--model $(MODEL) \
		--report $(DETERMINISM_REPORT)

score:
	@python3 -c 'import json; r=json.load(open("$(REPORT)")); s=(r.get("primary_score",{}) or {}).get("value_pct"); print(s if s is not None else r.get("local_metrics_summary",{}).get("aggregate_local_score_pct", 0.0))'

report:
	uv run view_report.py --report $(REPORT)

test:
	uv run \
		--with 'openai>=2.0.0' \
		--with 'pydantic>=2.8.0' \
		--with 'pytest>=8.0.0' \
		python -m pytest tests

typecheck:
	MYPYPATH=tests uv run \
		--with 'openai>=2.0.0' \
		--with 'pydantic>=2.8.0' \
		--with 'pytest>=8.0.0' \
		--with 'mypy>=1.10' \
		mypy --strict --explicit-package-bases \
		models.py engine.py preprocess.py scorer.py \
		schemas.py extract.py verify.py reconcile.py packet.py core.py \
		tests/fact_builders.py tests/llm_fixtures.py tests/test_engine.py \
		tests/test_gold_cases.py tests/test_preprocess.py tests/test_scorer.py \
		tests/test_extract.py tests/test_verify.py tests/test_reconcile.py \
		tests/test_triage_submission.py

all: baseline evals determinism score

clean:
	rm -f data/baseline_outputs.jsonl \
		data/eval_report.json \
		data/determinism_report.json
