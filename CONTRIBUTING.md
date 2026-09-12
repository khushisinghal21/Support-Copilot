# Contributing

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate

# Install the PINNED set. requirements.txt uses ">=" with no upper bound, which is
# right for declared compatibility and wrong for reproducibility -- a fresh clone
# resolving freely is how this project once ended up installing a full CUDA torch
# and running out of memory on a 512MB host.
pip install --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            -r requirements.lock

# The dev tooling (ruff, mypy, pytest-cov, pre-commit) is NOT in the runtime
# requirements, because src/ does not import it -- but you need it to run the
# checks CI gates on, and ruff is pinned here to the same version
# .pre-commit-config.yaml uses so your hook and CI cannot disagree.
pip install -r requirements-dev.txt

cp .env.example .env      # optional: add a real GEMINI_API_KEY
pre-commit install        # runs the same checks CI does, before the commit
```

The CPU-only index URL is not optional on a GPU-less machine, and it must be in
the **same** pip invocation as the requirements file. A separate `pip install
torch --index-url .../cpu` beforehand does not survive: the second command is a
fresh resolver with no knowledge of that index and pulls the CUDA build straight
back in from PyPI. `render.yaml`, the `Dockerfile` and `.github/workflows/ci.yml`
all carry the same single-call form for the same reason.

## Everyday commands

| Command | What it does |
| :--- | :--- |
| `pytest tests/ -q` | Full suite. Runs offline -- no network, no API key, no flake. |
| `pytest tests/ --cov=src --cov-report=term-missing` | Suite with coverage. CI fails under 85% overall, 95% on `src/triage/` and `src/drafting/guardrails.py`. |
| `ruff check . && ruff format --check .` | Lint and format, exactly as CI runs them. |
| `mypy src/` | Types. Clean across all of `src/`; strict on `src/models.py`, `src/pipeline.py`, `src/triage/`. |
| `python -m src.eval.runner` | Full benchmark. Headline numbers on held-out rows, with the calibration gap printed beside them. |
| `python -m src.eval.runner --quick` | 50-row subset, for a fast sanity check. |
| `python scripts/calibrate_thresholds.py` | Threshold sweep. Reads the **calibration** split only -- see below. |
| `python scripts/add_golden_split.py` | Re-assigns the persisted calibration/held-out split. Idempotent. |
| `./run.sh` | Dashboard + API on `http://localhost:8000`. |
| `docker compose up --build` | Same thing in a container. |

## Things that will get a change rejected

These are invariants, not preferences. Each one exists because it was violated
once already.

1. **Never report a number that did not come from a run.** Every figure in
   `README.md` and `docs/REPORT.md` comes from `docs/benchmark_summary.json`, which
   the eval harness writes. `tests/test_doc_code_consistency.py` enforces it. If
   you cannot run something, say so and label the gap -- do not estimate.
2. **Headline metrics are reported on the held-out split only.** The two triage
   thresholds were tuned against the `calibration` rows; scoring on those rows
   flatters the system by construction. `scripts/calibrate_thresholds.py` must
   never read a held-out row (there is a test).
3. **Input-side triage gates run before retrieval and generation.** Gates 1-5
   exist to stop a ticket reaching the model. A pipeline that generates first
   cannot honour that, whatever the diagram says. Tested by asserting
   `generator.generate.call_count == 0` for an input-gated tweet.
4. **Fail closed.** Any unhandled exception in the pipeline must produce
   `ESCALATE` / `SYSTEM_EXCEPTION_FAIL_CLOSED` with no draft leaked.
5. **Keep gate order and reason codes stable.** The reason code is an
   auditability contract: it tells a human operator why a ticket reached them.
   Renaming one silently breaks every downstream consumer and every historical
   audit row.
6. **Golden-set source rows stay out of the RAG corpus.** Without that exclusion
   the system can retrieve the exact reply it is being scored against.
7. **Explain *why*, and update the explanation when the code moves.** The comments
   in `src/config.py`, `render.yaml` and `src/drafting/prompts.py` record evidence
   for decisions -- threshold sweeps, real incidents, SDK bugs. If you change what
   a comment describes, update the comment; never delete the reasoning.
8. **Tests run offline.** No network, no API key. Mock the encoder; the grounding
   check already falls back to its lexical path when no model is available.

## Regenerating the eval

```bash
python -m src.eval.runner        # writes docs/REPORT.md + docs/benchmark_summary.json
```

`docs/REPORT.md` is **generated**. Edit `src/eval/report_generator.py`, not the
markdown -- a hand-edit is overwritten by the next run and, worse, becomes a number
nobody can trace to a run.
