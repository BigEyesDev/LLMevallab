# Tutorial: CI Pipeline for an ML Evaluation Project

> **Context:** This tutorial documents how CI was set up for LLMevallab (v0.2.1, Phase 2b Priority 5).
> It's written as a *learning record* — not just what to do, but **why** each decision was made
> and what traps to avoid.

---

## 1. What is a CI pipeline and why does it matter here?

**Continuous Integration (CI)** automatically runs your tests every time code is pushed to a shared branch.
For an evaluation project this is especially important because:

- Metric logic is subtle — a one-line bug in ROUGE/BLEU aggregation silently produces wrong numbers.
- The pipeline wires together many moving parts: loaders, processors, evaluator, provenance. Breaking one can corrupt results without an obvious error.
- You want to be confident that `main` is always green before promoting results or writing a blog post about them.

---

## 2. Anatomy of a GitHub Actions workflow

A workflow file lives at `.github/workflows/<name>.yml`.
Here is the structure used in this project:

```yaml
name: Tests                  # displayed in the GitHub UI

on:                          # when to trigger
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  test:                      # one job (can have many)
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12"]   # easy to add 3.11 later
    steps:
      - uses: actions/checkout@v4          # step 1: get the code
      - uses: astral-sh/setup-uv@v5       # step 2: install uv
      - uses: actions/setup-python@v5     # step 3: pin Python
        with:
          python-version: ${{ matrix.python-version }}
      - uses: actions/cache@v4            # step 4: restore venv from cache
        with:
          path: .venv
          key: venv-${{ runner.os }}-py${{ matrix.python-version }}-${{ hashFiles('uv.lock') }}
      - run: uv sync --frozen             # step 5: install (fast if cache hit)
      - run: |                            # step 6: set dummy env vars
          echo "GEMINI_API_KEY=ci-dummy" >> $GITHUB_ENV
      - run: uv run pytest tests/ -v      # step 7: run tests
```

### The `on:` triggers

```yaml
on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]
```

- `push` fires when you push directly to a branch.
- `pull_request` fires when a PR is opened or updated targeting those branches.

Together these guarantee: every commit that could end up on `main` or `dev` is tested — including the merge commit that GitHub synthesises for a PR.

---

## 3. Why `uv` instead of `pip`?

[`uv`](https://github.com/astral-sh/uv) is a drop-in replacement for pip/virtualenv written in Rust.
It is **10–100× faster** for dependency resolution and installation.

In CI this matters because:

- A cold pip install of this project (torch, transformers, bert-score…) takes 5-10 minutes.
- With `uv sync --frozen` + a warm cache, it takes **under 10 seconds**.

`--frozen` means: do not update `uv.lock`, just install exactly what is pinned.
This is critical for reproducibility — CI must run against the exact locked dependency tree.

---

## 4. Caching the virtual environment

```yaml
- uses: actions/cache@v4
  with:
    path: .venv
    key: venv-${{ runner.os }}-py${{ matrix.python-version }}-${{ hashFiles('uv.lock') }}
    restore-keys: |
      venv-${{ runner.os }}-py${{ matrix.python-version }}-
```

**How it works:**

1. GitHub Actions computes the cache key: `venv-ubuntu-latest-py3.12-<sha256 of uv.lock>`.
2. If the key is found in the cache store, it restores `.venv/` before the install step.
3. `uv sync --frozen` then has nothing to do and completes in ~1 second.
4. If `uv.lock` changes (a dependency was updated), the key changes, cache misses, and a full install runs.

**`restore-keys`** is a fallback prefix: even if the exact key is not found, GitHub will restore the most recent cache that starts with `venv-ubuntu-latest-py3.12-`, and `uv` will top-up only what has changed.

---

## 5. Excluding API-dependent tests

Some tests call real LLM APIs (Gemini, Claude, OpenAI). These:

- Require live API keys (secrets).
- Are non-deterministic.
- Cost money.
- Can fail due to rate limits or outages — not because our code is broken.

The solution is to exclude them in CI and mock all external I/O in the tests that do run:

```bash
uv run pytest tests/ \
  --ignore=tests/test_gemini_processor.py \
  --ignore=tests/test_claude_processor.py \
  --ignore=tests/test_openai_compatible_processor.py \
  -v --tb=short
```

Provider contract tests (`test_processor_contract.py`) test the *interface* of each processor class using mocks — they don't call any API but do verify that all three providers implement the required methods.

**Lesson learned:** mock at the HTTP layer, not the class layer.
If you mock `GeminiProcessor.translate()`, you miss bugs in the retry logic, token counting, and error handling.
If you mock the HTTP client (`google.generativeai`), those code paths all run.

---

## 6. Dummy API keys in CI

Some import paths do:

```python
api_key = os.environ.get(model_config["api_key_env"])
if not api_key:
    raise EnvironmentError(f"{model_config['api_key_env']} not set in environment.")
```

In CI this would abort tests that only need imports — before a single test runs.

The fix is to set dummy values:

```yaml
- name: Set dummy API keys
  run: |
    echo "GEMINI_API_KEY=ci-dummy" >> $GITHUB_ENV
    echo "ANTHROPIC_API_KEY=ci-dummy" >> $GITHUB_ENV
```

These are not secrets. They prevent early EnvironmentError. They will not succeed in making a real API call (any test that tries will fail with an auth error — which is expected, and those tests are excluded).

---

## 7. The `strategy.matrix` pattern

```yaml
strategy:
  fail-fast: false
  matrix:
    python-version: ["3.12"]
```

Even with one version today, this pattern is worth using from the start because:

- Adding `"3.11"` later is a one-line change.
- `fail-fast: false` means if 3.11 fails and 3.12 passes, you still see both results — not just the first failure.

---

## 8. `--tb=short` for readable CI logs

```bash
pytest ... --tb=short
```

The default (`--tb=auto`) prints full tracebacks for each failure.
In CI, where you are scanning a wall of log text, short tracebacks are easier to triage.
Use `--tb=long` locally when actively debugging a failure.

---

## 9. What CI does NOT test (and why that is acceptable)

| Not tested in CI | Reason |
|---|---|
| Real Gemini/Claude/OpenAI calls | Requires secrets; non-deterministic; costs money |
| Downloading HuggingFace datasets | Requires network; datasets are large; loaders are tested against pre-saved fixtures |
| End-to-end manifest + report round-trip with real data | Covered by `test_manifest.py` with mocked MetricsRunner; real data test is manual |
| BERTScore with real model weights | `tests/test_evaluator.py` mocks MetricsRunner; full BERTScore is slow and needs GPU |

The principle: **CI tests the correctness of your code, not the correctness of your dependencies or external services.**

---

## 10. Local CI parity

Run the exact same command locally before pushing:

```bash
uv run pytest tests/ \
  --ignore=tests/test_gemini_processor.py \
  --ignore=tests/test_claude_processor.py \
  --ignore=tests/test_openai_compatible_processor.py \
  -v --tb=short
```

If this passes locally, CI will pass — and vice versa.
There are no hidden environment differences (we use `uv` and `--frozen` both locally and in CI).

---

## 11. What a green CI badge actually guarantees

When `.github/workflows/test.yml` shows ✅ on `main`:

- All provenance/hash logic works correctly (`test_provenance.py`, `test_manifest.py`).
- Truncation limits are correctly applied and recorded (`test_truncation.py`).
- The evaluator refuses to run on a hash mismatch (`test_manifest.py`).
- All three provider processor contracts are met (`test_processor_contract.py`).
- Config loading, model catalog, and dataset path resolution are correct (`test_config.py`).
- The benchmark runner produces deterministic output for the same documents (`test_benchmark.py`).

**What it does not guarantee:** the quality of LLM outputs, the correctness of API integration, or the meaningfulness of metric scores on real data. Those require human review of evaluation reports.

---

## 12. Extending CI later

When Phase 3 is underway, consider:

```yaml
# Run a fast smoke test against real APIs using a tiny sample (1 doc)
# only when a secret is present (skips on forks / external PRs).
- name: Smoke test — real API call (1 doc)
  if: secrets.GEMINI_API_KEY != ''
  run: |
    uv run python -m src.pipeline.orchestrator \
      --task summarisation \
      --model gemini-2.5-flash \
      --dataset cnn_dailymail \
      --sample 1
  env:
    GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

Add `secrets.GEMINI_API_KEY` in **GitHub → Settings → Secrets and variables → Actions**.

---

## Summary

| Decision | Why |
|---|---|
| GitHub Actions | Native to GitHub, zero infrastructure, free for public repos |
| `uv` | 10-100× faster installs; lockfile-first reproducibility |
| Cache keyed on `uv.lock` | Exact cache hit on unchanged deps; auto-invalidates on updates |
| Exclude API tests | Non-deterministic, expensive, require secrets |
| Dummy API keys in CI | Prevent EnvironmentError at import time in non-API tests |
| `--frozen` | CI must not silently upgrade a dependency |
| `matrix.python-version` | Ready to expand; `fail-fast: false` gives full visibility |
| `--tb=short` | Readable CI logs |
