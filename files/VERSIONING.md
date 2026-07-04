# Versioning

How this repo tracks **package releases**, **git tags**, and (separately) **prompt versions**.

## Rule: one patch bump per merged feature

Every time a **feature branch merges into `dev`**, bump the **patch** version:

```text
0.1.0  →  Phase 1 baseline (main)
0.1.1  →  feature/config-catalog merged
0.1.2  →  feature/pricing-retry-models merged
0.1.3  →  feature/providers merged
0.1.4  →  feature/factory-cli merged
0.1.5  →  feature/benchmark-runner merged
0.1.6  →  feature/integration merged — Phase 2 complete on dev
0.1.7+ →  Phase 3 feature branches (same rule)
```

The canonical version lives in **`pyproject.toml`** (`[project].version`).

Read it in code:

```python
from src import __version__
print(__version__)  # e.g. 0.1.3
```

```bash
uv run python -c "from src import __version__; print(__version__)"
```

## Version map (Phase 2)

| Version | Feature branch | Merged to `dev`? |
|---------|----------------|------------------|
| `0.1.0` | Phase 1 (`main`) | yes |
| `0.1.1` | `feature/config-catalog` | yes |
| `0.1.2` | `feature/pricing-retry-models` | yes |
| `0.1.3` | `feature/providers` | yes |
| `0.1.4` | `feature/factory-cli` | yes |
| `0.1.5` | `feature/benchmark-runner` | yes |
| `0.1.6` | `feature/integration` | yes — **Phase 2 complete** |

When Phase 2 is complete and merged to **`main`**, you may optionally tag **`v0.1.6`** (or continue patch series into Phase 3).

## Merge workflow (includes version bump)

After you say **"good to go"** on a feature:

1. Merge `feature/<name>` → `dev`
2. Increment patch in `pyproject.toml` (`0.1.3` → `0.1.4`)
3. Move `[Unreleased]` notes in `CHANGELOG.md` into `## [0.1.4] - YYYY-MM-DD`
4. Run `uv sync && uv run pytest`
5. Commit on `dev`:

```text
Release v0.1.4: factory CLI tests.

Bump version after merging feature/factory-cli.
```

6. Push `dev`
7. *(Optional)* Tag on `dev`: `git tag -a v0.1.4 -m "..." && git push origin v0.1.4`

**Do not bump** on individual task-pair commits inside a feature branch — only when the **whole feature** merges to `dev`.

## Git tags

| Tag | When |
|-----|------|
| `v0.1.0` | Phase 1 on `main` (tag once if not already) |
| `v0.1.1` … `v0.1.N` | Optional after each feature merge to `dev` |
| `v0.2.0` | Optional later if you want a **minor** “Phase 2 shipped on main” marker |

Patch tags are enough for this project until you reach `1.0.0`.

### Tag Phase 1 (one-time, on `main`)

```bash
git checkout main
git tag -a v0.1.0 -m "Phase 1: Gemini pipeline and evaluation metrics"
git push origin v0.1.0
```

## What to bump when

| Change | Bump | Example |
|--------|------|---------|
| Feature branch merged to `dev` | **Patch** | `0.1.3` → `0.1.4` |
| Bug fix on `dev` / `main` (no feature branch) | **Patch** | `0.1.3` → `0.1.4` |
| Full phase merged to `main` (optional) | **Minor** | `0.1.6` → `0.2.0` |
| Stable public API (future) | **Major** | `0.x` → `1.0.0` |

Always update **`CHANGELOG.md`** in the same commit as the version bump.

## Branch workflow vs version

```text
main     ← Phase 1 at 0.1.0; later receives merged dev at 0.1.N
  ↑
dev      ← version increments after each merged feature (0.1.3 now)
  ↑
feature/* ← no version bump until merge
```

## Prompt versioning (separate)

**Package version** (`0.1.3`) ≠ **prompt version** (`1.0.0` in `prompts.yaml`).

Prompt versions track template changes per run (Phase 3). See [files/PHASE_3.md](../files/PHASE_3.md).

## Run metadata (future)

Benchmark reports should include `app_version` from `src.__version__` — wire when `BenchmarkReport` I/O lands.

## Checklist after each feature merge

- [ ] `pyproject.toml` patch incremented
- [ ] `CHANGELOG.md` section added for new version
- [ ] `README.md` version line updated (if shown)
- [ ] `uv sync && uv run pytest` green
- [ ] Commit + push `dev`
- [ ] Optional git tag `v0.1.N`
