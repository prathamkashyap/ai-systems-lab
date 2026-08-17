# Contributing

## Setup

```bash
git clone https://github.com/prathamkashyap/ai-systems-lab.git
cd ai-systems-lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest   # should show all tests passing before you change anything
```

## Making a change

1. Branch off `main`: `git checkout -b feat/short-description`.
2. Write the change. If you're adding a provider, chunker, or embedder,
   implement it against the existing interface
   (`backend/providers/base.py`, `data/embeddings.py`'s `Embedder` protocol)
   rather than special-casing it into the gateway/pipeline — that's the
   whole point of those seams.
3. Add or update tests in `tests/`. A PR that changes behavior without a
   test covering it will be asked to add one.
4. Run the full local check before opening a PR:
   ```bash
   make lint
   make test
   ```
5. If the change affects request handling, caching, or batching, sanity
   check the benchmark numbers still make sense:
   ```bash
   make bench
   ```
   and update `benchmarks/results/README.md` if the numbers materially
   change.
6. If the change is architectural (new component, changed data flow, a
   tradeoff worth recording), add or update an ADR in
   `system-design/adr/` — see existing ADRs for the format. Small
   refactors and bug fixes don't need one.

## Commit / PR conventions

- Keep commits scoped to one logical change.
- PR description should state *what* changed and *why*, not just *what*
  files were touched — the diff already shows that.
- CI (`.github/workflows/ci.yml`) runs lint, tests, and a Docker build on
  every PR; it must be green before merge.

## Code style

- Enforced by `ruff` (`make lint` / `make fmt` to autofix). No separate
  style guide beyond what the linter checks.
- Type hints on public function signatures; the codebase targets Python
  3.11+ and uses `from __future__ import annotations` throughout.
