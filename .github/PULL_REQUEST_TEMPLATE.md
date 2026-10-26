## What

<!-- What changes, and for whom? One or two sentences. -->

## Why

<!-- The failure, measurement, or requirement behind it. `Fixes #NNN` when an issue exists. -->

## How it was verified

<!-- Commands run, and what they printed. A claim without evidence is a request, not a PR. -->

- `torch` / `torch_musa`:
- driver / MUSA SDK:
- hardware:
- commands run:
- result:

## Version bump

Every PR that changes code, tests, or packaged data carries the work commit(s)
**and** one separate bump commit. See
[CONTRIBUTING.md](CONTRIBUTING.md#version-bumps).

- [ ] Work commit(s) first, then exactly one `chore: bump torchada to <version>` commit
- [ ] The bump replaced **all seven carriers**: `git grep -nF '<old version>' -- pyproject.toml src/torchada/__init__.py src/torchada/csrc/ops.h tests/test_platform.py examples/extension_setup.py README.md README_CN.md` prints nothing
- [ ] The newest `benchmarks/benchmark_history.json` row carries `<version>` (`git diff --stat` there: one insertion, one deletion)
- [ ] The bump commit contains nothing else
- [ ] Not applicable — documentation-only as defined in CONTRIBUTING.md (no bump commit)

## Compatibility impact

- [ ] Nothing user-visible changed
- [ ] `README.md` **and** `README_CN.md` updated
- [ ] Added a `TORCHADA_*` switch → documented in both READMEs and `AGENTS.md`
- [ ] Added or changed a `torch_musa` version gate → both sides of the bound tested
- [ ] Added a mapping rule → covered in `tests/test_mappings.py` (or `test_stable_abi_mappings.py`)
- [ ] Touched a hot path → `benchmarks/benchmark_overhead.py` re-run, README tables updated

## Notes for reviewers

<!-- Tradeoffs you rejected, follow-ups you deferred, anything surprising. -->
