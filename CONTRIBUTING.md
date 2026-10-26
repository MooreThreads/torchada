# Contributing to torchada

Thank you for your interest in contributing to torchada! This guide will help you get started.

- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Running Tests](#running-tests)
- [Pull Request Requirements](#pull-request-requirements) — read this before you open a PR
- [Version Bumps](#version-bumps) — every PR ends with its own bump commit
- [Adding New Patches](#adding-new-patches)
- [Critical Constraints](#critical-constraints)
- [Submitting Changes](#submitting-changes)

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/MooreThreads/torchada.git
cd torchada
pip install -e ".[dev]"
```

Python 3.8 or newer is required (`requires-python = ">=3.8"` in `pyproject.toml`).

### 2. Set Up Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically check and format code before each commit.

```bash
# Install the hooks (one-time setup)
pre-commit install

# (Optional) Run hooks against all files to verify setup
pre-commit run --all-files
```

Once installed, the hooks run automatically on `git commit`. They will:

- **Format** code with [black](https://github.com/psf/black) (line-length 100)
- **Sort imports** with [isort](https://github.com/PyCQA/isort) (black profile)
- **Lint** with [ruff](https://github.com/astral-sh/ruff) (unused imports F401, undefined names F821)
- **Check** for common issues (trailing whitespace, YAML/TOML syntax, merge conflicts, debug statements, large files, private keys, and `no-commit-to-branch` — `main` cannot be committed to directly)
- **Spell-check** with [codespell](https://github.com/codespell-project/codespell)

If a hook modifies files (e.g., reformatting), the commit will be aborted. Simply `git add` the changes and commit again. Do not pass `--no-verify`.

### 3. Manual Hook Usage

```bash
# Run all hooks on all files
pre-commit run --all-files

# Run a specific hook
pre-commit run black --all-files
pre-commit run isort --all-files

# Run hooks only on staged files (same as what happens on commit)
pre-commit run

# Update hook versions
pre-commit autoupdate
```

## Code Style

- **Formatter**: black with line-length 100
- **Import sorting**: isort with black profile
- **Python version**: >=3.8

These are enforced automatically by pre-commit hooks. You can also run them manually:

```bash
isort src/ tests/ && black src/ tests/
```

When formatting inside a Docker container, restore ownership so files remain editable on the host:

```bash
docker exec -w /ws <container> bash -c "isort src/ tests/ && black src/ tests/ && chown -R 1000:1000 src/ tests/"
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test class
pytest tests/test_cuda_patching.py::TestLibraryImpl -v

# Run with short traceback
pytest tests/ --tb=short

# Version metadata checks only (no GPU required)
pytest tests/test_platform.py -q
```

### Test Markers

- `@pytest.mark.musa` — Requires MUSA platform
- `@pytest.mark.cuda` — Requires CUDA platform
- `@pytest.mark.gpu` — Requires any GPU
- `@pytest.mark.slow` — Slow tests

Tests are grouped one module per feature area (`test_mappings.py`, `test_cpp_extension.py`,
`test_device_strings.py`, `test_graph_rotation.py`, `test_triton_patching.py`, ...). Add a
case to the module that owns the area rather than a new catch-all file.

### Docker Testing

Example development containers on the MUSA host:

```bash
docker exec -w /ws yeahdongcn python -m pytest tests/ --tb=short   # torch_musa 2.7.0
docker exec -w /ws yeahdongcn1 python -m pytest tests/ --tb=short  # torch_musa 2.7.1
```

When a change touches a `torch_musa` version gate, validate both sides of the bound.

## Pull Request Requirements

**Every pull request that changes code, tests, or packaged data must contain at least
two commits:**

1. **The work** — one or more commits that contain the actual change (feature, fix,
   tests). Split it logically if that makes review easier.
2. **The version bump** — a final, separate commit that only raises the version,
   with the message:

   ```
   chore: bump torchada to 0.1.102
   ```

   (substitute the version you are releasing).

A pull request that folds the version bump into the functional commit, or that bumps
only some of the files that carry the version, will be sent back for a redo. If you are
an AI agent, `AGENTS.md` states the same rule together with the rest of the repository
conventions.

### Documentation-only pull requests are exempt

A diff that touches nothing but Markdown — `README.md`, `README_CN.md`, `AGENTS.md`,
`CONTRIBUTING.md`, `docs/*.md`, `examples/*.md` — is a **documentation-only** PR. It
merges as a single `docs: ...` commit and **does not bump the version**: the bump marks
what ships, and a docs-only change ships nothing.

The exemption ends the moment the same PR also touches source, tests, `pyproject.toml`,
or a bundled tuning config. Mixed PRs are normal PRs and need both commits, so keep a
doc fix and a code fix in separate pull requests if you can.

### Why the split is required

The two commits have completely different review criteria. The work commit needs a
careful read of behavior, tests, and compatibility gates; the bump commit is a
mechanical `replace all` that must be verified by a search, not by judgment. Keeping
them apart means a reviewer can see the semantics of the change without a version
diff mixed in, and a release can be reconstructed from `git log` alone.

### The bump must replace **all** occurrences

Never upgrade "the ones you remember". The version string lives in seven tracked
files, plus one conditional, and only two of them are covered by tests:

| File | Expression |
|---|---|
| `pyproject.toml` | `version = "0.1.102"` |
| `src/torchada/__init__.py` | `__version__ = "0.1.102"` |
| `src/torchada/csrc/ops.h` | `constexpr const char* VERSION = "0.1.102";` |
| `tests/test_platform.py` | `assert version == "0.1.102"` (`test_get_version`) |
| `examples/extension_setup.py` | `version="0.1.102"` |
| `README.md` | `torchada>=0.1.102` |
| `README_CN.md` | `torchada>=0.1.102` |
| `benchmarks/benchmark_history.json` | only if the old version appears — relabel the current-release row |

`test_get_version` and `test_project_version_matches_runtime_version` will fail on a
partial bump, but they only cover `pyproject.toml`, `__init__.py`, and the test
itself. A missed `ops.h`, `README.md`, or `examples/extension_setup.py` passes CI and
ships a package whose C++ extension and documentation report a different version than
the wheel. That is why the checklist is `git grep`, not memory.

## Version Bumps

### Versioning scheme

`MAJOR.MINOR.PATCH` with the current line at `0.1.N`. Every PR that changes code,
tests, or packaged data and merges to `main` raises `N` by exactly one
(`0.1.102` -> `0.1.103`). Do not skip numbers, do not reuse a number, and do not leave
a merged PR without a bump — except a documentation-only PR, which ships nothing and
therefore does not bump.

### Procedure

```bash
OLD=0.1.102
NEW=0.1.103

# 1. Finish and commit the work first
git add src/torchada/_patch.py tests/test_thing.py
git commit -m "fix(_patch): ..."

# 2. Raise every carrier in one sweep
sed -i "s/${OLD//./\\.}/$NEW/g" \
  pyproject.toml \
  src/torchada/__init__.py \
  src/torchada/csrc/ops.h \
  tests/test_platform.py \
  examples/extension_setup.py \
  README.md \
  README_CN.md

# 3. Prove nothing is left behind
git grep -nF "$OLD"
python -m pytest tests/test_platform.py -q

# 4. Commit the bump on its own
git add pyproject.toml src/torchada/__init__.py src/torchada/csrc/ops.h \
        tests/test_platform.py examples/extension_setup.py README.md README_CN.md
git commit -m "chore: bump torchada to $NEW"
```

`benchmarks/benchmark_history.json` is conditional: edit it only when the old version
string appears there, and only relabel the row that tracks the current release line.
Never invent or hand-edit measurement numbers — a changed hot path should be measured
with `benchmarks/benchmark_overhead.py --save`, and the new numbers then also need to
be reflected in the performance tables of both READMEs.

### Keep the split when merging

Merge with rebase or a merge commit so `main` keeps the work commit and the bump
commit separate. If the pull request has to be squashed, keep both log entries in the
squash message:

```
* fix(_patch): ...

* chore: bump torchada to 0.1.103
```

## Commit Messages

Use a conventional prefix, optionally scoped to the touched module:

```
fix(_patch): synchronize CUDA and MUSA visible devices
feat(device): register the device factory as the FX device builtin
perf(moe): tune the Nemotron MTP decode row on S5000
docs: refresh the READMEs against recent merged PRs
test: skip the legacy Inductor heuristic copy tests on torch_musa >= 2.11.0.post2
chore: bump torchada to 0.1.102
```

- Imperative mood, lowercase after the colon, no trailing period, aim for <= 72 characters.
- Put the *why* and the evidence in the body: measured numbers, the `torch` /
  `torch_musa` versions involved, the failure that motivated the change.
- Reference the issue when there is one (`Fixes #101`).
- Keep the bump commit message to exactly the bump — nothing else in that commit.

## Adding New Patches

1. Add a patch function in `src/torchada/_patch.py`:

```python
@patch_function
@requires_import("torch_musa")
def _patch_feature_name():
    """Docstring explaining what this patch does and why it is safe."""
    original_func = torch.module.func

    def patched_func(*args, **kwargs):
        # Translation logic
        return original_func(*args, **kwargs)

    torch.module.func = patched_func
```

2. Gate version-dependent behavior on the `torch_musa` version through the shared proxy
   rather than ad-hoc comparisons. An unknown version ranks lowest, so the
   conservative workaround stays armed:

```python
from ._version import version_of

if version_of(getattr(torch.musa, "__version__", None)) < "2.11.0.post2":
    ...  # workaround for everything older than the fix
```

3. Keep hot paths cheap — cache attributes in `__dict__`, keep the string translation
   cache, avoid per-call imports. Overhead is measured in hundreds of nanoseconds.

4. Add tests next to the matching feature area, including the disabled/older-version path.

5. Update documentation in `README.md` **and** `README_CN.md`. If the patch adds a new
   `TORCHADA_*` environment variable or a new invariant, update `AGENTS.md` too.

## Critical Constraints

1. **Never patch** `torch.cuda.is_available()` or `torch.version.cuda` by default — downstream projects use them for platform detection. The only allowed exception is the explicit opt-in switch `TORCHADA_FORCE_CUDA_AVAILABLE`, which redirects `torch.cuda.is_available()` to `torch.musa.is_available()` for unmodifiable CUDA-only projects; an unset or non-truthy value must keep the original behavior
2. **Import order matters**: `import torchada` must come before other torch imports in user code
3. **MUSA tensors use `PrivateUse1` dispatch key**, not `CUDA` — always translate dispatch keys
4. **Keep file ownership 1000:1000** when running formatters in Docker
5. **Version bumps are their own commit** — see [Version Bumps](#version-bumps)

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes (pre-commit hooks will auto-format on commit)
4. Add tests for new functionality
5. Ensure all tests pass
6. Unless this is a documentation-only PR, add the version bump commit
   (`chore: bump torchada to X`) as the final commit
7. Submit a pull request and fill in the checklist below

### Pull request checklist

- [ ] At least two commits: the work, then `chore: bump torchada to <version>` (documentation-only PR: a single `docs:` commit, no bump)
- [ ] `git grep -nF '<old version>'` returns nothing
- [ ] `pre-commit run --all-files` passes
- [ ] Tests pass (`pytest tests/`), or the PR states which subset ran and why
- [ ] `README.md` and `README_CN.md` updated for anything user-visible
- [ ] Benchmarks re-run (`--save`) if a hot path changed, with the README tables updated
- [ ] The PR description names the exact `torch`, `torch_musa`, driver, and hardware versions used for validation
