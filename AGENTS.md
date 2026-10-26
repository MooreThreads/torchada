# AGENTS.md

Working instructions for AI agents (and humans) on **torchada**. `CLAUDE.md` only
imports this file; the workspace-level `../AGENTS.md` adds the multi-repository
contract on top. Most PRs in this repository are authored by agents, so this file
is the primary place to record a convention instead of leaving it in a PR comment.

Everything below was verified against `origin/main` at `0.1.102`. Patch files move
fast: re-verify a version gate, a file path, or an env-var name before relying on
it, and update this file in the same PR that invalidates it.

## Project Overview

**torchada** is a CUDA-to-MUSA compatibility adapter for PyTorch on Moore Threads
GPUs. It enables PyTorch code written for NVIDIA CUDA GPUs to run on Moore Threads
MUSA GPUs without code changes.

**Key Design Principle**: Developers import `torchada` once, then use standard
`torch.cuda.*`, `torch.utils.cpp_extension.*` and Triton APIs normally. All MUSA
handling is invisible to users; anything user-visible must be documented in
`README.md` **and** `README_CN.md`.

Two boundaries are deliberate and must not be "fixed":

- `torch.cuda.is_available()` stays `False` on MUSA by default, and
  `torch.version.cuda` is never patched, so downstream projects can keep telling
  the two platforms apart. The only exception is the explicit opt-in
  `TORCHADA_FORCE_CUDA_AVAILABLE=1`.
- `import torchada` must come before torch-ecosystem imports that capture CUDA
  symbols (`torch`, `torch_musa`, `vllm`, `sglang`, ...). A patch that only works
  if imported later is not acceptable.

## Repository Layout

```
src/torchada/
├── __init__.py                 # Public API; bootstrap -> load_cpp_ops() -> apply_patches()
│                               #   -> set_default_moe_config_dir()
├── _patch.py                   # @patch_function registry, ~2.9k lines: the patches themselves
├── _platform.py                # Platform enum, is_musa/cuda/cpu_platform(), device helpers
├── _mapping.py                 # 16-line aggregator re-exporting MAPPING_RULE / EXT_REPLACED_MAPPING
├── _mappings/                  # 20 per-domain rule modules: aten, c10, compiler, cublas, cudnn,
│                               #   cuda_runtime, cuda_driver, cuda_arch, cufft, curand, cusolver,
│                               #   cusparse, cutlass, thrust, nccl, nvjpeg, flashinfer,
│                               #   libtorch_stable, data_types, device
├── _runtime.py                 # ctypes symbol translation (cuda_to_musa_name, nccl_to_mccl_name, ...)
├── _cpp_ops.py                 # JIT loader for the ATen-override and graph-rotation aux extensions
├── _graph_rotation.py          # transparent CUDA-graph executable rotation (driver caps live execs)
├── _graph_rotation_src/        # C++ source of the graph-rotation aux extension
├── _visible_devices.py         # MUSA/CUDA_VISIBLE_DEVICES mirroring, before torch is imported
├── _version.py                 # version_of(): infix PEP 440 proxy used by every version gate
├── csrc/                       # ops.h / ops.cpp / musa_ops.mu / rope.mu + stable_compat/ headers
├── cuda/                       # CUDA-compatible submodules: amp, nvtx, random
├── utils/cpp_extension.py      # CUDAExtension / BuildExtension / CUDA_HOME + in-place source
│                               #   porting via SimplePorting (~1.5k lines)
└── triton/                     # vLLM/SGLang Triton-Fused-MoE compat, autotune tooling,
│                               #   bundled per-shape S5000 configs

tests/       29 test modules, one per feature area (see Testing)
docs/        custom_musa_ops.md, tune_triton_moe.md
examples/    basic_usage.py, model_training.py, extension_setup.py, profiler_demo.py,
             migrate_existing_project.md
benchmarks/  benchmark_overhead.py, benchmark_history.json
scripts/     publish.sh (build + twine upload to PyPI/TestPyPI)
```

## Architecture

Import order in `src/torchada/__init__.py` is load-bearing; keep new bootstrap work
before `import torch` and new patches inside a `@patch_function`:

1. **Visible-device bootstrap** (`_visible_devices.py`) — mirrors
   `MUSA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` before either runtime
   initializes, because a MUSA build can initialize its runtime while extension
   modules are imported. Honor `TORCHADA_PLATFORM=cuda|cpu` as an opt-out.
2. **Platform detection** (`_platform.py`) — `is_musa_platform()`,
   `is_cuda_platform()`, `is_cpu_platform()`, `is_gpu_device()`,
   `is_cuda_like_device()`. Never re-implement these checks locally.
3. **Patching** (`_patch.py`) — `@patch_function`-decorated functions collected and
   applied by `apply_patches()`; `is_patched()` reports state.
4. **Source porting for extensions** (`utils/cpp_extension.py` + `_mappings/`) —
   `SimplePorting` rewrites CUDA sources in place (no `<dir>_musa` mirror) using
   `MAPPING_RULE`; `.cu`/`.cuh` compile as MUSA by selecting the `-x musa` rule in
   `_is_musa_file`, so renaming to `.mu`/`.muh` is unnecessary.
5. **Runtime symbol translation** (`_runtime.py`) — patched `ctypes.CDLL` maps CUDA
   names onto musa* libraries, so `libmusart.so.cudaIpcOpenMemHandle` works.
6. **C++ operator overrides** (`_cpp_ops.py` + `csrc/`) — `TORCH_LIBRARY_IMPL(aten,
   PrivateUse1, m)` implementations JIT-built on first use into torch's extension
   build directory (`TORCH_EXTENSIONS_DIR` respected).
7. **CUDA-graph rotation** (`_graph_rotation.py`) — the MUSA driver caps live
   `musaGraphExec_t` per process (~2048); rotation keeps every graph *template*
   alive, LRU-caps live *executables* (default 1900) and re-instantiates on replay.
   Zero cost until the cap is exceeded.
8. **Version gating** (`_version.py`) — write gates as
   `version_of(getattr(torch.musa, "__version__", None)) < "2.11.0.post2"`.

MUSA tensors dispatch to `PrivateUse1`, not `CUDA`: every dispatch key, device
string, or factory kwarg crossing the boundary must be translated.

## Build and Install

```bash
# Development install
pip install -e .

# With dev dependencies (pytest, pytest-cov, black, isort, mypy, pre-commit)
pip install -e ".[dev]"

# Release build (no upload)
./scripts/publish.sh --build
```

Python floor is 3.8 (`requires-python` in `pyproject.toml`); a new syntax or stdlib
dependency must not break 3.8. `setup.py` is only an editable-install shim —
package metadata lives in `pyproject.toml`.

## Code Style

- **Formatter**: black, line-length 100 (pinned 24.10.0 in pre-commit)
- **Import sorting**: isort, black profile, line-length 100
- **Lint**: ruff `--select=F401,F821` over `src/ tests/ examples/ benchmarks/`
  (`__init__.py` files are excluded because re-exports are the point)
- **Spelling**: codespell (`musa`, `musa_`, `musas` are allowed words)
- **Python version**: >=3.8

```bash
# One-time hook setup — only with the repository/workspace owner's agreement,
# since the hooks run formatters that rewrite staged files
pre-commit install

# Run everything, or one hook
pre-commit run --all-files
pre-commit run black --all-files

# Manual formatting
isort src/ tests/ && black src/ tests/

# In Docker containers, preserve file ownership
docker exec -w /ws <container> bash -c "isort src/ tests/ && black src/ tests/ && chown -R 1000:1000 src/ tests/"
```

Hooks also run `no-commit-to-branch`, so `main` cannot be committed to directly.
If a hook rewrites files, the commit aborts: `git add` the result and commit again.
Never bypass with `--no-verify`. If a formatter only fires while you are making the
bump commit, put the reformat in its own commit *ahead of* the bump, or fold it into a
work commit you are still composing. Never rewrite an existing commit to absorb it, and
keep the bump version-only.

## Testing

```bash
# Full suite
pytest tests/

# One area / one test class
pytest tests/test_cuda_patching.py::TestLibraryImpl -v
pytest tests/test_mappings.py tests/test_inplace_porting.py -q

# Version metadata gates (no GPU needed)
pytest tests/test_platform.py -q
```

**Test Markers** (defined in `tests/conftest.py`):

- `@pytest.mark.musa` — requires MUSA platform
- `@pytest.mark.cuda` — requires CUDA platform
- `@pytest.mark.gpu` — requires any GPU
- `@pytest.mark.slow` — slow test

`conftest.py` imports `torchada` before pytest collects anything, so patches are
active in every test. Skip by marker, never by deleting a test.

**Where a new test goes** — one module per feature area already exists; add to it
instead of creating `test_misc.py`:

| Area | Module |
|---|---|
| General patching | `test_cuda_patching.py` |
| Mapping rules / porting | `test_mappings.py`, `test_inplace_porting.py`, `test_stable_abi_mappings.py` |
| Extensions | `test_cpp_extension.py`, `test_extension_build.py`, `test_cpp_ops_guard.py` |
| Device translation | `test_device_strings.py`, `test_fx_device_builtin.py`, `test_infer_schema_device.py`, `test_tensor_cuda_torch_function_mode.py` |
| Version gates / version metadata | `test_version.py`, `test_platform.py` |
| Opt-in CUDA availability | `test_cuda_availability_optin.py` |
| Graphs / rotation | `test_graph_rotation.py` |
| Triton MoE | `test_triton_patching.py`, `test_triton_descriptor_capability.py`, `test_moe_config_dir.py`, `test_moe_pipeline.py`, `test_tune_moe_recipe.py`, `test_nemotron_moe_config.py` |

**Docker containers for testing**: run the suite in the container that belongs to
your own task or GPU lease — `<your-container>` below — never in one you did not
create. The MUSA development host still carries older `yeahdongcn*` containers from
earlier sessions; treat them as someone else's: do not run tests in them, do not
rewrite their files, do not clean them up. For an external checkout, any environment
carrying the `torch_musa` version you are gating against works.

```bash
docker exec -w /ws <your-container> python -m pytest tests/ --tb=short
```

Primary validation target is the newest `torch_musa` release the code gates
against; when a change touches a version gate, test both sides of the bound.

## Version Bumps: a PR needs at least two commits

**Every PR that changes code, tests, or packaged data contains at least two
commits: the work, then the bump.** Reviewers must be able to diff the behavior
change and the version bump separately. A PR that mixes them in one commit will be
asked to split.

A **documentation-only PR** — a diff that touches nothing but Markdown and the
issue/PR templates (`AGENTS.md`, `CONTRIBUTING.md`, `README.md`, `README_CN.md`,
`docs/`, `examples/*.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/`)
— carries **no bump commit** and **does not change the version**. The bump marks what
ships; a docs-only change ships nothing. It may still be several `docs:` commits (a
refresh, a correction, a template), normally squashed into one entry on merge. Workflows
under `.github/workflows/` are automation, not documentation, and are **not** exempt.

```
<work commit(s)>                       # feat/fix/perf/docs/test: what the PR is about
chore: bump torchada to <new version>  # exactly the version bump, nothing else
```

Rules:

1. **One bump per PR, exactly one patch increment.** Current line is `0.1.N`;
   `0.1.102` -> `0.1.103`. Never skip a number, never reuse one, and never bump
   inside a work commit.
2. **Replace every carrier, never a subset.** Partial bumps have shipped before and
   leave the C++ extension or the docs reporting an older version than the wheel. The
   scoped `git grep` below is the gate, not your memory of the file list.
3. **The bump commit touches nothing else.** No source fixes, no formatting
   drive-bys, no test changes riding along with it.
4. **A bump-only PR is not a feature PR.** Release management owns those; do not
   open one unless asked.
5. **Land the two commits separately.** Prefer rebase/merge over squash so `main`
   keeps the split. If the PR must be squashed, keep both entries in the squash
   body (`* feat(...): ...`, `* chore: bump torchada to X`).
6. **Documentation-only PRs skip the bump** — no `chore:` commit at all.
   The moment the same PR also touches source, tests, `pyproject.toml`, a bundled
   training/tuning config, or a workflow, it is a normal PR and needs both commits.

**The bump surface** — seven carriers, with `<version>` standing for the value
currently in `pyproject.toml`. The placeholders are deliberate: this file and
`CONTRIBUTING.md` are **not** version carriers, so a bump must not rewrite the
examples below.

| File | Expression |
|---|---|
| `pyproject.toml` | `version = "<version>"` |
| `src/torchada/__init__.py` | `__version__ = "<version>"` |
| `src/torchada/csrc/ops.h` | `constexpr const char* VERSION = "<version>";` |
| `tests/test_platform.py` | `assert version == "<version>"` (in `test_get_version`) |
| `examples/extension_setup.py` | `version="<version>"` |
| `README.md` | `torchada>=<version>` |
| `README_CN.md` | `torchada>=<version>` |

Only two of those are test-enforced — `test_get_version` pins the string and
`test_project_version_matches_runtime_version` compares `pyproject.toml` against
`__version__`. `ops.h`, both READMEs, and `examples/extension_setup.py` are not
gated, which is exactly how a partial bump survives review.

Bump procedure:

```bash
OLD=0.1.102
NEW=0.1.103

# 1. Work first: commit the actual change
git add src/torchada/_patch.py tests/test_thing.py
git commit -m "fix(_patch): ..."

# 2. Then the bump, with every carrier at once
sed -i "s/${OLD//./\\.}/$NEW/g" \
  pyproject.toml \
  src/torchada/__init__.py \
  src/torchada/csrc/ops.h \
  tests/test_platform.py \
  examples/extension_setup.py \
  README.md \
  README_CN.md

# 3. Verify: no stale reference left in any carrier
git grep -nF "$OLD" -- pyproject.toml src/torchada/__init__.py \
    src/torchada/csrc/ops.h tests/test_platform.py \
    examples/extension_setup.py README.md README_CN.md
# No output expected. Scope matters: this file and CONTRIBUTING.md quote versions as
# examples, so an unscoped `git grep -nF "$OLD"` always hits them and proves nothing.
python -m pytest tests/test_platform.py -q

# 4. Commit only what the bump touched
git add pyproject.toml src/torchada/__init__.py src/torchada/csrc/ops.h \
        tests/test_platform.py examples/extension_setup.py README.md README_CN.md
git commit -m "chore: bump torchada to $NEW"
```

`benchmarks/benchmark_history.json` is **not** a bump carrier. It is written by
`benchmarks/benchmark_overhead.py --save`, which keys rows on `(version, platform)`:
it appends a row for a version that has none and replaces the row of the version it
runs on. Never hand-edit or relabel a row — relabelling asserts that a measurement
taken on the old release was taken on the new one, and the file's value is that it
records version, date, and stack together. If you did not run `--save`, the file does
not change and stays out of the bump commit.

## Commit Messages

The history mixes conventional prefixes; use them instead of bespoke prose:

```
fix(_patch): synchronize CUDA and MUSA visible devices
feat(device): register the device factory as the FX device builtin
perf(moe): tune the Nemotron MTP decode row on S5000
docs: refresh the READMEs against recent merged PRs
test: skip the legacy Inductor heuristic copy tests on torch_musa >= 2.11.0.post2
chore: bump torchada to 0.1.102
```

- `type(scope):` where scope names the touched module (`_patch`, `_mappings`,
  `device`, `musa`, `cpp_extension`, `cpp_ops`, `fused_moe`, `cuda`) or is omitted
  for repo-wide changes.
- Imperative, lowercase after the colon, no trailing period, aim for <= 72 chars.
- The body carries the *why* and the evidence: measured numbers, the
  `torch_musa`/`torch` versions involved, the failure that motivated the fix.
- Reference the issue when one exists (`Fixes #101`).

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

2. Gate on the `torch_musa` version with `version_of()` instead of ad-hoc
   comparisons or per-release helper predicates. An unknown or unparsable version
   ranks lowest, so write "workaround while older than the fix":

```python
from ._version import version_of

if version_of(getattr(torch.musa, "__version__", None)) < "2.11.0.post2":
    ...  # workaround for everything older than the fix
```

3. Keep the patch cheap: cache attribute lookups in `__dict__`, keep the string
   translation cache, and avoid per-call imports or `hasattr` chains on hot paths
   (`torch.cuda.Stream`, `Event`, `device_count`, `_translate_device`).
4. Add tests next to the matching feature area (table under Testing) and cover the
   disabled path too — a gate that arms on an unknown version is a real case.
5. Update `README.md` **and** `README_CN.md`. If the patch introduces a new
   invariant or a new `TORCHADA_*` switch, update this file as well.

## Adding a Mapping Rule

`_mapping.py` is only the aggregator; rules live in `src/torchada/_mappings/`:

1. Add the entry to the domain module that owns the symbol (`aten.py`, `c10.py`,
   `cublas.py`, `cuda_runtime.py`, `libtorch_stable.py`, ...). Domains are disjoint,
   so order inside `MAPPING_RULE` is irrelevant — `SimplePorting` sorts by key
   length.
2. Keep CUDA->MUSA `#define`s out of in-place porting when the mapping would
   change compile-time behavior for code that also compiles on CUDA.
3. Cover it in `tests/test_mappings.py` (or `test_stable_abi_mappings.py` for the
   stable-ABI surface) and, for extension-visible rules, in
   `tests/test_extension_build.py`.
4. Mention the domain in `README.md` / `README_CN.md` only when the rule set grows
   by a whole new API family.

## C++ Operator Overrides

torchada can override ATen operators at C++ level for correctness or performance.
See [docs/custom_musa_ops.md](docs/custom_musa_ops.md).

The extension is loaded automatically on MUSA when torchada is imported and
provides CUDA-compatible entry points including the memory-pool functions
(`_cuda_beginAllocateCurrentThreadToPool`, `_cuda_endAllocateToPool`,
`_cuda_releasePool`) that get injected into `torch.cuda.memory`.

**Adding an override**:

1. Edit `src/torchada/csrc/musa_ops.mu` for MUSA kernels (or `ops.cpp` for pure C++).
2. Register with `TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)`.
3. The extension is JIT-compiled on first use; stale build locks are recovered
   automatically — do not "fix" a lock by deleting the cache manually in CI.
4. Add a guard test next to `tests/test_cpp_ops_guard.py`.

Stable-ABI shims live in `src/torchada/csrc/stable_compat/` and must stay
header-only additions; they exist so kernels built against `libtorch_stable`
compile on MUSA without patching the dependency headers in place.

## Environment Variables

`README.md#environment-variables` is the canonical table; keep it and this list in
sync when a switch is added.

| Variable | Default | Effect |
|---|---|---|
| `TORCHADA_PLATFORM` | auto-detect | Force platform: `cuda`, `musa`, or `cpu` |
| `TORCHADA_FORCE_CUDA_AVAILABLE` | unset | `1` makes `torch.cuda.is_available()` report `torch.musa.is_available()` |
| `TORCHADA_EXCLUDE_DIRS` | unset | Extra include roots excluded from in-place source porting |
| `TORCHADA_GRAPH_ROTATION` | `1` | `0` disables CUDA-graph executable rotation |
| `TORCHADA_GRAPH_EXEC_CAP` | `1900` | Live graph-executable cap per process |
| `TORCHADA_GRAPH_AUTOPROBE` | `0` | `1` probes the driver's real executable cap at startup |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `128` | Margin kept below the probed cap |
| `TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH` | unset | Directory for MUSA graph `.dot` dumps |
| `TORCHADA_CPP_OPS_VERBOSE` | `0` | `1` prints the C++ operator-override build log |
| `TORCHADA_DEBUG_CPP_OPS` | `0` | `1` logs operator-override calls |
| `TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME>` | unset | `1` disables one override, e.g. `TORCHADA_DISABLE_OP_OVERRIDE_log=1` |

`TORCHADA_STABLE_ACCESSORS`, `TORCHADA_HAS_NATIVE_STABLE_DEVICE`,
`TORCHADA_NV_BFLOAT16_ALIASED`, `TORCHADA_HAVE_AOTI_CUDA_STREAM`,
`TORCHADA_PROBE_LIMIT`, and `TORCHADA_TUNE_USE_GRAPH` are internal probe
overrides written by build/probe code — do not document them as user knobs.
`MTGPU_TARGET=mp_XX` overrides the GPU architecture (normally auto-detected via
`musaInfo`).

## Performance Benchmarking

Overhead here is measured in hundreds of nanoseconds, so a "small" regression on a
fast path is a real regression. Numbers are checked in and cited by both READMEs.

```bash
# Print only
docker exec -w /ws <your-container> python benchmarks/benchmark_overhead.py

# Record a row in benchmarks/benchmark_history.json (appends or updates your version's row)
docker exec -w /ws <your-container> python benchmarks/benchmark_overhead.py --save
```

- Fast operations (<200ns): `torch.cuda.device_count()`, `torch.cuda.Stream`,
  `torch.cuda.Event`, `_translate_device()`, `torch.backends.cuda.is_built()`.
- Medium operations (200-800ns): inherent costs (runtime calls, object creation)
  that cannot be optimized further without changing behavior.

Run benchmarks when adding a patch to a hot path, when optimizing one, and before a
release (with `--save`). When the recorded numbers change, update the performance
table in `README.md` and `README_CN.md` in the same PR.

Techniques already in use, keep using them: attribute caching in `__dict__`,
platform-check caching (`_is_musa_platform_cached`), string translation caching
(`_device_str_cache`), closure variable caching in wrappers.

## Critical Constraints

1. **Never patch** `torch.cuda.is_available()` or `torch.version.cuda` by default —
   downstream projects use them for platform detection. The only allowed exception
   is the explicit opt-in `TORCHADA_FORCE_CUDA_AVAILABLE`; an unset or non-truthy
   value must keep the original behavior.
2. **Import order matters**: `import torchada` must come before other torch imports
   in user code, and bootstrap work (`_visible_devices`) must stay before
   `import torch`.
3. **MUSA tensors use `PrivateUse1` dispatch**, not `CUDA` — always translate
   dispatch keys, device strings, and factory `device=` kwargs.
4. **Never invent versions in gates.** Use `version_of()`; an unknown version must
   keep the conservative (workaround-enabled) branch armed.
5. **Do not require a full `torch_musa` rebuild** for a torchada change; graph
   rotation and C++ ops build small aux extensions against installed headers.
6. **Keep file ownership 1000:1000** when running formatters in Docker.
7. **Version bumps are their own commit** — see "Version Bumps" above.

## Side Effects and Security

- Patches are applied at import time via `apply_patches()`; they affect torch APIs,
  not system resources. No network access at import.
- Import-time filesystem writes are limited to:
  - `set_default_moe_config_dir()` — best-effort, idempotent space-free filename
    aliases (symlink, or atomic copy) for the bundled MoE tuning configs inside
    torchada's own installed config directory, so vLLM's `VLLM_TUNED_CONFIG_FOLDER`
    lookup resolves. Skipped silently when the install location is read-only.
  - JIT-built aux extensions (`torchada_cpp_ops`, `torchada_graph_rotation_ops`)
    into torch's extension build directory, honoring `TORCH_EXTENSIONS_DIR`.
- Extension building uses standard torch/setuptools mechanisms; sources are ported
  in place, never mirrored into the user's tree.
- Do not add telemetry, background threads, or hooks that outlive import.

## Definition of Done

Before claiming a change is ready:

- [ ] Work commit(s) and one `chore: bump torchada to X` commit, in that order
      (a docs-only PR has no bump commit at all).
- [ ] `git grep -nF '<old version>' -- <the seven carriers>` returns nothing.
- [ ] `pre-commit run --all-files` clean (no `--no-verify`).
- [ ] `pytest tests/ -q` passing, or a named subset plus the reason the rest is not run.
- [ ] `README.md` **and** `README_CN.md` updated for anything user-visible.
- [ ] This file updated if a convention, invariant, path, or env var changed.
- [ ] Benchmarks re-run (with `--save`) when a hot path changed.
- [ ] Evidence states exact `torch`, `torch_musa`, driver, and hardware versions,
      the commands run, and the observed output.
