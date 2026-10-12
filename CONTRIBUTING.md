# Contributing to torchada

Thank you for your interest in contributing to torchada! This guide will help you get started.

## Development Setup

### 1. Clone and Install

```bash
git clone https://github.com/MooreThreads/torchada.git
cd torchada
pip install -e ".[dev]"
```

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
- **Check** for common issues (trailing whitespace, YAML/TOML syntax, merge conflicts, debug statements, etc.)
- **Spell-check** with [codespell](https://github.com/codespell-project/codespell)

If a hook modifies files (e.g., reformatting), the commit will be aborted. Simply `git add` the changes and commit again.

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

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test class
pytest tests/test_cuda_patching.py::TestLibraryImpl -v

# Run with short traceback
pytest tests/ --tb=short
```

### Test Markers

- `@pytest.mark.musa` — Requires MUSA platform
- `@pytest.mark.cuda` — Requires CUDA platform
- `@pytest.mark.gpu` — Requires any GPU
- `@pytest.mark.slow` — Slow tests

### Docker Testing

Tests should pass on both torch_musa versions:

```bash
docker exec -w /ws yeahdongcn python -m pytest tests/ --tb=short   # torch_musa 2.7.0
docker exec -w /ws yeahdongcn1 python -m pytest tests/ --tb=short  # torch_musa 2.7.1
```

## Adding New Patches

1. Add patch function in `src/torchada/_patch.py`:

```python
@patch_function
@requires_import("torch_musa")
def _patch_feature_name():
    """Docstring explaining what this patch does."""
    original_func = torch.module.func
    def patched_func(*args, **kwargs):
        # Translation logic
        return original_func(*args, **kwargs)
    torch.module.func = patched_func
```

2. Add tests in `tests/test_cuda_patching.py`
3. Update documentation in `README.md` and `README_CN.md`

## Critical Constraints

1. **Never patch** `torch.cuda.is_available()` or `torch.version.cuda` by default — downstream projects use these for platform detection. The only allowed exception is the explicit opt-in switch `TORCHADA_FORCE_CUDA_AVAILABLE`, which redirects `torch.cuda.is_available()` to `torch.musa.is_available()` for unmodifiable CUDA-only projects; an unset or non-truthy value must keep the original behavior
2. **Import order matters**: `import torchada` must come before other torch imports in user code
3. **MUSA tensors use `PrivateUse1` dispatch key**, not `CUDA` — always translate dispatch keys

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes (pre-commit hooks will auto-format on commit)
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request
