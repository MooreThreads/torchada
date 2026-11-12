"""
Pytest configuration and fixtures for torchada tests.
"""

import os

import pytest

# Ensure torchada is imported first to apply patches
import torchada

_ABSENT = object()
_WATCHED_ENV = ("MUSA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "musa: mark test as requiring MUSA platform")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA platform")
    config.addinivalue_line("markers", "gpu: mark test as requiring any GPU (CUDA or MUSA)")
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture(autouse=True)
def _no_leaked_patch_state():
    """Fail the test that leaves state a torchada patch wrote behind for the next one.

    ``@patch_function`` bodies assign ``torch.*`` attributes and ``os.environ``
    entries themselves, and ``monkeypatch`` can only undo a binding it owns, so a
    test that arms a patch leaks whatever it did not pin.  The leak is invisible
    where it happens and only bites some later test, which is why the check is
    suite-wide rather than module-local.  Two have cost real debugging time:

    * ``torch.mm`` / ``torch.bmm``.  With ``_original_torch_mm`` back to ``None``,
      the next promoted ``out_dtype`` call sends ``_probe_out_dtype_broken_ops()``
      into the leaked wrapper, which asks the probe for its verdict again and
      deadlocks on the non-reentrant ``_MM_OUT_DTYPE_PROBE_LOCK``.
    * ``MUSA_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES``.  A leaked mask is
      inherited by every subprocess a later test spawns, so tests that compile or
      probe in a child process stop seeing the GPU.

    Autouse fixtures are set up before ``monkeypatch``, so this teardown runs after
    its undo and sees exactly what the next test inherits.  Keep it argument-free:
    requesting ``monkeypatch`` would make this a dependent of it, pytest would then
    set ``monkeypatch`` up first and finalize it last, and the comparison would run
    *before* the undo and flag every test that pins anything.

    Compare the env half by value, not identity - ``os.environ`` decodes a fresh
    ``str`` on every lookup, so ``is`` reports a leak for any mask that is merely
    present.
    """
    import torch

    from torchada import _patch

    def bindings():
        return {
            "torch.mm": torch.mm,
            "torch.bmm": torch.bmm,
            "_original_torch_mm": _patch._original_torch_mm,
            "_original_torch_bmm": _patch._original_torch_bmm,
        }

    def masks():
        return {name: os.environ.get(name, _ABSENT) for name in _WATCHED_ENV}

    bindings_before, masks_before = bindings(), masks()
    yield
    bindings_after, masks_after = bindings(), masks()
    leaked = [name for name, was in bindings_before.items() if bindings_after[name] is not was]
    leaked += [name for name, was in masks_before.items() if masks_after[name] != was]
    assert not leaked, f"left behind for the next test: {', '.join(leaked)}"


@pytest.fixture(scope="session")
def platform():
    """Return the detected platform."""
    return torchada.detect_platform()


@pytest.fixture(scope="session")
def is_musa():
    """Return True if running on MUSA platform."""
    return torchada.is_musa_platform()


@pytest.fixture(scope="session")
def is_cuda():
    """Return True if running on CUDA platform."""
    return torchada.is_cuda_platform()


def _gpu_available():
    """Check if any GPU is actually available (works on both CUDA and MUSA platforms)."""
    import torch

    if torchada.is_musa_platform():
        try:
            return torch.musa.is_available()
        except Exception:
            return False
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def has_gpu():
    """Return True if any GPU is available."""
    return _gpu_available()


@pytest.fixture(scope="function")
def gpu_tensor():
    """Create a GPU tensor fixture."""
    import torch

    if _gpu_available():
        return torch.randn(10, 10, device="cuda")
    else:
        pytest.skip("No GPU available")


@pytest.fixture(scope="function")
def cpu_tensor():
    """Create a CPU tensor fixture."""
    import torch

    return torch.randn(10, 10)


def pytest_collection_modifyitems(config, items):
    """Skip tests based on platform markers."""
    skip_musa = pytest.mark.skip(reason="MUSA platform required")
    skip_cuda = pytest.mark.skip(reason="CUDA platform required")
    skip_gpu = pytest.mark.skip(reason="GPU required")

    for item in items:
        if "musa" in item.keywords and not torchada.is_musa_platform():
            item.add_marker(skip_musa)
        if "cuda" in item.keywords and not torchada.is_cuda_platform():
            item.add_marker(skip_cuda)
        if "gpu" in item.keywords and not _gpu_available():
            item.add_marker(skip_gpu)
