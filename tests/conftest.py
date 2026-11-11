"""
Pytest configuration and fixtures for torchada tests.
"""

import pytest

# Ensure torchada is imported first to apply patches
import torchada


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "musa: mark test as requiring MUSA platform")
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA platform")
    config.addinivalue_line("markers", "gpu: mark test as requiring any GPU (CUDA or MUSA)")
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture(autouse=True)
def _no_leaked_patch_bindings():
    """Fail the test that leaves torchada's ``out_dtype`` backport bindings behind.

    ``@patch_function`` bodies assign ``torch.*`` attributes themselves and
    ``monkeypatch`` can only undo an assignment it made, so a test that arms
    ``_patch._patch_mm_out_dtype()`` leaks a wrapper unless it pins every op the
    patch touches.  The leak is invisible in the test that causes it and lethal
    later: with ``_original_torch_mm`` back to ``None``, the next promoted
    ``out_dtype`` call sends the probe into the leaked wrapper, which asks the
    probe for its verdict again and deadlocks on the non-reentrant
    ``_MM_OUT_DTYPE_PROBE_LOCK``.  Hence a suite-wide check rather than one
    module's: autouse fixtures are set up before ``monkeypatch``, so this
    teardown runs after its undo and sees what the next test inherits.

    Keep it argument-free.  Requesting ``monkeypatch`` here would make this a
    dependent of it, so pytest would set ``monkeypatch`` up first and finalize it
    last - the comparison would then run *before* the undo and flag every test
    that pins a binding.
    """
    import torch

    from torchada import _patch

    names = ("torch.mm", "torch.bmm", "_original_torch_mm", "_original_torch_bmm")

    def bindings():
        return (torch.mm, torch.bmm, _patch._original_torch_mm, _patch._original_torch_bmm)

    before = bindings()
    yield
    leaked = [name for name, was, now in zip(names, before, bindings()) if was is not now]
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
