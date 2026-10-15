"""Tests for the CUDA->MUSA ``torch._C`` export aliases.

``torchada`` reports ``torch.backends.cuda.is_built()`` as True on MUSA, which
makes ``torch._inductor.cudagraph_trees`` take the branch that imports
CUDA-named symbols from ``torch._C``. Those symbols do not exist in the MUSA
build, so ``_patch_torch_c_exports`` aliases each of them to its real MUSA
counterpart from ``torch_musa._MUSAC``. Without the alias,
``torch.compiler.cudagraph_mark_step_begin()`` fails even when compilation
itself is disabled.

The import-chain tests below run in fresh subprocesses: module import side
effects (``torch._C`` patching, ``sys.modules`` caching) do not reset within
one process, so in-process reload would give false results.
"""

import subprocess
import sys

import pytest
import torch

import torchada  # noqa: F401  (apply patches on import)


def _musa_available() -> bool:
    musa = getattr(torch, "musa", None)
    return musa is not None and musa.is_available()


def _require_musa() -> None:
    if not _musa_available():
        pytest.skip("MUSA device required")


def _require_torch_musa():
    torch_musa = pytest.importorskip("torch_musa")
    if not hasattr(torch_musa, "_MUSAC"):
        pytest.skip("torch_musa._MUSAC required")
    return torch_musa


def _run_fresh(code: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-u", "-c", code],
        capture_output=True, text=True, timeout=timeout,
    )


class TestCExportsAlias:
    def test_cuda_allocator_state_available(self):
        """The CUDA-named symbol is either native or aliased to MUSA."""
        torch_musa = _require_torch_musa()
        if _musa_available():
            assert hasattr(torch._C, "_cuda_CUDAAllocator_AllocatorState")
            assert (
                torch._C._cuda_CUDAAllocator_AllocatorState
                is torch_musa._MUSAC._musa_MUSAAllocator_AllocatorState
            )
        else:
            assert hasattr(torch._C, "_cuda_CUDAAllocator_AllocatorState") or hasattr(
                torch_musa._MUSAC, "_musa_MUSAAllocator_AllocatorState"
            )


class TestInductorImportChain:
    """End-to-end import chain with torchada applied (needs MUSA hardware)."""

    def test_cudagraph_trees_import_with_torchada(self):
        _require_musa()
        proc = _run_fresh(
            "import torchada;"
            "from torch._inductor import cudagraph_trees;"
            "print('PASS')"
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.strip().splitlines()[-1] == "PASS"

    def test_cudagraph_mark_step_begin_with_torchada(self):
        _require_musa()
        proc = _run_fresh(
            "import torchada;"
            "from torch.compiler import cudagraph_mark_step_begin;"
            "cudagraph_mark_step_begin();"
            "print('PASS')"
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.strip().splitlines()[-1] == "PASS"

    def test_compile_with_torchada(self):
        _require_musa()
        proc = _run_fresh(
            "import torchada;"
            "import torch, torch_musa;"
            "torch.musa.set_device(0);"
            "m = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(),"
            "torch.nn.Linear(64, 64)).to('musa');"
            "y = torch.compile(m)(torch.randn(8, 64, device='musa'));"
            "torch.musa.synchronize();"
            "print('PASS', tuple(y.shape))"
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert proc.stdout.strip().splitlines()[-1] == "PASS (8, 64)"
