"""End-to-end test: the torchada libtorch-stable ABI shim is generally usable.

Builds tests/csrc/stable_abi_ops.cu through torchada (cuda->musa porting + the
shipped stable_compat headers) and runs the ops on MUSA, exercising diverse
stable-ABI signature shapes (void / Tensor / int returns; Tensor + scalar args;
THO_DISPATCH over float/half/bfloat16). This is the runtime counterpart to the
pure-Python rules in test_stable_abi_mappings.py.
"""

import glob
import os
import shutil
import subprocess
import sys

import pytest

import torchada  # noqa: F401  apply MUSA patches first
from torchada.utils.cpp_extension import (
    stable_compat_box_header,
    stable_compat_include_dir,
)

CSRC = os.path.join(os.path.dirname(__file__), "csrc")
OPS_CU = os.path.join(CSRC, "stable_abi_ops.cu")


def _gpu_available() -> bool:
    import torch

    return hasattr(torch, "musa") and torch.musa.is_available()


_SETUP_TEMPLATE = """\
import os
import pathlib
import torchada  # noqa: F401
import torchada.utils.cpp_extension as cpp_extension
os.environ["TORCHADA_EXCLUDE_DIRS"] = {exclude_dir!r}

if {expect_backport!r}:
    _original_backport = cpp_extension._ensure_stable_headers_patched
    def _record_backport():
        pathlib.Path("backport-called").write_text("1")
        _original_backport()
    cpp_extension._ensure_stable_headers_patched = _record_backport
else:
    def _unexpected_backport():
        raise AssertionError("torchada patched stable headers on torch >= 2.11")
    cpp_extension._ensure_stable_headers_patched = _unexpected_backport

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="torchada_stable_test",
    ext_modules=[CUDAExtension(
        name="torchada_stable_test",
        sources=["stable_abi_ops.cu"],
        include_dirs={include_dirs!r},
        extra_compile_args={{
            "nvcc": {nvcc_args!r},
            "cxx": {cxx_args!r},
        }},
        # AOTI stable-ABI shims live in libtorch_cpu.so.
        libraries=["c10", "torch", "torch_cpu", "torch_python", "musart"],
    )],
    cmdclass={{"build_ext": BuildExtension}},
)
"""


def _run_stable_abi_shim_test(tmp_path, expect_backport, use_compat_headers):
    import torch

    shutil.copy(OPS_CU, tmp_path)
    (tmp_path / "setup.py").write_text(
        _SETUP_TEMPLATE.format(
            expect_backport=expect_backport,
            exclude_dir=stable_compat_include_dir(),
            include_dirs=[stable_compat_include_dir()] if use_compat_headers else [],
            nvcc_args=["-DUSE_MUSA"] + ([
                "-DCUDA_VERSION=0", "-DENABLE_FP8", "-include",
                stable_compat_box_header()
            ] if use_compat_headers else []),
            cxx_args=["force_mcc", "-include", stable_compat_box_header()]
            if use_compat_headers else ["force_mcc"],
        ))

    res = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, f"build failed:\n{res.stderr[-4000:]}"
    marker = tmp_path / "backport-called"
    assert marker.exists() is expect_backport

    so = glob.glob(str(tmp_path / "torchada_stable_test*.so"))
    assert so, "extension .so not produced"
    torch.ops.load_library(so[0])

    ns = torch.ops.torchada_stable_test
    dev = "cuda"  # torchada -> musa

    for dt in (torch.float32, torch.float16, torch.bfloat16):
        # bf16 carries ~3 significant digits; scale-by-3 amplifies its rounding.
        tol = 1e-1 if dt == torch.bfloat16 else 1e-2
        x = torch.randn(128, device=dev, dtype=dt)
        # (1) void return + two Tensor args + THO_DISPATCH (negation is exact)
        out = torch.empty_like(x)
        ns.negate(out, x)
        torch.musa.synchronize()
        assert torch.allclose(out.float(), -x.float(), atol=1e-2), f"negate {dt}"
        # (2) void return + Tensor args + double scalar arg
        out2 = torch.empty_like(x)
        ns.scale(out2, x, 3.0)
        torch.musa.synchronize()
        assert torch.allclose(out2.float(), x.float() * 3.0, atol=tol, rtol=tol), \
            f"scale {dt}"

    # (3) single Tensor return  -> from<Tensor>
    x = torch.randn(8, device=dev)
    assert ns.passthrough(x).data_ptr() == x.data_ptr()
    # (4) scalar int return     -> from<int64_t>
    assert int(ns.numel_of(x)) == 8

    x = torch.arange(24, device=dev).reshape(4, 6)[:, ::2]
    alias = ns.weak_ref_tensor(x)
    assert alias.data_ptr() == x.data_ptr()
    assert alias.shape == x.shape
    assert alias.stride() == x.stride()
    assert alias.dtype == x.dtype
    alias.add_(1)
    assert torch.equal(alias, x)


def _torch_minor() -> int:
    import torch

    return int(torch.__version__.split(".")[1])


@pytest.mark.musa
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _gpu_available(), reason="requires a MUSA GPU")
@pytest.mark.skipif(_torch_minor() != 9, reason="requires torch 2.9")
def test_stable_abi_ops_with_torch29_backport(tmp_path):
    _run_stable_abi_shim_test(tmp_path, expect_backport=True, use_compat_headers=True)


@pytest.mark.musa
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _gpu_available(), reason="requires a MUSA GPU")
@pytest.mark.skipif(_torch_minor() < 11, reason="requires torch 2.11 or newer")
def test_stable_abi_ops_with_native_torch211_abi(tmp_path):
    _run_stable_abi_shim_test(tmp_path, expect_backport=False, use_compat_headers=False)
