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
import torchada  # noqa: F401
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="torchada_stable_test",
    ext_modules=[CUDAExtension(
        name="torchada_stable_test",
        sources=["stable_abi_ops.cu"],
        include_dirs=[{inc!r}],
        extra_compile_args={{
            "nvcc": ["-DCUDA_VERSION=0", "-DENABLE_FP8", "-include", {box!r}],
            "cxx": ["force_mcc", "-include", {box!r}],
        }},
        # AOTI stable-ABI shims live in libtorch_cpu.so.
        libraries=["c10", "torch", "torch_cpu", "torch_python", "musart"],
    )],
    cmdclass={{"build_ext": BuildExtension}},
)
"""


@pytest.mark.musa
@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not _gpu_available(), reason="requires a MUSA GPU")
def test_stable_abi_shim_general_signatures(tmp_path):
    import torch

    shutil.copy(OPS_CU, tmp_path)
    (tmp_path / "setup.py").write_text(
        _SETUP_TEMPLATE.format(inc=stable_compat_include_dir(),
                               box=stable_compat_box_header()))

    res = subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=tmp_path, capture_output=True, text=True)
    assert res.returncode == 0, f"build failed:\n{res.stderr[-4000:]}"

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
