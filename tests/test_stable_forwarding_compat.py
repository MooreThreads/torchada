"""Compile the legacy stable_compat header interface without forwarding -I."""

import os
import pathlib
import shutil
import subprocess

import pytest

from torchada.utils.cpp_extension import stable_compat_include_dir


@pytest.mark.parametrize("standard", ["c++17", "c++20"])
def test_stable_compat_headers_compile_without_forwarding_include(tmp_path, standard):
    compiler = shutil.which("mcc")
    if compiler is None:
        pytest.skip("requires mcc")
    source = pathlib.Path(__file__).parent / "csrc" / "stable_forwarding_compat.cu"
    result = subprocess.run(
        [
            compiler,
            "-x",
            "musa",
            "-std=" + standard,
            "--offload-arch=" + os.environ.get("MTGPU_TARGET", "mp_31"),
            "-I",
            stable_compat_include_dir(),
            "-c",
            str(source),
            "-o",
            str(tmp_path / "stable_forwarding_compat.o"),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
