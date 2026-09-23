"""SGLang's custom ninja JIT must compile with mcc on MUSA."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from importlib.util import module_from_spec
from types import ModuleType

from torchada._patch import _apply_sglang_jit_toolchain, _patch_sglang_jit_toolchain

TOOLCHAIN_NAME = "sglang.kernels.jit.utils.compile.toolchain"


def _toolchain_module() -> ModuleType:
    module = ModuleType(TOOLCHAIN_NAME)
    module.cuda_home = lambda: "/usr/local/cuda"
    module.device_compiler_path = lambda: "/usr/local/cuda/bin/nvcc"
    module.target_flags = lambda: ["-gencode=arch=compute_90,code=sm_90"]
    module.base_cuda_flags = lambda: ["-Xcompiler", "-fPIC"]
    module.base_include_paths = lambda: ["/opt/tvm-ffi/include"]
    module.base_link_flags = lambda *, with_device: (
        ["-shared", "-Ltvm", "-ltvm_ffi", "-L/usr/local/cuda/lib64", "-lcudart"]
        if with_device
        else ["-shared", "-Ltvm", "-ltvm_ffi"]
    )
    module.is_hip_runtime = lambda: False
    return module


def _assert_mapped(toolchain, musa_home: str) -> None:
    assert toolchain.cuda_home() == musa_home
    assert toolchain.device_compiler_path() == f"{musa_home}/bin/mcc"
    assert toolchain.target_flags() == ["--offload-arch=mp_31"]
    assert toolchain.base_cuda_flags()[-2:] == ["-x", "musa"]
    assert f"{musa_home}/include" in toolchain.base_include_paths()
    assert toolchain.base_link_flags(with_device=False) == [
        "-shared",
        "-Ltvm",
        "-ltvm_ffi",
    ]
    link_flags = toolchain.base_link_flags(with_device=True)
    assert "-lcudart" not in link_flags
    assert "-lmusart" in link_flags
    assert f"-L{musa_home}/lib" in link_flags


def test_sglang_jit_toolchain_maps_nvcc_gencode_and_cudart(monkeypatch, tmp_path):
    musa_home = tmp_path / "musa"
    (musa_home / "include").mkdir(parents=True)
    (musa_home / "lib").mkdir()
    monkeypatch.setattr("torchada.utils.cpp_extension.CUDA_HOME", str(musa_home))
    monkeypatch.setenv("MTGPU_TARGET", "mp_31")
    monkeypatch.setattr("torchada._platform.is_musa_platform", lambda: True)
    monkeypatch.setattr("torchada._patch.is_musa_platform", lambda: True)

    toolchain = _toolchain_module()
    _apply_sglang_jit_toolchain(toolchain)
    _assert_mapped(toolchain, str(musa_home))


def test_sglang_jit_import_hook_patches_the_first_import(monkeypatch, tmp_path):
    musa_home = tmp_path / "musa"
    (musa_home / "include").mkdir(parents=True)
    (musa_home / "lib").mkdir()
    monkeypatch.setattr("torchada.utils.cpp_extension.CUDA_HOME", str(musa_home))
    monkeypatch.setenv("MTGPU_TARGET", "mp_31")
    monkeypatch.setattr("torchada._patch.is_musa_platform", lambda: True)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [finder for finder in sys.meta_path if not getattr(finder, "_torchada_sglang_jit", False)],
    )
    sys.modules.pop(TOOLCHAIN_NAME, None)

    class DummyLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            source = _toolchain_module()
            module.__dict__.update(
                {
                    name: getattr(source, name)
                    for name in (
                        "cuda_home",
                        "device_compiler_path",
                        "target_flags",
                        "base_cuda_flags",
                        "base_include_paths",
                        "base_link_flags",
                        "is_hip_runtime",
                    )
                }
            )

    class DummyFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname != TOOLCHAIN_NAME:
                return None
            return ModuleSpec(fullname, DummyLoader(), is_package=False)

    sys.meta_path.append(DummyFinder())
    _patch_sglang_jit_toolchain()
    hook = next(
        finder for finder in sys.meta_path if getattr(finder, "_torchada_sglang_jit", False)
    )
    spec = hook.find_spec(TOOLCHAIN_NAME, None)
    assert spec is not None
    toolchain = module_from_spec(spec)
    spec.loader.exec_module(toolchain)
    try:
        assert getattr(toolchain, "_torchada_sglang_jit", False)
        _assert_mapped(toolchain, str(musa_home))
    finally:
        sys.modules.pop(TOOLCHAIN_NAME, None)
