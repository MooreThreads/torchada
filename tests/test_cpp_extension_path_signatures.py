"""CPU-only contracts for legacy and modern extension path discovery."""

import sys
import types

import pytest

from torchada import Platform
from torchada.utils import cpp_extension as ext


@pytest.fixture
def musa_paths(monkeypatch):
    root = types.ModuleType("torch_musa")
    utils = types.ModuleType("torch_musa.utils")
    backend = types.ModuleType("torch_musa.utils.musa_extension")
    backend.include_paths = lambda musa=False: ["/torch/include"] + (
        ["/sdk/include"] if musa else []
    )
    backend.library_paths = lambda musa=False: ["/torch/lib"] + (["/sdk/lib"] if musa else [])
    root.utils = utils
    utils.musa_extension = backend
    for name, module in (
        ("torch_musa", root),
        ("torch_musa.utils", utils),
        ("torch_musa.utils.musa_extension", backend),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(ext, "detect_platform", lambda: Platform.MUSA)
    monkeypatch.setattr(ext, "_get_cuda_home", lambda: "/sdk")
    return backend


@pytest.mark.parametrize(
    "tested_device",
    ["cpu", "cuda", "musa", False, True],
    ids=["host-name", "mapped-gpu-name", "native-gpu-name", "old-host-flag", "old-gpu-flag"],
)
def test_positional_device_and_torch_flag(musa_paths, tested_device):
    gpu = tested_device in ("cuda", "musa", True)
    includes = ext.include_paths(tested_device, False)
    libraries = ext.library_paths(tested_device, False)
    assert "/torch/include" not in includes
    assert "/torch/lib" not in libraries
    assert ("/sdk/include" in includes) == gpu
    assert ("/sdk/lib" in libraries) == gpu


@pytest.mark.parametrize("flag", [True, False])
def test_legacy_cuda_keyword_and_positional_flag(musa_paths, flag):
    assert ext.include_paths(flag) == ext.include_paths(cuda=flag)
    assert ext.library_paths(flag) == ext.library_paths(cuda=flag)


def test_cpu_inductor_211_calls(musa_paths):
    assert ext.include_paths("cpu", False) == []
    assert ext.library_paths("cpu", torch_include_dirs=True, cross_target_platform=None) == [
        "/torch/lib"
    ]
    assert ext.library_paths("cpu", torch_include_dirs=False, cross_target_platform=None) == []
    assert ext.library_paths("cpu", True, None) == ["/torch/lib"]


@pytest.mark.parametrize("flag, override", [(False, "musa"), (True, "cpu"), (None, "cpu")])
def test_legacy_two_positional_arguments(musa_paths, flag, override):
    assert ext.include_paths(flag, override) == ext.include_paths(cuda=flag, device_type=override)
    assert ext.library_paths(flag, override) == ext.library_paths(cuda=flag, device_type=override)


def test_keep_legacy_defaults_and_device_precedence(musa_paths):
    assert "/sdk/include" in ext.include_paths()
    assert "/sdk/lib" in ext.library_paths()
    assert ext.include_paths(cuda=True, device_type="cpu") == ["/torch/include"]
    assert ext.library_paths(cuda=True, device_type="cpu") == []


def test_musa_cross_target_is_explicitly_unsupported(musa_paths):
    with pytest.raises(NotImplementedError, match="cross-target"):
        ext.library_paths("musa", cross_target_platform="windows")


@pytest.mark.parametrize("kind", ["include_paths", "library_paths"])
def test_modern_native_forwarding(monkeypatch, kind):
    import torch.utils.cpp_extension as native

    calls = []

    def paths(device_type="cuda", torch_include_dirs=True, cross_target_platform=None):
        calls.append((device_type, torch_include_dirs, cross_target_platform))
        return (["/torch/base"] if torch_include_dirs else []) + ["/native/sdk"]

    monkeypatch.setattr(ext, "detect_platform", lambda: Platform.CUDA)
    monkeypatch.setattr(native, kind, paths)
    target = ext.include_paths if kind == "include_paths" else ext.library_paths
    assert target("cuda", False) == ["/native/sdk"]
    assert calls == [("cuda", False, None)]
    if kind == "library_paths":
        target("cuda", True, cross_target_platform="windows")
        assert calls[-1] == ("cuda", True, "windows")


@pytest.mark.parametrize("kind", ["include_paths", "library_paths"])
def test_legacy_native_forwarding(monkeypatch, kind):
    import torch.utils.cpp_extension as native

    def paths(cuda=False):
        return ["/torch/base"] + (["/native/sdk"] if cuda else [])

    monkeypatch.setattr(ext, "detect_platform", lambda: Platform.CUDA)
    monkeypatch.setattr(native, kind, paths)
    target = ext.include_paths if kind == "include_paths" else ext.library_paths
    assert target(True, False) == ["/native/sdk"]
    assert target(False, False) == []
    if kind == "library_paths":
        assert target(cuda=False) == []
