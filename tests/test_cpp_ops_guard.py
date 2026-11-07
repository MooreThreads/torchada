"""Tests for the flock guard around torchada's JIT C++ extension builds."""

import errno
import fcntl
import sys
import types
import warnings
from contextlib import nullcontext

import pytest
import torch
import torch.utils.cpp_extension as cpp_ext

from torchada import _cpp_ops, _platform, arch
from torchada.arch import MusaArch
from torchada.utils import cpp_extension

EXT_NAME = "fake_ext"
SUBDIR = "mp31_mu43"


@pytest.fixture
def ext_dir(tmp_path, monkeypatch):
    """Point torch's JIT build root at tmp_path and make torch look FileBaton-based."""
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(cpp_ext, "FileBaton", object, raising=False)
    return tmp_path / EXT_NAME


def _load(fake_load):
    return _cpp_ops._locked_load(fake_load, name=EXT_NAME, musa=False)


def test_stale_baton_is_removed(ext_dir):
    """A torch ``lock`` left by a killed builder is deleted before the build."""
    ext_dir.mkdir()
    (ext_dir / "lock").write_bytes(b"")

    def fake_load(**kwargs):
        assert not (ext_dir / "lock").exists()
        return kwargs

    with pytest.warns(UserWarning, match="stale JIT build lock"):
        kwargs = _load(fake_load)
    assert kwargs == {"name": EXT_NAME, "build_directory": str(ext_dir)}


def test_flock_is_held_during_load(ext_dir):
    """Another opener of the flock file is excluded while the build runs."""

    def fake_load(**kwargs):
        with open(ext_dir / _cpp_ops._FLOCK_FILE) as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)

    _load(fake_load)


def test_flock_unavailable_keeps_the_baton(ext_dir, monkeypatch):
    """Without flock (e.g. ENOLCK on NFS) the baton cannot be proven stale, so it stays."""
    ext_dir.mkdir()
    (ext_dir / "lock").write_bytes(b"")

    def no_flock(*args):
        raise OSError(errno.ENOLCK, "No locks available")

    monkeypatch.setattr(fcntl, "flock", no_flock)
    with pytest.warns(UserWarning, match="flock unavailable"):
        _load(lambda **kwargs: None)
    assert (ext_dir / "lock").exists()


def test_filelock_based_torch_is_left_alone(ext_dir, monkeypatch):
    """Newer torch locks JIT builds itself, so the call passes straight through."""
    monkeypatch.delattr(cpp_ext, "FileBaton")

    assert _load(lambda **kwargs: kwargs) == {"name": EXT_NAME}
    assert not (ext_dir / _cpp_ops._FLOCK_FILE).exists()


def test_baton_is_removed_even_when_warnings_are_errors(ext_dir):
    """Under ``-W error`` the warning raises, but only after the stale baton is gone."""
    ext_dir.mkdir()
    (ext_dir / "lock").write_bytes(b"")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="stale JIT build lock"):
            _load(lambda **kwargs: None)
    assert not (ext_dir / "lock").exists()


def test_musa_loader_build_directory_is_used(tmp_path, monkeypatch):
    """MUSA sources lock and build in torch_musa's build directory."""
    musa_dir = tmp_path / "musa" / EXT_NAME

    def get_build_directory(name, verbose):
        musa_dir.mkdir(parents=True, exist_ok=True)
        return str(musa_dir)

    loader = types.ModuleType("torch_musa.utils.musa_extension")
    loader.FileBaton = object
    loader._get_build_directory = get_build_directory
    utils = types.ModuleType("torch_musa.utils")
    utils.musa_extension = loader
    package = types.ModuleType("torch_musa")
    package.utils = utils
    for module in (package, utils, loader):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    musa_dir.mkdir(parents=True)
    (musa_dir / "lock").write_bytes(b"")
    with pytest.warns(UserWarning, match="stale JIT build lock"):
        kwargs = _cpp_ops._locked_load(lambda **kw: kw, name=EXT_NAME, musa=True)
    assert kwargs["build_directory"] == str(musa_dir)
    assert not (musa_dir / "lock").exists()


def test_subdir_holds_the_flock_and_the_baton(ext_dir):
    """With a sub-directory, it is created and both lock files live in it."""
    ext_dir.mkdir()
    (ext_dir / "lock").write_bytes(b"")
    subdir = ext_dir / SUBDIR

    def fake_load(**kwargs):
        with open(subdir / _cpp_ops._FLOCK_FILE) as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return kwargs

    kwargs = _cpp_ops._locked_load(fake_load, name=EXT_NAME, musa=False, subdir=SUBDIR)
    assert kwargs == {"name": EXT_NAME, "build_directory": str(subdir)}
    assert subdir.is_dir()
    # A baton beside the sub-directory belongs to another layout and is left alone.
    assert (ext_dir / "lock").exists()
    assert not (ext_dir / _cpp_ops._FLOCK_FILE).exists()


def test_stale_baton_in_subdir_is_removed(ext_dir):
    """The stale-baton recovery looks inside the sub-directory."""
    subdir = ext_dir / SUBDIR
    subdir.mkdir(parents=True)
    (subdir / "lock").write_bytes(b"")

    with pytest.warns(UserWarning, match="stale JIT build lock"):
        _cpp_ops._locked_load(lambda **kwargs: None, name=EXT_NAME, musa=False, subdir=SUBDIR)
    assert not (subdir / "lock").exists()


def test_filelock_based_torch_builds_in_the_subdir(ext_dir, monkeypatch):
    """Without FileBaton the sub-directory is still created and used, with no flock file."""
    monkeypatch.delattr(cpp_ext, "FileBaton")
    subdir = ext_dir / SUBDIR

    kwargs = _cpp_ops._locked_load(
        lambda **kwargs: kwargs, name=EXT_NAME, musa=False, subdir=SUBDIR, verbose=True
    )
    assert kwargs == {"name": EXT_NAME, "build_directory": str(subdir), "verbose": True}
    assert subdir.is_dir()
    assert not (subdir / _cpp_ops._FLOCK_FILE).exists()


def test_musa_loader_subdir_is_used(tmp_path, monkeypatch):
    """torch_musa's build directory gets the sub-directory too."""
    musa_dir = tmp_path / "musa" / EXT_NAME
    loader = types.ModuleType("torch_musa.utils.musa_extension")
    loader.FileBaton = object
    loader._get_build_directory = lambda name, verbose: str(musa_dir)
    utils = types.ModuleType("torch_musa.utils")
    utils.musa_extension = loader
    package = types.ModuleType("torch_musa")
    package.utils = utils
    for module in (package, utils, loader):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    kwargs = _cpp_ops._locked_load(lambda **kw: kw, name=EXT_NAME, musa=True, subdir=SUBDIR)
    assert kwargs["build_directory"] == str(musa_dir / SUBDIR)
    assert (musa_dir / SUBDIR / _cpp_ops._FLOCK_FILE).exists()


@pytest.fixture
def musa_cpp_ops(monkeypatch):
    """Run ``load_cpp_ops`` as on MUSA, with torch's JIT loader and no device."""
    monkeypatch.setattr(_platform, "is_musa_platform", lambda: True)
    monkeypatch.setattr(_cpp_ops, "_cpp_ops_module", None)
    monkeypatch.delattr(torch, "musa", raising=False)
    monkeypatch.setitem(sys.modules, "torch_musa.utils.musa_extension", None)
    for name in ("MTGPU_TARGET", "TORCH_MUSA_ARCH_LIST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(arch, "_initialized_device_archs", lambda: pytest.fail("device check"))


@pytest.fixture
def locked_loads(musa_cpp_ops, monkeypatch):
    """Arguments of every ``_locked_load`` call ``load_cpp_ops`` makes."""
    calls = []

    def fake_locked_load(load, **kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(_mark_loaded=lambda: None)

    monkeypatch.setattr(_cpp_ops, "_locked_load", fake_locked_load)
    return calls


def test_cpp_ops_mtgpu_target_decides_without_probing(locked_loads, monkeypatch):
    """A set MTGPU_TARGET picks the arch flags and sub-directory; no device is probed."""
    monkeypatch.setenv("MTGPU_TARGET", "mp_31")
    monkeypatch.setattr(torch.version, "musa", "4.3.1", raising=False)
    monkeypatch.setattr(arch, "_device_archs", lambda: pytest.fail("probed the device"))

    assert _cpp_ops.load_cpp_ops() is not None
    (call,) = locked_loads
    assert call["name"] == "torchada_cpp_ops"
    assert call["musa"] is True
    assert call["extra_cuda_cflags"] == ["--offload-arch=mp_31"]
    assert call["subdir"] == "mp31_mu431"


def test_cpp_ops_cover_every_visible_device(locked_loads, monkeypatch):
    """Without MTGPU_TARGET every visible device's arch is compiled."""
    monkeypatch.setattr(torch.version, "musa", None, raising=False)
    monkeypatch.setattr(arch, "_device_archs", lambda: (MusaArch(2, 2), MusaArch(3, 1)))

    _cpp_ops.load_cpp_ops()
    (call,) = locked_loads
    assert call["extra_cuda_cflags"] == ["--offload-arch=mp_22", "--offload-arch=mp_31"]
    assert call["subdir"] == "mp22_mp31"


def test_cpp_ops_default_ignores_torch_musa_arch_list(locked_loads, monkeypatch):
    """torchada's own ops read only MTGPU_TARGET, then fall back to mp_31."""
    monkeypatch.setenv("TORCH_MUSA_ARCH_LIST", "22")
    monkeypatch.setattr(arch, "_device_archs", lambda: ())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpp_ops.load_cpp_ops()
    (call,) = locked_loads
    assert call["extra_cuda_cflags"] == ["--offload-arch=mp_31"]
    assert call["subdir"] == arch.cache_tag([MusaArch(3, 1)])


def test_cpp_ops_malformed_mtgpu_target_warns(locked_loads, monkeypatch):
    """A malformed MTGPU_TARGET fails the build with a warning, not the import."""
    monkeypatch.setenv("MTGPU_TARGET", "mp_310")

    with pytest.warns(UserWarning, match=r"Failed to load torchada C\+\+ ops: MTGPU_TARGET"):
        assert _cpp_ops.load_cpp_ops() is None
    assert locked_loads == []


def test_cpp_ops_build_in_the_arch_subdirectory(musa_cpp_ops, tmp_path, monkeypatch):
    """End to end on disk: the stale baton inside the arch sub-directory is recovered."""
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(cpp_ext, "FileBaton", object, raising=False)
    monkeypatch.setenv("MTGPU_TARGET", "mp_31")
    subdir = tmp_path / "torchada_cpp_ops" / arch.cache_tag([MusaArch(3, 1)])
    subdir.mkdir(parents=True)
    (subdir / "lock").write_bytes(b"")
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        assert not (subdir / "lock").exists()
        assert (subdir / _cpp_ops._FLOCK_FILE).exists()
        return types.SimpleNamespace(_mark_loaded=lambda: None)

    monkeypatch.setattr(cpp_extension, "load", fake_load)
    with pytest.warns(UserWarning, match="stale JIT build lock"):
        assert _cpp_ops.load_cpp_ops() is not None
    assert subdir.is_dir()
    assert seen["name"] == "torchada_cpp_ops"
    assert seen["build_directory"] == str(subdir)
    assert seen["extra_cuda_cflags"] == ["--offload-arch=mp_31"]


@pytest.mark.musa
@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("custom_stream", [False, True])
def test_log_uses_input_device_and_stream(is_musa, dtype_name, inplace, strided, custom_stream):
    """An input on GPU 1 must not launch the log kernel on the current GPU 0."""
    import torch

    if not is_musa or torch.musa.device_count() < 2:
        pytest.skip("Requires two MUSA GPUs")

    dtype = getattr(torch, dtype_name)
    reference = torch.linspace(0.01, 16, 64, dtype=dtype).reshape(2, 32)
    stream = torch.musa.Stream(device=1) if custom_stream else torch.musa.default_stream(1)
    with torch.musa.stream(stream) if custom_stream else nullcontext():
        tensor = reference.to("musa:1")
        if strided:
            tensor = tensor[:, ::2]
            reference = reference[:, ::2]
        torch.musa.synchronize(1)
        with torch.musa.device(0):
            result = tensor.log_() if inplace else torch.log(tensor)
            assert torch.musa.current_device() == 0
        stream.synchronize()
    # Also surface erroneous launches on GPU 0 before comparing CPU values.
    torch.musa.synchronize(0)
    expected = reference.double().log().to(dtype)
    torch.testing.assert_close(result.cpu(), expected)
    assert result.device == tensor.device
    if inplace:
        assert result.data_ptr() == tensor.data_ptr()
