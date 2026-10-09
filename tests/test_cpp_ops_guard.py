"""Tests for the flock guard around torchada's JIT C++ extension builds."""

import errno
import fcntl
import sys
import types
import warnings

import pytest
import torch.utils.cpp_extension as cpp_ext

from torchada import _cpp_ops

EXT_NAME = "fake_ext"


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
