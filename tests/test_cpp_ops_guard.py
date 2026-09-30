"""Tests for the flock guard around torchada's JIT C++ extension builds."""

import errno
import fcntl

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
