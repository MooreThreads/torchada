"""Tests for torchada.arch: the MUSA arch table, target resolution and build flags."""

import dataclasses
import logging
import re
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch

import torchada
from torchada import arch
from torchada.arch import MusaArch
from torchada.utils import cpp_extension

MP22 = MusaArch(2, 2)
MP31 = MusaArch(3, 1)
MP32 = MusaArch(3, 2)

ARCH_SOURCE = Path(__file__).parent.parent / "src" / "torchada" / "arch.py"


@pytest.fixture(autouse=True)
def isolated_probes(monkeypatch):
    """No real device probe, a fresh musaInfo cache and no arch variables per test."""
    for name in ("TEST_ARCH_A", "TEST_ARCH_B", "TEST_ARCH_C"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delattr(torch, "musa", raising=False)
    arch._musainfo_archs.cache_clear()
    yield
    arch._musainfo_archs.cache_clear()


class FakeRun:
    """Stands in for ``subprocess.run`` and records every command it is given."""

    def __init__(self, stdout="", error=None):
        self.stdout = stdout
        self.error = error
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.error is not None:
            raise self.error
        return subprocess.CompletedProcess(cmd, 0, stdout=self.stdout, stderr="")


@pytest.fixture
def musainfo(monkeypatch):
    """A fake ``musaInfo`` on PATH; set ``.stdout`` or ``.error`` before probing."""
    fake = FakeRun()
    monkeypatch.setattr(arch.shutil, "which", lambda name: "/fake/bin/" + name)
    monkeypatch.setattr(arch.subprocess, "run", fake)
    return fake


class FakeMusa(types.ModuleType):
    """A ``torch.musa`` stand-in that records every device query."""

    def __init__(self, initialized, capabilities=()):
        super().__init__("torch.musa")
        self.initialized = initialized
        self.capabilities = list(capabilities)
        self.device_calls = []

    def is_initialized(self):
        return self.initialized

    def device_count(self):
        self.device_calls.append("device_count")
        if not self.initialized:
            raise AssertionError("device_count() on an uninitialized runtime")
        return len(self.capabilities)

    def get_device_capability(self, index=None):
        self.device_calls.append("get_device_capability")
        if not self.initialized:
            raise AssertionError("get_device_capability() on an uninitialized runtime")
        return self.capabilities[index]

    def current_device(self):
        self.device_calls.append("current_device")
        raise AssertionError("current_device() must not be called")


class RecordingEnviron(dict):
    """A copy of the environment that records every variable read from it."""

    def __init__(self, data):
        super().__init__(data)
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.reads.append(key)
        return super().__getitem__(key)

    def __contains__(self, key):
        self.reads.append(key)
        return super().__contains__(key)


def _raise(error):
    raise error


def _musainfo_output(*capabilities):
    lines = [
        "Device {}\n  compute capability:              {}".format(i, c)
        for i, c in enumerate(capabilities)
    ]
    return "\n".join(["musaInfo"] + lines) + "\n"


class TestModule:
    def test_public_names(self):
        assert set(arch.__all__) == {
            "MusaArch",
            "ARCHS",
            "target_archs",
            "arch_flags",
            "torch_arch_list",
            "cache_tag",
        }
        assert torchada.arch is arch
        assert "arch" in torchada.__all__

    def test_import_does_not_import_torch(self):
        code = (
            "import importlib.util, sys\n"
            "spec = importlib.util.spec_from_file_location('arch_alone', sys.argv[1])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(module)\n"
            "assert module.ARCHS[0].name == 'mp_31'\n"
            "assert 'torch' not in sys.modules, sorted(m for m in sys.modules if 'torch' in m)\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(ARCH_SOURCE)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr


class TestMusaArch:
    def test_properties(self):
        assert MP31.capability == (3, 1)
        assert MP31.name == "mp_31"
        assert str(MP31) == "mp_31"

    @pytest.mark.parametrize(
        "major, minor",
        [(0, 1), (10, 0), (3, 10), (3, -1), (3.0, 1), ("3", 1), (3, None), (True, 1)],
    )
    def test_rejects_out_of_range_and_non_int(self, major, minor):
        with pytest.raises(ValueError, match="MusaArch"):
            MusaArch(major, minor)

    @pytest.mark.parametrize(
        "major, minor, name", [(1, 0, "mp_10"), (1, 9, "mp_19"), (9, 0, "mp_90"), (9, 9, "mp_99")]
    )
    def test_accepts_the_range_bounds(self, major, minor, name):
        assert MusaArch(major, minor).name == name

    def test_identity_is_the_capability(self):
        assert MusaArch(3, 1) == MP31
        assert hash(MusaArch(3, 1)) == hash(MP31)
        assert MusaArch(3, 1) in arch.ARCHS
        assert len({MusaArch(3, 1), MP31}) == 1
        assert MusaArch(3, 1).fp8 is MP31.fp8 is True

    def test_frozen(self):
        with pytest.raises(dataclasses.FrozenInstanceError):
            MP31.minor = 2

    def test_ordering(self):
        assert MP22 < MP31 < MP32 < MusaArch(4, 0)
        assert sorted([MusaArch(4, 0), MP31, MP22, MP32]) == [MP22, MP31, MP32, MusaArch(4, 0)]

    def test_fp8_comes_from_the_table(self, monkeypatch):
        assert MP31.fp8 is True
        assert MP22.fp8 is False
        assert MP32.fp8 is False
        monkeypatch.setitem(arch._ROWS, (3, 2), arch._Row(fp8=True))
        assert MP32.fp8 is True

    def test_fp4_comes_from_the_table(self, monkeypatch):
        assert MP31.fp4 is False
        assert MP22.fp4 is False
        assert MP32.fp4 is False
        monkeypatch.setitem(arch._ROWS, (3, 2), arch._Row(fp4=True))
        assert MP32.fp4 is True
        assert MP32.fp8 is False
        monkeypatch.setitem(arch._ROWS, (3, 1), arch._Row(fp8=True, fp4=True))
        assert MP31.fp4 is True

    def test_archs_follow_the_rows(self):
        assert arch.ARCHS == (MP31,)
        assert arch.ARCHS == tuple(MusaArch(*capability) for capability in sorted(arch._ROWS))


class TestParse:
    @pytest.mark.parametrize("token", ["mp_31", "MP_31", "mp31", "Mp31", "3.1", "31", " 31\t"])
    def test_accepted_forms(self, token):
        assert arch._parse(token) == MP31

    @pytest.mark.parametrize(
        "token, capability",
        [("10", (1, 0)), ("mp_19", (1, 9)), ("9.0", (9, 0)), ("mp99", (9, 9)), ("9.9", (9, 9))],
    )
    def test_range_bounds_parse(self, token, capability):
        assert arch._parse(token) == MusaArch(*capability)

    @pytest.mark.parametrize(
        "token",
        [
            "310",
            "3.10",
            "mp_310",
            "mp_3.1",
            "sm_80",
            "3_1",
            "3",
            "",
            "0.0",
            "mp_01",
            "3.1.0",
            "mp-31",
        ],
    )
    def test_rejected_tokens_list_the_forms(self, token):
        with pytest.raises(ValueError, match=r"mp_XY, mpXY, X\.Y or XY"):
            arch._parse(token)


class TestTargetArchsEnvironment:
    def test_first_set_variable_wins_and_later_ones_are_not_read(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "")
        monkeypatch.setenv("TEST_ARCH_B", "mp_22")
        monkeypatch.setenv("TEST_ARCH_C", "not-an-arch")
        environ = RecordingEnviron(arch.os.environ)
        monkeypatch.setattr(arch.os, "environ", environ)
        archs = arch.target_archs(env=("TEST_ARCH_A", "TEST_ARCH_B", "TEST_ARCH_C"))
        assert archs == (MP22,)
        assert environ.reads == ["TEST_ARCH_A", "TEST_ARCH_B"]

    def test_environment_never_probes(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "31")
        monkeypatch.setattr(arch, "_device_archs", lambda: pytest.fail("probed the device"))
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP31,)

    def test_separators_dedupe_and_sort(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", " 3.2;mp_31 22\t31;;MP_22 ")
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP22, MP31, MP32)

    def test_malformed_token_names_the_variable(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "31;310")
        with pytest.raises(ValueError, match=r"TEST_ARCH_A='31;310'.*'310'"):
            arch.target_archs(env=("TEST_ARCH_A", "TEST_ARCH_B"))

    @pytest.mark.parametrize("value", [" ", ";", " ; \t"])
    def test_value_without_tokens_names_the_variable(self, monkeypatch, value):
        monkeypatch.setenv("TEST_ARCH_A", value)
        monkeypatch.setenv("TEST_ARCH_B", "31")
        with pytest.raises(ValueError, match="TEST_ARCH_A"):
            arch.target_archs(env=("TEST_ARCH_A", "TEST_ARCH_B"))

    def test_env_must_be_a_sequence_of_names(self):
        with pytest.raises(TypeError):
            arch.target_archs(env="TEST_ARCH_A")


class TestTargetArchsFallback:
    def test_visible_devices_decide_without_environment(self, monkeypatch):
        monkeypatch.setattr(arch, "_device_archs", lambda: (MP22, MP31))
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP22, MP31)

    def test_default_is_silent(self, monkeypatch, capsys, caplog):
        monkeypatch.setattr(arch, "_device_archs", lambda: ())
        caplog.set_level(logging.DEBUG)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP31,)
            assert arch.target_archs(env=(), default="2.2") == (MP22,)
        assert caught == []
        assert capsys.readouterr() == ("", "")
        assert caplog.records == []

    def test_malformed_default_raises(self, monkeypatch):
        monkeypatch.setattr(arch, "_device_archs", lambda: ())
        with pytest.raises(ValueError, match="'mp_310'"):
            arch.target_archs(env=(), default="mp_310")


class TestTargetArchsDeviceCheck:
    def test_mismatch_names_result_devices_and_decider(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "2.2")
        monkeypatch.setattr(arch, "_initialized_device_archs", lambda: (MP31,))
        with pytest.raises(RuntimeError) as excinfo:
            arch.target_archs(env=("TEST_ARCH_A",), check_device=True)
        message = str(excinfo.value)
        assert "[mp_22]" in message
        assert "[mp_31]" in message
        assert "TEST_ARCH_A='2.2'" in message

    def test_every_device_must_be_covered(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "31")
        monkeypatch.setattr(arch, "_initialized_device_archs", lambda: (MP22, MP31))
        with pytest.raises(RuntimeError, match=r"\[mp_22, mp_31\]"):
            arch.target_archs(env=("TEST_ARCH_A",), check_device=True)

    def test_covered_devices_pass(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "22;31")
        monkeypatch.setattr(arch, "_initialized_device_archs", lambda: (MP31,))
        assert arch.target_archs(env=("TEST_ARCH_A",), check_device=True) == (MP22, MP31)

    def test_skipped_when_not_initialized(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "2.2")
        monkeypatch.setattr(arch, "_initialized_device_archs", lambda: None)
        assert arch.target_archs(env=("TEST_ARCH_A",), check_device=True) == (MP22,)

    def test_not_run_without_check_device(self, monkeypatch):
        monkeypatch.setenv("TEST_ARCH_A", "2.2")
        monkeypatch.setattr(arch, "_initialized_device_archs", lambda: pytest.fail("checked"))
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP22,)


class TestRenderers:
    def test_arch_flags(self):
        flags = arch.arch_flags([MP31, MP22, MP31])
        assert flags == ["--offload-arch=mp_22", "--offload-arch=mp_31"]
        assert arch.arch_flags(a for a in (MP31,)) == ["--offload-arch=mp_31"]

    def test_torch_arch_list(self):
        assert arch.torch_arch_list([MP32, MP31, MP32]) == "31;32"
        assert arch.torch_arch_list(arch.ARCHS) == "31"

    def test_cache_tag_without_musa_toolkit(self, monkeypatch):
        monkeypatch.setattr(torch.version, "musa", None, raising=False)
        assert arch.cache_tag([MP31, MP22, MP31]) == "mp22_mp31"

    @pytest.mark.parametrize(
        "version, suffix", [("4.3.1", "_mu431"), ("rc4.3-Beta+1", "_murc43beta1"), ("43", "_mu43")]
    )
    def test_cache_tag_carries_the_musa_toolkit(self, monkeypatch, version, suffix):
        monkeypatch.setattr(torch.version, "musa", version, raising=False)
        tag = arch.cache_tag([MP31])
        assert tag == "mp31" + suffix
        assert re.fullmatch(r"[a-z0-9_]+", tag)

    @pytest.mark.parametrize("render", [arch.arch_flags, arch.torch_arch_list, arch.cache_tag])
    def test_empty_input_raises(self, render):
        with pytest.raises(ValueError, match="at least one"):
            render([])

    @pytest.mark.parametrize("render", [arch.arch_flags, arch.torch_arch_list, arch.cache_tag])
    def test_non_arch_input_raises(self, render):
        with pytest.raises(TypeError, match="MusaArch"):
            render(["mp_31"])


class TestMusaInfoProbe:
    def test_one_device(self, musainfo):
        musainfo.stdout = _musainfo_output("3.1")
        assert arch._device_archs() == (MP31,)
        cmd, kwargs = musainfo.calls[0]
        assert cmd == ["/fake/bin/musaInfo"]
        assert kwargs["timeout"] == 5

    def test_every_device_line_is_read(self, musainfo):
        musainfo.stdout = _musainfo_output("3.1", "2.2", "3.1")
        assert arch._device_archs() == (MP22, MP31)

    def test_invalid_values_are_skipped(self, musainfo):
        musainfo.stdout = (
            _musainfo_output("0.0", "abc", "3", "3.1") + "Compute Capability: 2.2\nmemory: 80 GB\n"
        )
        assert arch._device_archs() == (MP22, MP31)

    def test_no_capability_line(self, musainfo):
        musainfo.stdout = "no MUSA device found\n"
        assert arch._device_archs() == ()

    @pytest.mark.parametrize(
        "error", [subprocess.TimeoutExpired("musaInfo", 5), FileNotFoundError(), OSError()]
    )
    def test_failure_returns_empty(self, musainfo, error):
        musainfo.error = error
        assert arch._device_archs() == ()

    def test_result_is_cached_until_reset(self, musainfo):
        musainfo.stdout = _musainfo_output("3.1")
        assert arch._device_archs() == (MP31,)
        musainfo.stdout = _musainfo_output("2.2")
        assert arch._device_archs() == (MP31,)
        assert len(musainfo.calls) == 1
        arch._musainfo_archs.cache_clear()
        assert arch._device_archs() == (MP22,)
        assert len(musainfo.calls) == 2

    def test_failure_is_cached(self, musainfo):
        musainfo.error = subprocess.TimeoutExpired("musaInfo", 5)
        assert arch._device_archs() == ()
        assert arch._device_archs() == ()
        assert len(musainfo.calls) == 1

    def test_falls_back_to_the_musa_home(self, monkeypatch, tmp_path):
        fake = FakeRun(stdout=_musainfo_output("3.1"))
        monkeypatch.setattr(arch.shutil, "which", lambda name: None)
        monkeypatch.setattr(arch.subprocess, "run", fake)
        monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
        assert arch._device_archs() == (MP31,)
        assert fake.calls[0][0] == [str(tmp_path / "bin" / "musaInfo")]

    def test_no_musainfo_anywhere(self, monkeypatch):
        fake = FakeRun()
        monkeypatch.setattr(arch.shutil, "which", lambda name: None)
        monkeypatch.setattr(arch.subprocess, "run", fake)
        monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
        assert arch._device_archs() == ()
        assert fake.calls == []

    def test_target_archs_uses_the_probe(self, musainfo):
        musainfo.stdout = _musainfo_output("3.1", "2.2")
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP22, MP31)


class TestInProcessProbe:
    def test_uninitialized_runtime_is_never_touched(self, monkeypatch, musainfo):
        musa = FakeMusa(initialized=False)
        monkeypatch.setattr(torch, "musa", musa, raising=False)
        musainfo.stdout = _musainfo_output("3.1")

        assert arch._initialized_device_archs() is None
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP31,)
        assert len(musainfo.calls) == 1
        assert musa.device_calls == []

    def test_check_is_skipped_without_a_subprocess(self, monkeypatch, musainfo):
        musa = FakeMusa(initialized=False)
        monkeypatch.setattr(torch, "musa", musa, raising=False)
        monkeypatch.setenv("TEST_ARCH_A", "2.2")

        assert arch.target_archs(env=("TEST_ARCH_A",), check_device=True) == (MP22,)
        assert musainfo.calls == []
        assert musa.device_calls == []

    def test_initialized_runtime_is_read_in_process(self, monkeypatch, musainfo):
        musa = FakeMusa(initialized=True, capabilities=[(3, 1), (2, 2), (0, 0)])
        monkeypatch.setattr(torch, "musa", musa, raising=False)

        assert arch._device_archs() == (MP22, MP31)
        assert arch._initialized_device_archs() == (MP22, MP31)
        assert arch.target_archs(env=("TEST_ARCH_A",)) == (MP22, MP31)
        assert "current_device" not in musa.device_calls
        assert musainfo.calls == []

    def test_check_reads_the_initialized_runtime(self, monkeypatch, musainfo):
        musa = FakeMusa(initialized=True, capabilities=[(3, 1), (2, 2)])
        monkeypatch.setattr(torch, "musa", musa, raising=False)
        monkeypatch.setenv("TEST_ARCH_A", "31")

        with pytest.raises(RuntimeError, match=r"visible devices \[mp_22, mp_31\]"):
            arch.target_archs(env=("TEST_ARCH_A",), check_device=True)
        musa.capabilities = [(3, 1), (3, 1)]
        assert arch.target_archs(env=("TEST_ARCH_A",), check_device=True) == (MP31,)
        assert musainfo.calls == []

    def test_in_process_failure(self, monkeypatch, musainfo):
        musa = FakeMusa(initialized=True)
        musa.device_count = lambda: _raise(RuntimeError("driver gone"))
        monkeypatch.setattr(torch, "musa", musa, raising=False)

        assert arch._device_archs() == ()
        assert arch._initialized_device_archs() is None
        assert musainfo.calls == []

    def test_is_initialized_failure_counts_as_uninitialized(self, monkeypatch, musainfo):
        musa = FakeMusa(initialized=False)
        musa.is_initialized = lambda: _raise(RuntimeError("broken"))
        monkeypatch.setattr(torch, "musa", musa, raising=False)
        musainfo.stdout = _musainfo_output("3.1")

        assert arch._initialized_device_archs() is None
        assert arch._device_archs() == (MP31,)
        assert musa.device_calls == []

    def test_musa_without_is_initialized_uses_musainfo(self, monkeypatch, musainfo):
        musa = types.ModuleType("torch.musa")
        monkeypatch.setattr(torch, "musa", musa, raising=False)
        musainfo.stdout = _musainfo_output("3.1")

        assert arch._initialized_device_archs() is None
        assert arch._device_archs() == (MP31,)

    def test_torch_without_musa_uses_musainfo(self, musainfo):
        assert not hasattr(torch, "musa")
        musainfo.stdout = _musainfo_output("2.2")

        assert arch._initialized_device_archs() is None
        assert arch._device_archs() == (MP22,)
