"""
Tests for the shared MUSA infrastructure: device facts and build identity.

Everything here is device-independent.  The device is a fake object exposing the
same attribute names ``torch_musa.get_device_properties()`` reports, which is also
the contract ``torchada.musa.device`` documents.
"""

import dataclasses
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

import torchada  # noqa: F401  (import first: applies patches)
from torchada.musa import build, device, targets  # noqa: E402


class FakeProperties:
    """Stand-in for ``torch_musa.get_device_properties()``."""

    def __init__(self, major=3, minor=1, name="MTT S5000", mp_count=56, warp_size=32, **overrides):
        self.major = major
        self.minor = minor
        self.name = name
        self.multi_processor_count = mp_count
        self.warp_size = warp_size
        self.total_memory = 80 << 30
        self.L2_cache_size = 8 << 20
        # An S5000 board as it reports itself (48 KiB is the legacy per-block
        # default, not what this part returns); matching keeps the fake consistent
        # with the recorded bin profile, so unrelated tests see no mismatch warning.
        self.shared_memory_per_block = 196608
        self.shared_memory_per_block_optin = 196608
        self.shared_memory_per_multiprocessor = 196608
        self.regs_per_multiprocessor = 131072
        self.max_threads_per_block = 1024
        self.max_threads_per_multiprocessor = 3072
        self.uuid = "fake-uuid"
        # A test that needs a board disagreeing with the recorded bin profile says so
        # by attribute name, the way a real stepping would differ.
        for attribute, value in overrides.items():
            setattr(self, attribute, value)


@pytest.fixture
def fake_device(monkeypatch):
    """Point the probe at a fake device; skip the real one entirely."""

    def install(**kwargs):
        props = FakeProperties(**kwargs)
        monkeypatch.setattr(device, "_get_properties", lambda device_id: props)
        device.clear_device_spec_cache()
        return props

    yield install
    device.clear_device_spec_cache()


class TestDeviceSpec:
    def test_capability_tuple_is_an_identity_not_a_threshold(self, fake_device):
        fake_device(major=3, minor=1)
        spec = device.get_device_spec(0)

        assert spec.arch_code == 310
        assert spec.arch_name == "mp_31"
        assert spec.device_capability == (3, 1)
        assert spec.supports_native_fp8

    def test_an_unknown_arch_answers_false_rather_than_a_comparison(self, fake_device):
        """mp_32 must not inherit mp_31's tuning just by being numerically later."""
        fake_device(major=3, minor=2)
        spec = device.get_device_spec(0)

        assert spec.arch_code == 320
        assert not spec.supports_native_fp8
        assert not device.supports_native_fp8(0)
        # The table does not answer "is this arch validated": that is a framework's
        # policy, and an unknown card must not inherit a neighbour's row either way.
        assert spec.arch_spec is targets.UNKNOWN_ARCH

    def test_a_mismatch_is_not_folded_into_false(self, fake_device, monkeypatch):
        """``False`` would send a wrong-arch binary down the generic path.

        The generic path is still that same wrong binary, so the check must raise
        from the convenience wrapper too instead of being answered.
        """
        fake_device(major=3, minor=1)

        def boom(device_id=None):
            raise build.MusaBuildTargetMismatch("built for mp_22")

        monkeypatch.setattr(device, "current_device_spec", boom)
        monkeypatch.setattr(device, "get_device_spec", boom)

        with pytest.raises(build.MusaBuildTargetMismatch):
            device.supports_native_fp8(None)
        with pytest.raises(build.MusaBuildTargetMismatch):
            device.supports_native_fp8(0)

    def test_a_probe_failure_is_still_folded_into_false(self, fake_device, monkeypatch):
        """Only a mismatch propagates; "device not ready" must not enable anything."""
        fake_device(major=3, minor=1)

        def not_ready(device_id=None):
            raise RuntimeError("device not ready")

        monkeypatch.setattr(device, "current_device_spec", not_ready)

        assert device.supports_native_fp8(None) is False

    def test_fp4_is_an_explicit_empty_set(self, fake_device):
        """No MUSA arch has FP4 arithmetic today, S5000 included.

        The question a caller needs answered is "does this arch execute FP4
        instructions", not "can I run an FP4 checkpoint" -- the latter works here
        through dequantisation.  An empty set answers False everywhere, where a
        capability comparison would answer True on S5000.
        """
        fake_device(major=3, minor=1)
        spec = device.get_device_spec(0)

        assert device.NATIVE_FP4_ARCHS == frozenset()
        assert spec.supports_native_fp4 is False
        assert spec.supports_native_fp4 is False

    def test_mp_count_comes_from_the_device_not_the_arch(self, fake_device):
        """Two bins of one arch differ in SM count; the spec must follow the bin."""
        fake_device(mp_count=48)
        small = device.get_device_spec(0)
        device.clear_device_spec_cache()
        fake_device(mp_count=60)
        large = device.get_device_spec(0)

        assert small.arch_name == large.arch_name == "mp_31"
        assert (small.mp_count, large.mp_count) == (48, 60)

    def test_a_failed_probe_is_retried_not_cached(self, monkeypatch):
        """One early failure must not pin every gate for the process."""
        calls = {"n": 0}

        def flaky(device_id):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("device not ready")
            return FakeProperties()

        monkeypatch.setattr(device, "_get_properties", flaky)
        device.clear_device_spec_cache()

        assert device.get_device_spec(0) is None
        recovered = device.get_device_spec(0)
        assert calls["n"] == 2
        assert recovered is not None and recovered.arch_name == "mp_31"

    def test_the_probe_does_not_pass_a_validation_verdict(self, fake_device, caplog):
        """Which kernels are validated is the consumer's policy, not a device fact.

        An arch the table does not list is described, not judged: the probe reports
        its numbers and says nothing about what a framework may run on it.
        """
        fake_device(major=2, minor=2)

        with caplog.at_level("WARNING", logger=device.__name__):
            spec = device.get_device_spec(0)
            device.get_device_spec(0)

        assert spec is not None and spec.arch_name == "mp_22"
        assert not [r for r in caplog.records if "validated" in r.message]


class TestArchStrings:
    def test_arch_code_is_the_value_device_code_compares(self):
        assert device.arch_code(3, 1) == 310
        assert device.arch_code(2, 2) == 220
        assert device.arch_name(3, 1) == "mp_31"


def _write_stamp(tmp_path, monkeypatch, targets, **kwargs):
    """Write a stamp into ``tmp_path`` and import it, dropping any stale module.

    ``read_build_stamp`` goes through the import system, which caches by module
    name, so two test files writing ``_build_stamp`` into different directories
    would otherwise read the first one back.
    """
    monkeypatch.syspath_prepend(str(tmp_path))
    path = build.write_build_stamp(tmp_path, targets, **kwargs)
    monkeypatch.delitem(sys.modules, path.stem, raising=False)
    return path


class TestBuildStamp:
    def test_stamp_roundtrip(self, tmp_path, monkeypatch):
        path = _write_stamp(
            tmp_path, monkeypatch, ["31"], fp8=True, compiler_version="5.2.0", source="environment"
        )

        stamp = build.read_build_stamp(path.stem)

        assert stamp is not None
        assert stamp.targets == ("mp_31",)
        assert stamp.fp8 is True
        assert stamp.mcc_version == "5.2.0"
        assert stamp.source == "environment"

    def test_an_absent_stamp_is_unknown_not_a_mismatch(self):
        assert build.read_build_stamp("torchada.no_such_stamp_module") is None
        # Unknown must not refuse anything: an editable install has no stamp.
        build.assert_build_matches_device(_spec("mp_22"), None)

    def test_a_mismatched_build_is_refused(self, tmp_path, monkeypatch):
        path = _write_stamp(tmp_path, monkeypatch, ["31"], compiler_version="5.2.0")

        with pytest.raises(build.MusaBuildTargetMismatch):
            build.assert_build_matches_device(_spec("mp_22"), build.read_build_stamp(path.stem))

    def test_a_matching_build_is_accepted(self, tmp_path, monkeypatch):
        path = _write_stamp(tmp_path, monkeypatch, ["31"], compiler_version="5.2.0")

        build.assert_build_matches_device(_spec("mp_31"), build.read_build_stamp(path.stem))

    def test_a_stamp_carrying_a_bare_string_is_tolerated(self):
        """A hand-written stamp may hold one name instead of a tuple."""
        stamp = build.BuildStamp(targets="mp_31")

        build.assert_build_matches_device(_spec("mp_31"), stamp)
        with pytest.raises(build.MusaBuildTargetMismatch):
            build.assert_build_matches_device(_spec("mp_22"), stamp)

    def test_the_reader_keeps_a_hand_written_name_whole(self, tmp_path, monkeypatch):
        """The tolerant guard is only reachable if the reader trusts a string.

        ``tuple("mp_31")`` is ``('m','p','_','3','1')``, which used to raise
        ``ValueError`` before any comparison -- so the guard could not be exercised
        from the one function that reads stamps.
        """
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "_build_stamp.py").write_text('MTGPU_TARGETS = "mp_31"\n')
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "pkg._build_stamp", raising=False)

        stamp = build.read_build_stamp("pkg._build_stamp")

        assert stamp.targets == ("mp_31",)
        assert build.assert_build_matches_device(
            _spec("mp_31"), stamp, module_name="pkg._build_stamp", name="pkg"
        )

    def test_the_check_names_the_package_it_is_checking(self):
        """A consumer's log must say whose build is wrong.

        The message used to read "this package", which is ambiguous when the
        library, the framework and the kernel package all use this module.
        """
        with pytest.raises(build.MusaBuildTargetMismatch) as raised:
            build.assert_build_matches_device(
                _spec("mp_31"), build.BuildStamp(("mp_22",)), name="vllm_musa"
            )

        assert str(raised.value).startswith("vllm_musa was built for MUSA arch mp_22")

    def test_the_check_reads_the_stamp_when_given_a_module_name(self, tmp_path, monkeypatch):
        """One call, no separate read: consumer wiring is what it always forgot."""
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / "_one_call_stamp.py").write_text(
            "MTGPU_TARGETS = ('mp_31',)\nENABLE_FP8 = True\n"
        )
        monkeypatch.delitem(sys.modules, "_one_call_stamp", raising=False)

        matched = build.assert_build_matches_device(
            _spec("mp_31"), module_name="_one_call_stamp", name="vllm_musa"
        )
        with pytest.raises(build.MusaBuildTargetMismatch):
            build.assert_build_matches_device(
                _spec("mp_22"), module_name="_one_call_stamp", name="vllm_musa"
            )

        assert matched is True

    def test_the_return_value_distinguishes_matched_from_nothing_to_compare(self):
        stamp = build.BuildStamp(("mp_31",))

        assert build.assert_build_matches_device(_spec("mp_31"), stamp) is True
        assert build.assert_build_matches_device(_spec("mp_31"), None) is False
        assert build.assert_build_matches_device(_spec("mp_31"), build.BuildStamp(())) is False

    def test_an_unstamped_check_can_be_made_loud(self, caplog):
        """A project whose only wrong-arch defence is this check wants to hear that it is off."""
        build._reset_unstamped_warnings()
        with caplog.at_level(logging.WARNING, logger="torchada.musa.build"):
            quiet = build.assert_build_matches_device(_spec("mp_31"), None)
        assert quiet is False
        assert not caplog.text

    def test_an_unstamped_check_warns_once(self, caplog):
        """The check runs on every device query; the notice must not repeat."""
        build._reset_unstamped_warnings()
        with caplog.at_level(logging.WARNING, logger="torchada.musa.build"):
            for _ in range(3):
                result = build.assert_build_matches_device(
                    _spec("mp_31"),
                    None,
                    # Never resolvable: a consumer's real stamp module is present in
                    # any environment that has that consumer installed, and this test
                    # is about the absent-stamp path.
                    module_name="torchada_tests._absent_build_stamp",
                    name="vllm_musa",
                    warn_when_unstamped=True,
                )

        assert result is False
        assert caplog.text.count("no build stamp found for vllm_musa") == 1

    def test_an_unstamped_check_warns_anonymously_too(self, caplog):
        build._reset_unstamped_warnings()
        with caplog.at_level(logging.WARNING, logger="torchada.musa.build"):
            result = build.assert_build_matches_device(
                _spec("mp_31"),
                None,
                module_name="torchada_tests._absent_build_stamp",
                name="vllm_musa",
                warn_when_unstamped=True,
            )

        assert result is False
        assert "no build stamp found for vllm_musa" in caplog.text

    def test_a_hand_written_string_stamp_survives_the_read_path(self, tmp_path, monkeypatch):
        """The read path must not shred the name the guard is asked to tolerate.

        ``tuple("mp_31")`` yields ('m', 'p', ...), which raised ``ValueError``
        before any arch comparison, so the tolerance above was unreachable from
        the only code that actually reads a stamp.
        """
        monkeypatch.syspath_prepend(str(tmp_path))
        (tmp_path / "_hand_written_stamp.py").write_text(
            "MTGPU_TARGETS = 'mp_31'\nENABLE_FP8 = True\nMCC_VERSION = '5.2.0'\n"
        )
        monkeypatch.delitem(sys.modules, "_hand_written_stamp", raising=False)

        stamp = build.read_build_stamp("_hand_written_stamp")

        assert stamp.targets == ("mp_31",)
        build.assert_build_matches_device(_spec("mp_31"), stamp)
        with pytest.raises(build.MusaBuildTargetMismatch):
            build.assert_build_matches_device(_spec("mp_22"), stamp)


class TestArchFlags:
    @pytest.mark.parametrize("target", ["31", "mp_31", "MP31", "3.1", "310", 310])
    def test_every_spelling_lands_on_the_three_digit_define(self, target):
        """``arch_code`` (310) and ``arch_name`` (mp_31) name the same arch.

        ``single_arch_define(targets=[spec.arch_code])`` is one composition away from the
        README example, so accepting only the two-digit form would emit
        ``-D__MUSA_ARCH_LIST__=3100`` and compile the wrong template branch.
        """
        assert build.single_arch_define(targets=[target]) == "-D__MUSA_ARCH_LIST__=310"

    @pytest.mark.parametrize("target", ["3100", "31000", "3", "3101"])
    def test_an_arch_that_cannot_be_rescaled_is_rejected(self, target):
        """Only the exact code mapping is accepted; the rest must fail loudly."""
        with pytest.raises(ValueError, match="not a MUSA arch target"):
            build.single_arch_define(targets=[target])

    def test_the_code_form_names_the_same_cache_directory(self):
        assert build.build_cache_tag(targets=[310], compiler_version="5.2.0") == "mp_31__mcc5.2.0"

    def test_the_define_uses_the_three_digit_form(self):
        """Shipped templates compare against ``== 310``, not the ``31`` of the name."""
        assert build.single_arch_define(targets=["22"]) == "-D__MUSA_ARCH_LIST__=220"

    def test_a_fat_build_cannot_use_the_template_define(self):
        """The rule moved: the build takes a set, the *macro* still cannot.

        ``arch_define`` is the one part of a fat build that has no honest answer,
        so the refusal lives there, with the alternative named.
        """
        with pytest.raises(RuntimeError, match="cannot describe 2 arches"):
            build.single_arch_define(targets=["22", "31"])

    def test_no_target_at_all_is_an_error_not_an_mp31_guess(self, monkeypatch):
        for name in build._TARGET_ENV:
            monkeypatch.delenv(name, raising=False)

        with pytest.raises(RuntimeError, match="cannot determine the MUSA arch"):
            build.single_arch_define(targets=None)

    def test_a_bogus_target_is_rejected(self):
        with pytest.raises(ValueError, match="not a MUSA arch target"):
            build.single_arch_define(targets=["bogus"])

    def test_the_environment_can_pin_the_target(self, monkeypatch):
        monkeypatch.setenv("MTGPU_TARGET", "mp_22")
        assert build.arch_codes() == ["22"]


class TestBuildTag:
    def test_the_tag_carries_arch_and_compiler(self):
        assert build.build_cache_tag(targets=["31"], compiler_version="5.2.0") == "mp_31__mcc5.2.0"

    def test_the_tag_does_not_depend_on_the_multiprocessor_count(self):
        """Codegen input is arch plus compiler; SM count only moves the tiling."""
        assert build.build_cache_tag(
            targets=["31"], compiler_version="5.2.0"
        ) == build.build_cache_tag(targets=["mp_31"], compiler_version="5.2.0")


class TestFatBuild:
    """One wheel per arch is the default; a fat build is supported, not promoted.

    ``mcc`` bundles several ``--offload-arch`` passes into one
    ``__CLANG_OFFLOAD_BUNDLE__``, so the build side has to be able to name a set of
    arches even while the shipped configuration names one.
    """

    def test_arch_codes_takes_a_set(self):
        assert build.arch_codes(["mp_31", "mp_22"]) == ["31", "22"]

    def test_a_repeated_arch_is_one_target(self):
        assert build.arch_codes(["mp_31", "310", "MP31"]) == ["31"]

    def test_arch_flags_are_one_offload_arch_per_target(self):
        assert build.mcc_arch_flags(["mp_22", "mp_31"]) == [
            "--offload-arch=mp_22",
            "--offload-arch=mp_31",
        ]

    def test_the_fat_tag_is_order_independent(self):
        one = build.build_cache_tag(["mp_31", "mp_22"], compiler_version="5.2.0")
        two = build.build_cache_tag(["mp_22", "mp_31"], compiler_version="5.2.0")

        assert one == two == "mp_22_mp_31__mcc5.2.0"

    def test_the_template_macro_refuses_a_list(self):
        """A macro compared against one ``__MUSA_ARCH__`` has no list value."""
        with pytest.raises(RuntimeError, match="cannot describe 2 arches"):
            build.single_arch_define(targets=["mp_31", "mp_22"])

    def test_a_fat_stamp_is_accepted_on_either_arch(self, tmp_path, monkeypatch):
        build.write_build_stamp(
            tmp_path / "pkg", ["mp_22", "mp_31"], fp8=True, compiler_version="5.2.0"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "pkg._build_stamp", raising=False)

        assert build.read_build_stamp("pkg._build_stamp").targets == ("mp_22", "mp_31")
        for arch in ("mp_22", "mp_31"):
            assert build.assert_build_matches_device(
                _spec(arch), module_name="pkg._build_stamp", name="pkg"
            )

    def test_a_fat_stamp_still_refuses_an_arch_it_does_not_carry(self, tmp_path, monkeypatch):
        build.write_build_stamp(
            tmp_path / "pkg2", ["mp_22", "mp_31"], fp8=True, compiler_version="5.2.0"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, "pkg2._build_stamp", raising=False)

        with pytest.raises(build.MusaBuildTargetMismatch, match="mp_32"):
            build.assert_build_matches_device(
                _spec("mp_32"), module_name="pkg2._build_stamp", name="pkg"
            )


class TestMccVersion:
    def test_only_the_anchored_mcc_line_counts(self, monkeypatch):
        """``mcc --version`` prints clang first; a loose match would tag every cache."""
        version_output = (
            "clang version 14.0.0 (musa)\n"
            "Target: x86_64-unknown-linux-gnu\n"
            "mcc version 5.2.0 (build abc)\n"
        )

        class Result:
            stdout = version_output
            stderr = ""

        monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: Result())

        assert build.mcc_version() == "5.2.0"

    def test_an_unparsable_version_is_unknown_not_clang(self, monkeypatch):
        class Result:
            stdout = "clang version 14.0.0\n"
            stderr = ""

        monkeypatch.setattr(build.subprocess, "run", lambda *a, **k: Result())

        assert build.mcc_version() == "unknown"

    @pytest.fixture(autouse=True)
    def _fresh_compiler_version(self):
        """The version is cached per executable, so each test starts uncached."""
        build.clear_mcc_version_cache()
        yield
        build.clear_mcc_version_cache()

    def test_a_missing_compiler_is_unknown(self, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("no mcc")

        monkeypatch.setattr(build.subprocess, "run", boom)

        assert build.mcc_version() == "unknown"


def _spec(arch_name):
    """Minimal duck-typed device spec; ``build`` takes it untyped on purpose."""

    class Spec:
        pass

    spec = Spec()
    spec.arch_name = arch_name
    spec.device_name = "fake device"
    spec.supports_native_fp8 = False
    return spec


class TestTargetSelection:
    """Which arch(es) a build targets: env, then device, then the default.

    These run inside ``setup.py`` before anything exists to inspect, so they are the
    only defence against a wheel stamped for one arch while ``--offload-arch``
    compiled another.  On MUSA that combination does not fault; it returns wrong
    numbers, which is why the refusals are tested as refusals.
    """

    def test_the_environment_decides_first(self):
        targets = build.resolve_targets("mp_31", device_arch="mp_22")

        assert targets.names == ("mp_31",)
        assert targets.source == "environment"
        assert targets.fp8 is True
        assert targets.is_fat is False

    def test_the_device_decides_when_nothing_is_requested(self):
        targets = build.resolve_targets(None, device_arch="mp_22")

        assert targets.names == ("mp_22",)
        assert targets.source == "device"
        assert targets.fp8 is False

    def test_the_default_is_recorded_as_a_default(self):
        """``mp_31`` is a fallback, and the stamp says so rather than "detected"."""
        targets = build.resolve_targets(None, device_arch=None)

        assert targets.names == ("mp_31",)
        assert targets.source == "default"

    def test_a_jit_build_gets_no_default(self):
        """No wheel to check later, so an unanswerable arch must not be guessed."""
        with pytest.raises(build.MusaTargetSelectionError, match="no device answered"):
            build.resolve_targets(None, device_arch=None, default=None)

    @pytest.mark.parametrize(
        "raw", ["mp_21,mp_22", "mp_21;mp_22", "mp_21 mp_22", "210,220", "21;22"]
    )
    def test_every_spelling_of_a_fat_selection_agrees(self, raw):
        targets = build.resolve_targets(raw, device_arch=None)

        assert targets.names == ("mp_21", "mp_22")
        assert targets.is_fat is True
        assert targets.codes == ("21", "22")
        assert targets.compiler_arch_list == "21;22"

    def test_a_repeated_arch_is_one_target(self):
        assert build.parse_targets("mp_31,310,31") == ("mp_31",)

    def test_resolving_a_mixed_set_is_not_this_function_s_decision(self):
        """Whether a mixed build is legal is a source-layout question, not a target one.

        A tree that guards its FP8 kernels per arch can build one artifact for
        several arches; a tree that compiles them unconditionally cannot.  Resolving
        the target set therefore answers, and the check is a separate opt-in call.
        """
        targets = build.resolve_targets("mp_31,mp_22", device_arch=None)

        assert targets.names == ("mp_31", "mp_22")
        assert targets.is_fat is True
        # The flag describes the artifact, so it is "does it carry FP8 at all".
        assert targets.fp8 is True

    def test_a_consumer_whose_layout_needs_one_answer_can_refuse_the_set(self):
        with pytest.raises(build.MusaTargetSelectionError, match="one flag for the whole build"):
            build.uniform_native_fp8(build.resolve_targets("mp_31,mp_22", device_arch=None))

    def test_the_uniform_check_passes_a_set_that_agrees(self):
        single = build.resolve_targets("mp_31", device_arch=None)
        non_fp8 = build.resolve_targets("mp_21,mp_22", device_arch=None)

        assert build.uniform_native_fp8(single) is single
        assert build.uniform_native_fp8(non_fp8) is non_fp8

    def test_a_bare_arch_name_can_be_written_to_a_stamp(self, tmp_path):
        """``read_build_stamp`` tolerates one name; writing one must produce it."""
        build.write_build_stamp(tmp_path, "mp_31")

        assert "MTGPU_TARGETS = ('mp_31',)" in (tmp_path / "_build_stamp.py").read_text()

    def test_an_absent_spec_answers_before_the_warning_that_names_it(self):
        """No probeable device means nothing to compare -- not an AttributeError."""
        assert build.assert_build_matches_device(None, warn_when_unstamped=True) is False

    def test_a_card_the_table_does_not_know_answers_from_the_fallback_row(self, fake_device):
        """Deriving the answers from the row is what keeps an unknown card generic."""
        fake_device(major=3, minor=2)
        spec = device.get_device_spec(0)

        assert spec.arch_spec is targets.UNKNOWN_ARCH
        assert spec.arch_spec.arch_name == ""
        assert spec.supports_native_fp8 is False

    def test_a_stamp_that_names_no_arch_warns_instead_of_answering(self, caplog):
        """An empty stamp recorded nothing, so it is "unknown", not a silent match."""
        spec = _spec("mp_31")
        build._WARNED_UNSTAMPED.discard(("pkg", "pkg._build_stamp"))
        try:
            with caplog.at_level("WARNING"):
                compared = build.assert_build_matches_device(
                    spec,
                    build.BuildStamp(targets=()),
                    module_name="pkg._build_stamp",
                    name="pkg",
                    warn_when_unstamped=True,
                )
        finally:
            build._WARNED_UNSTAMPED.discard(("pkg", "pkg._build_stamp"))

        assert compared is False
        assert "names no arch" in caplog.text

    def test_one_arch_list_in_every_spelling_the_consumers_parse(self):
        """The three spellings of a set, from one list, so they cannot drift."""
        assert build.arch_list_forms(["mp_31", "mp_22"]) == {
            "target": "mp_22,mp_31",
            "torch": "22;31",
            "mate": "2.2;3.1",
        }
        assert build.arch_list_forms(["mp_31"]) == {
            "target": "mp_31",
            "torch": "31",
            "mate": "3.1",
        }

    def test_the_spellings_of_a_set_agree_whatever_the_order(self):
        assert build.arch_list_forms("mp_31,mp_22") == build.arch_list_forms(["mp_22", "mp_31"])

    def test_no_spelling_is_offered_for_an_unsupported_arch(self):
        """A spelling helper that accepts mp_99 would launder it into a build."""
        with pytest.raises(build.MusaTargetSelectionError, match="unknown MUSA arch mp_99"):
            build.arch_list_forms(["mp_31", "mp_99"])

    def test_the_compiler_version_is_asked_once_per_executable(self, monkeypatch):
        """A JIT load asks three times; each ask was a subprocess."""
        calls = []

        def fake_run(command, **_kwargs):
            calls.append(command[0])

            class _Result:
                stdout = "mcc version 5.2.0\n"
                stderr = ""

            return _Result()

        monkeypatch.setattr(build.subprocess, "run", fake_run)
        build.clear_mcc_version_cache()
        try:
            assert build.mcc_version("/opt/musa/bin/mcc") == "5.2.0"
            assert build.mcc_version("/opt/musa/bin/mcc") == "5.2.0"
            assert calls == ["/opt/musa/bin/mcc"]
            build.mcc_version("/other/mcc")
            assert calls == ["/opt/musa/bin/mcc", "/other/mcc"]
        finally:
            build.clear_mcc_version_cache()

    def test_the_stamp_records_the_fp4_answer_too(self, tmp_path):
        """The reader of a stamp asks both questions, so the writer answers both."""
        build.write_build_stamp(tmp_path, "mp_31", fp8=True, compiler_version="5.2.0")
        text = (tmp_path / "_build_stamp.py").read_text()

        assert "ENABLE_FP8 = True" in text
        assert "ENABLE_FP4 = False" in text

    def test_no_supported_arch_carries_fp4(self):
        """The flag exists for the arch that will have it; none does today."""
        for name in build.parse_targets("mp_21,mp_22,mp_31"):
            assert build.resolve_targets(name).fp4 is False

    def test_an_unknown_arch_is_refused(self):
        with pytest.raises(build.MusaTargetSelectionError, match="unknown MUSA arch mp_99"):
            build.resolve_targets("mp_99", device_arch=None)

    def test_the_compiler_list_is_derived_from_the_targets(self):
        targets = build.resolve_targets("mp_21,mp_22", device_arch=None)

        assert build.compiler_arch_list(None, targets) == "21;22"
        assert build.compiler_arch_list("", targets) == "21;22"

    def test_an_agreeing_compiler_list_is_kept(self):
        targets = build.resolve_targets("mp_21,mp_22", device_arch=None)

        assert build.compiler_arch_list("22;21", targets) == "21;22"

    def test_every_spelling_of_an_arch_agrees_with_the_compiler_list(self):
        """``310`` / ``3.1`` / ``MP31`` name the same arch ``parse_targets`` accepts."""
        targets = build.resolve_targets("mp_31", device_arch=None)

        for declared in ("31", "mp_31", "MP31", "310", "3.1"):
            assert build.compiler_arch_list(declared, targets) == "31"

    def test_the_env_is_read_the_way_the_target_parser_reads_it(self, monkeypatch):
        """A ``31;22`` selection must not be rejected for its separator."""
        monkeypatch.setenv("MTGPU_TARGET", "31;22")

        assert list(build.arch_codes()) == ["31", "22"]

    def test_a_disagreeing_compiler_list_is_refused(self):
        """The stamp and ``--offload-arch`` must describe the same hardware."""
        targets = build.resolve_targets("mp_21,mp_22", device_arch=None)

        with pytest.raises(build.MusaTargetSelectionError, match="disagrees with the selected"):
            build.compiler_arch_list("21", targets)

    def test_the_table_is_the_single_source_of_the_capability_sets(self):
        """Every set a call site reads comes from ``targets``, codes included."""
        from torchada.musa import targets

        assert device.NATIVE_FP8_ARCHS == frozenset({310})
        assert device.NATIVE_FP4_ARCHS == frozenset()
        # S5000 is the only FP8 arch, and FP8 does not imply FP4.
        assert targets.get_arch_spec("mp_31").native_fp8 is True
        assert targets.get_arch_spec("mp_31").native_fp4 is False
        assert targets.get_arch_spec("mp_22").native_fp8 is False

    def test_the_4bit_atom_is_named_after_the_instruction(self):
        """``int4=True`` would promise arithmetic the arch does not have in general.

        What exists is one packed SQMMA atom with a signed 4-bit B operand, so the
        field is named for it; a caller asking "may I feed int4 weights" has to read
        the name that answers exactly that.
        """
        from torchada.musa import targets

        assert targets.get_arch_spec("mp_31").s4_sqmma_fp16 is True
        assert targets.get_arch_spec("mp_22").s4_sqmma_fp16 is False
        assert targets.s4_sqmma_arch_names() == ("mp_31",)
        assert targets.native_fp8_arch_names() == ("mp_31",)
        assert targets.native_fp4_arch_names() == ()
        # No generic dtype predicate to misread: the questions are the fields.
        assert not hasattr(targets.get_arch_spec("mp_31"), "supports_native")

    def test_a_tested_compiler_is_evidence_not_a_floor(self):
        """``tested_mcc`` records what built the arch; it refuses nothing.

        "We built and validated mp_31 with 5.2" does not establish that 5.1 cannot
        compile it, so a floor has to come from a project's support policy
        (``check_toolchain(floor=...)``) rather than from this column.
        """
        from torchada.musa import targets

        assert targets.get_arch_spec("mp_31").tested_mcc == "5.2"
        assert targets.get_arch_spec("mp_22").tested_mcc is None
        assert targets.get_arch_spec("mp_21").tested_mcc is None
        assert targets.UNKNOWN_ARCH.tested_mcc is None
        # With no policy floor, an ancient compiler is still only reported.
        assert build.check_toolchain(["mp_31"], version="5.1") == "5.1"

    def test_the_flags_survive_a_one_shot_iterable(self):
        """Two passes read the names; a spent iterator would emit no flag at all.

        An empty flag list is not an error at this point -- it is mcc's own
        multi-arch default, i.e. a binary that runs on the wrong arch and returns
        wrong numbers, which is the failure this module exists to prevent.
        """
        from torchada.musa import build

        assert build.mcc_arch_flags(["mp_31"]) == ["--offload-arch=mp_31"]
        assert build.mcc_arch_flags(("mp_31",)) == ["--offload-arch=mp_31"]
        assert build.mcc_arch_flags(iter(["mp_31"])) == ["--offload-arch=mp_31"]
        assert build.mcc_arch_flags(iter(["mp_22", "mp_31"])) == [
            "--offload-arch=mp_22",
            "--offload-arch=mp_31",
        ]
        assert build.mcc_arch_flags(name for name in ["mp_31"]) == ["--offload-arch=mp_31"]

    def test_no_recorded_floor_means_nothing_is_refused(self, monkeypatch):
        """A tested compiler is evidence, not a requirement; only a policy refuses.

        The table's ``tested_mcc`` is what built the arch once.  Nothing in this
        module turns that into a minimum, so an older compiler builds and the flags
        come out -- the flags path is what ``setup.py`` actually calls.
        """
        monkeypatch.setattr(build, "mcc_version", lambda *a, **k: "5.1.0")

        assert build.check_toolchain(["mp_31"], "5.1.0") == "5.1.0"
        assert build.mcc_arch_flags(["mp_31"]) == ["--offload-arch=mp_31"]
        # No compiler on this host is normal while *consuming* a wheel: skip, not fail.
        assert build.check_toolchain(["mp_31"], "unknown") == "unknown"
        assert build.check_toolchain(["mp_31"], "unknown", floor="5.2") == "unknown"

    def test_a_supplied_floor_names_the_arch_being_built(self):
        assert build.check_toolchain(["mp_31"], "5.2.0", floor="5.2") == "5.2.0"
        with pytest.raises(build.MusaTargetSelectionError, match="mp_31 needs mcc >= 5.2"):
            build.check_toolchain(["mp_31"], "5.1.0", floor="5.2")

    def test_a_bin_profile_is_the_same_facts_without_the_card(self):
        """The tuning layer has to answer for a bin it is not running on."""
        from torchada.musa import profiles

        assert profiles.known_mp_counts("mp_31") == (48, 52, 56, 60)
        assert profiles.known_mp_counts("mp_22") == ()

        profile = profiles.bin_profile("mp_31", 56)
        assert profile is not None
        assert (profile.warp_size, profile.shared_memory_per_block) == (32, 196608)
        # The ceiling a kernel has to ask for, not just the default it gets.
        assert profile.shared_memory_per_block_optin == 196608
        assert (profile.regs_per_multiprocessor, profile.max_threads_per_multiprocessor) == (
            131072,
            3072,
        )
        assert "MUSA-100083" in profile.source

        # Four rows for one arch; the row's other fields are shared.
        assert {p.mp_count for p in profiles.bin_profiles("mp_31")} == {48, 52, 56, 60}
        assert {p.shared_memory_per_multiprocessor for p in profiles.bin_profiles("mp_31")} == {
            196608
        }
        # Not modelled is None, never a neighbouring bin's numbers: an arch with no
        # row at all, and a count that is not one of that arch's bins.
        assert profiles.bin_profile("mp_31", 44) is None
        assert profiles.bin_profile("mp_31", None) is None
        assert profiles.bin_profile("mp_22", 0) is None
        assert profiles.bin_profile("mp_99", 60) is None

    def test_the_bins_a_row_is_reused_for_are_named(self):
        """Reusing one probe across an arch's bins is a claim; it is written down.

        The resource numbers are per arch, so a probe at one bin answers for its
        siblings.  That is only honest while the module says which bins were
        actually probed -- otherwise "one row per arch" silently becomes "I made up
        mp52".
        """
        from torchada.musa import profiles

        assert profiles.verified_mp_counts("mp_31") == (48, 56, 60)
        assert profiles.verified_mp_counts("mp_22") == ()
        # Every bin the table lists is modelled, and every probed bin is one of
        # those too: no verified bin may sit outside the arch's list.
        for arch_name in ("mp_31",):
            modelled = set(profiles.known_mp_counts(arch_name))
            probed = set(profiles.verified_mp_counts(arch_name))
            assert probed <= modelled
            assert all(profiles.bin_profile(arch_name, mp) is not None for mp in modelled)
        # The same row answers for a probed bin and an inferred one, so a caller
        # distinguishes them by asking rather than by comparing numbers.
        probed = profiles.bin_profile("mp_31", 56)
        inferred = profiles.bin_profile("mp_31", 52)
        assert probed is not None and inferred is not None
        assert dataclasses.replace(probed, mp_count=52) == inferred
        assert probed.source == inferred.source

    def test_a_board_that_disagrees_with_its_recorded_profile_says_so(self):
        """The probe wins, but a stale table must not be silently applied."""
        from torchada.musa import profiles

        class Board:
            arch_name = "mp_31"
            mp_count = 56
            warp_size = 32
            shared_memory_per_block = 196608
            shared_memory_per_multiprocessor = 196608
            regs_per_multiprocessor = 131072
            max_threads_per_block = 1024
            max_threads_per_multiprocessor = 3072

        assert profiles.profile_mismatch(Board()) is None

        Board.regs_per_multiprocessor = 65536
        reported = profiles.profile_mismatch(Board())
        assert reported is not None
        assert "regs_per_multiprocessor=65536" in reported and "recorded 131072" in reported

        # The opt-in ceiling is compared too: a part where asking for 192 KiB gets
        # you no more than the default is the case that bites a kernel author.
        Board.regs_per_multiprocessor = 131072
        Board.shared_memory_per_block_optin = 65536
        reported = profiles.profile_mismatch(Board())
        assert reported is not None
        assert "shared_memory_per_block_optin=65536" in reported
        assert "recorded 196608" in reported
        Board.shared_memory_per_block_optin = 196608

        # An unmodelled bin has nothing to disagree with, so it is not reported.
        Board.mp_count = 44
        assert profiles.profile_mismatch(Board()) is None

    def test_a_disagreeing_bin_warns_once_and_the_probe_still_wins(self, fake_device, monkeypatch):
        """Warn, never raise: a new stepping of the card must still run.

        The consumer of this (vllm-musa's tactic layer) keys its tables on
        ``(arch, mp_count)``, so a stale row that says nothing would be applied as if
        it had been measured.  One warning per bin is the contract: two disagreeing
        bins on one heterogeneous node are two warnings, and a repeated call for the
        same bin is not a third.
        """
        warnings = []
        monkeypatch.setattr(
            device.logger, "warning", lambda message, *args: warnings.append(message % args)
        )

        fake_device(minor=1, mp_count=56, regs_per_multiprocessor=65536)
        spec = device.get_device_spec(0)
        other = device.get_device_spec(0)  # cached probe, same bin

        assert len(warnings) == 1, warnings
        assert "mp_31/mp56" in warnings[0] and "regs_per_multiprocessor=65536" in warnings[0]
        # The board is the authority, not the table.
        assert spec.regs_per_multiprocessor == 65536
        assert other is spec

        # A second, different bin of the same arch is its own warning.
        fake_device(minor=1, mp_count=52, regs_per_multiprocessor=65536)
        device.get_device_spec(0)
        assert len(warnings) == 2, warnings
        assert "mp_31/mp52" in warnings[1]

    def test_a_matching_bin_warns_about_nothing(self, fake_device, monkeypatch):
        warnings = []
        monkeypatch.setattr(
            device.logger, "warning", lambda message, *args: warnings.append(message % args)
        )

        fake_device(minor=1, mp_count=60)
        spec = device.get_device_spec(0)

        assert spec.mp_count == 60
        assert spec.bin_profile is not None and spec.bin_profile.mp_count == 60
        assert warnings == []

    def test_a_field_the_probe_does_not_carry_is_not_a_disagreement(self, fake_device, monkeypatch):
        """``_int_attr`` says 0 for an absent key; the row must not be blamed for it.

        This is the case the tolerant two-spelling read exists for: a torch_musa that
        renamed or dropped a property.  Warning "the probe wins, edit the table" there
        would send a reader to fix data that is still correct.
        """
        props = fake_device(minor=1, mp_count=56)
        del props.shared_memory_per_block_optin

        warnings = []
        monkeypatch.setattr(
            device.logger, "warning", lambda message, *args: warnings.append(message % args)
        )
        spec = device.get_device_spec(0)

        assert spec.shared_memory_per_block_optin == 0  # unreported, not "zero bytes"
        assert spec.shared_memory_per_block == 196608
        assert warnings == []

    def test_the_spec_answers_what_a_kernel_may_request(self, fake_device):
        """Occupancy planning needs the opt-in ceiling, not only the default."""
        fake_device(minor=1, shared_memory_per_block_optin=163840)
        spec = device.get_device_spec(0)

        assert spec.shared_memory_per_block == 196608
        assert spec.shared_memory_per_block_optin == 163840

    def test_the_compiler_is_the_one_the_extension_builder_runs(self, monkeypatch):
        """Read the version of the ``mcc`` torch_musa will invoke, not of ``mcc`` on PATH.

        ``torch_musa.utils.musa_extension`` writes its ninja file with ``$PYTORCH_MCC``
        when set and ``$MUSA_HOME/bin/mcc`` otherwise.  Reading a different executable
        would tag a cache directory with a version that produced none of its object
        code -- the very disagreement the tag exists to prevent.
        """
        from torchada.musa import build

        monkeypatch.setenv("MUSA_HOME", "/opt/musa")
        monkeypatch.delenv("PYTORCH_MCC", raising=False)
        monkeypatch.setenv("MCC", "/somewhere/else/mcc")

        # MUSA_HOME answers before PATH or MCC, because that is what the builder does.
        assert build.compiler_executable() == "/opt/musa/bin/mcc"

        monkeypatch.setenv("PYTORCH_MCC", "/ccache/mcc")
        assert build.compiler_executable() == "/ccache/mcc"

        # MUSA_PATH is the same variable under another name, checked before the
        # builder's own PATH and standard-prefix guesses (which the next test pins).
        monkeypatch.delenv("PYTORCH_MCC", raising=False)
        monkeypatch.delenv("MUSA_HOME", raising=False)
        monkeypatch.setenv("MUSA_PATH", "/opt/from-musa-path")
        assert build.compiler_executable() == "/opt/from-musa-path/bin/mcc"

    def test_an_empty_selection_is_refused_rather_than_left_to_mcc(self, monkeypatch):
        """``iter([])`` is truthy, so the emptiness check has to run after a tuple().

        No ``--offload-arch`` at all is not an mcc error: it is mcc's own multi-arch
        default, i.e. an object for an arch nobody selected.  The three entry points
        that derive flags must refuse it, and an explicitly empty selection must not be
        quietly replaced by the environment, which is a different statement.
        """
        for name in build._TARGET_ENV:
            monkeypatch.setenv(name, "mp_22")  # must not be used to rescue the call

        for empty in ([], (), iter(()), (name for name in [])):
            with pytest.raises(build.MusaTargetSelectionError, match="empty MUSA arch"):
                build.mcc_arch_flags(empty)
            with pytest.raises(build.MusaTargetSelectionError, match="empty MUSA arch"):
                build.arch_codes(empty)
            with pytest.raises(build.MusaTargetSelectionError, match="empty MUSA arch"):
                build.single_arch_define(targets=empty)

        # A floor only asks which arches it applies to once it is *not* met, so the
        # arch lookup that refuses the empty set is reached with an unmet floor.
        with pytest.raises(build.MusaTargetSelectionError, match="empty MUSA arch"):
            build.check_toolchain([], version="5.2", floor="9.9")

    def test_a_launcher_in_the_compiler_command_still_reports_the_version(
        self, tmp_path, monkeypatch
    ):
        """``$PYTORCH_MCC`` goes into a Ninja rule, so it may carry a launcher.

        torch_musa's own comment invites ``ccache`` there.  Treating the whole value as
        one filename executed nothing and answered ``unknown``, which then became part
        of the cache tag; splitting it into argv (no shell) reports the version of the
        compiler behind the launcher.
        """
        real = tmp_path / "mcc-real"
        real.write_text(
            '#!/bin/sh\nif [ "$1" = "--version" ]; then\n'
            '  echo "clang version 15.0.0"\n  echo "MUSA mcc version 5.2.0"\n'
            "  exit 0\nfi\nexit 1\n"
        )
        real.chmod(0o755)
        launcher = tmp_path / "ccache"
        launcher.write_text('#!/bin/sh\nexec "$@"\n')
        launcher.chmod(0o755)

        for command in (
            str(real),
            "env {}".format(real),
            "{} {}".format(launcher, real),
        ):
            build.clear_mcc_version_cache()
            monkeypatch.setenv("PYTORCH_MCC", command)

            # The identity is the whole command; only its version is the compiler's.
            assert build.compiler_executable() == command
            assert build.compiler_argv() == tuple(command.split())
            assert build.mcc_version() == "5.2.0"
            assert build.build_cache_tag(["mp_31"], build.mcc_version()) == "mp_31__mcc5.2.0"

    def test_the_compiler_home_follows_the_builder_s_guesses(self, tmp_path, monkeypatch):
        """``MUSA_HOME`` > the ``mcc`` on ``PATH`` > ``/usr/local/musa``.

        The builder reconstructs ``<home>/bin/mcc`` and compiles with it, so a helper
        that stopped at ``mcc`` from ``PATH`` would read the version of a different
        executable -- or of none, on a host where the standard prefix is all there is.
        """
        monkeypatch.delenv("MUSA_HOME", raising=False)
        monkeypatch.delenv("MUSA_PATH", raising=False)
        monkeypatch.delenv("PYTORCH_MCC", raising=False)

        standard = tmp_path / "standard-musa"
        (standard / "bin").mkdir(parents=True)
        (standard / "bin" / "mcc").write_text("#!/bin/sh\nexit 1\n")
        monkeypatch.setattr(build, "MUSA_DEFAULT_HOME", str(standard))

        # Guess 2 first: a mcc on PATH names its own home.
        monkeypatch.setattr(build.shutil, "which", lambda name: "/opt/musa52/bin/mcc")
        assert build.musa_home() == "/opt/musa52"
        assert build.compiler_executable() == "/opt/musa52/bin/mcc"

        # Guess 3 when PATH has none: the standard prefix, if it exists.
        monkeypatch.setattr(build.shutil, "which", lambda name: None)
        assert build.musa_home() == str(standard)
        assert build.compiler_executable() == str(standard / "bin" / "mcc")

        # Nothing at all: no path is invented; the builder raises its own OSError here.
        monkeypatch.setattr(build, "MUSA_DEFAULT_HOME", str(tmp_path / "absent"))
        assert build.musa_home() is None
        assert build.compiler_executable() == "mcc"

        # The environment wins over both guesses.
        monkeypatch.setenv("MUSA_HOME", "/opt/from-env")
        assert build.compiler_executable() == "/opt/from-env/bin/mcc"

    def test_the_version_and_the_tag_name_the_same_compiler(self, monkeypatch):
        """One identity through the tag and the stamp, not two resolutions.

        A call site resolves the executable once and passes the version it read, so the
        cache directory cannot disagree with the compiler that filled it.
        """
        from torchada.musa import build

        monkeypatch.setattr(build, "mcc_version", lambda mcc=None, timeout=10.0: "5.1.0")

        assert build.build_cache_tag(["mp_31"]) == "mp_31__mcc5.1.0"
        assert build.build_cache_tag(["mp_31"], "5.2.0") == "mp_31__mcc5.2.0"

    def test_an_mp_21_build_is_not_refused_for_what_mp_31_records(self, monkeypatch):
        """The reported reproducer: ``MTGPU_TARGET=mp_21`` with a 5.1 compiler.

        The old check iterated the whole table, so ``mp_31``'s higher recorded floor
        rejected a build that never asked for ``mp_31``.
        """
        from torchada.musa import build

        monkeypatch.setenv("MTGPU_TARGET", "mp_21")
        monkeypatch.setattr(build, "mcc_version", lambda *a, **k: "5.1.0")

        assert build.check_toolchain() == "5.1.0"
        assert build.mcc_arch_flags() == ["--offload-arch=mp_21"]
        # A policy floor still refuses, and names the arch that was selected.
        with pytest.raises(build.MusaTargetSelectionError, match="mp_21 needs mcc >= 5.2"):
            build.check_toolchain(floor="5.2")

    def test_the_shorthand_spellings_reach_the_flags(self, monkeypatch):
        """``arch_codes()`` always accepted ``"21"``; the flag path rejected it.

        Every entry point has to normalise the same way, or one spelling builds and the
        other raises halfway through a setup.py.
        """
        from torchada.musa import build

        monkeypatch.setattr(build, "mcc_version", lambda mcc=None, timeout=10.0: "5.2.0")

        for spelling in ("21", "2.1", "210", "mp_21"):
            assert build.mcc_arch_flags([spelling]) == ["--offload-arch=mp_21"]
            assert build.build_cache_tag([spelling]) == "mp_21__mcc5.2.0"

    def test_the_build_only_entry_point_imports_these_modules_without_torch(self, tmp_path):
        """``import torchada_build`` gets the arch table with no torch and no torchada.

        The modules are standard-library only, but a plain ``from torchada.musa
        import build`` still runs ``torchada/__init__`` and therefore torch.  This
        checks the entry point downstream build backends are meant to use, with
        ``torch``, ``torch_musa`` and ``numpy`` blocked outright -- in this process
        they are already imported, so a new import inside one of these files would
        resolve from the cache and the test would pass regardless.
        """
        root = Path(__file__).resolve().parents[1]
        script = """
import os, sys

BLOCKED = ("torch", "torch_musa", "numpy")


class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("this module must not import " + name)
        return None


sys.meta_path.insert(0, Blocker())

import torchada_build as tbuild

print(tbuild.profiles.known_mp_counts("mp_31"))
print(tbuild.profiles.verified_mp_counts("mp_31"))
print(tbuild.build.arch_codes(["mp_31"]))
print(tbuild.targets.native_fp8_arch_names())
print(tbuild.build.build_cache_tag(["mp_31"], "5.2.0"))
assert "torchada" not in sys.modules, "the entry point must not import the package"
assert "torch" not in sys.modules
"""
        out = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "PYTHONPATH": str(root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )

        assert out.returncode == 0, out.stderr
        assert out.stdout.splitlines() == [
            "(48, 52, 56, 60)",
            "(48, 56, 60)",
            "['31']",
            "('mp_31',)",
            "mp_31__mcc5.2.0",
        ]

    def test_a_card_the_table_does_not_list_still_reports_its_own_numbers(self, fake_device):
        fake_device(major=3, minor=2)
        spec = device.get_device_spec(0)
        assert spec.supports_native_fp8 is False
        assert spec.has_s4_sqmma is False
        # Its bin is not in the table, so there is no profile to compare against.
        assert spec.bin_profile is None

    def test_an_arch_the_table_does_not_know_answers_nothing(self):
        """A newer card must not inherit S5000's answers by being "at least" its code."""
        from torchada.musa import targets

        assert targets.get_arch_spec("mp_99") is None
        assert not targets.UNKNOWN_ARCH.native_fp8
        assert not targets.UNKNOWN_ARCH.native_fp4
