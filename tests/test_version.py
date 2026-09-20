"""Tests for the infix version comparisons in ``torchada._version``.

No GPU and no torch_musa build is required: the proxy is pure string handling.
"""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from torchada._version import VersionComparison, version_of


class TestVersionOf:
    """The accepted shapes of a version source."""

    def test_accepts_a_string(self):
        assert version_of("2.11.0") == "2.11.0"

    def test_accepts_none(self):
        assert isinstance(version_of(None), VersionComparison)
        assert not version_of(None).is_known

    def test_accepts_an_object_with_a_version_attribute(self):
        class _Module:
            __version__ = "2.12.0+musa6.0.0"

        assert version_of(_Module()) >= "2.11.0.post2"
        assert version_of(_Module).is_known  # a module/class works too

    def test_accepts_another_proxy(self):
        proxy = version_of("2.11.0.post1")
        assert version_of(proxy) is proxy

    def test_missing_attribute_is_unknown(self):
        class _Module:
            pass

        assert not version_of(_Module()).is_known


class TestComparisonSemantics:
    """The two documented policies: public version, unknown ranks lowest."""

    def test_local_segment_is_ignored(self):
        # ``+musa5.2.0`` identifies the MUSA stack build, not the fix level.
        assert version_of("2.11.0.post1+musa5.2.0") < "2.11.0.post2"
        assert not (version_of("2.11.0.post1+musa5.2.0") >= "2.11.0.post2")
        assert version_of("2.11.0.post1+musa5.2.0") == version_of("2.11.0.post1")
        assert version_of("2.11.0.post2+musa5.2.0") >= "2.11.0.post2"

    def test_post_releases_compare_semantically(self):
        assert version_of("2.11.0.post10+musa5.2.0") >= "2.11.0.post2"
        assert version_of("2.11.0.post1") < "2.11.0.post10"

    def test_unparsable_version_ranks_lowest(self):
        # The "unknown ⇒ keep the workaround" gates rely on this: any upper
        # bound is satisfied, so the patch stays enabled.
        for unknown in ("not-a-version", "not-a-version+musa5.2.0", "", None):
            proxy = version_of(unknown)
            assert not proxy.is_known
            assert proxy < "2.11.0.post2"
            assert proxy < "0.0.1"
            assert not (proxy >= "2.11.0.post2")

    def test_lower_bound_gate_skips_only_known_versions(self):
        # The mm/bmm gate is a lower bound: an unknown version must NOT be read
        # as "newer than the line" (that would disable the backport).
        minimum = "2.11.0"
        for unknown in ("not-a-version", None):
            proxy = version_of(unknown)
            assert not (proxy.is_known and proxy < minimum)
        assert version_of("2.10.0") < minimum
        assert not (version_of("2.11.0") < minimum)

    def test_operators_work_in_both_operand_orders(self):
        proxy = version_of("2.11.0.post1")
        assert proxy < "2.11.0.post2"
        assert "2.11.0.post2" > proxy
        assert proxy <= "2.11.0.post1"
        assert "2.11.0.post1" >= proxy
        assert proxy != "2.11.0.post2"
        assert not (proxy == "2.11.0.post2")

    def test_hashing_and_repr(self):
        assert len({version_of("2.11.0"), version_of("2.11.0+musa5.2.0")}) == 1
        assert "2.11.0" in repr(version_of("2.11.0"))
        assert "unknown" in repr(version_of("nonsense"))


class TestTorchMusaGate:
    """The gate helper now built on the proxy keeps its old behaviour."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        (
            ("2.10.0", True),
            ("2.11.0", True),
            ("2.11.0.post1+musa5.2.0", True),
            ("2.12.0+musa6.0.0", True),
            ("2.13.0", False),
            ("2.13.0.post1", False),
            ("2.14.0", False),
            ("not-a-version+musa5.2.0", True),
            (None, True),
        ),
    )
    def test_mm_out_dtype_bound_is_a_single_upper_bound(self, version, expected):
        """The arming gate is one inline comparison against the committed fix release.

        Below 2.13.0 the wrappers are armed, from 2.13.0 on nothing is installed,
        and an unknown or unparsable version ranks lowest in ``version_of``, so it
        stays armed.
        """
        assert (version_of(version) < "2.13.0") is expected


class TestOutDtypeArming:
    """The arming decision, driven through the real patch function (no device)."""

    @pytest.mark.parametrize(
        "version,expected",
        (
            ("2.10.0", True),
            ("2.11.0.post1+musa5.2.0", True),
            ("2.12.9+musa6.0.0", True),
            ("2.13.0", False),
            ("2.13.0.post1+musa6.1.0", False),
            ("not-a-version", True),
            (None, True),
        ),
    )
    def test_arming_is_a_single_upper_bound(self, monkeypatch, version, expected):
        import sys
        from types import ModuleType

        import torch

        from torchada import _patch

        musa = ModuleType("fake_torch_musa")
        if version is not None:
            musa.__version__ = version
        installed = []
        # The patch function is guarded by ``requires_import("torch_musa")``, and this
        # test needs the real body to run, so stand in for the import.
        monkeypatch.setitem(sys.modules, "torch_musa", ModuleType("torch_musa"))
        monkeypatch.setattr(_patch, "is_musa_platform", lambda: True)
        monkeypatch.setattr(torch, "musa", musa, raising=False)
        monkeypatch.setattr(_patch, "_original_torch_mm", None)
        monkeypatch.setattr(_patch, "_original_torch_bmm", None)
        monkeypatch.setattr(
            _patch, "_register_jit_builtin_alias", lambda original, wrapper: installed.append(wrapper)
        )
        original_mm, original_bmm = torch.mm, torch.bmm
        try:
            _patch._patch_mm_out_dtype()
        finally:
            torch.mm, torch.bmm = original_mm, original_bmm
            _patch._original_torch_mm = None
            _patch._original_torch_bmm = None

        assert bool(installed) is expected
        pass
