"""Tests for the MUSA ``out_dtype`` backport of ``torch.mm`` / ``torch.bmm``.

torch_musa accepts ``out_dtype`` on both ops but its implementation never
writes the result (all zeros for ``mm``, non-zero garbage for ``bmm``), while
the plain overloads and the argument validation are correct.  torchada arms a
wrapper on the affected stack only, and the defect probe that decides whether
the vendor overload is broken runs on the first call that actually passes
``out_dtype`` - probing touches the device and was measured to perturb an
unrelated serving process.  The wrapper tests below need no GPU (a recording op
stands in for the vendor entry point), the contract tests need MUSA hardware.
"""

import logging

import pytest
import torch

from torchada import _patch

ATOL = 1e-2
RTOL = 1.6e-2


def _musa_available() -> bool:
    musa = getattr(torch, "musa", None)
    return musa is not None and musa.is_available()


def _require_musa() -> None:
    if not _musa_available():
        pytest.skip("MUSA device required")


def _recording_op(calls, result=None):
    """Stand-in for the vendor op that records how the wrapper called it."""

    def op(input, mat2, *args, **kwargs):
        calls.append((input, mat2, args, kwargs))
        return result

    return op


def _computing_op(calls):
    """Stand-in for the vendor op that records the call and computes the product."""

    def op(input, mat2, *args, **kwargs):
        calls.append((input, mat2, args, kwargs))
        return input @ mat2

    return op


class TestMMOutDtypeGating:
    """The backport must stay a no-op unless the probe reports a broken op."""

    @staticmethod
    def _install_patch(monkeypatch, broken, version="2.11.0.post1+musa5.2.0"):
        """Run ``_patch_mm_out_dtype`` as if it were applied on MUSA."""
        import sys
        from types import ModuleType, SimpleNamespace

        monkeypatch.setitem(sys.modules, "torch_musa", ModuleType("torch_musa"))
        monkeypatch.setattr(_patch, "is_musa_platform", lambda: True)
        monkeypatch.setattr(_patch, "_original_torch_mm", None)
        monkeypatch.setattr(_patch, "_original_torch_bmm", None)
        monkeypatch.setattr(_patch, "_mm_out_dtype_probe_cache", None)
        monkeypatch.setattr(_patch, "_mm_out_dtype_probe_warned", False)
        # These tests are about the eager path, so the capture state is pinned
        # instead of being read from a host whose checker may raise.
        monkeypatch.setattr(_patch, "_in_device_capture", lambda: False)
        monkeypatch.setattr(torch, "musa", SimpleNamespace(__version__=version), raising=False)
        monkeypatch.setattr(_patch, "_probe_out_dtype_broken_ops", lambda: dict(broken))

    def test_arming_does_not_touch_the_device(self, monkeypatch):
        """The probe must not run while the patch is applied: it perturbs serving.

        Probing means running ``mm``/``bmm`` on the device, which initialises the
        vendor libraries ahead of the host process' own warm-up and changed the
        KV cache budget and decode TPOT of an unrelated vLLM server.
        """
        probes = []
        self._install_patch(monkeypatch, {"mm": True, "bmm": True})
        monkeypatch.setattr(
            _patch, "_probe_out_dtype_broken_ops", lambda: probes.append(1) or {"mm": True, "bmm": True}
        )

        _patch._patch_mm_out_dtype()

        assert probes == []
        assert _patch._mm_out_dtype_probe_cache is None

    def test_healthy_vendor_op_is_delegated_to(self, monkeypatch):
        """A healthy vendor op must not be reimplemented (no GPU needed)."""
        calls = []
        vendor_mm = _recording_op(calls, result="vendor")
        monkeypatch.setattr(torch, "mm", vendor_mm)
        self._install_patch(monkeypatch, {"mm": False, "bmm": False})

        _patch._patch_mm_out_dtype()
        result = torch.mm(torch.randn(2, 3, dtype=torch.bfloat16), torch.randn(3, 2, dtype=torch.bfloat16), out_dtype=torch.float32)

        assert result == "vendor"
        assert len(calls) == 1 and calls[0][3] == {"out_dtype": torch.float32}

    def test_plain_calls_never_probe(self, monkeypatch):
        """Only a call that asks for ``out_dtype`` may trigger the probe."""
        probes = []
        calls = []
        monkeypatch.setattr(torch, "mm", _recording_op(calls, result="vendor"))
        self._install_patch(monkeypatch, {"mm": True, "bmm": True})
        monkeypatch.setattr(
            _patch, "_probe_out_dtype_broken_ops", lambda: probes.append(1) or {"mm": True, "bmm": True}
        )

        _patch._patch_mm_out_dtype()
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        torch.mm(a, b)
        torch.mm(a, b, out=torch.empty(2, 2, dtype=torch.bfloat16))

        assert probes == []
        assert len(calls) == 2

    def test_same_dtype_calls_never_probe(self, monkeypatch):
        """The two same-dtype fast paths are decided without touching the GPU."""
        probes = []
        calls = []
        monkeypatch.setattr(torch, "mm", _recording_op(calls, result="vendor"))
        self._install_patch(monkeypatch, {"mm": True, "bmm": True})
        monkeypatch.setattr(
            _patch, "_probe_out_dtype_broken_ops", lambda: probes.append(1) or {"mm": True, "bmm": True}
        )
        _patch._patch_mm_out_dtype()
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        torch.mm(a, b, out_dtype=torch.bfloat16)
        torch.mm(a.float(), b.float(), out_dtype=torch.float32)

        assert probes == []
        assert len(calls) == 2
        assert all("out_dtype" not in call[3] for call in calls)

    def test_probe_runs_once_and_is_cached(self, monkeypatch):
        probes = []
        monkeypatch.setattr(torch, "mm", _recording_op([], result="vendor"))
        self._install_patch(monkeypatch, {"mm": True, "bmm": True})
        monkeypatch.setattr(
            _patch, "_probe_out_dtype_broken_ops", lambda: probes.append(1) or {"mm": True, "bmm": True}
        )

        _patch._patch_mm_out_dtype()
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        for _ in range(3):
            torch.mm(a, b, out_dtype=torch.float32)

        assert len(probes) == 1

    def test_a_failing_probe_keeps_the_backport_and_is_not_cached(
        self, monkeypatch, caplog
    ):
        """A probe that cannot decide must never be remembered as "healthy".

        The defect this locks: a probe exception used to be read as "nothing to
        patch" and cached for the whole process, so a device error while probing
        would silently disable the backport and let ``out_dtype=float32`` return
        the vendor's zeros again. Only explicit conditions (no MUSA platform, no
        usable device, a binding that does not accept the keyword) may report
        health; every other failure keeps the backport and stays uncached.
        """
        calls = []
        real_probe = _patch._probe_out_dtype_broken_ops
        monkeypatch.setattr(torch, "mm", _computing_op(calls))
        self._install_patch(monkeypatch, {"mm": True, "bmm": True})
        # Use the real probe, not the helper's stub: the whole point is what the
        # probe does with a device error.
        monkeypatch.setattr(_patch, "_probe_out_dtype_broken_ops", real_probe)
        monkeypatch.setattr(_patch, "_musa_devices_available", lambda: True)
        _patch._patch_mm_out_dtype()

        def exploding_op(*args, **kwargs):
            # The way the broken vendor overload fails: a device/kernel error, not
            # a missing keyword.
            raise RuntimeError("Run MUDNN failed in: mudnnMatmulGetWorkspaceSize")

        monkeypatch.setattr(_patch, "_original_torch_mm", exploding_op)
        monkeypatch.setattr(_patch, "_original_torch_bmm", exploding_op)

        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        expected = a.to(torch.float32) @ b.to(torch.float32)
        with caplog.at_level(logging.WARNING, logger=_patch.logger.name):
            results = [torch.mm(a, b, out_dtype=torch.float32) for _ in range(2)]

        for result in results:
            assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
        assert [call[3] for call in calls] == [{}, {}], "the vendor kwarg must not be forwarded"
        assert all(call[0].dtype == torch.float32 for call in calls), "the promoted path is used"
        assert _patch._mm_out_dtype_probe_cache is None, (
            "an undecidable probe must not be stored as a verdict"
        )
        assert caplog.text.count("probe could not decide") == 1, "the warning is throttled"

    def test_a_failing_capture_check_never_probes(self, monkeypatch):
        """A capture check that raises answers "capturing", so nothing is probed.

        Locks the direction of the default: an exception here used to answer "not
        capturing" and ran the probe *inside* a capture - the failure mode that
        both kills the capture and latches a wrong "healthy" verdict. The wrapper
        must keep using the emulation and resolve later instead.
        """
        probes = []
        calls = []
        real_capture_check = _patch._in_device_capture
        monkeypatch.setattr(torch, "mm", _computing_op(calls))
        self._install_patch(monkeypatch, {"mm": True, "bmm": True})
        # Undo the helper's pin so the real capture check (with the exploding
        # checker installed below) is what the wrapper consults.
        monkeypatch.setattr(_patch, "_in_device_capture", real_capture_check)
        monkeypatch.setattr(
            _patch,
            "_probe_out_dtype_broken_ops",
            lambda: probes.append(1) or {"mm": True, "bmm": True},
        )

        def exploding_checker():
            raise RuntimeError("no device to query")

        for owner in (getattr(torch, "musa", None), torch.cuda):
            if owner is not None:
                monkeypatch.setattr(
                    owner, "is_current_stream_capturing", exploding_checker, raising=False
                )

        assert _patch._in_device_capture() is True

        _patch._patch_mm_out_dtype()
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        result = torch.mm(a, b, out_dtype=torch.float32)

        assert probes == [], "no probe may run when the capture state is unknown"
        assert [call[3] for call in calls] == [{}], "the vendor kwarg must not be forwarded"
        assert all(call[0].dtype == torch.float32 for call in calls)
        assert torch.allclose(result, a.to(torch.float32) @ b.to(torch.float32), atol=1e-5, rtol=1e-5)

    def test_healthy_bmm_probe_is_resolved_on_its_own_op(self, monkeypatch):
        """A broken ``mm`` must not make a healthy ``bmm`` take the backport."""
        calls = []
        monkeypatch.setattr(torch, "bmm", _recording_op(calls, result="vendor"))
        self._install_patch(monkeypatch, {"mm": True, "bmm": False})

        _patch._patch_mm_out_dtype()
        a = torch.randn(1, 2, 3, dtype=torch.bfloat16)
        b = torch.randn(1, 3, 2, dtype=torch.bfloat16)
        result = torch.bmm(a, b, out_dtype=torch.float32)

        assert result == "vendor"
        assert len(calls) == 1

    def test_unknown_torch_musa_version_keeps_the_patch_enabled(self, monkeypatch):
        """An unparsable torch_musa version must not disable the backport."""
        original_mm = _recording_op([])
        monkeypatch.setattr(torch, "mm", original_mm)
        self._install_patch(monkeypatch, {"mm": True, "bmm": True}, version="not-a-version")

        _patch._patch_mm_out_dtype()

        assert torch.mm is not original_mm

    def test_below_the_committed_fix_release_is_armed(self, monkeypatch):
        """Everything below 2.13.0 is armed, including releases older than 2.11.0."""
        original_mm = _recording_op([])
        monkeypatch.setattr(torch, "mm", original_mm)
        self._install_patch(monkeypatch, {"mm": True, "bmm": True}, version="2.7.1+musa4.3.0")

        _patch._patch_mm_out_dtype()

        assert torch.mm is not original_mm


class TestMMOutDtypeWrapper:
    """Wrapper semantics on CPU: a recording op stands in for the MUSA op."""

    @pytest.fixture(autouse=True)
    def _broken_probe(self, monkeypatch):
        """Wrapper tests exercise the broken stack, i.e. the backport path."""
        monkeypatch.setattr(_patch, "_mm_out_dtype_probe_cache", {"mm": True, "bmm": True})

    @staticmethod
    def _wrap(calls, result=None, op_name="mm"):
        return _patch._wrap_mm_out_dtype(_recording_op(calls, result), op_name)

    def test_plain_call_is_forwarded_unchanged(self):
        """No ``out_dtype``: the call reaches the original untouched."""
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b)

        assert calls == [(a, b, (), {})]

    def test_out_dtype_float32_promotes_both_operands(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, out_dtype=torch.float32)

        called_a, called_b, args, kwargs = calls[0]
        assert called_a.dtype == torch.float32
        assert called_b.dtype == torch.float32
        assert args == ()
        assert "out_dtype" not in kwargs

    def test_fp16_inputs_are_promoted_too(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.float16)
        b = torch.randn(3, 2, dtype=torch.float16)

        wrapped(a, b, out_dtype=torch.float32)

        called_a, called_b, _, kwargs = calls[0]
        assert called_a.dtype == torch.float32
        assert called_b.dtype == torch.float32
        assert kwargs == {}

    def test_same_dtype_out_dtype_drops_the_keyword(self):
        """``out_dtype == input dtype`` is the plain op: the spy must see no kwarg."""
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, out_dtype=torch.bfloat16)

        assert len(calls) == 1
        called_a, called_b, args, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert args == ()
        assert kwargs == {}
        assert "out_dtype" not in kwargs

    def test_fp32_out_dtype_on_fp32_inputs_drops_the_keyword(self):
        """fp32 in / fp32 out is the same no-op fast path as same-dtype."""
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(3, 2, dtype=torch.float32)

        wrapped(a, b, out_dtype=torch.float32)

        assert len(calls) == 1
        called_a, called_b, args, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert args == ()
        assert kwargs == {}
        assert "out_dtype" not in kwargs

    def test_the_probe_is_not_resolved_inside_a_capture(self, monkeypatch):
        """No probe and no device synchronization while a graph is capturing."""
        calls = []
        probes = []

        def _probe():
            probes.append(True)
            return {"mm": True, "bmm": True}

        monkeypatch.setattr(_patch, "_mm_out_dtype_probe_cache", None)
        monkeypatch.setattr(_patch, "_probe_out_dtype_broken_ops", _probe)
        monkeypatch.setattr(_patch, "_in_device_capture", lambda: True)
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, out_dtype=torch.float32)

        assert probes == []
        called_a, called_b, _, kwargs = calls[0]
        assert called_a.dtype == torch.float32
        assert called_b.dtype == torch.float32
        assert "out_dtype" not in kwargs

    def test_illegal_dtype_pair_keeps_the_vendor_validation(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, out_dtype=torch.float16)

        called_a, called_b, args, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert args == ()
        assert kwargs == {"out_dtype": torch.float16}

    def test_mismatched_input_dtypes_keep_the_vendor_validation(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.float32)

        wrapped(a, b, out_dtype=torch.float32)

        called_a, called_b, _, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert kwargs == {"out_dtype": torch.float32}

    def test_positional_out_dtype_is_backported(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, torch.float32)

        called_a, called_b, args, kwargs = calls[0]
        assert called_a.dtype == torch.float32
        assert called_b.dtype == torch.float32
        assert args == ()
        assert kwargs == {}

    def test_over_long_argument_lists_are_forwarded(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, torch.float32, torch.float32)

        called_a, called_b, args, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert args == (torch.float32, torch.float32)
        assert kwargs == {}

    def test_positional_and_keyword_out_dtype_are_forwarded(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, torch.float32, out_dtype=torch.float32)

        called_a, called_b, args, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert args == (torch.float32,)
        assert kwargs == {"out_dtype": torch.float32}

    def test_non_tensor_operands_are_forwarded(self):
        calls = []
        wrapped = self._wrap(calls)

        wrapped([[1.0, 2.0]], [[3.0], [4.0]], out_dtype=torch.float32)

        assert calls[0][0] == [[1.0, 2.0]]
        assert calls[0][3] == {"out_dtype": torch.float32}

    def test_out_tensor_receives_the_result(self):
        result = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        calls = []
        wrapped = self._wrap(calls, result=result)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        out = torch.empty(2, 2, dtype=torch.float32)

        returned = wrapped(a, b, out=out, out_dtype=torch.float32)

        assert returned is out
        assert torch.equal(out, result)

    def test_out_tensor_of_the_wrong_dtype_keeps_the_vendor_validation(self):
        calls = []
        wrapped = self._wrap(calls, result=torch.zeros(2, 2, dtype=torch.float32))
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)
        out = torch.empty(2, 2, dtype=torch.bfloat16)

        wrapped(a, b, out=out, out_dtype=torch.float32)

        # The fp32 promotion runs first, then the dtype mismatch is handed back
        # to the vendor op with both keywords intact.
        assert calls[0][0].dtype == torch.float32
        assert calls[-1][3] == {"out_dtype": torch.float32, "out": out}

    def test_plain_call_with_out_tensor_is_forwarded(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(3, 2, dtype=torch.float32)
        out = torch.empty(2, 2, dtype=torch.float32)

        wrapped(a, b, out=out)

        assert calls[0][3] == {"out": out}


@pytest.mark.musa
class TestMMOutDtypeContract:
    """Hardware contract: what CUDA promises must hold on MUSA."""

    @staticmethod
    def _pair(shape, dtype):
        a = torch.randn(*shape, device="musa", dtype=dtype)
        b = torch.randn(shape[-1], 5, device="musa", dtype=dtype)
        return a, b

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_mm_out_dtype_float32_matches_fp32_reference(self, dtype):
        _require_musa()
        a, b = self._pair((7, 256), dtype)
        reference = torch.mm(a.to(torch.float32), b.to(torch.float32))

        result = torch.mm(a, b, out_dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.shape == reference.shape
        assert result.abs().max().item() > 0.0, "out_dtype result is all zeros"
        assert torch.allclose(result, reference, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_bmm_out_dtype_float32_matches_fp32_reference(self, dtype):
        _require_musa()
        a, b = self._pair((7, 256), dtype)
        a, b = a.unsqueeze(0), b.unsqueeze(0)
        reference = torch.bmm(a.to(torch.float32), b.to(torch.float32))

        result = torch.bmm(a, b, out_dtype=torch.float32)

        assert result.dtype == torch.float32
        assert result.abs().max().item() > 0.0, "out_dtype result is all zeros"
        assert torch.allclose(result, reference, atol=ATOL, rtol=RTOL)

    def test_positional_out_dtype_is_backported(self):
        _require_musa()
        a, b = self._pair((7, 256), torch.bfloat16)
        reference = torch.mm(a.to(torch.float32), b.to(torch.float32))

        result = torch.mm(a, b, torch.float32)

        assert result.dtype == torch.float32
        assert torch.allclose(result, reference, atol=ATOL, rtol=RTOL)

        a, b = a.unsqueeze(0), b.unsqueeze(0)
        assert torch.allclose(
            torch.bmm(a, b, torch.float32),
            torch.bmm(a.to(torch.float32), b.to(torch.float32)),
            atol=ATOL,
            rtol=RTOL,
        )

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_same_dtype_out_dtype_matches_the_plain_call(self, dtype):
        _require_musa()
        a, b = self._pair((7, 256), dtype)

        assert torch.equal(torch.mm(a, b, out_dtype=dtype), torch.mm(a, b))

        a, b = a.unsqueeze(0), b.unsqueeze(0)
        assert torch.equal(torch.bmm(a, b, out_dtype=dtype), torch.bmm(a, b))

    def test_fp32_out_dtype_is_bitwise_equal_to_the_plain_call(self):
        _require_musa()
        a, b = self._pair((7, 256), torch.float32)

        assert torch.equal(torch.mm(a, b, out_dtype=torch.float32), torch.mm(a, b))

    @pytest.mark.parametrize(
        "in_dtype,out_dtype",
        [
            (torch.bfloat16, torch.float16),
            (torch.float32, torch.float16),
            (torch.float32, torch.bfloat16),
        ],
    )
    def test_illegal_out_dtype_still_raises(self, in_dtype, out_dtype):
        _require_musa()
        a, b = self._pair((7, 256), in_dtype)

        with pytest.raises(RuntimeError, match="out_dtype must be the same as input dtype"):
            torch.mm(a, b, out_dtype=out_dtype)

        # bmm validates one stage later than mm on MUSA (the rejection comes from
        # the kernel rather than from the binding), so assert only that the patch
        # keeps the vendor's own error instead of substituting one.
        a, b = a.unsqueeze(0), b.unsqueeze(0)
        vendor_bmm = _patch._original_torch_bmm or torch.bmm
        with pytest.raises(RuntimeError) as vendor_error:
            vendor_bmm(a, b, out_dtype=out_dtype)
        with pytest.raises(RuntimeError) as patched_error:
            torch.bmm(a, b, out_dtype=out_dtype)
        assert str(patched_error.value) == str(vendor_error.value)

    def test_fp16_out_dtype_on_fp32_inputs_still_raises(self):
        _require_musa()
        a, b = self._pair((7, 256), torch.float32)

        with pytest.raises(RuntimeError, match="out_dtype must be the same as input dtype"):
            torch.mm(a, b, out_dtype=torch.float16)

    def test_capture_before_the_first_eager_call_uses_the_emulation_path(
        self, monkeypatch, caplog
    ):
        """A capture must not resolve the verdict, so it records the emulation path.

        If the first ``out_dtype`` call happens inside a MUSA graph capture, the
        wrapper treats the unresolved verdict as untrusted: the capture contains
        the fp32 emulation, no probe runs, and nothing synchronizes (a graph is
        static, so that graph keeps the emulation until it is re-captured). One
        eager call afterwards resolves the real verdict and logs it, which is what
        restores the vendor path for later captures.

        This is a necessity, not an optimization. Measured on MUSA with the guard
        disabled and a fresh wrapper whose first ``out_dtype`` call happens inside
        a capture: the capture dies with ``RuntimeError: MUSA error: operation
        failed due to a previous error during capture`` (the probe runs the vendor
        overloads and a device readback on a stream that must not block), and -
        worse than the failure - the probe swallows it and caches
        ``{'mm': False, 'bmm': False}``, i.e. "the vendor overload is healthy", so
        every later call would be sent to the broken overload and return zeros.
        The guard removes that failure mode: the capture records the emulation and
        the verdict is resolved later, from eager code.

        A fresh wrapper is built here because the verdict is remembered per
        wrapper instance: the module-level ``torch.mm`` may already be resolved by
        an earlier test in the session.
        """
        probes = []
        original = (
            _patch._original_torch_mm if _patch._original_torch_mm is not None else torch.mm
        )
        monkeypatch.setattr(_patch, "_mm_out_dtype_probe_cache", None)
        monkeypatch.setattr(
            _patch,
            "_probe_out_dtype_broken_ops",
            lambda: probes.append(True) or {"mm": True, "bmm": True},
        )
        wrapped = _patch._wrap_mm_out_dtype(original, "mm")
        a = torch.randn(32, 32, dtype=torch.bfloat16, device="musa")
        b = torch.randn(32, 32, dtype=torch.bfloat16, device="musa")
        reference = torch.mm(a.float(), b.float())

        graph = torch.musa.MUSAGraph()
        with torch.musa.graph(graph):
            captured = wrapped(a, b, out_dtype=torch.float32)
        torch.musa.synchronize()

        assert probes == [], "the verdict must not be resolved inside a capture"

        graph.replay()
        torch.musa.synchronize()
        assert torch.allclose(captured, reference, atol=1e-2, rtol=1e-2)

        with caplog.at_level(logging.INFO, logger=_patch.logger.name):
            eager = wrapped(a, b, out_dtype=torch.float32)

        assert probes == [True], "the next eager call resolves the verdict exactly once"
        assert torch.allclose(eager, reference, atol=1e-2, rtol=1e-2)
        assert "ignores out_dtype" in caplog.text
        assert "backport emulation in use" in caplog.text

    def test_backport_is_armed_and_the_probe_resolves_lazily(self):
        """A qualifying stack is armed at import; the verdict comes on first use."""
        _require_musa()
        from torchada._version import version_of

        gated = version_of(torch.musa.__version__) < "2.13.0"

        assert (getattr(torch.mm, "__wrapped__", None) is not None) == gated
        if not gated:
            return

        a = torch.randn(2, 3, dtype=torch.bfloat16, device="musa")
        b = torch.randn(3, 2, dtype=torch.bfloat16, device="musa")
        result = torch.mm(a, b, out_dtype=torch.float32)

        assert set(_patch._mm_out_dtype_probe_cache) == {"mm", "bmm"}, "the first call resolves the verdict"
        if _patch._mm_out_dtype_probe_cache["mm"]:
            reference = a.to(torch.float32) @ b.to(torch.float32)
            assert torch.allclose(result.cpu(), reference.cpu(), atol=ATOL, rtol=RTOL)
        else:
            assert result.dtype == torch.float32

    def test_probe_verdict_matches_a_direct_measurement(self):
        """A reported-broken op must really be broken, and vice versa."""
        _require_musa()
        vendor_mm = _patch._original_torch_mm or torch.mm
        a = torch.arange(1, 33, dtype=torch.float32, device="musa").reshape(4, 8)
        b = torch.arange(1, 41, dtype=torch.float32, device="musa").reshape(8, 5)
        measured = not torch.equal(vendor_mm(a, b, out_dtype=torch.float32), vendor_mm(a, b))

        assert _patch._probe_out_dtype_broken_ops()["mm"] == measured
