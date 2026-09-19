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

    def test_pre_2_11_version_needs_no_backport(self, monkeypatch):
        original_mm = _recording_op([])
        monkeypatch.setattr(torch, "mm", original_mm)
        self._install_patch(monkeypatch, {"mm": True, "bmm": True}, version="2.7.1+musa4.3.0")

        _patch._patch_mm_out_dtype()

        assert torch.mm is original_mm


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

    def test_same_dtype_out_dtype_uses_the_plain_op(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.bfloat16)
        b = torch.randn(3, 2, dtype=torch.bfloat16)

        wrapped(a, b, out_dtype=torch.bfloat16)

        called_a, called_b, args, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert args == ()
        assert kwargs == {}

    def test_fp32_out_dtype_on_fp32_inputs_uses_the_plain_op(self):
        calls = []
        wrapped = self._wrap(calls)
        a = torch.randn(2, 3, dtype=torch.float32)
        b = torch.randn(3, 2, dtype=torch.float32)

        wrapped(a, b, out_dtype=torch.float32)

        called_a, called_b, _, kwargs = calls[0]
        assert called_a is a
        assert called_b is b
        assert kwargs == {}

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

    def test_backport_is_armed_and_the_probe_resolves_lazily(self):
        """A qualifying stack is armed at import; the verdict comes on first use."""
        _require_musa()
        gated = _patch._torch_musa_may_break_mm_out_dtype(torch.musa.__version__)

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
