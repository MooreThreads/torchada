import pytest
import triton.language as tl

from torchada.triton.kernels.moe.kernel import (
    support_tensor_descriptor as kernel_support_tensor_descriptor,
)
from torchada.triton.musa_compat import triton_backend_is_musa
from torchada.triton.runtime.fused_moe.fused_moe import (
    support_tensor_descriptor as runtime_support_tensor_descriptor,
)


def test_tensor_descriptor_support_matches_language_api():
    """Off the MUSA backend, select the TMA path only when Triton has descriptor support."""

    if triton_backend_is_musa():
        pytest.skip("Triton backend is MUSA")
    expected = hasattr(tl, "make_tensor_descriptor")
    assert runtime_support_tensor_descriptor() is expected
    assert kernel_support_tensor_descriptor() is expected


def test_tensor_descriptor_disabled_on_musa_backend():
    """The host-TensorDescriptor TMA path is not validated on the MUSA backend."""

    if not triton_backend_is_musa():
        pytest.skip("Triton backend is not MUSA")
    assert runtime_support_tensor_descriptor() is False
    assert kernel_support_tensor_descriptor() is False
