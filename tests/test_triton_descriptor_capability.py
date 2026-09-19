import triton.language as tl

from torchada.triton.kernels.moe.kernel import (
    support_tensor_descriptor as kernel_support_tensor_descriptor,
)
from torchada.triton.runtime.fused_moe.fused_moe import (
    support_tensor_descriptor as runtime_support_tensor_descriptor,
)


def test_tensor_descriptor_support_matches_language_api():
    """Do not select the TMA path when Triton lacks its language constructor."""

    expected = hasattr(tl, "make_tensor_descriptor")
    assert runtime_support_tensor_descriptor() is expected
    assert kernel_support_tensor_descriptor() is expected
