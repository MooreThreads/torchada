import functools
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from torchada.triton.kernels.moe.kernel import invoke_fused_moe_kernel
from torchada.triton.runtime.fused_moe.config import (
    get_config_dtype_str,
    try_get_optimal_moe_config,
)

try:
    _support_tensor_descriptor = True
except:
    _support_tensor_descriptor = False


def support_tensor_descriptor():
    return _support_tensor_descriptor


def _is_current_stream_capturing() -> bool:
    """Return whether the active accelerator stream is graph-capturing.

    The CUDA compatibility namespace can exist on CPU-only test hosts while
    the underlying capture query is unavailable.  Treat that case as an
    ordinary eager call; on an actual capture stream the backend query is
    expected to return a boolean.
    """

    query = getattr(torch.cuda, "is_current_stream_capturing", None)
    if query is None:
        return False
    try:
        return bool(query())
    except (AttributeError, RuntimeError):
        return False


@functools.lru_cache()
def _down_moe_use_tma():
    return support_tensor_descriptor()


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.
    """

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # Try to import from sgl_kernel
    try:
        from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

        cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)

        sgl_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    except (ImportError, AttributeError):
        pass

    # Try to import from vllm._custom_ops
    try:
        from vllm._custom_ops import moe_align_block_size as vllm_moe_align_block_size

        vllm_moe_align_block_size(
            topk_ids,
            num_experts,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            None,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    except (ImportError, AttributeError):
        if _is_current_stream_capturing():
            raise RuntimeError(
                "MoE graph capture requires the native moe_align_block_size "
                "operator; the Python fallback is not graph-safe"
            ) from None
        # Standalone torchada tuning images may contain vLLM Python sources
        # without the optional _moe_C extension. Keep the reference path
        # usable for correctness/tuning; production vLLM uses the native op.
        flat_ids = topk_ids.reshape(-1)
        routes = []
        block_experts = []
        for expert in range(num_experts):
            positions = torch.nonzero(flat_ids == expert, as_tuple=False).flatten()
            if positions.numel() == 0:
                continue
            pad = (-positions.numel()) % block_size
            if pad:
                positions = torch.cat(
                    [positions, torch.full_like(positions[:1], flat_ids.numel()).expand(pad)]
                )
            routes.append(positions)
            block_experts.extend([expert] * (positions.numel() // block_size))
        if routes:
            sorted_routes = torch.cat(routes)
            blocks = torch.tensor(block_experts, dtype=torch.int32, device=topk_ids.device)
        else:
            sorted_routes = torch.empty(0, dtype=torch.int32, device=topk_ids.device)
            blocks = torch.empty(0, dtype=torch.int32, device=topk_ids.device)
        sorted_ids[: sorted_routes.numel()].copy_(sorted_routes)
        expert_ids[: blocks.numel()].copy_(blocks)
        num_tokens_post_pad[0] = sorted_routes.numel()
        return sorted_ids, expert_ids, num_tokens_post_pad


def _prepare_fused_moe_run(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[List[int]],
):
    """Resolve config, down_config, TMA flag, and aligned expert routing ids.

    Shared by ``fused_experts_impl`` and ``pre_permute_standard_to_triton`` so
    both paths compute alignment from the same source.
    """
    padded_size = 128
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None:
        padded_size = 0

    num_tokens = hidden_states.shape[0]
    E = w1.shape[0]
    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    config, (down_config, _) = try_get_optimal_moe_config(
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
        config_dtype,
        num_tokens,
        block_shape=block_shape,
        per_channel_quant=per_channel_quant,
        return_down_config=True,
    )
    down_moe_use_tma = (
        _down_moe_use_tma() and down_config is not None and down_config.pop("USE_TMA", False)
    )

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E
    )

    return (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    )


def _fused_moe_kernel_sequence(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    config: Dict[str, Any],
    down_config: Optional[Dict[str, Any]],
    down_moe_use_tma: bool,
    *,
    b1: Optional[torch.Tensor],
    b2: Optional[torch.Tensor],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    w1_scale: Optional[torch.Tensor],
    w2_scale: Optional[torch.Tensor],
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    block_shape: Optional[List[int]],
    activation: str,
    is_gated: bool,
    no_combine: bool,
    inplace: bool,
    apply_router_weight_on_input: bool,
    routed_scaling_factor: Optional[float],
    gemm1_alpha: Optional[float],
    gemm1_limit: Optional[float],
    filter_expert: bool,
    hooks: Optional[Any] = None,
) -> torch.Tensor:
    """Run the MoE kernel/activation/kernel/combine sequence in a single shot.

    Inputs are already aligned and the block-size config is already resolved.
    Supports optional LoRA hooks that fire between the two kernels and before
    combine. Returns ``out_hidden_states``.
    """
    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    topk = topk_ids.shape[1]
    if (is_gated and activation != "silu") or (not is_gated and activation != "relu2_no_mul"):
        raise ValueError(
            f"Unsupported MoE activation/layout: activation={activation!r}, " f"is_gated={is_gated}"
        )
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    padded_tokens = (
        min(num_tokens * topk, E + 1) * (config["BLOCK_SIZE_M"] - 1) if down_moe_use_tma else 0
    )
    total_tokens = num_tokens * topk + padded_tokens

    if no_combine:
        assert not inplace
        out_hidden_states = torch.empty(
            (num_tokens, topk, w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
    elif inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    use_fused_moe_sum_all_reduce = False

    intermediate_cache1 = torch.empty(
        (total_tokens, N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    invoke_fused_moe_kernel(
        hidden_states,
        w1,
        b1,
        intermediate_cache1,
        a1_scale,
        w1_scale,
        w1_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk,
        config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        c_sorted=down_moe_use_tma,
        filter_expert=filter_expert,
    )

    if hooks and hooks.after_gate_up:
        # Hooks expect intermediate_cache1 shaped (num_tokens, topk, N); the
        # underlying buffer is laid out as (total_tokens, N) where
        # total_tokens = num_tokens * topk (+ TMA padding). Slice off any
        # padding and reshape for the hook, which writes in-place on the view.
        hooks.after_gate_up(
            hidden_states,
            intermediate_cache1[: num_tokens * topk].view(num_tokens, topk, N),
            topk_weights,
            topk_ids,
        )

    if is_gated:
        intermediate_cache2 = torch.empty(
            (total_tokens, N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        gate, up = intermediate_cache1.chunk(2, dim=-1)
        intermediate_cache2.copy_(torch.nn.functional.silu(gate) * up)
    else:
        intermediate_cache2 = torch.empty_like(intermediate_cache1)
        # Nemotron-H's relu2 path is non-gated: relu(x)^2, with no second
        # projection half to multiply. Keep this explicit so the tuner cannot
        # benchmark an uninitialized intermediate buffer.
        intermediate_cache2.copy_(torch.relu(intermediate_cache1).square())

    intermediate_cache3 = torch.empty(
        (num_tokens, topk, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # LoRA hooks force the second kernel to write to intermediate_cache3 so
    # hooks.after_down can inspect/modify it before reduction.
    _use_intermediate = not no_combine and (topk != 1 or hooks)

    out_slice = None
    if use_fused_moe_sum_all_reduce:
        out_slice = out_hidden_states
        out_slice.zero_()

    invoke_fused_moe_kernel(
        intermediate_cache2,
        w2,
        b2,
        (
            out_slice
            if use_fused_moe_sum_all_reduce
            else (
                intermediate_cache3
                if no_combine or _use_intermediate
                else out_hidden_states.unsqueeze(0)
            )
        ),
        a2_scale,
        w2_scale,
        w2_zp,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        down_config or config,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
        a_use_tma=down_moe_use_tma,
        b_use_tma=down_moe_use_tma,
        filter_expert=filter_expert,
        fuse_sum_all_reduce=use_fused_moe_sum_all_reduce,
        router_topk=topk,
    )

    if no_combine:
        return intermediate_cache3

    if _use_intermediate:
        combined = intermediate_cache3
        if apply_router_weight_on_input:
            combined = combined * topk_weights.to(combined.dtype).unsqueeze(-1)
        combined = combined.float().sum(dim=1).to(out_hidden_states.dtype)
        out_hidden_states.copy_(combined)
    if routed_scaling_factor is not None:
        out_hidden_states.mul_(routed_scaling_factor)
    return out_hidden_states


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    inplace: bool = False,
    activation: str = "silu",
    is_gated: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    no_combine: bool = False,
    routed_scaling_factor: Optional[float] = None,
    gemm1_alpha: Optional[float] = None,
    gemm1_limit: Optional[float] = None,
    filter_expert: bool = True,
):
    if (is_gated and activation != "silu") or (not is_gated and activation != "relu2_no_mul"):
        raise ValueError(
            f"Unsupported MoE activation/layout: activation={activation!r}, " f"is_gated={is_gated}"
        )
    padded_size = 128
    if not (use_fp8_w8a8 or use_int8_w8a8) or block_shape is not None:
        padded_size = 0

    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2] - padded_size, f"Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    (
        config,
        down_config,
        down_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        w1,
        w2,
        topk_ids,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        block_shape=block_shape,
    )

    return _fused_moe_kernel_sequence(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        config,
        down_config,
        down_moe_use_tma,
        b1=b1,
        b2=b2,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        apply_router_weight_on_input=apply_router_weight_on_input,
        routed_scaling_factor=routed_scaling_factor,
        gemm1_alpha=gemm1_alpha,
        gemm1_limit=gemm1_limit,
        filter_expert=filter_expert,
        hooks=None,
    )


def fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_output,
    moe_runner_config,
    b1: Optional[torch.Tensor] = None,
    b2: Optional[torch.Tensor] = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
) -> torch.Tensor:
    """
    This function computes a Mixture of Experts (MoE) layer using two sets of
    weights, w1 and w2, and top-k gating mechanism.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the MoE layer.
    - w1 (torch.Tensor): The first set of expert weights.
    - w2 (torch.Tensor): The second set of expert weights.
    - topk_output (StandardTopKOutput): The top-k output of the experts.
    - moe_runner_config (MoeRunnerConfig): The configuration for the MoE runner.
    - b1 (Optional[torch.Tensor]): Optional bias for w1.
    - b2 (Optional[torch.Tensor]): Optional bias for w2.
    - use_fp8_w8a8 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a8 (bool): If True, use int8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int8_w8a16 (bool): If True, use fp8 arithmetic to compute the inner
        products for w1 and w2. Defaults to False.
    - use_int4_w4a16 (bool): If True, use matmul of int4 weight and bf16/fp16
        activation to compute the inner products for w1 and w2.
        Defaults to False.
    - w1_scale (Optional[torch.Tensor]): Optional scale to be used for
        w1.
    - w2_scale (Optional[torch.Tensor]): Optional scale to be used for
        w2.
    - a1_scale (Optional[torch.Tensor]): Optional scale to be used for
        a1.
    - a2_scale (Optional[torch.Tensor]): Optional scale to be used for
        a2.
    - block_shape: (Optional[List[int]]): Optional block size for block-wise
        quantization.
    - gemm1_alpha (Optional[float]): Optional gemm1_alpha for the activation
        function.
    - gemm1_limit (Optional[float]): Optional gemm1_limit for the swiglu activation
        function.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    topk_weights, topk_ids, _ = topk_output
    filter_expert = (
        moe_runner_config.num_experts is None
        or moe_runner_config.num_experts != moe_runner_config.num_local_experts
    )
    return fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        b1=b1,
        b2=b2,
        inplace=getattr(moe_runner_config, "inplace", False),
        activation=moe_runner_config.activation,
        is_gated=moe_runner_config.is_gated,
        apply_router_weight_on_input=moe_runner_config.apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        no_combine=getattr(moe_runner_config, "no_combine", False),
        routed_scaling_factor=moe_runner_config.routed_scaling_factor,
        gemm1_alpha=moe_runner_config.gemm1_alpha,
        gemm1_limit=moe_runner_config.gemm1_clamp_limit,
        filter_expert=filter_expert,
    )
