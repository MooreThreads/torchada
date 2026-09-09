"""CPU contract tests for the out-of-tree fused MoE implementation.

The Triton launch is replaced with a tiny reference GEMM.  These tests cover
the Python pipeline and argument plumbing without requiring a CUDA/MUSA device.
"""

from types import SimpleNamespace

import pytest
import torch

from torchada.triton.runtime.fused_moe import fused_moe as moe


def _fake_gemm(a, w, bias, out, *args, **kwargs):
    """Reference implementation of invoke_fused_moe_kernel for CPU tests."""
    ids, sorted_ids, expert_ids = args[4], args[5], args[6]
    # The down projection launch passes top_k=1 because inputs are already
    # expanded; routing metadata still carries the original top-k width.
    topk = ids.shape[1]
    call_index = getattr(_fake_gemm, "calls", 0)
    _fake_gemm.calls = call_index + 1
    # The first launch receives [tokens, hidden], the second receives routed
    # [tokens * topk, intermediate].  sorted_ids is identity in this test.
    out_rows = out.reshape(-1, out.shape[-1])
    nrows = out_rows.shape[0]

    def write(row, value):
        out_rows[row].copy_(value)

    if call_index == 0:
        for row in range(nrows):
            token = row // topk
            e = int(ids[token, row % topk])
            write(row, a[token].to(out.dtype) @ w[e].to(out.dtype).T)
            if bias is not None:
                out_rows[row].add_(bias[e].to(out.dtype))
    else:
        for row in range(nrows):
            token = row // topk
            choice = row % topk
            e = int(ids[token, choice])
            write(row, a[row].to(out.dtype) @ w[e].to(out.dtype).T)
            if bias is not None:
                out_rows[row].add_(bias[e].to(out.dtype))
            # The down launch receives MUL_ROUTED_WEIGHT=True when the
            # caller asks the kernel to apply routing weights on output.
            if args[8]:
                out_rows[row].mul_(args[3][token, choice])


def _args(*, activation="relu2_no_mul", is_gated=False, no_combine=False, inplace=False):
    tokens, hidden, inter, experts, topk = 2, 3, 4, 2, 2
    x = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.0]])
    w1 = (
        torch.arange(experts * inter * hidden, dtype=torch.float32).reshape(experts, inter, hidden)
        / 10
    )
    w2_width = inter // 2 if is_gated else inter
    w2 = (
        torch.arange(experts * hidden * w2_width, dtype=torch.float32).reshape(
            experts, hidden, w2_width
        )
        / 10
    )
    weights = torch.tensor([[0.25, 0.75], [0.4, 0.6]])
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    ident = torch.arange(tokens * topk, dtype=torch.int32)
    config = {"BLOCK_SIZE_M": 1}
    return dict(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weights=weights,
        topk_ids=ids,
        sorted_token_ids=ident,
        expert_ids=torch.zeros(tokens, dtype=torch.int32),
        num_tokens_post_padded=torch.tensor([tokens], dtype=torch.int32),
        config=config,
        down_config=config,
        down_moe_use_tma=False,
        b1=None,
        b2=None,
        use_fp8_w8a8=False,
        use_int8_w8a8=False,
        use_int8_w8a16=False,
        use_int4_w4a16=False,
        per_channel_quant=False,
        w1_scale=None,
        w2_scale=None,
        w1_zp=None,
        w2_zp=None,
        a1_scale=None,
        a2_scale=None,
        block_shape=None,
        activation=activation,
        is_gated=is_gated,
        no_combine=no_combine,
        inplace=inplace,
        # Exercise Python-side routing-weight application for combined output.
        apply_router_weight_on_input=True,
        routed_scaling_factor=None,
        gemm1_alpha=None,
        gemm1_limit=None,
        filter_expert=False,
    )


@pytest.mark.parametrize("activation,is_gated", [("relu2_no_mul", False), ("silu", True)])
@pytest.mark.parametrize("no_combine,inplace", [(False, False), (False, True), (True, False)])
def test_fused_moe_pipeline_activation_and_output_modes(
    monkeypatch, activation, is_gated, no_combine, inplace
):
    monkeypatch.setattr(moe, "invoke_fused_moe_kernel", _fake_gemm)
    _fake_gemm.calls = 0
    kwargs = _args(activation=activation, is_gated=is_gated, no_combine=no_combine, inplace=inplace)
    original = kwargs["hidden_states"].clone()
    result = moe._fused_moe_kernel_sequence(**kwargs)
    assert isinstance(result, torch.Tensor)
    assert result.shape == ((2, 2, 3) if no_combine else (2, 3))
    if inplace:
        assert result.data_ptr() == kwargs["hidden_states"].data_ptr()
    # Ensure both activation branches actually contribute finite values.
    assert torch.isfinite(result).all()
    # Check the complete GEMM1 -> activation -> GEMM2 pipeline against a
    # compact PyTorch reference (the fake launcher only replaces the kernels).
    expected_rows = []
    for token in range(kwargs["hidden_states"].shape[0]):
        token_outputs = []
        for choice, expert in enumerate(kwargs["topk_ids"][token].tolist()):
            gate_up = original[token] @ kwargs["w1"][expert].T
            if is_gated:
                width = gate_up.shape[-1] // 2
                activated = torch.nn.functional.silu(gate_up[:width]) * gate_up[width:]
            else:
                activated = torch.relu(gate_up).square()
            token_outputs.append(activated @ kwargs["w2"][expert].T)
        expected_rows.append(torch.stack(token_outputs))
    expected = torch.stack(expected_rows)
    if no_combine:
        assert torch.allclose(result, expected, atol=1e-5, rtol=1e-5)
    else:
        weighted = (expected * kwargs["topk_weights"].unsqueeze(-1)).sum(dim=1)
        assert torch.allclose(result, weighted, atol=1e-5, rtol=1e-5)
    if inplace:
        assert not torch.equal(result, original)


def test_fused_moe_rejects_unknown_activation(monkeypatch):
    monkeypatch.setattr(moe, "invoke_fused_moe_kernel", _fake_gemm)
    _fake_gemm.calls = 0
    with pytest.raises(ValueError, match="activation"):
        moe._fused_moe_kernel_sequence(**_args(activation="gelu", is_gated=False))


def test_fused_moe_wrapper_forwards_runner_options_by_keyword(monkeypatch):
    captured = {}

    def fake_impl(*args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return torch.tensor([7.0])

    monkeypatch.setattr(moe, "fused_experts_impl", fake_impl)
    topk = (torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.long), None)
    runner = SimpleNamespace(
        num_experts=2,
        num_local_experts=2,
        activation="relu2_no_mul",
        is_gated=False,
        apply_router_weight_on_input=True,
        routed_scaling_factor=1.5,
        gemm1_alpha=0.25,
        gemm1_clamp_limit=2.0,
        no_combine=True,
        inplace=False,
    )
    x = torch.ones(1, 3)
    out = moe.fused_moe(x, torch.ones(2, 4, 3), torch.ones(2, 3, 4), topk, runner)
    assert out.item() == 7.0
    assert captured["activation"] == "relu2_no_mul"
    assert captured["is_gated"] is False
    assert captured["no_combine"] is True
    assert captured["inplace"] is False
    assert captured["apply_router_weight_on_input"] is True
    assert captured["routed_scaling_factor"] == 1.5
    assert captured["gemm1_alpha"] == 0.25
    assert captured["gemm1_limit"] == 2.0
