"""Tests for the MUSA joint rotary-position kernel."""

import sys
import types

import pytest
import torch

import torchada


def reference_rope(
    tensor: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    rope_dimension = tensor.size(-1)
    pair_count = rope_dimension // 2
    selected_cache = cos_sin_cache.index_select(0, positions.to(torch.int64))
    cosine = selected_cache[:, :pair_count].unsqueeze(1)
    sine = selected_cache[:, pair_count:].unsqueeze(1)
    result = tensor.to(torch.float32).clone()
    if is_neox:
        first_values = tensor[..., :pair_count].to(torch.float32)
        second_values = tensor[..., pair_count:].to(torch.float32)
        result[..., :pair_count] = first_values * cosine - second_values * sine
        result[..., pair_count:] = second_values * cosine + first_values * sine
    else:
        first_values = tensor[..., 0::2].to(torch.float32)
        second_values = tensor[..., 1::2].to(torch.float32)
        result[..., 0::2] = first_values * cosine - second_values * sine
        result[..., 1::2] = second_values * cosine + first_values * sine
    return result.to(tensor.dtype)


@pytest.mark.musa
@pytest.mark.gpu
@pytest.mark.parametrize("tensor_dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("is_neox", [False, True])
def test_apply_rope_inplace_matches_reference(
    tensor_dtype: torch.dtype,
    position_dtype: torch.dtype,
    is_neox: bool,
) -> None:
    torch.manual_seed(0)
    query = torch.randn(5, 3, 16, device="musa", dtype=tensor_dtype)
    key = torch.randn(5, 2, 16, device="musa", dtype=tensor_dtype)
    angles = torch.randn(7, 8, device="musa", dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.tensor([0, 3, 1, 6, 2], device="musa", dtype=position_dtype)
    expected_query = reference_rope(query, cos_sin_cache, positions, is_neox)
    expected_key = reference_rope(key, cos_sin_cache, positions, is_neox)
    query_pointer = query.data_ptr()
    key_pointer = key.data_ptr()

    torchada.apply_rope_inplace(
        query,
        key,
        cos_sin_cache,
        positions,
        is_neox=is_neox,
    )

    assert query.data_ptr() == query_pointer
    assert key.data_ptr() == key_pointer
    torch.testing.assert_close(query, expected_query, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key, expected_key, rtol=1e-3, atol=1e-3)
    assert torch.isfinite(query).all()
    assert torch.isfinite(key).all()


@pytest.mark.musa
@pytest.mark.gpu
@pytest.mark.parametrize("is_neox", [False, True])
def test_apply_rope_inplace_supports_partial_rotary_views(is_neox: bool) -> None:
    torch.manual_seed(2)
    head_dimension = 24
    rope_dimension = 16
    query_storage = torch.randn(5, 3, head_dimension, device="musa", dtype=torch.bfloat16)
    key_storage = torch.randn(5, 2, head_dimension, device="musa", dtype=torch.bfloat16)
    query = query_storage[..., :rope_dimension]
    key = key_storage[..., :rope_dimension]
    angles = torch.randn(7, rope_dimension // 2, device="musa", dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.tensor([0, 3, 1, 6, 2], device="musa", dtype=torch.int32)
    expected_query = reference_rope(query, cos_sin_cache, positions, is_neox)
    expected_key = reference_rope(key, cos_sin_cache, positions, is_neox)
    query_tail = query_storage[..., rope_dimension:].clone()
    key_tail = key_storage[..., rope_dimension:].clone()
    query_pointer = query.data_ptr()
    key_pointer = key.data_ptr()

    assert query.stride(1) == head_dimension
    assert key.stride(1) == head_dimension
    torchada.apply_rope_inplace(
        query,
        key,
        cos_sin_cache,
        positions,
        is_neox=is_neox,
    )

    assert query.data_ptr() == query_pointer
    assert key.data_ptr() == key_pointer
    torch.testing.assert_close(query, expected_query, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key, expected_key, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        query_storage[..., rope_dimension:],
        query_tail,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        key_storage[..., rope_dimension:],
        key_tail,
        rtol=0,
        atol=0,
    )


def test_install_sglang_rope_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    sglang = types.ModuleType("sglang")
    sglang.__path__ = []
    kernels = types.ModuleType("sglang.kernels")
    kernels.__path__ = []
    ops = types.ModuleType("sglang.kernels.ops")
    ops.__path__ = []
    attention = types.ModuleType("sglang.kernels.ops.attention")
    attention.__path__ = []
    rope = types.ModuleType("sglang.kernels.ops.attention.rope")
    attention.rope = rope
    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.kernels", kernels)
    monkeypatch.setitem(sys.modules, "sglang.kernels.ops", ops)
    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.attention", attention)
    monkeypatch.setitem(sys.modules, "sglang.kernels.ops.attention.rope", rope)

    torchada.install_sglang_rope_compat()

    assert rope.apply_rope_inplace is torchada.apply_rope_inplace


@pytest.mark.musa
@pytest.mark.gpu
def test_apply_rope_inplace_replays_with_new_graph_inputs() -> None:
    torch.manual_seed(1)
    query = torch.randn(6, 4, 32, device="musa", dtype=torch.bfloat16)
    key = torch.randn(6, 2, 32, device="musa", dtype=torch.bfloat16)
    angles = torch.randn(8, 16, device="musa", dtype=torch.float32)
    cos_sin_cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.tensor([0, 2, 4, 6, 1, 3], device="musa", dtype=torch.int64)

    warmup_query = query.clone()
    warmup_key = key.clone()
    torchada.apply_rope_inplace(
        warmup_query,
        warmup_key,
        cos_sin_cache,
        positions,
        is_neox=False,
    )
    torch.musa.synchronize()

    graph = torch.musa.MUSAGraph()
    with torch.musa.graph(graph):
        torchada.apply_rope_inplace(
            query,
            key,
            cos_sin_cache,
            positions,
            is_neox=False,
        )

    next_query = torch.randn_like(query)
    next_key = torch.randn_like(key)
    expected_query = reference_rope(next_query, cos_sin_cache, positions, False)
    expected_key = reference_rope(next_key, cos_sin_cache, positions, False)
    query.copy_(next_query)
    key.copy_(next_key)
    graph.replay()
    torch.musa.synchronize()

    torch.testing.assert_close(query, expected_query, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(key, expected_key, rtol=1e-3, atol=1e-3)
