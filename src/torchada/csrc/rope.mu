// Copyright (c) torchada contributors
// SPDX-License-Identifier: MIT

#include "ops.h"

#include "torch_musa/csrc/core/MUSAGuard.h"
#include <ATen/musa/MUSAContext.h>
#include <c10/core/ScalarType.h>
#include <limits>

namespace torchada {

template <typename scalar_t>
__device__ __forceinline__ void rotate_head(scalar_t *head,
                                            int64_t pair_index,
                                            int64_t pair_count, float cosine,
                                            float sine, bool is_neox) {
  int64_t first_dimension;
  int64_t second_dimension;
  if (is_neox) {
    first_dimension = pair_index;
    second_dimension = pair_index + pair_count;
  } else {
    first_dimension = pair_index * 2;
    second_dimension = first_dimension + 1;
  }

  const float first_value = static_cast<float>(head[first_dimension]);
  const float second_value = static_cast<float>(head[second_dimension]);
  head[first_dimension] =
      static_cast<scalar_t>(first_value * cosine - second_value * sine);
  head[second_dimension] =
      static_cast<scalar_t>(second_value * cosine + first_value * sine);
}

template <typename query_scalar_t, typename key_scalar_t,
          typename cache_scalar_t, typename index_t>
__global__ void apply_rope_inplace_kernel(
    query_scalar_t *query, key_scalar_t *key,
    const cache_scalar_t *cos_sin_cache, const index_t *positions,
    int64_t query_head_count, int64_t key_head_count,
    int64_t query_token_stride, int64_t key_token_stride, int64_t head_stride,
    int64_t rope_dimension, bool is_neox) {
  const int64_t pair_count = rope_dimension / 2;
  const int64_t token_index = blockIdx.x;
  const int64_t cache_offset =
      static_cast<int64_t>(positions[token_index]) * rope_dimension;
  query_scalar_t *query_token = query + token_index * query_token_stride;
  key_scalar_t *key_token = key + token_index * key_token_stride;

  for (int64_t pair_index = threadIdx.x; pair_index < pair_count;
       pair_index += blockDim.x) {
    const float cosine =
        static_cast<float>(cos_sin_cache[cache_offset + pair_index]);
    const float sine =
        static_cast<float>(cos_sin_cache[cache_offset + pair_index + pair_count]);
    for (int64_t head_index = 0; head_index < query_head_count; ++head_index) {
      rotate_head(query_token + head_index * head_stride, pair_index,
                  pair_count, cosine, sine, is_neox);
    }
    for (int64_t head_index = 0; head_index < key_head_count; ++head_index) {
      rotate_head(key_token + head_index * head_stride, pair_index, pair_count,
                  cosine, sine, is_neox);
    }
  }
}

template <typename query_scalar_t, typename key_scalar_t,
          typename cache_scalar_t, typename index_t>
void launch_apply_rope_inplace(at::Tensor &query, at::Tensor &key,
                               const at::Tensor &cos_sin_cache,
                               const at::Tensor &positions, bool is_neox,
                               int64_t rope_dimension) {
  const int64_t pair_count = rope_dimension / 2;
  const int64_t token_count = query.size(0);
  if (token_count == 0) {
    return;
  }
  TORCH_CHECK(token_count <= std::numeric_limits<int>::max(),
              "token count exceeds the MUSA grid limit");
  int threads = 32;
  while (threads < pair_count && threads < 256) {
    threads *= 2;
  }
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  apply_rope_inplace_kernel<query_scalar_t, key_scalar_t, cache_scalar_t,
                            index_t>
      <<<static_cast<int>(token_count), threads, 0, stream>>>(
          query.data_ptr<query_scalar_t>(), key.data_ptr<key_scalar_t>(),
          cos_sin_cache.data_ptr<cache_scalar_t>(),
          positions.data_ptr<index_t>(), query.size(1), key.size(1),
          query.stride(0), key.stride(0), query.stride(1), rope_dimension,
          is_neox);
}

template <typename index_t>
void dispatch_apply_rope_dtypes(at::Tensor &query, at::Tensor &key,
                                const at::Tensor &cos_sin_cache,
                                const at::Tensor &positions, bool is_neox,
                                int64_t rope_dimension) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, query.scalar_type(),
      "torchada_apply_rope_inplace_query", [&] {
        using query_scalar_t = scalar_t;
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half, at::ScalarType::BFloat16, key.scalar_type(),
            "torchada_apply_rope_inplace_key", [&] {
              using key_scalar_t = scalar_t;
              AT_DISPATCH_FLOATING_TYPES_AND2(
                  at::ScalarType::Half, at::ScalarType::BFloat16,
                  cos_sin_cache.scalar_type(),
                  "torchada_apply_rope_inplace_cache", [&] {
                    using cache_scalar_t = scalar_t;
                    launch_apply_rope_inplace<
                        query_scalar_t, key_scalar_t, cache_scalar_t, index_t>(
                        query, key, cos_sin_cache, positions, is_neox,
                        rope_dimension);
                  });
            });
      });
}

template <typename index_t>
void dispatch_apply_rope_positions(at::Tensor &query, at::Tensor &key,
                                   const at::Tensor &cos_sin_cache,
                                   const at::Tensor &positions, bool is_neox,
                                   int64_t rope_dimension) {
  dispatch_apply_rope_dtypes<index_t>(query, key, cos_sin_cache, positions,
                                      is_neox, rope_dimension);
}

void apply_rope_inplace_musa(at::Tensor &query, at::Tensor &key,
                             const at::Tensor &cos_sin_cache,
                             const at::Tensor &positions, bool is_neox,
                             int64_t rope_dimension) {
  TORCH_CHECK(query.is_privateuseone(), "query must be a MUSA tensor");
  TORCH_CHECK(key.device() == query.device(),
              "query and key must be on the same device");
  TORCH_CHECK(cos_sin_cache.device() == query.device() &&
                  positions.device() == query.device(),
              "all RoPE tensors must be on the same device");
  TORCH_CHECK(query.dim() == 3 && key.dim() == 3,
              "query and key must be three-dimensional");
  TORCH_CHECK(query.size(0) == key.size(0),
              "query and key must have the same token count");
  TORCH_CHECK(query.size(2) == key.size(2),
              "query and key must have the same head dimension");
  TORCH_CHECK(query.stride(2) == 1 && key.stride(2) == 1,
              "query and key must have a contiguous last dimension");
  TORCH_CHECK(query.stride(1) == key.stride(1),
              "query and key must have matching head strides");
  TORCH_CHECK(query.scalar_type() == at::ScalarType::Half ||
                  query.scalar_type() == at::ScalarType::BFloat16 ||
                  query.scalar_type() == at::ScalarType::Float,
              "query and key must use float16, bfloat16, or float32");
  TORCH_CHECK(key.scalar_type() == at::ScalarType::Half ||
                  key.scalar_type() == at::ScalarType::BFloat16 ||
                  key.scalar_type() == at::ScalarType::Float,
              "query and key must use float16, bfloat16, or float32");
  TORCH_CHECK(query.element_size() == key.element_size(),
              "query and key dtypes must have the same element size");
  TORCH_CHECK(cos_sin_cache.dim() == 2 && cos_sin_cache.is_contiguous(),
              "cos_sin_cache must be a contiguous matrix");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Half ||
                  cos_sin_cache.scalar_type() == at::ScalarType::BFloat16 ||
                  cos_sin_cache.scalar_type() == at::ScalarType::Float,
              "cos_sin_cache must use float16, bfloat16, or float32");
  TORCH_CHECK(positions.dim() == 1 && positions.is_contiguous(),
              "positions must be a contiguous vector");
  TORCH_CHECK(positions.scalar_type() == at::ScalarType::Int ||
                  positions.scalar_type() == at::ScalarType::Long,
              "positions must use int32 or int64");
  TORCH_CHECK(positions.size(0) == query.size(0),
              "positions must contain one value per token");
  TORCH_CHECK(rope_dimension > 0 && rope_dimension % 2 == 0,
              "rope_dimension must be a positive even number");
  TORCH_CHECK(rope_dimension == query.size(2) &&
                  rope_dimension == cos_sin_cache.size(1),
              "rope_dimension must match the head and cache dimensions");

  const c10::musa::MUSAGuard device_guard(query.device());
  if (positions.scalar_type() == at::ScalarType::Int) {
    dispatch_apply_rope_positions<int32_t>(
        query, key, cos_sin_cache, positions, is_neox, rope_dimension);
  } else {
    dispatch_apply_rope_positions<int64_t>(
        query, key, cos_sin_cache, positions, is_neox, rope_dimension);
  }

  const musaError_t error = musaGetLastError();
  TORCH_CHECK(error == musaSuccess, "MUSA apply_rope_inplace kernel failed: ",
              musaGetErrorString(error));
}

} // namespace torchada

TORCH_LIBRARY(torchada, m) {
  m.def("apply_rope_inplace(Tensor(a!) query, Tensor(b!) key, Tensor "
        "cos_sin_cache, "
        "Tensor positions, bool is_neox, int rope_dimension) -> ()");
}

TORCH_LIBRARY_IMPL(torchada, PrivateUse1, m) {
  m.impl("apply_rope_inplace", torchada::apply_rope_inplace_musa);
}
