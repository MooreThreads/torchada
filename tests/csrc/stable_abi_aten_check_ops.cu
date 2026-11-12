/*
 * torchada libtorch-stable ABI shim coverage: the ATen check a ported kernel calls.
 *
 * The shape vLLM-MUSA's ported kernels present, which stable_abi_ops.cu does not:
 * stable-ABI registration and stable::Tensor, but the argument check is ATen's
 * TORCH_CHECK with a ScalarType streamed into the message. Nothing included here
 * declares either name — they come from the force-included box header, so this TU
 * stops compiling if that header drops them.
 *
 * Written in plain CUDA; torchada ports it to MUSA.
 */
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <cuda_runtime.h>

__global__ void fill_ones_kernel(float* __restrict__ out, int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = 1.0f;
  }
}

void fill_ones(torch::stable::Tensor out) {
  auto dtype = out.scalar_type();
  // ATen's check, with the ScalarType in the message: the two things the box
  // header has to declare for a ported kernel.
  TORCH_CHECK(dtype == at::ScalarType::Float, "fill_ones wants float, got ", dtype);
  int64_t n = out.numel();
  int64_t threads = 256;
  fill_ones_kernel<<<(n + threads - 1) / threads, threads>>>(
      reinterpret_cast<float*>(out.data_ptr()), n);
}

STABLE_TORCH_LIBRARY(torchada_stable_aten_check_test, m) {
  m.def("fill_ones(Tensor(a!) out) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(torchada_stable_aten_check_test, CUDA, m) {
  m.impl("fill_ones", TORCH_BOX(&fill_ones));
}
