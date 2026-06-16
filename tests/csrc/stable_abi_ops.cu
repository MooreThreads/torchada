/*
 * torchada libtorch-stable ABI shim coverage.
 *
 * Exercises the torchada stable-ABI compat shim (TORCH_BOX boxer +
 * THO_DISPATCH Dispatch.h shim) across DIVERSE stable-ABI signature shapes, so
 * we know the shim is generally usable — not just for the activation case:
 *   - void return + two Tensor args          (negate)
 *   - void return + Tensor args + scalar arg (scale, double)
 *   - single Tensor return                    (passthrough -> from<Tensor>)
 *   - scalar (int) return                     (numel_of  -> from<int64_t>)
 *   - THO_DISPATCH over float / half / bfloat16
 *
 * Written in plain CUDA; torchada ports it to MUSA. Uses only stable::Tensor
 * methods present on stock torch_musa (data_ptr/numel/scalar_type) so the test
 * is self-contained (no torch_musa header patch required).
 */
// NOTE: stable-ABI kernels are intentionally ATen-independent — do NOT include
// ATen headers here (ATen/Dispatch.h also defines ::detail::scalar_type and
// would clash with the headeronly Dispatch shim). Launch on the default stream.
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>  // torchada THO_DISPATCH shim
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void affine_kernel(const scalar_t* __restrict__ in,
                              scalar_t* __restrict__ out, double a, double b,
                              int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = static_cast<scalar_t>(a * static_cast<double>(in[i]) + b);
  }
}

template <typename scalar_t>
static void launch_affine_typed(const torch::stable::Tensor& in,
                                torch::stable::Tensor& out, double a, double b,
                                int64_t n) {
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);
  affine_kernel<scalar_t><<<blocks, threads>>>(  // default stream
      static_cast<const scalar_t*>(in.data_ptr()),
      static_cast<scalar_t*>(out.data_ptr()), a, b, n);
}

static void affine(const torch::stable::Tensor& in, torch::stable::Tensor& out,
                   double a, double b) {
  const int64_t n = in.numel();
  if (n == 0) return;
  THO_DISPATCH_SWITCH(
      in.scalar_type(), "torchada_stable_affine",
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float,
                        [&] { launch_affine_typed<scalar_t>(in, out, a, b, n); })
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Half,
                        [&] { launch_affine_typed<scalar_t>(in, out, a, b, n); })
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::BFloat16,
                        [&] { launch_affine_typed<scalar_t>(in, out, a, b, n); }));
}

// void return, two Tensor args
void negate(torch::stable::Tensor& out, torch::stable::Tensor& input) {
  affine(input, out, -1.0, 0.0);
}
// void return, Tensor args + double scalar arg
void scale(torch::stable::Tensor& out, torch::stable::Tensor& input, double s) {
  affine(input, out, s, 0.0);
}
// single Tensor return (boxer must from<Tensor>)
torch::stable::Tensor passthrough(torch::stable::Tensor& input) { return input; }
// scalar int return (boxer must from<int64_t>)
int64_t numel_of(torch::stable::Tensor& input) { return input.numel(); }

STABLE_TORCH_LIBRARY(torchada_stable_test, m) {
  m.def("negate(Tensor! out, Tensor input) -> ()");
  m.def("scale(Tensor! out, Tensor input, float s) -> ()");
  m.def("passthrough(Tensor input) -> Tensor");
  m.def("numel_of(Tensor input) -> int");
}

// MUSA tensors are PrivateUse1; torchada's _mapping.py rewrites the upstream
// STABLE_TORCH_LIBRARY_IMPL(..., CUDA, ...) device key to PrivateUse1.
STABLE_TORCH_LIBRARY_IMPL(torchada_stable_test, PrivateUse1, m) {
  m.impl("negate", TORCH_BOX(&negate));
  m.impl("scale", TORCH_BOX(&scale));
  m.impl("passthrough", TORCH_BOX(&passthrough));
  m.impl("numel_of", TORCH_BOX(&numel_of));
}
