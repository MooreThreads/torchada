// torchada MUSA operator overrides
//
// This file contains MUSA kernel implementations that can override torch_musa's
// default ATen operator implementations.
//
// NOTE: No operators are overridden by default. The implementations below serve
// as examples. To activate an override, uncomment the corresponding m.impl()
// line in the TORCH_LIBRARY_IMPL block at the bottom of this file.

#include "ops.h"
#include <ATen/musa/MUSAContext.h>

namespace torchada {

// Example MUSA kernel for overriding aten::neg on PrivateUse1 tensors.

template <typename scalar_t>
__global__ void neg_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = -input[idx];
    }
}

at::Tensor neg_musa_impl(const at::Tensor& self) {
    log_op_call("neg");

    // Ensure the input is contiguous.
    auto self_contig = self.contiguous();

    // Allocate the output tensor.
    auto output = at::empty_like(self_contig);

    if (self_contig.numel() == 0) {
        return output;
    }

    // Use the current MUSA stream.
    musaStream_t stream = at::musa::getCurrentMUSAStream();

    // Launch the kernel.
    const int64_t numel = self_contig.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        self_contig.scalar_type(), "neg_musa", [&] {
            neg_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                self_contig.data_ptr<scalar_t>(),
                numel);
        });

    // Check launch errors.
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        TORCH_CHECK(false, "MUSA kernel launch failed: ", musaGetErrorString(err));
    }

    return output;
}

}  // namespace torchada

// Register operator overrides for PrivateUse1 (MUSA).
//
// Each operator checks TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME>=1 at registration
// time. If set, torch_musa's default implementation remains active.
//
// Uncomment m.impl() lines to activate custom implementations.

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Example: register neg override only when enabled.
    // if (torchada::is_override_enabled("neg")) {
    //     m.impl("neg", torchada::neg_musa_impl);
    // }
}
