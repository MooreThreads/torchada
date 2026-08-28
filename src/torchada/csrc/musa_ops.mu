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
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>
#include <limits>

namespace torchada {

// ============================================================================
// Example: MUSA kernel for neg (negation)
// This demonstrates how to override aten::neg for PrivateUse1 (MUSA) tensors
// ============================================================================

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

    // Ensure contiguous tensor
    auto self_contig = self.contiguous();

    // Allocate output tensor
    auto output = at::empty_like(self_contig);

    if (self_contig.numel() == 0) {
        return output;
    }

    // Get MUSA stream
    musaStream_t stream = at::musa::getCurrentMUSAStream();

    // Launch kernel
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

    // Check for launch errors
    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        TORCH_CHECK(false, "MUSA kernel launch failed: ", musaGetErrorString(err));
    }

    return output;
}

namespace {

__device__ unsigned long long multinomial_counter = 0;

__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ double uniform01(
    unsigned long long seed,
    int64_t row,
    int64_t sample) {
    unsigned long long x = splitmix64(
        seed ^ (static_cast<unsigned long long>(row) * 0xD1B54A32D192ED03ull) ^
        (static_cast<unsigned long long>(sample) * 0x94D049BB133111EBull));
    constexpr double scale = 1.0 / 9007199254740992.0;
    return static_cast<double>(x >> 11) * scale;
}

template <typename scalar_t>
__device__ __forceinline__ double read_weight(
    const scalar_t* input,
    int64_t idx) {
    double v = static_cast<double>(input[idx]);
    return isfinite(v) && v > 0.0 ? v : 0.0;
}

__device__ __forceinline__ bool already_selected(
    const int64_t* output,
    int64_t row,
    int64_t num_samples,
    int64_t current_sample,
    int64_t candidate) {
    const int64_t base = row * num_samples;
    for (int64_t i = 0; i < current_sample; ++i) {
        if (output[base + i] == candidate) {
            return true;
        }
    }
    return false;
}

template <typename scalar_t, int BLOCK>
__global__ void multinomial_kernel(
    const scalar_t* __restrict__ input,
    int64_t* __restrict__ output,
    int64_t rows,
    int64_t cols,
    int64_t num_samples,
    bool replacement,
    unsigned long long seed_base) {
    __shared__ double partial[BLOCK];
    __shared__ double prefix[BLOCK];
    __shared__ double total_sum;
    __shared__ unsigned long long block_seed;

    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    if (row >= rows) {
        return;
    }

    const int64_t row_offset = row * cols;
    const int64_t chunk = (cols + BLOCK - 1) / BLOCK;
    const int64_t begin = static_cast<int64_t>(tid) * chunk;
    const int64_t end = min(begin + chunk, cols);

    if (tid == 0) {
        unsigned long long counter = atomicAdd(&multinomial_counter, 1ull);
        block_seed = splitmix64(
            seed_base ^ counter ^ static_cast<unsigned long long>(clock64()));
    }
    __syncthreads();
    unsigned long long seed = block_seed;

    for (int64_t sample = 0; sample < num_samples; ++sample) {
        double sum = 0.0;
        for (int64_t col = begin; col < end; ++col) {
            if (replacement || !already_selected(output, row, num_samples, sample, col)) {
                sum += read_weight(input, row_offset + col);
            }
        }
        partial[tid] = sum;
        __syncthreads();

        if (tid == 0) {
            double running = 0.0;
            for (int i = 0; i < BLOCK; ++i) {
                prefix[i] = running;
                running += partial[i];
            }
            total_sum = running;
        }
        __syncthreads();

        double total = total_sum;
        int64_t selected = 0;
        if (total > 0.0) {
            double target = uniform01(seed, row, sample) * total;
            double before = prefix[tid];
            double after = before + partial[tid];
            if (target >= before && target < after) {
                double running = before;
                for (int64_t col = begin; col < end; ++col) {
                    if (!replacement && already_selected(output, row, num_samples, sample, col)) {
                        continue;
                    }
                    running += read_weight(input, row_offset + col);
                    if (target < running) {
                        selected = col;
                        break;
                    }
                }
            } else {
                selected = -1;
            }
        } else {
            selected = tid == 0 ? 0 : -1;
        }
        partial[tid] = static_cast<double>(selected);
        __syncthreads();

        if (tid == 0) {
            int64_t chosen = 0;
            for (int i = 0; i < BLOCK; ++i) {
                int64_t candidate = static_cast<int64_t>(partial[i]);
                if (candidate >= 0) {
                    chosen = candidate;
                    break;
                }
            }
            output[row * num_samples + sample] = chosen;
        }
        __syncthreads();
    }
}

}  // namespace

at::Tensor multinomial_musa_impl(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator) {
    log_op_call("multinomial");

    TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "prob_dist must be 1 or 2 dim");
    TORCH_CHECK(num_samples >= 0, "cannot sample n_sample < 0 samples");

    int64_t rows = self.dim() == 1 ? 1 : self.size(0);
    int64_t cols = self.dim() == 1 ? self.size(0) : self.size(1);
    if (!replacement) {
        TORCH_CHECK(
            num_samples <= cols,
            "cannot sample n_sample > prob_dist.size(-1) samples without replacement");
    }

    auto options = self.options().dtype(at::kLong);
    at::Tensor output = self.dim() == 1
        ? at::empty({num_samples}, options)
        : at::empty({rows, num_samples}, options);

    if (num_samples == 0 || rows == 0) {
        return output;
    }

    auto input = self.contiguous();
    constexpr int BLOCK = 256;
    musaStream_t stream = at::musa::getCurrentMUSAStream();
    unsigned long long seed_base = generator.has_value()
        ? static_cast<unsigned long long>(generator->current_seed())
        : static_cast<unsigned long long>(clock());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "torchada_multinomial_musa",
        [&] {
            multinomial_kernel<scalar_t, BLOCK><<<rows, BLOCK, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<int64_t>(),
                rows,
                cols,
                num_samples,
                replacement,
                seed_base);
        });

    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        TORCH_CHECK(false, "MUSA multinomial kernel launch failed: ", musaGetErrorString(err));
    }

    return output;
}

template <typename scalar_t>
__global__ void log_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        double value = static_cast<double>(input[idx]);
        output[idx] = static_cast<scalar_t>(log(value));
    }
}

at::Tensor log_musa_impl(const at::Tensor& self) {
    log_op_call("log");
    TORCH_CHECK(
        at::isFloatingType(self.scalar_type()),
        "torchada MUSA log only supports floating point tensors");

    auto input = self.contiguous();
    auto output = at::empty_like(input);
    if (input.numel() == 0) {
        return output;
    }

    constexpr int threads = 256;
    const int64_t numel = input.numel();
    const int blocks = static_cast<int>((numel + threads - 1) / threads);
    musaStream_t stream = at::musa::getCurrentMUSAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        input.scalar_type(),
        "torchada_log_musa",
        [&] {
            log_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                output.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                numel);
        });

    musaError_t err = musaGetLastError();
    if (err != musaSuccess) {
        TORCH_CHECK(false, "MUSA log kernel launch failed: ", musaGetErrorString(err));
    }

    if (!self.is_contiguous()) {
        return output.view(self.sizes());
    }
    return output;
}

at::Tensor& log_inplace_musa_impl(at::Tensor& self) {
    log_op_call("log_");
    TORCH_CHECK(
        at::isFloatingType(self.scalar_type()),
        "torchada MUSA log_ only supports floating point tensors");

    if (self.numel() == 0) {
        return self;
    }

    auto output = log_musa_impl(self);
    self.copy_(output);
    return self;
}

}  // namespace torchada

// ============================================================================
// Register operator overrides for PrivateUse1 (MUSA)
//
// Each operator checks TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME>=1 at registration
// time. If set, the override is not registered and torch_musa's default
// implementation is used.
//
// ============================================================================

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    if (torchada::is_override_enabled("multinomial")) {
        m.impl("multinomial", torchada::multinomial_musa_impl);
    }
    if (torchada::is_override_enabled("log")) {
        m.impl("log", torchada::log_musa_impl);
    }
    if (torchada::is_override_enabled("log_")) {
        m.impl("log_", torchada::log_inplace_musa_impl);
    }
}
