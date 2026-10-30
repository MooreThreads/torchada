# Writing Custom MUSA Operators in torchada

This guide explains how to write custom MUSA C++ operators that override torch_musa's default ATen implementations.

## Overview

torchada allows you to override ATen operators at the C++ level for the `PrivateUse1` (MUSA) dispatch key. This is useful when you need:

- Better performance than the default torch_musa implementation
- Custom behavior for specific operators
- Workarounds for torch_musa bugs

## Quick Start

C++ extensions are automatically loaded on MUSA platform when torchada is imported.

### 1. Write Your Kernel

Edit `src/torchada/csrc/musa_ops.mu`:

```cpp
#include "ops.h"
#include <ATen/musa/MUSAContext.h>

namespace torchada {

template <typename scalar_t>
__global__ void my_kernel(scalar_t* output, const scalar_t* input, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = /* your computation */;
    }
}

at::Tensor my_op_impl(const at::Tensor& self) {
    log_op_call("my_op");

    auto input = self.contiguous();
    auto output = at::empty_like(input);
    if (input.numel() == 0) return output;

    musaStream_t stream = at::musa::getCurrentMUSAStream();
    const int64_t n = input.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "my_op", [&] {
        my_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            n);
    });

    return output;
}

}  // namespace torchada

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // Check env var at registration time - allows disabling via
    // TORCHADA_DISABLE_OP_OVERRIDE_my_op=1
    if (torchada::is_override_enabled("my_op")) {
        m.impl("my_op", torchada::my_op_impl);
    }
}
```

### 2. Test Your Kernel

```bash
TORCHADA_DEBUG_CPP_OPS=1 python -c "
import torch
import torchada

x = torch.randn(1000, device='cuda')
y = torch.neg(x)  # Should print '[torchada] neg called'
print('Result:', y.cpu()[:5])
"
```

## File Structure

| File | Purpose |
|------|---------|
| `src/torchada/csrc/ops.h` | Header with utilities (`log_op_call`, `is_override_enabled`) |
| `src/torchada/csrc/ops.cpp` | Python bindings, C++-only implementations, and CUDA-compatible APIs |
| `src/torchada/csrc/musa_ops.mu` | MUSA kernel implementations |

## Built-in Functions

torchada's C++ extension provides CUDA-compatible implementations of some torch_musa memory management APIs:

### Memory Pool Functions

These functions are automatically injected into `torch.cuda.memory` and allow CUDA code using memory pools to work transparently on MUSA:

- `_cuda_beginAllocateCurrentThreadToPool(device, mempool_id)` - Begin allocating memory from current thread to a memory pool
- `_cuda_endAllocateToPool(device, mempool_id)` - End allocating memory to a memory pool
- `_cuda_releasePool(device, mempool_id)` - Release a memory pool

**Usage example:**
```python
import torchada
import torch

# This works transparently on MUSA - no code changes needed
from torch.cuda.memory import _cuda_beginAllocateCurrentThreadToPool
from torch.cuda.memory import _cuda_endAllocateToPool
from torch.cuda.memory import _cuda_releasePool

# Use the functions as in CUDA code
device = 0
pool_id = torch.cuda.graph_pool_handle()
_cuda_beginAllocateCurrentThreadToPool(device, pool_id)
# ... allocations ...
_cuda_endAllocateToPool(device, pool_id)
_cuda_releasePool(device, pool_id)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `TORCHADA_CPP_OPS_VERBOSE=1` | Show compilation output |
| `TORCHADA_DEBUG_CPP_OPS=1` | Log operator calls to stdout |
| `TORCHADA_DISABLE_OP_OVERRIDE_<NAME>=1` | Disable specific operator override |
| `MTGPU_TARGET=mp_XX` | Architecture when no device answers (the device wins) |

### Disabling Specific Operators

To disable a specific operator override at runtime, set the environment variable before importing torchada:

```bash
# Disable the 'neg' operator override, use torch_musa's default instead
TORCHADA_DISABLE_OP_OVERRIDE_neg=1 python my_script.py
```

**Important**: The operator name in the environment variable should match the name passed to `is_override_enabled()` in the C++ code. For example, if the code uses `is_override_enabled("neg")`, set `TORCHADA_DISABLE_OP_OVERRIDE_neg=1`.

This check happens at **registration time** (when the extension is loaded), not at runtime. Once the extension is loaded, the operator registrations are fixed.

## GPU Architecture

The arch is read from `major.minor` in the device properties, never guessed from a
product name or from `torch_musa.get_arch_list()` (which reports what the *wheel*
was built for):

| GPU | Compute Capability | Architecture |
|-----|-------------------|--------------|
| MTT S80 | 2.1 | mp_21 |
| MTT S4000 | 2.2 | mp_22 |
| MTT S5000 | 3.1 | mp_31 |

There are two build paths and they read the target in opposite orders, on purpose.

**torchada's own operators** (`torchada/csrc`, compiled on first import by
`load_cpp_ops()`): the device comes first, and `MTGPU_TARGET` /
`VLLM_MUSA_ARCH_LIST` / `MATE_MUSA_ARCH_LIST` (then `musaInfo`) answer only when no
device does. These operators are compiled on the machine that runs them, so the card
in the box is the authority — a stale literal baked into an image would otherwise
produce an object that runs on another card and returns plausible wrong numbers.

**A wheel or a project extension** built through `torchada.musa.build.resolve_targets()`
reads the environment first and the device second. A build host must be able to state
its target, including when a card of a different class happens to be in the box
(cross-compiling, a mixed CI fleet). With `default=None` (the JIT form) and nothing
set and no device, it raises rather than guessing.

On a CPU-only build server nothing is detected, which is the normal case for a wheel:
the environment is the input, and `MTGPU_TARGET=mp_22` is how you state it.

The built object is cached under
`<torch extensions>/torchada_cpp_ops/<arch>__mcc<version>/`, keyed by the arch and the
compiler rather than by the module name alone. That costs one compile per machine per
(arch, compiler) — the same first build you would pay anyway — and buys the guarantee
that a copied or shared cache cannot hand an `mp_22` object to an `mp_31` host, where
it would run and return plausible wrong numbers.

## Best Practices

### Avoid Infinite Recursion

When overriding an operator, don't call the same operator:

```cpp
// BAD - causes infinite recursion
at::Tensor bad_neg_impl(const at::Tensor& self) {
    return -self;  // Calls aten::neg again!
}

// GOOD - use lower-level primitives
at::Tensor good_neg_impl(const at::Tensor& self) {
    auto output = at::empty_like(self);
    // Launch custom kernel or use in-place ops
    return output;
}
```

### Handle Edge Cases

```cpp
at::Tensor my_impl(const at::Tensor& self) {
    auto input = self.contiguous();  // Ensure contiguous
    if (input.numel() == 0) {
        return at::empty_like(input);  // Handle empty tensors
    }
    // ... kernel launch
}
```

### Check for Errors

```cpp
musaError_t err = musaGetLastError();
if (err != musaSuccess) {
    TORCH_CHECK(false, "MUSA kernel failed: ", musaGetErrorString(err));
}
```

### Use Type Dispatching

```cpp
AT_DISPATCH_ALL_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(), "my_kernel", [&] {
        my_kernel<scalar_t><<<blocks, threads, 0, stream>>>(...);
    });
```

## Overridable ATen Operators

Any ATen operator with a `PrivateUse1` dispatch can be overridden. Common categories:

### Unary Operations
`abs`, `neg`, `exp`, `exp2`, `log`, `log2`, `log10`, `sqrt`, `rsqrt`, `ceil`, `floor`, `round`, `trunc`, `sign`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `sigmoid`, `erf`, `erfc`, `reciprocal`, `bitwise_not`

### Binary Operations
`add`, `sub`, `mul`, `div`, `pow`, `fmod`, `remainder`, `maximum`, `minimum`, `atan2`, `bitwise_and`, `bitwise_or`, `bitwise_xor`, `logical_and`, `logical_or`, `logical_xor`

### Reduction Operations
`sum`, `prod`, `mean`, `std`, `var`, `max`, `min`, `argmax`, `argmin`, `all`, `any`, `norm`, `logsumexp`

### Matrix Operations
`mm`, `bmm`, `addmm`, `addmv`, `addr`, `matmul`, `dot`, `mv`, `ger`, `linear`

### Activation Functions
`relu`, `relu_`, `leaky_relu`, `gelu`, `silu`, `mish`, `hardswish`, `hardsigmoid`, `softplus`, `softshrink`, `threshold`

### Normalization
`batch_norm`, `layer_norm`, `group_norm`, `instance_norm`, `local_response_norm`

### Pooling
`max_pool1d`, `max_pool2d`, `max_pool3d`, `avg_pool1d`, `avg_pool2d`, `avg_pool3d`, `adaptive_max_pool2d`, `adaptive_avg_pool2d`

### Convolution
`conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`

### Memory Operations
`copy_`, `clone`, `contiguous`, `fill_`, `zero_`, `ones_like`, `zeros_like`, `empty_like`

### Indexing
`index`, `index_put_`, `gather`, `scatter`, `scatter_add`, `masked_fill`, `masked_select`, `where`

### Shape Operations
`view`, `reshape`, `transpose`, `permute`, `squeeze`, `unsqueeze`, `expand`, `repeat`, `cat`, `stack`, `split`, `chunk`

To find the exact operator signature, use:

```python
import torch
# Search for specific operator:
for s in torch._C._jit_get_all_schemas():
    if 'neg' in str(s):
        print(s)
```

## Complete Example

See `src/torchada/csrc/musa_ops.mu` for a complete example of an `aten::neg` override (commented out by default — uncomment the `m.impl()` line to activate).

## Debugging

1. **Verify kernel is called**: Set `TORCHADA_DEBUG_CPP_OPS=1`
2. **Check compilation**: Set `TORCHADA_CPP_OPS_VERBOSE=1`
3. **Clear cache**: `rm -rf ~/.cache/torch_extensions/*/torchada_cpp_ops`
4. **Check architecture**: Run `musaInfo | grep "compute capability"`
