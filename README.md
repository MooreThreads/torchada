<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/MooreThreads/torchada/main/assets/logo.png" alt="logo" width="250" margin="10px"></img>
</div>

--------------------------------------------------------------------------------

# torchada

English | [中文](README_CN.md)

**Run your CUDA code on Moore Threads GPUs — zero code changes required**

torchada is an adapter that makes [torch_musa](https://github.com/MooreThreads/torch_musa) (Moore Threads GPU support for PyTorch) compatible with standard PyTorch CUDA APIs. Import it once, and your existing `torch.cuda.*` code works on MUSA hardware.

## Why torchada?

Many PyTorch projects are written for NVIDIA GPUs using `torch.cuda.*` APIs. To run these on Moore Threads GPUs, you would normally need to change every `cuda` reference to `musa`. torchada eliminates this by automatically translating CUDA API calls to MUSA equivalents at runtime.

## Prerequisites

- **torch_musa**: You must have [torch_musa](https://github.com/MooreThreads/torch_musa) installed (this provides MUSA support for PyTorch)
- **Moore Threads GPU**: A Moore Threads GPU with proper driver installed

## Installation

```bash
pip install torchada

# Or install from source
git clone https://github.com/MooreThreads/torchada.git
cd torchada
pip install -e .
```

## Quick Start

```python
import torchada  # ← Add this one line at the top
import torch

# Your existing CUDA code works unchanged:
x = torch.randn(10, 10).cuda()
print(torch.cuda.device_count())
torch.cuda.synchronize()
```

That's it! All `torch.cuda.*` APIs are automatically redirected to `torch.musa.*`.

## What Works

| Feature | Example |
|---------|---------|
| Device operations | `tensor.cuda()`, `model.cuda()`, `torch.device("cuda")` |
| Memory management | `torch.cuda.memory_allocated()`, `empty_cache()` |
| Synchronization | `torch.cuda.synchronize()`, `Stream`, `Event` |
| Mixed precision | `torch.cuda.amp.autocast()`, `GradScaler()` |
| CUDA Graphs | `torch.cuda.CUDAGraph`, `torch.cuda.graph()` |
| CUDA Runtime | `torch.cuda.cudart()` → uses MUSA runtime |
| Profiler | `ProfilerActivity.CUDA` → uses PrivateUse1 |
| Custom Ops | `Library.impl(..., "CUDA")` → uses PrivateUse1 |
| Distributed | `dist.init_process_group(backend='nccl')` → uses MCCL |
| torch.compile | `torch.compile(model)` with all backends |
| C++ Extensions | `CUDAExtension`, `BuildExtension`, `load()` |
| FlexAttention | `torch.nn.attention.flex_attention` works on MUSA |
| ctypes Libraries | `ctypes.CDLL` with CUDA function names → MUSA equivalents |

## Examples

### Mixed Precision Training

```python
import torchada
import torch

model = MyModel().cuda()
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    output = model(data.cuda())
    loss = criterion(output, target.cuda())

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Distributed Training

```python
import torchada
import torch.distributed as dist

# 'nccl' is automatically mapped to 'mccl' on MUSA
dist.init_process_group(backend='nccl')
```

### CUDA Graphs

```python
import torchada
import torch

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g):  # cuda_graph= keyword works on MUSA
    y = model(x)
```

### torch.compile

```python
import torchada
import torch

compiled_model = torch.compile(model.cuda(), backend='inductor')
```

### Building C++ Extensions

```python
import torchada  # Must import before torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Standard CUDAExtension works — torchada handles CUDA→MUSA translation
ext = CUDAExtension("my_ext", sources=["kernel.cu"])
```

### Custom Ops

```python
import torchada
import torch

my_lib = torch.library.Library("my_lib", "DEF")
my_lib.define("my_op(Tensor x) -> Tensor")
my_lib.impl("my_op", my_func, "CUDA")  # Works on MUSA!
```

### Profiler

```python
import torchada
import torch

# ProfilerActivity.CUDA works on MUSA
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    model(x)
```

### ctypes Library Loading

```python
import torchada
import ctypes

# Load MUSA runtime library with CUDA function names
lib = ctypes.CDLL("libmusart.so")
func = lib.cudaMalloc  # Automatically translates to musaMalloc

# Works with MCCL too
nccl_lib = ctypes.CDLL("libmccl.so")
func = nccl_lib.ncclAllReduce  # Automatically translates to mcclAllReduce
```

## Platform Detection

```python
import torchada
from torchada import detect_platform, Platform

platform = detect_platform()
if platform == Platform.MUSA:
    print("Running on Moore Threads GPU")
elif platform == Platform.CUDA:
    print("Running on NVIDIA GPU")

# Or use torch.version-based detection
def is_musa():
    import torch
    return hasattr(torch.version, 'musa') and torch.version.musa is not None
```

## Performance

torchada uses aggressive caching to minimize runtime overhead. All frequently-called operations complete in under 200 nanoseconds:

| Operation | Overhead |
|-----------|----------|
| `torch.cuda.device_count()` | ~140ns |
| `torch.cuda.Stream` (attribute access) | ~130ns |
| `torch.cuda.Event` (attribute access) | ~130ns |
| `_translate_device('cuda')` | ~140ns |
| `torch.backends.cuda.is_built()` | ~155ns |

For comparison, a typical GPU kernel launch takes 5,000-20,000ns. The patching overhead is negligible for real-world applications.

Operations with inherent costs (runtime calls, object creation) take 300-600ns but cannot be optimized further without changing behavior.

## Known Limitation

**Device type string comparisons fail on MUSA:**

```python
device = torch.device("cuda:0")  # On MUSA, this becomes musa:0
device.type == "cuda"  # Returns False!
```

**Solution:** Use `torchada.is_gpu_device()`:

```python
import torchada

if torchada.is_gpu_device(device):  # Works on both CUDA and MUSA
    ...
# Or: device.type in ("cuda", "musa")
```

## API Reference

| Function | Description |
|----------|-------------|
| `detect_platform()` | Returns `Platform.CUDA`, `Platform.MUSA`, or `Platform.CPU` |
| `is_musa_platform()` | Returns True if running on MUSA |
| `is_cuda_platform()` | Returns True if running on CUDA |
| `is_gpu_device(device)` | Returns True if device is CUDA or MUSA |
| `CUDA_HOME` | Path to CUDA/MUSA installation |
| `cuda_to_musa_name(name)` | Convert `cudaXxx` → `musaXxx` |
| `nccl_to_mccl_name(name)` | Convert `ncclXxx` → `mcclXxx` |
| `cublas_to_mublas_name(name)` | Convert `cublasXxx` → `mublasXxx` |
| `curand_to_murand_name(name)` | Convert `curandXxx` → `murandXxx` |

**Note**: `torch.cuda.is_available()` is intentionally NOT redirected — it returns `False` on MUSA. This allows proper platform detection. For GPU availability checks, see the `has_gpu()` pattern in [examples/migrate_existing_project.md](examples/migrate_existing_project.md#important-note-on-gpu-detection).

**Note**: The name conversion utilities are exported for manual use, but `ctypes.CDLL` is automatically patched to translate function names when loading MUSA libraries.

## C++ Extension Symbol Mapping

When building C++ extensions, torchada automatically translates CUDA symbols to MUSA:

| CUDA | MUSA |
|------|------|
| `cudaMalloc` | `musaMalloc` |
| `cudaStream_t` | `musaStream_t` |
| `cublasHandle_t` | `mublasHandle_t` |
| `at::cuda` | `at::musa` |
| `c10::cuda` | `c10::musa` |
| `#include <cuda/*>` | `#include <musa/*>` |

See `src/torchada/_mapping.py` for the complete mapping table (380+ mappings).

## Integrating torchada into Your Project

### Step 1: Add Dependency

```
# pyproject.toml or requirements.txt
torchada>=0.1.38
```

### Step 2: Conditional Import

```python
# At your application entry point
def is_musa():
    import torch
    return hasattr(torch.version, "musa") and torch.version.musa is not None

if is_musa():
    import torchada  # noqa: F401

# Rest of your code uses torch.cuda.* as normal
```

### Step 3: Extend Feature Flags (if applicable)

```python
# Include MUSA in GPU capability checks
if is_nvidia() or is_musa():
    ENABLE_FLASH_ATTENTION = True
```

### Step 4: Fix Device Type Checks (if applicable)

```python
# Instead of: device.type == "cuda"
# Use: device.type in ("cuda", "musa")
# Or: torchada.is_gpu_device(device)
```

## Projects Using torchada

| Project | Category | Status |
|---------|----------|--------|
| [Xinference](https://github.com/xorbitsai/inference) | Model Serving | ✅ Merged |
| [LightLLM](https://github.com/ModelTC/LightLLM) | Model Serving | ✅ Merged |
| [LightX2V](https://github.com/ModelTC/LightX2V) | Image/Video Generation | ✅ Merged |
| [Chitu](https://github.com/thu-pacman/chitu) | Model Serving | ✅ Merged |
| [SGLang](https://github.com/sgl-project/sglang) | Model Serving | In Progress |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | Image/Video Generation | In Progress |

## License

MIT License
