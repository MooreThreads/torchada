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

That's it! Supported `torch.cuda.*` APIs are automatically redirected to `torch.musa.*`.

## What Works

| Feature | Example |
|---------|---------|
| Device operations | `tensor.cuda()`, `model.cuda()`, `torch.device("cuda")` |
| Tensor factories | `torch.zeros(..., device="cuda")`, `torch.asarray(..., device="cuda")` → MUSA |
| Memory management | `torch.cuda.memory_allocated()`, `empty_cache()` |
| Synchronization | `torch.cuda.synchronize()`, `Stream`, `Event` |
| Mixed precision | `torch.cuda.amp.autocast()`, `GradScaler()` |
| CUDA Graphs | `torch.cuda.CUDAGraph`, `torch.cuda.graph()` |
| CUDA Runtime | `torch.cuda.cudart()` → uses MUSA runtime |
| Profiler | `ProfilerActivity.CUDA` → uses PrivateUse1 |
| Custom Ops | `Library.impl(..., "CUDA")` → uses PrivateUse1 |
| Distributed | `dist.init_process_group(backend='nccl')` → uses MCCL |
| torch.compile | Inductor with AOT-cacheable tensor factory wrappers |
| C++ Extensions | `CUDAExtension`, `BuildExtension`, in-place source porting, stable-ABI shims |
| FlexAttention | `torch.nn.attention.flex_attention` works on MUSA |
| C++ nvJPEG porting | nvJPEG source and build settings → MTJPEG |
| ctypes Libraries | `ctypes.CDLL` with CUDA function names → MUSA equivalents |
| Unified Accelerator API | `torch.accelerator.empty_cache()`, `memory_stats()`, `Stream`, `Event`, ... |
| Triton CUDA Extra | `tl.extra.cuda` → `tl.extra.musa` compatibility on MUSA |
| Triton Fused MoE | Triton 3.2.0 MTT S5000 tuning configs for vLLM and SGLang |

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

To dump MUSA graph dot files for debugging, set
`TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH` before running your program. torchada will
call `enable_debug_mode()` before each graph capture and `debug_dump(path)` after
the capture completes:

```bash
TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH=./graph_dumps \
python serve.py
```

The value is a dump directory. torchada creates it if needed and writes
timestamped files such as `graph_1783512345678900000.dot` inside it, so
repeated captures do not overwrite one another.

### torch.compile

```python
import torchada
import torch

compiled_model = torch.compile(model.cuda(), backend='inductor')
```

Tensor factory calls such as `torch.zeros(..., device="cuda")`,
`torch.asarray(..., device="cuda")`, and the `*_like` family also translate
explicit CUDA devices to MUSA. The factory wrappers remain compatible with
CUDA Graph capture and `torch.compile` AOT caching. The patched
`torch.device(...)` also remains usable from TorchScript.

### Triton Fused MoE Tuning

torchada bundles Triton 3.2.0 fused-MoE configurations tuned on MTT S5000 for
vLLM and SGLang. The bundled set includes BF16, FP8 W8A8, and shared-expert
shapes, plus a consistent up/down-projection layout for JoyAI-LLM-Flash. Tuning
results are environment-specific; use custom configurations for other Triton,
hardware, or workload combinations.

On import, torchada points SGLang and vLLM to the bundled configurations through
`SGLANG_MOE_CONFIG_DIR` and `VLLM_TUNED_CONFIG_FOLDER`. Existing environment
values are never overwritten, so set either variable before importing torchada
to use custom configurations.

### SGLang FlashAttention

When the MUSA `flash_attn_interface` package is available, torchada redirects
`sgl_kernel.flash_attn` imports to it. For legacy MUSA FA3 entry points whose
signature cannot accept the newer SGLang `only_qv` keyword, the wrapper drops
only that keyword; implementations that natively accept it are left unchanged.

### Building C++ Extensions

```python
import torchada  # Must import before torch.utils.cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Standard CUDAExtension works — torchada handles CUDA→MUSA translation.
ext = CUDAExtension("my_ext", sources=["kernel.cu"])
```

If an extension uses nvJPEG, keep its existing CUDA build settings:

```python
jpeg_ext = CUDAExtension(
    "jpeg_ext",
    sources=["decode.cu"],
    libraries=["nvjpeg"],
    define_macros=[("NVJPEG_FOUND", "1")],
)
```

On MUSA, `BuildExtension` ports project-local C/C++/CUDA source and header
contents **in place**. Original `.cu`/`.cuh` names and paths are preserved and
no `<dir>_musa` mirror is created; native `.mu`/`.muh` files and non-source
files are left unchanged. Because eligible source files are rewritten during
the build, use a clean or disposable checkout when the original CUDA contents
must be preserved. Symlinked portable sources and headers are rejected to avoid
modifying a target outside the project tree.

The porter translates CUDA architecture guards together with their thresholds
and preserves already-correct CUDA-to-MUSA mapping defines that would otherwise
collapse into self-references. It also maps canonical `nvjpeg*`/`NVJPEG*`
symbols and exact `nvjpeg.h` includes to MTJPEG, plus `libraries=["nvjpeg"]` to
`mtjpeg` and `NVJPEG_FOUND` to `MTJPEG_FOUND` on MUSA. CUDA builds keep their
original settings.

torchada also provides compatibility headers and source porting for the
libtorch stable-ABI kernels used by recent vLLM and SGLang releases on
torch_musa 2.9. The patched `torch.utils.cpp_extension.include_paths()` exposes
the compatibility include directory on MUSA. Custom stable-ABI builds should
add `stable_compat_include_dir()` explicitly, and kernels that use `TORCH_BOX`
must force-include the path returned by `stable_compat_box_header()`. Both
helpers are available from `torchada.utils.cpp_extension`. The torch_musa 2.9
header backport runs lazily and best-effort at MUSA extension
build time on torch 2.9; read-only headers are left unchanged. Torch 2.11 and
newer provide the stable ABI directly, so the backport is skipped. A plain
`import torchada` does not modify PyTorch or torch_musa headers.

In-place CUDA-to-MUSA porting protects system include roots by default. Set
`TORCHADA_EXCLUDE_DIRS` to add excluded roots for your environment. Entries may
be directory paths or directory names and are separated by the platform path
separator; commas are also accepted. A name matches a complete path component,
so `TORCHADA_EXCLUDE_DIRS=torch_musa` excludes `/home/torch_musa` without the
full path. Explicit source directories remain eligible for porting even when
they are below an excluded root.

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

### Unified Accelerator API (`torch.accelerator`)

`torch.accelerator` is PyTorch's unified backend-agnostic entry point. Its API
surface is expanding across PyTorch releases, so APIs such as `empty_cache()`,
`memory_stats()`, `Stream`, and `Event` are not yet present in torch 2.7 even
though they exist on `torch.musa`. torchada wraps `torch.accelerator` so code
written against the newer unified API works today:

```python
import torchada
import torch

# APIs that exist in torch 2.7 keep their official implementation
torch.accelerator.is_available()
torch.accelerator.device_count()

# APIs missing from torch 2.7 transparently fall back to torch.musa
torch.accelerator.empty_cache()
torch.accelerator.memory_allocated()
torch.accelerator.memory_stats()
torch.accelerator.manual_seed(42)
s = torch.accelerator.Stream()
e = torch.accelerator.Event()

# Patched to delegate to torch.musa.synchronize() (the default MUSA
# implementation does not support synchronizing all streams on a device)
torch.accelerator.synchronize()

# Context managers for forward compatibility with PyTorch 2.9+
with torch.accelerator.device_index(0):
    ...
with torch.accelerator.stream(torch.musa.Stream()):
    ...
```

**Forward compatibility:** The wrapper prefers the real `torch.accelerator`
implementation and only falls back to `torch.musa` when an attribute is
missing. The exception is torch_musa releases before `2.11.0.post2`, where
known-broken accelerator memory APIs are forced through `torch.musa`. Starting
with `2.11.0.post2`, the fixed official implementations are used automatically.

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

## Selected API Reference

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
| `__CUDA_ARCH__ < 800` | `__MUSA_ARCH__ < 220` |
| `nvjpeg.h`, `nvjpeg*`, `NVJPEG*` | `mtjpeg.h`, `mtjpeg*`, `MTJPEG*` |
| `libraries=["nvjpeg"]` | `libraries=["mtjpeg"]` |
| `NVJPEG_FOUND` | `MTJPEG_FOUND` |

See `src/torchada/_mappings/` for 400+ mapping rules grouped by API domain.
`src/torchada/_mapping.py` remains the compatibility aggregation entry point.

## Integrating torchada into Your Project

### Step 1: Add Dependency

```
# pyproject.toml or requirements.txt
torchada>=0.1.85
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

| Project | Category | Status | Tracking |
|---------|----------|--------|----------|
| [SGLang](https://github.com/sgl-project/sglang) | Model Serving | ✅ Merged | — |
| [vLLM-MUSA](https://github.com/MooreThreads/vllm-musa) | Model Serving | ✅ Merged | — |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | Model Serving (Omni) | ✅ Merged | — |
| [Xinference](https://github.com/xorbitsai/inference) | Model Serving | ✅ Merged | — |
| [LightLLM](https://github.com/ModelTC/LightLLM) | Model Serving | ✅ Merged | — |
| [LightX2V](https://github.com/ModelTC/LightX2V) | Image/Video Generation | ✅ Merged | — |
| [Chitu](https://github.com/thu-pacman/chitu) | Model Serving | ✅ Merged | — |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | KVCache | ✅ Merged | — |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Image/Video Generation | 🚧 In Progress | [ComfyUI#11618](https://github.com/Comfy-Org/ComfyUI/pull/11618) |

## License

MIT License
