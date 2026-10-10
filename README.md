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
| Tensor factories | `torch.zeros(..., device="cuda")`, `torch.asarray(..., device="cuda")`, the `*_like` family |
| Memory management | `torch.cuda.memory_allocated()`, `empty_cache()`, `torch.cuda.memory.*`, CUDA memory-pool APIs |
| Synchronization | `torch.cuda.synchronize()`, `Stream`, `Event`, `torch.cuda.streams` |
| Mixed precision | `torch.cuda.amp.autocast()`, `GradScaler()` |
| CUDA Graphs | `torch.cuda.CUDAGraph`, `torch.cuda.graph()`, executable rotation for deep models |
| CUDA Runtime | `torch.cuda.cudart()` → uses MUSA runtime |
| Profiler | `ProfilerActivity.CUDA` → uses PrivateUse1 |
| Custom Ops | `Library.impl(..., "CUDA")` → uses PrivateUse1 |
| Distributed | `dist.init_process_group(backend='nccl')` → uses MCCL |
| torch.compile | Inductor, AOT-cacheable factory wrappers, FX `device` builtin, `torch.cuda._get_device_index` |
| C++ Extensions | `CUDAExtension`, `BuildExtension`, in-place source porting, stable-ABI shims |
| FlexAttention | `torch.nn.attention.flex_attention` works on MUSA |
| C++ nvJPEG porting | nvJPEG source and build settings → MTJPEG |
| ctypes Libraries | `ctypes.CDLL` with CUDA function names → MUSA equivalents |
| Unified Accelerator API | `torch.accelerator.empty_cache()`, `memory_stats()`, `get_memory_info()`, `Stream`, `Event`, ... |
| FlashAttention providers | `flash_attn_interface` redirection, plus `only_qv` and `_flash_attn_forward` shims |
| Triton CUDA Extra | `tl.extra.cuda` → `tl.extra.musa` compatibility on MUSA |
| Triton Fused MoE | Triton 3.2.0 MTT S5000 tuning configs for vLLM and SGLang |
| MUSA-specific operator fixes | float64 in-place `log_`, asynchronous `isfinite`, `mm`/`bmm` `out_dtype=` — see [torch_musa compatibility](#torch_musa-compatibility) |

## torch_musa Compatibility

Some patches exist only because a particular torch_musa release was missing a kernel
or had a broken one. Each of those is version-gated: it installs itself only on the
releases that need it and steps aside on the release that fixes the problem, so
nothing has to be configured when the underlying stack moves.

| Shim | Installed when | What it does |
|------|----------------|--------------|
| float64 in-place `Tensor.log_` | torch_musa < `2.11.0.post2` | Reuses the supported out-of-place `log` and writes the result back, preserving the in-place contract |
| Compiled `multinomial` / `log` / `log_` | torch_musa < `2.11.0.post2` | MUSA `PrivateUse1` operator overrides in torchada's JIT-built C++ extension, replay-safe under CUDA graph capture |
| Inductor MUSA template heuristics | torch_musa < `2.11.0.post2` | Fills the `musa` keys of Inductor's template-heuristic registry from the CUDA entries |
| `torch.accelerator` memory APIs | torch_musa < `2.11.0.post2` | Routes `empty_cache()` / `memory_stats()` / ... to `torch.musa`, because the pre-release dispatch was broken |
| `torch.mm` / `torch.bmm` `out_dtype=` | torch_musa < `2.13.0` | Reuses the plain overloads and accumulates in fp32; a runtime probe decides per process whether the native overload is healthy |
| Asynchronous `torch.isfinite` | Always, on MUSA float tensors | Evaluates `abs() < inf` on float16/bfloat16/float32/float64 tensors, because ATen's composite runs a boolean `mul` that blocks the host until the device queue drains. Other dtypes keep the original operator |
| libtorch stable-ABI headers | torch < 2.11 on MUSA | Backports the `torch::stable` accessors that torch_musa's 2.9 snapshot omits, so vLLM and SGLang stable kernels build |
| `torch.cuda.streams` module path | MUSA, whenever `torch_musa.core.stream` is importable | Exposes it as `torch.cuda.streams`, which the PyTorch 2.11 Dynamo guards resolve |

The `out_dtype` backport is armed **below `torch_musa 2.13.0`**, the release torch_musa committed to
fix the overloads in. That is a vendor release commitment, not a measurement of ours, and trusting it
is the accepted risk: if `2.13.0` does not actually fix them, a `>= 2.13.0` stack installs no wrapper
and the silent all-zero result can come back. The gate only decides whether a Python wrapper sits in
front of `torch.mm`/`torch.bmm`; correctness is decided per process by the runtime probe, which
forwards to a healthy overload and emulates a broken one - so a fix backported into `2.12.x` is picked
up automatically. An unknown or unparsable `__version__` ranks lowest and therefore stays armed. Once
`2.13.0` is released and verified fixed here, the shim is deleted; if the fix slips, the bound moves to
the newly committed release. The overload was measured broken on `2.11.0.post1+musa5.2.0` (`mm` writes
zeros, `bmm` writes wrong values).

**Not covered:** builds whose binding has no `*_Dtype` overload keep raising on `out_dtype=`. That is
the binding's contract, not a defect torchada repairs, so CUDA parity for that case is out of scope.

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

Deep models get a second transparent fix. The MUSA driver caps the number of live
graph *executables* per process (~2048), and piecewise CUDA graphs instantiate
`capture_sizes * num_layers` of them, so models deeper than roughly 40 layers used to
exceed the cap and could not use piecewise graphs at all. torchada keeps every
captured *template* alive, LRU-caps the live executables (1900 by default), and
re-instantiates an evicted graph's executable from its template on the next replay,
which costs about 0.3 ms and needs no forward re-run. It is zero-cost until the cap
is exceeded, and can be tuned or disabled through the
[environment variables](#environment-variables) below.

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

Three more patches keep compiled code working end to end:

- `torch.cuda._get_device_index` is mapped to its MUSA counterpart, which is what
  TorchDynamo calls to model `torch.cuda.device(...)`. Without it, compiling a
  function that opens a device context raises `AttributeError: module 'torch_musa'
  has no attribute '_get_device_index'`.
- The patched `torch.device` is registered as the FX `device` builtin, so a
  `GraphModule` holding a device constant still binds the name `device` and runs with
  the `eager` and `aot_eager` backends. (Inductor executes its own generated code, so
  it is unaffected either way.)
- `MUSA_VISIBLE_DEVICES` is mirrored to `CUDA_VISIBLE_DEVICES` when both are present,
  and Inductor's autotune subprocess reads the MUSA variable.

On torch_musa releases before `2.11.0.post2`, missing `musa` entries in Inductor's
matmul template-heuristic registry are filled in from the CUDA entries; newer releases
register their own and torchada leaves them alone.

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

The tables are generated rather than typed. `ci/shapes.json` and `ci/models.json`
next to `tune_moe.py` record the kernel shape, the bucket, and the configuration each
bucket is pinned to; `tune_moe.py --config .../ci/shapes.json --materialize
--merge-configs` re-derives the rows without a GPU, and
`tests/test_tune_moe_recipe.py` asserts the result matches the shipped table row for
row and key for key. `--merge-configs` keeps every row a run did not measure, so a
change shows up as a reviewable diff. See [docs/tune_triton_moe.md](docs/tune_triton_moe.md)
for the tuning and benchmarking tool itself.

### FlashAttention Providers (SGLang, vLLM-Omni)

When the MUSA `flash_attn_interface` package is available, torchada redirects
`sgl_kernel.flash_attn` imports to it. For legacy MUSA FA3 entry points whose
signature cannot accept the newer SGLang `only_qv` keyword, the wrapper drops
only that keyword; implementations that natively accept it are left unchanged.

torchada also provides `flash_attn_interface._flash_attn_forward` when the provider
exposes the public output+softmax-LSE API but not that private symbol. vLLM-Omni's
Ring Attention imports the private name because it needs the LSE for partial-attention
accumulation, and its FA3 availability probe evaluates false without it. The shim is a
thin adapter over `flash_attn_func(..., return_softmax_lse=True)`, returns `None` for
the dropout-only auxiliary values the public inference API cannot supply, and fails
closed if the provider does not return both output and LSE. A provider that ships its
own `_flash_attn_forward` is left untouched.

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

Porting reaches nested PyTorch headers as well: both `<torch/cuda.h>` and
`"torch/cuda.h"` become `torch/musa.h`, while project-local names such as
`decode_jpegs_cuda.h` are left alone.

For stable-ABI kernels, porting also rekeys `STABLE_TORCH_LIBRARY_IMPL(<namespace>,
CUDA, ...)` to the `PrivateUse1` dispatch key for any namespace and any whitespace
layout, including multiline macro invocations, and maps the stable-ABI CUDA stream
helpers (`aoti_torch_get_current_cuda_stream`, `torch_set_current_cuda_stream`,
`torch_get_cuda_stream_from_pool`, `torch_cuda_stream_synchronize`) plus the BLAS
handle accessors to their MUSA equivalents.

`include_paths()` and `library_paths()` follow PyTorch's 2.6+ signature, positionally
and by keyword. This matters whenever `TORCHINDUCTOR_CACHE_DIR` is cold: Inductor's
C++ builder calls both helpers for CPU kernels too, and the older MUSA-aware
signatures raise there, which made a plain CPU `torch.compile` fail after
`import torchada`.

The C++ operator-override extension is JIT-built on first use. torchada builds it
under its own `flock`, so a `lock` file left behind by a process killed mid-build no
longer makes every later `import torchada` block forever in the JIT loader's
`FileBaton`, and concurrent imports compile once instead of racing. If `flock` is
unavailable (some NFS mounts), torchada warns and falls back to torch's original
behaviour.

### Custom Ops

```python
import torchada
import torch

my_lib = torch.library.Library("my_lib", "DEF")
my_lib.define("my_op(Tensor x) -> Tensor")
my_lib.impl("my_op", my_func, "CUDA")  # Works on MUSA!
```

To override ATen operators at the C++ level on the `PrivateUse1` dispatch key
instead, see [docs/custom_musa_ops.md](docs/custom_musa_ops.md).

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
torch.accelerator.get_memory_info()
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

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `TORCHADA_PLATFORM` | auto-detect | Force platform detection: `cuda`, `musa`, or `cpu` |
| `TORCHADA_EXCLUDE_DIRS` | unset | Extra include roots excluded from in-place source porting |
| `TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH` | unset | Directory for MUSA graph `.dot` dumps |
| `TORCHADA_GRAPH_ROTATION` | `1` | `0` disables CUDA-graph executable rotation |
| `TORCHADA_GRAPH_EXEC_CAP` | `1900` | Live graph-executable cap per process |
| `TORCHADA_GRAPH_AUTOPROBE` | `0` | `1` probes the driver's real executable cap at startup |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `128` | Margin kept below the probed cap |
| `TORCHADA_CPP_OPS_VERBOSE` | `0` | `1` prints the C++ operator-override build log |
| `TORCHADA_DEBUG_CPP_OPS` | `0` | `1` logs operator-override calls |
| `TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME>` | unset | `1` disables one operator override, e.g. `TORCHADA_DISABLE_OP_OVERRIDE_log=1` |

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

torchada uses aggressive caching to minimize runtime overhead. The numbers below are
the checked-in `benchmarks/benchmark_history.json` measurement for torchada 0.1.95
(2026-10-09, MTT S5000, PyTorch and torch_musa `2.11.0.post2+musa5.2.0`), as median
per-call overhead:

| Operation | Overhead |
|-----------|----------|
| `torch.cuda.Stream` (attribute access) | ~170ns |
| `torch.cuda.Event` (attribute access) | ~170ns |
| `_translate_device('cuda')` | ~180ns |
| `torch.cuda.device_count()` | ~200ns |
| `torch.backends.cuda.is_built()` | ~260ns |

For comparison, a typical GPU kernel launch takes 5,000-20,000ns, so the patching
overhead is negligible for real-world applications.

Operations with inherent costs (runtime calls, object creation) take 400-600ns and
cannot be optimized further without changing behavior.

The 0.1.94 entry in the same file, measured on torch_musa 2.7.1, reports ~120-160ns
for the same fast paths.

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
| `#include <torch/cuda.h>` | `#include <torch/musa.h>` |
| `__CUDA_ARCH__ < 800` | `__MUSA_ARCH__ < 220` |
| `nvjpeg.h`, `nvjpeg*`, `NVJPEG*` | `mtjpeg.h`, `mtjpeg*`, `MTJPEG*` |
| `libraries=["nvjpeg"]` | `libraries=["mtjpeg"]` |
| `NVJPEG_FOUND` | `MTJPEG_FOUND` |
| `aoti_torch_get_current_cuda_stream`, `torch_cuda_stream_synchronize`, ... | MUSA stream helpers |
| `STABLE_TORCH_LIBRARY_IMPL(<namespace>, CUDA, ...)` | `STABLE_TORCH_LIBRARY_IMPL(<namespace>, PrivateUse1, ...)` |
| `cudaGridDependencySynchronize()`, `cudaTriggerProgrammaticLaunchCompletion()` | `((void)0)` — no-ops, since MUSA has no equivalent |

Programmatic dependent launch is the one CUDA feature torchada deliberately maps to a
no-op rather than an equivalent: the two calls become `((void)0)` so kernels that use
them still build and run.

See `src/torchada/_mappings/` for 400+ mapping rules grouped by API domain (ATen,
c10, cuBLAS, CUDA runtime and driver, cuDNN, cuFFT, cuRAND, cuSOLVER, cuSPARSE, NCCL,
nvJPEG, libtorch-stable, FlashInfer, cutlass, thrust, ...).
`src/torchada/_mapping.py` remains the compatibility aggregation entry point.

## Integrating torchada into Your Project

### Step 1: Add Dependency

```
# pyproject.toml or requirements.txt
torchada>=0.1.95
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
| [ms-swift](https://github.com/modelscope/ms-swift) | Training / Fine-tuning | ✅ Merged | — |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Image/Video Generation | 🚧 In Progress | [ComfyUI#11618](https://github.com/Comfy-Org/ComfyUI/pull/11618) |

## License

MIT License
