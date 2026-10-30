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
| CUDA Runtime | `torch.cuda.cudart()` → uses MUSA runtime, `torch._C._cuda_getCurrentRawStream` shim |
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
- `MUSA_VISIBLE_DEVICES` is authoritative and mirrored to `CUDA_VISIBLE_DEVICES`; when
  only the CUDA variable is set, it is copied to the MUSA variable. Inductor's autotune
  subprocess reads the MUSA variable.

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

## MUSA Device Facts and Build Identity

`torchada.musa` is one place to ask what the hardware is and what a build targets,
instead of every project deriving its own copy: the per-arch table, the device
probe, the recorded profile for when no board is attached, and the build identity.

A wrong-arch mubin is not a crash on MUSA: it loads, runs, and leaves the output
buffer untouched while `musaDeviceSynchronize` reports success. That is why the
build side is a refusal rather than a warning.

Every question is named after the answer it gives, and lives on the object that
owns the fact -- the same shape `vllm` uses for `platform.supports_fp8()`:

```python
from torchada.musa import device

spec = device.get_device_spec(0)   # None when the probe cannot answer
spec.arch_name                     # "mp_31"
spec.arch_code                     # 310, the value device code compares against
spec.mp_count                      # the count this card reports at runtime
spec.supports_native_fp8           # FP8 arithmetic in hardware
spec.supports_native_fp4           # False on every arch as of MUSA 5.2.0
spec.has_s4_sqmma                  # the packed FP16 x S4 SQMMA atom
spec.bin_profile                   # the recorded row for this arch and bin, or None

# The same questions when you hold a device id instead of a spec:
device.supports_native_fp8(0)
device.supports_native_fp4(0)
device.has_s4_sqmma(0)
```

Two answers a hardware fact must not give. Whether the tuned kernels of *your*
framework have been validated on this arch is a property of that implementation and
its version, so the policy stays with the consumer: a shared flag cannot say which
kernel, which version or which input contract was validated. And whether the object
code in front of you *carries* the FP8 kernels is a build fact (`BuildTargets.fp8`
below), not a hardware one -- the two disagree whenever a wheel is built without
`-DENABLE_FP8`.

The per-arch facts themselves are one table, so "what arches exist and what can
each do" is a file you can read rather than a comparison to trace:

```python
from torchada.musa import targets

targets.get_arch_spec("mp_31")     # MusaArchSpec(arch_name='mp_31', arch_code=310,
                                   #             native_fp8=True, native_fp4=False,
                                   #             s4_sqmma_fp16=True, tested_mcc='5.2')
targets.native_fp8_arch_names()    # ("mp_31",): the instructions exist
targets.native_fp4_arch_names()    # (): no arch has FP4 arithmetic
targets.s4_sqmma_arch_names()      # ("mp_31",)
targets.target_names()             # the arches we build for: mp_21, mp_22, mp_31
targets.target_mp_counts("mp_31")  # (48, 52, 56, 60): the bins the table lists
```

The 4-bit field is named after the instruction rather than after a dtype:
`s4_sqmma_fp16` says that one packed SQMMA atom takes a signed 4-bit B operand.
Calling it `int4` would promise arithmetic the arch does not have in general, and a
caller reading "int4 support" would enable kernels that do not exist. Every other
capability is one boolean per question for the same reason: a generic
`supports_native(dtype)` reads as "this arch implements dtype X", which is exactly
the claim only the specific atom makes true.

When the card is not in front of you -- a tuning table to generate, a CI job, an
AOT variant list -- the same facts come from the recorded profile instead of a
probe:

```python
from torchada.musa import profiles

profile = profiles.bin_profile("mp_31", 56)  # None: no row for that arch or bin
profile.shared_memory_per_block_optin        # 196608 -- what a kernel may request
profile.shared_memory_per_multiprocessor     # 196608 (192 KiB)
profile.regs_per_multiprocessor              # 131072
profile.source                               # which probe these numbers came from
profiles.known_mp_counts("mp_31")            # (48, 52, 56, 60); () when unrecorded
profiles.verified_mp_counts("mp_31")         # (48, 56, 60): probed, not inferred
profiles.bin_profiles("mp_31")               # every modelled bin of the arch
profiles.profile_mismatch(spec)              # how the live board differs, or None
```

The row is per *arch* and serves every bin the table lists for it. That is an
assumption, so it is visible rather than implicit: `verified_mp_counts()` names the
bins that were actually probed, `source` names the probe, and every bin in that list
agreed field for field. `None` therefore means "this arch, or this bin, is not
modelled" -- never "a number I did not have". When a live board disagrees with its
row, `get_device_spec()` warns once per bin with the differing fields: the probe
wins, and a stale table is not applied silently.

```python
from torchada.musa import build

build.compiler_executable()                  # the mcc the build will actually run
build.mcc_version()                          # "5.2.0"; "unknown" when unparsable
build.check_toolchain(["mp_31"], "5.1.0")     # reports the version, refuses nothing
build.check_toolchain(["mp_31"], "5.1.0", floor="5.2")   # raises: a policy floor
build.check_toolchain(["mp_31"], "unknown")   # no compiler here: skipped, not failed
build.single_arch_define(targets=["mp_31"])   # "-D__MUSA_ARCH_LIST__=310"
build.mcc_arch_flags(["mp_22", "mp_31"])      # ["--offload-arch=mp_22", "--offload-arch=mp_31"]
build.build_cache_tag(["mp_22", "mp_31"])     # "mp_22_mp_31__mcc5.2.0" (JIT cache dir)
build.write_build_stamp(package_dir, ["mp_31"], fp8=True)   # from setup.py
build.assert_build_matches_device(spec, stamp)              # raises on mismatch
build.assert_build_matches_device(                          # or one call: read + compare
    spec, module_name="vllm_musa._build_stamp", name="vllm_musa"
)

# Which arch(es) should this build target?  One rule, shared by setup.py and JIT:
targets = build.resolve_targets(os.environ.get(build.TARGET_ENV), device_arch="mp_31")
targets.names                 # ("mp_31",) or ("mp_21", "mp_22") for a fat build
targets.fp8                   # may this build carry -DENABLE_FP8?
targets.fp4                   # ... and -DENABLE_FP4 (no arch has it yet)
targets.source                # "environment" | "device" | "default"
build.uniform_native_fp8(targets)   # opt in: refuse a set whose arches disagree
build.compiler_arch_list("31", targets)      # the TORCH_MUSA_ARCH_LIST value to build with

# Three consumers read the target, each in its own spelling.  One list, three forms:
build.arch_list_forms(["mp_31", "mp_22"])
# {'target': 'mp_22,mp_31', 'torch': '22;31', 'mate': '2.2;3.1'}
```

torchada's own operators take the same two answers from this module: `load_cpp_ops()`
builds `--offload-arch` with `mcc_arch_flags()` and caches the object under
`torchada_cpp_ops/<build_cache_tag()>`, i.e. per arch and compiler, which is what
stops a shared cache from serving another card's object. The tag's compiler comes
from `compiler_executable()`, which resolves like the builder does: `$PYTORCH_MCC` when
set, otherwise `<home>/bin/mcc` where `home` is `$MUSA_HOME`/`$MUSA_PATH`, then the
`mcc` on `PATH`, then `/usr/local/musa`. That is the command `torch_musa`'s extension
builder actually writes into its ninja file, and it may carry a launcher
(`ccache /path/to/mcc`), so `compiler_argv()` splits it into an argv and the version is
read without a shell. Reading whichever `mcc` happens to be first on `PATH`, or `$MCC`,
would tag a directory with a version that produced none of its object code.

`torchada.musa.build` imports only the standard library, so a project's `setup.py`
can read the arch table and write a stamp before the package it builds is
importable:

```python
import torchada_build            # no torch, no `torchada/__init__`

torchada_build.targets.get_arch_spec("mp_31").native_fp8   # True
torchada_build.build.build_cache_tag(["mp_31"], "5.2.0")   # "mp_31__mcc5.2.0"
```

`import torchada_build` loads `targets.py`, `profiles.py` and `build.py` under a
private package name, so nothing under `torchada` is executed. A plain
`from torchada.musa import build` is *not* a build-only path even though those
modules are stdlib-only: the package `__init__` imports torch, so that spelling
needs a torch install. (The modules loaded by the entry point are the same source
files but not the same module objects, so do not mix the two spellings in one
process.) The property is checked rather than promised: a test imports the entry
point with `torch`, `torch_musa` and `numpy` blocked and asserts that neither
`torch` nor `torchada` ends up in `sys.modules`.

`resolve_targets(..., default=None)` is the variant a JIT build uses: with no
stamp to check later, an unanswerable arch has to raise rather than be guessed and
cached.

Rules worth knowing:
- A fat build is a set, not a bigger number: `arch_codes()`, `mcc_arch_flags()`,
  `build_cache_tag()` and the stamp all take several targets (`mcc` bundles repeated
  `--offload-arch` into one `__CLANG_OFFLOAD_BUNDLE__`). `single_arch_define()`
  cannot, because the template macro is compared against a single `__MUSA_ARCH__`;
  a fat template branches on `__MUSA_ARCH__` instead.
- `resolve_targets()` answers the target set; whether a *mixed* FP8/non-FP8 set is
  legal is a source-layout question, so it is `uniform_native_fp8(targets)` and only
  the projects whose layout compiles those sources unconditionally call it.
  `-DENABLE_FP8` is one flag for the whole compilation, so `targets.fp8` is `any`
  over the arch answers.
- No MUSA arch has FP4 arithmetic today, so `NATIVE_FP4_ARCHS` is empty and every
  arch answers `False`. That is a statement about instructions, not about FP4
  checkpoints, which run through dequantisation.
- `MusaArchSpec.tested_mcc` records the compiler the arch has actually been built
  and validated with (`mp_31`: 5.2). It is evidence, not a floor: it refuses
  nothing, because "we tested 5.2" does not establish that 5.1 cannot build the same
  arch. A project with a support policy passes that minimum to
  `check_toolchain(floor=...)` itself, and a version of `"unknown"` (no `mcc` on
  this host, normal when consuming a wheel) skips the check rather than failing a
  build that is not happening here.
- Every entry point normalises the arch the same way, so `"21"`, `"2.1"`, `"210"` and
  `"mp_21"` reach the flags, the cache tag and the stamp identically; anything else
  is refused rather than scaled. The names are materialised once inside
  `arch_codes()`, so an iterator argument cannot be spent before the flags are
  produced, and an empty selection -- `[]`, `()`, `iter(())`, a spent generator --
  raises rather than returning no flags at all: no `--offload-arch` is mcc's own
  multi-arch default, i.e. the wrong-arch binary these functions exist to prevent.

- The arch of a device comes from `get_device_properties().major/minor`, never
  from `torch_musa.get_arch_list()` — that reports what the **wheel** was built
  for, which is `build.read_build_stamp()`'s job.
- A failed probe is not cached: a probe taken before the device is ready must not
  pin a process to the generic path for its whole lifetime.
- An absent stamp means *unknown*, not *mismatch*, so an editable install still
  runs; a stamp naming another arch is refused.

## Environment Variables

| Variable | Default | Effect |
|----------|---------|--------|
| `TORCHADA_PLATFORM` | auto-detect | Force platform detection: `cuda`, `musa`, or `cpu` |
| `TORCHADA_FORCE_CUDA_AVAILABLE` | unset | `1` makes `torch.cuda.is_available()` report `torch.musa.is_available()` |
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
the checked-in `benchmarks/benchmark_history.json` measurement for torchada 0.1.103
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

**`torch.cuda.is_available()` returns `False` on MUSA:**

```python
device = "cuda" if torch.cuda.is_available() else "cpu"  # silently picks the CPU on MUSA
```

This is intentional: it lets a project tell the two platforms apart. Use
`torchada.is_musa_platform()`, or the `has_gpu()` pattern in
[examples/migrate_existing_project.md](examples/migrate_existing_project.md#important-note-on-gpu-detection),
when you need to know whether a GPU is present.

For a project that is written for CUDA only and cannot be modified for MUSA, set
`TORCHADA_FORCE_CUDA_AVAILABLE=1` **before importing torchada**.
`torch.cuda.is_available()` then reports `torch.musa.is_available()`:

```bash
TORCHADA_FORCE_CUDA_AVAILABLE=1 python serve.py
```

```python
import torchada
import torch

torch.cuda.is_available()  # True on a MUSA GPU
device = torch.device("cuda")  # translated to musa
```

Only that single attribute changes. `torch.cuda.device_count()`,
`torch.cuda.get_device_name()`, `torch.cuda.memory_*()`, and the rest are
already redirected to `torch.musa` in both modes, and
`torchada.detect_platform()` still reports `Platform.MUSA`. The variable is read
once, when the patches are applied, so set it in the environment (or in
`os.environ`) before torchada is imported.

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

**Note**: `torch.cuda.is_available()` is intentionally NOT redirected by default — it returns `False` on MUSA. This allows proper platform detection. Set `TORCHADA_FORCE_CUDA_AVAILABLE=1` before importing torchada to have it report `torch.musa.is_available()` instead (see [Known Limitation](#known-limitation)). For GPU availability checks, see the `has_gpu()` pattern in [examples/migrate_existing_project.md](examples/migrate_existing_project.md#important-note-on-gpu-detection).

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
torchada>=0.1.104
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
