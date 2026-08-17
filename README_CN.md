<div align="center" id="sglangtop">
<img src="https://raw.githubusercontent.com/MooreThreads/torchada/main/assets/logo.png" alt="logo" width="250" margin="10px"></img>
</div>

--------------------------------------------------------------------------------

# torchada

[English](README.md) | 中文

**在摩尔线程 GPU 上运行你的 CUDA 代码 — 无需任何代码改动**

torchada 是一个适配器，让 [torch_musa](https://github.com/MooreThreads/torch_musa)（摩尔线程 GPU 的 PyTorch 支持）兼容标准的 PyTorch CUDA API。只需导入一次，你现有的 `torch.cuda.*` 代码就能在 MUSA 硬件上运行。

## 为什么需要 torchada？

许多 PyTorch 项目使用 `torch.cuda.*` API 为 NVIDIA GPU 编写。要在摩尔线程 GPU 上运行这些项目，通常需要把每个 `cuda` 引用改成 `musa`。torchada 通过在运行时自动将 CUDA API 调用转换为 MUSA 等效调用来消除这一问题。

## 前置条件

- **torch_musa**：必须安装 [torch_musa](https://github.com/MooreThreads/torch_musa)（提供 PyTorch 的 MUSA 支持）
- **摩尔线程 GPU**：已安装正确驱动的摩尔线程 GPU

## 安装

```bash
pip install torchada

# 或从源码安装
git clone https://github.com/MooreThreads/torchada.git
cd torchada
pip install -e .
```

## 快速开始

```python
import torchada  # ← 在文件顶部添加这一行
import torch

# 你现有的 CUDA 代码无需改动：
x = torch.randn(10, 10).cuda()
print(torch.cuda.device_count())
torch.cuda.synchronize()
```

就这么简单！支持的 `torch.cuda.*` API 会自动重定向到 `torch.musa.*`。

## 支持的功能

| 功能 | 示例 |
|------|------|
| 设备操作 | `tensor.cuda()`, `model.cuda()`, `torch.device("cuda")` |
| 张量工厂函数 | `torch.zeros(..., device="cuda")`、`torch.asarray(..., device="cuda")` → MUSA |
| 显存管理 | `torch.cuda.memory_allocated()`, `empty_cache()` |
| 同步 | `torch.cuda.synchronize()`, `Stream`, `Event` |
| 混合精度 | `torch.cuda.amp.autocast()`, `GradScaler()` |
| CUDA Graphs | `torch.cuda.CUDAGraph`, `torch.cuda.graph()` |
| CUDA 运行时 | `torch.cuda.cudart()` → 使用 MUSA 运行时 |
| 性能分析 | `ProfilerActivity.CUDA` → 使用 PrivateUse1 |
| 自定义算子 | `Library.impl(..., "CUDA")` → 使用 PrivateUse1 |
| 分布式训练 | `dist.init_process_group(backend='nccl')` → 使用 MCCL |
| torch.compile | Inductor，以及支持 AOT 缓存的张量工厂函数包装器 |
| C++ 扩展 | `CUDAExtension`、`BuildExtension`、源码原地移植、稳定 ABI 兼容层 |
| FlexAttention | `torch.nn.attention.flex_attention` 支持 MUSA 设备 |
| C++ nvJPEG 移植 | nvJPEG 源码及构建配置 → MTJPEG |
| ctypes 库加载 | `ctypes.CDLL` 使用 CUDA 函数名 → 自动转换为 MUSA |
| 统一加速器 API | `torch.accelerator.empty_cache()`、`memory_stats()`、`Stream`、`Event` 等 |
| Triton CUDA Extra | MUSA 上的 `tl.extra.cuda` → `tl.extra.musa` 兼容 |
| Triton 融合 MoE | 面向 vLLM 和 SGLang 的 Triton 3.2.0 MTT S5000 调优配置 |

## 示例

### 混合精度训练

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

### 分布式训练

```python
import torchada
import torch.distributed as dist

# 'nccl' 会自动映射到 MUSA 上的 'mccl'
dist.init_process_group(backend='nccl')
```

### CUDA Graphs

```python
import torchada
import torch

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(cuda_graph=g):  # cuda_graph= 关键字参数在 MUSA 上也能工作
    y = model(x)
```

如果需要 dump MUSA graph 的 dot 文件用于调试，可以在运行前设置
`TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH`。torchada 会在每次 graph capture 前调用
`enable_debug_mode()`，并在 capture 结束后调用 `debug_dump(path)`：

```bash
TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH=./graph_dumps \
python serve.py
```

该变量表示 dump 目录。torchada 会按需创建目录，并在其中写入带时间戳的文件，
例如 `graph_1783512345678900000.dot`，避免多次 capture 时互相覆盖。

### torch.compile

```python
import torchada
import torch

compiled_model = torch.compile(model.cuda(), backend='inductor')
```

`torch.zeros(..., device="cuda")`、`torch.asarray(..., device="cuda")` 以及
`*_like` 系列张量工厂函数也会把显式 CUDA 设备转换为 MUSA。这些包装器与
CUDA Graph capture 和 `torch.compile` AOT 缓存保持兼容。打补丁后的
`torch.device(...)` 也仍可在 TorchScript 中使用。

### Triton 融合 MoE 调优

torchada 为 vLLM 和 SGLang 内置在 MTT S5000 上调优的 Triton 3.2.0 融合 MoE
配置。内置配置包括 BF16、FP8 W8A8 和共享专家形状，以及布局一致的
JoyAI-LLM-Flash 上、下投影配置。调优结果与环境相关；其他 Triton 版本、硬件或
工作负载组合应使用自定义配置。

导入时，torchada 会通过 `SGLANG_MOE_CONFIG_DIR` 和
`VLLM_TUNED_CONFIG_FOLDER` 将 SGLang 与 vLLM 指向内置配置。已有环境变量不会
被覆盖；如需使用自定义配置，请在导入 torchada 前设置相应变量。

### SGLang FlashAttention

当 MUSA `flash_attn_interface` 包可用时，torchada 会将
`sgl_kernel.flash_attn` 导入重定向到该实现。如果旧版 MUSA FA3 入口的函数签名
无法接受 SGLang 新版调用方传入的 `only_qv` 参数，包装器只会丢弃这一关键字；
如果实现原生支持该参数，则保持不变。

### 构建 C++ 扩展

```python
import torchada  # 必须在 torch.utils.cpp_extension 之前导入
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 标准 CUDAExtension 可直接使用 — torchada 处理 CUDA→MUSA 转换。
ext = CUDAExtension("my_ext", sources=["kernel.cu"])
```

如果扩展使用 nvJPEG，可以保留现有 CUDA 构建配置：

```python
jpeg_ext = CUDAExtension(
    "jpeg_ext",
    sources=["decode.cu"],
    libraries=["nvjpeg"],
    define_macros=[("NVJPEG_FOUND", "1")],
)
```

在 MUSA 上，`BuildExtension` 会**原地**移植项目内的 C/C++/CUDA 源码及头文件
内容。原有 `.cu`/`.cuh` 文件名和路径保持不变，也不会创建 `<dir>_musa` 镜像；
原生 `.mu`/`.muh` 文件及非源码文件保持不变。由于符合条件的源码会在构建时被
改写，如果需要保留原始 CUDA 内容，请使用干净或一次性的检出目录进行构建。为
避免修改项目目录之外的链接目标，移植器会拒绝符号链接形式的可移植源码和头文件。

移植器会同时转换 CUDA 架构条件及其阈值，并保留那些原本正确、但转换后会坍缩
为自引用的 CUDA→MUSA 映射宏。它还会将规范形式的 `nvjpeg*`/`NVJPEG*` 符号及
精确的 `nvjpeg.h` include 转换为 MTJPEG，并在 MUSA 上把
`libraries=["nvjpeg"]` 转换为 `mtjpeg`、把 `NVJPEG_FOUND` 转换为
`MTJPEG_FOUND`。CUDA 构建仍保留原始配置。

torchada 还为近期 vLLM 和 SGLang 在 torch_musa 2.9 上使用的 libtorch 稳定 ABI
内核提供兼容头文件及源码移植支持。在 MUSA 上，打补丁后的
`torch.utils.cpp_extension.include_paths()` 会返回该兼容 include 目录。自定义
稳定 ABI 构建应显式加入 `stable_compat_include_dir()`；使用 `TORCH_BOX` 的内核
还必须通过编译器强制 include `stable_compat_box_header()` 返回的头文件。这两个
辅助函数都位于 `torchada.utils.cpp_extension`。自定义 stable ABI 扩展应通过
上述方式显式加入兼容头。torch_musa 2.9 头文件回补会在 MUSA 扩展构建时延迟、
尽力执行；torch 2.11 及更新版本已经原生提供 stable ABI，因此会跳过回补。单纯
`import torchada` 不会修改 PyTorch 或 torch_musa 头文件。

原地 CUDA 到 MUSA 的转换会继续按原有规则保护系统 include 目录。可以通过环境变量
`TORCHADA_EXCLUDE_DIRS` 额外配置要排除的目录；每一项可以是目录路径，也可以是目录
名称，使用平台路径分隔符，也支持逗号分隔。名称会按完整路径组件匹配，因此
`TORCHADA_EXCLUDE_DIRS=torch_musa` 可以直接排除 `/home/torch_musa`，无需填写完整路径。
即使源目录位于排除目录下，扩展显式提供的源目录仍会执行转换。

### 自定义算子

```python
import torchada
import torch

my_lib = torch.library.Library("my_lib", "DEF")
my_lib.define("my_op(Tensor x) -> Tensor")
my_lib.impl("my_op", my_func, "CUDA")  # 在 MUSA 上也能工作！
```

### 性能分析

```python
import torchada
import torch

# ProfilerActivity.CUDA 在 MUSA 上也能工作
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
) as prof:
    model(x)
```

### ctypes 库加载

```python
import torchada
import ctypes

# 使用 CUDA 函数名加载 MUSA 运行时库
lib = ctypes.CDLL("libmusart.so")
func = lib.cudaMalloc  # 自动转换为 musaMalloc

# 同样适用于 MCCL
nccl_lib = ctypes.CDLL("libmccl.so")
func = nccl_lib.ncclAllReduce  # 自动转换为 mcclAllReduce
```

### 统一加速器 API（`torch.accelerator`）

`torch.accelerator` 是 PyTorch 的统一后端无关入口。它的 API 在不同的 PyTorch 版本中逐步扩展，
因此像 `empty_cache()`、`memory_stats()`、`Stream` 和 `Event` 等 API 在 torch 2.7 中尚未存在，
即使它们已经在 `torch.musa` 中提供。torchada 封装了 `torch.accelerator`，使得针对更新统一 API
编写的代码可以立即使用：

```python
import torchada
import torch

# torch 2.7 中已存在的 API 保持使用官方实现
torch.accelerator.is_available()
torch.accelerator.device_count()

# torch 2.7 中缺失的 API 透明地回退到 torch.musa
torch.accelerator.empty_cache()
torch.accelerator.memory_allocated()
torch.accelerator.memory_stats()
torch.accelerator.manual_seed(42)
s = torch.accelerator.Stream()
e = torch.accelerator.Event()

# 修复为委托给 torch.musa.synchronize()（默认 MUSA 实现不支持同步设备上的所有流）
torch.accelerator.synchronize()

# 前向兼容 PyTorch 2.9+ 的上下文管理器
with torch.accelerator.device_index(0):
    ...
with torch.accelerator.stream(torch.musa.Stream()):
    ...
```

**前向兼容性：** 包装器始终优先使用真正的 `torch.accelerator` 实现，只有在缺少属性时才回退到
`torch.musa`，因此升级到提供官方实现的未来 PyTorch 版本时无需任何更改 —— 您将自动获得上游版本。

## 平台检测

```python
import torchada
from torchada import detect_platform, Platform

platform = detect_platform()
if platform == Platform.MUSA:
    print("在摩尔线程 GPU 上运行")
elif platform == Platform.CUDA:
    print("在 NVIDIA GPU 上运行")

# 或使用基于 torch.version 的检测
def is_musa():
    import torch
    return hasattr(torch.version, 'musa') and torch.version.musa is not None
```

## 性能

torchada 使用激进的缓存策略来最小化运行时开销。所有频繁调用的操作都在 200 纳秒内完成：

| 操作 | 开销 |
|------|------|
| `torch.cuda.device_count()` | ~140ns |
| `torch.cuda.Stream`（属性访问） | ~130ns |
| `torch.cuda.Event`（属性访问） | ~130ns |
| `_translate_device('cuda')` | ~140ns |
| `torch.backends.cuda.is_built()` | ~155ns |

作为对比，典型的 GPU 内核启动耗时 5,000-20,000ns。补丁开销对于实际应用来说可以忽略不计。

具有固有成本的操作（运行时调用、对象创建）耗时 300-600ns，但在不改变行为的情况下无法进一步优化。

## 已知限制

**设备类型字符串比较在 MUSA 上会失败：**

```python
device = torch.device("cuda:0")  # 在 MUSA 上会变成 musa:0
device.type == "cuda"  # 返回 False！
```

**解决方案：** 使用 `torchada.is_gpu_device()`：

```python
import torchada

if torchada.is_gpu_device(device):  # 在 CUDA 和 MUSA 上都能工作
    ...
# 或者: device.type in ("cuda", "musa")
```

## 常用 API 参考

| 函数 | 描述 |
|------|------|
| `detect_platform()` | 返回 `Platform.CUDA`、`Platform.MUSA` 或 `Platform.CPU` |
| `is_musa_platform()` | 在 MUSA 上运行时返回 True |
| `is_cuda_platform()` | 在 CUDA 上运行时返回 True |
| `is_gpu_device(device)` | 设备是 CUDA 或 MUSA 时返回 True |
| `CUDA_HOME` | CUDA/MUSA 安装路径 |
| `cuda_to_musa_name(name)` | 转换 `cudaXxx` → `musaXxx` |
| `nccl_to_mccl_name(name)` | 转换 `ncclXxx` → `mcclXxx` |
| `cublas_to_mublas_name(name)` | 转换 `cublasXxx` → `mublasXxx` |
| `curand_to_murand_name(name)` | 转换 `curandXxx` → `murandXxx` |

**注意**：`torch.cuda.is_available()` 故意没有重定向 — 在 MUSA 上返回 `False`。这是为了支持正确的平台检测。关于 GPU 可用性检查，请参见 [examples/migrate_existing_project.md](examples/migrate_existing_project.md#important-note-on-gpu-detection) 中的 `has_gpu()` 模式。

**注意**：名称转换工具函数可供手动使用，但 `ctypes.CDLL` 已自动打补丁，加载 MUSA 库时会自动转换函数名。

## C++ 扩展符号映射

构建 C++ 扩展时，torchada 会自动将 CUDA 符号转换为 MUSA：

| CUDA | MUSA |
|------|------|
| `cudaMalloc` | `musaMalloc` |
| `cudaStream_t` | `musaStream_t` |
| `cublasHandle_t` | `mublasHandle_t` |
| `at::cuda` | `at::musa` |
| `c10::cuda` | `c10::musa` |
| `#include <cuda/*>` | `#include <musa/*>` |
| `__CUDA_ARCH__ < 800` | `__MUSA_ARCH__ < 220` |
| `nvjpeg.h`、`nvjpeg*`、`NVJPEG*` | `mtjpeg.h`、`mtjpeg*`、`MTJPEG*` |
| `libraries=["nvjpeg"]` | `libraries=["mtjpeg"]` |
| `NVJPEG_FOUND` | `MTJPEG_FOUND` |

按 API 领域组织的 400+ 条映射规则请参见 `src/torchada/_mappings/`。
`src/torchada/_mapping.py` 保留为兼容性聚合入口。

## 将 torchada 集成到你的项目

### 步骤 1：添加依赖

```
# pyproject.toml 或 requirements.txt
torchada>=0.1.82
```

### 步骤 2：条件导入

```python
# 在应用入口处
def is_musa():
    import torch
    return hasattr(torch.version, "musa") and torch.version.musa is not None

if is_musa():
    import torchada  # noqa: F401

# 其余代码正常使用 torch.cuda.*
```

### 步骤 3：扩展功能标志（如适用）

```python
# 在 GPU 能力检查中包含 MUSA
if is_nvidia() or is_musa():
    ENABLE_FLASH_ATTENTION = True
```

### 步骤 4：修复设备类型检查（如适用）

```python
# 不要用: device.type == "cuda"
# 改用: device.type in ("cuda", "musa")
# 或者: torchada.is_gpu_device(device)
```

## 使用 torchada 的项目

| 项目 | 类别 | 状态 | 跟踪 |
|------|------|------|------|
| [SGLang](https://github.com/sgl-project/sglang) | 模型服务 | ✅ 已合并 | — |
| [vLLM-MUSA](https://github.com/MooreThreads/vllm-musa) | 模型服务 | ✅ 已合并 | — |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | 模型服务 (Omni) | ✅ 已合并 | — |
| [Xinference](https://github.com/xorbitsai/inference) | 模型服务 | ✅ 已合并 | — |
| [LightLLM](https://github.com/ModelTC/LightLLM) | 模型服务 | ✅ 已合并 | — |
| [LightX2V](https://github.com/ModelTC/LightX2V) | 图像/视频生成 | ✅ 已合并 | — |
| [赤兔](https://github.com/thu-pacman/chitu) | 模型服务 | ✅ 已合并 | — |
| [Mooncake](https://github.com/kvcache-ai/Mooncake) | KV 缓存 | ✅ 已合并 | — |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 图像/视频生成 | 🚧 进行中 | [ComfyUI#11618](https://github.com/Comfy-Org/ComfyUI/pull/11618) |


## 许可证

MIT License
