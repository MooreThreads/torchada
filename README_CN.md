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

就这么简单！所有 `torch.cuda.*` API 会自动重定向到 `torch.musa.*`。

## 支持的功能

| 功能 | 示例 |
|------|------|
| 设备操作 | `tensor.cuda()`, `model.cuda()`, `torch.device("cuda")` |
| 显存管理 | `torch.cuda.memory_allocated()`, `empty_cache()` |
| 同步 | `torch.cuda.synchronize()`, `Stream`, `Event` |
| 混合精度 | `torch.cuda.amp.autocast()`, `GradScaler()` |
| CUDA Graphs | `torch.cuda.CUDAGraph`, `torch.cuda.graph()` |
| CUDA 运行时 | `torch.cuda.cudart()` → 使用 MUSA 运行时 |
| 性能分析 | `ProfilerActivity.CUDA` → 使用 PrivateUse1 |
| 自定义算子 | `Library.impl(..., "CUDA")` → 使用 PrivateUse1 |
| 分布式训练 | `dist.init_process_group(backend='nccl')` → 使用 MCCL |
| torch.compile | `torch.compile(model)` 支持所有后端 |
| C++ 扩展 | `CUDAExtension`, `BuildExtension`, `load()` |
| FlexAttention | `torch.nn.attention.flex_attention` 支持 MUSA 设备 |
| ctypes 库加载 | `ctypes.CDLL` 使用 CUDA 函数名 → 自动转换为 MUSA |
| 统一加速器 API | `torch.accelerator.empty_cache()`、`memory_stats()`、`Stream`、`Event` 等 |

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

### torch.compile

```python
import torchada
import torch

compiled_model = torch.compile(model.cuda(), backend='inductor')
```

### 构建 C++ 扩展

```python
import torchada  # 必须在 torch.utils.cpp_extension 之前导入
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# 标准 CUDAExtension 可直接使用 — torchada 处理 CUDA→MUSA 转换
ext = CUDAExtension("my_ext", sources=["kernel.cu"])
```

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

## API 参考

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

完整映射表（380+ 条映射）请参见 `src/torchada/_mapping.py`。

## 将 torchada 集成到你的项目

### 步骤 1：添加依赖

```
# pyproject.toml 或 requirements.txt
torchada>=0.1.52
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
| [Xinference](https://github.com/xorbitsai/inference) | 模型服务 | ✅ 已合并 | — |
| [LightLLM](https://github.com/ModelTC/LightLLM) | 模型服务 | ✅ 已合并 | — |
| [LightX2V](https://github.com/ModelTC/LightX2V) | 图像/视频生成 | ✅ 已合并 | — |
| [赤兔](https://github.com/thu-pacman/chitu) | 模型服务 | ✅ 已合并 | — |
| [vLLM-MUSA](https://github.com/MooreThreads/vllm-musa) | 模型服务 | ✅ 已合并 | — |
| [SGLang](https://github.com/sgl-project/sglang) | 模型服务 | 🚧 进行中 | [SGLang#16565](https://github.com/sgl-project/sglang/issues/16565) |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 图像/视频生成 | 🚧 进行中 | [ComfyUI#11618](https://github.com/Comfy-Org/ComfyUI/pull/11618) |
| [vLLM-Omni](https://github.com/vllm-project/vllm-omni) | 模型服务 (Omni) | 🚧 进行中 | [vLLM-Omni#2347](https://github.com/vllm-project/vllm-omni/issues/2347) |

## 许可证

MIT License
