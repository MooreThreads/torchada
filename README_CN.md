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
| 张量工厂函数 | `torch.zeros(..., device="cuda")`、`torch.asarray(..., device="cuda")`、`*_like` 系列 |
| 显存管理 | `torch.cuda.memory_allocated()`、`empty_cache()`、`torch.cuda.memory.*`、CUDA 显存池 API |
| 同步 | `torch.cuda.synchronize()`, `Stream`, `Event`, `torch.cuda.streams` |
| 混合精度 | `torch.cuda.amp.autocast()`, `GradScaler()` |
| CUDA Graphs | `torch.cuda.CUDAGraph`、`torch.cuda.graph()`、深层模型的 graph executable 轮换 |
| CUDA 运行时 | `torch.cuda.cudart()` → 使用 MUSA 运行时 |
| 性能分析 | `ProfilerActivity.CUDA` → 使用 PrivateUse1 |
| 自定义算子 | `Library.impl(..., "CUDA")` → 使用 PrivateUse1 |
| 分布式训练 | `dist.init_process_group(backend='nccl')` → 使用 MCCL |
| torch.compile | Inductor、支持 AOT 缓存的张量工厂函数包装器、FX `device` builtin、`torch.cuda._get_device_index` |
| C++ 扩展 | `CUDAExtension`、`BuildExtension`、源码原地移植、稳定 ABI 兼容层 |
| FlexAttention | `torch.nn.attention.flex_attention` 支持 MUSA 设备 |
| C++ nvJPEG 移植 | nvJPEG 源码及构建配置 → MTJPEG |
| ctypes 库加载 | `ctypes.CDLL` 使用 CUDA 函数名 → 自动转换为 MUSA |
| 统一加速器 API | `torch.accelerator.empty_cache()`、`memory_stats()`、`get_memory_info()`、`Stream`、`Event` 等 |
| FlashAttention 提供方 | `flash_attn_interface` 重定向，以及 `only_qv` 与 `_flash_attn_forward` 兼容层 |
| Triton CUDA Extra | MUSA 上的 `tl.extra.cuda` → `tl.extra.musa` 兼容 |
| Triton 融合 MoE | 面向 vLLM 和 SGLang 的 Triton 3.2.0 MTT S5000 调优配置 |
| MUSA 算子级修复 | float64 原地 `log_`、异步 `isfinite`、`mm`/`bmm` 的 `out_dtype=` —— 详见 [torch_musa 兼容性](#torch_musa-兼容性) |

## torch_musa 兼容性

有些补丁之所以存在，只是因为某个 torch_musa 版本缺失或写坏了对应的内核。这类补丁都按版本门控：
只在需要它们的版本上安装，在修好该问题的版本上自动让位，因此底层栈升级时不需要任何配置。

| 兼容层 | 安装条件 | 作用 |
|--------|----------|------|
| float64 原地 `Tensor.log_` | torch_musa < `2.11.0.post2` | 复用受支持的非原地 `log` 并把结果写回，保持原地操作契约 |
| 编译态 `multinomial` / `log` / `log_` | torch_musa < `2.11.0.post2` | torchada JIT 构建的 C++ 扩展中的 MUSA `PrivateUse1` 算子覆盖，在 CUDA graph capture 下可正确重放 |
| Inductor MUSA 模板启发式 | torch_musa < `2.11.0.post2` | 用 CUDA 条目补齐 Inductor 模板启发式注册表中缺失的 `musa` 键 |
| `torch.accelerator` 显存 API | torch_musa < `2.11.0.post2` | 把 `empty_cache()` / `memory_stats()` 等转发到 `torch.musa`，因为该版本之前的 dispatch 有问题 |
| `torch.mm` / `torch.bmm` 的 `out_dtype=` | torch_musa < `2.13.0` | 复用普通重载并以 fp32 累加；运行期探针按进程判断原生重载是否健康 |
| 异步 `torch.isfinite` | 始终（MUSA 浮点张量） | float16/bfloat16/float32/float64 张量改用 `abs() < inf`：ATen 组合实现里的 bool `mul` 会阻塞主机直到设备队列清空。其他 dtype 仍走原算子 |
| libtorch 稳定 ABI 头文件 | MUSA 且 torch < 2.11 | 回补 torch_musa 2.9 快照缺失的 `torch::stable` 访问器，使 vLLM 与 SGLang 的稳定 ABI 内核可以构建 |
| `torch.cuda.streams` 模块路径 | MUSA，且 `torch_musa.core.stream` 可导入 | 将其暴露为 `torch.cuda.streams`，PyTorch 2.11 的 Dynamo guard 会解析该路径 |

`out_dtype` 兼容层在 **`torch_musa 2.13.0` 以下**武装（该版本是 torch_musa 承诺修复重载的发布）。那是 vendor 的发布承诺、不是我们的实测，
信任它就是已接受的风险：若 `2.13.0` 实际没修好，`>= 2.13.0` 的栈不安装包装层，静默全零会重新出现。门控只决定是否在 `torch.mm`/`torch.bmm`
前加一层 Python 包装；正确性由运行期探针按进程裁决（健康则直接转发、损坏才仿真），因此若修复被 backport 到 `2.12.x` 会自动生效。
`__version__` 无法解析时 rank 最低 ⇒ 仍然武装。待 `2.13.0` 发布并在此**验证修好**后删除该 shim；若承诺延期，则把上界抬到新的承诺版本。
该重载实测损坏于 `2.11.0.post1+musa5.2.0`（`mm` 全零、`bmm` 错值）。

**未覆盖**：binding 本身不含 `*_Dtype` 重载的构建，`out_dtype=` 仍会报错。那是 binding 的契约、不是 torchada 要修的缺陷，
为该情形做 CUDA 等价支持不在范围内。

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

深层模型还有第二层透明修复。MUSA 驱动对每进程**存活**的 graph executable 数量有约
2048 的上限，而 piecewise CUDA graphs 会实例化 `capture_sizes * num_layers` 个 executable，
因此约 40 层以上的模型会超限、完全无法使用 piecewise CUDA graphs。torchada 会保留所有已
capture 的模板（template），只对存活的 executable 做 LRU 上限管理（默认 1900），并在被淘汰
的 graph 下次 replay 时用其 template 重新实例化，代价约 0.3 ms 且无需重跑 forward。在上限
被触及之前它是零开销的，可通过下面的[环境变量](#环境变量)调整或关闭。

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

还有三处补丁保证编译路径端到端可用：

- `torch.cuda._get_device_index` 会映射到 MUSA 对应实现，TorchDynamo 正是通过它来建模
  `torch.cuda.device(...)`。缺少该映射时，编译一个打开设备上下文的函数会抛出
  `AttributeError: module 'torch_musa' has no attribute '_get_device_index'`。
- 打补丁后的 `torch.device` 会注册为 FX 的 `device` builtin，因此持有 device 常量的
  `GraphModule` 仍能绑定 `device` 这个名字，并在 `eager` 与 `aot_eager` 后端下正常运行。
  （Inductor 执行的是自己生成的代码，因此两种情况都不受影响。）
- 当两者都存在时，`MUSA_VISIBLE_DEVICES` 会镜像到 `CUDA_VISIBLE_DEVICES`，Inductor 的
  autotune 子进程读取的是 MUSA 变量。

在 `2.11.0.post2` 之前的 torch_musa 上，Inductor matmul 模板启发式注册表中缺失的 `musa`
条目会从 CUDA 条目补齐；更新的版本会自行注册，torchada 不做干预。

### Triton 融合 MoE 调优

torchada 为 vLLM 和 SGLang 内置在 MTT S5000 上调优的 Triton 3.2.0 融合 MoE
配置。内置配置包括 BF16、FP8 W8A8 和共享专家形状，以及布局一致的
JoyAI-LLM-Flash 上、下投影配置。调优结果与环境相关；其他 Triton 版本、硬件或
工作负载组合应使用自定义配置。

导入时，torchada 会通过 `SGLANG_MOE_CONFIG_DIR` 和
`VLLM_TUNED_CONFIG_FOLDER` 将 SGLang 与 vLLM 指向内置配置。已有环境变量不会
被覆盖；如需使用自定义配置，请在导入 torchada 前设置相应变量。

这些表是生成出来的、不是手写的。`tune_moe.py` 同目录下的 `ci/shapes.json` 与
`ci/models.json` 记录了内核形状、bucket，以及每个 bucket 固定的配置；
`tune_moe.py --config .../ci/shapes.json --materialize --merge-configs` 无需 GPU 即可重新
导出这些行，`tests/test_tune_moe_recipe.py` 会逐行逐键断言结果与随包发布的表一致。
`--merge-configs` 会保留本次运行未测量的每一行，因此改动会以可评审的 diff 形式出现。
调优与基准测试工具本身的说明见 [docs/tune_triton_moe.md](docs/tune_triton_moe.md)。

### FlashAttention 提供方（SGLang、vLLM-Omni）

当 MUSA `flash_attn_interface` 包可用时，torchada 会将
`sgl_kernel.flash_attn` 导入重定向到该实现。如果旧版 MUSA FA3 入口的函数签名
无法接受 SGLang 新版调用方传入的 `only_qv` 参数，包装器只会丢弃这一关键字；
如果实现原生支持该参数，则保持不变。

如果提供方暴露了公开的 output+softmax-LSE API 但没有
`flash_attn_interface._flash_attn_forward` 这个私有符号，torchada 会补上它。vLLM-Omni
的 Ring Attention 需要 LSE 来做部分注意力累加，因此会导入这个私有名字，缺少它时其
FA3 可用性探测会判定为不可用。该兼容层是对
`flash_attn_func(..., return_softmax_lse=True)` 的薄封装：公开推理 API 无法提供的
dropout 相关辅助返回值一律返回 `None`；若提供方没有同时返回 output 和 LSE 则直接失败。
提供方自带 `_flash_attn_forward` 时不做任何改动。

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

移植也会处理嵌套的 PyTorch 头文件：`<torch/cuda.h>` 和 `"torch/cuda.h"` 都会变成
`torch/musa.h`，而形如 `decode_jpegs_cuda.h` 的项目内文件名保持不变。

对稳定 ABI 内核，移植还会把 `STABLE_TORCH_LIBRARY_IMPL(<namespace>, CUDA, ...)` 改写为
`PrivateUse1` dispatch key（支持任意 namespace 与任意空白布局，包括跨行宏调用），并将稳定
ABI 的 CUDA 流辅助函数（`aoti_torch_get_current_cuda_stream`、
`torch_set_current_cuda_stream`、`torch_get_cuda_stream_from_pool`、
`torch_cuda_stream_synchronize`）以及 BLAS handle 访问器映射到 MUSA 对应实现。

`include_paths()` 与 `library_paths()` 遵循 PyTorch 2.6+ 的函数签名，位置参数和关键字
参数都支持。这在 `TORCHINDUCTOR_CACHE_DIR` 为空时尤其重要：Inductor 的 C++ builder 对
CPU 内核同样会调用这两个函数，而旧的 MUSA 专用签名会在那里报错，导致 `import torchada`
之后一个普通的 CPU `torch.compile` 直接失败。

C++ 算子覆盖扩展在首次使用时 JIT 构建。torchada 会在自己的 `flock` 下构建它，因此进程
在构建中途被杀而残留的 `lock` 文件不会再让之后每次 `import torchada` 都永久阻塞在 JIT
loader 的 `FileBaton` 上，并发导入也只会编译一次而不再互相竞争。若 `flock` 不可用
（部分 NFS 挂载），torchada 会给出警告并回退到 torch 的原有行为。

### 自定义算子

```python
import torchada
import torch

my_lib = torch.library.Library("my_lib", "DEF")
my_lib.define("my_op(Tensor x) -> Tensor")
my_lib.impl("my_op", my_func, "CUDA")  # 在 MUSA 上也能工作！
```

如需在 `PrivateUse1` dispatch key 上以 C++ 方式覆盖 ATen 算子，请参见
[docs/custom_musa_ops.md](docs/custom_musa_ops.md)。

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
torch.accelerator.get_memory_info()
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

**前向兼容性：** 包装器优先使用真正的 `torch.accelerator` 实现，只有在缺少属性时才回退到
`torch.musa`。唯一例外是 `2.11.0.post2` 之前的 torch_musa：已知有问题的 accelerator 内存 API
会被强制转发到 `torch.musa`。从 `2.11.0.post2` 开始，将自动使用已修复的官方实现。

## 环境变量

| 环境变量 | 默认值 | 作用 |
|----------|--------|------|
| `TORCHADA_PLATFORM` | 自动检测 | 强制指定平台：`cuda`、`musa` 或 `cpu` |
| `TORCHADA_EXCLUDE_DIRS` | 未设置 | 原地源码移植额外排除的 include 根目录 |
| `TORCHADA_CUDA_GRAPH_DEBUG_DUMP_PATH` | 未设置 | MUSA graph `.dot` dump 目录 |
| `TORCHADA_GRAPH_ROTATION` | `1` | 设为 `0` 关闭 CUDA graph executable 轮换 |
| `TORCHADA_GRAPH_EXEC_CAP` | `1900` | 每进程存活的 graph executable 上限 |
| `TORCHADA_GRAPH_AUTOPROBE` | `0` | 设为 `1` 时在启动阶段探测驱动真实的 executable 上限 |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `128` | 在探测到的上限之下保留的余量 |
| `TORCHADA_CPP_OPS_VERBOSE` | `0` | 设为 `1` 时打印 C++ 算子覆盖扩展的构建日志 |
| `TORCHADA_DEBUG_CPP_OPS` | `0` | 设为 `1` 时记录算子覆盖调用 |
| `TORCHADA_DISABLE_OP_OVERRIDE_<OP_NAME>` | 未设置 | 设为 `1` 可禁用单个算子覆盖，例如 `TORCHADA_DISABLE_OP_OVERRIDE_log=1` |

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

torchada 使用激进的缓存策略来最小化运行时开销。下表的数字取自仓库中已提交的
`benchmarks/benchmark_history.json`（torchada 0.1.95，2026-10-09，MTT S5000，
PyTorch 与 torch_musa `2.11.0.post2+musa5.2.0`），为每次调用的开销中位数：

| 操作 | 开销 |
|------|------|
| `torch.cuda.Stream`（属性访问） | ~170ns |
| `torch.cuda.Event`（属性访问） | ~170ns |
| `_translate_device('cuda')` | ~180ns |
| `torch.cuda.device_count()` | ~200ns |
| `torch.backends.cuda.is_built()` | ~260ns |

作为对比，典型的 GPU 内核启动耗时 5,000-20,000ns，因此补丁开销对于实际应用来说可以忽略不计。

具有固有成本的操作（运行时调用、对象创建）耗时 400-600ns，但在不改变行为的情况下无法进一步优化。

同一文件中 0.1.94 的记录是在 torch_musa 2.7.1 上测得的，上述快路径约为 120-160ns。

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
| `#include <torch/cuda.h>` | `#include <torch/musa.h>` |
| `__CUDA_ARCH__ < 800` | `__MUSA_ARCH__ < 220` |
| `nvjpeg.h`、`nvjpeg*`、`NVJPEG*` | `mtjpeg.h`、`mtjpeg*`、`MTJPEG*` |
| `libraries=["nvjpeg"]` | `libraries=["mtjpeg"]` |
| `NVJPEG_FOUND` | `MTJPEG_FOUND` |
| `aoti_torch_get_current_cuda_stream`、`torch_cuda_stream_synchronize` 等 | MUSA 对应流辅助函数 |
| `STABLE_TORCH_LIBRARY_IMPL(<namespace>, CUDA, ...)` | `STABLE_TORCH_LIBRARY_IMPL(<namespace>, PrivateUse1, ...)` |
| `cudaGridDependencySynchronize()`、`cudaTriggerProgrammaticLaunchCompletion()` | `((void)0)` —— MUSA 无对应实现，直接空操作 |

程序化依赖启动（programmatic dependent launch）是 torchada 唯一有意映射为空操作而非等价
实现的 CUDA 特性：这两个调用会被替换为 `((void)0)`，使用它们的 kernel 仍可正常构建与运行。

按 API 领域组织的 400+ 条映射规则请参见 `src/torchada/_mappings/`（包括 ATen、c10、
cuBLAS、CUDA 运行时与驱动、cuDNN、cuFFT、cuRAND、cuSOLVER、cuSPARSE、NCCL、nvJPEG、
libtorch-stable、FlashInfer、cutlass、thrust 等）。
`src/torchada/_mapping.py` 保留为兼容性聚合入口。

## 将 torchada 集成到你的项目

### 步骤 1：添加依赖

```
# pyproject.toml 或 requirements.txt
torchada>=0.1.95
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
| [ms-swift](https://github.com/modelscope/ms-swift) | 训练/微调 | ✅ 已合并 | — |
| [ComfyUI](https://github.com/Comfy-Org/ComfyUI) | 图像/视频生成 | 🚧 进行中 | [ComfyUI#11618](https://github.com/Comfy-Org/ComfyUI/pull/11618) |


## 许可证

MIT License
