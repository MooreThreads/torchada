"""
Automatic patch orchestration for torchada.

This module patches PyTorch to automatically translate 'cuda' device strings
to 'musa' when running on Moore Threads hardware.

Usage:
    import torchada  # Apply all patches automatically.
    import torch

    # Use torch.cuda APIs normally; they resolve to MUSA on MUSA platforms.
    torch.cuda.is_available()
    x = torch.randn(3, 3).cuda()
    from torch.cuda.amp import autocast, GradScaler

    # Distributed training with NCCL resolves to MCCL on MUSA platforms.
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")  # Uses MCCL on MUSA.

    # CUDA graph APIs resolve to MUSA graph APIs on MUSA platforms.
    g = torch.cuda.CUDAGraph()  # Uses MUSAGraph on MUSA.
"""

import functools
import inspect
import sys
import warnings
from types import ModuleType
from typing import Callable, List, Optional

import torch

from . import _accelerator_compat as _accelerator_compat
from . import _ctypes_compat as _ctypes_compat
from . import _device_compat as _device_compat
from ._accelerator_compat import patch_torch_accelerator
from ._cpp_ops import get_module
from ._ctypes_compat import patch_ctypes_cdll
from ._cuda_compat import (
    _CudaModuleWrapper,
    install_cuda_memory_compat,
    install_cuda_module_aliases,
    install_cuda_public_api_shims,
)
from ._device_compat import (
    _FACTORY_FUNCTIONS,
    _translate_device,
    _wrap_factory_function,
    _wrap_module_cuda,
    _wrap_tensor_cuda,
    _wrap_to_method,
    patch_torch_device,
    patch_torch_generator,
)
from ._platform import is_musa_platform

_patched = False
_original_init_process_group = None

_DYNAMIC_COMPAT_ATTRS = {
    "_original_torch_device": _device_compat.get_original_torch_device,
    "_original_torch_generator": _device_compat.get_original_torch_generator,
    "_original_c_generator": _device_compat.get_original_c_generator,
    "_original_ctypes_CDLL": _ctypes_compat.get_original_ctypes_cdll,
    "_original_torch_accelerator": _accelerator_compat.get_original_torch_accelerator,
}


def __getattr__(name: str):
    """Expose moved compatibility state for existing internal imports/tests."""
    getter = _DYNAMIC_COMPAT_ATTRS.get(name)
    if getter is not None:
        return getter()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Patch registry.
_patch_registry: List[Callable[[], None]] = []


def patch_function(func: Callable[[], None]) -> Callable[[], None]:
    """
    Decorator to register a function to be called during patching.

    This follows the registration pattern used in frameworks like Flask (@app.route),
    pytest (@pytest.fixture), and Django (@receiver). It allows patch functions
    to be defined anywhere in the module and automatically collected for application.

    Usage:
        @patch_function
        def _patch_something():
            # Patching logic.
            pass

    The decorated function will be called by apply_patches() in registration order.
    """
    _patch_registry.append(func)
    return func


def requires_import(*module_names: str) -> Callable[[Callable], Callable]:
    """
    Decorator to guard a patch function with import checks.

    If any of the specified modules cannot be imported, the decorated function
    returns early without executing. This replaces repetitive try/except patterns.

    Usage:
        @patch_function
        @requires_import('torch_musa')
        def _patch_something():
            # This only runs if torch_musa is importable.
            import torch_musa
            # Patching logic.

        @patch_function
        @requires_import('torch._inductor.autotune_process')
        def _patch_autotune():
            import torch._inductor.autotune_process as ap
            # Patching logic.

    Args:
        *module_names: Variable number of module names to check for importability

    Returns:
        A decorator that wraps the function with import guards
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for module_name in module_names:
                try:
                    __import__(module_name)
                except ImportError:
                    return None
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _has_param(func: Callable, param_name: str) -> bool:
    """
    Check if a function has a specific parameter in its signature.

    Args:
        func: The function to check
        param_name: The parameter name to look for

    Returns:
        True if the function has the parameter, False otherwise
    """
    try:
        sig = inspect.signature(func)
        return param_name in sig.parameters
    except (ValueError, TypeError):
        return False


@patch_function
def _patch_torch_device():
    """
    Patch torch.device to translate 'cuda' to 'musa' on MUSA platform.

    This ensures that torch.device("cuda:0") creates a musa device when on MUSA.
    """
    patch_torch_device(torch)


@patch_function
def _patch_torch_generator():
    """
    Patch torch.Generator to translate 'cuda' device to 'musa' on MUSA platform.

    This ensures that torch.Generator(device="cuda") creates a MUSA generator
    instead of failing with "Cannot get CUDA generator without ATen_cuda library".

    Uses a metaclass to properly implement __instancecheck__ so that
    isinstance(gen, torch.Generator) works correctly.
    """
    patch_torch_generator(torch)


# Saved graph class used by the graph context-manager patch.
_original_graph_class = None


def _patch_graph_context_manager():
    """
    Patch torch.cuda.graph context manager to accept cuda_graph= keyword argument.

    MUSA's graph class uses musa_graph= as the first parameter, but CUDA code
    uses cuda_graph=. This wrapper translates cuda_graph= to musa_graph= so that
    existing CUDA code works transparently on MUSA.
    """
    global _original_graph_class

    if _original_graph_class is not None:
        return

    # Read from torch.cuda after module redirection so this is torch.musa on MUSA.
    if not hasattr(torch.cuda, "graph"):
        return

    _original_graph_class = torch.cuda.graph

    class GraphWrapper:
        """Wrapper for torch.cuda.graph that accepts cuda_graph= keyword argument."""

        # Preserve class attributes used by callers.
        default_capture_stream = None

        def __init__(
            self,
            cuda_graph=None,
            pool=None,
            stream=None,
            capture_error_mode: str = "global",
            *,
            musa_graph=None,  # Also accept musa_graph for compatibility.
        ):
            # Accept CUDA and MUSA keyword spellings.
            graph_obj = cuda_graph if cuda_graph is not None else musa_graph
            if graph_obj is None:
                raise TypeError("graph() missing required argument: 'cuda_graph'")

            self._wrapped = _original_graph_class(
                graph_obj,
                pool=pool,
                stream=stream,
                capture_error_mode=capture_error_mode,
            )

        def __enter__(self):
            return self._wrapped.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            return self._wrapped.__exit__(exc_type, exc_value, traceback)

    # Preserve metadata for introspection.
    GraphWrapper.__doc__ = _original_graph_class.__doc__
    GraphWrapper.__module__ = _original_graph_class.__module__

    torch.cuda.graph = GraphWrapper

    # Keep the backend module consistent when it exposes graph directly.
    if hasattr(torch, "musa") and hasattr(torch.musa, "graph"):
        torch.musa.graph = GraphWrapper


# Saved original torch.cuda module.
_original_torch_cuda = None


@patch_function
@requires_import("torch_musa")
def _patch_torch_cuda_module():
    """
    Patch torch.cuda to redirect to torch.musa on MUSA platform.

    This allows developers to use torch.cuda.* APIs transparently.

    Note: torch.cuda.is_available() is NOT redirected - it keeps the original
    behavior to allow downstream projects to detect the platform properly.
    """
    global _original_torch_cuda

    # torch_musa registers itself as torch.musa when imported.
    if hasattr(torch, "musa"):
        if _original_torch_cuda is None:
            _original_torch_cuda = torch.cuda

        # Preserve CUDA-only detection APIs while redirecting the rest to MUSA.
        cuda_wrapper = _CudaModuleWrapper(_original_torch_cuda, torch.musa)

        # Keep import statements and attribute access on the same wrapper.
        sys.modules["torch.cuda"] = cuda_wrapper
        torch.cuda = cuda_wrapper

        install_cuda_module_aliases(torch)
        install_cuda_memory_compat(torch, get_module(), _translate_device)

        # Accept CUDA graph keyword spelling on top of the MUSA graph class.
        _patch_graph_context_manager()
        install_cuda_public_api_shims(torch, _translate_device)


@patch_function
@requires_import("torch.distributed")
def _patch_distributed_backend():
    """
    Patch torch.distributed to automatically use MCCL when NCCL is requested.

    This allows code using 'nccl' backend to work transparently on MUSA.
    """
    global _original_init_process_group

    import torch.distributed as dist

    if _original_init_process_group is not None:
        return

    _original_init_process_group = dist.init_process_group

    @functools.wraps(_original_init_process_group)
    def patched_init_process_group(
        backend: Optional[str] = None,
        init_method: Optional[str] = None,
        timeout=None,
        world_size: int = -1,
        rank: int = -1,
        store=None,
        group_name: str = "",
        pg_options=None,
        device_id=None,
    ):
        # Translate NCCL backend requests to MCCL on MUSA.
        if is_musa_platform() and backend is not None:
            if backend.lower() == "nccl":
                backend = "mccl"

        # Translate CUDA device IDs before delegating.
        if device_id is not None:
            device_id = _translate_device(device_id)

        # Preserve the original signature while allowing version-specific args.
        kwargs = {
            "backend": backend,
            "init_method": init_method,
            "world_size": world_size,
            "rank": rank,
            "store": store,
            "group_name": group_name,
            "pg_options": pg_options,
            "device_id": device_id,
        }
        if timeout is not None:
            kwargs["timeout"] = timeout

        return _original_init_process_group(**kwargs)

    dist.init_process_group = patched_init_process_group

    # Patch new_group with the same backend and device translation.
    original_new_group = dist.new_group

    # Cache device_id support because it was added in PyTorch 2.6.
    _new_group_has_device_id = _has_param(original_new_group, "device_id")

    @functools.wraps(original_new_group)
    def patched_new_group(
        ranks=None,
        timeout=None,
        backend=None,
        pg_options=None,
        use_local_synchronization=False,
        group_desc=None,
        device_id=None,
    ):
        # Translate NCCL backend requests to MCCL on MUSA.
        if is_musa_platform() and backend is not None:
            if isinstance(backend, str) and backend.lower() == "nccl":
                backend = "mccl"

        # Preserve the original signature while allowing version-specific args.
        kwargs = {
            "ranks": ranks,
            "backend": backend,
            "pg_options": pg_options,
            "use_local_synchronization": use_local_synchronization,
            "group_desc": group_desc,
        }

        # Translate CUDA device IDs only when the installed PyTorch accepts them.
        if device_id is not None and _new_group_has_device_id:
            kwargs["device_id"] = _translate_device(device_id)

        if timeout is not None:
            kwargs["timeout"] = timeout

        return original_new_group(**kwargs)

    dist.new_group = patched_new_group


@patch_function
def _patch_tensor_is_cuda():
    """
    Patch torch.Tensor.is_cuda property to return True for MUSA tensors.

    This allows code that checks tensor.is_cuda to work on MUSA.
    We patch the is_cuda property to also return True for MUSA tensors.

    Performance: Uses try/except with direct attribute access for speed.
    Benchmarks show getattr(self, 'is_musa', False) is faster than self.device.type.
    """
    # Keep the descriptor so CUDA tensors retain their native fast path.
    original_is_cuda = torch.Tensor.is_cuda

    @property
    def patched_is_cuda(self):
        """Return True if tensor is on CUDA or MUSA device."""
        # Use direct descriptor access for actual CUDA tensors.
        result = original_is_cuda.__get__(self)
        if result:
            return True
        # Direct attribute access is faster than getattr with a default here.
        try:
            return self.is_musa
        except AttributeError:
            return False

    torch.Tensor.is_cuda = patched_is_cuda


@patch_function
@requires_import("torch_musa.core.stream")
def _patch_stream_cuda_stream():
    """
    Patch MUSA Stream class to add cuda_stream property.

    This allows code that accesses stream.cuda_stream to work on MUSA.
    The cuda_stream property returns the same value as musa_stream.
    """
    from torch_musa.core.stream import Stream as MUSAStream

    if not hasattr(MUSAStream, "cuda_stream"):

        @property
        def cuda_stream(self):
            """Return the underlying stream pointer, matching ``musa_stream``."""
            return self.musa_stream

        MUSAStream.cuda_stream = cuda_stream


@patch_function
@requires_import("torch_musa")
def _patch_autocast():
    """
    Ensure torch.amp.autocast works with 'cuda' device_type on MUSA.
    """
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        return

    original_autocast = torch.amp.autocast

    class PatchedAutocast(original_autocast):
        def __init__(self, device_type, *args, **kwargs):
            # Translate CUDA autocast contexts to MUSA contexts.
            if device_type == "cuda":
                device_type = "musa"
            super().__init__(device_type, *args, **kwargs)

    torch.amp.autocast = PatchedAutocast


@patch_function
@requires_import("torch_musa")
def _patch_profiler_activity():
    """
    Patch torch.profiler.profile to translate ProfilerActivity.CUDA to PrivateUse1 on MUSA.

    On MUSA, ProfilerActivity.CUDA doesn't work - you need to use ProfilerActivity.PrivateUse1.
    Simply assigning `ProfilerActivity.CUDA = ProfilerActivity.PrivateUse1` doesn't work because
    ProfilerActivity is an enum. Instead, we wrap the profile() function to translate
    CUDA activities to PrivateUse1 in the activities list.
    """
    if not hasattr(torch, "profiler") or not hasattr(torch.profiler, "profile"):
        return

    original_profile = torch.profiler.profile

    def _translate_activities(activities):
        """Translate ProfilerActivity.CUDA to PrivateUse1 on MUSA."""
        if activities is None:
            return None

        translated = []
        for activity in activities:
            if activity == torch.profiler.ProfilerActivity.CUDA:
                # MUSA profiler events use PrivateUse1 rather than CUDA.
                translated.append(torch.profiler.ProfilerActivity.PrivateUse1)
            else:
                translated.append(activity)
        return translated

    class ProfileWrapper:
        """Wrapper for torch.profiler.profile that translates CUDA activities."""

        def __init__(self, *args, activities=None, **kwargs):
            translated_activities = _translate_activities(activities)
            self._profiler = original_profile(*args, activities=translated_activities, **kwargs)

        def __enter__(self):
            return self._profiler.__enter__()

        def __exit__(self, *args):
            return self._profiler.__exit__(*args)

        def __getattr__(self, name):
            return getattr(self._profiler, name)

    torch.profiler.profile = ProfileWrapper


@patch_function
@requires_import("torch_musa")
def _patch_musa_warnings():
    """
    Suppress noisy MUSA-specific warnings from torch_musa.

    These warnings are informational but can clutter logs:
    - "In musa autocast, but the target dtype is not supported. Disabling autocast."
    - "Unsupported qk_head_dim: X v_head_dim: Y for FlashAttention in MUSA backend"

    We suppress them using Python's warnings.filterwarnings().
    """
    # Suppress autocast dtype warnings for unsupported MUSA dtypes.
    warnings.filterwarnings(
        "ignore",
        message=r"In musa autocast, but the target dtype is not supported.*",
        category=UserWarning,
    )

    # Suppress FlashAttention dimension warnings for unsupported MUSA head sizes.
    warnings.filterwarnings(
        "ignore",
        message=r"Unsupported qk_head_dim:.*for FlashAttention in MUSA backend.*",
        category=UserWarning,
    )


@patch_function
@requires_import("torch_musa")
def _patch_library_impl():
    """
    Patch torch.library.Library.impl() to translate CUDA dispatch keys to PrivateUse1.

    On MUSA, tensors dispatch to PrivateUse1, not CUDA. When code registers custom ops
    with CUDA backends, they won't work with MUSA tensors. This patch automatically
    translates CUDA dispatch keys to PrivateUse1 equivalents:

        CUDA -> PrivateUse1
        AutogradCUDA -> AutogradPrivateUse1
        AutocastCUDA -> AutocastPrivateUse1
        SparseCUDA -> SparsePrivateUse1
        SparseCsrCUDA -> SparseCsrPrivateUse1
        QuantizedCUDA -> QuantizedPrivateUse1
        NestedTensorCUDA -> NestedTensorPrivateUse1

    This patch preserves the full original signature including the with_keyset parameter.

    Example of code that needs this patch:
        my_lib.impl(op_name, op_func, "CUDA")  # Works on MUSA.
        my_lib.impl(op_name, op_func, "Autograd", with_keyset=True)  # Works on MUSA.
        my_lib.impl(op_name, op_func, "Autograd", with_keyset=True, allow_override=True)
    """
    if not hasattr(torch, "library") or not hasattr(torch.library, "Library"):
        return

    original_impl = torch.library.Library.impl

    # CUDA dispatch keys that should register against PrivateUse1 on MUSA.
    cuda_dispatch_key_map = {
        "CUDA": "PrivateUse1",
        "AutogradCUDA": "AutogradPrivateUse1",
        "AutocastCUDA": "AutocastPrivateUse1",
        "SparseCUDA": "SparsePrivateUse1",
        "SparseCsrCUDA": "SparseCsrPrivateUse1",
        "QuantizedCUDA": "QuantizedPrivateUse1",
        "NestedTensorCUDA": "NestedTensorPrivateUse1",
    }

    def patched_impl(self, *args, **kwargs):
        # Translate CUDA dispatch keys before registering custom operators.
        sig = inspect.signature(original_impl)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        if bound.arguments.get("dispatch_key") in cuda_dispatch_key_map:
            bound.arguments["dispatch_key"] = cuda_dispatch_key_map[bound.arguments["dispatch_key"]]

        return original_impl(*bound.args, **bound.kwargs)

    torch.library.Library.impl = patched_impl


@patch_function
@requires_import("torch_musa")
def _patch_torch_c_exports():
    """
    Patch torch._C to include MUSA-specific functions from torch_musa._MUSAC.

    Some functions like _storage_Use_Count exist in torch_musa._MUSAC but not
    in torch._C. Code that tries to do:
        from torch._C import _storage_Use_Count
    will fail without this patch.

    This patch adds missing functions from torch_musa._MUSAC to torch._C.
    """
    import torch_musa

    if not hasattr(torch_musa, "_MUSAC"):
        return

    musac = torch_musa._MUSAC

    # Common downstream imports that torch_musa exposes under _MUSAC only.
    _MUSAC_EXPORTS = [
        "_storage_Use_Count",
    ]

    for name in _MUSAC_EXPORTS:
        if hasattr(musac, name) and not hasattr(torch._C, name):
            setattr(torch._C, name, getattr(musac, name))


@patch_function
@requires_import("torch_musa")
def _patch_backends_cuda():
    """
    Patch torch.backends.cuda to work on MUSA platform.

    This patches:
    - is_built() to return True when MUSA is available (since we're using
      torch.cuda APIs that are redirected to MUSA)
    - torch.backends.cuda.matmul attribute access to MUSA matmul semantics
    """
    if not hasattr(torch, "backends") or not hasattr(torch.backends, "cuda"):
        return

    # Let CUDA build checks pass when torchada is redirecting CUDA APIs to MUSA.
    original_is_built = torch.backends.cuda.is_built

    # Cache the result because platform state does not change at runtime.
    _is_built_cache = {}

    def patched_is_built():
        if "result" not in _is_built_cache:
            # Treat MUSA as CUDA-built even in build-only environments.
            if is_musa_platform():
                _is_built_cache["result"] = True
            else:
                _is_built_cache["result"] = original_is_built()
        return _is_built_cache["result"]

    torch.backends.cuda.is_built = patched_is_built

    if not (
        is_musa_platform()
        and hasattr(torch.backends, "musa")
        and hasattr(torch.backends.musa, "matmul")
        and hasattr(torch.backends.cuda, "matmul")
    ):
        return

    cuda_matmul = torch.backends.cuda.matmul
    musa_matmul = torch.backends.musa.matmul
    matmul_class = cuda_matmul.__class__
    original_getattr = matmul_class.__getattr__
    original_setattr = matmul_class.__setattr__

    try:
        _ = cuda_matmul.fp32_precision
        has_native_fp32_precision = True
    except AttributeError:
        has_native_fp32_precision = False

    def patched_getattr(self, name):
        if name == "fp32_precision" and not has_native_fp32_precision:
            return torch.get_float32_matmul_precision()
        try:
            return getattr(musa_matmul, name)
        except (AttributeError, AssertionError):
            return original_getattr(self, name)

    def patched_setattr(self, name, value):
        if name == "fp32_precision" and not has_native_fp32_precision:
            return torch.set_float32_matmul_precision(value)
        try:
            return setattr(musa_matmul, name, value)
        except (AttributeError, AssertionError):
            return original_setattr(self, name, value)

    matmul_class.__getattr__ = patched_getattr
    matmul_class.__setattr__ = patched_setattr


@patch_function
@requires_import("torchada.utils.cpp_extension", "torch.utils.cpp_extension")
def _patch_cpp_extension():
    """
    Patch torch.utils.cpp_extension to use torchada's MUSA-compatible versions.

    This allows developers to use standard imports like:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension

    And have them work transparently on MUSA platform.

    Also patches include_paths and library_paths to support both:
    - PyTorch < 2.6: include_paths(cuda=True)
    - PyTorch 2.6+: include_paths(device_type="cuda")
    """
    import torch.utils.cpp_extension as torch_cpp_ext

    from .utils import cpp_extension as torchada_cpp_ext

    # Patch the key classes and constants.
    torch_cpp_ext.CUDAExtension = torchada_cpp_ext.CUDAExtension
    torch_cpp_ext.BuildExtension = torchada_cpp_ext.BuildExtension
    torch_cpp_ext.CUDA_HOME = torchada_cpp_ext.CUDA_HOME

    # Delegate include/library path handling to the torchada compatibility layer.
    torch_cpp_ext.include_paths = torchada_cpp_ext.include_paths
    torch_cpp_ext.library_paths = torchada_cpp_ext.library_paths

    # Keep future imports on the patched module object.
    sys.modules["torch.utils.cpp_extension"] = torch_cpp_ext


@patch_function
@requires_import("torch._inductor.autotune_process")
def _patch_autotune_process():
    """
    Patch torch._inductor.autotune_process to use MUSA_VISIBLE_DEVICES on MUSA platform.

    The autotune subprocess uses CUDA_VISIBLE_DEVICES to control GPU visibility.
    On MUSA platform, we need to use MUSA_VISIBLE_DEVICES instead.

    Reference: https://github.com/pytorch/pytorch/blob/main/torch/_inductor/autotune_process.py#L61
    """
    import torch._inductor.autotune_process as autotune_process

    # Use the MUSA visibility environment variable in autotune subprocesses.
    if hasattr(autotune_process, "CUDA_VISIBLE_DEVICES"):
        autotune_process.CUDA_VISIBLE_DEVICES = "MUSA_VISIBLE_DEVICES"


@patch_function
@requires_import("torch_musa", "torch.nn.attention.flex_attention")
def _patch_validate_device():
    """
    Patch torch.nn.attention.flex_attention._validate_device to accept MUSA devices.

    The original upstream validator only allows certain device types (cuda, cpu, etc.)
    and rejects MUSA tensors. Instead of replacing the entire function (which varies
    across PyTorch versions), this wraps the original and short-circuits for MUSA
    devices, delegating all other cases to the upstream implementation.
    """
    import torch.nn.attention.flex_attention

    _orig_validate_device = None
    if hasattr(torch.nn.attention.flex_attention, "_validate_device"):
        _orig_validate_device = torch.nn.attention.flex_attention._validate_device

    def _validate_device(query, key, value):
        if query.device.type == "musa" or _orig_validate_device is None:
            return
        return _orig_validate_device(query, key, value)

    torch.nn.attention.flex_attention._validate_device = _validate_device


@patch_function
@requires_import("flash_attn_interface")
def _patch_flash_attn():
    """
    Redirect sgl_kernel.flash_attn imports to the MUSA flash_attn_interface package.

    On CUDA (NVIDIA), sgl_kernel provides its own flash_attn submodule:
        from sgl_kernel.flash_attn import flash_attn_varlen_func

    On MUSA, the mate package provides an equivalent flash_attn_interface package.
    This patch registers flash_attn_interface as sgl_kernel.flash_attn in sys.modules
    so that code using sgl_kernel.flash_attn works transparently on MUSA.

    If sgl_kernel is not installed, a stub module is created so that
    sgl_kernel.flash_attn imports still resolve correctly.
    """
    import flash_attn_interface

    # Prefer the real package; create a stub only when sgl_kernel is absent.
    if "sgl_kernel" not in sys.modules:
        try:
            import sgl_kernel  # noqa: F401
        except ImportError:
            sgl_kernel_stub = ModuleType("sgl_kernel")
            sgl_kernel_stub.__path__ = []  # Mark the stub as a package.
            sgl_kernel_stub.__package__ = "sgl_kernel"
            sys.modules["sgl_kernel"] = sgl_kernel_stub

    # Register flash_attn_interface as the sgl_kernel.flash_attn submodule.
    sgl_kernel = sys.modules["sgl_kernel"]
    sgl_kernel.flash_attn = flash_attn_interface
    sys.modules["sgl_kernel.flash_attn"] = flash_attn_interface


@patch_function
@requires_import("torch_musa", "torch.accelerator")
def _patch_torch_accelerator():
    """
    Wrap torch.accelerator with an _AcceleratorModuleWrapper on MUSA platform.

    This provides:

    1. A fix for torch.accelerator.synchronize() - the MUSA backend does not
       implement the all-streams synchronization hook, so the default
       implementation raises. The wrapper installs a patched synchronize that
       delegates to torch.musa.synchronize().

    2. Overrides for memory APIs that exist on torch.accelerator (PyTorch 2.9+)
       but are broken on MUSA because they route through torch._C._accelerator_*
       C++ functions that don't dispatch to the MUSA allocator. These are
       redirected to torch.musa implementations (see _AcceleratorModuleWrapper
       ._MUSA_OVERRIDES).

    3. Forward compatibility for APIs that PyTorch is expected to add to
       torch.accelerator in future releases but are not yet present (Stream,
       Event, manual_seed, get_device_name, ...). Any attribute missing from
       the current torch.accelerator module is looked up on torch.musa instead.

    4. device_index(idx) and stream(s) context managers, which are not yet
       present on torch.accelerator in torch 2.7.
    """
    patch_torch_accelerator(torch)


@patch_function
def _patch_ctypes_cdll():
    """
    Patch ctypes.CDLL to automatically translate CUDA/NCCL function names to MUSA/MCCL.

    This allows code that uses ctypes to directly call CUDA runtime or NCCL functions
    (like sglang's cuda_wrapper.py and pynccl.py) to work transparently on MUSA
    without requiring code changes.

    When loading MUSA libraries (libmusart.so, libmccl.so, etc.), the returned CDLL
    wrapper will automatically translate function name lookups:
        - cudaXxx -> musaXxx (for libmusart)
        - ncclXxx -> mcclXxx (for libmccl)
        - cublasXxx -> mublasXxx (for libmublas)
        - curandXxx -> murandXxx (for libmurand)

    Example (in sglang):
        lib = ctypes.CDLL("libmusart.so")
        # This lookup resolves to musaIpcOpenMemHandle:
        func = lib.cudaIpcOpenMemHandle
    """
    patch_ctypes_cdll()


def apply_patches():
    """
    Apply all necessary patches for CUDA to MUSA translation.

    After calling this, developers can use torch.cuda.* APIs normally
    and they will be transparently redirected to torch.musa on MUSA platform.

    This includes:
    - torch.device("cuda") -> torch.device("musa")
    - torch.cuda.* API -> torch.musa.*
    - torch.cuda.nvtx -> no-op stub
    - torch.cuda.Stream.cuda_stream -> musa_stream
    - torch.Tensor.cuda() -> torch.Tensor.musa()
    - torch.Tensor.is_cuda -> True for MUSA tensors
    - torch.nn.Module.cuda() -> torch.nn.Module.musa()
    - Device string translation ("cuda" -> "musa")
    - torch.distributed with 'nccl' backend -> 'mccl'
    - torch.cuda.CUDAGraph -> torch.musa.MUSAGraph
    - torch.cuda.nccl -> torch.musa.mccl
    - torch.amp.autocast(device_type='cuda') -> 'musa'
    - torch.utils.cpp_extension (CUDAExtension, BuildExtension) -> MUSA versions
    - torch._inductor.autotune_process.CUDA_VISIBLE_DEVICES -> MUSA_VISIBLE_DEVICES
    - torch.accelerator.synchronize() -> torch.musa.synchronize()
    - torch.accelerator context managers (device_index, stream) for forward compatibility
    - ctypes.CDLL function name translation for MUSA libraries:
        - cudaXxx -> musaXxx (for libmusart)
        - ncclXxx -> mcclXxx (for libmccl)
        - cublasXxx -> mublasXxx, curandXxx -> murandXxx (for libmublas, libmurand)

    This function should be called once at import time.

    Patch functions are registered via the @patch_function decorator and
    can be guarded with @requires_import for optional module dependencies.
    """
    global _patched

    if _patched:
        return

    if not is_musa_platform():
        _patched = True
        return

    # Import torch_musa so torch.musa is registered before applying patches.
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        _patched = True
        return

    # Apply registered patch functions in definition order.
    for patch_fn in _patch_registry:
        patch_fn()

    if hasattr(torch.Tensor, "to"):
        torch.Tensor.to = _wrap_to_method(torch.Tensor.to)

    if hasattr(torch.Tensor, "cuda"):
        torch.Tensor.cuda = _wrap_tensor_cuda(torch.Tensor.cuda)

    if hasattr(torch.nn.Module, "cuda"):
        torch.nn.Module.cuda = _wrap_module_cuda(torch.nn.Module.cuda)

    # Wrap tensor factories and keep originals for PyTorch device-context dispatch.
    original_fns = []
    for fn_name in _FACTORY_FUNCTIONS:
        if hasattr(torch, fn_name):
            original_fn = getattr(torch, fn_name)
            original_fns.append(original_fn)
            setattr(torch, fn_name, _wrap_factory_function(original_fn))

    # PyTorch's __torch_function__ path receives original C functions, not our wrappers.
    try:
        from torch.utils._device import _device_constructors

        constructors = _device_constructors()

        for orig_fn in original_fns:
            constructors.add(orig_fn)

    except (ImportError, AttributeError):
        pass  # Older PyTorch versions may not expose this helper.

    _patched = True


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _patched


# Additional exports for advanced usage.
def get_original_init_process_group():
    """Get the original torch.distributed.init_process_group function."""
    return _original_init_process_group
