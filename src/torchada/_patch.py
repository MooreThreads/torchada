"""
Automatic patching module for torchada.

This module patches PyTorch to automatically translate 'cuda' device strings
to 'musa' when running on Moore Threads hardware.

Usage:
    import torchada  # This applies all patches automatically
    import torch

    # Then use torch.cuda as normal - it will work on MUSA
    torch.cuda.is_available()
    x = torch.randn(3, 3).cuda()
    from torch.cuda.amp import autocast, GradScaler

    # Distributed training with NCCL also works transparently
    import torch.distributed as dist
    dist.init_process_group(backend="nccl")  # Uses MCCL on MUSA

    # CUDA Graphs work transparently
    g = torch.cuda.CUDAGraph()  # Uses MUSAGraph on MUSA
"""

import functools
import sys
import warnings
from types import ModuleType
from typing import Any, Callable, List, Optional

import torch

from ._platform import is_musa_platform

_patched = False
_original_init_process_group = None

# Registry for patch functions
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
            # patching logic
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
            # This only runs if torch_musa is importable
            import torch_musa
            # ... patching logic

        @patch_function
        @requires_import('torch._inductor.autotune_process')
        def _patch_autotune():
            import torch._inductor.autotune_process as ap
            # ... patching logic

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


# Cache for translated device strings - avoids repeated string operations
_device_str_cache = {}

# Cache for is_musa_platform result - computed once on first call
_is_musa_platform_cached = None


def _translate_device(device: Any) -> Any:
    """
    Translate 'cuda' device references to 'musa' on MUSA platform.

    Args:
        device: Device specification (string, torch.device, int, or None)

    Returns:
        Translated device specification

    Performance: Platform check and string translations are cached.
    """
    global _is_musa_platform_cached

    # Cache the platform check result (computed once)
    if _is_musa_platform_cached is None:
        _is_musa_platform_cached = is_musa_platform()

    if not _is_musa_platform_cached:
        return device

    if device is None:
        return device

    if isinstance(device, str):
        # Check cache first for common strings
        if device in _device_str_cache:
            return _device_str_cache[device]

        # Handle 'cuda', 'cuda:0', 'cuda:1', etc.
        if device == "cuda" or device.startswith("cuda:"):
            result = device.replace("cuda", "musa")
            _device_str_cache[device] = result
            return result
        # Cache non-cuda strings too to avoid repeated startswith checks
        _device_str_cache[device] = device
        return device

    if isinstance(device, torch.device):
        if device.type == "cuda":
            return torch.device("musa", device.index)
        return device

    # For integer device IDs, keep as-is (context determines device type)
    return device


def _wrap_to_method(original_to: Callable) -> Callable:
    """Wrap tensor.to() to translate device strings."""

    @functools.wraps(original_to)
    def wrapped_to(self, *args, **kwargs):
        # Translate device in positional args
        if args and len(args) >= 1:
            first_arg = args[0]
            # Check if first arg looks like a device
            if isinstance(first_arg, (str, torch.device)):
                args = (_translate_device(first_arg),) + args[1:]
            elif isinstance(first_arg, torch.dtype):
                # .to(dtype) case, check for device in kwargs or second arg
                if len(args) >= 2:
                    args = (first_arg, _translate_device(args[1])) + args[2:]

        # Translate device in keyword args
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])

        return original_to(self, *args, **kwargs)

    return wrapped_to


def _wrap_tensor_cuda(original_cuda: Callable) -> Callable:
    """Wrap tensor.cuda() to use musa on MUSA platform."""
    # Cache platform check at wrapper creation time
    _is_musa = is_musa_platform()

    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None, non_blocking=False):
        if _is_musa:
            # Use .musa() instead
            if hasattr(self, "musa"):
                return self.musa(device=device, non_blocking=non_blocking)
            else:
                # Fallback to .to()
                target_device = f"musa:{device}" if device is not None else "musa"
                return self.to(target_device, non_blocking=non_blocking)
        return original_cuda(self, device=device, non_blocking=non_blocking)

    return wrapped_cuda


def _wrap_module_cuda(original_cuda: Callable) -> Callable:
    """Wrap nn.Module.cuda() to use musa on MUSA platform."""
    # Cache platform check at wrapper creation time
    _is_musa = is_musa_platform()

    @functools.wraps(original_cuda)
    def wrapped_cuda(self, device=None):
        if _is_musa:
            if hasattr(self, "musa"):
                return self.musa(device=device)
            else:
                target_device = f"musa:{device}" if device is not None else "musa"
                return self.to(target_device)
        return original_cuda(self, device=device)

    return wrapped_cuda


_original_torch_device = None


class _DeviceFactoryMeta(type):
    """Metaclass to make isinstance(x, torch.device) work with our factory."""

    def __instancecheck__(cls, instance):
        if _original_torch_device is not None:
            return isinstance(instance, _original_torch_device)
        return False

    def __subclasscheck__(cls, subclass):
        if _original_torch_device is not None:
            return issubclass(subclass, _original_torch_device)
        return False


class DeviceFactoryWrapper(metaclass=_DeviceFactoryMeta):
    """
    A wrapper class that acts as torch.device but translates cuda to musa.

    Uses a metaclass to properly handle isinstance() checks.

    Supports all calling conventions of torch.device:
        torch.device("cuda:0")
        torch.device("cuda", 0)
        torch.device(type="cuda", index=0)
        torch.device(device="cuda:0")
    """

    _original = None

    def __new__(cls, device=None, index=None, *, type=None):
        original = cls._original
        if original is None:
            raise RuntimeError("DeviceFactoryWrapper not initialized")

        # Handle 'type' keyword argument (alias for device in original torch.device)
        if type is not None:
            device = type

        # Handle the case where device is already a torch.device
        if isinstance(device, original):
            if device.type == "cuda":
                index = device.index if index is None else index
                device = "musa"
            else:
                return device

        # Handle string device
        if isinstance(device, str):
            device = _translate_device(device)

        # Create the actual device
        if index is not None:
            return original(device, index)
        elif device is not None:
            return original(device)
        else:
            return original()


@patch_function
def _patch_torch_device():
    """
    Patch torch.device to translate 'cuda' to 'musa' on MUSA platform.

    This ensures that torch.device("cuda:0") creates a musa device when on MUSA.
    """
    global _original_torch_device

    if _original_torch_device is not None:
        return  # Already patched

    _original_torch_device = torch.device
    DeviceFactoryWrapper._original = _original_torch_device

    # Replace torch.device with our wrapper
    torch.device = DeviceFactoryWrapper


# Store original torch.Generator for patching
_original_torch_generator = None
# Store the underlying C Generator class for isinstance checks
_original_c_generator = None


@patch_function
def _patch_torch_generator():
    """
    Patch torch.Generator to translate 'cuda' device to 'musa' on MUSA platform.

    This ensures that torch.Generator(device="cuda") creates a MUSA generator
    instead of failing with "Cannot get CUDA generator without ATen_cuda library".

    Uses a metaclass to properly implement __instancecheck__ so that
    isinstance(gen, torch.Generator) works correctly.
    """
    global _original_torch_generator, _original_c_generator

    if _original_torch_generator is not None:
        return  # Already patched

    _original_torch_generator = torch.Generator
    # Get the underlying C Generator class for isinstance checks
    # torch_musa may have already wrapped torch.Generator, but instances are still
    # of type torch._C.Generator
    _original_c_generator = torch._C.Generator

    class GeneratorMeta(type):
        """Metaclass that properly implements __instancecheck__ for isinstance() to work."""

        def __instancecheck__(cls, instance):
            # Check if instance is a torch._C.Generator (the actual C class)
            return isinstance(instance, _original_c_generator)

        def __subclasscheck__(cls, subclass):
            # Check if subclass is or inherits from torch._C.Generator
            if subclass is _original_c_generator:
                return True
            return super().__subclasscheck__(subclass)

    class GeneratorWrapper(metaclass=GeneratorMeta):
        """Wrapper for torch.Generator that translates cuda -> musa."""

        def __new__(cls, device=None):
            # Translate device if needed
            if device is not None:
                device = _translate_device(device)
            return _original_torch_generator(device=device)

    # Copy over class attributes
    GeneratorWrapper.__doc__ = _original_torch_generator.__doc__
    GeneratorWrapper.__module__ = _original_torch_generator.__module__

    torch.Generator = GeneratorWrapper


# Store original graph class for patching
_original_graph_class = None


@patch_function
def _patch_graph_context_manager():
    """
    Patch torch.cuda.graph context manager to accept cuda_graph= keyword argument.

    MUSA's graph class uses musa_graph= as the first parameter, but CUDA code
    uses cuda_graph=. This wrapper translates cuda_graph= to musa_graph= so that
    existing CUDA code works transparently on MUSA.
    """
    global _original_graph_class

    if _original_graph_class is not None:
        return  # Already patched

    # Get the graph class from torch.cuda (which is torch.musa after patching)
    if not hasattr(torch.cuda, "graph"):
        return

    _original_graph_class = torch.cuda.graph

    class GraphWrapper:
        """Wrapper for torch.cuda.graph that accepts cuda_graph= keyword argument."""

        # Preserve class attributes
        default_capture_stream = None

        def __init__(
            self,
            cuda_graph=None,
            pool=None,
            stream=None,
            capture_error_mode: str = "global",
            *,
            musa_graph=None,  # Also accept musa_graph for compatibility
        ):
            # Allow either cuda_graph= or musa_graph= or positional argument
            graph_obj = cuda_graph if cuda_graph is not None else musa_graph
            if graph_obj is None:
                raise TypeError("graph() missing required argument: 'cuda_graph'")

            # Create the original graph instance
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

    # Copy over class attributes and docstring
    GraphWrapper.__doc__ = _original_graph_class.__doc__
    GraphWrapper.__module__ = _original_graph_class.__module__

    # Replace torch.cuda.graph with our wrapper
    torch.cuda.graph = GraphWrapper

    # Also update torch.musa.graph if it exists
    if hasattr(torch, "musa") and hasattr(torch.musa, "graph"):
        torch.musa.graph = GraphWrapper


def _wrap_factory_function(original_fn: Callable) -> Callable:
    """Wrap tensor factory functions (empty, zeros, ones, etc.) to translate device."""

    @functools.wraps(original_fn)
    def wrapped_fn(*args, **kwargs):
        if "device" in kwargs:
            kwargs["device"] = _translate_device(kwargs["device"])
        return original_fn(*args, **kwargs)

    return wrapped_fn


# List of torch factory functions that accept a device argument
_FACTORY_FUNCTIONS = [
    # Basic tensor creation
    "tensor",
    "as_tensor",
    "asarray",
    # Uninitialized/initialized tensors
    "empty",
    "zeros",
    "ones",
    "full",
    # Random tensors
    "rand",
    "randn",
    "randint",
    "randperm",
    "normal",
    # Sequences
    "arange",
    "range",
    "linspace",
    "logspace",
    # Special tensors
    "eye",
    "empty_strided",
    "empty_permuted",
    # From file
    "from_file",
    # Like variants
    "empty_like",
    "zeros_like",
    "ones_like",
    "full_like",
    "rand_like",
    "randn_like",
    "randint_like",
    # Sparse tensors
    "sparse_coo_tensor",
    "sparse_csr_tensor",
    "sparse_csc_tensor",
    "sparse_bsr_tensor",
    "sparse_bsc_tensor",
    "sparse_compressed_tensor",
    # Index tensors
    "tril_indices",
    "triu_indices",
    # Window functions
    "bartlett_window",
    "blackman_window",
    "hamming_window",
    "hann_window",
    "kaiser_window",
]


class _CudartWrapper:
    """
    Wrapper for CUDA runtime that translates calls to MUSA runtime.

    This allows code like `torch.cuda.cudart().cudaHostRegister(...)` to work
    on MUSA by translating to `torch_musa.musart().musaHostRegister(...)`.

    Performance optimization: Resolved attributes are cached in __dict__ to avoid
    repeated __getattr__ calls.
    """

    # Mapping from CUDA runtime function names to MUSA equivalents
    _CUDA_TO_MUSA = {
        "cudaHostRegister": "musaHostRegister",
        "cudaHostUnregister": "musaHostUnregister",
        "cudaMemGetInfo": "musaMemGetInfo",
        "cudaGetErrorString": "musaGetErrorString",
        "cudaStreamCreate": "musaStreamCreate",
        "cudaStreamDestroy": "musaStreamDestroy",
    }

    def __init__(self, musart_module):
        self._musart = musart_module

    def __getattr__(self, name):
        # Translate CUDA runtime function names to MUSA equivalents
        if name in self._CUDA_TO_MUSA:
            musa_name = self._CUDA_TO_MUSA[name]
            value = getattr(self._musart, musa_name)
            # Cache in __dict__ for faster subsequent access
            object.__setattr__(self, name, value)
            return value

        # Try direct access (for any functions with same name)
        if hasattr(self._musart, name):
            value = getattr(self._musart, name)
            # Cache in __dict__ for faster subsequent access
            object.__setattr__(self, name, value)
            return value

        raise AttributeError(f"CUDA runtime has no attribute '{name}'")


class _CudaModuleWrapper(ModuleType):
    """
    A wrapper module that redirects torch.cuda to torch.musa,
    but keeps certain attributes (like is_available) pointing to the original.

    This allows downstream projects to detect MUSA platform using:
        torch.cuda.is_available()  # Returns False on MUSA (original behavior)
    While still using torch.cuda.* APIs that redirect to torch.musa.

    Performance optimization: Resolved attributes are cached in __dict__ to avoid
    repeated __getattr__ calls. This reduces overhead from ~800ns to ~50ns for
    cached attributes.
    """

    # Attributes that should NOT be redirected to torch.musa
    _NO_REDIRECT = {"is_available"}

    # Special attribute mappings for attributes not at top level of torch_musa
    # Maps attribute name -> dot-separated path within torch_musa
    _SPECIAL_ATTRS = {
        "StreamContext": "core.stream.StreamContext",
    }

    # Attribute name remappings (CUDA name -> MUSA name)
    # For CUDA-specific APIs that have different names in MUSA
    _REMAP_ATTRS = {
        "_device_count_nvml": "device_count",  # NVML is NVIDIA-specific
    }

    # Attributes that should NOT be cached (functions that may return different values)
    # Most functions are safe to cache since they're module-level functions
    _NO_CACHE = {
        # These are typically not called in hot paths anyway
    }

    def __init__(self, original_cuda, musa_module):
        super().__init__("torch.cuda")
        self._original_cuda = original_cuda
        self._musa_module = musa_module
        self._cudart_wrapper = None

    def cudart(self):
        """
        Return a CUDA runtime wrapper that translates to MUSA runtime.

        This allows code like `torch.cuda.cudart().cudaHostRegister(...)` to work
        on MUSA by translating to the equivalent MUSA runtime calls.
        """
        if self._cudart_wrapper is None:
            if hasattr(self._musa_module, "musart"):
                musart_module = self._musa_module.musart()
                self._cudart_wrapper = _CudartWrapper(musart_module)
            else:
                # Fallback to original if musart not available
                return self._original_cuda.cudart()
        return self._cudart_wrapper

    def __getattr__(self, name):
        # Keep original is_available behavior
        if name in self._NO_REDIRECT:
            value = getattr(self._original_cuda, name)
            # Cache in __dict__ for faster subsequent access
            if name not in self._NO_CACHE:
                object.__setattr__(self, name, value)
            return value

        # Handle special attributes that need nested lookup
        if name in self._SPECIAL_ATTRS:
            obj = self._musa_module
            for part in self._SPECIAL_ATTRS[name].split("."):
                obj = getattr(obj, part)
            # Cache the resolved value
            if name not in self._NO_CACHE:
                object.__setattr__(self, name, obj)
            return obj

        # Handle attribute name remapping (CUDA-specific names -> MUSA equivalents)
        if name in self._REMAP_ATTRS:
            value = getattr(self._musa_module, self._REMAP_ATTRS[name])
            # Cache the resolved value
            if name not in self._NO_CACHE:
                object.__setattr__(self, name, value)
            return value

        # Redirect everything else to torch.musa
        value = getattr(self._musa_module, name)
        # Cache the resolved value for faster subsequent access
        # This is safe because module attributes don't change at runtime
        if name not in self._NO_CACHE:
            object.__setattr__(self, name, value)
        return value

    def __dir__(self):
        # Combine attributes from both modules
        attrs = set(dir(self._musa_module))
        attrs.update(self._NO_REDIRECT)
        attrs.add("cudart")
        return list(attrs)


# Store original torch.cuda module before patching
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

    # torch_musa registers itself as torch.musa when imported
    # Now patch torch.cuda to point to torch.musa (which is torch_musa)
    if hasattr(torch, "musa"):
        # Save original torch.cuda before patching
        if _original_torch_cuda is None:
            _original_torch_cuda = torch.cuda

        # Create wrapper module that redirects most things to torch.musa
        # but keeps is_available pointing to the original
        cuda_wrapper = _CudaModuleWrapper(_original_torch_cuda, torch.musa)

        # Replace torch.cuda with our wrapper in sys.modules
        # This makes 'from torch.cuda import ...' work
        sys.modules["torch.cuda"] = cuda_wrapper

        # Also patch torch.cuda attribute directly
        torch.cuda = cuda_wrapper

        # Patch torch.cuda.amp
        if hasattr(torch.musa, "amp"):
            sys.modules["torch.cuda.amp"] = torch.musa.amp

        # Patch torch.cuda.graphs - MUSAGraph should be accessible as CUDAGraph
        if hasattr(torch.musa, "graphs"):
            sys.modules["torch.cuda.graphs"] = torch.musa.graphs

        # Add CUDAGraph alias pointing to MUSAGraph
        if hasattr(torch.musa, "MUSAGraph") and not hasattr(torch.musa, "CUDAGraph"):
            torch.musa.CUDAGraph = torch.musa.MUSAGraph

        # Patch torch.cuda.graph context manager to accept cuda_graph= keyword
        # MUSA's graph class uses musa_graph= but CUDA code uses cuda_graph=
        _patch_graph_context_manager()

        # Patch torch.cuda.nccl -> torch.musa.mccl
        if hasattr(torch.musa, "mccl"):
            sys.modules["torch.cuda.nccl"] = torch.musa.mccl

        # Patch torch.cuda.profiler
        if hasattr(torch.musa, "profiler"):
            sys.modules["torch.cuda.profiler"] = torch.musa.profiler

        # Patch torch.cuda.nvtx - use our stub since MUSA doesn't have nvtx
        try:
            from .cuda import nvtx as nvtx_stub

            sys.modules["torch.cuda.nvtx"] = nvtx_stub
            torch.musa.nvtx = nvtx_stub
        except ImportError:
            pass

        # Patch missing _lazy_call from torch_musa.core._lazy_init
        # torch_musa only maps _lazy_init but not _lazy_call
        # This is needed for code that does: from torch.cuda import _lazy_call
        # We add it to torch.musa so _CudaModuleWrapper can redirect it
        try:
            from torch_musa.core._lazy_init import _lazy_call

            # Only add if not already present (forward compatible with torch_musa fix)
            if not hasattr(torch.musa, "_lazy_call"):
                torch.musa._lazy_call = _lazy_call
        except ImportError:
            pass

        # Add _is_compiled to torch_musa if not present
        # This is needed for code that checks torch.cuda._is_compiled()
        # (e.g., vLLM's CUDA kernel availability checks)
        if not hasattr(torch.musa, "_is_compiled"):
            torch.musa._is_compiled = lambda: True


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
        # Already patched
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
        # Translate 'nccl' to 'mccl' on MUSA platform
        if is_musa_platform() and backend is not None:
            if backend.lower() == "nccl":
                backend = "mccl"

        # Translate device_id if it's a cuda device
        if device_id is not None:
            device_id = _translate_device(device_id)

        # Build kwargs for the original function
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

    # Also patch new_group to translate 'nccl' to 'mccl'
    original_new_group = dist.new_group

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
        # Translate 'nccl' to 'mccl' on MUSA platform
        if is_musa_platform() and backend is not None:
            if isinstance(backend, str) and backend.lower() == "nccl":
                backend = "mccl"

        # Translate device_id if it's a cuda device
        if device_id is not None:
            device_id = _translate_device(device_id)

        # Build kwargs for the original function
        kwargs = {
            "ranks": ranks,
            "backend": backend,
            "pg_options": pg_options,
            "use_local_synchronization": use_local_synchronization,
            "group_desc": group_desc,
            "device_id": device_id,
        }
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
    # Store the original is_cuda property (it's a getset_descriptor)
    original_is_cuda = torch.Tensor.is_cuda

    @property
    def patched_is_cuda(self):
        """Return True if tensor is on CUDA or MUSA device."""
        # Check original is_cuda first (fast path for actual CUDA tensors)
        # Use direct property access - original_is_cuda is a getset_descriptor
        result = original_is_cuda.__get__(self)
        if result:
            return True
        # Check if tensor is on MUSA device
        # Use try/except with direct attribute access - faster than getattr with default
        try:
            return self.is_musa
        except AttributeError:
            return False

    # Replace is_cuda with our patched version
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

    # Add cuda_stream property that returns musa_stream
    if not hasattr(MUSAStream, "cuda_stream"):

        @property
        def cuda_stream(self):
            """Return the underlying stream pointer (same as musa_stream)."""
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
            # Translate 'cuda' to 'musa'
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
                # On MUSA, use PrivateUse1 instead of CUDA
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
    # Suppress autocast dtype warning from torch/amp/autocast_mode.py
    # This happens when autocast is used with unsupported dtypes on MUSA
    warnings.filterwarnings(
        "ignore",
        message=r"In musa autocast, but the target dtype is not supported.*",
        category=UserWarning,
    )

    # Suppress FlashAttention unsupported dimension warning from torch_musa
    # This happens when SDP attention is used with unsupported head dimensions
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
        my_lib.impl(op_name, op_func, "CUDA")  # Now works on MUSA!
        my_lib.impl(op_name, op_func, "Autograd", with_keyset=True)  # Also works!
    """
    if not hasattr(torch, "library") or not hasattr(torch.library, "Library"):
        return

    original_impl = torch.library.Library.impl

    # Mapping of CUDA dispatch keys to PrivateUse1 equivalents
    cuda_dispatch_key_map = {
        "CUDA": "PrivateUse1",
        "AutogradCUDA": "AutogradPrivateUse1",
        "AutocastCUDA": "AutocastPrivateUse1",
        "SparseCUDA": "SparsePrivateUse1",
        "SparseCsrCUDA": "SparseCsrPrivateUse1",
        "QuantizedCUDA": "QuantizedPrivateUse1",
        "NestedTensorCUDA": "NestedTensorPrivateUse1",
    }

    def patched_impl(self, op_name, fn, dispatch_key="", *, with_keyset=False):
        # Translate CUDA dispatch keys to PrivateUse1 equivalents for MUSA compatibility
        if dispatch_key in cuda_dispatch_key_map:
            dispatch_key = cuda_dispatch_key_map[dispatch_key]
        return original_impl(self, op_name, fn, dispatch_key, with_keyset=with_keyset)

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

    # List of functions/classes to copy from _MUSAC to torch._C
    # These are commonly imported by downstream code
    _MUSAC_EXPORTS = [
        "_storage_Use_Count",
        # Add more as needed
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
    - torch.backends.cuda.matmul.fp32_precision to use torch.get/set_float32_matmul_precision
      (this attribute is missing in some torch_musa versions)

    Note: Other torch.backends.cuda.matmul properties (allow_tf32, etc.) work
    as-is because they are settings that apply to the internal PyTorch
    operations regardless of backend.
    """
    if not hasattr(torch, "backends") or not hasattr(torch.backends, "cuda"):
        return

    # Patch is_built() to return True when MUSA is available
    # This allows code that checks torch.backends.cuda.is_built() to proceed
    original_is_built = torch.backends.cuda.is_built

    # Cache the result since it won't change at runtime
    _is_built_cache = {}

    def patched_is_built():
        if "result" not in _is_built_cache:
            # On MUSA platform, report as "built" since we redirect cuda->musa.
            # Use is_musa_platform() instead of torch.musa.is_available() so this
            # works even when no GPU card is present (build-only environments).
            if is_musa_platform():
                _is_built_cache["result"] = True
            else:
                _is_built_cache["result"] = original_is_built()
        return _is_built_cache["result"]

    torch.backends.cuda.is_built = patched_is_built

    # Patch cuBLASModule to support fp32_precision attribute
    # This attribute is in newer PyTorch but may be missing in torch_musa's version
    matmul = torch.backends.cuda.matmul
    matmul_class = matmul.__class__

    # Check if fp32_precision is already supported
    try:
        _ = matmul.fp32_precision
        # Already supported, no need to patch
        return
    except AttributeError:
        pass

    # Store original methods
    original_getattr = matmul_class.__getattr__
    original_setattr = matmul_class.__setattr__

    def patched_getattr(self, name):
        if name == "fp32_precision":
            return torch.get_float32_matmul_precision()
        return original_getattr(self, name)

    def patched_setattr(self, name, value):
        if name == "fp32_precision":
            return torch.set_float32_matmul_precision(value)
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

    # Patch the key classes and functions
    torch_cpp_ext.CUDAExtension = torchada_cpp_ext.CUDAExtension
    torch_cpp_ext.BuildExtension = torchada_cpp_ext.BuildExtension
    torch_cpp_ext.CUDA_HOME = torchada_cpp_ext.CUDA_HOME

    # Patch include_paths and library_paths to handle both old and new signatures
    # and to correctly translate "cuda" to MUSA on MUSA platform
    torch_cpp_ext.include_paths = torchada_cpp_ext.include_paths
    torch_cpp_ext.library_paths = torchada_cpp_ext.library_paths

    # Also update sys.modules entry
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

    # Patch the CUDA_VISIBLE_DEVICES constant to use MUSA_VISIBLE_DEVICES
    if hasattr(autotune_process, "CUDA_VISIBLE_DEVICES"):
        autotune_process.CUDA_VISIBLE_DEVICES = "MUSA_VISIBLE_DEVICES"


@patch_function
@requires_import("torch_musa")
def _patch_validate_device():
    import torch.nn.attention.flex_attention
    from torch import Tensor
    def _validate_device(query: Tensor, key: Tensor, value: Tensor):
        """TODO: Remove once non cuda/cpu devices support is added
        We only need to check query since we have already that q,k,v are on the same device
        """
        if query.device.type != "cuda" and query.device.type != "cpu" and query.device.type != "musa":
            raise ValueError(
                "FlexAttention is only supported on CUDA or CPU or MUSA devices. "
                f"Found input tensors on {query.device.type} device."
            )
    torch.nn.attention.flex_attention._validate_device = _validate_device


class _CDLLWrapper:
    """
    Wrapper for ctypes.CDLL that automatically translates CUDA/NCCL function names
    to MUSA/MCCL equivalents when accessing library functions.

    This allows code that uses ctypes to load CUDA libraries (libcudart, libnccl) and
    access CUDA-named functions to work transparently on MUSA without code changes.

    Example:
        # Original code uses CUDA function names:
        lib = ctypes.CDLL("libmusart.so")
        func = lib.cudaIpcOpenMemHandle  # Automatically translates to musaIpcOpenMemHandle

        lib = ctypes.CDLL("libmccl.so")
        func = lib.ncclAllReduce  # Automatically translates to mcclAllReduce
    """

    # Detect library type from filename patterns
    _MUSART_PATTERNS = ("libmusart", "musart.so", "libmusa_runtime")
    _MCCL_PATTERNS = ("libmccl", "mccl.so")
    _MUBLAS_PATTERNS = ("libmublas", "mublas.so")
    _MURAND_PATTERNS = ("libmurand", "murand.so")

    def __init__(self, cdll_instance, lib_path: str):
        # Store the original CDLL instance
        object.__setattr__(self, "_cdll", cdll_instance)
        object.__setattr__(self, "_lib_path", lib_path)
        object.__setattr__(self, "_lib_type", self._detect_lib_type(lib_path))

    def _detect_lib_type(self, lib_path: str) -> str:
        """Detect the type of library from its path."""
        lib_path_lower = lib_path.lower()
        if any(p in lib_path_lower for p in self._MUSART_PATTERNS):
            return "musart"
        elif any(p in lib_path_lower for p in self._MCCL_PATTERNS):
            return "mccl"
        elif any(p in lib_path_lower for p in self._MUBLAS_PATTERNS):
            return "mublas"
        elif any(p in lib_path_lower for p in self._MURAND_PATTERNS):
            return "murand"
        return "unknown"

    def _translate_name(self, name: str) -> str:
        """Translate CUDA/NCCL function name to MUSA/MCCL equivalent."""
        lib_type = object.__getattribute__(self, "_lib_type")

        if lib_type == "musart":
            # cudaXxx -> musaXxx
            if name.startswith("cuda"):
                return "musa" + name[4:]
        elif lib_type == "mccl":
            # ncclXxx -> mcclXxx
            if name.startswith("nccl"):
                return "mccl" + name[4:]
        elif lib_type == "mublas":
            # cublasXxx -> mublasXxx
            if name.startswith("cublas"):
                return "mublas" + name[6:]
        elif lib_type == "murand":
            # curandXxx -> murandXxx
            if name.startswith("curand"):
                return "murand" + name[6:]

        return name

    def __getattr__(self, name: str):
        cdll = object.__getattribute__(self, "_cdll")
        translated_name = self._translate_name(name)
        value = getattr(cdll, translated_name)
        # Cache in __dict__ for faster subsequent access
        object.__setattr__(self, name, value)
        return value

    def __setattr__(self, name: str, value):
        cdll = object.__getattribute__(self, "_cdll")
        translated_name = self._translate_name(name)
        setattr(cdll, translated_name, value)

    def __getitem__(self, name: str):
        cdll = object.__getattribute__(self, "_cdll")
        translated_name = self._translate_name(name)
        return cdll[translated_name]


# Store original ctypes.CDLL for patching
_original_ctypes_CDLL = None


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
        # This will automatically find musaIpcOpenMemHandle:
        func = lib.cudaIpcOpenMemHandle
    """
    import ctypes

    global _original_ctypes_CDLL

    # Only patch once
    if _original_ctypes_CDLL is not None:
        return

    _original_ctypes_CDLL = ctypes.CDLL

    class PatchedCDLL:
        """Patched CDLL that wraps MUSA libraries with function name translation."""

        def __new__(cls, name, *args, **kwargs):
            # Create the original CDLL instance
            cdll_instance = _original_ctypes_CDLL(name, *args, **kwargs)

            # Check if this is a MUSA library that needs wrapping
            name_str = str(name) if name else ""
            if any(
                pattern in name_str.lower()
                for pattern in (
                    "libmusart",
                    "musart.so",
                    "libmusa_runtime",
                    "libmccl",
                    "mccl.so",
                    "libmublas",
                    "mublas.so",
                    "libmurand",
                    "murand.so",
                )
            ):
                return _CDLLWrapper(cdll_instance, name_str)

            # For non-MUSA libraries, return the original CDLL instance
            return cdll_instance

    ctypes.CDLL = PatchedCDLL


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

    # Import torch_musa to ensure it's initialized
    try:
        import torch_musa  # noqa: F401
    except ImportError:
        _patched = True
        return

    # Apply all registered patch functions
    # These are registered via @patch_function decorator in definition order
    for patch_fn in _patch_registry:
        patch_fn()

    # Patch torch.Tensor.to()
    if hasattr(torch.Tensor, "to"):
        torch.Tensor.to = _wrap_to_method(torch.Tensor.to)

    # Patch torch.Tensor.cuda()
    if hasattr(torch.Tensor, "cuda"):
        torch.Tensor.cuda = _wrap_tensor_cuda(torch.Tensor.cuda)

    # Patch torch.nn.Module.cuda()
    if hasattr(torch.nn.Module, "cuda"):
        torch.nn.Module.cuda = _wrap_module_cuda(torch.nn.Module.cuda)

    # Patch tensor factory functions to translate device argument
    # We also need to update _device_constructors cache to include
    # the original (unwrapped) functions, because PyTorch's __torch_function__
    # dispatch receives the original C function, not our Python wrapper.
    original_fns = []
    for fn_name in _FACTORY_FUNCTIONS:
        if hasattr(torch, fn_name):
            original_fn = getattr(torch, fn_name)
            original_fns.append(original_fn)
            setattr(torch, fn_name, _wrap_factory_function(original_fn))

    # Update _device_constructors to include original functions
    # This ensures the device context manager (with torch.device(...):) works
    # because __torch_function__ receives the original C function
    try:
        from torch.utils._device import _device_constructors

        # Get the current set of constructors
        constructors = _device_constructors()

        # Add original (unwrapped) functions to the constructors set
        # PyTorch's __torch_function__ receives these, not our wrappers
        for orig_fn in original_fns:
            constructors.add(orig_fn)

    except (ImportError, AttributeError):
        pass  # Older PyTorch versions may not have this

    _patched = True


def is_patched() -> bool:
    """Check if patches have been applied."""
    return _patched


# Additional exports for advanced usage
def get_original_init_process_group():
    """Get the original torch.distributed.init_process_group function."""
    return _original_init_process_group
