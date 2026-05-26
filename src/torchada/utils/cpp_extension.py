"""
torchada.utils.cpp_extension - C++/CUDA extension utilities.

This module provides CUDAExtension, BuildExtension, and related utilities
that work on both CUDA and MUSA platforms.

Note: After importing torchada, you can use standard torch.utils.cpp_extension
imports - they are automatically patched to use these implementations on MUSA.

Usage (preferred):
    import torchada  # Apply patches first
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

    ext_modules = [
        CUDAExtension(
            name="my_extension",
            sources=["my_extension.cpp", "my_extension_kernel.cu"],
        )
    ]

    setup(
        name="my_package",
        ext_modules=ext_modules,
        cmdclass={"build_ext": BuildExtension},
    )
"""

import logging
import os
from typing import Any, Dict, List, Optional

from .._mapping import _MAPPING_RULE, EXT_REPLACED_MAPPING
from .._platform import Platform, detect_platform, is_musa_platform

# Flag to track if torch_musa patches have been applied
_musa_patches_applied = False


def _get_cuda_home() -> Optional[str]:
    """
    Get the CUDA or MUSA home directory.

    On MUSA platform, this returns MUSA_HOME but is still called CUDA_HOME
    so developers don't need to change their code.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Check MUSA_HOME first, then common paths
        musa_home = os.environ.get("MUSA_HOME")
        if musa_home:
            return musa_home

        # Common MUSA installation paths
        common_paths = [
            "/usr/local/musa",
            "/opt/musa",
        ]
        for path in common_paths:
            if os.path.exists(path):
                return path
        return None

    elif platform == Platform.CUDA:
        # Use torch's CUDA_HOME
        try:
            from torch.utils.cpp_extension import CUDA_HOME as TORCH_CUDA_HOME

            return TORCH_CUDA_HOME
        except ImportError:
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
            if cuda_home:
                return cuda_home
            if os.path.exists("/usr/local/cuda"):
                return "/usr/local/cuda"
            return None

    return None


def _is_cuda_file(path: str) -> bool:
    """Check if a file is a CUDA source file."""
    ext = os.path.splitext(path)[1]
    return ext in [".cu", ".cuh"]


def _is_musa_file(path: str) -> bool:
    """
    Check if a file is a MUSA source file.
    Also recognizes .cu/.cuh files as MUSA files for compatibility.

    This function is used to patch torch_musa so it recognizes .cu files.
    """
    ext = os.path.splitext(path)[1]
    return ext in [".cu", ".cuh", ".mu", ".muh"]


def _patch_simple_porting_load_replaced_mapping(musa_sp):
    """
    Patch SimplePorting.load_replaced_mapping to suppress unwanted print output.

    Some versions of torch_musa have `print(self.mapping_rule)` in this method.
    This patch wraps the method to redirect stdout during execution.

    This is forward-compatible: if future versions remove the print, this still works.
    """
    import io
    import sys

    original_method = musa_sp.SimplePorting.load_replaced_mapping

    def patched_load_replaced_mapping(self):
        # Temporarily redirect stdout to suppress the print
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return original_method(self)
        finally:
            sys.stdout = old_stdout

    musa_sp.SimplePorting.load_replaced_mapping = patched_load_replaced_mapping


def _patch_simple_porting_open(musa_sp):
    """
    Patch simple_porting.open to tolerate non-UTF-8 bytes in source files.

    Some source files processed by SimplePorting may contain comments or string
    literals that are not valid UTF-8. The original implementation opens files
    with UTF-8 decoding, which can raise UnicodeDecodeError during porting.

    This patch wraps text-mode UTF-8 file opens with errors="surrogateescape"
    so undecodable bytes can round-trip safely. The original SimplePorting
    modify_file logic is preserved, keeping this patch forward-compatible with
    future torch_musa changes.
    """

    import builtins

    def open_with_surrogateescape(file, mode="r", *args, **kwargs):
        if "b" not in mode and kwargs.get("encoding") == "utf-8" and "errors" not in kwargs:
            kwargs["errors"] = "surrogateescape"
        return builtins.open(file, mode, *args, **kwargs)

    musa_sp.open = open_with_surrogateescape


def _apply_musa_patches():
    """
    Apply patches to torch_musa modules for CUDA compatibility.

    This function patches:
    1. musa_ext._is_musa_file - to recognize .cu/.cuh files
    2. musa_sp.EXT_REPLACED_MAPPING - to convert .cu/.cuh to .mu/.muh
    3. musa_sp._MAPPING_RULE - to apply CUDA->MUSA symbol mapping

    These patches are required to compile .cu files on MUSA platform.
    """
    global _musa_patches_applied

    if _musa_patches_applied:
        return

    if not is_musa_platform():
        return

    try:
        import torch_musa.utils.musa_extension as musa_ext
        import torch_musa.utils.simple_porting as musa_sp

        # Patch _is_musa_file to recognize .cu/.cuh files
        musa_ext._is_musa_file = _is_musa_file

        # Patch EXT_REPLACED_MAPPING to convert .cu/.cuh to .mu/.muh
        musa_sp.EXT_REPLACED_MAPPING = EXT_REPLACED_MAPPING

        # Patch _MAPPING_RULE with our comprehensive CUDA->MUSA mappings
        # This is the critical patch that enables source code porting
        musa_sp._MAPPING_RULE = _MAPPING_RULE

        # Patch load_replaced_mapping to suppress print(self.mapping_rule)
        # Some versions of torch_musa have an extra print statement that we want to disable
        # This patch is forward-compatible - if the print is removed, this still works
        _patch_simple_porting_load_replaced_mapping(musa_sp)
        # Patch simple_porting.open to tolerate non-UTF-8 source files
        # This preserves SimplePorting's original logic while allowing undecodable bytes to round-trip
        _patch_simple_porting_open(musa_sp)

        _musa_patches_applied = True

    except ImportError:
        # torch_musa not available, patches not needed
        pass


# Apply MUSA patches at module import time
_apply_musa_patches()

# Export CUDA_HOME - always use this name, even on MUSA platform
# This way developers don't need to change their code
CUDA_HOME = _get_cuda_home()


def _port_cuda_source(source_code: str, mapping_rules: Optional[Dict[str, str]] = None) -> str:
    """
    Port CUDA source code to MUSA by applying mapping rules.

    Args:
        source_code: The CUDA source code to port
        mapping_rules: Optional custom mapping rules (defaults to _MAPPING_RULE)

    Returns:
        The ported MUSA source code
    """
    if mapping_rules is None:
        mapping_rules = _MAPPING_RULE

    result = source_code

    # Sort rules by length (longest first) to avoid partial replacements
    sorted_rules = sorted(mapping_rules.items(), key=lambda x: len(x[0]), reverse=True)

    for cuda_symbol, musa_symbol in sorted_rules:
        if cuda_symbol != musa_symbol:
            result = result.replace(cuda_symbol, musa_symbol)

    return result


def include_paths(cuda: Optional[bool] = None, device_type: Optional[str] = None) -> List[str]:
    """
    Get include paths for compiling extensions.

    Supports both PyTorch < 2.6 (cuda=True) and PyTorch 2.6+ (device_type="cuda")
    signatures for compatibility.

    Args:
        cuda: (PyTorch < 2.6) Whether to include CUDA/MUSA paths. Deprecated in 2.6+.
        device_type: (PyTorch 2.6+) Device type string, e.g. "cuda", "cpu", "musa".

    Returns:
        List of include paths
    """
    # Handle both old (cuda=bool) and new (device_type=str) signatures
    if device_type is not None:
        # PyTorch 2.6+ style: device_type="cuda" or "cpu"
        # Translate "cuda" to MUSA include paths on MUSA platform
        include_device = device_type.lower() in ("cuda", "musa")
    elif cuda is not None:
        include_device = cuda
    else:
        # Default: include device paths
        include_device = True

    platform = detect_platform()

    if platform == Platform.MUSA:
        try:
            import torch_musa.utils.musa_extension as musa_ext

            if hasattr(musa_ext, "include_paths"):
                # musa_ext uses musa=bool parameter, not cuda= or device_type=
                return musa_ext.include_paths(musa=include_device)
        except ImportError:
            pass

        # Fallback: construct paths manually
        paths = []
        musa_home = _get_cuda_home()
        if musa_home:
            paths.append(os.path.join(musa_home, "include"))
        return paths

    else:
        # Check which signature the torch version supports
        import inspect

        from torch.utils.cpp_extension import include_paths as torch_include_paths

        sig = inspect.signature(torch_include_paths)
        if "device_type" in sig.parameters:
            # PyTorch 2.6+
            if device_type is not None:
                return torch_include_paths(device_type=device_type)
            else:
                return torch_include_paths(device_type="cuda" if include_device else "cpu")
        else:
            # PyTorch < 2.6
            return torch_include_paths(cuda=include_device)


def library_paths(cuda: Optional[bool] = None, device_type: Optional[str] = None) -> List[str]:
    """
    Get library paths for compiling extensions.

    Supports both PyTorch < 2.6 (cuda=True) and PyTorch 2.6+ (device_type="cuda")
    signatures for compatibility.

    Args:
        cuda: (PyTorch < 2.6) Whether to include CUDA/MUSA library paths. Deprecated in 2.6+.
        device_type: (PyTorch 2.6+) Device type string, e.g. "cuda", "cpu", "musa".

    Returns:
        List of library paths
    """
    # Handle both old (cuda=bool) and new (device_type=str) signatures
    if device_type is not None:
        # PyTorch 2.6+ style: device_type="cuda" or "cpu"
        # Translate "cuda" to MUSA library paths on MUSA platform
        include_device = device_type.lower() in ("cuda", "musa")
    elif cuda is not None:
        include_device = cuda
    else:
        # Default: include device paths
        include_device = True

    platform = detect_platform()

    if platform == Platform.MUSA:
        if not include_device:
            return []

        try:
            import torch_musa.utils.musa_extension as musa_ext

            if hasattr(musa_ext, "library_paths"):
                # musa_ext uses musa=bool parameter, not cuda= or device_type=
                return musa_ext.library_paths(musa=include_device)
        except ImportError:
            pass

        # Fallback: construct paths manually
        paths = []
        musa_home = _get_cuda_home()
        if musa_home:
            paths.append(os.path.join(musa_home, "lib"))
            paths.append(os.path.join(musa_home, "lib64"))
        return [p for p in paths if os.path.exists(p)]

    else:
        # Check which signature the torch version supports
        import inspect

        from torch.utils.cpp_extension import library_paths as torch_library_paths

        sig = inspect.signature(torch_library_paths)
        if "device_type" in sig.parameters:
            # PyTorch 2.6+
            if device_type is not None:
                return torch_library_paths(device_type=device_type)
            else:
                return torch_library_paths(device_type="cuda" if include_device else "cpu")
        else:
            # PyTorch < 2.6
            return torch_library_paths(cuda=include_device)


class CUDAExtension:
    """
    A wrapper that creates either a torch CUDAExtension or MUSA MUSAExtension.

    This class provides a unified interface for building CUDA extensions
    that works transparently on both CUDA and MUSA platforms.
    """

    def __new__(cls, name: str, sources: List[str], *args, **kwargs):
        """
        Create a new extension module.

        Args:
            name: The name of the extension
            sources: List of source files
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        platform = detect_platform()

        if platform == Platform.MUSA:
            return _create_musa_extension(name, sources, *args, **kwargs)
        else:
            return _create_cuda_extension(name, sources, *args, **kwargs)


class CppExtension:
    """
    A wrapper for creating C++ extensions (no CUDA/MUSA).
    """

    def __new__(cls, name: str, sources: List[str], *args, **kwargs):
        """Create a C++ extension module."""
        from torch.utils.cpp_extension import CppExtension as TorchCppExtension

        return TorchCppExtension(name, sources, *args, **kwargs)


def _create_cuda_extension(name: str, sources: List[str], *args, **kwargs):
    """Create a CUDA extension using torch's CUDAExtension."""
    from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension

    return TorchCUDAExtension(name, sources, *args, **kwargs)


def _translate_compile_args(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate CUDA-style compile args to MUSA-style.

    This function maps:
    - 'nvcc' key to 'mcc' for MUSA compiler flags
    - Keeps 'cxx' key as-is for C++ compiler flags

    This allows developers to use standard 'nvcc' key in extra_compile_args
    and have it work transparently on MUSA platform.
    """
    if "extra_compile_args" not in kwargs:
        return kwargs

    extra_compile_args = kwargs["extra_compile_args"]
    if not isinstance(extra_compile_args, dict):
        return kwargs

    # Create a new dict with translated keys
    new_compile_args = {}
    for key, value in extra_compile_args.items():
        if key == "nvcc":
            # Map 'nvcc' to 'mcc' for MUSA
            new_compile_args["mcc"] = value
        else:
            new_compile_args[key] = value

    # Return a copy of kwargs with updated extra_compile_args
    new_kwargs = kwargs.copy()
    new_kwargs["extra_compile_args"] = new_compile_args
    return new_kwargs


def _create_musa_extension(name: str, sources: List[str], *args, **kwargs):
    """Create a MUSA extension using torch_musa's MUSAExtension.

    The patches applied by _apply_musa_patches() make MUSAExtension accept
    .cu/.cuh files directly by:
    1. Patching musa_ext._is_musa_file to recognize .cu/.cuh as valid MUSA files
    2. Patching musa_sp.EXT_REPLACED_MAPPING to convert .cu/.cuh to .mu/.muh
    3. Patching musa_sp._MAPPING_RULE to convert CUDA symbols to MUSA in source code
    4. Translating 'nvcc' compile args key to 'mcc' for MUSA compiler
    """
    # Ensure patches are applied
    _apply_musa_patches()

    # Translate 'nvcc' to 'mcc' in extra_compile_args
    kwargs = _translate_compile_args(kwargs)

    try:
        import torch_musa.utils.musa_extension as musa_ext

        # Simply pass sources to MUSAExtension - patches make it accept .cu files
        return musa_ext.MUSAExtension(name, sources, *args, **kwargs)
    except ImportError:
        # Fallback to torch's CUDAExtension if torch_musa is not available
        from torch.utils.cpp_extension import CUDAExtension as TorchCUDAExtension

        return TorchCUDAExtension(name, sources, *args, **kwargs)


def _get_build_extension_class():
    """
    Get the BuildExtension class for the current platform.

    On MUSA platform, returns a custom class that:
    1. Uses SimplePorting to convert CUDA sources to MUSA in run() (like torch's HIPIFY)
    2. Registers .cu/.cuh as valid source extensions in build_extensions()
    3. Provides extensible mapping rules via get_mapping_rule() method

    The porting process is automatic and transparent - developers use csrc/*.cu paths
    and the build system handles conversion to csrc_musa/*.cu internally.
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Ensure patches are applied
        _apply_musa_patches()
        try:
            import torch_musa.utils.musa_extension as musa_ext
            import torch_musa.utils.simple_porting as musa_sp

            # Patch _is_musa_file to also recognize .cu/.cuh files as MUSA sources
            # This allows keeping original CUDA file extensions while still compiling
            # them with the MUSA compiler (mcc with -x musa flag)
            _original_is_musa_file = musa_ext._is_musa_file

            def _patched_is_musa_file(path: str) -> bool:
                """Check if a file is a MUSA source file (including .cu/.cuh)."""
                ext = os.path.splitext(path)[1].lower()
                # Include .cu/.cuh in addition to .mu/.muh
                if ext in [".cu", ".cuh"]:
                    return True
                return _original_is_musa_file(path)

            musa_ext._is_musa_file = _patched_is_musa_file

            class _MUSABuildExtension(musa_ext.BuildExtension):
                """
                Custom BuildExtension that handles CUDA->MUSA source porting.

                This class works like torch's HIPIFY for ROCm:
                - run(): Automatically ports CUDA sources to MUSA using SimplePorting
                - build_extensions(): Registers .cu/.cuh as valid extensions
                - get_mapping_rule(): Returns mapping rules (override in subclass to extend)

                Subclasses can override get_mapping_rule() to add project-specific mappings:

                    class MyBuildExt(_MUSABuildExtension):
                        def get_mapping_rule(self):
                            base_rules = super().get_mapping_rule()
                            return {
                                **base_rules,
                                "my_cuda_func": "my_musa_func",
                            }
                """

                # Track directories that have been ported (class-level for persistence)
                _ported_dirs = set()

                def get_mapping_rule(self):
                    """
                    Get the CUDA->MUSA mapping rules for source porting.

                    Override this method in subclasses to add project-specific mappings.
                    Call super().get_mapping_rule() and merge with additional rules.

                    Returns:
                        dict: Mapping from CUDA symbols to MUSA equivalents
                    """
                    return _MAPPING_RULE.copy()

                def build_extensions(self):
                    # Register .cu, .cuh as valid source extensions
                    self.compiler.src_extensions += [".cu", ".cuh"]
                    super().build_extensions()

                def _port_directory(self, source_dir, mapping_rule=None):
                    """
                    Port a directory containing CUDA sources to MUSA.

                    When both .cu and .mu files exist with the same base name,
                    the .mu file takes precedence (it's the hand-written MUSA version).

                    Args:
                        source_dir: Path to directory containing CUDA sources
                        mapping_rule: Optional custom mapping rules (uses get_mapping_rule() if None)

                    Returns:
                        str: Path to the ported directory (source_dir + "_musa")
                    """
                    if mapping_rule is None:
                        mapping_rule = self.get_mapping_rule()

                    source_dir = os.path.abspath(source_dir)
                    musa_dir = source_dir + "_musa"

                    if source_dir not in self._ported_dirs:
                        musa_sp.LOGGER.setLevel(logging.ERROR)
                        musa_sp.SimplePorting(
                            cuda_dir_path=source_dir, mapping_rule=mapping_rule
                        ).run()
                        self._ported_dirs.add(source_dir)

                        # After porting, copy any original .mu/.muh files to ensure
                        # hand-written MUSA files take precedence over ported .cu files.
                        # This fixes the case where both foo.cu and foo.mu exist -
                        # SimplePorting processes both, but order is non-deterministic.
                        import shutil

                        for root, _, files in os.walk(source_dir):
                            for file in files:
                                ext = os.path.splitext(file)[1].lower()
                                if ext in [".mu", ".muh"]:
                                    src_path = os.path.join(root, file)
                                    rel_path = os.path.relpath(root, source_dir)
                                    dst_dir = os.path.join(musa_dir, rel_path)
                                    dst_path = os.path.join(dst_dir, file)
                                    os.makedirs(dst_dir, exist_ok=True)
                                    shutil.copy2(src_path, dst_path)

                    return musa_dir

                def _convert_source_path(self, source):
                    """
                    Convert a CUDA source path to its ported MUSA equivalent.

                    Args:
                        source: Original source file path (e.g., "csrc/kernel.cu")

                    Returns:
                        tuple: (converted_path, needs_porting)
                            - converted_path: Path to ported file (e.g., "csrc_musa/kernel.mu")
                            - needs_porting: True if the source directory needs porting
                    """
                    source_path = os.path.abspath(source)
                    source_dir = os.path.dirname(source_path)
                    source_file = os.path.basename(source_path)
                    base_name, ext_name = os.path.splitext(source_file)
                    ext_name_lower = ext_name.lower()

                    # Port all source files that may contain CUDA references:
                    # - .cu/.cuh: CUDA source/header files
                    # - .cc/.cpp/.cxx: C++ files that may reference CUDA symbols
                    if ext_name_lower in [".cu", ".cuh", ".cc", ".cpp", ".cxx"]:
                        # Get the ported extension (kept same with our EXT_REPLACED_MAPPING)
                        new_ext = EXT_REPLACED_MAPPING.get(ext_name_lower[1:], ext_name_lower[1:])
                        musa_dir = source_dir + "_musa"
                        new_source = os.path.join(musa_dir, base_name + "." + new_ext)
                        return new_source, True
                    # Handle .mu/.muh files that don't exist at original path
                    # Try to find them in the _musa directory (already ported files)
                    elif ext_name_lower in [".mu", ".muh"]:
                        if not os.path.exists(source_path):
                            musa_dir = source_dir + "_musa"
                            musa_source = os.path.join(musa_dir, source_file)
                            if os.path.exists(musa_source):
                                # File exists in _musa dir, use it (no porting needed)
                                return musa_source, False
                        # File exists at original path, use it as-is
                        return source, False
                    else:
                        return source, False

                def run(self):
                    """
                    Run the build process with automatic CUDA->MUSA porting.

                    This method:
                    1. Identifies CUDA source directories from extension sources
                    2. Ports each directory using SimplePorting (like torch's HIPIFY)
                    3. Updates source paths to point to ported files
                    4. Ports include directories as well
                    5. Calls parent run() to perform actual compilation
                    """
                    mapping_rule = self.get_mapping_rule()

                    for ext in self.extensions:
                        new_sources = []
                        dirs_to_port = set()

                        # First pass: identify directories that need porting
                        for source in ext.sources:
                            (
                                new_source,
                                needs_porting,
                            ) = self._convert_source_path(source)
                            new_sources.append(new_source)
                            if needs_porting:
                                source_dir = os.path.dirname(os.path.abspath(source))
                                dirs_to_port.add(source_dir)
                        # Sort directories by depth (deepest first) to ensure proper porting order
                        dirs_to_port = sorted(
                            dirs_to_port, key=lambda p: p.count("/"), reverse=True
                        )
                        # Port each unique directory
                        for cuda_dir in dirs_to_port:
                            self._port_directory(cuda_dir, mapping_rule)

                        # Update extension sources to point to ported files
                        ext.sources = new_sources

                        # Port include directories and update include_dirs
                        # Only port project-local directories, not system paths
                        if hasattr(ext, "include_dirs") and ext.include_dirs:
                            new_include_dirs = []
                            for inc_dir in ext.include_dirs:
                                inc_dir_abs = os.path.abspath(inc_dir)
                                # Skip system directories - only port project-local dirs
                                # System dirs typically start with /usr, /opt, or site-packages
                                is_system_dir = (
                                    inc_dir_abs.startswith("/usr/")
                                    or inc_dir_abs.startswith("/opt/")
                                    or "site-packages" in inc_dir_abs
                                    or "dist-packages" in inc_dir_abs
                                )
                                if os.path.isdir(inc_dir_abs) and not is_system_dir:
                                    # Check if directory might contain CUDA headers (recursively)
                                    has_cuda_headers = False
                                    try:
                                        for root, dirs, files in os.walk(inc_dir_abs):
                                            for f in files:
                                                if f.endswith(
                                                    (
                                                        ".h",
                                                        ".hpp",
                                                        ".cuh",
                                                        ".cu",
                                                    )
                                                ):
                                                    has_cuda_headers = True
                                                    break
                                            if has_cuda_headers:
                                                break
                                    except OSError:
                                        pass

                                    if has_cuda_headers:
                                        ported_dir = self._port_directory(inc_dir_abs, mapping_rule)
                                        # Add ported dir first so ported headers take precedence
                                        if os.path.isdir(ported_dir):
                                            new_include_dirs.append(ported_dir)
                                new_include_dirs.append(inc_dir_abs)
                            ext.include_dirs = new_include_dirs

                    super().run()

            return _MUSABuildExtension
        except ImportError:
            pass

    # Fallback to torch's BuildExtension
    from torch.utils.cpp_extension import BuildExtension as TorchBuildExtension

    return TorchBuildExtension


# Get the actual BuildExtension class at module load time
# This ensures BuildExtension is a proper class that inherits from Command
BuildExtension = _get_build_extension_class()


def load(
    name: str,
    sources: List[str],
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
    with_cuda: Optional[bool] = None,
    is_python_module: bool = True,
    is_standalone: bool = False,
    keep_intermediates: bool = True,
):
    """
    Load a PyTorch C++/CUDA extension at runtime (JIT compilation).

    This function works on both CUDA and MUSA platforms.

    Args:
        name: The name of the extension
        sources: List of source files
        extra_cflags: Extra C++ compiler flags
        extra_cuda_cflags: Extra CUDA/MUSA compiler flags
        extra_ldflags: Extra linker flags
        extra_include_paths: Extra include paths
        build_directory: Directory to build in
        verbose: Whether to print build output
        with_cuda: Whether to include CUDA/MUSA support
        is_python_module: Whether this is a Python module
        is_standalone: Whether this is a standalone executable
        keep_intermediates: Whether to keep intermediate files

    Returns:
        The loaded extension module
    """
    platform = detect_platform()

    if platform == Platform.MUSA:
        # Ensure patches are applied
        _apply_musa_patches()

        try:
            import torch_musa.utils.musa_extension as musa_ext

            # Use MUSA's load function if available
            # Note: MUSA uses different parameter names:
            #   extra_cuda_cflags -> extra_musa_cflags
            #   with_cuda -> with_musa
            if hasattr(musa_ext, "load"):
                return musa_ext.load(
                    name=name,
                    sources=sources,
                    extra_cflags=extra_cflags,
                    extra_musa_cflags=extra_cuda_cflags,
                    extra_ldflags=extra_ldflags,
                    extra_include_paths=extra_include_paths,
                    build_directory=build_directory,
                    verbose=verbose,
                    with_musa=with_cuda,
                    is_python_module=is_python_module,
                    is_standalone=is_standalone,
                    keep_intermediates=keep_intermediates,
                )
        except ImportError:
            pass

    # Fallback to torch's load
    from torch.utils.cpp_extension import load as torch_load

    return torch_load(
        name=name,
        sources=sources,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        is_standalone=is_standalone,
        keep_intermediates=keep_intermediates,
    )


def load_inline(
    name: str,
    cpp_sources: List[str],
    cuda_sources: Optional[List[str]] = None,
    functions: Optional[List[str]] = None,
    extra_cflags: Optional[List[str]] = None,
    extra_cuda_cflags: Optional[List[str]] = None,
    extra_ldflags: Optional[List[str]] = None,
    extra_include_paths: Optional[List[str]] = None,
    build_directory: Optional[str] = None,
    verbose: bool = False,
    with_cuda: Optional[bool] = None,
    is_python_module: bool = True,
    with_pytorch_error_handling: bool = True,
    keep_intermediates: bool = True,
):
    """
    Load a PyTorch C++/CUDA extension from inline source code.

    This function works on both CUDA and MUSA platforms.
    """
    platform = detect_platform()

    # On MUSA platform, apply patches and port CUDA sources to MUSA
    if platform == Platform.MUSA:
        _apply_musa_patches()
        if cuda_sources:
            cuda_sources = [_port_cuda_source(src) for src in cuda_sources]

    from torch.utils.cpp_extension import load_inline as torch_load_inline

    return torch_load_inline(
        name=name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
        build_directory=build_directory,
        verbose=verbose,
        with_cuda=with_cuda,
        is_python_module=is_python_module,
        with_pytorch_error_handling=with_pytorch_error_handling,
        keep_intermediates=keep_intermediates,
    )


# Export all public symbols
# Note: We only export CUDA_HOME, not MUSA_HOME. On MUSA platform, CUDA_HOME
# points to the MUSA installation so developers don't need to change their code.
__all__ = [
    "CUDA_HOME",
    "CUDAExtension",
    "CppExtension",
    "BuildExtension",
    "include_paths",
    "library_paths",
    "load",
    "load_inline",
]
