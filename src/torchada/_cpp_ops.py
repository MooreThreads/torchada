# Copyright (c) torchada contributors
# SPDX-License-Identifier: MIT
"""
C++ operator overrides infrastructure for torchada.

This module handles building and loading C++ extensions that can override
ATen operator implementations for the PrivateUse1 (MUSA) dispatch key.

C++ extensions are automatically loaded on MUSA platform when torchada is imported.

Usage:
    import torchada  # C++ extensions are loaded automatically on MUSA

    # Or explicitly load
    from torchada._cpp_ops import load_cpp_ops
    load_cpp_ops()
"""

import os
import subprocess
from typing import Optional

_cpp_ops_module: Optional[object] = None
_graph_rotation_module: Optional[object] = None
_musa_arch_cached: Optional[str] = None


def _detect_musa_arch() -> str:
    """
    Detect MUSA GPU architecture from compute capability.

    Uses musaInfo to get compute capability and maps it to architecture:
    - 2.1 -> mp_21 (MTT S80)
    - 2.2 -> mp_22 (MTT S4000)
    - 3.1 -> mp_31 (MTT S5000)

    Returns:
        Architecture string like "mp_21", "mp_22", or "mp_31".
        Defaults to "mp_31" if detection fails.
    """
    global _musa_arch_cached
    if _musa_arch_cached is not None:
        return _musa_arch_cached

    arch = "mp_31"  # Default fallback
    try:
        result = subprocess.run(
            ["musaInfo"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if "compute capability:" in line.lower():
                # Parse "compute capability:              2.1"
                parts = line.split(":")
                if len(parts) >= 2:
                    version = parts[1].strip()
                    # Convert "2.1" -> "mp_21", "3.1" -> "mp_31"
                    version_parts = version.split(".")
                    if len(version_parts) >= 2:
                        major = version_parts[0].strip()
                        minor = version_parts[1].strip()
                        arch = f"mp_{major}{minor}"
                        break
    except Exception:
        pass

    _musa_arch_cached = arch
    return arch


def load_cpp_ops(force_reload: bool = False) -> Optional[object]:
    """
    Load the C++ operator overrides extension.

    The extension is automatically loaded on MUSA platform.

    Args:
        force_reload: If True, reload the extension even if already loaded.

    Returns:
        The loaded extension module, or None if not loaded.
    """
    global _cpp_ops_module

    if _cpp_ops_module is not None and not force_reload:
        return _cpp_ops_module

    # Check if on MUSA platform
    from ._platform import is_musa_platform

    if not is_musa_platform():
        return None

    try:
        import os.path as osp

        csrc_dir = osp.join(osp.dirname(__file__), "csrc")

        # Collect all source files
        cpp_sources = []
        musa_sources = []

        for fname in os.listdir(csrc_dir):
            fpath = osp.join(csrc_dir, fname)
            if fname.endswith(".cpp"):
                cpp_sources.append(fpath)
            elif fname.endswith((".cu", ".mu")):
                musa_sources.append(fpath)

        if not cpp_sources and not musa_sources:
            import warnings

            warnings.warn("torchada C++ ops: no source files found")
            return None

        verbose = os.environ.get("TORCHADA_CPP_OPS_VERBOSE") == "1"
        all_sources = cpp_sources + musa_sources

        # Use MUSA extension loader if we have MUSA sources, otherwise use torch's
        if musa_sources:
            # Use torchada's load which handles MUSA properly
            from .utils.cpp_extension import load

            # Get MUSA architecture flags
            # Use MTGPU_TARGET env var if set, otherwise auto-detect from GPU
            extra_cuda_cflags = []
            mtgpu_target = os.environ.get("MTGPU_TARGET", "")
            if not mtgpu_target:
                mtgpu_target = _detect_musa_arch()
            extra_cuda_cflags.append(f"--offload-arch={mtgpu_target}")

            _cpp_ops_module = load(
                name="torchada_cpp_ops",
                sources=all_sources,
                extra_include_paths=[csrc_dir],
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=verbose,
            )
        else:
            # Pure C++ extension - use torch's loader directly
            from torch.utils.cpp_extension import load

            _cpp_ops_module = load(
                name="torchada_cpp_ops",
                sources=all_sources,
                extra_include_paths=[csrc_dir],
                verbose=verbose,
            )

        _cpp_ops_module._mark_loaded()
        return _cpp_ops_module

    except Exception as e:
        import warnings

        warnings.warn(f"Failed to load torchada C++ ops: {e}")
        return None


def load_graph_rotation_ops(force_reload: bool = False) -> Optional[object]:
    """Build/load the MUSA CUDA-graph executable-rotation ops.

    This is a *separate* extension from the ATen op overrides built by
    ``load_cpp_ops``: it needs torch_musa's internal headers (``MUSAGraph.h``) and
    links ``libmusa_python``, and a build failure here must not break the core op
    overrides. It is host-only (no device kernels), so torch's plain loader is used.
    Built lazily by the graph-rotation module the first time rotation engages.

    Returns the extension module exposing ``free_exec``/``inst_exec``/``has_exec``,
    or ``None`` on a non-MUSA platform or build failure.
    """
    global _graph_rotation_module

    if _graph_rotation_module is not None and not force_reload:
        return _graph_rotation_module

    from ._platform import is_musa_platform

    if not is_musa_platform():
        return None

    try:
        import os.path as osp

        import torch_musa
        from torch.utils.cpp_extension import load

        src = osp.join(osp.dirname(__file__), "_graph_rotation_src", "graph_exec_aux.cpp")
        tm = osp.dirname(torch_musa.__file__)
        musa_home = os.environ.get("MUSA_HOME", "/usr/local/musa")
        include_dirs = [
            osp.join(tm, "csrc"),
            osp.join(tm, "share", "generated_cuda_compatible", "include"),
            osp.join(musa_home, "include"),
        ]
        libdir = osp.join(tm, "lib")
        verbose = os.environ.get("TORCHADA_CPP_OPS_VERBOSE") == "1"

        _graph_rotation_module = load(
            name="torchada_graph_rotation_ops",
            sources=[src],
            extra_include_paths=include_dirs,
            extra_cflags=["-O2"],
            extra_ldflags=[
                "-L" + osp.join(musa_home, "lib"),
                "-lmusart",
                "-L" + libdir,
                "-lmusa_python",
                "-Wl,-rpath," + libdir,
            ],
            verbose=verbose,
        )
        return _graph_rotation_module

    except Exception as e:
        import warnings

        warnings.warn(f"Failed to load torchada graph-rotation ops: {e}")
        return None


def is_loaded() -> bool:
    """Check if C++ ops extension is loaded."""
    return _cpp_ops_module is not None


def get_version() -> Optional[str]:
    """Get C++ ops extension version."""
    if _cpp_ops_module is None:
        return None
    return _cpp_ops_module.get_version()


def get_module() -> Optional[object]:
    """Get the loaded C++ ops module."""
    return _cpp_ops_module
