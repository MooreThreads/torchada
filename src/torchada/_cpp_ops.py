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
from typing import Optional

from . import arch

_cpp_ops_module: Optional[object] = None
_graph_rotation_module: Optional[object] = None

_FLOCK_FILE = ".torchada.flock"


def _jit_loader_module(musa: bool):
    """torch_musa's JIT loader for MUSA sources when importable, else torch's."""
    if musa:
        try:
            import torch_musa.utils.musa_extension as loader

            return loader
        except ImportError:
            pass
    import torch.utils.cpp_extension as loader

    return loader


def _locked_load(load, name: str, musa: bool, subdir: Optional[str] = None, **kwargs):
    """Call a JIT ``load`` under a flock that, unlike torch's ``lock`` baton, dies with its holder.

    With ``subdir`` the extension builds in that sub-directory of its build directory,
    and the flock and the stale-baton check live there too.
    """
    import fcntl
    import warnings

    loader = _jit_loader_module(musa)
    if not hasattr(loader, "FileBaton"):  # newer torch guards JIT builds with a FileLock itself
        if subdir is None:
            return load(name=name, **kwargs)
        build_dir = os.path.join(loader._get_build_directory(name, False), subdir)
        os.makedirs(build_dir, exist_ok=True)
        return load(name=name, build_directory=build_dir, **kwargs)
    build_dir = loader._get_build_directory(name, False)
    if subdir is not None:
        build_dir = os.path.join(build_dir, subdir)
    os.makedirs(build_dir, exist_ok=True)
    with open(os.path.join(build_dir, _FLOCK_FILE), "a") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except OSError as e:
            warnings.warn(
                f"torchada: flock unavailable on {lock_file.name}, building without it: {e}"
            )
        else:
            baton = os.path.join(build_dir, "lock")
            if os.path.exists(baton):
                os.remove(baton)
                warnings.warn(f"torchada: removed stale JIT build lock {baton}")
        return load(name=name, build_directory=build_dir, **kwargs)


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

    import torch

    from ._version import version_of

    musa_module = getattr(torch, "musa", None)
    # Legacy shim (compiled only below 2.11.0.post2): those releases lack the
    # ``multinomial`` / ``log`` / ``log_`` kernels these overrides provide. From
    # that release on the native implementations are used, so the overrides are
    # switched off through their env flags.
    if version_of(musa_module) >= "2.11.0.post2":
        for op_name in ("multinomial", "log", "log_"):
            os.environ[f"TORCHADA_DISABLE_OP_OVERRIDE_{op_name}"] = "1"

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

            # MTGPU_TARGET overrides; otherwise every visible device's arch, else mp_31.
            archs = arch.target_archs(env=("MTGPU_TARGET",))
            _cpp_ops_module = _locked_load(
                load,
                name="torchada_cpp_ops",
                musa=True,
                subdir=arch.cache_tag(archs),
                sources=all_sources,
                extra_include_paths=[csrc_dir],
                extra_cuda_cflags=arch.arch_flags(archs),
                verbose=verbose,
            )
        else:
            # Pure C++ extension - use torch's loader directly
            from torch.utils.cpp_extension import load

            _cpp_ops_module = _locked_load(
                load,
                name="torchada_cpp_ops",
                musa=False,
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

        _graph_rotation_module = _locked_load(
            load,
            name="torchada_graph_rotation_ops",
            musa=False,
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
