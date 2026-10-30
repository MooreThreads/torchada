# SPDX-License-Identifier: Apache-2.0
"""The build side of :mod:`torchada.musa`, importable without torch.

``torchada/__init__.py`` imports torch and applies the CUDA-to-MUSA patches, so
``from torchada.musa import build`` in a ``setup.py`` or a PEP 517 backend pulls a
torch install and a runtime into a process that wants neither.  Those modules are
standard-library only on purpose (the FP8 answer has to be available before the
package being built exists), and the only thing standing between them and a plain
import is the package ``__init__`` above them.

This module is that entry point: it registers the same files under a private package
name, so the relative imports inside them resolve and nothing under ``torchada`` is
executed.

    import torchada_build

    torchada_build.targets.get_arch_spec("mp_31").native_fp8   # True
    torchada_build.build.resolve_targets("mp_31").fp8          # True

Two things to know.  The modules loaded here are *the same source files* as
``torchada.musa.targets`` but not the same module objects: if a process has already
imported ``torchada``, prefer the package's own names, and do not compare types or
``isinstance`` across the two.  And this is the *build* surface only --
:mod:`torchada.musa.device`, which probes the card, imports torch and is not here.
"""

from __future__ import annotations

import importlib.util as _importlib_util
import os as _os
import sys as _sys
import types as _types

__all__ = ["build", "profiles", "targets"]

_HERE = _os.path.dirname(_os.path.abspath(__file__))
_MUSA_DIR = _os.path.join(_HERE, "torchada", "musa")

#: Private package name, so loading these files cannot shadow or preempt the real
#: ``torchada``/``torchada.musa`` entries in :data:`sys.modules`.
_PACKAGE = "_torchada_build_standalone"


def _ensure_package() -> None:
    package = _sys.modules.get(_PACKAGE)
    if package is not None:
        return
    package = _types.ModuleType(_PACKAGE)
    package.__path__ = [_MUSA_DIR]
    package.__package__ = _PACKAGE
    _sys.modules[_PACKAGE] = package


def _load(name: str) -> _types.ModuleType:
    """Import ``torchada/musa/<name>.py`` as ``<private package>.<name>``."""
    qualified = "{}.{}".format(_PACKAGE, name)
    loaded = _sys.modules.get(qualified)
    if loaded is not None:
        return loaded
    path = _os.path.join(_MUSA_DIR, name + ".py")
    if not _os.path.exists(path):
        raise ImportError(
            "torchada_build cannot find {!r}: expected it beside this module at {}. "
            "A source checkout needs no extra step; a broken install does.".format(
                _os.path.relpath(path, _HERE), path
            )
        )
    spec = _importlib_util.spec_from_file_location(qualified, path)
    module = _importlib_util.module_from_spec(spec)
    _sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


_ensure_package()
targets = _load("targets")
profiles = _load("profiles")
build = _load("build")
