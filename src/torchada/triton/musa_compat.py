"""
Compatibility fixes for the MUSA Triton backend.

This module imports neither torch nor Triton at module level; Triton is
imported inside functions. ``import torchada`` itself needs torch, so use
without torch requires loading this file directly. The ``torchada._patch``
entries decide when to call the ``install_*`` functions.
"""

import functools
import importlib.metadata
import inspect
import itertools
import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

INPLACE_ALIAS_ENV = "TORCHADA_TRITON_INPLACE_ALIAS"
FAST_EXP_ENV = "TORCHADA_TRITON_FAST_EXP"
F32_DEFAULT_ENV = "TORCHADA_TRITON_F32_DEFAULT"
TRITON_F32_DEFAULT_ENV = "TRITON_F32_DEFAULT"

INPLACE_ALIAS_MODES = ("fix", "off", "vendor")
DEFAULT_DOT_INPUT_PRECISIONS = ("ieee", "tf32", "tf32x3", "bf16x3", "bf16x6")

# Bump when the code emitted by the fast exp builtin changes.
FAST_EXP_REVISION = 1

_MUSA_BACKEND_NAMES = ("musa", "mtgpu")
_LOG2E = 1.4426950408889634

_DESERIALIZE_MARK = "_torchada_attrs_on_ir_args_only"
_ALIAS_MARK = "_torchada_pointer_alias_spec"
_FAST_EXP_MARK = "_torchada_fast_exp"
_HASH_MARK = "_torchada_hash_salt"


# ---------------------------------------------------------------------------
# Installed Triton description
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MusaTritonInfo:
    """Description of the imported Triton package.

    ``backend`` is the discovered Triton backend: ``"musa"`` (registered through
    the ``triton.backends`` entry point) or ``"mtgpu"`` (in-tree backend
    directory) when present, otherwise the first discovered name or ``None``.
    ``dist_matches_import`` is True when a dist-info whose files include the
    imported ``triton/__init__.py`` was found; ``entry_point_backends``,
    ``commit`` and ``musa_version`` come from that dist-info only.
    """

    version: Optional[str]
    backend: Optional[str]
    backends: Tuple[str, ...]
    entry_point_backends: Tuple[str, ...]
    commit: Optional[str]
    musa_version: Optional[str]
    dist_matches_import: bool

    @property
    def release(self) -> Tuple[int, ...]:
        return _release_tuple(self.version)

    @property
    def is_musa_backend(self) -> bool:
        return self.backend == "musa"

    @property
    def is_musa_triton_36(self) -> bool:
        return self.is_musa_backend and self.release[:2] == (3, 6)


_NO_TRITON = MusaTritonInfo(None, None, (), (), None, None, False)


def _release_tuple(version: Optional[str]) -> Tuple[int, ...]:
    parts = []
    for part in (version or "").split("."):
        digits = ""
        for ch in part:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
        if digits != part:
            break
    return tuple(parts)


def _pick_backend(backends: Iterable[str]) -> Optional[str]:
    names = tuple(backends)
    for name in _MUSA_BACKEND_NAMES:
        if name in names:
            return name
    return names[0] if names else None


def _same_file(lhs: Any, rhs: Any) -> bool:
    try:
        return os.path.realpath(str(lhs)) == os.path.realpath(str(rhs))
    except (OSError, TypeError, ValueError):
        return False


def _triton_distribution_names():
    yield "triton"
    packages_distributions = getattr(importlib.metadata, "packages_distributions", None)
    if packages_distributions is None:
        return
    try:
        names = packages_distributions().get("triton", ())
    except Exception:  # noqa: BLE001 - malformed metadata elsewhere on sys.path
        return
    for name in names:
        if name != "triton":
            yield name


def _find_triton_distribution(triton_file: Optional[str]):
    """Return the distribution that installed ``triton_file``, or None."""
    if not triton_file:
        return None
    for name in _triton_distribution_names():
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        if _same_file(dist.locate_file("triton/__init__.py"), triton_file):
            return dist
    return None


def describe_triton(triton_module: Any, backend_names: Iterable[str]) -> MusaTritonInfo:
    """Build a :class:`MusaTritonInfo` for an imported ``triton`` module."""
    triton_file = getattr(triton_module, "__file__", None)
    dist = _find_triton_distribution(triton_file)
    entry_points: Tuple[str, ...] = ()
    commit = musa_version = None
    if dist is not None:
        entry_points = tuple(
            sorted(ep.name for ep in dist.entry_points if ep.group == "triton.backends")
        )
        for classifier in dist.metadata.get_all("Classifier") or ():
            key, _, value = classifier.partition(":")
            if key.strip() == "Commit ID":
                commit = value.strip() or None
            elif key.strip() == "MUSA Version":
                musa_version = value.strip() or None
    backends = tuple(sorted(backend_names))
    return MusaTritonInfo(
        version=getattr(triton_module, "__version__", None),
        backend=_pick_backend(backends),
        backends=backends,
        entry_point_backends=entry_points,
        commit=commit,
        musa_version=musa_version,
        dist_matches_import=dist is not None,
    )


@functools.lru_cache(maxsize=1)
def musa_triton_info() -> MusaTritonInfo:
    """Describe the imported Triton: version, backend, wheel commit."""
    try:
        import triton
        import triton.backends as triton_backends
    except Exception:  # noqa: BLE001 - absent or broken Triton
        return _NO_TRITON
    backends = getattr(triton_backends, "backends", None) or {}
    return describe_triton(triton, backends.keys())


def triton_backend_is_musa() -> bool:
    """True when the imported Triton compiles for a MUSA backend."""
    return musa_triton_info().backend in _MUSA_BACKEND_NAMES


# ---------------------------------------------------------------------------
# Backend hash salt
# ---------------------------------------------------------------------------


def _musa_backend_class():
    try:
        from triton.backends.musa.compiler import MUSABackend
    except Exception:  # noqa: BLE001 - no MUSA backend in this Triton
        return None
    return MUSABackend


def _salt_state(backend_cls: Any = None) -> Optional[Dict[str, Any]]:
    """Salt state stored on the installed hash wrapper, or None when not installed.

    The state lives on the wrapper rather than in this module so that every
    copy of this module (reloads, alternate import names) shares it.
    """
    cls = backend_cls if backend_cls is not None else _musa_backend_class()
    if cls is None:
        return None
    return getattr(cls.__dict__.get("hash", getattr(cls, "hash", None)), _HASH_MARK, None)


def backend_hash_suffix(backend_cls: Any = None) -> str:
    """Text appended to ``MUSABackend.hash()``; empty when no salt is registered."""
    state = _salt_state(backend_cls)
    return state["suffix"] if state else ""


def _clear_torch_triton_hash_cache() -> None:
    torch_triton = sys.modules.get("torch.utils._triton")
    cached = getattr(torch_triton, "triton_hash_with_backend", None)
    cache_clear = getattr(cached, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def install_backend_hash_salt(backend_cls: Any = None) -> bool:
    """Wrap ``MUSABackend.hash`` so registered salts reach every Triton cache key.

    With no salt registered the wrapper returns the original hash unchanged.
    Returns True when the wrapper is in place.
    """
    cls = backend_cls if backend_cls is not None else _musa_backend_class()
    if cls is None:
        return False
    current = cls.__dict__.get("hash", getattr(cls, "hash", None))
    if current is None:
        return False
    if getattr(current, _HASH_MARK, None) is not None:
        return True
    state: Dict[str, Any] = {"salts": {}, "suffix": ""}

    def hash(self):  # noqa: A001 - mirrors the backend method name
        base = current(self)
        suffix = state["suffix"]
        return base + suffix if suffix else base

    setattr(hash, _HASH_MARK, state)
    hash.__wrapped__ = current
    hash.__doc__ = getattr(current, "__doc__", None)
    cls.hash = hash
    _clear_torch_triton_hash_cache()
    return True


def add_backend_hash_salt(name: str, version: Any, backend_cls: Any = None) -> bool:
    """Register ``name:version`` in the MUSA backend hash.

    Salts are appended in sorted name order. Registering the same name again
    replaces its version. Use this only for patches whose codegen change is
    invisible to Triton's own cache key. Returns True when the salt is active.
    """
    if not install_backend_hash_salt(backend_cls):
        return False
    state = _salt_state(backend_cls)
    if state is None:
        return False
    salts = state["salts"]
    token = str(version)
    if salts.get(name) == token:
        return True
    salts[name] = token
    state["suffix"] = "-torchada:" + ",".join(f"{k}:{v}" for k, v in sorted(salts.items()))
    _clear_torch_triton_hash_cache()
    return True


# ---------------------------------------------------------------------------
# ASTFunction.deserialize: attributes only on IR arguments
# ---------------------------------------------------------------------------


def ast_function_needs_attr_fix(ast_function: Any) -> bool:
    """True for an ``ASTFunction(ret_types, arg_types, attrs)`` without ``constants``."""
    if ast_function is None:
        return False
    try:
        params = inspect.signature(ast_function.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "attrs" in params and "constants" not in params


def install_deserialize_fix(
    code_generator: Any = None, info: Optional[MusaTritonInfo] = None
) -> bool:
    """Apply argument attributes only to argument paths that own an IR argument.

    ``ASTFunction.deserialize`` sets each path's attributes at the current IR
    argument cursor, but a constexpr path consumes no IR argument, so its
    attributes land on the next argument or past the last one. Callers that
    attach attributes to constexpr paths (torch Inductor's ``generate_ttir``)
    then get a mislabelled signature or an ``IndexError``. Native Triton
    launches never attach attributes to constexpr paths and compile unchanged.

    Only MUSA Triton 3.6 frontends with ``ASTFunction(ret_types, arg_types,
    attrs)`` are patched. Returns True when the fix is in place.
    """
    info = musa_triton_info() if info is None else info
    if not info.is_musa_triton_36:
        return False
    if code_generator is None:
        try:
            from triton.compiler import code_generator
        except Exception:  # noqa: BLE001
            return False
    ast_function = getattr(code_generator, "ASTFunction", None)
    if ast_function is None:
        return False
    if getattr(ast_function.deserialize, _DESERIALIZE_MARK, False):
        return True
    if not ast_function_needs_attr_fix(ast_function):
        return False

    from triton import language
    from triton._utils import apply_with_path, set_iterable_path

    original = ast_function.deserialize

    def deserialize(self, fn):
        def make_template(ty):
            if isinstance(ty, (list, tuple, language.tuple_type)):
                return language.tuple([make_template(x) for x in ty], ty)
            return language.constexpr(None)

        vals = make_template(self.arg_types)
        handles = [fn.args(i) for i in range(fn.get_num_args())]
        cursor = 0

        def build_value(path, ty):
            nonlocal cursor
            first = cursor
            val, cursor = ty._unflatten_ir(handles, cursor)
            if cursor > first:
                for attr_name, attr_val in self.attrs.get(path, []):
                    fn.set_arg_attr(first, attr_name, attr_val)
            set_iterable_path(vals, path, val)

        apply_with_path(self.arg_types, build_value)
        return vals

    setattr(deserialize, _DESERIALIZE_MARK, True)
    deserialize.__wrapped__ = original
    deserialize.__doc__ = original.__doc__
    ast_function.deserialize = deserialize
    return True


# ---------------------------------------------------------------------------
# inplace_alias_pairs: number IR arguments, not parameters
# ---------------------------------------------------------------------------


def inplace_alias_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Selected ``TORCHADA_TRITON_INPLACE_ALIAS`` mode: ``fix`` (default), ``off`` or ``vendor``."""
    env = os.environ if env is None else env
    raw = env.get(INPLACE_ALIAS_ENV, "")
    mode = raw.strip().lower() or "fix"
    if mode not in INPLACE_ALIAS_MODES:
        logger.warning(
            "torchada: ignoring %s=%r; expected one of %s",
            INPLACE_ALIAS_ENV,
            raw,
            ", ".join(INPLACE_ALIAS_MODES),
        )
        return "fix"
    return mode


def _ir_arg_count(ty: Any) -> Optional[int]:
    """Number of IR arguments a specialization type flattens to; None if unknown."""
    if isinstance(ty, (tuple, list)):
        total = 0
        for item in ty:
            count = _ir_arg_count(item)
            if count is None:
                return None
            total += count
        return total
    if ty == "constexpr":
        return 0
    if not isinstance(ty, str) or ty.startswith("tensordesc"):
        return None
    return 1


def make_pointer_alias_spec(specialize_impl: Callable, base_backend: Any) -> Callable:
    """Build a ``_make_pointer_alias_spec`` that numbers real IR arguments.

    The backend pass that consumes ``inplace_alias_pairs`` indexes kernel IR
    arguments. Parameters the launch binder specializes to constexpr (``None``,
    unannotated ints equal to 1, JIT functions) are not IR arguments, and a
    tuple flattens to one IR argument per non-constexpr leaf. Pointers inside
    tuples are not recorded. Common Python values are classified inline; other
    values go through ``specialize_impl``. Any failure yields ``""``, which
    compiles the launch as if no pointer were shared.
    """

    def _make_pointer_alias_spec(params, bound_vals):
        """Return compile-time IR argument alias pairs for runtime-equal pointers."""
        indices_by_ptr: Dict[int, list] = {}
        shared = False
        ir_idx = 0
        try:
            for param, value in zip(params, bound_vals):
                if param.is_constexpr:
                    continue
                annotation = param.annotation_type
                if annotation:
                    if annotation.startswith("*"):
                        data_ptr = getattr(value, "data_ptr", None)
                        ptr = data_ptr() if callable(data_ptr) else 0
                        if ptr:
                            seen = indices_by_ptr.setdefault(ptr, [])
                            shared = shared or bool(seen)
                            seen.append(ir_idx)
                    ir_idx += 1
                    continue
                if value is None:
                    continue
                data_ptr = getattr(value, "data_ptr", None)
                if callable(data_ptr) and hasattr(value, "dtype"):
                    ptr = data_ptr()
                    if ptr:
                        seen = indices_by_ptr.setdefault(ptr, [])
                        shared = shared or bool(seen)
                        seen.append(ir_idx)
                    ir_idx += 1
                    continue
                value_type = type(value)
                if value_type is bool or value_type is float:
                    ir_idx += 1
                    continue
                if value_type is int:
                    if value != 1 or param.do_not_specialize:
                        ir_idx += 1
                    continue
                ty = specialize_impl(
                    base_backend,
                    value,
                    param.is_const,
                    not param.do_not_specialize,
                    not param.do_not_specialize_on_alignment,
                )[0]
                count = _ir_arg_count(ty)
                if count is None:
                    return ""
                ir_idx += count
        except Exception:  # noqa: BLE001 - a cache hint must never fail a launch
            return ""
        if not shared:
            return ""
        return ",".join(
            f"{lhs}:{rhs}"
            for indices in indices_by_ptr.values()
            if len(indices) > 1
            for lhs, rhs in itertools.combinations(indices, 2)
        )

    setattr(_make_pointer_alias_spec, _ALIAS_MARK, "fix")
    return _make_pointer_alias_spec


def make_no_pointer_alias_spec() -> Callable:
    """Build a ``_make_pointer_alias_spec`` that never reports aliasing."""

    def _make_pointer_alias_spec(params, bound_vals):
        """Return no alias pairs."""
        return ""

    setattr(_make_pointer_alias_spec, _ALIAS_MARK, "off")
    return _make_pointer_alias_spec


class _ProbePointer:
    dtype = "float32"

    def data_ptr(self):
        return 4096


def vendor_alias_spec_has_param_numbering(jit_module: Any) -> bool:
    """True when the vendor helper numbers a ``None`` parameter as an IR argument.

    The probe ``(maybe_ptr=None, x_ptr=p, out_ptr=p)`` has IR arguments
    ``x_ptr, out_ptr``; parameter numbering reports ``"1:2"`` instead of
    ``"0:1"``.
    """

    def _probe(maybe_ptr, x_ptr, out_ptr):
        pass

    params = [
        jit_module.KernelParam(i, p, False, False)
        for i, p in enumerate(inspect.signature(_probe).parameters.values())
    ]
    ptr = _ProbePointer()
    return jit_module._make_pointer_alias_spec(params, [None, ptr, ptr]) == "1:2"


def install_inplace_alias_fix(
    mode: str, jit_module: Any = None, info: Optional[MusaTritonInfo] = None
) -> str:
    """Replace ``triton.runtime.jit._make_pointer_alias_spec`` according to ``mode``.

    ``fix`` installs :func:`make_pointer_alias_spec` when the Triton helper
    shows parameter numbering, ``off`` emits no alias pairs, ``vendor`` keeps
    the Triton helper. Only MUSA Triton 3.6 is patched. Returns ``"fix"``,
    ``"off"`` or ``"skip: <reason>"``.
    """
    info = musa_triton_info() if info is None else info
    if not info.is_musa_triton_36:
        return "skip: not MUSA Triton 3.6"
    if jit_module is None:
        import triton.runtime.jit as jit_module
    current = getattr(jit_module, "_make_pointer_alias_spec", None)
    if current is None:
        return "skip: no _make_pointer_alias_spec"
    installed = getattr(current, _ALIAS_MARK, None)
    if installed:
        return installed if installed == mode else f"skip: {installed} already installed"
    if mode == "vendor":
        return "skip: vendor selected"
    if mode == "off":
        replacement = make_no_pointer_alias_spec()
    else:
        try:
            known_bug = vendor_alias_spec_has_param_numbering(jit_module)
        except Exception:  # noqa: BLE001 - the helper contract changed; leave it alone
            known_bug = False
        if not known_bug:
            return "skip: vendor helper not recognized"
        from triton.backends.compiler import BaseBackend

        replacement = make_pointer_alias_spec(jit_module.native_specialize_impl, BaseBackend)
    replacement.__wrapped__ = current
    jit_module._make_pointer_alias_spec = replacement
    return mode


# ---------------------------------------------------------------------------
# tl.exp fast path
# ---------------------------------------------------------------------------


def _env_flag(env: Mapping[str, str], name: str) -> bool:
    return env.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def fast_exp_requested(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when ``TORCHADA_TRITON_FAST_EXP`` enables the fast exp path."""
    return _env_flag(os.environ if env is None else env, FAST_EXP_ENV)


def _make_fast_exp(original_exp: Callable) -> Callable:
    from triton.language import core
    from triton.language.math import _check_dtype

    @core.builtin
    @_check_dtype(dtypes=["fp32", "fp64"])
    def exp(x, _semantic=None):
        x = _semantic.to_tensor(x)
        builder = _semantic.builder
        backend_name = getattr(getattr(builder, "options", None), "backend_name", None)
        if x.type.scalar.name != "fp32" or backend_name != "musa":
            return original_exp(x, _semantic=_semantic)
        log2e = builder.get_fp32(_LOG2E)
        if x.type.is_block():
            log2e = builder.create_splat(x.type.to_ir(builder), log2e)
        return core.tensor(builder.create_exp2(builder.create_fmul(x.handle, log2e)), x.type)

    exp.__doc__ = original_exp.__doc__
    setattr(exp, _FAST_EXP_MARK, True)
    exp.__wrapped__ = original_exp
    return exp


def install_fast_exp(info: Optional[MusaTritonInfo] = None) -> bool:
    """Lower fp32 ``tl.exp`` to ``exp2(x * log2(e))`` on the MUSA backend.

    Rebinds ``triton.language.math.exp``, ``triton.language.exp`` and the
    ``tensor.exp`` member; other dtypes and backends use the original builtin.
    Registers the ``fast-exp`` backend hash salt, since Triton does not hash
    builtins. The salt is registered only once the replacement is built.
    Returns True when the fast path is installed; otherwise logs why and
    leaves Triton and its hash unchanged.
    """
    info = musa_triton_info() if info is None else info
    if not info.is_musa_triton_36:
        logger.info(
            "torchada: ignoring %s; it requires MUSA Triton 3.6 (found %s, backend %s)",
            FAST_EXP_ENV,
            info.version,
            info.backend,
        )
        return False
    try:
        import triton.language as tl
        import triton.language.math as tl_math
        from triton.language import core

        if getattr(tl_math.exp, _FAST_EXP_MARK, False):
            return add_backend_hash_salt("fast-exp", FAST_EXP_REVISION)

        fast_exp = _make_fast_exp(tl_math.exp)
        original_member = core.tensor.exp

        def member(self, _semantic=None):
            if self.type.scalar.name == "fp32":
                return fast_exp(self, _semantic=_semantic)
            return original_member(self, _semantic=_semantic)

        setattr(member, core.TRITON_BUILTIN, True)
        setattr(member, _FAST_EXP_MARK, True)
        member.__wrapped__ = original_member
        member.__doc__ = getattr(original_member, "__doc__", None)
    except Exception as exc:  # noqa: BLE001 - Triton internals differ from 3.6
        logger.warning("torchada: %s not applied: %s: %s", FAST_EXP_ENV, type(exc).__name__, exc)
        return False
    if not add_backend_hash_salt("fast-exp", FAST_EXP_REVISION):
        logger.warning("torchada: %s not applied: no MUSA backend hash to salt", FAST_EXP_ENV)
        return False

    tl_math.exp = fast_exp
    tl.exp = fast_exp
    core.tensor.exp = member
    return True


# ---------------------------------------------------------------------------
# fp32 tl.dot default precision
# ---------------------------------------------------------------------------


def allowed_dot_input_precisions(backend_module: Any = None) -> Tuple[str, ...]:
    """Precisions the MUSA backend accepts for ``tl.dot``."""
    if backend_module is None:
        try:
            from triton.backends.musa import compiler as backend_module
        except Exception:  # noqa: BLE001
            backend_module = None
    options = getattr(backend_module, "MUSAOptions", None)
    allowed = getattr(options, "allowed_dot_input_precisions", None)
    if isinstance(allowed, (tuple, list)) and all(isinstance(p, str) for p in allowed):
        return tuple(allowed)
    return DEFAULT_DOT_INPUT_PRECISIONS


def resolve_f32_default(
    env: Optional[Mapping[str, str]] = None, allowed: Iterable[str] = DEFAULT_DOT_INPUT_PRECISIONS
) -> Optional[str]:
    """Value to export as ``TRITON_F32_DEFAULT``, or None to leave it unset.

    ``TRITON_F32_DEFAULT`` set by the user always wins. An unsupported
    ``TORCHADA_TRITON_F32_DEFAULT`` is refused with a warning, since Triton
    would reject every fp32 ``tl.dot`` that uses the default precision. A
    value other than ``ieee`` is logged as a warning because Triton applies it
    to dots that pass ``allow_tf32=False`` as well.
    """
    env = os.environ if env is None else env
    raw = env.get(F32_DEFAULT_ENV, "")
    value = raw.strip().lower()
    if not value:
        return None
    if TRITON_F32_DEFAULT_ENV in env:
        logger.info(
            "torchada: %s is set; ignoring %s=%r", TRITON_F32_DEFAULT_ENV, F32_DEFAULT_ENV, raw
        )
        return None
    allowed = tuple(allowed)
    if value not in allowed:
        logger.warning(
            "torchada: ignoring %s=%r; the MUSA backend accepts %s",
            F32_DEFAULT_ENV,
            raw,
            ", ".join(allowed),
        )
        return None
    if value != "ieee":
        logger.warning(
            "torchada: %s=%s also applies to fp32 tl.dot calls that pass allow_tf32=False",
            F32_DEFAULT_ENV,
            value,
        )
    return value


def install_f32_default(
    env: Optional[Mapping[str, str]] = None, info: Optional[MusaTritonInfo] = None
) -> Optional[str]:
    """Export ``TRITON_F32_DEFAULT`` from ``TORCHADA_TRITON_F32_DEFAULT``.

    Triton gives ``TRITON_F32_DEFAULT`` precedence over ``allow_tf32=`` in
    ``tl.dot``, so the value applies to every fp32 dot without an explicit
    ``input_precision=``, including Inductor's fp32 mm templates. It is one of
    Triton's cache-invalidating environment variables, and child processes
    inherit it; Inductor's autotune caches do not key on it. The in-process
    kernel cache does not key on it either, so it is set once, before the first
    compilation. Applies to any Triton with the ``musa`` backend. Returns the
    exported value, or None when nothing was set.
    """
    info = musa_triton_info() if info is None else info
    target = os.environ if env is None else env
    if not info.is_musa_backend:
        if target.get(F32_DEFAULT_ENV, "").strip():
            logger.info(
                "torchada: ignoring %s; it requires the musa Triton backend (found %s)",
                F32_DEFAULT_ENV,
                info.backend,
            )
        return None
    value = resolve_f32_default(target, allowed_dot_input_precisions())
    if value is not None:
        target[TRITON_F32_DEFAULT_ENV] = value
    return value
