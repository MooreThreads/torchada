"""Device-type dispatch for torch_musa's FSDP2 (``fully_shard``) replacements.

torch_musa replaces parts of PyTorch's FSDP2 with versions that create MUSA streams
whatever the device mesh is, so ``fully_shard`` on a non-MUSA mesh, for example a CPU
mesh on gloo, fails in the root pre-forward. ``install()`` keeps torch_musa's functions
for MUSA meshes and runs PyTorch's own for every other mesh:

* ``torch.distributed.fsdp.fully_shard`` sends a non-MUSA mesh straight to PyTorch's
  ``fully_shard``, so torch_musa's lazy FSDP2 setup and custom communication never run
  for it.
* Each FSDP2 method that torch_musa replaces dispatches per call on its device type:
  ``"musa"`` runs torch_musa's function, any other device type runs PyTorch's.

PyTorch's own methods are rebuilt from the installed source of their module when
torch_musa has already replaced them. Nothing is installed unless all of them can be
recovered and verified against the loaded code.

Only the ``torch.distributed.fsdp.fully_shard`` attribute is routed. A reference to
torch_musa's wrapper bound before ``import torchada`` (``from torch.distributed.fsdp
import fully_shard``) still runs torch_musa's lazy FSDP2 setup and custom communication
for every mesh, so bind ``fully_shard`` after importing torchada.
"""

import ast
import functools
import inspect
import logging
import os
import sys
import tokenize
from types import CodeType, FunctionType, ModuleType
from typing import Any, Callable, Dict, Iterable, List, NamedTuple, Optional, Set, Tuple

import torch

logger = logging.getLogger(__name__)

_MUSA = "musa"

_FSDP_API = "torch.distributed.fsdp"
_FULLY_SHARD_PKG = "torch.distributed.fsdp._fully_shard"
_PARAM_GROUP = "torch.distributed.fsdp._fully_shard._fsdp_param_group"
_STATE = "torch.distributed.fsdp._fully_shard._fsdp_state"
_COLLECTIVES = "torch.distributed.fsdp._fully_shard._fsdp_collectives"
_MUSA_PATCH = "torch_musa.distributed._composable.fsdp.patch"
_MUSA_OVERLAP = "torch_musa.distributed._composable.fsdp.custom_overlap_patch"

# Attribute on a dispatcher holding PyTorch's function; its __wrapped__ is torch_musa's.
UPSTREAM_ATTR = "_torchada_upstream"
# Attribute marking the fully_shard router.
ROUTER_ATTR = "_torchada_fsdp2_router"


class Target(NamedTuple):
    module: str
    cls: str
    name: str
    # Device type of the call, read from the method's own arguments.
    device_type: Callable[..., str]
    # torch_musa module globals that torch_musa assigns to the method on its first
    # fully_shard call.
    hooks: Tuple[Tuple[str, str], ...]


TARGETS = (
    Target(
        _PARAM_GROUP,
        "FSDPCommContext",
        "lazy_init",
        lambda self, device, *args, **kwargs: device.type,
        ((_MUSA_OVERLAP, "comm_context_lazy_init"),),
    ),
    Target(
        _PARAM_GROUP,
        "FSDPParamGroup",
        "wait_for_unshard",
        lambda self, *args, **kwargs: self.device.type,
        ((_MUSA_PATCH, "wait_for_unshard_non_overlap"),),
    ),
    Target(
        _PARAM_GROUP,
        "FSDPParamGroup",
        "post_backward",
        lambda self, *args, **kwargs: self.device.type,
        ((_MUSA_PATCH, "post_backward_non_overlap"),),
    ),
    Target(
        _STATE,
        "FSDPState",
        "_root_post_backward_final_callback",
        lambda self, *args, **kwargs: self._device.type,
        (),
    ),
)


class _Resolved(NamedTuple):
    target: Target
    cls: type
    upstream: FunctionType


_ClassSource = Optional[Tuple[ast.ClassDef, CodeType]]


def _unwrap(fn: Any) -> Any:
    try:
        return inspect.unwrap(fn)
    except ValueError:
        return fn


def _globals_name(fn: Any) -> str:
    return getattr(fn, "__globals__", {}).get("__name__", "")


def _is_torch_musa_name(name: str) -> bool:
    return name == "torch_musa" or name.startswith("torch_musa.")


def owner(fn: Any) -> str:
    """Name of the module whose namespace ``fn`` runs in, wrappers unwrapped.

    ``__module__`` is not used: ``functools.wraps`` copies it from the wrapped function.
    """
    return _globals_name(_unwrap(fn))


def is_dispatch(fn: Any) -> bool:
    return hasattr(fn, UPSTREAM_ATTR)


def is_torch_musa(fn: Any) -> bool:
    return isinstance(fn, FunctionType) and _is_torch_musa_name(owner(fn))


def _class_source(module: ModuleType, class_name: str) -> _ClassSource:
    """The class's definition and body code, compiled from the module's source file."""
    path = getattr(module, "__file__", None)
    if not isinstance(path, str) or not path.endswith(".py"):
        return None
    try:
        with tokenize.open(path) as f:
            source = f.read()
        tree = ast.parse(source, path)
        code = compile(source, path, "exec", dont_inherit=True)
    except (OSError, SyntaxError, TypeError, ValueError):
        logger.debug("cannot compile the source of %s", module.__name__, exc_info=True)
        return None
    defs = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name]
    codes = [c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == class_name]
    if len(defs) != 1 or len(codes) != 1:
        return None
    return defs[0], codes[0]


def _matches_loaded(module: ModuleType, cls: type, class_code: CodeType) -> bool:
    """Every method of ``cls`` still owned by ``module`` compiles to its loaded code."""
    codes = {c.co_name: c for c in class_code.co_consts if isinstance(c, CodeType)}
    checked = 0
    for name, value in vars(cls).items():
        if not isinstance(value, FunctionType) or owner(value) != module.__name__:
            continue
        if codes.get(name) != _unwrap(value).__code__:
            return False
        checked += 1
    return checked > 0


def _names(code: CodeType) -> Set[str]:
    names = set(code.co_names)
    for const in code.co_consts:
        if isinstance(const, CodeType):
            names |= _names(const)
    return names


def _torch_musa_dir() -> Optional[str]:
    path = getattr(sys.modules.get("torch_musa"), "__file__", None)
    if not isinstance(path, str):
        return None
    return os.path.dirname(os.path.abspath(path)) + os.sep


def _defined_by_torch_musa(value: Any, musa_dir: Optional[str]) -> bool:
    if isinstance(value, type):
        return _is_torch_musa_name(getattr(value, "__module__", None) or "")
    if not isinstance(value, FunctionType):
        return False
    for fn in (value, _unwrap(value)):
        if _is_torch_musa_name(_globals_name(fn)):
            return True
        filename = getattr(getattr(fn, "__code__", None), "co_filename", "")
        if musa_dir and os.path.abspath(filename).startswith(musa_dir):
            return True
    return False


def _torch_musa_global(module: ModuleType, code: CodeType) -> Optional[str]:
    """A global of ``module`` read by ``code`` that torch_musa defines, if any."""
    musa_dir = _torch_musa_dir()
    for name in sorted(_names(code)):
        if _defined_by_torch_musa(module.__dict__.get(name), musa_dir):
            return name
    return None


def upstream_method(
    module: ModuleType,
    cls: type,
    name: str,
    sources: Optional[Dict[Tuple[str, str], _ClassSource]] = None,
) -> Optional[FunctionType]:
    """PyTorch's own ``cls.name``, bound to the live namespace of ``module``.

    Returns the live function while PyTorch's is still installed. Otherwise the method
    is rebuilt from the code object compiled from the module's source, which runs no
    module code and sees the globals the original saw. Returns None if the source is
    missing, does not match the loaded code, or the method cannot be rebuilt exactly.
    """
    if sources is None:
        sources = {}
    key = (module.__name__, cls.__name__)
    if key not in sources:
        sources[key] = _class_source(module, cls.__name__)
    found = sources[key]
    if found is None or getattr(module, cls.__name__, None) is not cls:
        return None
    class_def, class_code = found
    defs = [
        n
        for n in class_def.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    ]
    codes = [c for c in class_code.co_consts if isinstance(c, CodeType) and c.co_name == name]
    if len(defs) != 1 or len(codes) != 1 or not isinstance(defs[0], ast.FunctionDef):
        return None
    fdef, code = defs[0], codes[0]
    args = fdef.args
    if fdef.decorator_list or args.defaults or any(d is not None for d in args.kw_defaults):
        return None
    if code.co_freevars or not _matches_loaded(module, cls, class_code):
        return None
    if _torch_musa_global(module, code) is not None:
        return None
    live = vars(cls).get(name)
    if (
        isinstance(live, FunctionType)
        and live.__code__ == code
        and live.__globals__ is module.__dict__
        and live.__defaults__ is None
        and live.__kwdefaults__ is None
        and live.__closure__ is None
    ):
        return live
    fn = FunctionType(code, module.__dict__, name)
    fn.__qualname__ = f"{cls.__qualname__}.{name}"
    return fn


def dispatch(
    musa_fn: Callable[..., Any],
    upstream_fn: Callable[..., Any],
    device_type: Callable[..., str],
) -> Callable[..., Any]:
    """Call ``musa_fn`` when the call's device type is ``"musa"``, else ``upstream_fn``."""

    # torch_musa's name and module, where _wrap binds the dispatcher in place of
    # musa_fn so that it pickles by reference; the code object (what tracebacks and
    # profilers show) is this one.
    @functools.wraps(musa_fn)
    def dispatcher(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            on_musa = device_type(self, *args, **kwargs) == _MUSA
        except Exception:  # noqa: BLE001
            on_musa = True  # behave as torch_musa does without the dispatch
        if on_musa:
            return musa_fn(self, *args, **kwargs)
        return upstream_fn(self, *args, **kwargs)

    setattr(dispatcher, UPSTREAM_ATTR, upstream_fn)
    return dispatcher


def _wrap(fn: Any, resolved: _Resolved) -> Optional[Callable[..., Any]]:
    """The dispatcher for torch_musa's ``fn``, None if ``fn`` needs none.

    Where ``fn`` is bound under its own name in its module, the dispatcher replaces it
    there too, and later calls return that dispatcher.
    """
    if is_dispatch(fn) or not is_torch_musa(fn):
        return None
    module = sys.modules.get(fn.__module__)
    bound = getattr(module, fn.__qualname__, None)
    if is_dispatch(bound) and bound.__wrapped__ is fn:
        return bound
    wrapped = dispatch(fn, resolved.upstream, resolved.target.device_type)
    if bound is fn:
        setattr(module, fn.__qualname__, wrapped)
    return wrapped


def _wrap_hooks(resolved: List[_Resolved]) -> None:
    for item in resolved:
        for module_name, attr in item.target.hooks:
            module = sys.modules.get(module_name)
            wrapped = _wrap(getattr(module, attr, None), item)
            if wrapped is not None:
                setattr(module, attr, wrapped)


def _wrap_class_attrs(resolved: List[_Resolved]) -> None:
    for item in resolved:
        wrapped = _wrap(vars(item.cls).get(item.target.name), item)
        if wrapped is not None:
            setattr(item.cls, item.target.name, wrapped)


def _replacement_check(
    resolved: List[_Resolved], methods: Dict[type, Set[str]]
) -> Callable[[], None]:
    """Warns, once per method, about torch_musa functions replacing other methods.

    ``methods`` holds the names each class defines in PyTorch's source. A torch_musa
    function under one of those names runs for every device mesh; torch_musa's
    additions under new names are never called by PyTorch.
    """
    targets = {(item.cls, item.target.name) for item in resolved}
    reported: Set[str] = set()

    def check() -> None:
        found = sorted(
            f"{cls.__name__}.{name}"
            for cls, names in methods.items()
            for name, value in vars(cls).items()
            if name in names and (cls, name) not in targets and is_torch_musa(value)
        )
        new = [name for name in found if name not in reported]
        if new:
            reported.update(new)
            logger.warning(
                "torch_musa's replacements of %s apply to FSDP2 meshes of every device type",
                ", ".join(new),
            )

    return check


def _mesh_device_type(mesh: Any) -> str:
    # The device type fully_shard derives from the mesh; "cuda" maps to "musa" when
    # torch.device translates it.
    try:
        return torch.device(mesh.device_type).type
    except Exception:  # noqa: BLE001
        return _MUSA


def _router(
    musa_fully_shard: Callable[..., Any],
    upstream_fully_shard: Callable[..., Any],
    wrap_methods: Callable[[], None],
) -> Callable[..., Any]:
    # The public identity of fully_shard (``.state``, pickling by reference) stays.
    @functools.wraps(musa_fully_shard)
    def fully_shard(*args: Any, **kwargs: Any) -> Any:
        mesh = kwargs.get("mesh")
        if mesh is not None and _mesh_device_type(mesh) != _MUSA:
            wrap_methods()
            return upstream_fully_shard(*args, **kwargs)
        try:
            return musa_fully_shard(*args, **kwargs)
        finally:
            # torch_musa's first call assigns its methods through the hooked globals,
            # which are dispatchers already; this also covers ones assigned otherwise.
            wrap_methods()

    setattr(fully_shard, ROUTER_ATTR, True)
    return fully_shard


def _torch_musa_fully_shard(live: Any, upstream_fully_shard: Any) -> Any:
    """torch_musa's wrapper around ``upstream_fully_shard``, if ``live`` is it."""
    if (
        live is None
        or upstream_fully_shard is None
        or live is upstream_fully_shard
        or getattr(live, ROUTER_ATTR, False)
        or getattr(live, "__wrapped__", None) is not upstream_fully_shard
        or not _is_torch_musa_name(_globals_name(live))
    ):
        return None
    return live


def _detected(
    live_fully_shard: Any,
    upstream_fully_shard: Any,
    classes: Iterable[Tuple[Target, Optional[type]]],
) -> bool:
    if getattr(live_fully_shard, ROUTER_ATTR, False):
        return True
    if _torch_musa_fully_shard(live_fully_shard, upstream_fully_shard) is not None:
        return True
    for target, cls in classes:
        if cls is not None and is_torch_musa(vars(cls).get(target.name)):
            return True
        for module_name, attr in target.hooks:
            if getattr(sys.modules.get(module_name), attr, None) is not None:
                return True
    return False


def _is_upstream_all_gather(fn: Any, collectives: ModuleType) -> bool:
    return (
        isinstance(fn, FunctionType)
        and fn.__name__ == "foreach_all_gather"
        and fn.__globals__ is collectives.__dict__
    )


def _check_all_gather() -> None:
    """Warn unless torch_musa's ``foreach_all_gather`` code swap leaves PyTorch's in place.

    torch_musa assigns ``foreach_all_gather_non_overlap.__code__`` to PyTorch's
    ``foreach_all_gather``. Both are ``torch.no_grad()`` wrappers sharing one code
    object, and the closure keeps PyTorch's body, so the swap changes nothing.
    """
    collectives = sys.modules.get(_COLLECTIVES)
    non_overlap = getattr(sys.modules.get(_MUSA_PATCH), "foreach_all_gather_non_overlap", None)
    gather = getattr(collectives, "foreach_all_gather", None)
    if collectives is None or not isinstance(non_overlap, FunctionType):
        return
    if isinstance(gather, FunctionType) and gather.__code__ is non_overlap.__code__:
        cells = dict(zip(gather.__code__.co_freevars, gather.__closure__ or ()))
        body = cells["func"].cell_contents if "func" in cells else None
        if _is_upstream_all_gather(_unwrap(gather), collectives) and _is_upstream_all_gather(
            body, collectives
        ):
            return
    logger.warning(
        "torch_musa's foreach_all_gather replacement applies to FSDP2 meshes of every "
        "device type"
    )


def install() -> bool:
    """Route torch_musa's FSDP2 replacements by device type.

    Returns True when torch_musa's FSDP2 layer is present and dispatches by device
    type. Reads only modules that are already imported and is idempotent. Changes
    nothing if PyTorch's own methods cannot all be recovered.
    """
    api = sys.modules.get(_FSDP_API)
    package = sys.modules.get(_FULLY_SHARD_PKG)
    modules = {name: sys.modules.get(name) for name in (_PARAM_GROUP, _STATE)}
    if api is None or package is None:
        return False
    upstream_fully_shard = getattr(package, "fully_shard", None)
    live_fully_shard = getattr(api, "fully_shard", None)
    classes = [(t, getattr(modules[t.module], t.cls, None)) for t in TARGETS]
    if not _detected(live_fully_shard, upstream_fully_shard, classes):
        return False

    sources: Dict[Tuple[str, str], _ClassSource] = {}
    resolved: List[_Resolved] = []
    methods: Dict[type, Set[str]] = {}
    for target, cls in classes:
        module = modules[target.module]
        upstream = None
        if module is not None and isinstance(cls, type):
            upstream = upstream_method(module, cls, target.name, sources)
        if upstream is None:
            logger.warning(
                "torch_musa's FSDP2 stays in place for every device mesh: PyTorch's "
                "%s.%s cannot be recovered from its source",
                target.cls,
                target.name,
            )
            return False
        resolved.append(_Resolved(target, cls, upstream))
        found = sources.get((target.module, cls.__name__))
        if found is not None:
            methods[cls] = {c.co_name for c in found[1].co_consts if isinstance(c, CodeType)}

    check_replacements = _replacement_check(resolved, methods)

    def wrap_methods() -> None:
        _wrap_class_attrs(resolved)
        check_replacements()

    _wrap_hooks(resolved)
    wrap_methods()
    musa_fully_shard = _torch_musa_fully_shard(live_fully_shard, upstream_fully_shard)
    if musa_fully_shard is not None:
        api.fully_shard = _router(musa_fully_shard, upstream_fully_shard, wrap_methods)
    elif live_fully_shard is not upstream_fully_shard and not getattr(
        live_fully_shard, ROUTER_ATTR, False
    ):
        logger.warning(
            "torch.distributed.fsdp.fully_shard is not torch_musa's wrapper and is not "
            "routed by device type; bind or wrap fully_shard after importing torchada"
        )
    _check_all_gather()
    return True
