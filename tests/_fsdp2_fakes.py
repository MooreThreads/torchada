"""A fake torch_musa FSDP2 layer for CPU tests of torchada's FSDP2 device dispatch.

The stub ``torch_musa`` modules replace FSDP2 pieces the way torch_musa 2.11 does:

* at import, ``FSDPParamGroup.post_backward`` and
  ``FSDPState._root_post_backward_final_callback``, plus a ``fully_shard`` wrapper on
  ``torch.distributed.fsdp``;
* on the wrapper's first call, ``FSDPCommContext.lazy_init`` (one MUSA stream for all
  four comm streams, whatever the device), the no-overlap ``wait_for_unshard`` and
  ``post_backward``, and the ``foreach_all_gather`` code swap.

The fake MUSA stream waits on another stream the way torch_musa's does, so a CPU mesh
fails with ``'NoneType' object has no attribute 'wait'``; with ``MUSA_VISIBLE_DEVICES``
set to an empty string the fake ``current_stream()`` raises "No MUSA GPUs are
available" instead. The other replacements record their call and run PyTorch's method.

The functions are compiled from source text into the stub modules, so their defining
module is a ``torch_musa`` one, as for the real functions. Not collected by pytest.
"""

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable, Dict, Tuple

FAKE_DIR = os.path.join(os.sep, "torchada-fsdp2-fake", "torch_musa")

PATCH = "torch_musa.distributed._composable.fsdp.patch"
OVERLAP = "torch_musa.distributed._composable.fsdp.custom_overlap_patch"
MUSA_PARAM_GROUP = "torch_musa.distributed.fsdp._fully_shard._fsdp_param_group"
MUSA_STATE = "torch_musa.distributed.fsdp._fully_shard._fsdp_state"

PARAM_GROUP = "torch.distributed.fsdp._fully_shard._fsdp_param_group"
STATE = "torch.distributed.fsdp._fully_shard._fsdp_state"
COLLECTIVES = "torch.distributed.fsdp._fully_shard._fsdp_collectives"

_OVERLAP_SRC = """
import os

from torch.distributed.device_mesh import _get_device_handle
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPCommContext


class FakeMusaEvent:
    def wait(self, stream=None):
        pass


class FakeMusaStream:
    def wait_event(self, event):
        event.wait(self)

    def record_event(self, event=None):
        return FakeMusaEvent() if event is None else event

    def wait_stream(self, stream):
        self.wait_event(stream.record_event())


CURRENT_STREAM = FakeMusaStream()

# Set when torch_musa is imported, whatever the device mesh.
_FSDP2_OVERLAP_LEVEL = "FSDP2OverlapLevel.NO_OVERLAP"


def current_stream():
    if os.environ.get("MUSA_VISIBLE_DEVICES") == "":
        raise RuntimeError("No MUSA GPUs are available")
    return CURRENT_STREAM


def comm_context_lazy_init(self, device):
    STATE.calls.append("comm_context_lazy_init")
    self.device_handle = _get_device_handle(device.type)
    stream = current_stream()
    self.all_gather_copy_in_stream = stream
    self.all_gather_stream = stream
    self.reduce_scatter_stream = stream
    self.all_reduce_stream = stream
    self.all_gather_state = None
    self.reduce_scatter_state = None
    self.post_forward_order = []


def _apply_custom_overlap_patch():
    FSDPCommContext.lazy_init = comm_context_lazy_init


def _maybe_set_custom_comm(fsdp_module):
    STATE.custom_comm += 1
"""

_PATCH_SRC = """
import os
from functools import wraps

import torch
from torch.distributed.fsdp._fully_shard import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


def wait_for_unshard_non_overlap(self):
    STATE.calls.append("wait_for_unshard_non_overlap")
    return UPSTREAM["wait_for_unshard"](self)


def post_backward_non_overlap(self, *unused):
    STATE.calls.append("post_backward_non_overlap")
    return UPSTREAM["post_backward"](self, *unused)


@torch.no_grad()
def foreach_all_gather_non_overlap(
    fsdp_params,
    group,
    async_op,
    all_gather_copy_in_stream,
    all_gather_stream,
    device,
    all_gather_comm,
):
    raise AssertionError("not called: the code swap keeps the original body")


def _get_musa_arch():
    STATE.arch += 1
    return 22 if os.environ.get("MUSA_VISIBLE_DEVICES") == "" else 31


def _setup_fsdp2_patches():
    from .custom_overlap_patch import _apply_custom_overlap_patch

    _apply_custom_overlap_patch()
    _get_musa_arch()
    torch.distributed.fsdp._fully_shard._fsdp_collectives.foreach_all_gather.__code__ = (
        foreach_all_gather_non_overlap.__code__
    )
    FSDPParamGroup.wait_for_unshard = wait_for_unshard_non_overlap
    FSDPParamGroup.post_backward = post_backward_non_overlap


def monkey_patched_fully_shard(fully_shard_func):
    has_patched = False
    from .custom_overlap_patch import _maybe_set_custom_comm

    @wraps(fully_shard_func)
    def wrapper(*args, **kwargs):
        nonlocal has_patched
        if not has_patched:
            _setup_fsdp2_patches()
            has_patched = True
        fsdp_module = fully_shard_func(*args, **kwargs)
        _maybe_set_custom_comm(fsdp_module)
        return fsdp_module

    return wrapper


def _apply_fsdp2_patches():
    torch.distributed.fsdp.fully_shard = monkey_patched_fully_shard(fully_shard)
"""

_MUSA_PARAM_GROUP_SRC = """
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


def post_backward(self, *unused):
    STATE.calls.append("post_backward")
    return UPSTREAM["post_backward"](self, *unused)


def _apply_fsdp_param_group_patch():
    FSDPParamGroup.post_backward = post_backward
"""

_MUSA_STATE_SRC = """
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState


def _root_post_backward_final_callback(self):
    STATE.calls.append("_root_post_backward_final_callback")
    return UPSTREAM["_root_post_backward_final_callback"](self)


def _apply_fsdp_state_patch():
    FSDPState._root_post_backward_final_callback = _root_post_backward_final_callback
"""

# The torch_musa module globals that its first fully_shard call assigns to FSDP2
# methods.
HOOK_GLOBALS = (
    (OVERLAP, "comm_context_lazy_init"),
    (PATCH, "wait_for_unshard_non_overlap"),
    (PATCH, "post_backward_non_overlap"),
)

# The torch_musa module globals that its import assigns to FSDP2 methods.
IMPORT_GLOBALS = (
    (MUSA_PARAM_GROUP, "post_backward"),
    (MUSA_STATE, "_root_post_backward_final_callback"),
)

RENAMED_SUFFIX = "_renamed"


def _sources(renamed: bool, decorated_non_overlap: bool) -> Tuple[Tuple[str, str, str], ...]:
    overlap, patch = _OVERLAP_SRC, _PATCH_SRC
    if renamed:
        for _, name in HOOK_GLOBALS:
            overlap = overlap.replace(name, name + RENAMED_SUFFIX)
            patch = patch.replace(name, name + RENAMED_SUFFIX)
    if not decorated_non_overlap:
        patch = patch.replace("@torch.no_grad()\n", "")
    empty = ""
    return (
        ("torch_musa", "__init__.py", empty),
        ("torch_musa.distributed", "distributed/__init__.py", empty),
        ("torch_musa.distributed._composable", "distributed/_composable/__init__.py", empty),
        (
            "torch_musa.distributed._composable.fsdp",
            "distributed/_composable/fsdp/__init__.py",
            empty,
        ),
        (OVERLAP, "distributed/_composable/fsdp/custom_overlap_patch.py", overlap),
        (PATCH, "distributed/_composable/fsdp/patch.py", patch),
        ("torch_musa.distributed.fsdp", "distributed/fsdp/__init__.py", empty),
        (
            "torch_musa.distributed.fsdp._fully_shard",
            "distributed/fsdp/_fully_shard/__init__.py",
            empty,
        ),
        (
            MUSA_PARAM_GROUP,
            "distributed/fsdp/_fully_shard/_fsdp_param_group.py",
            _MUSA_PARAM_GROUP_SRC,
        ),
        (MUSA_STATE, "distributed/fsdp/_fully_shard/_fsdp_state.py", _MUSA_STATE_SRC),
    )


def upstream_methods() -> Dict[str, Callable]:
    import torch.distributed.fsdp  # noqa: F401

    group_cls = sys.modules[PARAM_GROUP].FSDPParamGroup
    state_cls = sys.modules[STATE].FSDPState
    methods = {
        "wait_for_unshard": vars(group_cls)["wait_for_unshard"],
        "post_backward": vars(group_cls)["post_backward"],
        "_root_post_backward_final_callback": vars(state_cls)["_root_post_backward_final_callback"],
    }
    for name, fn in methods.items():
        owner = getattr(fn, "__globals__", {}).get("__name__", "")
        assert owner.startswith("torch.distributed.fsdp."), f"{name} is already replaced"
    return methods


def build(
    register: Callable[[str, ModuleType], None],
    renamed: bool = False,
    decorated_non_overlap: bool = True,
    apply: bool = True,
) -> SimpleNamespace:
    """Create the fake torch_musa modules and apply its import-time FSDP2 patches.

    ``register(name, module)`` adds each module to ``sys.modules``. ``renamed`` gives
    the functions in ``HOOK_GLOBALS`` different names, ``decorated_non_overlap=False``
    drops ``torch.no_grad()`` from ``foreach_all_gather_non_overlap``, and
    ``apply=False`` leaves torch untouched.
    """
    state = SimpleNamespace(calls=[], arch=0, custom_comm=0)
    upstream = upstream_methods()
    modules: Dict[str, ModuleType] = {}
    for name, relpath, source in _sources(renamed, decorated_non_overlap):
        module = ModuleType(name)
        module.__file__ = os.path.join(FAKE_DIR, relpath)
        if relpath.endswith("__init__.py"):
            module.__path__ = [os.path.dirname(module.__file__)]
            module.__package__ = name
        else:
            module.__package__ = name.rpartition(".")[0]
        module.STATE = state
        module.UPSTREAM = upstream
        register(name, module)
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, module)
        exec(compile(source, module.__file__, "exec", dont_inherit=True), module.__dict__)
        modules[name] = module
    if apply:
        modules[PATCH]._apply_fsdp2_patches()
        modules[MUSA_PARAM_GROUP]._apply_fsdp_param_group_patch()
        modules[MUSA_STATE]._apply_fsdp_state_patch()
    return SimpleNamespace(
        state=state,
        modules=modules,
        patch=modules[PATCH],
        overlap=modules[OVERLAP],
        param_group=modules[MUSA_PARAM_GROUP],
        fsdp_state=modules[MUSA_STATE],
    )
