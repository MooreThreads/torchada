"""Multi-process FSDP2 (``fully_shard``) worker for torchada's FSDP2 dispatch tests.

Run as a script; the tests also import its ``load`` and ``compare`` helpers. Not
collected by pytest. Commands:

``run``
    Spawns ``--world-size`` ranks. For each mesh in ``--mesh``, one after the other,
    shards a small model with ``torch.distributed.fsdp.fully_shard`` and runs two
    ``no_grad`` forwards and ``--steps`` SGD training steps. Each rank writes its
    tensors (``rank<N>.pt``), a structural record (``rank<N>.json``) and its output
    (``rank<N>.log``) to ``--out``; rank 0 also writes the FSDP2 call trace
    (``trace.json``).
``structure``
    Records how FSDP2 is patched in a fresh process, without collectives, as JSON in
    ``--out``/``structure.json``, or on stdout without ``--out``.
``compare A B``
    Compares two ``run`` outputs, prints ``SAME`` or the differences, and exits 0 only
    when they are the same.
``signature FILE``
    Prints the exception that ended a failed ``run``, read from its captured output, or
    nothing when the output names none.

Exit status 2 means the worker could not start a run (bad arguments, or ``--out`` or the
temporary directory for the ranks' rendezvous file cannot be written); a failure inside
a rank exits 1.

Arms (``--arm``):

* ``upstream``: plain PyTorch; torch_musa must not be loaded (on MUSA hosts set
  ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``).
* ``fake`` / ``fake-torchada``: the fake torch_musa layer of ``_fsdp2_fakes``,
  without / with ``torchada._fsdp2.install()``. Needs a host without torch_musa
  loaded; set ``TORCHADA_PLATFORM=cpu TORCH_DEVICE_BACKEND_AUTOLOAD=0``.
* ``torch-musa`` / ``torchada``: the real torch_musa, without / with ``import torchada``.

``--early-bound`` binds ``fully_shard`` before torchada's dispatch is installed, as a
consumer that imported it first does.
"""

import argparse
import contextlib
import inspect
import json
import os
import re
import shutil
import sys
import tempfile
from types import FunctionType
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp

ARMS = ("upstream", "fake", "fake-torchada", "torch-musa", "torchada")
DIM = 8
COMM_STREAMS = (
    "all_gather_copy_in_stream",
    "all_gather_stream",
    "reduce_scatter_stream",
    "all_reduce_stream",
)
PARAM_GROUP = "torch.distributed.fsdp._fully_shard._fsdp_param_group"
STATE = "torch.distributed.fsdp._fully_shard._fsdp_state"
FULLY_SHARD = "torch.distributed.fsdp._fully_shard._fully_shard"
MUSA_PATCH = "torch_musa.distributed._composable.fsdp.patch"
MUSA_OVERLAP = "torch_musa.distributed._composable.fsdp.custom_overlap_patch"
TARGETS = (
    (PARAM_GROUP, "FSDPCommContext", "lazy_init"),
    (PARAM_GROUP, "FSDPParamGroup", "wait_for_unshard"),
    (PARAM_GROUP, "FSDPParamGroup", "post_backward"),
    (STATE, "FSDPState", "_root_post_backward_final_callback"),
)
HOOK_GLOBALS = (
    (MUSA_OVERLAP, "comm_context_lazy_init"),
    (MUSA_PATCH, "wait_for_unshard_non_overlap"),
    (MUSA_PATCH, "post_backward_non_overlap"),
)
# torch_musa additions to FSDP2 classes that the dispatch leaves alone.
TORCH_MUSA_ADDITIONS = (
    "post_backward_final_lazy_hsdp_all_reduce",
    "post_backward_final_lazy_hsdp_update_grads",
    "set_lazy_hsdp_allreduce",
)
TRACE_DIRS = ("/torch/distributed/fsdp/_fully_shard/", "/torch/cpu/", "/torch_musa/")
TRACE_SKIP = {"<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>", "<lambda>", "_with_fqn"}
IMPORTS_DONE = "fsdp2-worker: imports done"
STRUCTURE_FILE = "structure.json"


# ----------------------------------------------------------------------------- model


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.proj(x)) + x


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Linear(DIM, DIM)  # gives the root its own param group
        self.layers = torch.nn.ModuleList(Block() for _ in range(2))
        self.head = torch.nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def _inputs() -> torch.Tensor:
    return torch.randn(4, DIM, generator=torch.Generator().manual_seed(1))


# ---------------------------------------------------------------------- inspection


def _owner(fn: Any) -> str:
    return getattr(inspect.unwrap(fn), "__globals__", {}).get("__name__", "")


def _short(path: str) -> str:
    path = path.replace(os.sep, "/")
    for marker in ("/torch_musa/", "/torch/"):
        index = path.rfind(marker)
        if index >= 0:
            return path[index + 1 :]
    return path


def _describe(fn: Any) -> Optional[List[Any]]:
    fn = inspect.unwrap(fn) if callable(fn) else fn
    code = getattr(fn, "__code__", None)
    if code is None:
        return None
    return [fn.__module__, fn.__qualname__, _short(code.co_filename), code.co_firstlineno]


def _effective(attr: Any, device_type: str) -> Any:
    """The function a call on ``device_type`` runs for this class attribute."""
    upstream = getattr(attr, "_torchada_upstream", None)
    if upstream is None:
        return attr
    return attr.__wrapped__ if device_type == "musa" else upstream


def _class(module: str, name: str) -> type:
    return getattr(sys.modules[module], name)


def _torch_musa_owned() -> Dict[str, List[str]]:
    classes = {
        "FSDPCommContext": _class(PARAM_GROUP, "FSDPCommContext"),
        "FSDPParamGroup": _class(PARAM_GROUP, "FSDPParamGroup"),
        "FSDPState": _class(STATE, "FSDPState"),
        "FSDPModule": _class(FULLY_SHARD, "FSDPModule"),
    }
    return {
        name: sorted(
            attr
            for attr, value in vars(cls).items()
            if isinstance(value, FunctionType) and _owner(value).startswith("torch_musa")
        )
        for name, cls in classes.items()
    }


def _overlap_level() -> Optional[str]:
    module = sys.modules.get(MUSA_OVERLAP)
    if module is None or not hasattr(module, "_FSDP2_OVERLAP_LEVEL"):
        return None
    return str(module._FSDP2_OVERLAP_LEVEL)


def _musa_initialized() -> Optional[bool]:
    torch_musa = sys.modules.get("torch_musa")
    is_initialized = getattr(torch_musa, "is_initialized", None)
    return bool(is_initialized()) if callable(is_initialized) else None


def _fake_state() -> Optional[Dict[str, Any]]:
    state = getattr(sys.modules.get(MUSA_PATCH), "STATE", None)
    if state is None:
        return None
    return {"calls": list(state.calls), "arch": state.arch, "custom_comm": state.custom_comm}


# --------------------------------------------------------------------------- tracing


class _Tracer:
    def __init__(self) -> None:
        self.events: List[str] = []
        self._depth = 0

    def __call__(self, frame: Any, event: str, arg: Any) -> None:
        code = frame.f_code
        filename = code.co_filename.replace(os.sep, "/")
        if code.co_name in TRACE_SKIP or not any(d in filename for d in TRACE_DIRS):
            return
        if event == "call":
            line = f"{_short(filename)}:{code.co_firstlineno} {code.co_name}"
            self.events.append("  " * self._depth + line)
            self._depth += 1
        elif event == "return":
            self._depth -= 1

    @contextlib.contextmanager
    def section(self, enabled: bool):
        if not enabled:
            yield None
            return
        self.events, self._depth = [], 0
        sys.setprofile(self)
        try:
            yield self.events
        finally:
            sys.setprofile(None)


# --------------------------------------------------------------------------- setup


def _setup(arm: str, early_bound: bool) -> Callable[..., Any]:
    """Import the arm's layers and return how the consumer calls ``fully_shard``."""
    bound = None
    if arm == "upstream":
        assert "torch_musa" not in sys.modules, "torch_musa is loaded; set the autoload off"
        import torch.distributed.fsdp  # noqa: F401
    elif arm in ("fake", "fake-torchada"):
        assert "torch_musa" not in sys.modules, "a real torch_musa is loaded"
        import _fsdp2_fakes

        _fsdp2_fakes.build(sys.modules.__setitem__)
        if early_bound:
            from torch.distributed.fsdp import fully_shard as bound
        if arm == "fake-torchada":
            from torchada import _fsdp2

            assert _fsdp2.install()
    elif arm == "torch-musa":
        import torch.distributed.fsdp  # noqa: F401
        import torch_musa  # noqa: F401

        assert "torchada" not in sys.modules
    elif arm == "torchada":
        if early_bound:
            import torch_musa  # noqa: F401
            from torch.distributed.fsdp import fully_shard as bound
        import torch.distributed.fsdp  # noqa: F401

        import torchada  # noqa: F401
    else:
        raise ValueError(arm)
    if bound is not None:
        return bound
    return lambda *args, **kwargs: sys.modules["torch.distributed.fsdp"].fully_shard(
        *args, **kwargs
    )


def _backend(meshes: List[str]) -> str:
    on_device = [m for m in meshes if m != "cpu"]
    if not on_device:
        return "gloo"
    if len(on_device) == len(meshes):
        return "mccl"
    return "cpu:gloo,musa:mccl"


@contextlib.contextmanager
def _redirect_output(path: Optional[str]):
    if path is None:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(1), os.dup(2)]
    with open(path, "ab", buffering=0) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            for fd in saved:
                os.close(fd)


# ------------------------------------------------------------------------- workload


def _shard(fully_shard: Callable[..., Any], model: Model, mesh_type: str, shape: List[int]):
    from torch.distributed import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    names = ("replicate", "shard") if len(shape) == 2 else ("shard",)
    mesh = init_device_mesh(mesh_type, mesh_shape=tuple(shape), mesh_dim_names=names)
    kwargs = dict(
        mesh=mesh,
        reshard_after_forward=True,
        mp_policy=MixedPrecisionPolicy(cast_forward_inputs=False),
    )
    for layer in model.layers:
        fully_shard(layer, **kwargs)
    fully_shard(model, **kwargs)


def _full(tensor: Any) -> torch.Tensor:
    tensor = tensor.full_tensor() if hasattr(tensor, "full_tensor") else tensor
    return tensor.detach().cpu().clone()


def _run_model(model: Model, steps: int, reference: Optional[Model]) -> Dict[str, Any]:
    device = model._get_fsdp_state()._device
    x = _inputs().to(device)
    with torch.no_grad():
        outputs = [model(x) for _ in range(2)]
    if reference is not None:
        with torch.no_grad():
            expected = reference(_inputs())
        for out in outputs:
            assert torch.equal(out.cpu(), expected), (out.cpu() - expected).abs().max()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    grads = []
    for step in range(steps):
        model(x).square().sum().backward()
        grads.append({n: _full(p.grad) for n, p in model.named_parameters()})
        if step == 0 and reference is not None:
            reference(_inputs()).square().sum().backward()
            for name, param in reference.named_parameters():
                torch.testing.assert_close(grads[0][name], param.grad, msg=name)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return {
        "outputs": [out.detach().cpu() for out in outputs],
        "grads": grads,
        "params": {n: _full(p) for n, p in model.named_parameters()},
    }


def _structure(model: Model, label: str) -> Dict[str, Any]:
    state = model._get_fsdp_state()
    device_type = state._device.type
    streams = [getattr(state._comm_ctx, name) for name in COMM_STREAMS]
    group = state._fsdp_param_group
    functions = {}
    for module, cls_name, name in TARGETS:
        attr = vars(_class(module, cls_name))[name]
        functions[f"{cls_name}.{name}"] = _describe(_effective(attr, device_type))
    return {
        "label": label,
        "device_type": device_type,
        "device_handle": getattr(state._device_handle, "__name__", None),
        "stream_types": [f"{type(s).__module__}.{type(s).__qualname__}" for s in streams],
        "stream_aliases": [next(i for i, t in enumerate(streams) if t is s) for s in streams],
        "stream_priorities": [getattr(s, "priority", None) for s in streams],
        "functions": functions,
        "all_gather_comm": type(getattr(group, "_all_gather_comm", None)).__name__,
        "reduce_scatter_comm": type(getattr(group, "_reduce_scatter_comm", None)).__name__,
    }


def _rank_main(rank: int, args: argparse.Namespace, init_method: str) -> None:
    log = os.path.join(args.out, f"rank{rank}.log") if args.out else None
    with _redirect_output(log):
        _rank_body(rank, args, init_method)


def _rank_body(rank: int, args: argparse.Namespace, init_method: str) -> None:
    torch.set_num_threads(1)
    meshes = args.mesh.split(",")
    shape = [int(n) for n in args.shape.split(",")]
    fully_shard = _setup(args.arm, args.early_bound)
    print(IMPORTS_DONE, file=sys.stderr, flush=True)
    record: Dict[str, Any] = {"arm": args.arm, "rank": rank, "mesh": meshes}
    initialized = {"after_imports": _musa_initialized()}

    import torch.distributed as dist

    if any(m != "cpu" for m in meshes):
        torch.musa.set_device(rank)
    dist.init_process_group(
        _backend(meshes), init_method=init_method, rank=rank, world_size=args.world_size
    )
    tracer = _Tracer()
    traces: Dict[str, List[str]] = {}
    tensors, structures = [], []
    try:
        # One mesh after the other: shard, run, record.
        for index, mesh_type in enumerate(meshes):
            torch.manual_seed(0)
            model = Model()
            reference = None
            if mesh_type == "cpu":
                reference = Model()
                reference.load_state_dict(model.state_dict())
            with tracer.section(rank == 0) as events:
                _shard(fully_shard, model, mesh_type, shape)
            if events is not None:
                traces[f"{index}:shard"] = events
            initialized[f"{index}:after_shard"] = _musa_initialized()
            with tracer.section(rank == 0) as events:
                tensors.append(_run_model(model, args.steps, reference))
            if events is not None:
                traces[f"{index}:run"] = events
            initialized[f"{index}:after_run"] = _musa_initialized()
            structures.append(_structure(model, mesh_type))
        dist.barrier()
    finally:
        dist.destroy_process_group()
    record.update(
        meshes=structures,
        musa_initialized=initialized,
        overlap_level=_overlap_level(),
        torch_musa_owned=_torch_musa_owned(),
        fake=_fake_state(),
    )
    if args.out:
        torch.save(tensors, os.path.join(args.out, f"rank{rank}.pt"))
        with open(os.path.join(args.out, f"rank{rank}.json"), "w") as f:
            json.dump(record, f, indent=1, sort_keys=True)
        if rank == 0:
            with open(os.path.join(args.out, "trace.json"), "w") as f:
                json.dump(traces, f, indent=0)
    print(f"[rank {rank}] PASS", flush=True)


def _probe(directory: str) -> None:
    """Write, flush and remove a file in ``directory``; raises OSError if that fails."""
    path = os.path.join(directory, ".fsdp2-worker-probe")
    with open(path, "wb") as f:
        f.write(b"\0" * 4096)
        f.flush()
        os.fsync(f.fileno())
    os.remove(path)


def _run(args: argparse.Namespace) -> int:
    temp_dir = None
    try:
        if args.out:
            os.makedirs(args.out, exist_ok=True)
            _probe(args.out)
        temp_dir = tempfile.mkdtemp()
        _probe(temp_dir)
    except OSError as exc:
        # Otherwise the ranks fail writing their output or wait on the rendezvous file.
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"fsdp2-worker: cannot write the output or rendezvous files: {exc}", file=sys.stderr)
        return 2
    try:
        init_method = "file://" + os.path.join(temp_dir, "rendezvous")
        mp.spawn(_rank_main, args=(args, init_method), nprocs=args.world_size, join=True)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    print("PASS", flush=True)
    return 0


# ------------------------------------------------------------------------ structure


def _compiled_method(module_name: str, cls_name: str, name: str) -> Any:
    module = sys.modules[module_name]
    with open(module.__file__, "rb") as f:
        code = compile(f.read(), module.__file__, "exec", dont_inherit=True)
    (cls_code,) = [c for c in code.co_consts if getattr(c, "co_name", None) == cls_name]
    (method,) = [c for c in cls_code.co_consts if getattr(c, "co_name", None) == name]
    return method


def _structure_record(args: argparse.Namespace) -> Dict[str, Any]:
    api = sys.modules["torch.distributed.fsdp"]
    package = sys.modules["torch.distributed.fsdp._fully_shard"]
    router = api.fully_shard
    wrapped = getattr(router, "__wrapped__", None)
    record: Dict[str, Any] = {
        "router": bool(getattr(router, "_torchada_fsdp2_router", False)),
        "router_has_state": hasattr(router, "state"),
        "router_wraps_torch_musa": getattr(wrapped, "__globals__", {})
        .get("__name__", "")
        .startswith("torch_musa."),
        "router_chain_to_upstream": getattr(wrapped, "__wrapped__", None) is package.fully_shard,
        "targets": {},
        "hook_globals": {},
        "torch_musa_owned": _torch_musa_owned(),
        "device_types": {},
    }
    for module, cls_name, name in TARGETS:
        attr = vars(_class(module, cls_name))[name]
        upstream = getattr(attr, "_torchada_upstream", None)
        record["targets"][f"{cls_name}.{name}"] = {
            "dispatch": upstream is not None,
            "torch_musa": _describe(attr.__wrapped__) if upstream is not None else None,
            "upstream": _describe(upstream) if upstream is not None else _describe(attr),
            "upstream_matches_source": (upstream or attr).__code__
            == _compiled_method(module, cls_name, name),
        }
    for module, name in HOOK_GLOBALS:
        value = getattr(sys.modules.get(module), name, None)
        record["hook_globals"][f"{module}.{name}"] = {
            "present": value is not None,
            "dispatch": hasattr(value, "_torchada_upstream"),
        }
    from types import SimpleNamespace

    from torch.distributed.fsdp._fully_shard._fsdp_init import _get_device_from_mesh

    for device_type in args.device_types.split(","):
        mesh = SimpleNamespace(device_type=device_type)
        record["device_types"][device_type] = [
            torch.device(device_type).type,
            _get_device_from_mesh(mesh).type,
        ]
    return record


def _structure_cmd(args: argparse.Namespace) -> int:
    import logging

    messages: List[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = record.getMessage()
            if record.name == "torchada._fsdp2" or "FSDP2" in message:
                messages.append(f"{record.name}: {message}")

    handler = _Collect(level=logging.WARNING)
    logging.getLogger("torchada").addHandler(handler)
    _setup(args.arm, early_bound=False)
    result = {"before": _structure_record(args)}
    if args.after_lazy_setup:
        sys.modules[MUSA_PATCH]._setup_fsdp2_patches()
        result["after_lazy_setup"] = _structure_record(args)
    result["torchada_warnings"] = messages
    text = json.dumps(result, indent=1, sort_keys=True)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, STRUCTURE_FILE), "w") as f:
            f.write(text)
    else:
        print(text)
    return 0


# -------------------------------------------------------------------------- compare


def load(out: str) -> Tuple[List[Dict[str, Any]], List[Any], Dict[str, List[str]]]:
    """Records, tensors and the rank-0 trace of a ``run`` output directory."""
    records, tensors = [], []
    rank = 0
    while os.path.exists(os.path.join(out, f"rank{rank}.json")):
        with open(os.path.join(out, f"rank{rank}.json")) as f:
            records.append(json.load(f))
        tensors.append(torch.load(os.path.join(out, f"rank{rank}.pt")))
        rank += 1
    with open(os.path.join(out, "trace.json")) as f:
        trace = json.load(f)
    return records, tensors, trace


def load_structure(out: str) -> Dict[str, Any]:
    """The record of a ``structure --out`` run."""
    with open(os.path.join(out, STRUCTURE_FILE)) as f:
        return json.load(f)


def tensor_diffs(a: Any, b: Any, exact: bool, path: str = "") -> List[str]:
    if isinstance(a, torch.Tensor):
        same = torch.equal(a, b) if exact else torch.allclose(a, b, rtol=1e-5, atol=1e-6)
        return [] if same else [f"{path}: max |a-b| {(a - b).abs().max().item():.3g}"]
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return [f"{path}: keys differ"]
        return [d for k in a for d in tensor_diffs(a[k], b[k], exact, f"{path}.{k}")]
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: lengths differ"]
        return [
            d
            for i, (x, y) in enumerate(zip(a, b))
            for d in tensor_diffs(x, y, exact, f"{path}[{i}]")
        ]
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def structural(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{k: v for k, v in mesh.items() if k != "label"} for mesh in record["meshes"]]


def compare(a: str, b: str, exact: bool = True, trace: str = "all") -> List[str]:
    """Differences between two ``run`` outputs.

    ``trace`` selects the trace sections to compare: ``all``, ``shard``, ``run`` or
    ``none``. Mesh labels are ignored, so a "cuda" mesh compares with a "musa" one.
    torch_musa's overlap level is compared only when both runs loaded torch_musa.
    """
    records_a, tensors_a, trace_a = load(a)
    records_b, tensors_b, trace_b = load(b)
    if len(records_a) != len(records_b):
        return [f"world sizes differ: {len(records_a)} != {len(records_b)}"]
    diffs = []
    for rank, (ra, rb) in enumerate(zip(records_a, records_b)):
        if structural(ra) != structural(rb):
            diffs.append(f"rank {rank}: structure {structural(ra)} != {structural(rb)}")
        levels = (ra["overlap_level"], rb["overlap_level"])
        if None not in levels and levels[0] != levels[1]:
            diffs.append(f"rank {rank}: overlap {ra['overlap_level']} != {rb['overlap_level']}")
        diffs += tensor_diffs(tensors_a[rank], tensors_b[rank], exact, f"rank{rank}")
    if trace != "none":
        keys = sorted(set(trace_a) | set(trace_b))
        for key in keys:
            if trace in ("all", key.split(":")[-1]) and trace_a.get(key) != trace_b.get(key):
                diffs.append(f"trace {key} differs")
    return diffs


_TRACEBACK = "Traceback (most recent call last):"
_CHAINED = (
    "During handling of the above exception, another exception occurred:",
    "The above exception was the direct cause of the following exception:",
)
_RANK_ERROR = re.compile(r"^-- Process \d+ terminated with the following error:\s*$", re.MULTILINE)
_EXCEPTION_LINE = re.compile(
    r"^(?:[A-Za-z_][\w.]*)?(?:Error|Exception|Interrupt|Exit)(?::.*)?$", re.MULTILINE
)


def _final_exceptions(text: str) -> List[str]:
    """The exception line that ends each traceback in ``text``, one per exception chain.

    The exception line is the first unindented line after the traceback's frames, so a
    message continued on further lines is cut to its first line. A chained traceback
    replaces the one it follows. Tracebacks printed after ``Exception ignored in ...``
    (errors during interpreter shutdown) are left out.
    """
    lines = text.splitlines()
    found: List[str] = []
    previous, ignored = "", False
    for index, line in enumerate(lines):
        if line.rstrip() == _TRACEBACK:
            chained = previous in _CHAINED
            if not chained:
                ignored = previous.startswith("Exception ignored")
            unindented = (
                t.rstrip() for t in lines[index + 1 :] if t.strip() and not t[0].isspace()
            )
            exception = next(unindented, None)
            if exception is not None and not ignored:
                if chained and found:
                    found[-1] = exception
                else:
                    found.append(exception)
        if line.strip():
            previous = line.strip()
    return found


def failure_signature(text: str) -> Optional[str]:
    """The exception that ended a failed ``run``, from its captured output, or None.

    A rank's exception, which the parent prints after ``-- Process N terminated with the
    following error:``, is preferred over the parent's own. Output without a traceback
    falls back to its last line that is an exception name, alone or followed by a colon.
    """
    blocks = _RANK_ERROR.split(text)
    if len(blocks) > 1:
        rank = _final_exceptions(blocks[-1])
        if rank:
            return rank[0]
    chains = _final_exceptions(text)
    if chains:
        return chains[-1]
    lines = _EXCEPTION_LINE.findall(text)
    return lines[-1].strip() if lines else None


def _signature_cmd(args: argparse.Namespace) -> int:
    with open(args.file, errors="replace") as f:
        signature = failure_signature(f.read())
    if signature is not None:
        print(signature)
    return 0


def _compare_cmd(args: argparse.Namespace) -> int:
    diffs = compare(args.a, args.b, exact=not args.close, trace=args.trace)
    print("\n".join(diffs) if diffs else "SAME")
    return 1 if diffs else 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=ARMS, required=True)
    run.add_argument("--mesh", default="cpu", help="comma-separated mesh device types")
    run.add_argument("--shape", default="1,2", help="mesh shape, e.g. 2 or 1,2")
    run.add_argument("--world-size", type=int, default=None)
    run.add_argument("--steps", type=int, default=2)
    run.add_argument("--early-bound", action="store_true")
    run.add_argument("--out", default=None)
    structure = sub.add_parser("structure")
    structure.add_argument("--arm", choices=("fake-torchada", "torchada"), required=True)
    structure.add_argument("--device-types", default="musa,cuda")
    structure.add_argument(
        "--after-lazy-setup",
        action="store_true",
        help="also record after running torch_musa's first-call _setup_fsdp2_patches()",
    )
    structure.add_argument("--out", default=None, help=f"directory for {STRUCTURE_FILE}")
    comp = sub.add_parser("compare")
    comp.add_argument("a")
    comp.add_argument("b")
    comp.add_argument("--close", action="store_true", help="compare tensors with a tolerance")
    comp.add_argument("--trace", default="all", help="all, none, shard or run")
    sig = sub.add_parser("signature")
    sig.add_argument("file", help="the captured stdout and stderr of a run")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.world_size is None:
            args.world_size = 1
            for n in args.shape.split(","):
                args.world_size *= int(n)
        return _run(args)
    if args.command == "structure":
        return _structure_cmd(args)
    if args.command == "signature":
        return _signature_cmd(args)
    return _compare_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
