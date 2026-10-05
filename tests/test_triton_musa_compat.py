"""Tests for the MUSA Triton compatibility fixes in ``torchada.triton.musa_compat``."""

import functools
import importlib.util
import inspect
import logging
import json
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from torchada.triton import musa_compat as mc

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is optional for the unit tests
    triton = None


def _info(version="3.6.0", backend="musa"):
    backends = (backend,) if backend else ()
    entry_points = ("musa",) if backend == "musa" else ()
    return mc.MusaTritonInfo(version, backend, backends, entry_points, None, None, True)


INFO_36 = _info()
INFO_37 = _info("3.7.0")
INFO_32 = _info("3.2.0", "mtgpu")

REAL_INFO = mc.musa_triton_info()

requires_musa_triton_36 = pytest.mark.skipif(
    not REAL_INFO.is_musa_triton_36, reason="requires MUSA Triton 3.6"
)


# ---------------------------------------------------------------------------
# Triton description
# ---------------------------------------------------------------------------



@pytest.fixture
def mc_caplog(caplog, monkeypatch):
    """Capture this module's log records even if another test reconfigured logging.

    ``logging.config.dictConfig`` (used by packages imported elsewhere in the suite)
    disables loggers that already exist, which would hide these records from caplog.
    """
    monkeypatch.setattr(mc.logger, "disabled", False)
    monkeypatch.setattr(mc.logger, "propagate", True)
    caplog.set_level(logging.INFO, logger=mc.__name__)
    return caplog

class TestMusaTritonInfo:
    @pytest.mark.parametrize(
        "version, expected",
        [
            ("3.6.0", (3, 6, 0)),
            ("3.6.0+git1234", (3, 6, 0)),
            ("3.10.1rc1", (3, 10, 1)),
            ("", ()),
            (None, ()),
        ],
    )
    def test_release_tuple(self, version, expected):
        assert mc._release_tuple(version) == expected

    @pytest.mark.parametrize(
        "backends, backend, is_musa, is_36",
        [
            (["musa"], "musa", True, True),
            (["mtgpu"], "mtgpu", False, False),
            (["nvidia"], "nvidia", False, False),
            ([], None, False, False),
        ],
    )
    def test_describe_triton_backend(self, backends, backend, is_musa, is_36):
        fake = SimpleNamespace(__version__="3.6.0", __file__="/nonexistent/triton/__init__.py")
        info = mc.describe_triton(fake, backends)
        assert info.backend == backend
        assert info.is_musa_backend is is_musa
        assert info.is_musa_triton_36 is is_36
        assert info.dist_matches_import is False

    def test_flags(self):
        assert INFO_36.is_musa_triton_36
        assert not INFO_37.is_musa_triton_36
        assert INFO_37.is_musa_backend
        assert not INFO_32.is_musa_backend
        assert not INFO_32.is_musa_triton_36

    def test_cached(self):
        assert mc.musa_triton_info() is mc.musa_triton_info()

    def test_matches_imported_triton(self):
        if triton is None:
            pytest.skip("Triton is not installed")
        info = mc.musa_triton_info()
        assert info.version == triton.__version__
        if info.dist_matches_import:
            assert info.backend in info.backends


# ---------------------------------------------------------------------------
# Environment selection
# ---------------------------------------------------------------------------


class TestEnvironmentSelection:
    @pytest.mark.parametrize(
        "value, expected",
        [(None, "fix"), ("", "fix"), ("FIX", "fix"), ("off", "off"), ("vendor", "vendor")],
    )
    def test_inplace_alias_mode(self, value, expected):
        env = {} if value is None else {mc.INPLACE_ALIAS_ENV: value}
        assert mc.inplace_alias_mode(env) == expected

    def test_inplace_alias_mode_invalid_uses_default(self, mc_caplog):
        assert mc.inplace_alias_mode({mc.INPLACE_ALIAS_ENV: "bogus"}) == "fix"
        assert "bogus" in mc_caplog.text

    @pytest.mark.parametrize(
        "value, expected", [(None, False), ("0", False), ("1", True), ("true", True)]
    )
    def test_fast_exp_requested(self, value, expected):
        env = {} if value is None else {mc.FAST_EXP_ENV: value}
        assert mc.fast_exp_requested(env) is expected

    def test_resolve_f32_default(self, mc_caplog):
        allowed = mc.DEFAULT_DOT_INPUT_PRECISIONS
        assert mc.resolve_f32_default({}, allowed) is None
        assert mc.resolve_f32_default({mc.F32_DEFAULT_ENV: "IEEE"}, allowed) == "ieee"
        assert "allow_tf32=False" not in mc_caplog.text
        assert mc.resolve_f32_default({mc.F32_DEFAULT_ENV: "tf32x3"}, allowed) == "tf32x3"
        assert "allow_tf32=False" in mc_caplog.text
        assert mc.resolve_f32_default({mc.F32_DEFAULT_ENV: "bogus"}, allowed) is None
        assert "bogus" in mc_caplog.text
        env = {mc.F32_DEFAULT_ENV: "ieee", mc.TRITON_F32_DEFAULT_ENV: "tf32"}
        assert mc.resolve_f32_default(env, allowed) is None

    def test_allowed_dot_input_precisions(self):
        assert mc.allowed_dot_input_precisions(SimpleNamespace()) == (
            mc.DEFAULT_DOT_INPUT_PRECISIONS
        )

        class MUSAOptions:
            allowed_dot_input_precisions = ("ieee", "tf32")

        backend = SimpleNamespace(MUSAOptions=MUSAOptions)
        assert mc.allowed_dot_input_precisions(backend) == ("ieee", "tf32")

    def test_install_f32_default(self, mc_caplog):
        env = {mc.F32_DEFAULT_ENV: "ieee"}
        with mc_caplog.at_level(logging.INFO, logger=mc.__name__):
            assert mc.install_f32_default(env, info=INFO_32) is None
        assert f"ignoring {mc.F32_DEFAULT_ENV}" in mc_caplog.text
        assert mc.TRITON_F32_DEFAULT_ENV not in env
        assert mc.install_f32_default(env, info=INFO_36) == "ieee"
        assert env[mc.TRITON_F32_DEFAULT_ENV] == "ieee"

    def test_install_f32_default_keeps_user_value(self):
        env = {mc.F32_DEFAULT_ENV: "ieee", mc.TRITON_F32_DEFAULT_ENV: "tf32"}
        assert mc.install_f32_default(env, info=INFO_36) is None
        assert env[mc.TRITON_F32_DEFAULT_ENV] == "tf32"


# ---------------------------------------------------------------------------
# Backend hash salt
# ---------------------------------------------------------------------------


def _fake_backend_class():
    class FakeBackend:
        def __init__(self, arch):
            self.arch = arch

        @functools.lru_cache()
        def hash(self):
            return f"version-{self.arch}"

    return FakeBackend


class TestBackendHashSalt:
    def test_no_salt_keeps_hash(self):
        backend = _fake_backend_class()
        assert mc.install_backend_hash_salt(backend)
        wrapper = backend.hash
        assert backend(31).hash() == "version-31"
        assert mc.install_backend_hash_salt(backend)
        assert backend.hash is wrapper

    def test_salts_sorted_and_replaced(self):
        backend = _fake_backend_class()
        assert mc.add_backend_hash_salt("b-patch", 2, backend_cls=backend)
        assert mc.add_backend_hash_salt("a-patch", 1, backend_cls=backend)
        assert backend(31).hash() == "version-31-torchada:a-patch:1,b-patch:2"
        assert mc.add_backend_hash_salt("a-patch", 3, backend_cls=backend)
        assert backend(31).hash() == "version-31-torchada:a-patch:3,b-patch:2"
        assert mc.backend_hash_suffix(backend) == "-torchada:a-patch:3,b-patch:2"

    def test_no_backend(self, monkeypatch):
        monkeypatch.setattr(mc, "_musa_backend_class", lambda: None)
        assert not mc.add_backend_hash_salt("x", 1)
        assert mc.backend_hash_suffix() == ""

    def test_salts_shared_across_module_copies(self):
        spec = importlib.util.spec_from_file_location("_musa_compat_copy", mc.__file__)
        copy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(copy)
        backend = _fake_backend_class()
        assert mc.add_backend_hash_salt("a-patch", 1, backend_cls=backend)
        assert copy.add_backend_hash_salt("b-patch", 2, backend_cls=backend)
        assert backend(31).hash() == "version-31-torchada:a-patch:1,b-patch:2"
        assert mc.backend_hash_suffix(backend) == copy.backend_hash_suffix(backend)


# ---------------------------------------------------------------------------
# ASTFunction.deserialize gate
# ---------------------------------------------------------------------------


class _MainFrontendASTFunction:
    def __init__(self, ret_types, arg_types, attrs):
        pass

    def deserialize(self, fn):
        return None


class _TagFrontendASTFunction:
    def __init__(self, ret_types, arg_types, constants, attrs):
        pass

    def deserialize(self, fn):
        return None


class TestDeserializeGate:
    def test_needs_fix(self):
        assert mc.ast_function_needs_attr_fix(_MainFrontendASTFunction)
        assert not mc.ast_function_needs_attr_fix(_TagFrontendASTFunction)
        assert not mc.ast_function_needs_attr_fix(None)

    def test_untouched_without_matching_frontend(self):
        assert not mc.install_deserialize_fix(SimpleNamespace(), INFO_36)
        original = _TagFrontendASTFunction.deserialize
        frontend = SimpleNamespace(ASTFunction=_TagFrontendASTFunction)
        assert not mc.install_deserialize_fix(frontend, INFO_36)
        assert _TagFrontendASTFunction.deserialize is original

    @pytest.mark.parametrize("info", [INFO_32, INFO_37])
    def test_untouched_outside_musa_triton_36(self, info):
        class ASTFunction(_MainFrontendASTFunction):
            pass

        original = ASTFunction.deserialize
        assert not mc.install_deserialize_fix(SimpleNamespace(ASTFunction=ASTFunction), info)
        assert ASTFunction.deserialize is original


# ---------------------------------------------------------------------------
# inplace_alias_pairs
# ---------------------------------------------------------------------------


class _Ptr:
    dtype = "float32"

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


def _param(annotation_type="", is_constexpr=False, do_not_specialize=False):
    return SimpleNamespace(
        annotation_type=annotation_type,
        is_constexpr=is_constexpr,
        do_not_specialize=do_not_specialize,
        do_not_specialize_on_alignment=False,
        is_const=False,
    )


def _fake_specialize(backend, value, is_const, specialize, align):
    if isinstance(value, tuple):
        return tuple(_fake_specialize(backend, v, is_const, True, align)[0] for v in value), None
    if isinstance(value, _Ptr):
        return "*fp32", None
    if isinstance(value, str):
        raise TypeError("failed to specialize argument of type: str")
    if value is None or (type(value) is int and value == 1):
        return "constexpr", None
    if isinstance(value, (int, float)):
        return "i32", None
    if isinstance(value, _JitFunctionStandIn):
        return "constexpr", None
    raise TypeError(f"failed to specialize argument of type: {type(value).__name__}")


class _JitFunctionStandIn:
    """Specializes to constexpr, like a JIT function argument."""


def _vendor_numbering(params, bound_vals):
    """Parameter-position numbering, as in the helper the fix replaces."""
    by_ptr = {}
    idx = 0
    for param, value in zip(params, bound_vals):
        if param.is_constexpr:
            continue
        data_ptr = getattr(value, "data_ptr", None)
        if callable(data_ptr) and data_ptr():
            by_ptr.setdefault(data_ptr(), []).append(idx)
        idx += 1
    return ",".join(
        f"{ix[a]}:{ix[b]}"
        for ix in by_ptr.values()
        for a in range(len(ix))
        for b in range(a + 1, len(ix))
    )


def _fake_jit(spec=_vendor_numbering):
    def kernel_param(num, param, do_not_specialize, do_not_specialize_on_alignment):
        return _param()

    return SimpleNamespace(
        KernelParam=kernel_param,
        _make_pointer_alias_spec=spec,
        native_specialize_impl=_fake_specialize,
    )


X, W, Z = _Ptr(0x1000), _Ptr(0x2000), _Ptr(0x3000)
FIVE = [_param() for _ in range(5)] + [_param(is_constexpr=True)]


class TestPointerAliasSpec:
    @pytest.mark.parametrize(
        "params, values, expected",
        [
            (FIVE, [None, X, X, W, 100, 64], "0:1"),
            (FIVE, [Z, X, X, W, 100, 64], "1:2"),
            (FIVE, [Z, X, W, None, 100, 64], ""),
            (FIVE, [1, X, X, W, 100, 64], "0:1"),
            (FIVE, [7, X, X, W, 100, 64], "1:2"),
            (FIVE, [True, X, X, W, 100, 64], "1:2"),
            (FIVE, [1.0, X, X, W, 100, 64], "1:2"),
            (FIVE, [(Z, None), X, X, W, 100, 64], "1:2"),
            (FIVE, [(Z, W), X, X, W, 100, 64], "2:3"),
            (FIVE, [_JitFunctionStandIn(), X, X, W, 100, 64], "0:1"),
            (FIVE, [SimpleNamespace(), X, X, W, 100, 64], ""),
            (FIVE, [X, X, W, W, 100, 64], "0:1,2:3"),
            (FIVE, [X, X, X, W, 100, 64], "0:1,0:2,1:2"),
            (FIVE, ["DOUBLE", X, X, W, 100, 64], ""),
            ([_param(do_not_specialize=True)] + FIVE[1:], [1, X, X, W, 100, 64], "1:2"),
            ([_param("i32")] + FIVE[1:], [1, X, X, W, 100, 64], "1:2"),
            ([_param("*fp32")] + FIVE[1:], [X, X, Z, W, 100, 64], "0:1"),
        ],
    )
    def test_ir_argument_numbering(self, params, values, expected):
        spec = mc.make_pointer_alias_spec(_fake_specialize, object())
        assert spec(params, values) == expected

    def test_off_spec(self):
        assert mc.make_no_pointer_alias_spec()(FIVE, [None, X, X, W, 100, 64]) == ""

    def test_vendor_fingerprint(self):
        assert mc.vendor_alias_spec_has_param_numbering(_fake_jit())
        fixed = mc.make_pointer_alias_spec(_fake_specialize, object())
        assert not mc.vendor_alias_spec_has_param_numbering(_fake_jit(fixed))

    def test_untouched_on_32(self):
        jit = _fake_jit()
        assert mc.install_inplace_alias_fix("fix", jit, INFO_32).startswith("skip")
        assert mc.install_inplace_alias_fix("off", jit, INFO_37).startswith("skip")
        assert jit._make_pointer_alias_spec is _vendor_numbering

    def test_vendor_mode(self):
        jit = _fake_jit()
        assert mc.install_inplace_alias_fix("vendor", jit, INFO_36).startswith("skip")
        assert jit._make_pointer_alias_spec is _vendor_numbering

    def test_unrecognized_helper_kept(self):
        def already_fixed(params, bound_vals):
            return "0:1"

        jit = _fake_jit(already_fixed)
        assert mc.install_inplace_alias_fix("fix", jit, INFO_36).startswith("skip")
        assert jit._make_pointer_alias_spec is already_fixed

    def test_off_mode(self):
        jit = _fake_jit()
        assert mc.install_inplace_alias_fix("off", jit, INFO_36) == "off"
        assert jit._make_pointer_alias_spec(FIVE, [None, X, X, W, 100, 64]) == ""
        assert jit._make_pointer_alias_spec.__wrapped__ is _vendor_numbering
        assert mc.install_inplace_alias_fix("fix", jit, INFO_36).startswith("skip")

    def test_fix_mode(self):
        pytest.importorskip("triton.backends.compiler")
        jit = _fake_jit()
        assert mc.install_inplace_alias_fix("fix", jit, INFO_36) == "fix"
        spec = jit._make_pointer_alias_spec
        assert spec(FIVE, [None, X, X, W, 100, 64]) == "0:1"
        assert spec.__wrapped__ is _vendor_numbering
        assert mc.install_inplace_alias_fix("fix", jit, INFO_36) == "fix"
        assert jit._make_pointer_alias_spec is spec


class TestFastExpGate:
    def test_untouched_on_32(self, mc_caplog):
        with mc_caplog.at_level(logging.INFO, logger=mc.__name__):
            assert mc.install_fast_exp(info=INFO_32) is False
            assert mc.install_fast_exp(info=INFO_37) is False
        assert f"ignoring {mc.FAST_EXP_ENV}" in mc_caplog.text

    def test_build_failure_leaves_hash_unsalted(self, monkeypatch, mc_caplog):
        tl_math = pytest.importorskip("triton.language.math")
        backend = _fake_backend_class()
        original_exp = tl_math.exp

        def broken(original):
            raise ImportError("cannot import name '_check_dtype'")

        monkeypatch.setattr(mc, "_musa_backend_class", lambda: backend)
        monkeypatch.setattr(mc, "_make_fast_exp", broken)
        if getattr(original_exp, mc._FAST_EXP_MARK, False):
            pytest.skip("fast exp already installed in this process")
        assert mc.install_fast_exp(info=INFO_36) is False
        assert "_check_dtype" in mc_caplog.text
        assert tl_math.exp is original_exp
        assert backend(31).hash() == "version-31"


# ---------------------------------------------------------------------------
# Real MUSA Triton 3.6
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _copy_kernel(src_ptr, dst_ptr, n, row_stride, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        values = tl.load(src_ptr + offs * row_stride, mask=mask)
        tl.store(dst_ptr + offs * row_stride, values, mask=mask)

    @triton.jit
    def _scalar_first_kernel(n, x_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)

    @triton.jit
    def _str_constexpr_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr, ACT: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        if ACT == "DOUBLE":
            x = x * 2.0
        tl.store(out_ptr + offs, x, mask=mask)

    @triton.jit
    def _maybe_kernel(maybe_ptr, x_ptr, out_ptr, w_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        if maybe_ptr is not None:
            x += tl.load(maybe_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, x * tl.load(w_ptr + offs, mask=mask), mask=mask)


def _identify_mutated_tensors():
    torch = pytest.importorskip("torch")
    try:
        from torch._higher_order_ops.triton_kernel_wrap import identify_mutated_tensors
    except ImportError:
        pytest.skip("torch has no Triton kernel mutation analysis")
    params = list(inspect.signature(identify_mutated_tensors).parameters)
    if params != ["kernel", "kwargs", "tma_descriptor_metadata"]:
        pytest.skip(f"unsupported identify_mutated_tensors signature {params}")
    try:
        triton.runtime.driver.active.get_current_target()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no active Triton driver: {exc}")
    return torch, identify_mutated_tensors


@pytest.mark.musa
@requires_musa_triton_36
class TestRealDeserializeFix:
    def test_installed(self):
        from triton.compiler.code_generator import ASTFunction

        assert getattr(ASTFunction.deserialize, mc._DESERIALIZE_MARK, False)

    @pytest.mark.parametrize(
        "case, expected",
        [("tensor_first", ["dst_ptr"]), ("scalar_first", ["out_ptr"]), ("str", ["out_ptr"])],
    )
    def test_identify_mutated_tensors(self, case, expected):
        torch, identify = _identify_mutated_tensors()
        a, b = torch.empty(2), torch.empty(2)
        kernel, kwargs = {
            "tensor_first": (
                _copy_kernel,
                dict(src_ptr=a, dst_ptr=b, n=64, row_stride=64, BLOCK=64),
            ),
            "scalar_first": (_scalar_first_kernel, dict(n=64, x_ptr=a, out_ptr=b, BLOCK=64)),
            "str": (
                _str_constexpr_kernel,
                dict(x_ptr=a, out_ptr=b, n=64, BLOCK=64, ACT="DOUBLE"),
            ),
        }[case]
        assert sorted(identify(kernel, kwargs, {})) == expected


@pytest.mark.musa
@requires_musa_triton_36
class TestRealInplaceAliasFix:
    def _spec(self):
        if mc.inplace_alias_mode() != "fix":
            pytest.skip(f"{mc.INPLACE_ALIAS_ENV} selects {mc.inplace_alias_mode()}")
        import triton.runtime.jit as jit

        spec = jit._make_pointer_alias_spec
        if getattr(spec, mc._ALIAS_MARK, None) is None:
            if not mc.vendor_alias_spec_has_param_numbering(jit):
                pytest.skip("Triton's _make_pointer_alias_spec already numbers IR arguments")
        assert getattr(spec, mc._ALIAS_MARK, None) == "fix"
        return spec

    def test_none_parameter_not_counted(self):
        torch = pytest.importorskip("torch")
        spec = self._spec()
        x, w = torch.empty(16), torch.empty(16)
        values = [None, x, x, w, 16, 64]
        assert spec(_maybe_kernel.params, values) == "0:1"
        assert spec.__wrapped__(_maybe_kernel.params, values) == "1:2"

    def test_all_tensor_launch_unchanged(self):
        torch = pytest.importorskip("torch")
        spec = self._spec()
        x, w, z = torch.empty(16), torch.empty(16), torch.empty(16)
        values = [z, x, x, w, 16, 64]
        assert spec(_maybe_kernel.params, values) == "1:2"
        assert spec(_maybe_kernel.params, values) == spec.__wrapped__(_maybe_kernel.params, values)


_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import json, os, re
    import torchada  # noqa: F401
    import triton
    import triton.language as tl
    from triton._C.libtriton import ir
    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend

    @triton.jit
    def k_exp(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=m, other=0.0)), mask=m)

    target = GPUTarget("musa", 31, 32)
    backend = make_backend(target)
    src = ASTSource(
        k_exp,
        {"x_ptr": "*fp32", "o_ptr": "*fp32", "n": "i32", "BLOCK": "constexpr"},
        {"BLOCK": 256},
    )
    options = backend.parse_options(src.parse_options())
    stages = {}
    backend.add_stages(stages, options, src.language)
    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)
    module = src.make_ir(
        target, options, backend.get_codegen_implementation(options),
        backend.get_module_map(), context,
    )
    out = {"ttir": str(module)}
    metadata = {}
    for name, stage in stages.items():
        module = stage(module, metadata)
        out[name] = str(module)
        if name == "llir":
            break
    print(json.dumps({
        "hash": backend.hash(),
        "ttir_exp2": out["ttir"].count("math.exp2"),
        "llir_exp2": len(re.findall(r"call [^\\n]*@llvm\\.exp2\\.f32", out["llir"])),
        "f32_default": os.environ.get("TRITON_F32_DEFAULT"),
    }))
    """
)


def _run_subprocess(tmp_path, env_updates):
    script = tmp_path / "triton_probe.py"
    script.write_text(_SUBPROCESS_SCRIPT)
    env = dict(os.environ)
    for key in (mc.FAST_EXP_ENV, mc.F32_DEFAULT_ENV, mc.TRITON_F32_DEFAULT_ENV):
        env.pop(key, None)
    env.update(env_updates)
    env["TRITON_CACHE_DIR"] = str(tmp_path / "triton-cache")
    result = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.musa
@requires_musa_triton_36
class TestRealCodegenPatches:
    def test_fast_exp(self, tmp_path):
        default = _run_subprocess(tmp_path, {})
        assert default["ttir_exp2"] == 0
        assert default["llir_exp2"] == 0
        assert "-torchada:" not in default["hash"]

        fast = _run_subprocess(tmp_path, {mc.FAST_EXP_ENV: "1"})
        assert fast["ttir_exp2"] == 1
        assert fast["llir_exp2"] > 0
        assert fast["hash"] == default["hash"] + f"-torchada:fast-exp:{mc.FAST_EXP_REVISION}"

    def test_f32_default(self, tmp_path):
        assert _run_subprocess(tmp_path, {})["f32_default"] is None
        assert _run_subprocess(tmp_path, {mc.F32_DEFAULT_ENV: "ieee"})["f32_default"] == "ieee"
        assert _run_subprocess(tmp_path, {mc.F32_DEFAULT_ENV: "bogus"})["f32_default"] is None
        preset = {mc.F32_DEFAULT_ENV: "ieee", mc.TRITON_F32_DEFAULT_ENV: "tf32"}
        assert _run_subprocess(tmp_path, preset)["f32_default"] == "tf32"


@pytest.mark.musa
@pytest.mark.skipif(REAL_INFO.backend != "mtgpu", reason="requires MUSA Triton 3.2")
def test_musa_triton_32_untouched():
    import triton.language.math as tl_math
    import triton.runtime.jit as jit

    assert not getattr(tl_math.exp, mc._FAST_EXP_MARK, False)
    assert getattr(jit, "_make_pointer_alias_spec", None) is None
    code_generator = pytest.importorskip("triton.compiler.code_generator")
    ast_function = getattr(code_generator, "ASTFunction", None)
    if ast_function is not None:
        assert not getattr(ast_function.deserialize, mc._DESERIALIZE_MARK, False)
