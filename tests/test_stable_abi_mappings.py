"""Mapping-rule tests for the libtorch-stable ABI shim.

These are pure-Python (no GPU): they verify torchada's _mapping.py knows how to
port libtorch-stable kernels (as built by vLLM, SGLang, and other
torch_musa-dependent projects, which target a newer PyTorch stable ABI than
torch_musa ships) to MUSA. The end-to-end build+run is in test_stable_abi.py.
"""

import pytest


class TestStableAbiMappingRules:
    def test_aoti_stream_symbol_rule(self):
        """The AOTI C-ABI stream getter must map cuda->musa (the C++
        getCurrentCUDAStream rule does not match this C name)."""
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["aoti_torch_get_current_cuda_stream"]
            == "aoti_torch_get_current_musa_stream"
        )

    def test_stable_blas_handle_symbol_rule(self):
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["torch_get_current_cuda_blas_handle"]
            == "torch_get_current_musa_blas_handle"
        )

    @pytest.mark.parametrize(
        ("cuda_symbol", "musa_symbol"),
        [
            ("torch_set_current_cuda_stream", "torch_set_current_musa_stream"),
            ("torch_get_cuda_stream_from_pool", "torch_get_musa_stream_from_pool"),
            ("torch_cuda_stream_synchronize", "torch_musa_stream_synchronize"),
        ],
    )
    def test_stable_stream_symbol_rules(self, cuda_symbol, musa_symbol):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE[cuda_symbol] == musa_symbol

    def test_cuda_header_rules_present(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cuda.h"] == "musa.h"
        assert _MAPPING_RULE["cuda_runtime.h"] == "musa_runtime.h"


class TestStableAbiSourcePorting:
    """Porting a representative stable-ABI source applies the rules."""

    def _port(self, src: str) -> str:
        from torchada.utils.cpp_extension import _port_cuda_source

        return _port_cuda_source(src)

    def test_port_rewrites_aoti_stream(self):
        ported = self._port(
            "auto s = aoti_torch_get_current_cuda_stream(0, &p);")
        assert "aoti_torch_get_current_musa_stream" in ported
        assert "aoti_torch_get_current_cuda_stream" not in ported

    def test_port_rewrites_stable_blas_handle(self):
        ported = self._port(
            "auto e = torch_get_current_cuda_blas_handle(&handle);")
        assert "torch_get_current_musa_blas_handle" in ported
        assert "torch_get_current_cuda_blas_handle" not in ported

    def test_port_rewrites_stable_stream_symbols(self):
        ported = self._port(
            "torch_set_current_cuda_stream(stream, device);\n"
            "torch_get_cuda_stream_from_pool(false, device, &stream);\n"
            "torch_cuda_stream_synchronize(stream, device);\n"
        )
        assert "torch_set_current_musa_stream" in ported
        assert "torch_get_musa_stream_from_pool" in ported
        assert "torch_musa_stream_synchronize" in ported
        assert "cuda_stream" not in ported

    def test_ported_blas_handle_has_torch_29_fallback(self):
        """The symbol emitted by porting must exist on the torch 2.9 path."""
        from torchada.utils.cpp_extension import stable_compat_box_header

        ported = self._port(
            "auto e = torch_get_current_cuda_blas_handle(&handle);")
        symbol = "torch_get_current_musa_blas_handle"
        assert symbol in ported

        with open(stable_compat_box_header(), encoding="utf-8") as f:
            header = f.read()
        torch_29_branch = header.split(
            "// torch_musa 2.9 has no stable C shim", 1
        )[1].split("#endif", 1)[0]
        assert f"static inline AOTITorchError {symbol}(void** ret)" in torch_29_branch
        assert f"return {symbol}(ret);" in torch_29_branch

    @pytest.mark.parametrize("namespace", ["_C", "custom_ops", "third_party_ops"])
    def test_port_rekeys_stable_impl_block_for_any_namespace(self, namespace):
        ported = self._port(
            f"STABLE_TORCH_LIBRARY_IMPL({namespace}, CUDA, ops) {{ ops.impl(\"x\", f); }}")
        assert f"STABLE_TORCH_LIBRARY_IMPL({namespace}, PrivateUse1" in ported

    def test_port_rekeys_multiline_stable_impl_block(self):
        ported = self._port(
            "STABLE_TORCH_LIBRARY_IMPL(\n"
            "    third_party_ops,\n"
            "    CUDA,\n"
            "    ops) { ops.impl(\"x\", f); }"
        )
        assert "third_party_ops,\n    PrivateUse1," in ported

    @pytest.mark.parametrize("dispatch_key", ["CPU", "CompositeExplicitAutograd", "PrivateUse1"])
    def test_port_preserves_non_cuda_stable_impl_key(self, dispatch_key):
        source = f"STABLE_TORCH_LIBRARY_IMPL(custom_ops, {dispatch_key}, ops) {{}}"
        assert self._port(source) == source


class TestStableCompatHeadersShipped:
    """torchada ships the stable-ABI compat headers + exposes their paths."""

    def test_dispatch_shim_exists(self):
        import os
        from torchada.utils.cpp_extension import stable_compat_include_dir

        d = stable_compat_include_dir()
        assert os.path.isfile(
            os.path.join(d, "torch", "headeronly", "core", "Dispatch.h"))

    def test_box_header_exists_and_defines_torch_box(self):
        import os
        from torchada.utils.cpp_extension import stable_compat_box_header

        h = stable_compat_box_header()
        assert os.path.isfile(h)
        assert "define TORCH_BOX" in open(h).read()
