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

    def test_stable_impl_dispatch_key_rekey(self):
        """STABLE_TORCH_LIBRARY_IMPL registers under the literal dispatch-key
        token; MUSA tensors are PrivateUse1, so the block must be re-keyed."""
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["STABLE_TORCH_LIBRARY_IMPL(_C, CUDA"]
            == "STABLE_TORCH_LIBRARY_IMPL(_C, PrivateUse1"
        )

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

    def test_port_rekeys_stable_impl_block(self):
        ported = self._port(
            "STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) { ops.impl(\"x\", f); }")
        assert "STABLE_TORCH_LIBRARY_IMPL(_C, PrivateUse1" in ported


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
