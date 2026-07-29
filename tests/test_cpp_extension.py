"""
Tests for C++ extension building utilities.

These tests verify that after importing torchada, the standard torch imports
work correctly:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
"""

import os
from types import SimpleNamespace

# Import torchada first to apply patches
import torchada  # noqa: F401


class TestCppExtensionImports:
    """Test cpp_extension module imports using standard torch imports."""

    def test_import_cuda_home(self):
        """Test CUDA_HOME can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import CUDA_HOME

        # CUDA_HOME should be a string or None
        assert CUDA_HOME is None or isinstance(CUDA_HOME, str)

    def test_import_cuda_extension(self):
        """Test CUDAExtension can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import CUDAExtension

        assert CUDAExtension is not None

    def test_import_build_extension(self):
        """Test BuildExtension can be imported from torch.utils.cpp_extension."""
        from torch.utils.cpp_extension import BuildExtension

        assert BuildExtension is not None

    def test_cuda_home_on_musa(self):
        """Test CUDA_HOME points to MUSA on MUSA platform."""
        from torch.utils.cpp_extension import CUDA_HOME

        if torchada.is_musa_platform() and CUDA_HOME is not None:
            # On MUSA platform, CUDA_HOME should point to MUSA installation
            assert "musa" in CUDA_HOME.lower() or os.path.exists(
                os.path.join(CUDA_HOME, "bin", "mcc")
            )

    def test_torch_cpp_extension_is_patched(self):
        """Test that torch.utils.cpp_extension is patched correctly on MUSA."""
        import torch.utils.cpp_extension as torch_cpp_ext

        if torchada.is_musa_platform():
            # Verify CUDAExtension is our patched version
            assert torch_cpp_ext.CUDAExtension.__module__ == "torchada.utils.cpp_extension"

    def test_torch_cpp_extension_cuda_home_same_as_torchada(self):
        """Test that torch.utils.cpp_extension.CUDA_HOME matches torchada's."""
        import torch.utils.cpp_extension as torch_cpp_ext

        from torchada.utils.cpp_extension import CUDA_HOME as torchada_cuda_home

        if torchada.is_musa_platform():
            assert torch_cpp_ext.CUDA_HOME == torchada_cuda_home


class TestCUDAExtension:
    """Test CUDAExtension class using standard torch imports."""

    def test_create_extension_basic(self):
        """Test basic CUDAExtension creation."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
        )
        assert ext.name == "test_ext"
        assert "test.cu" in ext.sources

    def test_create_extension_with_include_dirs(self):
        """Test CUDAExtension with include_dirs."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
            include_dirs=["/usr/include"],
        )
        assert "/usr/include" in ext.include_dirs

    def test_create_extension_with_extra_compile_args(self):
        """Test CUDAExtension with extra_compile_args."""
        from torch.utils.cpp_extension import CUDAExtension

        ext = CUDAExtension(
            name="test_ext",
            sources=["test.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-arch=sm_70"]},
        )
        assert ext.extra_compile_args is not None

    def test_translate_nvjpeg_link_args(self):
        from torchada.utils.cpp_extension import _translate_link_args

        kwargs = {
            "libraries": ["jpeg", "nvjpeg"],
            "define_macros": [("JPEG_FOUND", 1), ("NVJPEG_FOUND", 1)],
        }

        translated = _translate_link_args(kwargs)

        assert translated["libraries"] == ["jpeg", "mtjpeg"]
        assert translated["define_macros"] == [
            ("JPEG_FOUND", 1),
            ("MTJPEG_FOUND", 1),
        ]
        assert kwargs["libraries"] == ["jpeg", "nvjpeg"]

    def test_porting_keeps_project_cuda_header_names(self):
        from torchada.utils.cpp_extension import _narrow_cuda_header_mapping

        rules = dict(
            _narrow_cuda_header_mapping(
                [("cudaMalloc", "musaMalloc"), ("cuda.h", "musa.h")]
            )
        )

        assert "cuda.h" not in rules
        assert rules['#include <cuda.h>'] == '#include <musa.h>'
        assert rules['#include "cuda.h"'] == '#include "musa.h"'

    def test_porting_translates_torch_cuda_header(self):
        from torchada.utils.cpp_extension import _narrow_cuda_header_mapping, _replace_porting_line

        rules = _narrow_cuda_header_mapping([("cuda.h", "musa.h")])

        assert (
            _replace_porting_line("#include <torch/cuda.h>\n", rules) == "#include <torch/musa.h>\n"
        )
        assert (
            _replace_porting_line('#include "torch/cuda.h"\n', rules) == '#include "torch/musa.h"\n'
        )

        project_include = '#include "project/decode_jpegs_cuda.h"\n'
        assert _replace_porting_line(project_include, rules) == project_include


class TestStableHeaderBackportVersionSelection:
    """Only torch versions predating 2.11 need the compatibility backport."""

    def test_torch_29_requires_backport(self, monkeypatch):
        import torch

        from torchada.utils.cpp_extension import _stable_header_backport_required

        for version in ("2.9.0", "2.9.0+mtgpu"):
            monkeypatch.setattr(torch, "__version__", version)
            assert _stable_header_backport_required() is True

    def test_torch_211_has_native_stable_abi(self, monkeypatch):
        import torch

        from torchada.utils.cpp_extension import _stable_header_backport_required

        for version in ("2.11.0", "2.11.0.dev20260730+mtgpu", "3.0.0"):
            monkeypatch.setattr(torch, "__version__", version)
            assert _stable_header_backport_required() is False


class TestDependencyIncludeProtection:
    """Editable dependency paths are never treated as project porting roots."""

    def test_dependency_package_and_checkout_paths_overlap(self):
        from torchada.utils.cpp_extension import _path_overlaps_any

        dependency_package = "/home/torch_musa/torch_musa"

        assert _path_overlaps_any("/home/torch_musa", [dependency_package])
        assert _path_overlaps_any(
            "/home/torch_musa/torch_musa/share/generated_cuda_compatible",
            [dependency_package],
        )
        assert not _path_overlaps_any("/home/torchvision", [dependency_package])

    def test_exclude_dirs_can_be_configured_by_environment(self, monkeypatch, tmp_path):
        from torchada.utils import cpp_extension

        excluded = tmp_path / "dependency"
        excluded.mkdir()
        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", str(excluded))

        assert cpp_extension._configured_exclude_dirs() == [str(excluded)]

    def test_exclude_dirs_accepts_package_names(self, monkeypatch, tmp_path):
        from torchada.utils import cpp_extension

        package = tmp_path / "torch_musa"
        package.mkdir()
        module_file = package / "__init__.py"
        module_file.write_text("", encoding="utf-8")
        monkeypatch.setattr(
            cpp_extension.importlib,
            "import_module",
            lambda name: SimpleNamespace(__file__=str(module_file)),
        )
        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", "torch_musa")

        assert cpp_extension._configured_exclude_dirs() == [str(package)]

    def test_exclude_dir_name_matches_path_component(self, monkeypatch):
        from torchada.utils import cpp_extension

        def unavailable(_name):
            raise ImportError

        monkeypatch.setattr(cpp_extension.importlib, "import_module", unavailable)
        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", "torch_musa")

        assert cpp_extension._is_configured_exclude_dir("/home/torch_musa")
        assert cpp_extension._is_configured_exclude_dir("/home/torch_musa/include")
        assert not cpp_extension._is_configured_exclude_dir("/home/torch_musa_extra")

    def test_empty_exclude_dirs_adds_nothing(self, monkeypatch):
        from torchada.utils import cpp_extension

        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", "")
        assert cpp_extension._configured_exclude_dirs() == []


class TestMusaPatches:
    """Test patches applied to torch_musa for extension building."""

    def test_is_musa_file_recognizes_cu(self):
        """Test _is_musa_file recognizes .cu files."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.cu")
            assert musa_ext._is_musa_file("path/to/kernel.cu")

    def test_is_musa_file_recognizes_cuh(self):
        """Test _is_musa_file recognizes .cuh files."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.cuh")
            assert musa_ext._is_musa_file("path/to/header.cuh")

    def test_is_musa_file_recognizes_mu(self):
        """Test _is_musa_file still recognizes .mu files."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.musa_extension as musa_ext

            assert musa_ext._is_musa_file("test.mu")

    def test_ext_replaced_mapping(self):
        """Test EXT_REPLACED_MAPPING keeps .cu/.cuh."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            # In-place porting retains filenames; mcc receives an explicit
            # ``-x musa`` flag instead of relying on .mu/.muh suffixes.
            assert musa_sp.EXT_REPLACED_MAPPING["cu"] == "cu"
            assert musa_sp.EXT_REPLACED_MAPPING["cuh"] == "cuh"

    def test_mapping_rule_exists(self):
        """Test _MAPPING_RULE is set."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            assert hasattr(musa_sp, "_MAPPING_RULE")
            assert len(musa_sp._MAPPING_RULE) > 0

    def test_mapping_rule_has_expected_entries(self):
        """Test _MAPPING_RULE has expected entries."""
        import torchada

        if torchada.is_musa_platform():
            import torch_musa.utils.simple_porting as musa_sp

            rules = musa_sp._MAPPING_RULE

            # Check some key mappings
            assert rules.get("cudaMalloc") == "musaMalloc"
            assert rules.get("cudaFree") == "musaFree"
            assert rules.get("cudaStream_t") == "musaStream_t"
            assert rules.get("at::cuda") == "at::musa"
            assert rules.get("c10::cuda") == "c10::musa"


class TestMusaNinjaLanguagePatch:
    """Test the filename-independent mcc language selection patch."""

    def test_adds_explicit_language_once(self):
        from torchada.utils.cpp_extension import _with_explicit_musa_language

        assert _with_explicit_musa_language(["-O3"]) == ["-O3", "-x", "musa"]
        assert _with_explicit_musa_language(["-x", "musa", "-O3"]) == [
            "-x",
            "musa",
            "-O3",
        ]
        assert _with_explicit_musa_language(["-x=musa", "-O3"]) == ["-x=musa", "-O3"]
        assert _with_explicit_musa_language(["-x", "cuda"]) == [
            "-x",
            "cuda",
            "-x",
            "musa",
        ]

    def test_patches_positional_and_keyword_musa_cflags(self):
        from torchada.utils.cpp_extension import _patch_musa_ninja_language

        calls = []

        def write_ninja(path, musa_cflags):
            calls.append((path, musa_cflags))

        module = SimpleNamespace(_write_ninja_file=write_ninja)
        _patch_musa_ninja_language(module)
        module._write_ninja_file("positional", ["-O2"])
        module._write_ninja_file(path="keyword", musa_cflags=["-g"])

        assert calls == [
            ("positional", ["-O2", "-x", "musa"]),
            ("keyword", ["-g", "-x", "musa"]),
        ]
