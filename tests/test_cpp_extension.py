"""
Tests for C++ extension building utilities.

These tests verify that after importing torchada, the standard torch imports
work correctly:
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
"""

import os

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

            # Extensions are converted: .cu -> .mu, .cuh -> .muh for mcc compiler
            assert musa_sp.EXT_REPLACED_MAPPING["cu"] == "mu"
            assert musa_sp.EXT_REPLACED_MAPPING["cuh"] == "muh"

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


class TestSimplePortingExcludeDirs:
    """Test directory exclusion helpers used with SimplePorting."""

    def test_get_env_exclude_dirs_uses_path_separator(self, monkeypatch, tmp_path):
        """TORCHADA_EXCLUDE_DIRS should accept multiple pathsep-delimited paths."""
        from torchada.utils.cpp_extension import _get_env_exclude_dirs

        first = tmp_path / "vendor"
        second = tmp_path / "third_party"
        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", f"{first}{os.pathsep}{second}")

        result = _get_env_exclude_dirs()

        assert os.path.realpath(str(first)) in result
        assert os.path.realpath(str(second)) in result

    def test_get_env_exclude_dirs_ignores_whitespace_entries(self, monkeypatch):
        """Whitespace-only env entries must not resolve to the current directory."""
        from torchada.utils.cpp_extension import _get_env_exclude_dirs

        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", f"   {os.pathsep}\t")

        assert _get_env_exclude_dirs() == []

    def test_is_path_in_dir_does_not_match_prefix_siblings(self, tmp_path):
        """A sibling with a shared prefix must not be treated as excluded."""
        from torchada.utils.cpp_extension import _is_path_in_dir

        excluded = tmp_path / "vendor"
        sibling = tmp_path / "vendor_extra"

        assert _is_path_in_dir(str(excluded), str(excluded))
        assert not _is_path_in_dir(str(sibling), str(excluded))

    def test_same_real_path_matches_equivalent_paths(self, tmp_path):
        """Equivalent absolute paths should be recognized before adding includes."""
        from torchada.utils.cpp_extension import _same_real_path

        include_dir = tmp_path / "include"
        include_dir.mkdir()

        assert _same_real_path(str(include_dir), str(include_dir / ".." / "include"))

    def test_collect_simple_porting_ignore_dirs_includes_nested_dirs(self, tmp_path):
        """SimplePorting needs exact ignore entries for nested excluded dirs."""
        from torchada.utils.cpp_extension import _collect_simple_porting_ignore_dirs

        source_dir = tmp_path / "csrc"
        excluded = source_dir / "vendor"
        nested = excluded / "cub"
        nested.mkdir(parents=True)

        result = _collect_simple_porting_ignore_dirs(str(source_dir), [str(excluded)])

        assert os.path.realpath(str(excluded)) in result
        assert os.path.realpath(str(nested)) in result

    def test_collect_simple_porting_ignore_dirs_ignores_outside_dirs(self, tmp_path):
        """Only excludes inside the ported source root should be passed down."""
        from torchada.utils.cpp_extension import _collect_simple_porting_ignore_dirs

        source_dir = tmp_path / "csrc"
        outside = tmp_path / "other_vendor"
        source_dir.mkdir()
        outside.mkdir()

        result = _collect_simple_porting_ignore_dirs(str(source_dir), [str(outside)])

        assert result == []

    def test_subclass_can_override_get_exclude_dirs(self, monkeypatch, tmp_path):
        """BuildExtension subclasses can merge env and project-specific excludes."""
        if not torchada.is_musa_platform():
            return

        from torchada.utils.cpp_extension import _get_build_extension_class

        env_exclude = tmp_path / "env_vendor"
        custom_exclude = tmp_path / "custom_vendor"
        monkeypatch.setenv("TORCHADA_EXCLUDE_DIRS", str(env_exclude))

        BaseClass = _get_build_extension_class()

        class CustomBuildExt(BaseClass):
            def get_exclude_dirs(self):
                return super().get_exclude_dirs() + [str(custom_exclude)]

        instance = CustomBuildExt.__new__(CustomBuildExt)
        result = instance.get_exclude_dirs()

        assert os.path.realpath(str(env_exclude)) in result
        assert str(custom_exclude) in result
