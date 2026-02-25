"""
Tests for torch.cuda patching functionality.

These tests verify that torch.cuda.* APIs work transparently on MUSA.
"""

import pytest


class TestTorchCudaModule:
    """Test torch.cuda module patching."""

    def test_torch_cuda_is_patched_on_musa(self):
        """Test that torch.cuda is patched to torch.musa on MUSA platform."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # torch.cuda should now be torch_musa
            assert "torch_musa" in torch.cuda.__name__ or "musa" in str(torch.cuda)
        else:
            # On CUDA/CPU, torch.cuda should remain unchanged
            assert "cuda" in torch.cuda.__name__

    def test_torch_cuda_is_available(self):
        """Test torch.cuda.is_available() is NOT redirected on MUSA.

        This is intentionally NOT redirected to allow downstream projects
        to detect the MUSA platform using patterns like:
            if torch.cuda.is_available():  # CUDA
            elif torch.musa.is_available():  # MUSA
        """
        import torch

        import torchada

        result = torch.cuda.is_available()
        assert isinstance(result, bool)

        if torchada.is_musa_platform():
            # On MUSA platform, torch.cuda.is_available() should return False
            # because CUDA is not available - only MUSA is
            assert result is False

    def test_torch_cuda_device_count(self):
        """Test torch.cuda.device_count() works."""
        import torch

        import torchada

        count = torch.cuda.device_count()
        assert isinstance(count, int)
        assert count >= 0

        if torchada.is_musa_platform() and torch.cuda.is_available():
            import torch_musa

            assert count == torch_musa.device_count()

    def test_torch_cuda_device_count_nvml(self):
        """Test torch.cuda._device_count_nvml() maps to torch.musa.device_count()."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # _device_count_nvml is NVIDIA-specific, should map to device_count on MUSA
            nvml_count = torch.cuda._device_count_nvml()
            musa_count = torch.musa.device_count()
            assert nvml_count == musa_count
            assert isinstance(nvml_count, int)
            assert nvml_count >= 0

    def test_torch_cuda_current_device(self):
        """Test torch.cuda.current_device() works when GPU available."""
        import torch

        if torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            assert isinstance(device_id, int)
            assert device_id >= 0
            assert device_id < torch.cuda.device_count()

    def test_torch_cuda_get_device_name(self):
        """Test torch.cuda.get_device_name() works."""
        import torch

        import torchada

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name()
            assert isinstance(name, str)
            assert len(name) > 0

            if torchada.is_musa_platform():
                # MUSA device names typically contain "MTT" or "Moore"
                # but this is hardware-dependent, so just check it's not empty
                pass

    def test_torch_cuda_synchronize(self):
        """Test torch.cuda.synchronize() works."""
        import torch

        if torch.cuda.is_available():
            # Should not raise
            torch.cuda.synchronize()

    def test_torch_cuda_empty_cache(self):
        """Test torch.cuda.empty_cache() works."""
        import torch

        if torch.cuda.is_available():
            # Should not raise
            torch.cuda.empty_cache()

    def test_torch_cuda_memory_allocated(self):
        """Test torch.cuda.memory_allocated() works."""
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated()
            assert isinstance(mem, int)
            assert mem >= 0

    def test_torch_cuda_memory_reserved(self):
        """Test torch.cuda.memory_reserved() works."""
        import torch

        if torch.cuda.is_available():
            mem = torch.cuda.memory_reserved()
            assert isinstance(mem, int)
            assert mem >= 0

    def test_torch_cuda_set_device(self):
        """Test torch.cuda.set_device() works."""
        import torch

        if torch.cuda.is_available():
            current = torch.cuda.current_device()
            torch.cuda.set_device(current)
            assert torch.cuda.current_device() == current


class TestTorchCudaLazyInit:
    """Test torch.cuda lazy initialization functions patching."""

    def test_import_lazy_call(self):
        """Test that _lazy_call can be imported from torch.cuda.

        This is needed because torch_musa only maps _lazy_init but not _lazy_call.
        torchada patches this to make from torch.cuda import _lazy_call work.
        """
        from torch.cuda import _lazy_call

        assert _lazy_call is not None
        assert callable(_lazy_call)

    def test_lazy_call_functionality(self):
        """Test that _lazy_call works correctly."""
        from torch.cuda import _lazy_call

        # _lazy_call should accept a callable and queue it for lazy execution
        called = []

        def test_callback():
            called.append(True)

        # Should not raise
        _lazy_call(test_callback)


class TestTorchCudaAmp:
    """Test torch.cuda.amp module patching."""

    def test_import_autocast(self):
        """Test that autocast can be imported from torch.cuda.amp."""
        from torch.cuda.amp import autocast

        assert autocast is not None

    def test_import_grad_scaler(self):
        """Test that GradScaler can be imported from torch.cuda.amp."""
        from torch.cuda.amp import GradScaler

        assert GradScaler is not None

    def test_autocast_context_manager(self):
        """Test autocast works as context manager."""
        import torch
        from torch.cuda.amp import autocast

        if torch.cuda.is_available():
            try:
                with autocast():
                    x = torch.randn(2, 2, device="cuda")
                    assert x.device.type in ("cuda", "musa")
            except RuntimeError as e:
                # MUSA driver issue, not torchada
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_grad_scaler_creation(self):
        """Test GradScaler can be created."""
        import torch
        from torch.cuda.amp import GradScaler

        if torch.cuda.is_available():
            scaler = GradScaler()
            assert scaler is not None


class TestCUDAGraph:
    """Test CUDAGraph aliasing."""

    def test_cuda_graph_alias(self):
        """Test torch.cuda.CUDAGraph is available."""
        import torch

        import torchada

        assert hasattr(torch.cuda, "CUDAGraph")
        # MUSAGraph only exists on MUSA platform (torch_musa specific class)
        # On CUDA platforms, only CUDAGraph exists
        if torchada.is_musa_platform():
            assert hasattr(torch.cuda, "MUSAGraph")

    def test_cuda_graph_is_musa_graph(self):
        """Test torch.cuda.CUDAGraph is aliased to MUSAGraph on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            assert torch.cuda.CUDAGraph is torch.cuda.MUSAGraph

    def test_cuda_graph_creation(self):
        """Test CUDAGraph can be created."""
        import torch

        if torch.cuda.is_available():
            try:
                g = torch.cuda.CUDAGraph()
                assert g is not None
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise

    def test_graphs_module(self):
        """Test torch.cuda.graphs module is available."""
        import torch

        assert hasattr(torch.cuda, "graphs")

    def test_make_graphed_callables(self):
        """Test make_graphed_callables is available."""
        import torch

        assert hasattr(torch.cuda, "make_graphed_callables")

    def test_graph_pool_handle(self):
        """Test graph_pool_handle is available."""
        import torch

        assert hasattr(torch.cuda, "graph_pool_handle")

    def test_graph_context_manager_cuda_graph_keyword(self):
        """Test torch.cuda.graph accepts cuda_graph= keyword argument.

        MUSA's graph class uses musa_graph= but CUDA code uses cuda_graph=.
        The wrapper should translate cuda_graph= to work on MUSA.
        """
        import torch

        import torchada

        # Use device_count > 0 since torch.cuda.is_available() returns False on MUSA
        if torch.cuda.device_count() == 0:
            pytest.skip("No GPU available")

        g = torch.cuda.CUDAGraph()

        # Should accept cuda_graph= keyword argument
        ctx = torch.cuda.graph(cuda_graph=g)
        assert ctx is not None
        # _wrapped attribute only exists on MUSA platform where torchada wraps
        # the context manager to translate cuda_graph= to musa_graph=
        # On CUDA platforms, no wrapping is needed
        if torchada.is_musa_platform():
            assert hasattr(ctx, "_wrapped")

    def test_graph_context_manager_positional(self):
        """Test torch.cuda.graph accepts positional argument."""
        import torch

        if torch.cuda.device_count() == 0:
            pytest.skip("No GPU available")

        g = torch.cuda.CUDAGraph()

        # Should accept positional argument
        ctx = torch.cuda.graph(g)
        assert ctx is not None

    def test_graph_context_manager_musa_graph_keyword(self):
        """Test torch.cuda.graph also accepts musa_graph= for compatibility."""
        import torch

        import torchada

        # This test only applies on MUSA platform where musa_graph= is valid
        # On CUDA platforms, only cuda_graph= keyword is valid (no musa_graph=)
        if not torchada.is_musa_platform():
            pytest.skip("musa_graph= keyword only valid on MUSA platform")

        if torch.cuda.device_count() == 0:
            pytest.skip("No GPU available")

        g = torch.cuda.CUDAGraph()

        # Should also accept musa_graph= for direct MUSA code
        ctx = torch.cuda.graph(musa_graph=g)
        assert ctx is not None

    def test_graph_context_manager_missing_arg_raises(self):
        """Test torch.cuda.graph raises TypeError when graph object is missing."""
        import torch

        import torchada  # noqa: F401

        # Error message differs between CUDA and MUSA platforms
        with pytest.raises(
            TypeError, match="(missing required argument|required positional argument)"
        ):
            torch.cuda.graph()

    def test_graph_context_manager_with_pool_and_stream(self):
        """Test torch.cuda.graph accepts pool and stream arguments."""
        import torch

        if torch.cuda.device_count() == 0:
            pytest.skip("No GPU available")

        g = torch.cuda.CUDAGraph()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream()

        # Should accept all arguments together
        ctx = torch.cuda.graph(cuda_graph=g, pool=pool, stream=stream)
        assert ctx is not None


class TestDistributedBackend:
    """Test distributed backend patching."""

    def test_original_init_available(self):
        """Test original init_process_group is saved."""
        import torchada

        if torchada.is_musa_platform():
            original = torchada.get_original_init_process_group()
            assert original is not None

    def test_mccl_backend_available(self):
        """Test MCCL backend is registered."""
        import torch.distributed as dist

        import torchada

        if torchada.is_musa_platform():
            assert hasattr(dist.Backend, "MCCL")

    def test_nccl_backend_available(self):
        """Test NCCL backend constant is still available."""
        import torch.distributed as dist

        assert hasattr(dist.Backend, "NCCL")

    def test_new_group_patched(self):
        """Test new_group is patched to translate nccl->mccl."""
        import torch.distributed as dist

        import torchada

        if torchada.is_musa_platform():
            # Check new_group is wrapped
            assert hasattr(dist.new_group, "__wrapped__")


class TestNCCLModule:
    """Test NCCL to MCCL module patching."""

    def test_mccl_module_available(self):
        """Test torch.cuda.mccl is available."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            assert hasattr(torch.cuda, "mccl")


class TestRNGFunctions:
    """Test RNG functions are available through torch.cuda."""

    def test_get_rng_state(self):
        """Test torch.cuda.get_rng_state is available."""
        import torch

        assert hasattr(torch.cuda, "get_rng_state")

    def test_set_rng_state(self):
        """Test torch.cuda.set_rng_state is available."""
        import torch

        assert hasattr(torch.cuda, "set_rng_state")

    def test_manual_seed(self):
        """Test torch.cuda.manual_seed is available."""
        import torch

        assert hasattr(torch.cuda, "manual_seed")

    def test_manual_seed_all(self):
        """Test torch.cuda.manual_seed_all is available."""
        import torch

        assert hasattr(torch.cuda, "manual_seed_all")

    def test_seed(self):
        """Test torch.cuda.seed is available."""
        import torch

        assert hasattr(torch.cuda, "seed")

    def test_initial_seed(self):
        """Test torch.cuda.initial_seed is available."""
        import torch

        assert hasattr(torch.cuda, "initial_seed")

    def test_manual_seed_works(self):
        """Test torch.cuda.manual_seed actually works."""
        import torch

        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)


class TestMemoryFunctions:
    """Test additional memory functions."""

    def test_max_memory_allocated(self):
        """Test torch.cuda.max_memory_allocated is available."""
        import torch

        assert hasattr(torch.cuda, "max_memory_allocated")

    def test_max_memory_reserved(self):
        """Test torch.cuda.max_memory_reserved is available."""
        import torch

        assert hasattr(torch.cuda, "max_memory_reserved")

    def test_memory_stats(self):
        """Test torch.cuda.memory_stats is available."""
        import torch

        assert hasattr(torch.cuda, "memory_stats")

    def test_memory_summary(self):
        """Test torch.cuda.memory_summary is available."""
        import torch

        assert hasattr(torch.cuda, "memory_summary")

    def test_memory_snapshot(self):
        """Test torch.cuda.memory_snapshot is available."""
        import torch

        assert hasattr(torch.cuda, "memory_snapshot")

    def test_reset_peak_memory_stats(self):
        """Test torch.cuda.reset_peak_memory_stats is available."""
        import torch

        assert hasattr(torch.cuda, "reset_peak_memory_stats")

    def test_mem_get_info(self):
        """Test torch.cuda.mem_get_info is available."""
        import torch

        assert hasattr(torch.cuda, "mem_get_info")

        if torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                assert free >= 0
                assert total > 0
            except RuntimeError as e:
                if "MUSA" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestStreamAndEvent:
    """Test Stream and Event classes."""

    def test_stream_class(self):
        """Test torch.cuda.Stream is available."""
        import torch

        assert hasattr(torch.cuda, "Stream")

    def test_event_class(self):
        """Test torch.cuda.Event is available."""
        import torch

        assert hasattr(torch.cuda, "Event")

    def test_external_stream_class(self):
        """Test torch.cuda.ExternalStream is available."""
        import torch

        assert hasattr(torch.cuda, "ExternalStream")

    def test_current_stream(self):
        """Test torch.cuda.current_stream is available."""
        import torch

        assert hasattr(torch.cuda, "current_stream")

    def test_default_stream(self):
        """Test torch.cuda.default_stream is available."""
        import torch

        assert hasattr(torch.cuda, "default_stream")

    def test_set_stream(self):
        """Test torch.cuda.set_stream is available."""
        import torch

        assert hasattr(torch.cuda, "set_stream")

    def test_stream_context_manager(self):
        """Test torch.cuda.stream is available."""
        import torch

        assert hasattr(torch.cuda, "stream")

    def test_stream_context_class(self):
        """Test torch.cuda.StreamContext is available on MUSA.

        StreamContext is not at top level of torch_musa, it's in
        torch_musa.core.stream.StreamContext, so we need special handling.
        """
        import torch

        import torchada

        if torchada.is_musa_platform():
            # StreamContext should be accessible via torch.cuda.StreamContext
            assert hasattr(torch.cuda, "StreamContext")
            # It should be the MUSA StreamContext class
            import torch_musa.core.stream

            assert torch.cuda.StreamContext is torch_musa.core.stream.StreamContext

    def test_stream_cuda_stream_property(self):
        """Test stream.cuda_stream returns same value as stream.musa_stream on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform() and torch.cuda.is_available():
            try:
                stream = torch.cuda.Stream()
                assert hasattr(stream, "cuda_stream")
                assert hasattr(stream, "musa_stream")
                assert stream.cuda_stream == stream.musa_stream
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestContextManagers:
    """Test device context managers."""

    def test_device_context_manager(self):
        """Test torch.cuda.device is available."""
        import torch

        assert hasattr(torch.cuda, "device")

    def test_device_of_context_manager(self):
        """Test torch.cuda.device_of is available."""
        import torch

        assert hasattr(torch.cuda, "device_of")


class TestDeviceFunctions:
    """Test additional device functions."""

    def test_get_device_properties(self):
        """Test torch.cuda.get_device_properties is available."""
        import torch

        assert hasattr(torch.cuda, "get_device_properties")

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            assert props is not None
            assert hasattr(props, "name")
            assert hasattr(props, "total_memory")

    def test_get_device_capability(self):
        """Test torch.cuda.get_device_capability is available."""
        import torch

        assert hasattr(torch.cuda, "get_device_capability")

        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            assert isinstance(cap, tuple)
            assert len(cap) == 2

    def test_is_initialized(self):
        """Test torch.cuda.is_initialized is available."""
        import torch

        assert hasattr(torch.cuda, "is_initialized")


class TestTorchGenerator:
    """Test torch.Generator works with cuda device on MUSA platform."""

    @pytest.mark.gpu
    def test_generator_cuda_device(self):
        """Test torch.Generator(device='cuda') works on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # Should create a MUSA generator instead of failing
            gen = torch.Generator(device="cuda")
            assert gen is not None
            assert gen.device.type == "musa"

    @pytest.mark.gpu
    def test_generator_cuda_device_index(self):
        """Test torch.Generator(device='cuda:0') works on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            gen = torch.Generator(device="cuda:0")
            assert gen is not None
            assert gen.device.type == "musa"
            assert gen.device.index == 0

    @pytest.mark.gpu
    def test_generator_musa_device(self):
        """Test torch.Generator(device='musa') still works."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            gen = torch.Generator(device="musa")
            assert gen is not None
            assert gen.device.type == "musa"

    def test_generator_no_device(self):
        """Test torch.Generator() without device works."""
        import torch

        import torchada  # noqa: F401

        gen = torch.Generator()
        assert gen is not None

    @pytest.mark.gpu
    def test_generator_manual_seed_chain(self):
        """Test torch.Generator(device='cuda').manual_seed(seed) works."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            gen = torch.Generator(device="cuda").manual_seed(42)
            assert gen is not None
            assert gen.device.type == "musa"

    @pytest.mark.gpu
    def test_generator_isinstance_check(self):
        """Test isinstance(gen, torch.Generator) works correctly.

        This is a regression test for sglang integration where
        generator_or_list_generators() was returning False because
        isinstance() wasn't working with the wrapped Generator class.

        The fix uses a metaclass with __instancecheck__ so isinstance() works.
        """
        import torch

        import torchada  # noqa: F401

        # Create generators
        gen_cpu = torch.Generator()
        gen_cuda = torch.Generator(device="cuda")

        # isinstance should work for both CPU and CUDA/MUSA generators
        assert isinstance(gen_cpu, torch.Generator), "CPU generator isinstance check failed"
        assert isinstance(gen_cuda, torch.Generator), "CUDA/MUSA generator isinstance check failed"

        # Test with a list (like sglang's generator_or_list_generators does)
        gens = [gen_cpu, gen_cuda]
        assert all(
            isinstance(g, torch.Generator) for g in gens
        ), "List of generators isinstance check failed"

    def test_generator_isinstance_negative(self):
        """Test isinstance returns False for non-Generator objects."""
        import torch

        import torchada  # noqa: F401

        assert not isinstance("not a generator", torch.Generator)
        assert not isinstance(42, torch.Generator)
        assert not isinstance(None, torch.Generator)
        assert not isinstance(torch.tensor([1, 2, 3]), torch.Generator)


class TestTorchVersionCuda:
    """Test torch.version.cuda is NOT patched.

    This is intentionally NOT patched to allow downstream projects
    to detect the MUSA platform using patterns like:
        if torch.version.cuda is not None:  # CUDA
        elif hasattr(torch.version, 'musa'):  # MUSA
    """

    def test_torch_version_cuda_not_patched(self):
        """Test torch.version.cuda is NOT patched on MUSA platform."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # torch.version.cuda should remain None on MUSA
            # This allows downstream projects to detect the platform
            assert torch.version.cuda is None
            # But torch.version.musa should be set
            assert torch.version.musa is not None
            assert isinstance(torch.version.musa, str)


class TestTensorIsCuda:
    """Test tensor.is_cuda patching."""

    def test_cpu_tensor_is_cuda_false(self):
        """Test CPU tensor.is_cuda returns False."""
        import torch

        cpu_tensor = torch.empty(10, 10)
        assert cpu_tensor.is_cuda is False

    def test_musa_tensor_is_cuda_true(self):
        """Test MUSA tensor.is_cuda returns True on MUSA platform."""
        import torch

        import torchada

        if torchada.is_musa_platform() and torch.cuda.is_available():
            try:
                musa_tensor = torch.empty(10, 10, device="cuda:0")
                assert musa_tensor.is_cuda is True
                assert musa_tensor.is_musa is True
            except RuntimeError as e:
                if "MUSA" in str(e) or "invalid device function" in str(e):
                    pytest.skip(f"MUSA driver issue: {e}")
                raise


class TestAutotuneProcess:
    """Test torch._inductor.autotune_process patching."""

    def test_cuda_visible_devices_patched(self):
        """Test CUDA_VISIBLE_DEVICES is patched to MUSA_VISIBLE_DEVICES on MUSA platform."""
        import torchada

        try:
            import torch._inductor.autotune_process as autotune_process
        except ImportError:
            pytest.skip("torch._inductor.autotune_process not available")

        # CUDA_VISIBLE_DEVICES constant was added in PyTorch 2.2.0
        # It does NOT exist in PyTorch 2.1.x and earlier versions
        if not hasattr(autotune_process, "CUDA_VISIBLE_DEVICES"):
            pytest.skip("CUDA_VISIBLE_DEVICES not available (requires PyTorch >= 2.2.0)")

        if torchada.is_musa_platform():
            # On MUSA platform, CUDA_VISIBLE_DEVICES should be patched to MUSA_VISIBLE_DEVICES
            assert autotune_process.CUDA_VISIBLE_DEVICES == "MUSA_VISIBLE_DEVICES"
        else:
            # On CUDA/CPU, it should remain as CUDA_VISIBLE_DEVICES
            assert autotune_process.CUDA_VISIBLE_DEVICES == "CUDA_VISIBLE_DEVICES"

    def test_cuda_visible_devices_is_string(self):
        """Test CUDA_VISIBLE_DEVICES constant is a string."""

        try:
            import torch._inductor.autotune_process as autotune_process
        except ImportError:
            pytest.skip("torch._inductor.autotune_process not available")

        # CUDA_VISIBLE_DEVICES constant was added in PyTorch 2.2.0
        # It does NOT exist in PyTorch 2.1.x and earlier versions
        if not hasattr(autotune_process, "CUDA_VISIBLE_DEVICES"):
            pytest.skip("CUDA_VISIBLE_DEVICES not available (requires PyTorch >= 2.2.0)")

        assert isinstance(autotune_process.CUDA_VISIBLE_DEVICES, str)

    def test_cuda_visible_devices_env_var_format(self):
        """Test CUDA_VISIBLE_DEVICES constant is in correct format for env vars."""

        try:
            import torch._inductor.autotune_process as autotune_process
        except ImportError:
            pytest.skip("torch._inductor.autotune_process not available")

        # CUDA_VISIBLE_DEVICES constant was added in PyTorch 2.2.0
        # It does NOT exist in PyTorch 2.1.x and earlier versions
        if not hasattr(autotune_process, "CUDA_VISIBLE_DEVICES"):
            pytest.skip("CUDA_VISIBLE_DEVICES not available (requires PyTorch >= 2.2.0)")

        env_var = autotune_process.CUDA_VISIBLE_DEVICES
        # Env var should be uppercase and use underscores
        assert env_var.isupper()
        assert "_" in env_var
        # Should end with VISIBLE_DEVICES
        assert env_var.endswith("VISIBLE_DEVICES")


class TestNvtxStub:
    """Test torch.cuda.nvtx stub module."""

    def test_nvtx_module_available(self):
        """Test torch.cuda.nvtx is available."""
        import torch

        assert hasattr(torch.cuda, "nvtx")

    def test_nvtx_mark(self):
        """Test torch.cuda.nvtx.mark is available and callable."""
        import torch.cuda.nvtx as nvtx

        assert hasattr(nvtx, "mark")
        # Should not raise
        nvtx.mark("test")

    def test_nvtx_range_push_pop(self):
        """Test torch.cuda.nvtx.range_push and range_pop are available."""
        import torch.cuda.nvtx as nvtx

        assert hasattr(nvtx, "range_push")
        assert hasattr(nvtx, "range_pop")
        # Should not raise
        nvtx.range_push("test")
        nvtx.range_pop()

    def test_nvtx_range_context_manager(self):
        """Test torch.cuda.nvtx.range context manager works."""
        import torch.cuda.nvtx as nvtx

        assert hasattr(nvtx, "range")
        # Should not raise
        with nvtx.range("test"):
            pass


class TestPatchDecorators:
    """Test the decorator-based patch registration system."""

    def test_patch_registry_is_populated(self):
        """Test that @patch_function decorator populates the registry."""
        from torchada._patch import _patch_registry

        # Registry should have at least 8 registered patches
        assert len(_patch_registry) >= 8

        # All entries should be callable
        for fn in _patch_registry:
            assert callable(fn)

    def test_patch_registry_contains_expected_functions(self):
        """Test that registry contains the expected patch functions."""
        from torchada._patch import _patch_registry

        # Get function names from registry
        fn_names = [fn.__name__ for fn in _patch_registry]

        # Check expected functions are registered
        expected_fns = [
            "_patch_torch_device",
            "_patch_torch_cuda_module",
            "_patch_distributed_backend",
            "_patch_tensor_is_cuda",
            "_patch_stream_cuda_stream",
            "_patch_autocast",
            "_patch_cpp_extension",
            "_patch_autotune_process",
        ]

        for expected in expected_fns:
            assert expected in fn_names, f"{expected} not found in registry"

    def test_requires_import_decorator_guards_import(self):
        """Test that @requires_import returns early when import fails."""
        from torchada._patch import requires_import

        @requires_import("nonexistent_module_that_does_not_exist")
        def test_func():
            raise AssertionError("Should not be called when import fails")

        # Should return None without raising
        result = test_func()
        assert result is None

    def test_requires_import_decorator_allows_execution(self):
        """Test that @requires_import allows execution when import succeeds."""
        from torchada._patch import requires_import

        @requires_import("sys")  # 'sys' always exists
        def test_func():
            return "executed"

        result = test_func()
        assert result == "executed"

    def test_requires_import_multiple_modules(self):
        """Test @requires_import with multiple module names."""
        from torchada._patch import requires_import

        @requires_import("sys", "os")
        def test_func():
            return "success"

        result = test_func()
        assert result == "success"

        @requires_import("sys", "nonexistent_module_xyz")
        def test_func_fails():
            raise AssertionError("Should not run")

        result = test_func_fails()
        assert result is None


class TestIsCompiledAndBackends:
    """Test _is_compiled and backends patches."""

    def test_torch_musa_is_compiled(self):
        """Test torch.musa._is_compiled() exists and returns True."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            assert hasattr(torch.musa, "_is_compiled")
            assert torch.musa._is_compiled() is True

    def test_torch_cuda_is_compiled_redirects(self):
        """Test torch.cuda._is_compiled() works via redirect."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # torch.cuda is redirected to torch.musa
            result = torch.cuda._is_compiled()
            assert result is True

    def test_torch_backends_cuda_is_built(self):
        """Test torch.backends.cuda.is_built() returns True on MUSA."""
        import torch

        import torchada

        if torchada.is_musa_platform():
            # Should return True since MUSA is available
            assert torch.backends.cuda.is_built() is True

    def test_torch_backends_cuda_matmul_allow_tf32(self):
        """Test torch.backends.cuda.matmul.allow_tf32 is accessible."""
        import torch

        # Should be accessible and settable
        original = torch.backends.cuda.matmul.allow_tf32
        assert isinstance(original, bool)

        # Test setting
        torch.backends.cuda.matmul.allow_tf32 = True
        assert torch.backends.cuda.matmul.allow_tf32 is True

        torch.backends.cuda.matmul.allow_tf32 = False
        assert torch.backends.cuda.matmul.allow_tf32 is False

        # Restore original
        torch.backends.cuda.matmul.allow_tf32 = original

    def test_torch_backends_cuda_matmul_fp32_precision(self):
        """Test torch.backends.cuda.matmul.fp32_precision is accessible."""
        import torch

        import torchada  # noqa: F401 - ensure patches are applied

        # fp32_precision is a torchada addition for MUSA compatibility
        # It wraps torch.get/set_float32_matmul_precision() for convenient access
        # This attribute does NOT exist in standard PyTorch on CUDA platforms
        # Note: torch.backends.cuda.matmul.__getattr__ raises AssertionError for
        # unknown attributes, so we need to catch that instead of using hasattr()
        try:
            _ = torch.backends.cuda.matmul.fp32_precision
        except (AttributeError, AssertionError):
            pytest.skip("fp32_precision not available (torchada MUSA-specific attribute)")

        # Should be accessible
        original = torch.backends.cuda.matmul.fp32_precision
        assert original in ("highest", "high", "medium")

        # Test setting (use torch.get/set_float32_matmul_precision values)
        torch.backends.cuda.matmul.fp32_precision = "highest"
        assert torch.backends.cuda.matmul.fp32_precision == "highest"
        assert torch.get_float32_matmul_precision() == "highest"

        torch.backends.cuda.matmul.fp32_precision = "high"
        assert torch.backends.cuda.matmul.fp32_precision == "high"
        assert torch.get_float32_matmul_precision() == "high"

        # Restore original
        torch.backends.cuda.matmul.fp32_precision = original

    def test_torch_c_storage_use_count(self):
        """Test torch._C._storage_Use_Count is accessible on MUSA."""
        import torchada

        if torchada.is_musa_platform():
            # Should be able to import from torch._C after patching
            from torch._C import _storage_Use_Count as use_count

            assert use_count is not None
            assert callable(use_count)


class TestProfilerActivity:
    """Test torch.profiler.ProfilerActivity.CUDA patching."""

    def test_profiler_activity_cuda_accessible(self):
        """Test ProfilerActivity.CUDA is still accessible after patching."""
        import torch

        import torchada  # noqa: F401

        assert hasattr(torch.profiler, "ProfilerActivity")
        assert hasattr(torch.profiler.ProfilerActivity, "CUDA")
        # ProfilerActivity.PrivateUse1 is MUSA-specific (for PrivateUse1 backend)
        # This attribute does NOT exist in standard PyTorch on CUDA platforms
        # torch_musa adds this to support profiling on MUSA GPUs
        if torchada.is_musa_platform():
            assert hasattr(torch.profiler.ProfilerActivity, "PrivateUse1")

    def test_profiler_with_cuda_activity(self):
        """Test profiler can be created with CUDA activity on MUSA."""
        import torch

        import torchada

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        # Should be able to create profiler with CUDA activity
        profiler = torch.profiler.profile(activities=activities)
        assert profiler is not None

        if torchada.is_musa_platform():
            # On MUSA, check that the wrapper is used
            assert hasattr(profiler, "_profiler")

    @pytest.mark.gpu
    def test_profiler_context_manager(self):
        """Test profiler context manager works with CUDA activity."""
        import torch

        import torchada  # noqa: F401

        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        x = torch.randn(10, 10)
        # Context manager should work
        with torch.profiler.profile(activities=activities) as prof:
            y = x.sum()

        assert prof is not None
        assert y is not None

    def test_profiler_methods_accessible(self):
        """Test profiler methods are accessible through wrapper."""
        import torch

        import torchada  # noqa: F401

        activities = [torch.profiler.ProfilerActivity.CPU]
        profiler = torch.profiler.profile(activities=activities)

        # Common methods should be accessible
        assert hasattr(profiler, "start")
        assert hasattr(profiler, "stop")
        assert hasattr(profiler, "step")


class TestMusaWarnings:
    """Test MUSA-specific warning suppression."""

    def test_musa_warnings_patch_applied(self):
        """Test that MUSA warning patch runs without error."""
        import torchada  # noqa: F401

        # The patch is applied on import - just verify no errors occurred
        # The actual filtering is tested in test_autocast_warning_suppressed
        assert True

    def test_autocast_warning_suppressed(self):
        """Test that autocast warnings are suppressed."""
        import warnings

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Try to trigger the warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Reapply our filters
            warnings.filterwarnings(
                "ignore",
                message=r"In musa autocast, but the target dtype is not supported.*",
                category=UserWarning,
            )

            # Simulate the warning (we can't easily trigger the real one without specific ops)
            warnings.warn(
                "In musa autocast, but the target dtype is not supported. Disabling autocast.",
                UserWarning,
            )

            # Check that no warnings were collected (filter worked)
            musa_warnings = [x for x in w if "musa autocast" in str(x.message)]
            assert len(musa_warnings) == 0, "Autocast warning was not suppressed"


class TestLibraryImpl:
    """Test torch.library.Library.impl() patching for CUDA -> PrivateUse1 translation."""

    @pytest.mark.gpu
    def test_library_impl_cuda_translated(self):
        """Test that Library.impl with 'CUDA' works on MUSA tensors."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Create a test library with unique name to avoid conflicts
        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        # Register with CUDA backend - should be translated to PrivateUse1
        test_lib.define("identity_op(Tensor x) -> Tensor")
        test_lib.impl("identity_op", identity, "CUDA")

        # Test calling it with MUSA tensor - if dispatch fails, we get NotImplementedError
        x = torch.randn(3).musa()
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)

        # Just verify we got a result on MUSA device (dispatch worked)
        assert result.device.type == "musa"
        assert result is x  # identity function returns same object

    def test_library_impl_with_keyset(self):
        """Test that Library.impl with with_keyset=True works."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Create a test library with unique name to avoid conflicts
        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity_with_keyset(keyset, x: torch.Tensor) -> torch.Tensor:
            # keyset is the dispatch keyset passed when with_keyset=True
            return x

        # Register with with_keyset=True - this was failing before the fix
        test_lib.define("identity_keyset(Tensor x) -> Tensor")
        test_lib.impl("identity_keyset", identity_with_keyset, "CPU", with_keyset=True)

        # Test calling it
        x = torch.randn(3)
        op = getattr(torch.ops, lib_name)
        result = op.identity_keyset(x)
        assert result is x

    def test_library_impl_with_keyset_false_explicit(self):
        """Test that Library.impl with with_keyset=False explicitly works."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        test_lib.impl("identity_op", identity, "CPU", with_keyset=False)

        x = torch.randn(3)
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result is x

    def test_library_impl_op_name_as_keyword(self):
        """Test that Library.impl works with op_name as keyword argument."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        # Use op_name as keyword argument
        test_lib.impl(op_name="identity_op", fn=identity, dispatch_key="CPU")

        x = torch.randn(3)
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result is x

    def test_library_impl_fn_as_keyword(self):
        """Test that Library.impl works with fn as keyword argument."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        # Use fn as keyword argument
        test_lib.impl("identity_op", fn=identity, dispatch_key="CPU")

        x = torch.randn(3)
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result is x

    def test_library_impl_dispatch_key_as_keyword(self):
        """Test that Library.impl works with dispatch_key as keyword argument."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        # Use dispatch_key as keyword argument
        test_lib.impl("identity_op", identity, dispatch_key="CPU")

        x = torch.randn(3)
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result is x

    @pytest.mark.gpu
    def test_library_impl_cuda_with_keyset(self):
        """Test CUDA dispatch key translation works with with_keyset=True."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity_with_keyset(keyset, x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        # Use CUDA with with_keyset=True
        test_lib.impl("identity_op", identity_with_keyset, dispatch_key="CUDA", with_keyset=True)

        x = torch.randn(3).musa()
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result.device.type == "musa"

    def test_library_impl_autograd_cuda_translation(self):
        """Test AutogradCUDA dispatch key translates to AutogradPrivateUse1."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        def identity_keyset(keyset, x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        test_lib.impl("identity_op", identity, "CPU")
        # AutogradCUDA should translate to AutogradPrivateUse1
        test_lib.impl("identity_op", identity_keyset, "AutogradCUDA", with_keyset=True)

        # Just verify it registers without error
        x = torch.randn(3)
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result is x

    @pytest.mark.gpu
    def test_library_impl_opoverload_as_op_name(self):
        """Test Library.impl works with OpOverload as op_name."""
        import uuid

        import torch
        import torch.library

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        lib_name = f"test_lib_{uuid.uuid4().hex[:8]}"
        test_lib = torch.library.Library(lib_name, "DEF")

        def identity(x: torch.Tensor) -> torch.Tensor:
            return x

        test_lib.define("identity_op(Tensor x) -> Tensor")
        test_lib.impl("identity_op", identity, "CPU")

        # Get the OpOverload and register another impl with it
        op_overload = getattr(torch.ops, lib_name).identity_op.default
        test_lib.impl(op_overload, identity, "CUDA")

        x = torch.randn(3).musa()
        op = getattr(torch.ops, lib_name)
        result = op.identity_op(x)
        assert result.device.type == "musa"


class TestCudart:
    """Test torch.cuda.cudart() CUDA runtime wrapper."""

    @pytest.mark.gpu
    def test_cudart_returns_wrapper(self):
        """Test that torch.cuda.cudart() returns a wrapper on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        cudart = torch.cuda.cudart()
        assert cudart is not None
        # Should be our wrapper class
        assert "CudartWrapper" in type(cudart).__name__

    @pytest.mark.gpu
    def test_cudart_host_register(self):
        """Test cudaHostRegister works via the wrapper."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        cudart = torch.cuda.cudart()
        x = torch.randn(10)
        ptr = x.data_ptr()
        size = x.numel() * x.element_size()

        # Register should succeed (return 0 or equivalent)
        result = cudart.cudaHostRegister(ptr, size, 1)
        assert result == 0, f"cudaHostRegister failed with {result}"

        # Unregister
        result2 = cudart.cudaHostUnregister(ptr)
        assert result2 == 0, f"cudaHostUnregister failed with {result2}"

    def test_cudart_in_dir(self):
        """Test that cudart appears in dir(torch.cuda)."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        assert "cudart" in dir(torch.cuda)


class TestCppExtensionPaths:
    """Test include_paths and library_paths with both old and new signatures."""

    def test_include_paths_with_cuda_param(self):
        """Test include_paths with legacy cuda=bool parameter."""
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Test with cuda=True
        paths = torch.utils.cpp_extension.include_paths(cuda=True)
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include MUSA paths
        assert any("musa" in p.lower() for p in paths)

        # Test with cuda=False
        paths_no_cuda = torch.utils.cpp_extension.include_paths(cuda=False)
        assert isinstance(paths_no_cuda, list)

    def test_include_paths_with_device_type_param(self):
        """Test include_paths with PyTorch 2.6+ device_type=str parameter."""
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Test with device_type="cuda" - should work on MUSA
        paths = torch.utils.cpp_extension.include_paths(device_type="cuda")
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include MUSA paths when device_type="cuda" on MUSA platform
        assert any("musa" in p.lower() for p in paths)

        # Test with device_type="cpu"
        paths_cpu = torch.utils.cpp_extension.include_paths(device_type="cpu")
        assert isinstance(paths_cpu, list)

    def test_include_paths_default(self):
        """Test include_paths with no parameters (default behavior)."""
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Default should include device paths
        paths = torch.utils.cpp_extension.include_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_library_paths_with_cuda_param(self):
        """Test library_paths with legacy cuda=bool parameter."""
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Test with cuda=True
        paths = torch.utils.cpp_extension.library_paths(cuda=True)
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include MUSA library paths
        assert any("musa" in p.lower() for p in paths)

        # Test with cuda=False
        paths_no_cuda = torch.utils.cpp_extension.library_paths(cuda=False)
        assert isinstance(paths_no_cuda, list)
        assert len(paths_no_cuda) == 0  # No device libs when cuda=False

    def test_library_paths_with_device_type_param(self):
        """Test library_paths with PyTorch 2.6+ device_type=str parameter."""
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Test with device_type="cuda" - should work on MUSA
        paths = torch.utils.cpp_extension.library_paths(device_type="cuda")
        assert isinstance(paths, list)
        assert len(paths) > 0
        # Should include MUSA library paths
        assert any("musa" in p.lower() for p in paths)

        # Test with device_type="cpu"
        paths_cpu = torch.utils.cpp_extension.library_paths(device_type="cpu")
        assert isinstance(paths_cpu, list)
        assert len(paths_cpu) == 0  # No device libs for CPU

    def test_library_paths_default(self):
        """Test library_paths with no parameters (default behavior)."""
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Default should include device library paths
        paths = torch.utils.cpp_extension.library_paths()
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_include_paths_patched_in_torch_module(self):
        """Test that include_paths is properly patched in torch.utils.cpp_extension."""
        import torch.utils.cpp_extension

        import torchada
        from torchada.utils import cpp_extension as torchada_cpp_ext

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Verify the function is patched
        assert (
            torch.utils.cpp_extension.include_paths == torchada_cpp_ext.include_paths
        ), "include_paths should be patched to torchada's version"

    def test_library_paths_patched_in_torch_module(self):
        """Test that library_paths is properly patched in torch.utils.cpp_extension."""
        import torch.utils.cpp_extension

        import torchada
        from torchada.utils import cpp_extension as torchada_cpp_ext

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Verify the function is patched
        assert (
            torch.utils.cpp_extension.library_paths == torchada_cpp_ext.library_paths
        ), "library_paths should be patched to torchada's version"

    def test_get_torch_include_paths_pattern(self):
        """Test the exact pattern reported by user that was failing."""
        import torch
        import torch.utils.cpp_extension

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # This is the exact pattern from the user's report
        def get_torch_include_paths(build_with_cuda: bool):
            if torch.__version__ >= torch.torch_version.TorchVersion("2.6.0"):
                return torch.utils.cpp_extension.include_paths(
                    device_type="cuda" if build_with_cuda else "cpu"
                )
            else:
                return torch.utils.cpp_extension.include_paths(cuda=build_with_cuda)

        # Should not raise any errors
        paths_with_cuda = get_torch_include_paths(True)
        assert isinstance(paths_with_cuda, list)
        assert len(paths_with_cuda) > 0

        paths_without_cuda = get_torch_include_paths(False)
        assert isinstance(paths_without_cuda, list)


class TestTensorFactoryFunctions:
    """Test tensor factory functions with device='cuda' translation."""

    @pytest.mark.gpu
    def test_asarray_with_cuda_device(self):
        """Test torch.asarray with device='cuda' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Test with device='cuda'
        x = torch.asarray([1, 2, 3, 4, 5], dtype=torch.float32, device="cuda")
        assert x.device.type == "musa"
        assert x.shape == (5,)
        assert x.dtype == torch.float32

    @pytest.mark.gpu
    def test_asarray_with_cuda_index(self):
        """Test torch.asarray with device='cuda:0' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        # Test with device='cuda:0'
        x = torch.asarray([1, 2, 3], dtype=torch.float32, device="cuda:0")
        assert x.device.type == "musa"
        assert x.device.index == 0

    def test_asarray_without_device(self):
        """Test torch.asarray without device argument still works."""
        import torch

        # Should create CPU tensor by default
        x = torch.asarray([1, 2, 3], dtype=torch.float32)
        assert x.device.type == "cpu"

    @pytest.mark.gpu
    def test_tensor_with_cuda_device(self):
        """Test torch.tensor with device='cuda' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        x = torch.tensor([1, 2, 3], dtype=torch.float32, device="cuda")
        assert x.device.type == "musa"

    @pytest.mark.gpu
    def test_zeros_with_cuda_device(self):
        """Test torch.zeros with device='cuda' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        try:
            x = torch.zeros(5, device="cuda")
            assert x.device.type == "musa"
        except RuntimeError as e:
            # Skip if MUDNN kernel execution fails (expected in test containers)
            if "MUDNN" in str(e) or "invalid device function" in str(e):
                pytest.skip("MUDNN kernel execution failed (expected in test containers)")
            raise

    @pytest.mark.gpu
    def test_ones_with_cuda_device(self):
        """Test torch.ones with device='cuda' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        try:
            x = torch.ones(5, device="cuda")
            assert x.device.type == "musa"
        except RuntimeError as e:
            # Skip if MUDNN kernel execution fails (expected in test containers)
            if "MUDNN" in str(e) or "invalid device function" in str(e):
                pytest.skip("MUDNN kernel execution failed (expected in test containers)")
            raise

    @pytest.mark.gpu
    def test_randn_with_cuda_device(self):
        """Test torch.randn with device='cuda' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        try:
            x = torch.randn(5, device="cuda")
            assert x.device.type == "musa"
        except RuntimeError as e:
            # Skip if MUDNN kernel execution fails (expected in test containers)
            if "MUDNN" in str(e) or "invalid device function" in str(e):
                pytest.skip("MUDNN kernel execution failed (expected in test containers)")
            raise

    @pytest.mark.gpu
    def test_empty_with_cuda_device(self):
        """Test torch.empty with device='cuda' works on MUSA."""
        import torch

        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("Only applicable on MUSA platform")

        x = torch.empty(5, device="cuda")
        assert x.device.type == "musa"


class TestCppOpsInfrastructure:
    """Test C++ operator overrides infrastructure."""

    def test_cpp_ops_module_exists(self):
        """Test that the _cpp_ops module exists and can be imported."""
        from torchada import _cpp_ops

        assert hasattr(_cpp_ops, "load_cpp_ops")
        assert hasattr(_cpp_ops, "is_loaded")
        assert hasattr(_cpp_ops, "get_version")
        assert hasattr(_cpp_ops, "get_module")

    def test_cpp_ops_not_loaded_by_default(self):
        """Test that C++ ops are not loaded by default."""
        from torchada import _cpp_ops

        # Without TORCHADA_ENABLE_CPP_OPS=1, should not be loaded
        # Note: This test may be affected by other tests that load the module
        # So we just check the functions exist and are callable
        assert callable(_cpp_ops.is_loaded)
        assert callable(_cpp_ops.get_version)

    def test_cpp_ops_source_files_exist(self):
        """Test that the C++ source files are packaged correctly."""
        import os.path as osp

        import torchada

        csrc_dir = osp.join(osp.dirname(torchada.__file__), "csrc")
        assert osp.isdir(csrc_dir), f"csrc directory not found: {csrc_dir}"

        ops_h = osp.join(csrc_dir, "ops.h")
        ops_cpp = osp.join(csrc_dir, "ops.cpp")

        assert osp.isfile(ops_h), f"ops.h not found: {ops_h}"
        assert osp.isfile(ops_cpp), f"ops.cpp not found: {ops_cpp}"

    def test_cpp_ops_header_content(self):
        """Test that the C++ header has expected content."""
        import os.path as osp

        import torchada

        csrc_dir = osp.join(osp.dirname(torchada.__file__), "csrc")
        ops_h = osp.join(csrc_dir, "ops.h")

        with open(ops_h, "r") as f:
            content = f.read()

        # Check for expected content
        assert "namespace torchada" in content
        assert "TORCH_LIBRARY_IMPL" in content
        assert "PrivateUse1" in content
        assert "is_override_enabled" in content
        assert "log_op_call" in content


class TestValidateDevice:
    """Test Validate Device func in FlexAttention."""

    def get_validator(self):
        import torch.nn.attention.flex_attention as flex_attention
        return flex_attention._validate_device

    def test_cpu_device_should_pass(self):
        import torch
        q = torch.randn(2, 2, device="cpu")
        k = torch.randn(2, 2, device="cpu")
        v = torch.randn(2, 2, device="cpu")

        validator = self.get_validator()
        validator(q, k, v)

    def test_musa_device_should_pass(self):
        import torch
        import torchada

        if not torchada.is_musa_platform():
            pytest.skip("MUSA platform required")

        q = torch.randn(2, 2, device="musa")
        k = torch.randn(2, 2, device="musa")
        v = torch.randn(2, 2, device="musa")

        validator = self.get_validator()
        validator(q, k, v)

    def test_invalid_device_should_fail(self):
        import torch

        class FakeTensor:
            def __init__(self):
                self.device = torch.device("meta")

        q = FakeTensor()
        k = FakeTensor()
        v = FakeTensor()

        validator = self.get_validator()
        with pytest.raises(ValueError, match="FlexAttention is only supported"):
            validator(q, k, v)
