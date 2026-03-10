"""
Tests for CUDA to MUSA mapping rules.
"""

import pytest


class TestMappingImports:
    """Test mapping module imports."""

    def test_import_mapping_rule(self):
        """Test _MAPPING_RULE can be imported."""
        from torchada._mapping import _MAPPING_RULE

        assert isinstance(_MAPPING_RULE, dict)
        assert len(_MAPPING_RULE) > 0

    def test_import_ext_replaced_mapping(self):
        """Test EXT_REPLACED_MAPPING can be imported."""
        from torchada._mapping import EXT_REPLACED_MAPPING

        assert isinstance(EXT_REPLACED_MAPPING, dict)


class TestATenMappings:
    """Test ATen CUDA to MUSA mappings."""

    def test_aten_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["at::cuda"] == "at::musa"

    def test_aten_cuda_includes(self):
        from torchada._mapping import _MAPPING_RULE

        # Check for specific ATen include mappings
        assert "#include <ATen/cuda/CUDAContext.h>" in _MAPPING_RULE


class TestC10Mappings:
    """Test C10 CUDA to MUSA mappings."""

    def test_c10_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["c10::cuda"] == "c10::musa"
        assert _MAPPING_RULE["c10/cuda"] == "c10/musa"

    def test_c10_device_type(self):
        from torchada._mapping import _MAPPING_RULE

        # MUSA uses PrivateUse1 as its device type
        assert _MAPPING_RULE["c10::DeviceType::CUDA"] == "c10::DeviceType::PrivateUse1"


class TestTorchMappings:
    """Test torch namespace mappings."""

    def test_torch_cuda_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["torch::cuda"] == "torch::musa"
        assert _MAPPING_RULE["torch.cuda"] == "torch.musa"

    def test_torch_device_type(self):
        from torchada._mapping import _MAPPING_RULE

        # All CUDA device type symbols map to PrivateUse1
        assert _MAPPING_RULE["at::kCUDA"] == "at::kPrivateUse1"
        assert _MAPPING_RULE["at::DeviceType::CUDA"] == "at::DeviceType::PrivateUse1"
        assert _MAPPING_RULE["torch::kCUDA"] == "torch::kPrivateUse1"


class TestCuBLASMappings:
    """Test cuBLAS to muBLAS mappings."""

    def test_cublas_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublas"] == "mublas"
        assert _MAPPING_RULE["CUBLAS"] == "MUBLAS"

    def test_cublas_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasHandle_t"] == "mublasHandle_t"
        assert _MAPPING_RULE["cublasStatus_t"] == "mublasStatus_t"

    def test_cublas_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasCreate"] == "mublasCreate"
        assert _MAPPING_RULE["cublasDestroy"] == "mublasDestroy"
        assert _MAPPING_RULE["cublasSetStream"] == "mublasSetStream"
        assert _MAPPING_RULE["cublasGetStream"] == "mublasGetStream"

    def test_cublas_gemm(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasSgemm"] == "mublasSgemm"
        assert _MAPPING_RULE["cublasDgemm"] == "mublasDgemm"
        assert _MAPPING_RULE["cublasHgemm"] == "mublasHgemm"
        assert _MAPPING_RULE["cublasGemmEx"] == "mublasGemmEx"

    def test_cublas_batched(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasGemmBatchedEx"] == "mublasGemmBatchedEx"
        assert _MAPPING_RULE["cublasGemmStridedBatchedEx"] == "mublasGemmStridedBatchedEx"
        assert _MAPPING_RULE["cublasSgemmBatched"] == "mublasSgemmBatched"
        assert _MAPPING_RULE["cublasDgemmBatched"] == "mublasDgemmBatched"

    def test_cublaslt(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cublasLtCreate"] == "mublasLtCreate"
        assert _MAPPING_RULE["cublasLtDestroy"] == "mublasLtDestroy"
        assert _MAPPING_RULE["cublasLtHandle_t"] == "mublasLtHandle_t"
        assert _MAPPING_RULE["cublasLtMatmul"] == "mublasLtMatmul"


class TestCuRANDMappings:
    """Test cuRAND to muRAND mappings."""

    def test_curand_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["curand"] == "murand"
        assert _MAPPING_RULE["CURAND"] == "MURAND"

    def test_curand_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["curandState"] == "murandState"
        assert _MAPPING_RULE["curandStatePhilox4_32_10_t"] == "murandStatePhilox4_32_10_t"

    def test_curand_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["curand_init"] == "murand_init"
        assert _MAPPING_RULE["curand_uniform"] == "murand_uniform"
        assert _MAPPING_RULE["curand_uniform4"] == "murand_uniform4"
        assert _MAPPING_RULE["curand_normal"] == "murand_normal"
        assert _MAPPING_RULE["curand_normal4"] == "murand_normal4"


class TestCuDNNMappings:
    """Test cuDNN to muDNN mappings."""

    def test_cudnn_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudnn"] == "mudnn"
        assert _MAPPING_RULE["CUDNN"] == "MUDNN"

    def test_cudnn_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudnnHandle_t"] == "mudnnHandle_t"
        assert _MAPPING_RULE["cudnnStatus_t"] == "mudnnStatus_t"

    def test_cudnn_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudnnCreate"] == "mudnnCreate"
        assert _MAPPING_RULE["cudnnDestroy"] == "mudnnDestroy"
        assert _MAPPING_RULE["cudnnSetStream"] == "mudnnSetStream"


class TestCUDARuntimeMappings:
    """Test CUDA runtime to MUSA runtime mappings."""

    def test_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaMalloc"] == "musaMalloc"
        assert _MAPPING_RULE["cudaFree"] == "musaFree"
        assert _MAPPING_RULE["cudaMemcpy"] == "musaMemcpy"
        assert _MAPPING_RULE["cudaMemcpyAsync"] == "musaMemcpyAsync"
        assert _MAPPING_RULE["cudaMemset"] == "musaMemset"
        assert _MAPPING_RULE["cudaMemsetAsync"] == "musaMemsetAsync"

    def test_host_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaHostAlloc"] == "musaHostAlloc"
        assert _MAPPING_RULE["cudaHostFree"] == "musaHostFree"
        assert _MAPPING_RULE["cudaMallocHost"] == "musaMallocHost"
        assert _MAPPING_RULE["cudaFreeHost"] == "musaFreeHost"
        assert _MAPPING_RULE["cudaMallocManaged"] == "musaMallocManaged"

    def test_async_memory_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaMallocAsync"] == "musaMallocAsync"
        assert _MAPPING_RULE["cudaFreeAsync"] == "musaFreeAsync"
        assert _MAPPING_RULE["cudaMemcpy2D"] == "musaMemcpy2D"
        assert _MAPPING_RULE["cudaMemcpy2DAsync"] == "musaMemcpy2DAsync"
        assert _MAPPING_RULE["cudaMemcpy3D"] == "musaMemcpy3D"
        assert _MAPPING_RULE["cudaMemcpy3DAsync"] == "musaMemcpy3DAsync"

    def test_device_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaDeviceSynchronize"] == "musaDeviceSynchronize"
        assert _MAPPING_RULE["cudaGetDevice"] == "musaGetDevice"
        assert _MAPPING_RULE["cudaSetDevice"] == "musaSetDevice"
        assert _MAPPING_RULE["cudaGetDeviceCount"] == "musaGetDeviceCount"
        assert _MAPPING_RULE["cudaGetDeviceProperties"] == "musaGetDeviceProperties"
        assert _MAPPING_RULE["cudaDeviceGetAttribute"] == "musaDeviceGetAttribute"

    def test_memcpy_kinds(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaMemcpyHostToDevice"] == "musaMemcpyHostToDevice"
        assert _MAPPING_RULE["cudaMemcpyDeviceToHost"] == "musaMemcpyDeviceToHost"
        assert _MAPPING_RULE["cudaMemcpyDeviceToDevice"] == "musaMemcpyDeviceToDevice"
        assert _MAPPING_RULE["cudaMemcpyHostToHost"] == "musaMemcpyHostToHost"

    def test_memory_constants(self):
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["cudaFuncAttributeMaxDynamicSharedMemorySize"]
            == "musaFuncAttributeMaxDynamicSharedMemorySize"
        )


class TestStreamEventMappings:
    """Test CUDA stream/event mappings."""

    def test_stream_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaStream_t"] == "musaStream_t"
        assert _MAPPING_RULE["cudaEvent_t"] == "musaEvent_t"

    def test_stream_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaStreamCreate"] == "musaStreamCreate"
        assert _MAPPING_RULE["cudaStreamDestroy"] == "musaStreamDestroy"
        assert _MAPPING_RULE["cudaStreamSynchronize"] == "musaStreamSynchronize"
        assert _MAPPING_RULE["cudaStreamQuery"] == "musaStreamQuery"
        assert _MAPPING_RULE["cudaStreamWaitEvent"] == "musaStreamWaitEvent"

    def test_stream_flags(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaStreamDefault"] == "musaStreamDefault"
        assert _MAPPING_RULE["cudaStreamNonBlocking"] == "musaStreamNonBlocking"
        assert _MAPPING_RULE["cudaStreamCreateWithFlags"] == "musaStreamCreateWithFlags"
        assert _MAPPING_RULE["cudaStreamCreateWithPriority"] == "musaStreamCreateWithPriority"

    def test_event_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaEventCreate"] == "musaEventCreate"
        assert _MAPPING_RULE["cudaEventDestroy"] == "musaEventDestroy"
        assert _MAPPING_RULE["cudaEventRecord"] == "musaEventRecord"
        assert _MAPPING_RULE["cudaEventSynchronize"] == "musaEventSynchronize"
        assert _MAPPING_RULE["cudaEventElapsedTime"] == "musaEventElapsedTime"
        assert _MAPPING_RULE["cudaEventQuery"] == "musaEventQuery"

    def test_event_flags(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaEventDefault"] == "musaEventDefault"
        assert _MAPPING_RULE["cudaEventBlockingSync"] == "musaEventBlockingSync"
        assert _MAPPING_RULE["cudaEventDisableTiming"] == "musaEventDisableTiming"
        assert _MAPPING_RULE["cudaEventCreateWithFlags"] == "musaEventCreateWithFlags"


class TestErrorHandlingMappings:
    """Test CUDA error handling mappings."""

    def test_error_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaError_t"] == "musaError_t"
        assert _MAPPING_RULE["cudaSuccess"] == "musaSuccess"

    def test_error_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cudaGetLastError"] == "musaGetLastError"
        assert _MAPPING_RULE["cudaGetErrorString"] == "musaGetErrorString"
        assert _MAPPING_RULE["cudaPeekAtLastError"] == "musaPeekAtLastError"

    def test_error_codes(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["CUDA_ERROR_NOT_PERMITTED"] == "MUSA_ERROR_NOT_PERMITTED"
        assert _MAPPING_RULE["CUDA_ERROR_NOT_SUPPORTED"] == "MUSA_ERROR_NOT_SUPPORTED"


class TestNCCLMappings:
    """Test NCCL to MCCL mappings."""

    def test_nccl_basic(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["nccl"] == "mccl"
        assert _MAPPING_RULE["NCCL"] == "MCCL"

    def test_nccl_types(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["ncclComm_t"] == "mcclComm_t"
        assert _MAPPING_RULE["ncclDataType_t"] == "mcclDataType_t"
        assert _MAPPING_RULE["ncclRedOp_t"] == "mcclRedOp_t"
        assert _MAPPING_RULE["ncclResult_t"] == "mcclResult_t"
        assert _MAPPING_RULE["ncclSuccess"] == "mcclSuccess"
        assert _MAPPING_RULE["ncclUniqueId"] == "mcclUniqueId"

    def test_nccl_comm_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["ncclCommInitRank"] == "mcclCommInitRank"
        assert _MAPPING_RULE["ncclCommInitAll"] == "mcclCommInitAll"
        assert _MAPPING_RULE["ncclCommDestroy"] == "mcclCommDestroy"
        assert _MAPPING_RULE["ncclCommCount"] == "mcclCommCount"
        assert _MAPPING_RULE["ncclCommCuDevice"] == "mcclCommCuDevice"
        assert _MAPPING_RULE["ncclCommUserRank"] == "mcclCommUserRank"
        assert _MAPPING_RULE["ncclGetUniqueId"] == "mcclGetUniqueId"

    def test_nccl_collective_functions(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["ncclAllReduce"] == "mcclAllReduce"
        assert _MAPPING_RULE["ncclBroadcast"] == "mcclBroadcast"
        assert _MAPPING_RULE["ncclReduce"] == "mcclReduce"
        assert _MAPPING_RULE["ncclAllGather"] == "mcclAllGather"
        assert _MAPPING_RULE["ncclReduceScatter"] == "mcclReduceScatter"
        assert _MAPPING_RULE["ncclSend"] == "mcclSend"
        assert _MAPPING_RULE["ncclRecv"] == "mcclRecv"
        assert _MAPPING_RULE["ncclGroupStart"] == "mcclGroupStart"
        assert _MAPPING_RULE["ncclGroupEnd"] == "mcclGroupEnd"


class TestLibraryMappings:
    """Test other library mappings."""

    def test_cusparse(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cusparse"] == "musparse"
        assert _MAPPING_RULE["CUSPARSE"] == "MUSPARSE"
        assert _MAPPING_RULE["cusparseHandle_t"] == "musparseHandle_t"
        assert _MAPPING_RULE["cusparseCreate"] == "musparseCreate"
        assert _MAPPING_RULE["cusparseDestroy"] == "musparseDestroy"

    def test_cusolver(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cusolver"] == "musolver"
        assert _MAPPING_RULE["CUSOLVER"] == "MUSOLVER"
        assert _MAPPING_RULE["cusolverDnHandle_t"] == "musolverDnHandle_t"
        assert _MAPPING_RULE["cusolverDnCreate"] == "musolverDnCreate"
        assert _MAPPING_RULE["cusolverDnDestroy"] == "musolverDnDestroy"

    def test_cufft(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cufft"] == "mufft"
        assert _MAPPING_RULE["CUFFT"] == "MUFFT"
        assert _MAPPING_RULE["cufftHandle"] == "mufftHandle"
        assert _MAPPING_RULE["cufftPlan1d"] == "mufftPlan1d"
        assert _MAPPING_RULE["cufftPlan2d"] == "mufftPlan2d"
        assert _MAPPING_RULE["cufftPlan3d"] == "mufftPlan3d"
        assert _MAPPING_RULE["cufftExecC2C"] == "mufftExecC2C"
        assert _MAPPING_RULE["cufftExecR2C"] == "mufftExecR2C"
        assert _MAPPING_RULE["cufftExecC2R"] == "mufftExecC2R"

    def test_cutlass(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cutlass"] == "mutlass"
        assert _MAPPING_RULE["CUTLASS"] == "MUTLASS"
        assert _MAPPING_RULE["cutlass/"] == "mutlass/"
        assert _MAPPING_RULE["cutlass::"] == "mutlass::"

    def test_cub(self):
        """CUB is provided directly by MUSA, no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        # CUB doesn't need mapping - MUSA provides it directly
        assert "cub::" not in _MAPPING_RULE
        assert "cub/" not in _MAPPING_RULE

    def test_thrust(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["thrust::cuda"] == "thrust::musa"


class TestIntrinsicMappings:
    """Test CUDA intrinsic and math function mappings.

    Note: Many CUDA intrinsics (shuffle, vote, sync, atomics, half precision)
    are the same in MUSA and don't require mapping. These tests verify that
    we correctly do NOT include identity mappings for these.
    """

    def test_shuffle_intrinsics_not_mapped(self):
        """Shuffle intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        # These should NOT be in the mapping (same syntax in MUSA)
        assert "__shfl_sync" not in _MAPPING_RULE
        assert "__shfl_xor_sync" not in _MAPPING_RULE

    def test_vote_intrinsics_not_mapped(self):
        """Vote intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "__ballot_sync" not in _MAPPING_RULE
        assert "__any_sync" not in _MAPPING_RULE

    def test_sync_intrinsics_not_mapped(self):
        """Sync intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "__syncthreads" not in _MAPPING_RULE
        assert "__threadfence" not in _MAPPING_RULE

    def test_atomic_operations_not_mapped(self):
        """Atomic operations are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "atomicAdd" not in _MAPPING_RULE
        assert "atomicCAS" not in _MAPPING_RULE

    def test_half_precision_not_mapped(self):
        """Half precision intrinsics are the same in MUSA - no mapping needed."""
        from torchada._mapping import _MAPPING_RULE

        assert "__float2half" not in _MAPPING_RULE
        assert "__half2float" not in _MAPPING_RULE
        assert "__hadd" not in _MAPPING_RULE

class TestCommonMacros:
    """Test Common macros."""

    def test_common_macros(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["CUDA_VERSION"] == "MUSA_VERSION"


class TestIncludeMappings:
    """Test header include mappings."""

    def test_cuda_headers(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["cuda_runtime.h"] == "musa_runtime.h"
        assert _MAPPING_RULE["cuda_runtime_api.h"] == "musa_runtime_api.h"
        assert _MAPPING_RULE["cuda.h"] == "musa.h"
        assert _MAPPING_RULE["cuda_fp16.h"] == "musa_fp16.h"
        assert _MAPPING_RULE["cuda_bf16.h"] == "musa_bf16.h"


class TestPyTorchCppMappings:
    """Test PyTorch C++ API mappings."""

    def test_pytorch_stream_utils(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["getCurrentCUDAStream"] == "getCurrentMUSAStream"
        assert _MAPPING_RULE["getDefaultCUDAStream"] == "getDefaultMUSAStream"
        assert _MAPPING_RULE["CUDAStream"] == "MUSAStream"

    def test_pytorch_guards(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["CUDAGuard"] == "MUSAGuard"
        assert _MAPPING_RULE["OptionalCUDAGuard"] == "OptionalMUSAGuard"
        assert _MAPPING_RULE["CUDAStreamGuard"] == "MUSAStreamGuard"
        assert _MAPPING_RULE["CUDAEvent"] == "MUSAEvent"

    def test_pytorch_handle(self):
        from torchada._mapping import _MAPPING_RULE

        assert _MAPPING_RULE["getCurrentCUDABlasHandle"] == "getCurrentMUSABlasHandle"

    def test_pytorch_torch_namespace(self):
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["torch::cuda::getCurrentCUDAStream"]
            == "torch::musa::getCurrentMUSAStream"
        )
        assert (
            _MAPPING_RULE["torch::cuda::getDefaultCUDAStream"]
            == "torch::musa::getDefaultMUSAStream"
        )
        assert _MAPPING_RULE["torch::cuda::getStreamFromPool"] == "torch::musa::getStreamFromPool"


class TestCUDADriverMappings:
    """Test CUDA driver API mappings."""

    def test_memory_constants(self):
        from torchada._mapping import _MAPPING_RULE

        assert (
            _MAPPING_RULE["CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED"]
            == "MU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_MUSA_VMM_SUPPORTED"
        )

        assert (
            _MAPPING_RULE["CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED"]
            == "MU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED"
        )

        assert (
            _MAPPING_RULE["CU_MEM_HANDLE_TYPE_FABRIC"] == "MU_MEM_HANDLE_TYPE_FABRIC"
        )

        assert (
            _MAPPING_RULE["CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR"] 
            == "MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR"
        )


class TestMappingCount:
    """Test total mapping count."""

    def test_mapping_count(self):
        from torchada._mapping import _MAPPING_RULE

        # We should have a substantial number of mappings
        assert len(_MAPPING_RULE) >= 250

    def test_ext_replaced_mapping(self):
        from torchada._mapping import EXT_REPLACED_MAPPING

        # Extensions are converted: .cu -> .mu, .cuh -> .muh for mcc compiler
        assert EXT_REPLACED_MAPPING["cu"] == "mu"
        assert EXT_REPLACED_MAPPING["cuh"] == "muh"


class TestMappingRobustness:
    """Tests to ensure mapping rules are robust and don't have issues."""

    def test_no_identity_mappings(self):
        """Ensure no mapping maps a key to itself (identity mapping)."""
        from torchada._mapping import _MAPPING_RULE

        identity_mappings = [(k, v) for k, v in _MAPPING_RULE.items() if k == v]
        assert len(identity_mappings) == 0, (
            f"Found {len(identity_mappings)} identity mappings that should be removed: "
            f"{identity_mappings[:5]}"
        )

    def test_no_empty_keys_or_values(self):
        """Ensure no mapping has empty key or value."""
        from torchada._mapping import _MAPPING_RULE

        for key, value in _MAPPING_RULE.items():
            assert key, "Found empty key in mapping"
            assert value is not None, f"Found None value for key: {key}"
            # Empty value is allowed for deletion, but we don't use that

    def test_all_keys_and_values_are_strings(self):
        """Ensure all keys and values are strings."""
        from torchada._mapping import _MAPPING_RULE

        for key, value in _MAPPING_RULE.items():
            assert isinstance(key, str), f"Key is not a string: {key}"
            assert isinstance(value, str), f"Value is not a string for key {key}: {value}"

    def test_cuda_to_musa_consistency(self):
        """Test that CUDA terms consistently map to MUSA equivalents."""
        from torchada._mapping import _MAPPING_RULE

        # Check that 'cuda' in key generally maps to 'musa' in value
        # (with some exceptions for special cases like torch::kCUDA -> PrivateUse1)
        exceptions = {
            "torch::kCUDA",  # Maps to PrivateUse1
            ".is_cuda()",  # Maps to .is_privateuseone()
        }

        for key, value in _MAPPING_RULE.items():
            if key in exceptions:
                continue
            # If key contains 'cuda' (case insensitive), value should contain 'musa'
            if "cuda" in key.lower() and "musa" not in value.lower():
                # Allow mappings where cuda -> privateuseone or similar
                if "privateuseone" not in value.lower() and "private" not in value.lower():
                    # Check for special patterns
                    if not any(
                        x in value.lower()
                        for x in [
                            "musa",
                            "privateuseone",
                            "ignore",
                            "torch_musa",
                        ]
                    ):
                        pytest.fail(
                            f"Inconsistent mapping: '{key}' -> '{value}' "
                            "(expected 'musa' or special case in value)"
                        )


class TestMappingApplication:
    """Test that mappings are correctly applied to source code."""

    def test_port_cuda_source_basic(self):
        """Test basic source code porting."""
        from torchada.utils.cpp_extension import _port_cuda_source

        # Test basic namespace replacement
        source = "at::cuda::getCurrentCUDAStream()"
        result = _port_cuda_source(source)
        assert "at::musa" in result
        assert "at::cuda" not in result

    def test_port_cuda_source_includes(self):
        """Test include statement porting."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = '#include <cuda_runtime.h>\n#include "my_file.cuh"'
        result = _port_cuda_source(source)
        assert "musa_runtime.h" in result
        assert 'my_file.muh"' in result

    def test_port_cuda_source_types(self):
        """Test type replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = "cudaStream_t stream; cudaError_t err;"
        result = _port_cuda_source(source)
        assert "musaStream_t" in result
        assert "musaError_t" in result
        assert "cudaStream_t" not in result

    def test_port_cuda_source_functions(self):
        """Test function name replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = "cudaMalloc(&ptr, size); cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);"
        result = _port_cuda_source(source)
        assert "musaMalloc" in result
        assert "musaMemcpy" in result
        assert "cudaMalloc" not in result

    def test_port_cuda_source_preserves_non_cuda(self):
        """Test that non-CUDA code is preserved."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = """
int main() {
    int x = 42;
    float y = 3.14f;
    return 0;
}
"""
        result = _port_cuda_source(source)
        assert "int main()" in result
        assert "int x = 42" in result
        assert "float y = 3.14f" in result

    def test_port_cuda_source_longer_patterns_first(self):
        """Test that longer patterns are applied before shorter ones."""
        from torchada.utils.cpp_extension import _port_cuda_source

        # This tests that 'cudaMemcpyHostToDevice' is replaced before 'cuda'
        source = "cudaMemcpyHostToDevice"
        result = _port_cuda_source(source)
        # Should be musaMemcpyHostToDevice, not something like musaMemcpyHostToDevice
        assert result == "musaMemcpyHostToDevice"

    def test_port_cuda_source_c10_macros(self):
        """Test C10 macro replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = "C10_CUDA_KERNEL_LAUNCH_CHECK();"
        result = _port_cuda_source(source)
        assert "C10_MUSA_KERNEL_LAUNCH_CHECK" in result
        assert "C10_CUDA" not in result

    def test_port_cuda_source_nccl(self):
        """Test NCCL to MCCL replacement."""
        from torchada.utils.cpp_extension import _port_cuda_source

        source = (
            "ncclComm_t comm; ncclAllReduce(buffer, buffer, count, datatype, op, comm, stream);"
        )
        result = _port_cuda_source(source)
        assert "mcclComm_t" in result
        assert "mcclAllReduce" in result
        assert "ncclComm_t" not in result


class TestMappingSubstringOrdering:
    """Test that mappings with substring relationships are handled correctly."""

    def test_specific_before_generic(self):
        """Test that specific mappings work correctly with generic ones."""
        from torchada._mapping import _MAPPING_RULE

        # The _port_cuda_source function sorts by length (longest first)
        # So specific mappings like 'cudaMemcpyHostToDevice' should be in the mapping
        # and the replacement algorithm should handle them correctly
        # Verify that both specific and generic patterns exist
        assert "cudaMemcpy" in _MAPPING_RULE
        assert "cudaMemcpyHostToDevice" in _MAPPING_RULE

        # Verify the specific one maps correctly
        assert _MAPPING_RULE["cudaMemcpyHostToDevice"] == "musaMemcpyHostToDevice"

    def test_include_specific_paths(self):
        """Test that specific include paths are correctly mapped."""
        from torchada._mapping import _MAPPING_RULE

        # Check that specific ATen includes exist
        assert "#include <ATen/cuda/CUDAContext.h>" in _MAPPING_RULE

        # And that generic at::cuda also exists
        assert "at::cuda" in _MAPPING_RULE


class TestRuntimeNameConversion:
    """Test runtime name conversion utilities."""

    def test_cuda_to_musa_name(self):
        """Test CUDA to MUSA function name conversion."""
        from torchada import cuda_to_musa_name

        # Basic conversions
        assert cuda_to_musa_name("cudaMalloc") == "musaMalloc"
        assert cuda_to_musa_name("cudaFree") == "musaFree"
        assert cuda_to_musa_name("cudaIpcOpenMemHandle") == "musaIpcOpenMemHandle"
        assert cuda_to_musa_name("cudaIpcGetMemHandle") == "musaIpcGetMemHandle"
        assert cuda_to_musa_name("cudaMemset") == "musaMemset"
        assert cuda_to_musa_name("cudaError_t") == "musaError_t"

        # Non-cuda names should be unchanged
        assert cuda_to_musa_name("someOtherFunc") == "someOtherFunc"
        assert cuda_to_musa_name("malloc") == "malloc"

    def test_nccl_to_mccl_name(self):
        """Test NCCL to MCCL function name conversion."""
        from torchada import nccl_to_mccl_name

        # Basic conversions
        assert nccl_to_mccl_name("ncclAllReduce") == "mcclAllReduce"
        assert nccl_to_mccl_name("ncclCommInitRank") == "mcclCommInitRank"
        assert nccl_to_mccl_name("ncclBroadcast") == "mcclBroadcast"
        assert nccl_to_mccl_name("ncclUniqueId") == "mcclUniqueId"
        assert nccl_to_mccl_name("ncclGetErrorString") == "mcclGetErrorString"

        # Non-nccl names should be unchanged
        assert nccl_to_mccl_name("someOtherFunc") == "someOtherFunc"

    def test_cublas_to_mublas_name(self):
        """Test cuBLAS to muBLAS function name conversion."""
        from torchada import cublas_to_mublas_name

        assert cublas_to_mublas_name("cublasCreate") == "mublasCreate"
        assert cublas_to_mublas_name("cublasSgemm") == "mublasSgemm"
        assert cublas_to_mublas_name("cublasDestroy") == "mublasDestroy"

        # Non-cublas names should be unchanged
        assert cublas_to_mublas_name("someOtherFunc") == "someOtherFunc"

    def test_curand_to_murand_name(self):
        """Test cuRAND to muRAND function name conversion."""
        from torchada import curand_to_murand_name

        assert curand_to_murand_name("curandCreate") == "murandCreate"
        assert curand_to_murand_name("curand_init") == "murand_init"

        # Non-curand names should be unchanged
        assert curand_to_murand_name("someOtherFunc") == "someOtherFunc"


class TestCDLLWrapper:
    """Test ctypes.CDLL wrapper for automatic function name translation.

    These tests load actual MUSA libraries from /usr/local/musa/lib/ and verify
    that CUDA function names are automatically translated to MUSA equivalents.
    """

    MUSA_LIB_PATH = "/usr/local/musa/lib"

    def test_cdll_wrapper_class_exists(self):
        """Test that _CDLLWrapper class is available."""
        from torchada._patch import _CDLLWrapper

        assert _CDLLWrapper is not None

    @pytest.mark.musa
    def test_libmusart_cuda_to_musa_translation(self):
        """Test that CUDA function names are translated when loading libmusart.so."""
        import ctypes
        import os

        import torchada  # noqa: F401 - Apply patches

        lib_path = os.path.join(self.MUSA_LIB_PATH, "libmusart.so")
        if not os.path.exists(lib_path):
            pytest.skip(f"libmusart.so not found at {lib_path}")

        # Load the library using patched ctypes.CDLL
        lib = ctypes.CDLL(lib_path)

        # Access using CUDA function names - should be translated to MUSA
        # These should NOT raise AttributeError because they get translated
        func = lib.cudaMalloc
        assert func is not None

        func = lib.cudaFree
        assert func is not None

        func = lib.cudaGetDevice
        assert func is not None

        func = lib.cudaIpcOpenMemHandle
        assert func is not None

        func = lib.cudaIpcGetMemHandle
        assert func is not None

    @pytest.mark.musa
    def test_libmccl_nccl_to_mccl_translation(self):
        """Test that NCCL function names are translated when loading libmccl.so."""
        import ctypes
        import os

        import torchada  # noqa: F401 - Apply patches

        lib_path = os.path.join(self.MUSA_LIB_PATH, "libmccl.so")
        if not os.path.exists(lib_path):
            pytest.skip(f"libmccl.so not found at {lib_path}")

        # Load the library using patched ctypes.CDLL
        lib = ctypes.CDLL(lib_path)

        # Access using NCCL function names - should be translated to MCCL
        func = lib.ncclAllReduce
        assert func is not None

        func = lib.ncclBroadcast
        assert func is not None

        func = lib.ncclCommInitRank
        assert func is not None

    @pytest.mark.musa
    def test_libmublas_cublas_to_mublas_translation(self):
        """Test that cuBLAS function names are translated when loading libmublas.so."""
        import ctypes
        import os

        import torchada  # noqa: F401 - Apply patches

        lib_path = os.path.join(self.MUSA_LIB_PATH, "libmublas.so")
        if not os.path.exists(lib_path):
            pytest.skip(f"libmublas.so not found at {lib_path}")

        # Load the library using patched ctypes.CDLL
        lib = ctypes.CDLL(lib_path)

        # Access using cuBLAS function names - should be translated to muBLAS
        # mublasCreate is the actual function name in libmublas.so
        func = lib.cublasCreate
        assert func is not None

        func = lib.cublasDestroy
        assert func is not None

    @pytest.mark.musa
    def test_libmurand_curand_to_murand_translation(self):
        """Test that cuRAND function names are translated when loading libmurand.so."""
        import ctypes
        import os

        import torchada  # noqa: F401 - Apply patches

        lib_path = os.path.join(self.MUSA_LIB_PATH, "libmurand.so")
        if not os.path.exists(lib_path):
            pytest.skip(f"libmurand.so not found at {lib_path}")

        # Load the library using patched ctypes.CDLL
        lib = ctypes.CDLL(lib_path)

        # Access using cuRAND function names - should be translated to muRAND
        func = lib.curandCreateGenerator
        assert func is not None

    @pytest.mark.musa
    def test_non_musa_lib_no_translation(self):
        """Test that non-MUSA libraries don't get function name translation."""
        import ctypes

        import torchada  # noqa: F401 - Apply patches

        # Load a standard library that exists on all systems
        try:
            lib = ctypes.CDLL("libc.so.6")
        except OSError:
            pytest.skip("libc.so.6 not found")

        # This should NOT be wrapped, so accessing cudaMalloc should fail
        # (libc doesn't have cudaMalloc or musaMalloc)
        with pytest.raises(AttributeError):
            _ = lib.cudaMalloc

    @pytest.mark.musa
    @pytest.mark.gpu
    def test_sglang_cuda_wrapper_pattern(self):
        """Test that sglang's cuda_wrapper.py pattern works seamlessly with torchada.

        This test simulates the approach used in sglang's cuda_wrapper.py:
        https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/cuda_wrapper.py

        The key insight is that sglang uses CUDA function names (cudaMalloc, cudaFree, etc.)
        when accessing functions from the library. With torchada's ctypes.CDLL patch,
        these names are automatically translated to MUSA equivalents.
        """
        import ctypes
        import os
        from dataclasses import dataclass
        from typing import Any, List

        import torchada  # noqa: F401 - Apply patches

        lib_path = os.path.join(self.MUSA_LIB_PATH, "libmusart.so")
        if not os.path.exists(lib_path):
            pytest.skip(f"libmusart.so not found at {lib_path}")

        # === Types from sglang's cuda_wrapper.py ===
        cudaError_t = ctypes.c_int

        @dataclass
        class Function:
            name: str
            restype: Any
            argtypes: List[Any]

        # === Subset of exported functions (same as sglang) ===
        exported_functions = [
            # cudaError_t cudaSetDevice ( int device )
            Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
            # cudaError_t cudaDeviceSynchronize ( void )
            Function("cudaDeviceSynchronize", cudaError_t, []),
            # cudaError_t cudaMalloc ( void** devPtr, size_t size )
            Function(
                "cudaMalloc",
                cudaError_t,
                [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
            ),
            # cudaError_t cudaFree ( void* devPtr )
            Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        ]

        # === Load library using sglang's pattern ===
        # sglang does: lib = ctypes.CDLL(so_file)
        lib = ctypes.CDLL(lib_path)

        # === Access functions using CUDA names (sglang's pattern) ===
        # sglang does: f = getattr(self.lib, func.name)
        # With torchada, this automatically translates cudaXxx -> musaXxx
        funcs = {}
        for func in exported_functions:
            # This is the key line - sglang uses CUDA names here
            f = getattr(lib, func.name)
            f.restype = func.restype
            f.argtypes = func.argtypes
            funcs[func.name] = f

        # Verify all functions were loaded successfully
        assert "cudaSetDevice" in funcs
        assert "cudaDeviceSynchronize" in funcs
        assert "cudaMalloc" in funcs
        assert "cudaFree" in funcs

        # === Actually call the functions to verify they work ===
        # Set device 0
        result = funcs["cudaSetDevice"](0)
        assert result == 0, f"cudaSetDevice failed with error {result}"

        # Allocate memory
        devPtr = ctypes.c_void_p()
        result = funcs["cudaMalloc"](ctypes.byref(devPtr), 1024)
        assert result == 0, f"cudaMalloc failed with error {result}"
        assert devPtr.value is not None

        # Free memory
        result = funcs["cudaFree"](devPtr)
        assert result == 0, f"cudaFree failed with error {result}"

        # Synchronize
        result = funcs["cudaDeviceSynchronize"]()
        assert result == 0, f"cudaDeviceSynchronize failed with error {result}"


# Import for compilation tests
import os
import shutil
import subprocess
import sys
import tempfile

import torchada

# Path to device type test source file
CSRC_DIR = os.path.join(os.path.dirname(__file__), "csrc")
DEVICE_TYPE_TEST_CU = os.path.join(CSRC_DIR, "device_type_test.cu")


def _is_gpu_available():
    """Check if CUDA or MUSA GPU is available."""
    import torch

    if torch.cuda.is_available():
        return True
    if hasattr(torch, "musa") and torch.musa.is_available():
        return True
    return False


@pytest.mark.skipif(
    not os.environ.get("TORCHADA_TEST_BUILD", "0") == "1",
    reason="Extension build tests are slow; set TORCHADA_TEST_BUILD=1 to run",
)
class TestDeviceTypeMappingCompilation:
    """
    Test that device type mappings compile and run correctly.

    These tests verify that:
    - at::DeviceType::CUDA -> at::DeviceType::PrivateUse1 compiles
    - c10::DeviceType::CUDA -> c10::DeviceType::PrivateUse1 compiles
    - at::kCUDA -> at::musa::kMUSA compiles
    - torch::kCUDA -> c10::DeviceType::PrivateUse1 compiles
    """

    def test_device_type_test_source_exists(self):
        """Verify the test source file exists."""
        assert os.path.exists(
            DEVICE_TYPE_TEST_CU
        ), f"Device type test source not found: {DEVICE_TYPE_TEST_CU}"

    def test_build_device_type_extension(self):
        """Test building the device_type_test extension with device type mappings."""
        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        if not torchada.is_musa_platform():
            pytest.skip("Device type mapping compilation test only applicable on MUSA")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(DEVICE_TYPE_TEST_CU, tmpdir)

            # Create setup.py using standard torch imports
            setup_content = """
import torchada  # noqa: F401 - Apply MUSA patches
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_device_type",
    ext_modules=[
        CUDAExtension(
            name="test_device_type",
            sources=["device_type_test.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            # Check build succeeded
            assert (
                result.returncode == 0
            ), f"Build failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"

    def test_run_device_type_extension(self):
        """Test running the device_type_test extension after building."""
        import torch

        if not _is_gpu_available():
            pytest.skip("CUDA/MUSA not available")

        if not torchada.is_musa_platform():
            pytest.skip("Device type mapping run test only applicable on MUSA platform")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(DEVICE_TYPE_TEST_CU, tmpdir)

            # Create setup.py
            setup_content = """
import torchada  # noqa: F401
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_device_type",
    ext_modules=[
        CUDAExtension(
            name="test_device_type",
            sources=["device_type_test.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, f"Build failed: {result.stderr}"

            # Add tmpdir to Python path and import the extension
            sys.path.insert(0, tmpdir)
            try:
                import test_device_type

                try:
                    # Test 1: create_cuda_tensor - tests torch::kCUDA mapping
                    tensor = test_device_type.create_cuda_tensor(10)
                    assert tensor.shape == (10,), "create_cuda_tensor shape mismatch"
                    assert tensor.device.type in (
                        "cuda",
                        "musa",
                    ), f"Wrong device type: {tensor.device.type}"

                    # Test 2: check_device_type - tests at::DeviceType::CUDA mapping
                    input_tensor = torch.randn(5, device="cuda")
                    output = test_device_type.check_device_type(input_tensor)
                    assert output.shape == (1,), "check_device_type shape mismatch"

                    # Test 3: get_device_info - tests multiple device type check methods
                    info = test_device_type.get_device_info(input_tensor)
                    # All three checks should return True
                    assert info[0], "at::DeviceType::CUDA check failed"
                    assert info[1], "c10::DeviceType::CUDA check failed"
                    assert info[2], "is_cuda() check failed"

                except RuntimeError as e:
                    # Skip if GPU kernel execution fails (MUDNN issues in test containers)
                    if "invalid device function" in str(e) or "MUDNN" in str(e):
                        pytest.skip("GPU kernel execution failed (expected in test containers)")
                    raise
            finally:
                sys.path.remove(tmpdir)
                # Clean up imported module
                if "test_device_type" in sys.modules:
                    del sys.modules["test_device_type"]

    def test_ported_source_contains_privateuse1(self):
        """Verify the ported source code contains PrivateUse1, not MUSA."""
        if not torchada.is_musa_platform():
            pytest.skip("Source porting test only applicable on MUSA platform")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Copy the source file
            shutil.copy(DEVICE_TYPE_TEST_CU, tmpdir)

            # Create setup.py
            setup_content = """
import torchada  # noqa: F401
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="test_device_type",
    ext_modules=[
        CUDAExtension(
            name="test_device_type",
            sources=["device_type_test.cu"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""
            setup_path = os.path.join(tmpdir, "setup.py")
            with open(setup_path, "w") as f:
                f.write(setup_content)

            # Build the extension (this triggers source porting)
            result = subprocess.run(
                [sys.executable, "setup.py", "build_ext", "--inplace"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            # Find all ported .mu files
            # The ported files are in {tmpdir}_musa directory (created by torchada)
            ported_files = []
            musa_dir = f"{tmpdir}_musa"
            if os.path.exists(musa_dir):
                for root, dirs, files in os.walk(musa_dir):
                    for f in files:
                        if f.endswith(".mu"):
                            ported_files.append(os.path.join(root, f))

            # Also check inside tmpdir in case porting puts files there
            for root, dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".mu"):
                        ported_files.append(os.path.join(root, f))

            assert (
                len(ported_files) > 0
            ), f"No .mu file found after porting. Build output:\n{result.stdout}"

            # Read ported content
            ported_content = ""
            for pf in ported_files:
                with open(pf, "r") as f:
                    ported_content += f.read()

            # Verify the mappings were applied correctly
            assert (
                "at::DeviceType::PrivateUse1" in ported_content
            ), "at::DeviceType::CUDA was not ported to PrivateUse1"
            assert (
                "c10::DeviceType::PrivateUse1" in ported_content
            ), "c10::DeviceType::CUDA was not ported to PrivateUse1"

            # Verify CUDA device type references are gone
            assert (
                "at::DeviceType::CUDA" not in ported_content
            ), "at::DeviceType::CUDA should be ported to PrivateUse1"
            assert (
                "c10::DeviceType::CUDA" not in ported_content
            ), "c10::DeviceType::CUDA should be ported to PrivateUse1"

            # Verify at::kCUDA is ported to at::kPrivateUse1
            assert (
                "at::kCUDA" not in ported_content
            ), "at::kCUDA should be ported to at::kPrivateUse1"
