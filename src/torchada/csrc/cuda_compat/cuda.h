#pragma once
// MUSA: the CUDA driver API surface for sources compiled by mcc without a
// Torchada source-porting pass. This directory must precede CUDA toolkit
// include directories so a stray CUDA header cannot supply the definitions.
#include <musa.h>

// Types.
#ifndef CUcontext
#define CUcontext MUcontext
#endif
#ifndef CUdevice
#define CUdevice MUdevice
#endif
#ifndef CUdevice_attribute
#define CUdevice_attribute MUdevice_attribute
#endif
#ifndef CUdeviceptr
#define CUdeviceptr MUdeviceptr
#endif
#ifndef CUdriverProcAddressQueryResult
#define CUdriverProcAddressQueryResult MUdriverProcAddressQueryResult
#endif
#ifndef cuuint32_t
#define cuuint32_t muuint32_t
#endif
#ifndef cuuint64_t
#define cuuint64_t muuint64_t
#endif
#ifndef CUevent
#define CUevent MUevent
#endif
#ifndef CUfunction
#define CUfunction MUfunction
#endif
#ifndef CUfunction_attribute
#define CUfunction_attribute MUfunction_attribute
#endif
#ifndef CUlaunchAttribute
#define CUlaunchAttribute MUlaunchAttribute
#endif
#ifndef CUlaunchAttributeID
#define CUlaunchAttributeID MUlaunchAttributeID
#endif
#ifndef CUlaunchAttributeValue
#define CUlaunchAttributeValue MUlaunchAttributeValue
#endif
#ifndef CUlaunchConfig
#define CUlaunchConfig MUlaunchConfig
#endif
#ifndef CUmemAccessDesc
#define CUmemAccessDesc MUmemAccessDesc
#endif
#ifndef CUmemAllocationProp
#define CUmemAllocationProp MUmemAllocationProp
#endif
#ifndef CUmemAllocationGranularity_flags
#define CUmemAllocationGranularity_flags MUmemAllocationGranularity_flags
#endif
#ifndef CUmemcpyAttributes
#define CUmemcpyAttributes MUmemcpyAttributes
#endif
#ifndef CUpointer_attribute
#define CUpointer_attribute MUpointer_attribute
#endif
#ifndef CUmemGenericAllocationHandle
#define CUmemGenericAllocationHandle MUmemGenericAllocationHandle
#endif
#ifndef CUmodule
#define CUmodule MUmodule
#endif
#ifndef CUresult
#define CUresult MUresult
#endif
#ifndef CUstream
#define CUstream MUstream
#endif

// Entry points.
#ifndef cuCtxGetCurrent
#define cuCtxGetCurrent muCtxGetCurrent
#endif
#ifndef cuCtxSetCurrent
#define cuCtxSetCurrent muCtxSetCurrent
#endif
#ifndef cuDeviceGet
#define cuDeviceGet muDeviceGet
#endif
#ifndef cuDeviceGetAttribute
#define cuDeviceGetAttribute muDeviceGetAttribute
#endif
#ifndef cuDeviceGetCount
#define cuDeviceGetCount muDeviceGetCount
#endif
#ifndef cuDevicePrimaryCtxRetain
#define cuDevicePrimaryCtxRetain muDevicePrimaryCtxRetain
#endif
#ifndef cuFuncSetAttribute
#define cuFuncSetAttribute muFuncSetAttribute
#endif
#ifndef cuGetErrorName
#define cuGetErrorName muGetErrorName
#endif
#ifndef cuGetErrorString
#define cuGetErrorString muGetErrorString
#endif
#ifndef cuGetProcAddress
#define cuGetProcAddress muGetProcAddress
#endif
#ifndef cuInit
#define cuInit muInit
#endif
#ifndef cuMemAddressFree
#define cuMemAddressFree muMemAddressFree
#endif
#ifndef cuMemAddressReserve
#define cuMemAddressReserve muMemAddressReserve
#endif
#ifndef cuMemCreate
#define cuMemCreate muMemCreate
#endif
#ifndef cuMemGetAddressRange
#define cuMemGetAddressRange muMemGetAddressRange
#endif
#ifndef cuMemGetAllocationGranularity
#define cuMemGetAllocationGranularity muMemGetAllocationGranularity
#endif
#ifndef cuMemMap
#define cuMemMap muMemMap
#endif
#ifndef cuMemRelease
#define cuMemRelease muMemRelease
#endif
#ifndef cuMemSetAccess
#define cuMemSetAccess muMemSetAccess
#endif
#ifndef cuMemUnmap
#define cuMemUnmap muMemUnmap
#endif
#ifndef cuMemcpyBatchAsync
#define cuMemcpyBatchAsync muMemcpyBatchAsync
#endif
#ifndef cuLaunchKernel
#define cuLaunchKernel muLaunchKernel
#endif
#ifndef cuLaunchKernelEx
#define cuLaunchKernelEx muLaunchKernelEx
#endif
#ifndef cuModuleGetFunction
#define cuModuleGetFunction muModuleGetFunction
#endif
#ifndef cuModuleLoadDataEx
#define cuModuleLoadDataEx muModuleLoadDataEx
#endif
#ifndef cuModuleUnload
#define cuModuleUnload muModuleUnload
#endif
#ifndef cuPointerGetAttribute
#define cuPointerGetAttribute muPointerGetAttribute
#endif

// Constants.
#ifndef CU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE MU_MEM_ACCESS_FLAGS_PROT_READWRITE
#endif
#ifndef CU_MEM_ALLOC_GRANULARITY_MINIMUM
#define CU_MEM_ALLOC_GRANULARITY_MINIMUM MU_MEM_ALLOC_GRANULARITY_MINIMUM
#endif
#ifndef CU_MEM_ALLOCATION_COMP_NONE
#define CU_MEM_ALLOCATION_COMP_NONE MU_MEM_ALLOCATION_COMP_NONE
#endif
#ifndef CU_MEM_ALLOCATION_TYPE_PINNED
#define CU_MEM_ALLOCATION_TYPE_PINNED MU_MEM_ALLOCATION_TYPE_PINNED
#endif
#ifndef CU_MEM_HANDLE_TYPE_FABRIC
#define CU_MEM_HANDLE_TYPE_FABRIC MU_MEM_HANDLE_TYPE_FABRIC
#endif
#ifndef CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#define CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
#endif
#ifndef CU_MEM_LOCATION_TYPE_DEVICE
#define CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
#endif
#ifndef CU_GET_PROC_ADDRESS_DEFAULT
#define CU_GET_PROC_ADDRESS_DEFAULT MU_GET_PROC_ADDRESS_DEFAULT
#endif
#ifndef CU_GET_PROC_ADDRESS_LEGACY_STREAM
#define CU_GET_PROC_ADDRESS_LEGACY_STREAM MU_GET_PROC_ADDRESS_LEGACY_STREAM
#endif
#ifndef CU_GET_PROC_ADDRESS_SUCCESS
#define CU_GET_PROC_ADDRESS_SUCCESS MU_GET_PROC_ADDRESS_SUCCESS
#endif
#ifndef CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#define CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES MU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES
#endif
#ifndef CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
#define CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION MU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
#endif
#ifndef CU_LAUNCH_ATTRIBUTE_COOPERATIVE
#define CU_LAUNCH_ATTRIBUTE_COOPERATIVE MU_LAUNCH_ATTRIBUTE_COOPERATIVE
#endif
#ifndef CU_LAUNCH_PARAM_BUFFER_POINTER
#define CU_LAUNCH_PARAM_BUFFER_POINTER MU_LAUNCH_PARAM_BUFFER_POINTER
#endif
#ifndef CU_LAUNCH_PARAM_BUFFER_SIZE
#define CU_LAUNCH_PARAM_BUFFER_SIZE MU_LAUNCH_PARAM_BUFFER_SIZE
#endif
#ifndef CU_LAUNCH_PARAM_END
#define CU_LAUNCH_PARAM_END MU_LAUNCH_PARAM_END
#endif
#ifndef CU_MEMCPY_SRC_ACCESS_ORDER_ANY
#define CU_MEMCPY_SRC_ACCESS_ORDER_ANY MU_MEMCPY_SRC_ACCESS_ORDER_ANY
#endif
#ifndef CU_MEMCPY_SRC_ACCESS_ORDER_STREAM
#define CU_MEMCPY_SRC_ACCESS_ORDER_STREAM MU_MEMCPY_SRC_ACCESS_ORDER_STREAM
#endif
#ifndef CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#endif
#ifndef CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED MU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_MUSA_VMM_SUPPORTED
#endif
#ifndef CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED MU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
#endif
#ifndef CU_POINTER_ATTRIBUTE_CONTEXT
#define CU_POINTER_ATTRIBUTE_CONTEXT MU_POINTER_ATTRIBUTE_CONTEXT
#endif
#ifndef CU_POINTER_ATTRIBUTE_DEVICE_POINTER
#define CU_POINTER_ATTRIBUTE_DEVICE_POINTER MU_POINTER_ATTRIBUTE_DEVICE_POINTER
#endif
#ifndef CU_POINTER_ATTRIBUTE_HOST_POINTER
#define CU_POINTER_ATTRIBUTE_HOST_POINTER MU_POINTER_ATTRIBUTE_HOST_POINTER
#endif
#ifndef CU_POINTER_ATTRIBUTE_MEMORY_TYPE
#define CU_POINTER_ATTRIBUTE_MEMORY_TYPE MU_POINTER_ATTRIBUTE_MEMORY_TYPE
#endif
#ifndef CU_POINTER_ATTRIBUTE_RANGE_SIZE
#define CU_POINTER_ATTRIBUTE_RANGE_SIZE MU_POINTER_ATTRIBUTE_RANGE_SIZE
#endif
#ifndef CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#define CU_POINTER_ATTRIBUTE_RANGE_START_ADDR MU_POINTER_ATTRIBUTE_RANGE_START_ADDR
#endif
#ifndef CUDA_ERROR_INVALID_VALUE
#define CUDA_ERROR_INVALID_VALUE MUSA_ERROR_INVALID_VALUE
#endif
#ifndef CUDA_ERROR_NOT_INITIALIZED
#define CUDA_ERROR_NOT_INITIALIZED MUSA_ERROR_NOT_INITIALIZED
#endif
#ifndef CUDA_ERROR_NOT_PERMITTED
#define CUDA_ERROR_NOT_PERMITTED MUSA_ERROR_NOT_PERMITTED
#endif
#ifndef CUDA_ERROR_NOT_SUPPORTED
#define CUDA_ERROR_NOT_SUPPORTED MUSA_ERROR_NOT_SUPPORTED
#endif
#ifndef CUDA_ERROR_OUT_OF_MEMORY
#define CUDA_ERROR_OUT_OF_MEMORY MUSA_ERROR_OUT_OF_MEMORY
#endif
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS MUSA_SUCCESS
#endif
