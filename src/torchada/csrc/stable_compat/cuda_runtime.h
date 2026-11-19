#pragma once
// MUSA: there is no <cuda_runtime.h> on MUSA; libtorch-stable kernels that
// include it (e.g. torch_utils.h) get the MUSA runtime instead. Resolved ahead
// of any toolchain header via the stable_compat include dir.
#include <musa_runtime.h>

// MUSA: libtorch-stable kernels use the CUDA runtime spelling for device
// query/property APIs; map them to the MUSA runtime so those kernels compile
// against upstream names. Guarded so an active MCC cuda-porting layer that
// already provides a name takes precedence.
#ifndef cudaDeviceProp
#define cudaDeviceProp musaDeviceProp
#endif
#ifndef cudaError_t
#define cudaError_t musaError_t
#endif
#ifndef cudaSuccess
#define cudaSuccess musaSuccess
#endif
#ifndef cudaStream_t
#define cudaStream_t musaStream_t
#endif
#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount musaGetDeviceCount
#endif
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties musaGetDeviceProperties
#endif
#ifndef cudaGetDevice
#define cudaGetDevice musaGetDevice
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString musaGetErrorString
#endif
#ifndef cudaGetLastError
#define cudaGetLastError musaGetLastError
#endif

// Launch APIs and types.
#ifndef cudaLaunchConfig_t
#define cudaLaunchConfig_t musaLaunchConfig_t
#endif
#ifndef cudaLaunchAttribute
#define cudaLaunchAttribute musaLaunchAttribute
#endif
#ifndef cudaLaunchKernelEx
#define cudaLaunchKernelEx musaLaunchKernelEx
#endif
#ifndef cudaLaunchAttributeProgrammaticStreamSerialization
#define cudaLaunchAttributeProgrammaticStreamSerialization musaLaunchAttributeIgnore
#endif
#ifndef cudaLaunchAttributeClusterDimension
#define cudaLaunchAttributeClusterDimension musaLaunchAttributeClusterDimension
#endif
#ifndef cudaLaunchAttributeCooperative
#define cudaLaunchAttributeCooperative musaLaunchAttributeCooperative
#endif

// Memory APIs and copy kinds.
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync musaMemcpyAsync
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer musaHostGetDevicePointer
#endif

// Device, function, and shared-memory attributes.
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor musaDevAttrComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor musaDevAttrComputeCapabilityMinor
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount musaDevAttrMultiProcessorCount
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlockOptin
#define cudaDevAttrMaxSharedMemoryPerBlockOptin musaDevAttrMaxSharedMemoryPerBlockOptin
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlock musaDevAttrMaxSharedMemoryPerBlock
#endif
#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute musaFuncSetAttribute
#endif
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#endif
#ifndef cudaFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncAttributePreferredSharedMemoryCarveout musaFuncAttributePreferredSharedMemoryCarveout
#endif
#ifndef cudaSharedmemCarveoutMaxL1
#define cudaSharedmemCarveoutMaxL1 musaSharedmemCarveoutMaxL1
#endif
#ifndef cudaSharedmemCarveoutMaxShared
#define cudaSharedmemCarveoutMaxShared musaSharedmemCarveoutMaxShared
#endif
#ifndef cudaSharedmemCarveoutDefault
#define cudaSharedmemCarveoutDefault musaSharedmemCarveoutDefault
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion musaRuntimeGetVersion
#endif
