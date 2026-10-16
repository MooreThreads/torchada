#pragma once
// MUSA: there is no <cuda_runtime.h> on MUSA; CUDA-named MUSA consumers that
// include it get the MUSA runtime instead. Resolved through
// forwarding_include_dir(), not the stable ABI compatibility include path.
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

#ifndef __grid_constant__
#define __grid_constant__
#endif
#ifndef cudaStreamPerThread
#define cudaStreamPerThread musaStreamPerThread
#endif
#ifndef cudaGetLastError
#define cudaGetLastError musaGetLastError
#endif
#ifndef cudaLaunchConfig_t
#define cudaLaunchConfig_t musaLaunchConfig_t
#endif
#ifndef cudaLaunchAttribute
#define cudaLaunchAttribute musaLaunchAttribute
#endif
#ifndef cudaLaunchAttributeProgrammaticStreamSerialization
#define cudaLaunchAttributeProgrammaticStreamSerialization musaLaunchAttributeIgnore
#endif
#ifndef cudaLaunchAttributeCooperative
#define cudaLaunchAttributeCooperative musaLaunchAttributeCooperative
#endif
#ifndef cudaLaunchAttributeClusterDimension
#define cudaLaunchAttributeClusterDimension musaLaunchAttributeClusterDimension
#endif
#ifndef cudaLaunchAttributeClusterSchedulingPolicyPreference
#define cudaLaunchAttributeClusterSchedulingPolicyPreference \
    musaLaunchAttributeClusterSchedulingPolicyPreference
#endif
#ifndef cudaLaunchKernel
#define cudaLaunchKernel musaLaunchKernel
#endif
#ifndef cudaLaunchKernelEx
#define cudaLaunchKernelEx musaLaunchKernelEx
#endif
#ifndef cudaMemcpy
#define cudaMemcpy musaMemcpy
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync musaMemcpyAsync
#endif
#ifndef cudaMemcpyHostToDevice
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#endif
#ifndef cudaMemcpyDeviceToHost
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#endif
#ifndef cudaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#endif
#ifndef cudaMemcpyHostToHost
#define cudaMemcpyHostToHost musaMemcpyHostToHost
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#endif
#ifndef cudaRuntimeGetVersion
// This reports the MUSA runtime encoding, not a CUDA toolkit version.
#define cudaRuntimeGetVersion musaRuntimeGetVersion
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount musaDevAttrMultiProcessorCount
#endif
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor musaDevAttrComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor musaDevAttrComputeCapabilityMinor
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlock musaDevAttrMaxSharedMemoryPerBlock
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlockOptin
#define cudaDevAttrMaxSharedMemoryPerBlockOptin musaDevAttrMaxSharedMemoryPerBlockOptin
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer musaHostGetDevicePointer
#endif
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaOccupancyAvailableDynamicSMemPerBlock
#define cudaOccupancyAvailableDynamicSMemPerBlock musaOccupancyAvailableDynamicSMemPerBlock
#endif
#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute musaFuncSetAttribute
#endif
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#endif
