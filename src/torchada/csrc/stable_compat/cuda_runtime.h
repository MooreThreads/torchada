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

#ifndef CUDA_VERSION
#ifdef MUSA_VERSION
#define CUDA_VERSION MUSA_VERSION
#else
#define CUDA_VERSION 12000
#endif
#endif
#ifndef CUDART_VERSION
#ifdef MUSART_VERSION
#define CUDART_VERSION MUSART_VERSION
#else
#define CUDART_VERSION CUDA_VERSION
#endif
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
#define cudaLaunchAttributeProgrammaticStreamSerialization musaLaunchAttributeProgrammaticStreamSerialization
#endif
#ifndef cudaLaunchAttributeClusterDimension
#define cudaLaunchAttributeClusterDimension musaLaunchAttributeClusterDimension
#endif
#ifndef cudaLaunchKernelEx
#define cudaLaunchKernelEx musaLaunchKernelEx
#endif
#ifndef cudaLaunchKernel
#define cudaLaunchKernel musaLaunchKernel
#endif
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaOccupancyAvailableDynamicSMemPerBlock
#define cudaOccupancyAvailableDynamicSMemPerBlock musaOccupancyAvailableDynamicSMemPerBlock
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute musaDeviceGetAttribute
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
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion musaRuntimeGetVersion
#endif
#ifndef cudaStreamPerThread
#define cudaStreamPerThread musaStreamPerThread
#endif
