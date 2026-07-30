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
