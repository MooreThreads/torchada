#pragma once
// MUSA: the CUDA-named runtime API surface, for sources that were written
// against CUDA and are compiled by mcc without a porting pass.
//
// WHY THIS FILE EXISTS. Two mechanisms replace CUDA names with MUSA ones:
//
//   * the porting table (`torchada/_mappings/cuda_runtime.py`), which rewrites
//     source text in place at build time, and
//   * this header, which the compiler resolves instead of an absent
//     `<cuda_runtime.h>`.
//
// A JIT that carries its own compiler driver (tvm-ffi, and anything else that
// does not go through `torch.utils.cpp_extension`) never reaches the first, so
// it needs the second. The two must agree: `TestCudaCompatHeaderParity` in
// `tests/test_mappings.py` asserts that every `cudaX -> musaX` entry in the runtime
// porting table appears here, so a name added to one mechanism cannot silently be
// missing from the other.
//
// This directory must be the FIRST include path. It shadows headers the MUSA SDK
// does not ship (`/usr/local/musa/include` carries `musa_*` names only); a
// `CUDA_HOME` left on a build box would otherwise win and the compile would bind
// to the wrong runtime.
//
// Every alias is guarded, so a porting layer that already provides a name - or a
// future MUSA header that defines the CUDA spelling itself - takes precedence.
//
// Out of scope on purpose: torch-level names (`at::kCUDA`, `c10::CUDAStream`,
// `torch::cuda::*`). Those are torch API, not the CUDA runtime API, and they
// belong to the stable-ABI shim (`csrc/stable_compat`) instead.

#include <musa_runtime.h>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
#ifndef cudaStream_t
#define cudaStream_t musaStream_t
#endif
#ifndef cudaEvent_t
#define cudaEvent_t musaEvent_t
#endif
#ifndef cudaError_t
#define cudaError_t musaError_t
#endif
#ifndef cudaDeviceProp
#define cudaDeviceProp musaDeviceProp
#endif
#ifndef cudaFuncAttributes
#define cudaFuncAttributes musaFuncAttributes
#endif
#ifndef cudaTextureObject_t
#define cudaTextureObject_t musaTextureObject_t
#endif
#ifndef cudaSurfaceObject_t
#define cudaSurfaceObject_t musaSurfaceObject_t
#endif
#ifndef cudaLaunchConfig_t
#define cudaLaunchConfig_t musaLaunchConfig_t
#endif

// ---------------------------------------------------------------------------
// Constants and enum values
//
// Same names, `musa` prefix: the runtime's flag and attribute namespaces are
// 1:1, so these are renames, not translations.
// ---------------------------------------------------------------------------
#ifndef cudaSuccess
#define cudaSuccess musaSuccess
#endif
#ifndef cudaErrorNotReady
#define cudaErrorNotReady musaErrorNotReady
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
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#endif
#ifndef cudaFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncAttributePreferredSharedMemoryCarveout musaFuncAttributePreferredSharedMemoryCarveout
#endif
#ifndef cudaDevAttrMaxThreadsPerBlock
#define cudaDevAttrMaxThreadsPerBlock musaDevAttrMaxThreadsPerBlock
#endif
#ifndef cudaDevAttrMaxBlockDimX
#define cudaDevAttrMaxBlockDimX musaDevAttrMaxBlockDimX
#endif
#ifndef cudaDevAttrMaxBlockDimY
#define cudaDevAttrMaxBlockDimY musaDevAttrMaxBlockDimY
#endif
#ifndef cudaDevAttrMaxBlockDimZ
#define cudaDevAttrMaxBlockDimZ musaDevAttrMaxBlockDimZ
#endif
#ifndef cudaDevAttrMaxGridDimX
#define cudaDevAttrMaxGridDimX musaDevAttrMaxGridDimX
#endif
#ifndef cudaDevAttrMaxGridDimY
#define cudaDevAttrMaxGridDimY musaDevAttrMaxGridDimY
#endif
#ifndef cudaDevAttrMaxGridDimZ
#define cudaDevAttrMaxGridDimZ musaDevAttrMaxGridDimZ
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlock
#define cudaDevAttrMaxSharedMemoryPerBlock musaDevAttrMaxSharedMemoryPerBlock
#endif
#ifndef cudaDevAttrMaxSharedMemoryPerBlockOptin
#define cudaDevAttrMaxSharedMemoryPerBlockOptin musaDevAttrMaxSharedMemoryPerBlockOptin
#endif
#ifndef cudaDevAttrWarpSize
#define cudaDevAttrWarpSize musaDevAttrWarpSize
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
#ifndef cudaStreamDefault
#define cudaStreamDefault musaStreamDefault
#endif
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking musaStreamNonBlocking
#endif
#ifndef cudaEventDefault
#define cudaEventDefault musaEventDefault
#endif
#ifndef cudaEventBlockingSync
#define cudaEventBlockingSync musaEventBlockingSync
#endif
#ifndef cudaEventDisableTiming
#define cudaEventDisableTiming musaEventDisableTiming
#endif
#ifndef cudaHostAllocDefault
#define cudaHostAllocDefault musaHostAllocDefault
#endif
#ifndef cudaHostAllocPortable
#define cudaHostAllocPortable musaHostAllocPortable
#endif
#ifndef cudaHostAllocMapped
#define cudaHostAllocMapped musaHostAllocMapped
#endif
#ifndef cudaHostAllocWriteCombined
#define cudaHostAllocWriteCombined musaHostAllocWriteCombined
#endif
#ifndef cudaMemoryTypeHost
#define cudaMemoryTypeHost musaMemoryTypeHost
#endif
#ifndef cudaMemoryTypeDevice
#define cudaMemoryTypeDevice musaMemoryTypeDevice
#endif
#ifndef cudaMemoryTypeManaged
#define cudaMemoryTypeManaged musaMemoryTypeManaged
#endif
#ifndef cudaSharedmemCarveoutDefault
#define cudaSharedmemCarveoutDefault musaSharedmemCarveoutDefault
#endif
#ifndef cudaSharedmemCarveoutMaxL1
#define cudaSharedmemCarveoutMaxL1 musaSharedmemCarveoutMaxL1
#endif
#ifndef cudaSharedmemCarveoutMaxShared
#define cudaSharedmemCarveoutMaxShared musaSharedmemCarveoutMaxShared
#endif
#ifndef cudaLaunchAttributeProgrammaticStreamSerialization
#define cudaLaunchAttributeProgrammaticStreamSerialization musaLaunchAttributeIgnore
#endif

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------
#ifndef cudaMalloc
#define cudaMalloc musaMalloc
#endif
#ifndef cudaFree
#define cudaFree musaFree
#endif
#ifndef cudaMemcpy
#define cudaMemcpy musaMemcpy
#endif
#ifndef cudaMemcpyAsync
#define cudaMemcpyAsync musaMemcpyAsync
#endif
#ifndef cudaMemset
#define cudaMemset musaMemset
#endif
#ifndef cudaMemsetAsync
#define cudaMemsetAsync musaMemsetAsync
#endif
#ifndef cudaDeviceSynchronize
#define cudaDeviceSynchronize musaDeviceSynchronize
#endif
#ifndef cudaStreamSynchronize
#define cudaStreamSynchronize musaStreamSynchronize
#endif
#ifndef cudaGetDevice
#define cudaGetDevice musaGetDevice
#endif
#ifndef cudaSetDevice
#define cudaSetDevice musaSetDevice
#endif
#ifndef cudaGetDeviceCount
#define cudaGetDeviceCount musaGetDeviceCount
#endif
#ifndef cudaGetDeviceProperties
#define cudaGetDeviceProperties musaGetDeviceProperties
#endif
#ifndef cudaDeviceGetAttribute
#define cudaDeviceGetAttribute musaDeviceGetAttribute
#endif
#ifndef cudaStreamCreate
#define cudaStreamCreate musaStreamCreate
#endif
#ifndef cudaStreamDestroy
#define cudaStreamDestroy musaStreamDestroy
#endif
#ifndef cudaEventCreate
#define cudaEventCreate musaEventCreate
#endif
#ifndef cudaEventDestroy
#define cudaEventDestroy musaEventDestroy
#endif
#ifndef cudaEventRecord
#define cudaEventRecord musaEventRecord
#endif
#ifndef cudaEventSynchronize
#define cudaEventSynchronize musaEventSynchronize
#endif
#ifndef cudaEventElapsedTime
#define cudaEventElapsedTime musaEventElapsedTime
#endif
#ifndef cudaStreamWaitEvent
#define cudaStreamWaitEvent musaStreamWaitEvent
#endif
#ifndef cudaGetLastError
#define cudaGetLastError musaGetLastError
#endif
#ifndef cudaGetErrorString
#define cudaGetErrorString musaGetErrorString
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion musaRuntimeGetVersion
#endif
#ifndef cudaPeekAtLastError
#define cudaPeekAtLastError musaPeekAtLastError
#endif
#ifndef cudaHostAlloc
#define cudaHostAlloc musaHostAlloc
#endif
#ifndef cudaHostFree
#define cudaHostFree musaHostFree
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer musaHostGetDevicePointer
#endif
#ifndef cudaMallocHost
#define cudaMallocHost musaMallocHost
#endif
#ifndef cudaFreeHost
#define cudaFreeHost musaFreeHost
#endif
#ifndef cudaMallocManaged
#define cudaMallocManaged musaMallocManaged
#endif
#ifndef cudaMallocAsync
#define cudaMallocAsync musaMallocAsync
#endif
#ifndef cudaFreeAsync
#define cudaFreeAsync musaFreeAsync
#endif
#ifndef cudaMemcpy2D
#define cudaMemcpy2D musaMemcpy2D
#endif
#ifndef cudaMemcpy2DAsync
#define cudaMemcpy2DAsync musaMemcpy2DAsync
#endif
#ifndef cudaMemcpy3D
#define cudaMemcpy3D musaMemcpy3D
#endif
#ifndef cudaMemcpy3DAsync
#define cudaMemcpy3DAsync musaMemcpy3DAsync
#endif
#ifndef cudaMemGetInfo
#define cudaMemGetInfo musaMemGetInfo
#endif
#ifndef cudaMemPrefetchAsync
#define cudaMemPrefetchAsync musaMemPrefetchAsync
#endif
#ifndef cudaPointerGetAttributes
#define cudaPointerGetAttributes musaPointerGetAttributes
#endif
#ifndef cudaStreamCreateWithFlags
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#endif
#ifndef cudaStreamCreateWithPriority
#define cudaStreamCreateWithPriority musaStreamCreateWithPriority
#endif
#ifndef cudaStreamQuery
#define cudaStreamQuery musaStreamQuery
#endif
#ifndef cudaStreamGetPriority
#define cudaStreamGetPriority musaStreamGetPriority
#endif
#ifndef cudaStreamGetFlags
#define cudaStreamGetFlags musaStreamGetFlags
#endif
#ifndef cudaEventCreateWithFlags
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#endif
#ifndef cudaEventQuery
#define cudaEventQuery musaEventQuery
#endif
#ifndef cudaDeviceReset
#define cudaDeviceReset musaDeviceReset
#endif
#ifndef cudaDeviceSetCacheConfig
#define cudaDeviceSetCacheConfig musaDeviceSetCacheConfig
#endif
#ifndef cudaDeviceGetCacheConfig
#define cudaDeviceGetCacheConfig musaDeviceGetCacheConfig
#endif
#ifndef cudaDeviceSetSharedMemConfig
#define cudaDeviceSetSharedMemConfig musaDeviceSetSharedMemConfig
#endif
#ifndef cudaDeviceGetSharedMemConfig
#define cudaDeviceGetSharedMemConfig musaDeviceGetSharedMemConfig
#endif
#ifndef cudaGetDeviceFlags
#define cudaGetDeviceFlags musaGetDeviceFlags
#endif
#ifndef cudaSetDeviceFlags
#define cudaSetDeviceFlags musaSetDeviceFlags
#endif
#ifndef cudaDeviceCanAccessPeer
#define cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#endif
#ifndef cudaDeviceEnablePeerAccess
#define cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#endif
#ifndef cudaDeviceDisablePeerAccess
#define cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#endif
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSize musaOccupancyMaxPotentialBlockSize
#endif
#ifndef cudaFuncGetAttributes
#define cudaFuncGetAttributes musaFuncGetAttributes
#endif
#ifndef cudaFuncSetAttribute
#define cudaFuncSetAttribute musaFuncSetAttribute
#endif
#ifndef cudaFuncSetCacheConfig
#define cudaFuncSetCacheConfig musaFuncSetCacheConfig
#endif
#ifndef cudaCreateTextureObject
#define cudaCreateTextureObject musaCreateTextureObject
#endif
#ifndef cudaDestroyTextureObject
#define cudaDestroyTextureObject musaDestroyTextureObject
#endif
#ifndef cudaCreateSurfaceObject
#define cudaCreateSurfaceObject musaCreateSurfaceObject
#endif
#ifndef cudaDestroySurfaceObject
#define cudaDestroySurfaceObject musaDestroySurfaceObject
#endif
#ifndef cudaLaunchAttribute
#define cudaLaunchAttribute musaLaunchAttribute
#endif

// ---------------------------------------------------------------------------
// Programmatic dependent launch (PDL)
//
// MUSA has no PDL, and `sglang`'s JIT helpers already compile it out on MUSA
// (`is_arch_support_pdl()` is false there). The porting table neutralises both
// calls to `((void)0)`; these macros give source that is compiled without a
// porting pass the same meaning instead of a link error.
// ---------------------------------------------------------------------------
#ifndef cudaGridDependencySynchronize
#define cudaGridDependencySynchronize() ((void)0)
#endif
#ifndef cudaTriggerProgrammaticLaunchCompletion
#define cudaTriggerProgrammaticLaunchCompletion() ((void)0)
#endif
