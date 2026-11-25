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
// A JIT that carries its own compiler driver (tvm-ffi and upstream torch's
// load_inline) never reaches the first, so it needs the second. Only
// `torchada.utils.cpp_extension.load_inline` applies the source porting. The two
// mechanisms must agree: `TestCudaCompatHeaderParity` in `tests/test_mappings.py`
// asserts that every `cudaX -> musaX` entry in the runtime porting table appears
// here, so a name added to one mechanism cannot silently be missing from the other.
//
// This directory must precede CUDA toolkit include directories. It shadows
// headers the MUSA SDK does not ship (`/usr/local/musa/include` carries
// `musa_*` names only); a `CUDA_HOME` left on a build box would otherwise win
// and the compile would bind to the wrong runtime.
//
// Every alias is guarded, so a porting layer that already provides a name - or a
// future MUSA header that defines the CUDA spelling itself - takes precedence.
//
// Out of scope on purpose: torch-level names (`at::kCUDA`, `c10::CUDAStream`,
// `torch::cuda::*`). Those are torch API, not the CUDA runtime API, and they
// belong to the stable-ABI shim (`csrc/stable_compat`) instead.

#include <musa_runtime.h>

#ifndef __grid_constant__
#define __grid_constant__
#endif

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
#ifndef cudaDeviceAttr
#define cudaDeviceAttr musaDeviceAttr
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
#ifndef cudaLaunchAttribute
#define cudaLaunchAttribute musaLaunchAttribute
#endif
#ifndef cudaIpcMemHandle_t
#define cudaIpcMemHandle_t musaIpcMemHandle_t
#endif
#ifndef cudaPointerAttributes
#define cudaPointerAttributes musaPointerAttributes
#endif
#ifndef cudaMemcpyAttributes
#define cudaMemcpyAttributes musaMemcpyAttributes
#endif
#ifndef cudaDriverEntryPointQueryResult
#define cudaDriverEntryPointQueryResult musaDriverEntryPointQueryResult
#endif
#ifndef cudaStreamCaptureStatus
#define cudaStreamCaptureStatus musaStreamCaptureStatus
#endif
#ifndef cudaStreamCaptureMode
#define cudaStreamCaptureMode musaStreamCaptureMode
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
#ifndef cudaErrorCallRequiresNewerDriver
#define cudaErrorCallRequiresNewerDriver musaErrorCallRequiresNewerDriver
#endif
#ifndef cudaErrorIllegalInstruction
#define cudaErrorIllegalInstruction musaErrorIllegalInstruction
#endif
#ifndef cudaErrorInvalidConfiguration
#define cudaErrorInvalidConfiguration musaErrorInvalidConfiguration
#endif
#ifndef cudaErrorInvalidValue
#define cudaErrorInvalidValue musaErrorInvalidValue
#endif
#ifndef cudaErrorNotSupported
#define cudaErrorNotSupported musaErrorNotSupported
#endif
#ifndef cudaErrorStreamCaptureInvalidated
#define cudaErrorStreamCaptureInvalidated musaErrorStreamCaptureInvalidated
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
#ifndef cudaMemcpyDefault
#define cudaMemcpyDefault musaMemcpyDefault
#endif
#ifndef cudaMemcpyKind
#define cudaMemcpyKind musaMemcpyKind
#endif
#ifndef cudaMemLocationTypeDevice
#define cudaMemLocationTypeDevice musaMemLocationTypeDevice
#endif
#ifndef cudaMemLocationTypeHost
#define cudaMemLocationTypeHost musaMemLocationTypeHost
#endif
#ifndef cudaMemLocationType
#define cudaMemLocationType musaMemLocationType
#endif
#ifndef cudaMemcpySrcAccessOrderStream
#define cudaMemcpySrcAccessOrderStream musaMemcpySrcAccessOrderStream
#endif
#ifndef cudaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#endif
#ifndef cudaFuncAttributePreferredSharedMemoryCarveout
#define cudaFuncAttributePreferredSharedMemoryCarveout musaFuncAttributePreferredSharedMemoryCarveout
#endif
#ifndef cudaFuncAttributeNonPortableClusterSizeAllowed
#define cudaFuncAttributeNonPortableClusterSizeAllowed musaFuncAttributeNonPortableClusterSizeAllowed
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
#ifndef cudaDevAttrMaxSharedMemoryPerMultiprocessor
#define cudaDevAttrMaxSharedMemoryPerMultiprocessor musaDevAttrMaxSharedMemoryPerMultiprocessor
#endif
#ifndef cudaDevAttrMaxThreadsPerMultiProcessor
#define cudaDevAttrMaxThreadsPerMultiProcessor musaDevAttrMaxThreadsPerMultiProcessor
#endif
#ifndef cudaDevAttrWarpSize
#define cudaDevAttrWarpSize musaDevAttrWarpSize
#endif
#ifndef cudaDevAttrMultiProcessorCount
#define cudaDevAttrMultiProcessorCount musaDevAttrMultiProcessorCount
#endif
#ifndef cudaDevAttrClockRate
#define cudaDevAttrClockRate musaDevAttrClockRate
#endif
#ifndef cudaStreamDefault
#define cudaStreamDefault musaStreamDefault
#endif
#ifndef cudaStreamNonBlocking
#define cudaStreamNonBlocking musaStreamNonBlocking
#endif
#ifndef cudaStreamPerThread
#define cudaStreamPerThread musaStreamPerThread
#endif
#ifndef cudaStreamLegacy
#define cudaStreamLegacy musaStreamLegacy
#endif
#ifndef cudaStreamCaptureStatusActive
#define cudaStreamCaptureStatusActive musaStreamCaptureStatusActive
#endif
#ifndef cudaStreamCaptureStatusNone
#define cudaStreamCaptureStatusNone musaStreamCaptureStatusNone
#endif
#ifndef cudaStreamCaptureModeRelaxed
#define cudaStreamCaptureModeRelaxed musaStreamCaptureModeRelaxed
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
#ifndef cudaLaunchAttributeProgrammaticStreamSerialization
#define cudaLaunchAttributeProgrammaticStreamSerialization musaLaunchAttributeIgnore
#endif
#ifndef cudaDriverEntryPointSuccess
#define cudaDriverEntryPointSuccess musaDriverEntryPointSuccess
#endif
#ifndef cudaEnableDefault
#define cudaEnableDefault musaEnableDefault
#endif
#ifndef cudaIpcMemLazyEnablePeerAccess
#define cudaIpcMemLazyEnablePeerAccess musaIpcMemLazyEnablePeerAccess
#endif
#ifndef cudaRoundNearest
#define cudaRoundNearest musaRoundNearest
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
#ifndef cudaDevAttrComputeCapabilityMajor
#define cudaDevAttrComputeCapabilityMajor musaDevAttrComputeCapabilityMajor
#endif
#ifndef cudaDevAttrComputeCapabilityMinor
#define cudaDevAttrComputeCapabilityMinor musaDevAttrComputeCapabilityMinor
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
#ifndef cudaGetErrorName
#define cudaGetErrorName musaGetErrorName
#endif
#ifndef cudaPeekAtLastError
#define cudaPeekAtLastError musaPeekAtLastError
#endif
#ifndef cudaHostAlloc
#define cudaHostAlloc musaHostAlloc
#endif
#ifndef cudaHostFree
#define cudaHostFree musaFreeHost
#endif
#ifndef cudaHostGetDevicePointer
#define cudaHostGetDevicePointer musaHostGetDevicePointer
#endif
#ifndef cudaHostRegister
#define cudaHostRegister musaHostRegister
#endif
#ifndef cudaHostUnregister
#define cudaHostUnregister musaHostUnregister
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
#ifndef cudaMemcpyToSymbol
#define cudaMemcpyToSymbol musaMemcpyToSymbol
#endif
#ifndef cudaMemcpyBatchAsync
#define cudaMemcpyBatchAsync musaMemcpyBatchAsync
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
#ifndef cudaDeviceGetStreamPriorityRange
#define cudaDeviceGetStreamPriorityRange musaDeviceGetStreamPriorityRange
#endif
#ifndef cudaOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor
#endif
#ifndef cudaOccupancyMaxPotentialBlockSize
#define cudaOccupancyMaxPotentialBlockSize musaOccupancyMaxPotentialBlockSize
#endif
#ifndef cudaOccupancyAvailableDynamicSMemPerBlock
#define cudaOccupancyAvailableDynamicSMemPerBlock musaOccupancyAvailableDynamicSMemPerBlock
#endif
#ifndef cudaOccupancyMaxActiveClusters
#define cudaOccupancyMaxActiveClusters musaOccupancyMaxActiveClusters
#endif
#ifndef cudaOccupancyMaxPotentialClusterSize
#define cudaOccupancyMaxPotentialClusterSize musaOccupancyMaxPotentialClusterSize
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
#ifndef cudaLaunchAttributeClusterDimension
#define cudaLaunchAttributeClusterDimension musaLaunchAttributeClusterDimension
#endif
#ifndef cudaLaunchAttributeCooperative
#define cudaLaunchAttributeCooperative musaLaunchAttributeCooperative
#endif
#ifndef cudaLaunchKernelEx
#define cudaLaunchKernelEx musaLaunchKernelEx
#endif
#ifndef cudaLaunchKernel
#define cudaLaunchKernel musaLaunchKernel
#endif
#ifndef cudaLaunchCooperativeKernel
#define cudaLaunchCooperativeKernel musaLaunchCooperativeKernel
#endif
#ifndef cudaIpcGetMemHandle
#define cudaIpcGetMemHandle musaIpcGetMemHandle
#endif
#ifndef cudaIpcOpenMemHandle
#define cudaIpcOpenMemHandle musaIpcOpenMemHandle
#endif
#ifndef cudaIpcCloseMemHandle
#define cudaIpcCloseMemHandle musaIpcCloseMemHandle
#endif
#ifndef cudaDriverGetVersion
#define cudaDriverGetVersion musaDriverGetVersion
#endif
#ifndef cudaGetDriverEntryPoint
#define cudaGetDriverEntryPoint musaGetDriverEntryPoint
#endif
#ifndef cudaGetDriverEntryPointByVersion
#define cudaGetDriverEntryPointByVersion musaGetDriverEntryPointByVersion
#endif
#ifndef cudaStreamGetCaptureInfo
#define cudaStreamGetCaptureInfo musaStreamGetCaptureInfo
#endif
#ifndef cudaStreamIsCapturing
#define cudaStreamIsCapturing musaStreamIsCapturing
#endif
#ifndef cudaThreadExchangeStreamCaptureMode
#define cudaThreadExchangeStreamCaptureMode musaThreadExchangeStreamCaptureMode
#endif
#ifndef cudaRuntimeGetVersion
#define cudaRuntimeGetVersion musaRuntimeGetVersion
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
