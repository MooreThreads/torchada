#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

__device__ int cuda_compat_symbol;

__global__ void cuda_compat_noop(float* value) {
  if (value != nullptr) {
    *value = 1.0f;
  }
}

void compile_cuda_compat_surface(float* value) {
  const dim3 grid(1);
  const dim3 block(1);
  void* args[] = {&value};
  cudaStream_t stream = nullptr;
  void* host_allocation = nullptr;
  cudaHostAlloc(&host_allocation, sizeof(float), cudaHostAllocDefault);
  cudaHostRegister(host_allocation, sizeof(float), 0);
  cudaHostUnregister(host_allocation);
  cudaHostFree(host_allocation);

  cudaLaunchKernel(reinterpret_cast<const void*>(&cuda_compat_noop), grid, block, args, 0, stream);
  cudaLaunchCooperativeKernel(
      reinterpret_cast<const void*>(&cuda_compat_noop), grid, block, args, 0, stream);

  cudaLaunchConfig_t config{};
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim.x = 1;
  attribute.val.clusterDim.y = 1;
  attribute.val.clusterDim.z = 1;
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  config.attrs = &attribute;
  config.numAttrs = 1;
  cudaLaunchKernelEx(&config, cuda_compat_noop, value);

  size_t shared_memory = 0;
  cudaOccupancyAvailableDynamicSMemPerBlock(&shared_memory, cuda_compat_noop, 1, 1);
  int cluster_count = 0;
  cudaOccupancyMaxActiveClusters(&cluster_count, cuda_compat_noop, &config);
  cudaOccupancyMaxPotentialClusterSize(&cluster_count, cuda_compat_noop, &config);

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &capture_status);
  cudaStreamCaptureMode capture_mode = cudaStreamCaptureModeRelaxed;
  cudaThreadExchangeStreamCaptureMode(&capture_mode);

  cudaPointerAttributes pointer_attributes{};
  cudaPointerGetAttributes(&pointer_attributes, value);
  cudaMemcpyAttributes copy_attributes{};
  copy_attributes.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
  cudaMemLocationType host_location = cudaMemLocationTypeHost;
  cudaMemLocationType device_location = cudaMemLocationTypeDevice;
  cudaMemcpyKind copy_kind = cudaMemcpyDefault;
  (void)host_location;
  (void)device_location;
  (void)copy_kind;

  void* dst = nullptr;
  const void* src = nullptr;
  size_t size = 0;
  size_t fail_index = 0;
  cudaMemcpyBatchAsync(
      &dst, &src, &size, 1, &copy_attributes, nullptr, 0, &fail_index, stream);
  cudaMemcpyToSymbol(cuda_compat_symbol, &shared_memory, sizeof(shared_memory));

  int driver_version = 0;
  cudaDriverGetVersion(&driver_version);
  int least_priority = 0;
  int greatest_priority = 0;
  cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
  void* entry = nullptr;
  cudaDriverEntryPointQueryResult query_result = cudaDriverEntryPointSuccess;
  cudaGetDriverEntryPoint("musaMalloc", &entry, cudaEnableDefault, &query_result);
  cudaGetDriverEntryPointByVersion(
      "musaMalloc", &entry, 12000, cudaEnableDefault, &query_result);

  cudaIpcMemHandle_t ipc_handle{};
  (void)ipc_handle;
  cudaError_t error = cudaSuccess;
  error = cudaErrorInvalidValue;
  error = cudaErrorNotSupported;
  error = cudaErrorCallRequiresNewerDriver;
  error = cudaErrorIllegalInstruction;
  error = cudaErrorStreamCaptureInvalidated;
  const char* error_name = cudaGetErrorName(error);
  (void)error;
  (void)error_name;

  const void* current_device = nullptr;
  cudaDeviceGetAttribute(
      nullptr, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  cudaDeviceGetAttribute(nullptr, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  (void)current_device;

  CUresult driver_result = CUDA_SUCCESS;
  const char* driver_error = nullptr;
  cuGetErrorName(driver_result, &driver_error);
  cuGetErrorString(driver_result, &driver_error);

  CUdeviceptr address = 0;
  size_t address_size = 0;
  cuMemGetAddressRange(nullptr, &address_size, address);
  int driver_attribute = 0;
  CUdevice_attribute device_attribute = CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR;
  cuDeviceGetAttribute(&driver_attribute, device_attribute, 0);

  CUmemAllocationGranularity_flags granularity = CU_MEM_ALLOC_GRANULARITY_MINIMUM;
  CUmemcpyAttributes driver_copy_attributes{};
  CUpointer_attribute pointer_attribute = CU_POINTER_ATTRIBUTE_MEMORY_TYPE;
  (void)granularity;
  (void)driver_copy_attributes;
  (void)pointer_attribute;

  CUdriverProcAddressQueryResult address_result = CU_GET_PROC_ADDRESS_SUCCESS;
  (void)address_result;
  CUfunction_attribute function_attribute =
      CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
  CUlaunchAttributeID launch_attribute = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  launch_attribute = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION;
  void* launch_param = CU_LAUNCH_PARAM_BUFFER_POINTER;
  launch_param = CU_LAUNCH_PARAM_BUFFER_SIZE;
  launch_param = CU_LAUNCH_PARAM_END;
  __nv_fp4x2_storage_t fp4_storage{};
  __nv_bfloat16_raw bfloat16_raw{};
  __nv_bfloat162_raw bfloat162_raw{};
  (void)device_attribute;
  (void)function_attribute;
  (void)launch_attribute;
  (void)launch_param;
  (void)fp4_storage;
  (void)bfloat16_raw;
  (void)bfloat162_raw;
}
