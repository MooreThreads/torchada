#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <initializer_list>
#include <type_traits>

static_assert(std::is_same_v<__nv_bfloat16, __mt_bfloat16>);
static_assert(std::is_same_v<__nv_bfloat162, __mt_bfloat162>);
static_assert(std::is_same_v<__nv_fp8_e4m3, __mt_fp8_e4m3>);
static_assert(std::is_same_v<__nv_fp8_e5m2, __mt_fp8_e5m2>);
static_assert(std::is_same_v<__nv_fp8x2_e4m3, __mt_fp8x2_e4m3>);
static_assert(std::is_same_v<__nv_fp8x2_e5m2, __mt_fp8x2_e5m2>);
static_assert(std::is_same_v<__nv_fp8x4_e4m3, __mt_fp8x4_e4m3>);
static_assert(std::is_same_v<__nv_fp8x4_e5m2, __mt_fp8x4_e5m2>);

__global__ void copy_half(__half* output, const __half* input) {
  output[threadIdx.x] = input[threadIdx.x];
}

cudaError_t launch_copy(__half* output, const __half* input) {
  cudaLaunchAttribute attribute{};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(1);
  config.blockDim = dim3(32);
  config.stream = cudaStreamPerThread;
  config.attrs = &attribute;
  config.numAttrs = 0;
  return cudaLaunchKernelEx(&config, copy_half, output, input);
}

void runtime_api(cudaStream_t stream, void* output, const void* input) {
  int device, value, blocks;
  std::size_t bytes;
  cudaGetDevice(&device);
  cudaRuntimeGetVersion(&value);
  for (auto attr : {cudaDevAttrComputeCapabilityMajor,
                    cudaDevAttrComputeCapabilityMinor,
                    cudaDevAttrMultiProcessorCount,
                    cudaDevAttrMaxSharedMemoryPerBlock,
                    cudaDevAttrMaxSharedMemoryPerBlockOptin}) {
    cudaDeviceGetAttribute(&value, attr, device);
  }
  cudaMemcpy(output, input, 32, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(output, input, 32, cudaMemcpyDeviceToHost, stream);
  cudaHostGetDevicePointer(&output, output, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, copy_half, 32, 0);
  cudaOccupancyAvailableDynamicSMemPerBlock(&bytes, copy_half, 1, 32);
  cudaFuncSetAttribute(copy_half, cudaFuncAttributeMaxDynamicSharedMemorySize, 0);
  cudaLaunchKernel(reinterpret_cast<const void*>(copy_half), dim3(1), dim3(32), nullptr, 0, stream);
  if (cudaGetLastError() != cudaSuccess) cudaGetErrorString(cudaGetLastError());
}
