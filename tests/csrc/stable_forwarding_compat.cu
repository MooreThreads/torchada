#include <cuda_fp8.h>
#include <cuda_runtime.h>

__nv_fp8_storage_t stable_fp8_storage;
__nv_fp8x2_storage_t stable_fp8x2_storage;
__nv_fp8x4_storage_t stable_fp8x4_storage;

cudaError_t stable_runtime_probe(cudaStream_t stream) {
  (void)stream;
  return cudaGetLastError();
}
