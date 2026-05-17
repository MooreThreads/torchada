# CUDA to MUSA Compatibility Gap Inventory

## Method

Compared selected `torch.cuda` attributes against `torch.musa` in the
`yeahdongcn1` torch_musa 2.7.1 container, then verified behavior after
`import torchada`.

## Fixed

- `torch.cuda.get_gencode_flags`
- `torch.cuda.get_sync_debug_mode`
- `torch.cuda.set_sync_debug_mode`
- `torch.cuda.nccl`
- Deprecated memory aliases and host-memory stat APIs
- `torch.cuda.streams`
- `torch.cuda.sparse`
- `torch.cuda.init`
- `torch.cuda.default_generators`
- `torch.cuda.get_stream_from_external`
- `torch.cuda.CUDAPluggableAllocator`

See:

- `docs/compat_gap_cuda_introspection.md`
- `docs/compat_gap_cuda_nccl_attr.md`
- `docs/compat_gap_cuda_public_aliases.md`

## Deferred

These remaining names are not patched as part of CUDA-to-MUSA runtime
compatibility:

- Imported helper symbols from the CUDA Python module: `Any`, `Callable`,
  `Optional`, `Union`, `cast`, `classproperty`, `importlib`, `lru_cache`,
  `threading`, `traceback`
- CUDA-only internal classes or APIs with no MUSA object model equivalent in the
  tested build: `CudaError`, `cudaStatus`, `DeferredCudaCallError`, `Device`,
  `ComplexFloatStorage`, `ComplexDoubleStorage`, `jiterator`

The following CUDA APIs are NVIDIA/NVML telemetry helpers and still have no
`torch.musa` equivalent in the tested torch_musa build:

- `torch.cuda.list_gpu_processes`
- `torch.cuda.utilization`
- `torch.cuda.memory_usage`
- `torch.cuda.temperature`
- `torch.cuda.power_draw`
- `torch.cuda.clock_rate`
- `torch.cuda.device_memory_used`
- `torch.cuda.caching_allocator_alloc`
- `torch.cuda.caching_allocator_delete`
- `torch.cuda.caching_allocator_enable`
- `torch.cuda.get_per_process_memory_fraction`
- `torch.cuda.gds`
- `torch.cuda.tunable`

In the same container, the original CUDA implementations depend on `pynvml` and
do not provide real values without NVIDIA NVML support. They are not patched in
this pass to avoid returning misleading MUSA telemetry. Allocator-control and
tunable/GDS APIs are also left unpatched because torch_musa does not expose an
equivalent behavior in this tested build.
