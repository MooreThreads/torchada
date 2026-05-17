# CUDA Public API Alias Compatibility Gap

## Status

- Fixed in `src/torchada/_patch.py`
- Covered by `tests/test_cuda_patching.py::TestCudaPublicApiAliases`

## Gap

A broader `dir(torch.cuda)` versus `dir(torch.musa)` comparison in the
`yeahdongcn1` torch_musa 2.7.1 container found additional CUDA public attributes
that are commonly used as imports or compatibility aliases but were missing
after torchada redirected `torch.cuda` to `torch.musa`.

## Fix

torchada now provides MUSA-backed or safe compatibility aliases for:

- Deprecated memory aliases: `memory_cached`, `max_memory_cached`
- Host-memory stat APIs with no MUSA counters: `host_memory_stats`,
  `host_memory_stats_as_nested_dict`, `reset_accumulated_host_memory_stats`,
  `reset_peak_host_memory_stats`
- Static CUDA build flags: `has_half`, `has_magma`
- Top-level `CUDAPluggableAllocator`
- `torch.cuda.streams`
- `torch.cuda.sparse`
- `torch.cuda.init`
- `torch.cuda.default_generators`
- `torch.cuda.get_stream_from_external`

## Deferred

Allocator-control APIs such as `caching_allocator_enable` and telemetry APIs
such as `utilization` remain deferred because torch_musa has no equivalent
behavior in the tested build.

## Verification

Run in the MUSA test container:

```bash
docker exec -w /ws yeahdongcn1 python -m pytest \
  tests/test_cuda_patching.py::TestCudaPublicApiAliases -v
```
