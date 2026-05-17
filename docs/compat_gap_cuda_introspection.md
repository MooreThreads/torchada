# CUDA Introspection Compatibility Gap

## Status

- Fixed in `src/torchada/_patch.py`
- Covered by `tests/test_cuda_patching.py::TestCudaBuildAndDebugIntrospection`

## Gap

In the `yeahdongcn1` torch_musa 2.7.1 container, these top-level CUDA APIs exist
on `torch.cuda` but are absent from `torch.musa`:

- `torch.cuda.get_gencode_flags`
- `torch.cuda.get_sync_debug_mode`
- `torch.cuda.set_sync_debug_mode`

After torchada redirects `torch.cuda` to `torch.musa`, those calls raised
`AttributeError` instead of preserving CUDA-compatible API access.

## Fix

torchada now installs MUSA-safe shims when torch_musa does not provide these
attributes:

- `get_gencode_flags()` returns `""` because NVCC gencode flags are CUDA-specific
  and should not be passed to the MUSA toolchain.
- `get_sync_debug_mode()` and `set_sync_debug_mode()` maintain a process-local
  debug mode value so CUDA-oriented code can call the public API without
  requiring unavailable CUDA C++ hooks.

## Verification

Run in the MUSA test container:

```bash
docker exec -w /ws yeahdongcn1 python -m pytest \
  tests/test_cuda_patching.py::TestCudaBuildAndDebugIntrospection -v
```
