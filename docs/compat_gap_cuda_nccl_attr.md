# CUDA NCCL Module Attribute Compatibility Gap

## Status

- Fixed in `src/torchada/_patch.py`
- Covered by `tests/test_cuda_patching.py::TestNCCLModule::test_nccl_module_alias_available`

## Gap

CUDA exposes `torch.cuda.nccl` as both an importable module and a module
attribute. torchada already registered `torch.cuda.nccl` in `sys.modules`, but
plain attribute access still failed on MUSA because the CUDA wrapper redirected
`torch.cuda.nccl` to missing `torch.musa.nccl` instead of `torch.musa.mccl`.

## Fix

torchada now aliases `torch.musa.nccl` to `torch.musa.mccl` when MCCL is
available and also remaps `torch.cuda.nccl` attribute access to `mccl`.

## Verification

Run in the MUSA test container:

```bash
docker exec -w /ws yeahdongcn1 python -m pytest \
  tests/test_cuda_patching.py::TestNCCLModule::test_nccl_module_alias_available -v
```
