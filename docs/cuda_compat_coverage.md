# CUDA compatibility coverage

`cuda_compat` supports direct compiler consumers of CUDA-named JIT sources.
`stable_compat` remains the libtorch Stable ABI layer.

The audit is intentionally scoped and classified:

- `mapped`: the identifier exists in the shared CUDA-to-MUSA mapping tables.
- `known-unmapped`: the identifier is intentionally listed for MUSA probing but
  has no automatic rewrite rule.
- `unclassified`: newly discovered CUDA runtime or driver API usage that still
  needs owner classification.

Run the source inventory with:

```bash
PYTHONPATH=src python3 scripts/audit_jit_cuda_usage.py \
  "$SGLANG_ROOT/python/sglang/kernels/jit" \
  --json /tmp/sglang-jit-cuda-audit.json
```

For vLLM, `vllm/distributed/device_communicators/pynccl_allocator.py` is the
current inline JIT source and uses no CUDA runtime API. The `csrc` tree is an
AOT extension rather than a JIT consumer, but its CUDA API inventory can be
audited separately:

```bash
PYTHONPATH=src python3 scripts/audit_jit_cuda_usage.py \
  "$VLLM_ROOT/csrc" "$VLLM_ROOT/vllm" \
  --json /tmp/vllm-upstream-cuda-audit.json
```

## Current inventory

| Tree | Source files | Unique symbols | Mapped | Known unmapped | Unclassified |
| --- | ---: | ---: | ---: | ---: | ---: |
| SGLang JIT | 342 | 83 | 31 | 19 | 33 |
| vLLM CUDA/AOT source inventory | 383 | 104 | 57 | 16 | 31 |

The vLLM row is not a claim that all of `csrc` is a JIT target.

## Known gaps

- Driver APIs such as `cuGetErrorName`, `cuGetProcAddress`,
  `cuDeviceGetAttribute`, and `cuMemcpyBatchAsync` are not part of the current
  `cuda_compat` header contract.
- TMA and tensor-map APIs such as `CUtensorMap`,
  `cuTensorMapEncodeTiled`, and `cudaTypedefs` require separate header and
  semantic coverage.
- IPC APIs such as `cudaIpcGetMemHandle`, `cudaIpcOpenMemHandle`, and
  `cudaIpcCloseMemHandle` are known unmapped because they need peer-access
  semantics rather than simple spelling aliases.
- Stream capture APIs and stream-capture status values are known unmapped.
- Batch/symbol memory-copy APIs and pointer/device attribute types are still
  unclassified and need MUSA capability probes.
- Cooperative and cluster launch names remain explicit `USE_MUSA` decisions in
  consumers unless their signatures and semantics are proven equivalent.

The JIT symbol probe performs the MUSA-side spelling check:

```bash
python3 scripts/probe_cuda_compat_symbols.py \
  --mcc /path/to/mcc --unmapped --json /tmp/cuda-compat-probe.json
```

`unclassified` is not a pass. A CUDA name is only promoted into
`cuda_compat` after it has a mapping rule and a real compiler test.
