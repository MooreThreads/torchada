from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_audit_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_jit_cuda_usage.py"
    spec = importlib.util.spec_from_file_location("audit_jit_cuda_usage", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_scan_classifies_mapped_known_and_unclassified_cuda_names(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text(
        """
// cudaIgnoredFromComment();
CUTLASS_UNUSED();
cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
cudaLaunchKernelEx;
cuNewUnclassifiedApi(0);
""",
        encoding="utf-8",
    )

    report = _load_audit_module().scan([tmp_path])

    assert report["files"] == 1
    assert report["findings"]["cudaMemcpyAsync"]["status"] == "mapped"
    assert report["findings"]["cudaLaunchKernelEx"]["status"] == "known-unmapped"
    assert report["findings"]["cuNewUnclassifiedApi"]["status"] == "unclassified"
    assert "cudaIgnoredFromComment" not in report["findings"]
    assert "CUTLASS_UNUSED" not in report["findings"]
