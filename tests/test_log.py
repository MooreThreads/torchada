import pytest
import torch


def _require_musa():
    import torchada

    if not torchada.is_musa_platform():
        pytest.skip("MUSA platform required")

    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA platform required")


def test_log_float64_privateuse1_smoke():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    x = torch.linspace(0.1, 2.0, 128, device="cuda", dtype=torch.float64)

    out = torch.log(x)
    torch.cuda.synchronize()

    expected = torch.log(x.cpu())
    torch.testing.assert_close(out.cpu(), expected, rtol=1e-12, atol=1e-12)


def test_log_inplace_float64_privateuse1_smoke():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    x = torch.linspace(0.1, 2.0, 128, device="cuda", dtype=torch.float64)
    expected = torch.log(x.cpu())

    ret = x.log_()
    torch.cuda.synchronize()

    assert ret is x
    torch.testing.assert_close(x.cpu(), expected, rtol=1e-12, atol=1e-12)


def test_log_inplace_float64_graph_capture_replay():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    x = torch.linspace(0.1, 2.0, 128, device="cuda", dtype=torch.float64)
    work = x.clone()
    for _ in range(3):
        work.copy_(x)
        work.log_()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        work.copy_(x)
        work.log_()

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    expected = torch.log(x.cpu())
    torch.testing.assert_close(work.cpu(), expected, rtol=1e-12, atol=1e-12)
