import pytest
import torch


def _require_musa():
    import torchada

    if (
        not torchada.is_musa_platform()
        or not hasattr(torch, "musa")
        or not torch.musa.is_available()
    ):
        pytest.skip("MUSA platform required")


def test_multinomial_privateuse1_smoke():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    probs = torch.softmax(torch.randn(4, 64, device="cuda"), dim=-1)

    out = torch.multinomial(probs, 1)
    torch.cuda.synchronize()

    assert out.shape == (4, 1)
    assert out.dtype == torch.long
    assert out.device.type in ("cuda", "musa")
    assert int(out.min()) >= 0
    assert int(out.max()) < 64


def test_multinomial_without_replacement_unique():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    probs = torch.full((128, 16), 1.0 / 16, device="cuda")

    out = torch.multinomial(probs, 8, replacement=False).cpu()

    assert all(len(set(row.tolist())) == 8 for row in out)


def test_multinomial_distribution_sanity():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    rows = 4096
    weights = torch.tensor(
        [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.37],
        device="cuda",
    ).repeat(rows, 1)

    out = torch.multinomial(weights, 1).flatten()
    counts = torch.bincount(out.cpu(), minlength=weights.shape[1])

    assert int(counts.argmax()) == 6
    assert counts[-1] > counts[-2] > counts[-3]


def test_multinomial_graph_capture_replay():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    probs = torch.softmax(torch.randn(2, 128, device="cuda"), dim=-1)
    for _ in range(3):
        out = torch.multinomial(probs, 1)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = torch.multinomial(probs, 1)

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    assert out.shape == (2, 1)
    assert int(out.min()) >= 0
    assert int(out.max()) < 128


def test_multinomial_accepts_generator_argument():
    _require_musa()
    from torchada import _cpp_ops

    _cpp_ops.load_cpp_ops(force_reload=True)
    probs = torch.full((4, 32), 1.0 / 32, device="cuda")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(1234)

    out = torch.multinomial(probs, 1, generator=generator)
    torch.cuda.synchronize()

    assert out.shape == (4, 1)
    assert int(out.min()) >= 0
    assert int(out.max()) < 32
