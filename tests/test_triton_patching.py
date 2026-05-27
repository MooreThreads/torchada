import pytest

pytestmark = pytest.mark.musa


def test_triton_extra_cuda_namespace_is_available_on_musa():
    import torchada  # noqa: F401

    tl = pytest.importorskip("triton.language")

    assert hasattr(tl.extra, "cuda")
    if hasattr(tl.extra, "musa") and hasattr(tl.extra.musa, "libdevice"):
        assert hasattr(tl.extra.cuda, "libdevice")


def test_triton_extra_cuda_gdc_functions_are_available_on_musa():
    import torchada  # noqa: F401

    tl = pytest.importorskip("triton.language")

    assert hasattr(tl.extra, "cuda")
    assert hasattr(tl.extra.cuda, "gdc_wait")
    assert hasattr(tl.extra.cuda, "gdc_launch_dependents")

    with pytest.raises(NotImplementedError, match="gdc_wait is not supported on MUSA"):
        tl.extra.cuda.gdc_wait()
    with pytest.raises(
        NotImplementedError,
        match="gdc_launch_dependents is not supported on MUSA",
    ):
        tl.extra.cuda.gdc_launch_dependents()
