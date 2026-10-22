"""Synchronize CUDA and MUSA visibility before either runtime initializes."""

import importlib.util
import os


def _should_sync_visible_devices_env() -> bool:
    """Detect a MUSA installation without importing torch or torch_musa."""
    forced_platform = os.environ.get("TORCHADA_PLATFORM", "").lower()
    if forced_platform in {"cuda", "cpu"}:
        return False
    if forced_platform == "musa":
        return True

    try:
        return importlib.util.find_spec("torch_musa") is not None
    except (ImportError, ValueError):
        return False


def _sync_visible_devices_env() -> None:
    """Mirror the authoritative visibility mask into its compatibility alias."""
    musa_visible = os.environ.get("MUSA_VISIBLE_DEVICES")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if musa_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = musa_visible
    elif cuda_visible is not None:
        os.environ["MUSA_VISIBLE_DEVICES"] = cuda_visible
