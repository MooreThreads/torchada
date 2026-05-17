"""
NVTX stub module for MUSA platform.

This module provides no-op implementations of NVTX (NVIDIA Tools Extension)
functions for profiling. On MUSA platform, these are stubs that do nothing.

Usage:
    import torchada  # Apply patches first.
    import torch.cuda.nvtx as nvtx

    nvtx.range_push("my_range")
    # Code to profile.
    nvtx.range_pop()

    # Or use the context manager.
    with nvtx.range("my_range"):
        # Code to profile.
"""

from contextlib import contextmanager

__all__ = [
    "mark",
    "range",
    "range_push",
    "range_pop",
    "range_start",
    "range_end",
]


def mark(msg: str) -> None:
    """
    Mark an instantaneous event in the timeline.

    This is a no-op on MUSA platform.

    Args:
        msg: The message to associate with the mark.
    """
    pass


def range_push(msg: str) -> int:
    """
    Push a range onto the stack.

    This is a no-op on MUSA platform.

    Args:
        msg: The message to associate with the range.

    Returns:
        The zero-based depth of the range that is started.
    """
    return 0


def range_pop() -> None:
    """
    Pop a range off the stack.

    This is a no-op on MUSA platform.
    """
    pass


def range_start(msg: str) -> int:
    """
    Start a range.

    This is a no-op on MUSA platform.

    Args:
        msg: The message to associate with the range.

    Returns:
        A range ID that can be passed to range_end.
    """
    return 0


def range_end(range_id: int) -> None:
    """
    End a range.

    This is a no-op on MUSA platform.

    Args:
        range_id: The range ID returned by range_start.
    """
    pass


@contextmanager
def range(msg: str, *args, **kwargs):
    """
    Context manager for NVTX ranges.

    This is a no-op on MUSA platform.

    Args:
        msg: The message to associate with the range.

    Yields:
        None
    """
    yield
