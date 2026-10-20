"""Minimal reproduction for torch.device and torch.library.infer_schema."""

from typing import Optional

import torchada  # noqa: F401
import torch


def test_infer_schema_with_torch_device_annotation():
    """Check whether infer_schema accepts a device annotation after torchada loads."""

    def custom_op(x: torch.Tensor, device: torch.device) -> None:
        pass

    schema = torch.library.infer_schema(custom_op, mutates_args=[])
    assert schema == "(Tensor x, Device device) -> ()"


def test_infer_schema_with_optional_torch_device_annotation():
    """Optional device annotations must resolve too."""

    def custom_op(x: torch.Tensor, device: Optional[torch.device] = None) -> None:
        pass

    schema = torch.library.infer_schema(custom_op, mutates_args=[])
    assert schema == "(Tensor x, Device? device=None) -> ()"
