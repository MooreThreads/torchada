"""
torchada.cuda.amp - Automatic Mixed Precision support.

This module provides AMP functionality compatible with both CUDA and MUSA platforms.
"""

from .._platform import Platform, detect_platform


def _get_amp_backend():
    """Get the appropriate AMP backend module."""
    platform = detect_platform()

    if platform == Platform.MUSA:
        import torch
        import torch_musa  # noqa: F401 - registers torch.musa

        if hasattr(torch.musa, "amp"):
            return torch.musa.amp
        # Fall back to torch.cuda.amp for API compatibility.
        return torch.cuda.amp
    else:
        import torch

        return torch.cuda.amp


# Re-export common AMP classes and functions.
def autocast(enabled=True, dtype=None, cache_enabled=True):
    """
    Context manager for automatic mixed precision.

    Args:
        enabled: Whether autocasting is enabled
        dtype: The dtype for autocasting
        cache_enabled: Whether to cache autocasted weights
    """
    backend = _get_amp_backend()
    if hasattr(backend, "autocast"):
        return backend.autocast(enabled=enabled, dtype=dtype, cache_enabled=cache_enabled)
    else:
        # Use torch.autocast with the detected device type.
        import torch

        from .._platform import get_device_name

        device_type = get_device_name()
        return torch.autocast(
            device_type=device_type,
            enabled=enabled,
            dtype=dtype,
            cache_enabled=cache_enabled,
        )


class GradScaler:
    """
    Gradient scaler for mixed precision training.

    Wraps the appropriate backend's GradScaler.
    """

    def __init__(
        self,
        init_scale=65536.0,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        enabled=True,
    ):
        backend = _get_amp_backend()
        self._scaler = backend.GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )

    def __getattr__(self, name):
        """Delegate backend-specific GradScaler APIs not wrapped explicitly."""
        scaler = self.__dict__.get("_scaler")
        if scaler is None:
            raise AttributeError(name)
        return getattr(scaler, name)

    def scale(self, outputs):
        """Scale the outputs."""
        return self._scaler.scale(outputs)

    def unscale_(self, optimizer):
        """Unscale the gradients."""
        return self._scaler.unscale_(optimizer)

    def step(self, optimizer, *args, **kwargs):
        """Step the optimizer."""
        return self._scaler.step(optimizer, *args, **kwargs)

    def update(self, new_scale=None):
        """Update the scale."""
        return self._scaler.update(new_scale)

    def get_scale(self):
        """Get the current scale."""
        return self._scaler.get_scale()

    def get_growth_factor(self):
        """Get the growth factor."""
        return self._scaler.get_growth_factor()

    def set_growth_factor(self, new_factor):
        """Set the growth factor."""
        return self._scaler.set_growth_factor(new_factor)

    def get_backoff_factor(self):
        """Get the backoff factor."""
        return self._scaler.get_backoff_factor()

    def set_backoff_factor(self, new_factor):
        """Set the backoff factor."""
        return self._scaler.set_backoff_factor(new_factor)

    def get_growth_interval(self):
        """Get the growth interval."""
        return self._scaler.get_growth_interval()

    def set_growth_interval(self, new_interval):
        """Set the growth interval."""
        return self._scaler.set_growth_interval(new_interval)

    def is_enabled(self):
        """Check if the scaler is enabled."""
        return self._scaler.is_enabled()

    def state_dict(self):
        """Return the state dict."""
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        """Load the state dict."""
        return self._scaler.load_state_dict(state_dict)
