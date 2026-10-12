"""
Tests for the opt-in TORCHADA_FORCE_CUDA_AVAILABLE switch.

By default torch.cuda.is_available() keeps its original torch.cuda behavior and
returns False on MUSA, so downstream projects can tell the platforms apart.
Setting the switch before importing torchada makes that single attribute report
torch.musa.is_available() instead, for projects written for CUDA only.

The switch is read when torchada applies its patches, so every case that
depends on it has to run in a child interpreter with a controlled environment.
"""

import functools
import json
import os
import subprocess
import sys

import pytest

import torchada
from torchada._patch import _FORCE_CUDA_AVAILABLE_ENV, _force_cuda_available_enabled

ENV_VAR = _FORCE_CUDA_AVAILABLE_ENV

requires_musa = pytest.mark.skipif(
    not torchada.is_musa_platform(), reason="the MUSA mappings only exist on the MUSA platform"
)

# ``original_cuda_is_available`` is captured before torchada is imported, so it
# identifies the unpatched torch.cuda attribute even when no GPU card is present.
_CHILD_PROBE = """
import json

import torch

original_cuda_is_available = torch.cuda.is_available

import torchada  # noqa: E402  (import before the torch.cuda attribute checks below)
from torch.cuda import is_available as imported_is_available  # noqa: E402

print("PROBE_JSON=" + json.dumps({
    "cuda": bool(torch.cuda.is_available()),
    "musa": bool(torch.musa.is_available()),
    "still_original": torch.cuda.is_available is original_cuda_is_available,
    "same_as_musa": torch.cuda.is_available is torch.musa.is_available,
    "from_import_same": imported_is_available is torch.cuda.is_available,
    "device_count": torch.cuda.device_count(),
    "musa_device_count": torch.musa.device_count(),
    "platform": torchada.detect_platform().value,
    "is_musa_platform": torchada.is_musa_platform(),
}))
"""

# The classic CUDA-only device selection that never considers MUSA: with the
# switch on it must both choose the GPU and actually compute on it.
_CUDA_ONLY_SCRIPT = """
import os

os.environ["TORCHADA_FORCE_CUDA_AVAILABLE"] = "1"

import torchada
import torch

assert torch.cuda.is_available(), "torch.cuda.is_available() is still False"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Linear(32, 32).to(device)
loss = model(torch.randn(4, 32, device=device)).sum()
loss.backward()
torch.cuda.synchronize()

print("DEVICE_TYPE=" + device.type)
print("PARAM_DEVICE=" + next(model.parameters()).device.type)
print("GRAD_DEVICE=" + model.bias.grad.device.type)
print("ALLOCATED=" + str(torch.cuda.memory_allocated() > 0))
"""


def _child_env(**overrides):
    """Environment for a child interpreter with no inherited opt-in."""
    env = dict(os.environ)
    env.pop(ENV_VAR, None)
    env.update(overrides)
    return env


def _run_child(code, **env_overrides):
    """Run code in a fresh interpreter and return its raw stdout."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=_child_env(**env_overrides),
        timeout=900,
    )
    assert result.returncode == 0, f"child interpreter failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


@functools.lru_cache(maxsize=None)
def _probe_cached(env_key):
    """Run the probe once per distinct environment and keep its result."""
    for line in _run_child(_CHILD_PROBE, **dict(env_key)).splitlines():
        if line.startswith("PROBE_JSON="):
            return json.loads(line[len("PROBE_JSON=") :])
    raise AssertionError("child interpreter did not report a probe payload")


def _probe(**env_overrides):
    """Import torchada in a child interpreter and report the availability state.

    The interpreter starts once per environment, so the cases that assert
    different facts about the same mode share one run.
    """
    return dict(_probe_cached(tuple(sorted(env_overrides.items()))))


class TestSwitchParsing:
    """The switch is off unless it holds one of the accepted truthy values."""

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on", " on ", "\ttrue\n"])
    def test_truthy_values_enable(self, value, monkeypatch):
        monkeypatch.setenv(ENV_VAR, value)
        assert _force_cuda_available_enabled() is True

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2", "enabled", "musa"])
    def test_other_values_keep_the_default(self, value, monkeypatch):
        monkeypatch.setenv(ENV_VAR, value)
        assert _force_cuda_available_enabled() is False

    def test_unset_keeps_the_default(self, monkeypatch):
        monkeypatch.delenv(ENV_VAR, raising=False)
        assert _force_cuda_available_enabled() is False


@requires_musa
class TestDefaultMode:
    """Without the switch, torch.cuda.is_available() behaves exactly as before."""

    def test_is_available_keeps_the_original_function(self):
        probe = _probe()
        assert probe["still_original"] is True
        assert probe["same_as_musa"] is False
        assert probe["cuda"] is False

    def test_from_import_also_keeps_the_original_function(self):
        probe = _probe()
        assert probe["from_import_same"] is True

    def test_platform_detection_reports_musa(self):
        probe = _probe()
        assert probe["platform"] == "musa"
        assert probe["is_musa_platform"] is True


@requires_musa
class TestForcedMode:
    """With the switch on, only is_available changes."""

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_is_available_reports_musa(self, value):
        probe = _probe(**{ENV_VAR: value})
        assert probe["still_original"] is False
        assert probe["same_as_musa"] is True
        assert probe["cuda"] is probe["musa"]

    def test_from_import_is_redirected_too(self):
        probe = _probe(**{ENV_VAR: "1"})
        assert probe["from_import_same"] is True

    def test_matches_the_musa_answer_on_a_gpu(self, has_gpu):
        if not has_gpu:
            pytest.skip("No GPU device available")
        probe = _probe(**{ENV_VAR: "1"})
        assert probe["cuda"] is True
        assert probe["musa"] is True

    def test_platform_detection_still_reports_musa(self):
        probe = _probe(**{ENV_VAR: "1"})
        assert probe["platform"] == "musa"
        assert probe["is_musa_platform"] is True

    def test_other_cuda_attributes_are_unchanged(self):
        default = _probe()
        forced = _probe(**{ENV_VAR: "1"})
        assert default["device_count"] == default["musa_device_count"]
        assert forced["device_count"] == forced["musa_device_count"]
        assert forced["device_count"] == default["device_count"]
        assert forced["platform"] == default["platform"]

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", "enabled", ""])
    def test_non_truthy_values_keep_the_default(self, value):
        probe = _probe(**{ENV_VAR: value})
        assert probe["still_original"] is True
        assert probe["cuda"] is False


@requires_musa
class TestCudaOnlyProjectPattern:
    """A CUDA-only project must select the GPU and compute on it, unmodified."""

    def test_cuda_device_selection_lands_on_musa(self, has_gpu):
        if not has_gpu:
            pytest.skip("No GPU device available")
        output = _run_child(_CUDA_ONLY_SCRIPT)
        assert "DEVICE_TYPE=musa" in output
        assert "PARAM_DEVICE=musa" in output
        assert "GRAD_DEVICE=musa" in output
        assert "ALLOCATED=True" in output
