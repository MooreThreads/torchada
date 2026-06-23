import importlib.metadata
import logging
import os
import re
import shutil
import sys
from typing import Optional

logger = logging.getLogger(__name__)

_TRITON_DIR_PREFIX = "triton_"


def _is_fused_moe_tuner() -> bool:
    argv0 = os.path.realpath(sys.argv[0]) if sys.argv else ""
    expected_suffix = os.path.join("autotune", "fused_moe", "tune_moe.py")
    return argv0.endswith(expected_suffix)


def _installed_triton_version() -> Optional[str]:
    """Installed triton release (e.g. "3.2.0") without the (heavy) `import triton`.

    Only the leading release segment is kept so PEP 440 variants such as
    "3.2.0.post1", "3.2.0rc1", or "3.2.0+gitabcdef" all match the
    "triton_3_2_0" config directory instead of falling back to "newest".
    """
    try:
        version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        return None
    match = re.match(r"\d+(?:\.\d+)*", version)
    return match.group(0) if match else None


def _pick_triton_config_dir(configs_root: str) -> Optional[str]:
    """Pick `configs/triton_<ver>/` for the installed triton, else the newest."""
    if not os.path.isdir(configs_root):
        return None
    subdirs = [
        d
        for d in os.listdir(configs_root)
        if d.startswith(_TRITON_DIR_PREFIX) and os.path.isdir(os.path.join(configs_root, d))
    ]
    if not subdirs:
        return None

    version = _installed_triton_version()
    if version is not None:
        exact = _TRITON_DIR_PREFIX + version.replace(".", "_")
        if exact in subdirs:
            return os.path.join(configs_root, exact)

    def _version_key(name: str):
        parts = name[len(_TRITON_DIR_PREFIX) :].split("_")
        return tuple(int(p) if p.isdigit() else -1 for p in parts)

    return os.path.join(configs_root, max(subdirs, key=_version_key))


def _alias_space_free_names(config_dir: str) -> None:
    """Alias `...block_shape=[128, 128].json` to `...block_shape=[128,128].json`.

    vLLM builds the lookup filename with spaces stripped
    (`get_config_file_name` does `.replace(" ", "")`), while the tuning
    scripts save filenames with `str(list)` spaces (which is what SGLang
    looks up). Keep both names: symlink where possible, copy otherwise.
    Idempotent and best-effort — on failure vLLM falls back to its default
    config, exactly the pre-alias behavior.
    """
    try:
        names = os.listdir(config_dir)
    except OSError as exc:
        logger.debug("Cannot list MoE config dir %s: %s", config_dir, exc)
        return
    for name in names:
        if " " not in name or not name.endswith(".json"):
            continue
        alias = os.path.join(config_dir, name.replace(" ", ""))
        if os.path.lexists(alias):
            continue
        try:
            os.symlink(name, alias)
        except FileExistsError:
            continue  # another process won the race
        except OSError:
            if os.path.lexists(alias):
                continue
            # Copy via a temp name + atomic replace so an interrupted copy
            # can never leave a truncated JSON behind at the alias path.
            tmp = f"{alias}.tmp.{os.getpid()}"
            try:
                shutil.copy2(os.path.join(config_dir, name), tmp)
                os.replace(tmp, alias)
            except OSError as exc:
                logger.debug("Cannot alias MoE config %s: %s", name, exc)
                try:
                    os.unlink(tmp)
                except OSError:
                    pass


def _vllm_tuned_config_dir(base: str) -> Optional[str]:
    """Resolve the directory vLLM's flat VLLM_TUNED_CONFIG_FOLDER lookup needs.

    vLLM (`get_moe_configs`) joins the env folder directly with the config
    filename — it does not walk `configs/triton_<ver>/` subdirectories the
    way SGLang does — so the env must point inside the version subdir, with
    space-free filename aliases in place.
    """
    config_dir = _pick_triton_config_dir(os.path.join(base, "configs"))
    if config_dir is None:
        return None
    _alias_space_free_names(config_dir)
    return config_dir


def set_default_moe_config_dir():
    default_path = os.path.dirname(os.path.realpath(__file__))

    if "SGLANG_MOE_CONFIG_DIR" not in os.environ:
        os.environ["SGLANG_MOE_CONFIG_DIR"] = default_path

    if "VLLM_TUNED_CONFIG_FOLDER" not in os.environ:
        vllm_dir = _vllm_tuned_config_dir(default_path)
        os.environ["VLLM_TUNED_CONFIG_FOLDER"] = vllm_dir or default_path
