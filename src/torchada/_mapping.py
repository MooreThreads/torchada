"""CUDA->MUSA source-porting rules for torch_musa SimplePorting (via torchada.utils.cpp_extension)."""

from ._mappings import MAPPING_RULE as _MAPPING_RULE  # noqa: F401

# Keep source extensions unchanged during porting. Sources are ported in place
# (no <dir>_musa mirror), and .cu/.cuh compile as MUSA via the patched
# _is_musa_file (which selects the -x musa rule), so renaming to .mu/.muh is
# unnecessary -- and avoiding it removes any need for include/source-path
# rewriting.
EXT_REPLACED_MAPPING = {
    'cuh': 'cuh',
    'cu': 'cu',
    'cc': 'cc',
    'cpp': 'cpp',
    'cxx': 'cxx',
}
