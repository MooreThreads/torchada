"""CUDA->MUSA source-porting rules for torch_musa SimplePorting (via torchada.utils.cpp_extension)."""

from ._mappings import MAPPING_RULE as _MAPPING_RULE  # noqa: F401

# Extension file suffix mappings: convert .cu/.cuh to .mu/.muh so torch_musa's
# musa_compile rule (which only adds -x musa for .mu/.muh) treats them correctly.
EXT_REPLACED_MAPPING = {
    'cuh': 'muh',
    'cu': 'mu',
    'cc': 'cc',
    'cpp': 'cpp',
    'cxx': 'cxx',
}
