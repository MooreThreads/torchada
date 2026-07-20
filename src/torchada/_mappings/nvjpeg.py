"""nvJPEG to MTJPEG porting rules."""

# These are the canonical lower- and upper-case spellings used by the C API;
# mixed-case variants are intentionally not inferred. cpp_extension protects
# project-local include paths before applying these prefix rules.
MAPPING = {
    'NVJPEG': 'MTJPEG',
    'nvjpeg': 'mtjpeg',
}
