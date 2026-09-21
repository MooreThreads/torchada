#!/usr/bin/env python3
"""
Example setup.py for building a C++/CUDA extension with torchada.

This example shows how to build CUDA extensions using standard torch imports.

Key point: Just import torchada first, then use standard torch imports.
No code changes needed!

Usage:
    python examples/extension_setup.py build_ext --inplace
"""


from setuptools import setup

# Now use standard torch imports - they work on any supported GPU!
from torch.utils.cpp_extension import CUDA_HOME, BuildExtension, CUDAExtension

# Import torchada first to apply patches
import torchada  # noqa: F401


def get_extensions():
    """Build the list of extensions."""
    extensions = []

    print(f"CUDA_HOME: {CUDA_HOME}")

    # Define source files
    # In a real project, these would be actual .cpp and .cu files
    sources = [
        # "src/my_extension.cpp",
        # "src/my_kernel.cu",
    ]

    # Skip if no source files (this is just an example)
    if not sources:
        print("No source files found, skipping extension build.")
        print("This is just an example showing the setup structure.")
        return extensions

    # Create the extension - torchada handles platform-specific details
    ext = CUDAExtension(
        name="my_extension",
        sources=sources,
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17"],  # torchada maps to correct compiler
        },
        include_dirs=[
            # Add your include directories here
            # os.path.join(os.path.dirname(__file__), "include"),
        ],
    )
    extensions.append(ext)

    return extensions


# Example of how to set up the package
if __name__ == "__main__":
    print("=" * 60)
    print("torchada Extension Build Example")
    print("=" * 60)

    extensions = get_extensions()

    if extensions:
        setup(
            name="my_cuda_extension",
            version="0.1.89",
            ext_modules=extensions,
            cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
            python_requires=">=3.8",
        )
    else:
        print("\nTo build an actual extension, add source files to the sources list.")
        print("\nExample directory structure:")
        print("  my_project/")
        print("  ├── setup.py              # Use this as a template")
        print("  ├── src/")
        print("  │   ├── my_extension.cpp  # C++ bindings")
        print("  │   └── my_kernel.cu      # CUDA kernels")
        print("  └── my_extension/")
        print("      └── __init__.py       # Python package")
