"""Regression tests for safe, deterministic in-place CUDA source porting."""

import os

import pytest
from setuptools import Distribution, Extension

import torchada

pytestmark = pytest.mark.skipif(
    not torchada.is_musa_platform(),
    reason="In-place SimplePorting integration requires torch_musa",
)

TOKEN = "TORCHADA_CUDA_TOKEN"
PORTED_TOKEN = "TORCHADA_MUSA_TOKEN"
MAPPING = {TOKEN: PORTED_TOKEN}


def _build_extension_command():
    from torchada.utils.cpp_extension import BuildExtension

    return BuildExtension(Distribution())


def test_existing_mirror_sibling_is_preserved(tmp_path):
    source_dir = tmp_path / "csrc"
    mirror_dir = tmp_path / "csrc_musa"
    source_dir.mkdir()
    mirror_dir.mkdir()
    source = source_dir / "kernel.cu"
    sentinel = mirror_dir / "DO_NOT_DELETE.txt"
    source.write_text(f"{TOKEN}\n", encoding="utf-8")
    sentinel.write_text("user-managed content\n", encoding="utf-8")

    command = _build_extension_command()
    command._port_directory(str(source_dir), MAPPING)

    assert sentinel.read_text(encoding="utf-8") == "user-managed content\n"
    assert source.read_text(encoding="utf-8") == f"{PORTED_TOKEN}\n"


def test_portable_symlink_fails_without_modifying_target(tmp_path):
    source_dir = tmp_path / "csrc"
    source_dir.mkdir()
    external = tmp_path / "user_owned.h"
    external.write_text(f"{TOKEN}\n", encoding="utf-8")
    link = source_dir / "linked.h"
    link.symlink_to(external)

    command = _build_extension_command()
    with pytest.raises(RuntimeError, match="symlinked CUDA/C\\+\\+ source"):
        command._port_directory(str(source_dir), MAPPING)

    assert link.is_symlink()
    assert external.read_text(encoding="utf-8") == f"{TOKEN}\n"


def test_dotless_files_do_not_create_or_overwrite_hidden_siblings(tmp_path):
    source_dir = tmp_path / "csrc"
    source_dir.mkdir()
    (source_dir / "kernel.cu").write_text(f"{TOKEN}\n", encoding="utf-8")
    makefile = source_dir / "Makefile"
    hidden_makefile = source_dir / ".Makefile"
    makefile.write_text("public makefile\n", encoding="utf-8")
    hidden_makefile.write_text("private sentinel\n", encoding="utf-8")

    command = _build_extension_command()
    command._port_directory(str(source_dir), MAPPING)

    assert makefile.read_text(encoding="utf-8") == "public makefile\n"
    assert hidden_makefile.read_text(encoding="utf-8") == "private sentinel\n"


@pytest.mark.parametrize("suffix", [".ipp", ".tpp", ".txx", ".ixx", ".cppm"])
def test_common_cpp_fragments_are_ported(tmp_path, suffix):
    source_dir = tmp_path / "csrc"
    source_dir.mkdir()
    fragment = source_dir / f"fragment{suffix}"
    fragment.write_text(f"{TOKEN}\n", encoding="utf-8")

    command = _build_extension_command()
    command._port_directory(str(source_dir), MAPPING)

    assert fragment.read_text(encoding="utf-8") == f"{PORTED_TOKEN}\n"


def test_overlapping_roots_are_ported_once(tmp_path, monkeypatch):
    from torchada.utils.cpp_extension import BuildExtension

    source_dir = tmp_path / "src"
    include_dir = source_dir / "include"
    include_dir.mkdir(parents=True)
    source = source_dir / "kernel.cu"
    header = include_dir / "header.h"
    source.write_text(f"{TOKEN}\n", encoding="utf-8")
    header.write_text(f"{TOKEN}\n", encoding="utf-8")

    class CustomBuildExtension(BuildExtension):
        def get_mapping_rule(self):
            return {TOKEN: TOKEN + "X"}

    command = CustomBuildExtension(Distribution())
    command.extensions = [
        Extension("test_overlap", sources=[str(source)], include_dirs=[str(include_dir)])
    ]
    monkeypatch.setattr(BuildExtension.__mro__[1], "run", lambda self: None)

    command.run()

    assert source.read_text(encoding="utf-8") == f"{TOKEN}X\n"
    assert header.read_text(encoding="utf-8") == f"{TOKEN}X\n"
    assert command._ported_dirs == {os.path.realpath(source_dir)}


def test_vendor_mapping_defines_are_kept_verbatim(tmp_path):
    """A header that already maps CUDA onto MUSA must survive porting.

    Substituting the defined name too would leave `#define musaX musaX`, which
    shadows the runtime's real value with a self-reference.
    """
    source_dir = tmp_path / "gpu_vendor"
    source_dir.mkdir()
    header = source_dir / "musa.h"
    header.write_text(
        "#define cudaEventDisableTiming musaEventDisableTiming\n"
        "#define CU_MEMORYTYPE_DEVICE MU_MEMORYTYPE_DEVICE\n"
        "#define cudaStreamWaitEvent(s) musaStreamWaitEvent(s)\n"
        "#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED \\\n"
        "    MU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED\n",
        encoding="utf-8",
    )

    command = _build_extension_command()
    command._port_directory(
        str(source_dir),
        {
            "cudaEventDisableTiming": "musaEventDisableTiming",
            "cudaStreamWaitEvent": "musaStreamWaitEvent",
            "CU_MEMORYTYPE_DEVICE": "MU_MEMORYTYPE_DEVICE",
            "CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED": (
                "MU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED"
            ),
        },
    )

    ported = header.read_text(encoding="utf-8")
    assert "#define cudaEventDisableTiming musaEventDisableTiming\n" in ported
    assert "#define CU_MEMORYTYPE_DEVICE MU_MEMORYTYPE_DEVICE\n" in ported
    assert "#define cudaStreamWaitEvent(s) musaStreamWaitEvent(s)\n" in ported
    assert (
        "#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED \\\n"
        "    MU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED\n"
    ) in ported
    assert "#define musaEventDisableTiming musaEventDisableTiming" not in ported


def test_annotated_and_parenthesized_mappings_are_kept_verbatim(tmp_path):
    """Mappings still collapse when annotated or wrapped in redundant parens.

    `#define X (X)` and `#define X X /* note */` shadow the real definition
    exactly as `#define X X` does, and vendor headers routinely write both.
    """
    source_dir = tmp_path / "gpu_vendor"
    source_dir.mkdir()
    header = source_dir / "musa.h"
    header.write_text(
        "#define cudaEventDisableTiming musaEventDisableTiming /**< no timing */\n"
        "#define cudaHostRegisterIoMemory musaHostRegisterIoMemory // mapped I/O\n"
        "#define cudaSuccess (musaSuccess)\n"
        "#define CU_MEMORYTYPE_DEVICE \\\n"
        "    MU_MEMORYTYPE_DEVICE /* device memory */\n",
        encoding="utf-8",
    )

    command = _build_extension_command()
    command._port_directory(
        str(source_dir),
        {
            "cudaEventDisableTiming": "musaEventDisableTiming",
            "cudaHostRegisterIoMemory": "musaHostRegisterIoMemory",
            "cudaSuccess": "musaSuccess",
            "CU_MEMORYTYPE_DEVICE": "MU_MEMORYTYPE_DEVICE",
        },
    )

    ported = header.read_text(encoding="utf-8")
    assert "#define cudaEventDisableTiming musaEventDisableTiming /**< no timing */\n" in ported
    assert "#define cudaHostRegisterIoMemory musaHostRegisterIoMemory // mapped I/O\n" in ported
    assert "#define cudaSuccess (musaSuccess)\n" in ported
    assert (
        "#define CU_MEMORYTYPE_DEVICE \\\n    MU_MEMORYTYPE_DEVICE /* device memory */\n"
    ) in ported


def test_aliasing_defines_still_port(tmp_path):
    """Only the degenerate mapping is skipped; a real alias must still port."""
    source_dir = tmp_path / "csrc"
    source_dir.mkdir()
    header = source_dir / "helpers.h"
    header.write_text(
        f"#define {TOKEN}_ALIAS {TOKEN}_IMPL\n"
        f"#define {TOKEN}_ALIAS2 {TOKEN}_IMPL // aliased\n"
        f"#define {TOKEN}_PAREN ({TOKEN}_IMPL)\n"
        f"#define {TOKEN}_FLAG 0x02\n"
        f"#define {TOKEN}_FLAG2 0x02 /**< annotated */\n"
        f"void call() {{ {TOKEN}(); }}\n",
        encoding="utf-8",
    )

    command = _build_extension_command()
    command._port_directory(str(source_dir), MAPPING)

    assert header.read_text(encoding="utf-8") == (
        f"#define {PORTED_TOKEN}_ALIAS {PORTED_TOKEN}_IMPL\n"
        f"#define {PORTED_TOKEN}_ALIAS2 {PORTED_TOKEN}_IMPL // aliased\n"
        f"#define {PORTED_TOKEN}_PAREN ({PORTED_TOKEN}_IMPL)\n"
        f"#define {PORTED_TOKEN}_FLAG 0x02\n"
        f"#define {PORTED_TOKEN}_FLAG2 0x02 /**< annotated */\n"
        f"void call() {{ {PORTED_TOKEN}(); }}\n"
    )


def test_source_parent_is_not_filtered_as_system_include(tmp_path, monkeypatch):
    from torchada.utils.cpp_extension import BuildExtension

    source_dir = tmp_path / "opt-like-project"
    source_dir.mkdir()
    source = source_dir / "kernel.cu"
    source.write_text(f"{TOKEN}\n", encoding="utf-8")

    class CustomBuildExtension(BuildExtension):
        def get_mapping_rule(self):
            return MAPPING

    command = CustomBuildExtension(Distribution())
    command.extensions = [
        Extension("test_system_source", sources=[str(source)], include_dirs=[str(source_dir)])
    ]
    monkeypatch.setattr(BuildExtension, "_is_system_include_dir", staticmethod(lambda _path: True))
    monkeypatch.setattr(BuildExtension.__mro__[1], "run", lambda self: None)

    command.run()

    assert source.read_text(encoding="utf-8") == f"{PORTED_TOKEN}\n"
    assert command._ported_dirs == {os.path.realpath(source_dir)}
