#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Ask mcc whether the MUSA names torchada's compat headers alias actually exist.

The header layer in ``csrc/cuda_compat`` states what the porting tables assume:
that ``musaMalloc``, ``mublasGemmEx``, ``__mt_fp8_e4m3`` and ~190 peers are
declared by the MUSA SDK on this box. Until it is measured, an assumption that
breaks shows up as an undeclared identifier inside someone else's JIT build, which
is the failure mode this whole layer exists to remove.

The probe needs the MUSA SDK and ``mcc`` but **no device**: every candidate is
compiled with ``-fsyntax-only``, so it runs on a build host or in a container
without leasing a GPU. Each name is tried three ways, because a rename can produce
any of the three and only one compiles: as a type (``using probe_t = NAME;``), as a
function or object (``(void)&NAME;``), and as a constant (``auto probe_use = (NAME);``,
which is how an enum value such as ``MUSA_ERROR_INVALID_VALUE`` is reachable).
"Exists" means one of the three compiled, and which one tells you what kind of name
it is.

    # every alias the tables produce, plus known unsupported names
    python3 scripts/probe_cuda_compat_symbols.py

    # only the CUDA names the tables have no rule for (compat.UNMAPPED_JIT_NAMES),
    # spelled mechanically. This exits nonzero while any known unsupported name
    # remains, which is the expected state for TensorMap and TMA today.
    python3 scripts/probe_cuda_compat_symbols.py --unmapped

    # machine-readable, for a ticket or a PR body
    python3 scripts/probe_cuda_compat_symbols.py --json /tmp/probe.json

Exit code is 0 when every probed name exists, 1 when any is missing (a missing
name is either a table rule to fix or a fallback to implement, never noise), 2 when
``mcc`` itself could not be found.
"""

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile

#: Which MUSA header should declare a name, per porting-table domain. A name that
#: exists but whose header we do not include looks missing, so this is part of the
#: measurement, not a convenience. ``data_types`` is absent because one domain spans
#: three headers; :func:`_header_for` splits it by name.
DOMAIN_HEADERS = {
    "cuda_runtime": "musa_runtime.h",
    "cuda_driver": "musa.h",
    "cublas": "mublas.h",
    "curand": "musa_curand.h",
    "cudnn": "musa_cudnn.h",
    "nccl": "mccl.h",
    "cufft": "musa_cufft.h",
    "cusparse": "musa_sparse.h",
    "cusolver": "musa_solver.h",
}


def _header_for(domain, name):
    """The MUSA header that should declare ``name`` in ``domain``, or ``None``."""
    if domain in DOMAIN_HEADERS:
        if domain == "cublas" and "lt" in name.lower():
            return "mublasLt.h"
        return DOMAIN_HEADERS[domain]
    if domain == "data_types":
        if "fp4" in name or name == "__MT_E2M1":
            return "musa_fp4.h"
        if (
            "fp8" in name
            or "_to_fp8" in name
            or name.startswith(("__NV_", "__MT_", "CUDA_R_", "MUSA_R_"))
        ):
            return "musa_fp8.h"
        if "bfloat" in name:
            return "musa_bf16.h"
        return "musa_fp16.h"
    return None


_TYPE_TU = """#include <{header}>

using probe_t = {name};
probe_t* probe_use = nullptr;
"""

_VALUE_TU = """#include <{header}>

void* probe_use = (void*)&{name};
"""

_EXPR_TU = """#include <{header}>

auto probe_use = ({name});
"""


def _compat_module():
    """Import ``torchada.cuda_compat`` without importing the torchada package."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "src", "torchada", "cuda_compat.py")
    if not os.path.exists(path):
        sys.exit("torchada/cuda_compat.py not found beside this script; run it from a checkout")
    spec = importlib.util.spec_from_file_location("_probe_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mapping_module(domain):
    """Load ``_mappings/<domain>.py``, which is a plain dict and imports nothing."""
    import importlib.util

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "src", "torchada", "_mappings", domain + ".py")
    if not os.path.exists(path):
        import importlib

        return importlib.import_module("torchada._mappings." + domain)
    name = "_probe_mapping_" + domain
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidates(only_unmapped):
    """``[(cuda_name, musa_name, header)]`` for everything worth asking about."""
    compat = _compat_module()

    def unmapped_row(name):
        if name.startswith("cuda"):
            return name, "musa" + name[len("cuda") :], "musa_runtime.h"
        if name.startswith("cu"):
            return name, "mu" + name[len("cu") :], "musa.h"
        if name.startswith("CU"):
            return name, "MU" + name[len("CU") :], "musa.h"
        return name, name, None

    rows = []
    if only_unmapped:
        for name in compat.UNMAPPED_JIT_NAMES:
            rows.append(unmapped_row(name))
        return rows

    torch_owned_cublas_names = {"getCurrentMUSABlasHandle"}
    for domain in compat.DOMAIN_HEADERS:
        mapping = _mapping_module(domain).MAPPING
        subword = compat.SUBWORD_RULES.get(domain, frozenset())
        for cuda_name, musa_name in mapping.items():
            if cuda_name in subword or not musa_name.isidentifier():
                continue
            if domain == "cublas" and musa_name in torch_owned_cublas_names:
                continue
            header = _header_for(domain, musa_name)
            if header is None:
                print(
                    "skipping %s/%s: no MUSA header is known to declare it" % (domain, musa_name),
                    file=sys.stderr,
                )
                continue
            rows.append((cuda_name, musa_name, header))
    for name in compat.UNMAPPED_JIT_NAMES:
        rows.append(unmapped_row(name))
    return rows


_FORMS = {"type": _TYPE_TU, "value": _VALUE_TU, "constant": _EXPR_TU}


def _compile(mcc, mcc_args, header, name, form, tmpdir):
    source = os.path.join(tmpdir, "probe_%s_%s.mu" % (form, abs(hash((header, name)))))
    with open(source, "w") as handle:
        handle.write(_FORMS[form].format(header=header, name=name))
    result = subprocess.run(
        [mcc, *mcc_args, "-x", "musa", "-fsyntax-only", "-std=c++20", source],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0, result.stderr


def _probe_one(mcc, mcc_args, cuda_name, musa_name, header):
    with tempfile.TemporaryDirectory() as tmpdir:
        errors = {}
        for form in ("type", "value", "constant"):
            ok, stderr = _compile(mcc, mcc_args, header, musa_name, form, tmpdir)
            if ok:
                return {"cuda": cuda_name, "musa": musa_name, "header": header, "kind": form}
            errors[form] = stderr

        value_error = errors["value"]
        if "address of overloaded function" in value_error or (
            "reference to overloaded function" in value_error
        ):
            return {
                "cuda": cuda_name,
                "musa": musa_name,
                "header": header,
                "kind": "overload",
            }

        first_error = errors["type"]
        detail = next(
            (line.strip() for line in first_error.splitlines() if "error" in line), "not declared"
        )
        return {
            "cuda": cuda_name,
            "musa": musa_name,
            "header": header,
            "kind": "missing",
            "detail": detail,
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mcc", default=None, help="mcc executable (default: search PATH)")
    parser.add_argument(
        "--mcc-arg",
        action="append",
        default=[],
        help="argument passed to every mcc invocation (repeatable)",
    )
    parser.add_argument("--jobs", type=int, default=8, help="parallel compiles (default: 8)")
    parser.add_argument(
        "--unmapped",
        action="store_true",
        help="probe only the names the porting tables have no rule for",
    )
    parser.add_argument("--json", default=None, help="write the full result to this path")
    args = parser.parse_args(argv)

    mcc = args.mcc or shutil.which("mcc")
    if not mcc or not os.access(mcc, os.X_OK):
        print(
            "mcc not found or not executable (%s); source the MUSA SDK environment "
            "or pass --mcc" % (mcc or "not on PATH"),
            file=sys.stderr,
        )
        return 2

    rows = _candidates(args.unmapped)
    print("probing %d names with %s" % (len(rows), mcc), file=sys.stderr)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(lambda row: _probe_one(mcc, args.mcc_arg, *row), rows))

    missing = [r for r in results if r["kind"] == "missing"]
    width = max(len(r["cuda"]) for r in results) + 2
    target = max(len(r["musa"]) for r in results) + 2
    for row in sorted(results, key=lambda r: (r["header"], r["cuda"])):
        print(
            "%-*s %-*s %-9s %s"
            % (width, row["cuda"], target, row["musa"], row["kind"], row.get("detail", ""))
        )

    counts = {}
    for row in results:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    print(
        "\n%d type, %d value, %d overload, %d constant, %d missing"
        % (
            counts.get("type", 0),
            counts.get("value", 0),
            counts.get("overload", 0),
            counts.get("constant", 0),
            len(missing),
        )
    )
    for header in sorted({r["header"] for r in missing}):
        names = [r["musa"] for r in missing if r["header"] == header]
        print("missing from <%s>: %s" % (header, ", ".join(names)))
    if missing:
        print(
            "\nA missing name is a decision, not noise: either the table rule is wrong, "
            "or the alias needs a behavioural fallback (a rename cannot supply one)."
        )

    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=2)
        print("wrote %s" % args.json, file=sys.stderr)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
