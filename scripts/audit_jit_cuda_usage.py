#!/usr/bin/env python3
"""Inventory CUDA runtime/driver names used by compiler-driven JIT sources."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from torchada._mapping import _MAPPING_RULE
from torchada.cuda_compat import UNMAPPED_JIT_NAMES

SOURCE_SUFFIXES = {".c", ".cc", ".cpp", ".cu", ".cuh", ".cxx", ".h", ".hpp"}
TOKEN_RE = re.compile(
    r"\b(?:cuda[A-Z][A-Za-z0-9_]*|cu[A-Z][A-Za-z0-9_]*|"
    r"CU(?:result|device|context|stream|event|module|function|mem|pointer|"
    r"tensorMap|array|tex|surface|graph|occupancy|launch|exec|library|kernel|"
    r"multicast)[A-Za-z0-9_]*)\b"
)
IGNORED_PREFIXES = (
    "CUB_",
    "CUDA_",
    "CUDART_",
    "CUDNN_",
    "CUTLASS_",
    "CUTE_",
    "CUPTI",
    "CUVID",
)
IGNORED_EXACT = {"CUDA", "CUR"}


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//.*", "", text)


def _status(name: str) -> str:
    if name in _MAPPING_RULE:
        return "mapped"
    if name in UNMAPPED_JIT_NAMES:
        return "known-unmapped"
    return "unclassified"


def scan(paths: list[Path]) -> dict:
    locations: dict[str, list[dict[str, int | str]]] = defaultdict(list)
    files = 0
    for root in paths:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            files += 1
            text = _strip_comments(path.read_text(errors="ignore"))
            for line_number, line in enumerate(text.splitlines(), 1):
                for match in TOKEN_RE.finditer(line):
                    name = match.group(0)
                    if name in IGNORED_EXACT or name.startswith(IGNORED_PREFIXES):
                        continue
                    locations[name].append(
                        {
                            "path": str(path.relative_to(root)),
                            "root": str(root),
                            "line": line_number,
                        }
                    )
    counts = Counter(_status(name) for name in locations)
    return {
        "roots": [str(path) for path in paths],
        "files": files,
        "symbols": len(locations),
        "counts": dict(sorted(counts.items())),
        "findings": {
            name: {"status": _status(name), "locations": refs}
            for name, refs in sorted(locations.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = scan(args.roots)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "files=%d symbols=%d mapped=%d known-unmapped=%d unclassified=%d"
        % (
            report["files"],
            report["symbols"],
            report["counts"].get("mapped", 0),
            report["counts"].get("known-unmapped", 0),
            report["counts"].get("unclassified", 0),
        )
    )
    for name, finding in report["findings"].items():
        if finding["status"] != "mapped":
            first = finding["locations"][0]
            print(f"{finding['status']:16} {name:55} {first['path']}:{first['line']}")


if __name__ == "__main__":
    main()
