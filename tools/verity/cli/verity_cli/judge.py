"""`verity lean judge [PKG ...]`: the Lean check's source judge (`sources_judge` of `tools/verity/lean/check.py`), which
reads each package's sources and lock and builds nothing, so what its audit would refuse before any build shows on a
checkout in seconds."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .deployed import lean_check


def judged(pkg: Path, check) -> list[str]:
    """``pkg``'s failures as lines: each one's code, entry and text, then its fix where the judge knows it."""
    try:
        policy = check.load(pkg)
    except (OSError, ValueError) as e:
        return [f"lock: {pkg / check.POLICY} can't be read: {type(e).__name__}: {e}"]
    if "roots" not in policy:
        return [f"lock: {pkg / check.POLICY} names no `roots`, so nothing is judged: add them"]
    return [f"{f.code} {f.entry or '-'}: {f}" + (f"; fix: {f.fix}" if f.fix else "")
            for f in check.sources_judge(pkg, policy, check.ROOT)]


def main(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="verity lean judge", description="The Lean check's source judge on each PKG, by "
                                 "default every package with a lean-audit.json: per package its failure count and each "
                                 "failure's code, entry and text, with its fix where the judge knows it. No build, no "
                                 "Lean. Exits 1 on any failure.")
    ap.add_argument("package", nargs="*", type=Path, help="a Lake package directory with a lean-audit.json")
    a = ap.parse_args(args)
    check = lean_check()
    pkgs = [p.resolve() for p in a.package] or [p for p in check.packages(check.ROOT) if (p / check.POLICY).is_file()]
    failed = 0
    for pkg in pkgs:
        lines = judged(pkg, check)
        failed += len(lines)
        print(f"{os.path.relpath(pkg, check.ROOT)}: {len(lines)} failure(s)")
        for line in lines:
            print(f"  {line}")
    return 1 if failed else 0
