#!/usr/bin/env python3
"""VLLM-289: run one command at each of several commits, in the job's own clone of the source root (scratch, lands nowhere).

    at.py LABEL=COMMIT... -- CMD

For a `research run --on auto --cwd source` job whose shipped commit is a carrier with the commits as parents. For each
LABEL=COMMIT it checks COMMIT out, detached, in a clone made as `--cwd clone` makes one (`research.clone.make`: a checkout
with the merge drivers, which `suites.py` asks for), and runs `bash -lc CMD` there with $OUT_AT=$OUT/LABEL. Each
command's exit code goes to $OUT/at.json; the job's is the largest.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SOURCE = os.getcwd()
OUT = os.environ.get("OUT") or os.path.join(os.environ.get("RESEARCH_RUN_DIR", SOURCE), "out")


def main() -> int:
    cut = sys.argv.index("--")
    jobs, cmd = [a.split("=", 1) for a in sys.argv[1:cut]], " ".join(sys.argv[cut + 1:])
    sys.path.insert(0, os.path.join(SOURCE, "tools", "research", "src"))
    from research import clone as research_clone

    scratch = tempfile.mkdtemp(prefix="restack289-at-")
    tree = os.path.join(scratch, "tree")
    done: list[dict] = []
    try:
        research_clone.make(Path(SOURCE), Path(tree))
        for label, commit in jobs:
            t0 = time.time()
            subprocess.run(["git", "checkout", "--quiet", "--detach", commit], cwd=tree, check=True)
            subprocess.run(["git", "clean", "-qfdx", "-e", ".venv"], cwd=tree, check=True)
            out = os.path.join(OUT, label)
            os.makedirs(out, exist_ok=True)
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True).stdout.strip()
            print(time.strftime("%H:%M:%SZ", time.gmtime()), f"== {label} at {head}: {cmd}", flush=True)
            rc = subprocess.run(["bash", "-lc", cmd], cwd=tree, env={**os.environ, "OUT_AT": out}).returncode
            done.append({"label": label, "commit": head, "rc": rc, "minutes": round((time.time() - t0) / 60, 2)})
            print(time.strftime("%H:%M:%SZ", time.gmtime()), f"== {label}: exit {rc} in {done[-1]['minutes']} min", flush=True)
            with open(os.path.join(OUT, "at.json"), "w") as f:
                json.dump({"command": cmd, "results": done}, f, indent=1)
        return max((d["rc"] for d in done), default=0)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
