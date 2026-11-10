"""What a move of a node's check cache onto /workspace must fit in and how fast the root disk reads: a measuring script that
reads only, not part of check.

    python3 tools/verity/check/move_facts.py [FAMILY] [--gib N]      writes $OUT/move-facts.json

It prints each disk's free bytes and inodes (`df`), whether `flock`, `rsync` and `cp` are here, and the rate at which the root
disk gives up FAMILY (a directory under `~/.cache/verity-check`, by default the largest one on the root disk among lean-env,
lean-deps, lean-changed, lean-artifacts): it reads files in walk order until N GiB (3) or 120 s, and counts bytes and files a
second.  Files the page cache already holds read faster than the disk, so the rate is a ceiling.  Stdlib only.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    argv = sys.argv[1:]
    gib = float(argv[argv.index("--gib") + 1]) if "--gib" in argv else 3.0
    names = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--gib")]
    out = Path(os.environ.get("OUT") or ".")
    out.mkdir(parents=True, exist_ok=True)
    cache = Path.home() / ".cache" / "verity-check"
    root = Path("/").stat().st_dev
    df = subprocess.run(["df", "-BG", "--output=target,size,avail,pcent,itotal,iused,iavail,ipcent", "/", "/workspace"],
                        capture_output=True, text=True).stdout.splitlines()
    doc: dict = {"df": df, "tools": {t: bool(shutil.which(t)) for t in ("flock", "rsync", "cp", "ionice")}}
    print("move-facts: " + " | ".join(" ".join(x.split()) for x in df), flush=True)
    print(f"move-facts: tools {doc['tools']}", flush=True)
    cands = names or ["lean-env", "lean-deps", "lean-changed", "lean-artifacts"]
    fam = next((cache / n for n in cands if (cache / n).is_dir() and not (cache / n).is_symlink() and (cache / n).stat().st_dev == root
                and any((cache / n).iterdir())), None)
    if fam is None:
        print(f"move-facts: none of {cands} is a directory on the root disk here", flush=True)
    else:
        t0, n, b = time.monotonic(), 0, 0
        for d, _dirs, files in os.walk(fam):
            for name in files:
                try:
                    with open(os.path.join(d, name), "rb") as f:
                        while chunk := f.read(4 << 20):
                            b += len(chunk)
                    n += 1
                except OSError:
                    continue
                if b >= gib * 2**30 or time.monotonic() - t0 > 120:
                    break
            else:
                continue
            break
        s = time.monotonic() - t0
        doc["read"] = {"family": fam.name, "bytes": b, "files": n, "seconds": round(s, 1), "mib_s": round(b / 2**20 / s, 1), "files_s": round(n / s)}
        print(f"move-facts: read {json.dumps(doc['read'])}", flush=True)
    (out / "move-facts.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
