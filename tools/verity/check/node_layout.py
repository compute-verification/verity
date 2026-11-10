"""Where a node keeps check's cache and how large each part is: a measuring script that reads only, not part of check.

    python3 tools/verity/check/node_layout.py [--du-s SECONDS]      writes $OUT/node-layout.json

It prints how full the root disk and /workspace are, whether the check cache (`~/.cache/verity-check`) is a link, and for every
entry directly under it: whether it is a link and to where, which disk it is on, and its size and file count (`du`, one
filesystem, all entries at once, each given SECONDS at most, 600 by default).  Then the same for the suites' caches beside it and
for what `/workspace/research/cache` holds, and the node's check slots, pools and Lean slots: names, cores and whether each is held
(never a holder's command).  Stdlib only; nothing is written outside $OUT.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCKS = Path("/workspace/research/locks")


def disk(p: Path, devs: dict[int, str]) -> str:
    try:
        return devs.get(p.stat().st_dev, f"another filesystem ({p.stat().st_dev})")
    except OSError:
        return "missing"


def du(p: Path, limit: float) -> dict:
    """``p``'s bytes and file count on its own filesystem, or how far `du` got in ``limit`` seconds."""
    t0 = time.monotonic()
    out: dict = {}
    for key, flag in (("bytes", "-B1"), ("files", "--inodes")):
        left = limit - (time.monotonic() - t0)
        if left <= 0:
            out[key] = None
            continue
        try:
            r = subprocess.run(["nice", "-n", "19", "du", "-s", flag, "--one-file-system", "--", str(p)], capture_output=True,
                               text=True, timeout=left)
            out[key] = int(r.stdout.split()[0]) if r.stdout.split() else None
        except (subprocess.TimeoutExpired, ValueError):
            out[key] = None
    out["seconds"] = round(time.monotonic() - t0, 1)
    return out


def entry(p: Path, devs: dict[int, str]) -> dict:
    d: dict = {"name": p.name, "link": p.is_symlink(), "dir": p.is_dir(), "disk": disk(p, devs)}
    if p.is_symlink():
        d["target"] = os.readlink(p)
    return d


def listing(base: Path, devs: dict[int, str], limit: float) -> list[dict]:
    """Every entry directly under ``base`` with its size, measured where it resolves."""
    if not base.is_dir():
        return []
    ps = sorted(base.iterdir())
    with ThreadPoolExecutor(max_workers=max(len(ps), 1)) as ex:
        sizes = list(ex.map(lambda p: du(p.resolve(), limit) if p.exists() else {"bytes": None, "files": None}, ps))
    return [{**entry(p, devs), **s} for p, s in zip(ps, sizes)]


def slots() -> dict:
    """The node's check slots and pools (`slot.py --status --json`) and Lean slots (`lean_slot.py`): names, cores and whether
    held, by whom and for how long; never a command."""
    out: dict = {}
    files = {"check": LOCKS / "slots", **{f"pool {p.parent.name}": p for p in sorted(LOCKS.glob("*/slots"))}}
    for name, f in files.items():
        if not f.exists():
            out[name] = None
            continue
        r = subprocess.run([sys.executable, str(HERE / "slot.py"), "--status", "--json"], capture_output=True, text=True,
                           env={**os.environ, "CHECK_SLOTS": str(f)}, timeout=60)
        try:
            doc = json.loads(r.stdout)
        except ValueError:
            out[name] = {"error": (r.stderr or r.stdout).strip().splitlines()[-1:]}
            continue
        out[name] = {"file": doc.get("file"), "waiters": len(doc.get("waiters") or []), "slots": [
            {"name": s["name"], "cpus": s["cpus"], "mem": s.get("mem"), "trains": s.get("trains"), "blocked": s.get("blocked"),
             "held": None if not s.get("holder") else {k: s["holder"].get(k) for k in ("minutes", "run", "by", "since")}}
            for s in doc.get("slots") or []]}
    try:
        r = subprocess.run([sys.executable, str(HERE / "lean_slot.py"), "--status", "--json"], capture_output=True, text=True, timeout=60)
        doc = json.loads(r.stdout)
        pools = doc.get("pools")
        out["lean"] = {"file": doc.get("file"), "pools": None if pools is None else {
            name: {"slots": len(p["slots"]), "held": sum(1 for s in p["slots"] if s.get("holder")), "waiters": len(p["waiters"])}
            for name, p in pools.items()}}
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        out["lean"] = {"error": type(e).__name__}
    return out


def main() -> int:
    limit = float(sys.argv[sys.argv.index("--du-s") + 1]) if "--du-s" in sys.argv else 600.0
    out = Path(os.environ.get("OUT") or ".")
    out.mkdir(parents=True, exist_ok=True)
    devs = {Path("/").stat().st_dev: "root"}
    if Path("/workspace").is_dir():
        devs.setdefault(Path("/workspace").stat().st_dev, "/workspace")
    df = subprocess.run(["df", "-BG", "--output=target,size,avail,pcent,ipcent", "/", "/workspace"], capture_output=True, text=True).stdout
    cache = Path.home() / ".cache" / "verity-check"
    doc: dict = {"machine": os.environ.get("RESEARCH_MACHINE") or os.environ.get("RESEARCH_ON"), "home": str(Path.home()),
                 "df": df.splitlines(), "cache": {"path": str(cache), **entry(cache, devs)}}
    print("node-layout: df " + " | ".join(" ".join(x.split()) for x in doc["df"]), flush=True)
    print(f"node-layout: cache {json.dumps(doc['cache'])}", flush=True)
    doc["slots"] = slots()
    print(f"node-layout: slots {json.dumps(doc['slots'])}", flush=True)
    (out / "node-layout.json").write_text(json.dumps(doc, indent=1) + "\n")
    with ThreadPoolExecutor(max_workers=3) as ex:
        a = ex.submit(listing, cache, devs, limit)
        b = ex.submit(listing, Path.home() / ".cache" / "verity", devs, limit)
        c = ex.submit(listing, Path("/workspace/research/cache"), devs, limit)
        doc["under_cache"], doc["suites_caches"], doc["workspace_cache"] = a.result(), b.result(), c.result()
    for key in ("under_cache", "suites_caches", "workspace_cache"):
        for e in doc[key]:
            print(f"node-layout: {key} {json.dumps(e)}", flush=True)
    for name in ("root", "/workspace"):
        rows = [e for e in doc["under_cache"] if e["disk"] == name]
        print(f"node-layout: the check cache's entries on {name}: {len(rows)}, "
              f"{sum(e['bytes'] or 0 for e in rows) / 2**30:.1f} GiB, {sum(e['files'] or 0 for e in rows)} files"
              + (", some not measured" if any(e["bytes"] is None or e["files"] is None for e in rows) else ""), flush=True)
    (out / "node-layout.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
