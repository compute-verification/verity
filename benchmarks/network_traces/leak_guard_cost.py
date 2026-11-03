"""The record's leak guard's cost on a gate of a given size. The gate run is synthetic: its public files are a published
gate bundle's (each repeated SCALE times) and its witness/ holds random files of the sizes the bundle's end.json records
(each times SCALE; a witness file that is a public file's twin is its copy). The run dir is as a gate wrapper leaves it:
out/bundle.tgz (that gate's public files) and a runner's resources.jsonl, with a hold beside it holding one file that
carries a witness piece as hex, which the guard must refuse. The guard reads the gate as its fetch (a tarball), the run
dir and the hold; each scale's time, peak memory and verdict go to stdout and OUT/cost.json, by counts and paths only.

    python3 leak_guard_cost.py BUNDLE.tgz --out DIR --scratch DIR [--scale K]... [--old GUARD.py --old-scale K
        --old-cap S]

--old runs another guard (one without --hold) on the same run dir, the planted file in out/, with --dry, for at most
--old-cap seconds.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEXT = (".json", ".jsonl", ".log")
BLOCK = 8 << 20


def _random(path: Path, n: int, text: bool) -> None:
    with open(path, "wb") as f:
        while n > 0:
            b = os.urandom(min(BLOCK, n))
            b = b.hex().encode() if text else b
            f.write(b[:n])
            n -= min(len(b), n)


def _repeat(path: Path, k: int) -> None:
    if k == 1:
        return
    tmp = path.with_name(path.name + ".x")
    with open(tmp, "wb") as w:
        for _ in range(k):
            with open(path, "rb") as r:
                shutil.copyfileobj(r, w, BLOCK)
    tmp.replace(path)


def _tar(path: Path, root: Path, members) -> int:
    with tarfile.open(path, "w:gz", compresslevel=1) as t:
        for p in members:
            t.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
    return path.stat().st_size


def build(bundle: Path, scale: int, w: Path) -> dict:
    """The synthetic gate at `scale` under `w`: its fetch, the run dir and the hold."""
    src = w / "gate"
    src.mkdir(parents=True)
    with tarfile.open(bundle) as t:
        t.extractall(src, filter="data")
    g = next(src.rglob("end.json")).parent
    public = sorted(p for p in g.rglob("*") if p.is_file())
    sizes = json.loads((g / "end.json").read_text())["witness"]
    for p in public:
        _repeat(p, scale)
    twins = {p.name: p for p in public}
    (g / "witness").mkdir()
    for name, size in sizes.items():
        n, p = size * scale, g / "witness" / name
        if name in twins and twins[name].stat().st_size == n:
            shutil.copyfile(twins[name], p)
        else:
            _random(p, n, name.endswith(TEXT))
    salts = (g / "witness" / "cells.salts").read_bytes()[:16]
    run, hold = w / "run", w / "hold"
    (run / "out").mkdir(parents=True)
    hold.mkdir()
    members = sorted(p for p in src.rglob("*") if "witness" not in p.relative_to(g).parts)
    bundle_bytes = _tar(run / "out" / "bundle.tgz", src, members)
    (run / "out" / "start.json").write_text(json.dumps({"gate": g.name, "scale": scale}) + "\n")
    _random(run / "resources.jsonl", 74_420_345 * scale, True)
    (hold / "planted.json").write_text(json.dumps({"x": salts.hex()}) + "\n")
    (hold / "session.json").write_text(json.dumps({"scale": scale}) + "\n")
    fetch = w / "fetch.tar.gz"
    fetch_bytes = _tar(fetch, src, sorted(src.rglob("*")))
    return {"gate": src, "fetch": fetch, "run": run, "hold": hold,
            "public_bytes": sum(p.stat().st_size for p in public), "witness_bytes": sum(sizes.values()) * scale,
            "bundle_bytes": bundle_bytes, "fetch_bytes": fetch_bytes,
            "run_bytes": sum(p.stat().st_size for p in run.rglob("*") if p.is_file())}


def _line(f: Path, i: int) -> str:
    return (f.read_text().strip().splitlines() or [""])[i][:300]


def timed(argv: list[str], logs: Path, cap: float | None = None) -> dict:
    """`argv`'s seconds, exit and peak RSS, killed past `cap` seconds; its first stdout line and last stderr line."""
    with open(logs.with_suffix(".out"), "w") as o, open(logs.with_suffix(".err"), "w") as e:
        t0 = time.monotonic()
        p = subprocess.Popen(argv, stdout=o, stderr=e)
        timer = threading.Timer(cap, p.kill) if cap else None
        if timer:
            timer.start()
        _, status, usage = os.wait4(p.pid, 0)
        s = time.monotonic() - t0
        if timer:
            timer.cancel()
    return {"s": round(s, 1), "rc": os.waitstatus_to_exitcode(status),
            "unfinished_at_cap": cap if cap and s >= cap else None, "maxrss_mb": usage.ru_maxrss // 1024,
            "said": _line(logs.with_suffix(".out"), 0), "err": _line(logs.with_suffix(".err"), -1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bundle", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--scale", type=int, action="append")
    ap.add_argument("--old", type=Path)
    ap.add_argument("--old-scale", type=int, action="append", default=[])
    ap.add_argument("--old-cap", type=float, default=1800)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    doc = {"python": sys.version.split()[0], "numpy": importlib.util.find_spec("numpy") is not None,
           "cpus": os.cpu_count(), "scales": rows}
    for k in a.scale or [1]:
        w = a.scratch / f"x{k}"
        t0 = time.monotonic()
        b = build(a.bundle, k, w)
        row = {"scale": k, "build_s": round(time.monotonic() - t0, 1),
               **{x: b[x] for x in ("public_bytes", "witness_bytes", "bundle_bytes", "fetch_bytes", "run_bytes")}}
        if a.old is not None and k in a.old_scale:
            shutil.copyfile(b["hold"] / "planted.json", b["run"] / "out" / "planted.json")
            row["old"] = timed([sys.executable, str(a.old), str(b["run"]), str(w / "q-old"), "--gates", str(b["gate"]),
                                "--dry", "--declared", str(HERE / "leak_guard.json"), "--report", str(w / "old.json")],
                               a.out / f"old-x{k}", a.old_cap)
            if (w / "old.json").is_file():
                r = json.loads((w / "old.json").read_text())
                row["old"] |= {"verdict": r.get("verdict"), "found": [d["path"] for d in r.get("found", [])]}
            (b["run"] / "out" / "planted.json").unlink()
        row["new"] = timed([sys.executable, str(HERE / "leak_guard.py"), str(b["run"]), str(w / "q"), "--gates",
                            str(b["fetch"]), "--hold", str(b["hold"]), "--release-to", str(b["run"] / "out"),
                            "--report", str(w / "new.json")], a.out / f"new-x{k}")
        if (w / "new.json").is_file():
            r = json.loads((w / "new.json").read_text())
            row["new"] |= {x: r[x] for x in ("verdict", "pieces", "files_scanned", "bytes_scanned", "public_files",
                                              "public_dropped", "held", "released")}
            row["new"]["found"] = [d["path"] for d in r["found"]]
            row["new"]["out"] = sorted(p.name for p in (b["run"] / "out").iterdir())
        rows.append(row)
        print(json.dumps(row), flush=True)
        (a.out / "cost.json").write_text(json.dumps(doc, indent=1) + "\n")
        shutil.rmtree(w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
