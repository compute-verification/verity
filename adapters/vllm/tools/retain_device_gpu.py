"""retain_device_gpu.sh's three parts: the draws, the drawer and the comparison of a ``--retain device`` serve with a
``--retain host`` one of the same load.

    python retain_device_gpu.py reads HOST_DIR READS_DIR          # each sealed window's draw, from the host serve's store
    python retain_device_gpu.py draw DEVICE_DIR READS_DIR --protocol P --delay-s S --stop FILE
    python retain_device_gpu.py compare DEVICE_DIR HOST_DIR READS_DIR

``reads`` picks, for each window the host serve sealed, a draw by hand (no subprotocol's draw is wired in yet): the window's first
and last steps, three members of each (the largest one's first two leaves, a middle member's middle leaf, the last member), and one
member of a middle step that no read names (``undrawn``). It prints the largest window's committed bytes, which sizes the device
pool. ``draw`` is the hand drawer: once a window's ``windows.jsonl`` line is out it waits ``--delay-s`` (so the next window's steps
fill the pool and wait on this window's release) and writes ``draws/w<k>.<P>.json`` whole, until ``--stop`` exists and every
window's line has had its draw. ``compare`` holds each window the device serve sealed to the host serve's: the device bundle keeps
exactly the leaves the draw opens (`serve_release.leaves_of`), each equal to the host bundle's leaf; every drawn read verifies
against the device bundle's own run root; the undrawn read is refused by name there and answered by the host bundle; the tree
levels are whole. Its last line is ``RETAIN-DEVICE PASS|FAIL``, with the device pool's peak and waits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from verity_vllm.commit import store_dump as SD
from verity_vllm.commit.opened import ORDINAL_SEP, OpenedReader
from verity_vllm.pipeline import replay_bundle as RB
from verity_vllm.pipeline import serve_release as SL


def lines(p: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def store_dir(serve: Path, k: int) -> Path:
    return RB.path(serve / "windows" / f"w{k}" / "commit", 0) / "store"


def _named(metas: list[list], i: int) -> str:
    name = metas[i][1]
    return f"{name}{ORDINAL_SEP}{metas[i][0]}" if sum(m[1] == name for m in metas) > 1 else name


def pick(doc: dict[str, Any]) -> dict[str, Any]:
    """A draw of one window by hand: reads of its first and last steps, and an undrawn read of a middle step."""
    C, steps = int(doc["chunk"]), doc["steps"]
    reads = []
    for s in sorted({0, len(steps) - 1}):
        metas = steps[s]["metas"]
        big = max(range(len(metas)), key=lambda i: metas[i][4])
        mid, last = len(metas) // 2, len(metas) - 1
        n = metas[mid][4]
        reads += [[s, _named(metas, big), 0, min(metas[big][4], 2 * C)],
                  [s, _named(metas, mid), n // 2, min(n, n // 2 + C)],
                  [s, _named(metas, last), 0, min(metas[last][4], 4 * C)]]
    u = len(steps) // 2
    metas = steps[u]["metas"]
    big = max(range(len(metas)), key=lambda i: metas[i][4])
    undrawn = [u, _named(metas, big), 0, min(metas[big][4], C)] if 0 < u < len(steps) - 1 else None
    return {"reads": [r for r in reads if r[3] > r[2]], "undrawn": undrawn}


def reads(a: argparse.Namespace) -> int:
    host, out = Path(a.serve), Path(a.reads)
    out.mkdir(parents=True, exist_ok=True)
    biggest, n = 0, 0
    for s in lines(host / "seals.jsonl"):
        k = s["window"]
        p = store_dir(host, k) / "store.json"
        if "error" in s or not p.exists():
            continue
        doc = json.loads(p.read_text())
        (out / f"w{k}.json").write_text(json.dumps({"window": k, **pick(doc)}))
        biggest, n = max(biggest, sum(max(int(st["used"]), int(st["n_bytes"])) for st in doc["steps"])), n + 1
    print(json.dumps({"windows": n, "max_window_bytes": biggest}))
    return 0 if n else 1


def draw(a: argparse.Namespace) -> int:
    serve, rd = Path(a.serve), Path(a.reads)
    d = Path(a.draws_dir) if a.draws_dir else serve / "draws"
    d.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    while True:
        stop = Path(a.stop).exists()
        new = [w["window"] for w in lines(serve / "windows.jsonl") if w["window"] not in done]
        if stop and not new:
            return 0
        for k in new:
            time.sleep(a.delay_s)
            f = rd / f"w{k}.json"
            got = json.loads(f.read_text())["reads"] if f.exists() else []
            body = {"window": k, "protocol": a.protocol, "reads": got}
            tmp = d / f".w{k}.{a.protocol}.json.tmp"
            tmp.write_text(json.dumps(body))
            os.replace(tmp, d / f"w{k}.{a.protocol}.json")
            done.add(k)
            print(json.dumps({"drew": k, "reads": len(got), "from_host_window": f.exists(), "t": time.time()}), flush=True)
        time.sleep(0.5)


def _kept_leaves(doc: dict[str, Any]) -> dict[int, set[int]]:
    C = int(doc["chunk"])
    out: dict[int, set[int]] = {}
    for st in doc["steps"]:
        for lo, hi in st["ranges"]:
            out.setdefault(int(st["step"]), set()).update(range(lo // C, -(-hi // C)))
    return {s: v for s, v in out.items() if v}


def window(dev: Path, host: Path, k: int, pk: dict[str, Any]) -> dict[str, Any]:
    ddoc = json.loads((dev / "store.json").read_text())
    hdoc = json.loads((host / "store.json").read_text())
    ds, drun = SD.load(dev)
    hs, hrun = SD.load(host)
    rd = [tuple(r) for r in pk["reads"]]
    want = {s: set(v) for s, v in SL.leaves_of(ds, rd).items()}
    kept = _kept_leaves(ddoc)
    bad: list[str] = []
    if kept != want:
        bad.append(f"kept leaves {sorted((s, len(v)) for s, v in kept.items())} are not the draw's {sorted((s, len(v)) for s, v in want.items())}")
    if [s["metas"] for s in ddoc["steps"]] != [s["metas"] for s in hdoc["steps"]]:
        bad.append("the device and host windows' layouts differ")
    leaves = 0
    for s, v in sorted(want.items()):
        for i in sorted(v):
            if bytes(ds.open_range(s, i, i + 1).value) != bytes(hs.open_range(s, i, i + 1).value):
                bad.append(f"step {s} leaf {i}: the device bundle's bytes are not the host bundle's")
            leaves += 1
    dr, hr = OpenedReader(ds, drun), OpenedReader(hs, hrun)
    for r in rd:
        a, b = dr(*r), hr(*r)
        if a is None or a != b:
            bad.append(f"read {list(r)}: device {None if a is None else len(a)} bytes, host {None if b is None else len(b)}: not equal")
    if dr.stats["failed"]:
        bad.append(f"{dr.stats['failed']} drawn read(s) did not verify against the device bundle's run root")
    refusal = None
    if pk.get("undrawn"):
        u = tuple(pk["undrawn"])
        try:
            dr(*u)
            bad.append(f"the undrawn read {list(u)} was answered from the device bundle")
        except RuntimeError as e:
            refusal = f"{type(e).__name__}: {e}"[:300]
        if not hr(*u):
            bad.append(f"the undrawn read {list(u)} is not in the host bundle")
    if [s["levels"] for s in ddoc["steps"]] != [s["levels"] for s in hdoc["steps"]]:
        bad.append("the device bundle's tree levels are not whole")
    return {"window": k, "steps": len(ddoc["steps"]), "reads": len(rd), "drawn_leaves": leaves,
            "kept_raw_bytes": _raw(ddoc), "host_raw_bytes": _raw(hdoc), "chunk": int(ddoc["chunk"]),
            "run_root": ddoc["run_root"][:16], "undrawn_refused": refusal, "ok": not bad, "bad": bad[:8]}


def _raw(doc: dict[str, Any]) -> int:
    return sum(hi - lo for st in doc["steps"] for lo, hi in st["ranges"])


def compare(a: argparse.Namespace) -> int:
    dev, host, rd = Path(a.serve), Path(a.host), Path(a.reads)
    seals = {s["window"]: s for s in lines(dev / "seals.jsonl")}
    roots = {w["window"]: w.get("run_root") for w in lines(dev / "windows.jsonl")}
    summary = json.loads((dev / "summary.json").read_text()) if (dev / "summary.json").exists() else {}
    rel = summary.get("release") or {}
    out: list[dict[str, Any]] = []
    for k, s in sorted(seals.items()):
        f = rd / f"w{k}.json"
        if "error" in s or not f.exists() or not (store_dir(host, k) / "store.json").exists():
            out.append({"window": k, "ok": False, "bad": [f"not compared: {s.get('error') or 'no host window ' + str(k)}"[:400]]})
            continue
        try:
            got = window(store_dir(dev, k), store_dir(host, k), k, json.loads(f.read_text()))
        except (OSError, ValueError, KeyError, IndexError, TypeError, RuntimeError, AssertionError) as e:
            got = {"window": k, "ok": False, "bad": [f"{type(e).__name__}: {e}"[:400]]}
        got["release"] = {x: (s.get("release") or {}).get(x) for x in
                          ("union_leaves", "kept_bytes", "dropped_bytes", "device_freed_bytes", "draw_wait_s", "discarded")}
        got["pre_coin_root_kept"] = (roots.get(k) or "")[:16] == got.get("run_root")
        got["ok"] = got["ok"] and got["pre_coin_root_kept"]
        out.append(got)
    pool = rel.get("device_pool") or {}
    ok = bool(out) and all(w["ok"] for w in out) and not rel.get("discarded") and not rel.get("failure")
    rec = {"windows": out, "device_pool": pool, "released": rel.get("released"), "kept_bytes": rel.get("kept_bytes"),
           "dropped_bytes": rel.get("dropped_bytes"), "discarded": rel.get("discarded"), "failure": rel.get("failure")}
    print(json.dumps(rec, indent=1))
    first = next((f"w{w['window']}: {(w.get('bad') or ['its pre-coin root is not its windows.jsonl line'])[0]}" for w in out
                  if not w["ok"]), None) or (f"discarded {rel.get('discarded')}, failure {rel.get('failure')}" if out else "no window sealed")
    what = ("each holding exactly its draw's leaves, byte-equal to --retain host's, the undrawn read refused" if ok
            else f"first problem: {first}"[:600])
    print(f"RETAIN-DEVICE {'PASS' if ok else 'FAIL'}: {len(out)} window(s) sealed from the device pool "
          f"({sum(w.get('drawn_leaves', 0) for w in out)} drawn leaves), {what}; pool peak {pool.get('peak_bytes')} of "
          f"{pool.get('capacity_bytes')} bytes, {pool.get('waits')} wait(s) ({pool.get('waited_s')} s); kept {rel.get('kept_bytes')} "
          f"dropped {rel.get('dropped_bytes')} bytes")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reads")
    p.add_argument("serve")
    p.add_argument("reads")
    p = sub.add_parser("draw")
    p.add_argument("serve")
    p.add_argument("reads")
    p.add_argument("--protocol", default="pouw")
    p.add_argument("--delay-s", type=float, default=20.0)
    p.add_argument("--draws-dir", default="")
    p.add_argument("--stop", required=True)
    p = sub.add_parser("compare")
    p.add_argument("serve")
    p.add_argument("host")
    p.add_argument("reads")
    a = ap.parse_args(argv)
    return {"reads": reads, "draw": draw, "compare": compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
