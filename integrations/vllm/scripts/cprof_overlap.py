#!/usr/bin/env python3
"""CPERF 2026-09-17 §6: overlap experiment -- replay/acquisition alone, commitment alone, both concurrently.

Three timed configurations over the SAME production record (workload-B0-small) on one worker:

  A  replay --mode commit as shipped                 (capture-time leaf hashing inside emit + serial build_commitments at the end)
  B  replay with H = framing-only (no SHA anywhere)  (acquisition without any hash arithmetic: D2H + tobytes + write + bookkeeping)
  C  replay as shipped + a concurrent WATCHER process that builds every RU tree (packed SHA-256, fasttree) as soon as its
     index.json is complete; reports how much of the tree work finished before the replay's own forward ended ("hidden")
     and the tail after the last RU landed.
  D  commitment alone: build_commitments over the finished store (serial, as replay.py does), and fasttree packed over the same

The replay is launched as a subprocess of this script (``verity_vllm_adapter.replay`` unchanged; the H patch of B is
installed by a tiny sitecustomize-style bootstrap module in this snapshot, never by editing production files).

    PYTHONPATH=. python scripts/cprof_overlap.py --production <dir> --out-root <dir> --checkpoints <json> [--arena-gb 4]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
POC = HERE.parent
if str(POC) not in sys.path:
    sys.path.insert(0, str(POC))

BOOT_NOOP = """
import sys
from verity_vllm import hashing
_Z = bytes(32)
def H_noop(tag, *parts):
    t = tag.encode('utf-8'); n = 4 + len(t)
    for p in parts: n += 8 + len(p)
    return _Z
hashing.H = H_noop
from verity_vllm_adapter import replay
sys.argv[0] = 'replay.py'
replay.main()
"""

BOOT_REAL = """
import sys
from verity_vllm_adapter import replay
sys.argv[0] = 'replay.py'
replay.main()
"""


def run_replay(py: str, boot: str, production: Path, out: Path, checkpoints: Path, arena_gb: float, rev: str, log: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    cmd = [py, "-c", boot, "--case", "B0", "--checkpoints", str(checkpoints), "--production", str(production), "--out", str(out),
           "--mode", "commit", "--weights-values", "store", "--adapter-revision", rev, "--arena-gb", str(arena_gb), "--replay-exec", "eager"]
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{POC}:{env.get('PYTHONPATH', '')}"
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env).returncode
    wall = time.perf_counter() - t0
    rj = json.loads((out / "replay.json").read_text()) if (out / "replay.json").exists() else {}
    return {"rc": rc, "process_wall_s": wall, "timing": rj.get("timing"), "commitments": rj.get("commitments"), "capture": rj.get("capture"),
            "profile_id": rj.get("profile_id")}


def _watcher(out: str, stop_path: str, result_path: str, n_expected: int) -> None:
    """Build each RU tree (packed sha256) as soon as its index.json exists; record per-RU arrival/finish times."""
    import numpy as np

    from verity_vllm import identity
    from verity_vllm.capture import RUReader
    from verity_vllm.commit_bench import fasttree as ft

    out_p = Path(out)
    done: dict[str, dict] = {}
    t_start = time.perf_counter()
    busy_s = 0.0
    while True:
        found = False
        for idx in sorted(out_p.glob("ru/*/*/index.json")):
            tid = f"{idx.parent.parent.name}/{idx.parent.name}"
            if tid in done:
                continue
            found = True
            t_seen = time.perf_counter()
            try:
                ru = RUReader(idx.parent, idx.parent.parent.name, int(idx.parent.name))
            except Exception:  # noqa: BLE001  (index being written)
                continue
            ids = identity.tree_sorted([e.canonical_id for e in ru.index.entries])
            by = {e.canonical_id: e for e in ru.index.entries}
            mm = memoryview(np.memmap(idx.parent / "values.bin", dtype=np.uint8, mode="r"))
            headers = [ft.leaf_header(c, by[c].dtype, by[c].shape) for c in ids]
            values = [mm[by[c].offset : by[c].offset + by[c].nbytes] for c in ids]
            root = ft.tree_root(headers, values)
            t_done = time.perf_counter()
            busy_s += t_done - t_seen
            done[tid] = {"seen": t_seen - t_start, "done": t_done - t_start, "s": t_done - t_seen, "root": root.hex(), "leaves": len(ids)}
        if not found:
            if Path(stop_path).exists() and len(done) >= n_expected:
                break
            if Path(stop_path).exists() and not found:
                # replay ended; one more sweep already happened with nothing new
                break
            time.sleep(0.02)
    Path(result_path).write_text(json.dumps({"trees": len(done), "busy_s": busy_s, "per_tree": done}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--production", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--arena-gb", type=float, default=4.0)
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--rev", default="cperf-overlap")
    ap.add_argument("--skip", default="", help="comma list of A,B,C,D to skip")
    args = ap.parse_args()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip.split(",")) if args.skip else set()
    prod = Path(args.production)
    res: dict = {"schema": "veritor/cperf/overlap/v1", "production": str(prod), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    from verity_vllm.commit_bench.prov import provenance
    res["provenance"] = provenance()

    if "A" not in skip:
        res["A_replay_as_shipped"] = run_replay(args.py, BOOT_REAL, prod, root / "A", Path(args.checkpoints), args.arena_gb, args.rev, root / "A.log")
        print("A", json.dumps({k: res["A_replay_as_shipped"][k] for k in ("rc", "process_wall_s", "commitments")}), flush=True)
    if "B" not in skip:
        res["B_replay_noop_hash"] = run_replay(args.py, BOOT_NOOP, prod, root / "B", Path(args.checkpoints), args.arena_gb, args.rev, root / "B.log")
        print("B", json.dumps({k: res["B_replay_noop_hash"][k] for k in ("rc", "process_wall_s", "commitments")}), flush=True)
    if "C" not in skip:
        out = root / "C"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        stop = root / "C.stop"
        result = root / "C.watcher.json"
        for p in (stop, result):
            if p.exists():
                p.unlink()
        n_expected = 64
        ctx = mp.get_context("fork")
        w = ctx.Process(target=_watcher, args=(str(out), str(stop), str(result), n_expected))
        t0 = time.perf_counter()
        w.start()
        rep = run_replay(args.py, BOOT_REAL, prod, out, Path(args.checkpoints), args.arena_gb, args.rev, root / "C.log")
        t_replay_end = time.perf_counter() - t0
        stop.write_text("1")
        w.join(timeout=600)
        t_all = time.perf_counter() - t0
        wres = json.loads(result.read_text()) if result.exists() else {}
        per = wres.get("per_tree", {})
        last_done = max((v["done"] for v in per.values()), default=None)
        last_seen = max((v["seen"] for v in per.values()), default=None)
        res["C_replay_plus_concurrent_watcher"] = {
            "replay": rep, "replay_process_wall_s": t_replay_end, "wall_to_last_tree_s": last_done, "last_ru_seen_s": last_seen,
            "watcher_trees": wres.get("trees"), "watcher_busy_s": wres.get("busy_s"), "total_wall_s": t_all,
            "tail_after_last_ru_s": (last_done - last_seen) if last_done is not None and last_seen is not None else None,
            "note": "watcher = separate forked process, packed SHA-256 fasttree per RU as its index.json lands; the replay still runs its own serial build_commitments at the end (its commitments_seconds is inside replay_process_wall_s)",
        }
        # do the watcher's roots agree with the replay's commitments.json?
        cj = out / "commitments.json"
        if cj.exists():
            trees = json.loads(cj.read_text())["trees"]
            agree = sum(1 for tid, v in per.items() if trees.get(f"ru/{tid}", {}).get("root") == v["root"])
            res["C_replay_plus_concurrent_watcher"]["watcher_roots_agree_with_commitments_json"] = f"{agree}/{len(per)}"
        print("C", json.dumps({k: v for k, v in res["C_replay_plus_concurrent_watcher"].items() if k != "replay"}), flush=True)
    if "D" not in skip:
        from verity_vllm.capture import CaptureReader
        from verity_vllm.commit import build_commitments
        from verity_vllm.commit_bench.prov import timed
        store = root / "A" if (root / "A" / "commitments.json").exists() else prod.parent / "commit"
        res["D_commit_alone"] = {"store": str(store),
                                 "build_commitments_recompute": timed(lambda: build_commitments(CaptureReader(store), recompute_leaves=True, write=False), 3),
                                 "build_commitments_trusted": timed(lambda: build_commitments(CaptureReader(store), recompute_leaves=False, write=False), 3)}
        print("D", json.dumps({k: (v.get("median_s") if isinstance(v, dict) else v) for k, v in res["D_commit_alone"].items()}), flush=True)
    (root / "overlap.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    small = json.loads(json.dumps(res, default=str))
    for k in ("A_replay_as_shipped", "B_replay_noop_hash"):
        if k in small:
            small[k] = {kk: small[k][kk] for kk in ("rc", "process_wall_s", "timing", "commitments")}
    if "C_replay_plus_concurrent_watcher" in small:
        c = small["C_replay_plus_concurrent_watcher"]
        c["replay"] = {kk: c["replay"][kk] for kk in ("rc", "process_wall_s", "timing", "commitments")}
    for k in ("D_commit_alone",):
        if k in small:
            for kk, v in small[k].items():
                if isinstance(v, dict):
                    v.pop("all_s", None)
    print("CPERF-OVERLAP-JSON " + json.dumps(small, separators=(",", ":"), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
