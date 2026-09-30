"""The backend sweep's feeder for vy-nebius-1's dispatcher: each unit shape of a cut row (73-sweep-shape.sh MODE=cut) as two ready
files, a CPU stage job and then, once that succeeded, a one-GPU prove job, so the GPU is requested only to prove.

    sweep_feed.py tick|loop|status --row ROW [--row ROW ...] [--limit N] [--ahead 6] [--every 60]

M0's pinned benches keep priority: a file is written only while `provers` has no pending workload, M0's ready directory
(flock-netlist) is empty and this lane's holds nothing unsubmitted, at most PER_TICK stage and prove files a tick (the dispatcher's
`provers` depth of 2 counts anyone's pending workloads, so ours never fill it ahead of a bench).
Ids are `<tag>-s<i>` (stage) and `<tag>-p<i>` (prove), `i` the shape's line in $SWEEP/<row>/shapes.tsv (smallest first), `tag` a
digest of the row. Every 25th prove job selftests on the GPU (the device's proofs and transcripts against the CPU's). `--limit`: only
the first N shapes of each row. State is the dispatcher's own: its ready directory and done.jsonl, and the feeder's $SWEEP_FEED_STATE/written.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

LANE = "backend-sweep-2"
READY = Path(os.environ.get("VY_READY", "/workspace/jobs/ready")) / LANE
DONE = Path(os.environ.get("VY_DISPATCH_ROOT", "/workspace/jobs/dispatch")) / "done.jsonl"
SWEEP = Path(os.environ.get("SWEEP", "/workspace/jobs/sweep2"))
STATE = Path(os.environ.get("SWEEP_FEED_STATE", "/workspace/jobs/sweep2-feed"))   # the feeder's (research's); SWEEP is the jobs'
TREE = os.environ.get("SWEEP_TREE", f"/workspace/research/trees/{LANE}")
SELFTEST_EVERY = 25
STAGE_RES = {"cpus": 2, "memory": 32, "gpus": 0}                 # the K = 2048 GEMM coordinate's stage: 10.7 GiB
PROVE_RES = {"cpus": 4, "memory": 48, "gpus": 1}
SELFTEST_RES = {"cpus": 8, "memory": 128, "gpus": 1}              # its 43-case GPU selftest at the full batch: 101.5 GiB
PER_TICK = 3


def provers_pending() -> int:
    wls = json.loads(subprocess.run(["kubectl", "get", "workloads", "-o", "json"], capture_output=True, text=True, check=True).stdout)["items"]
    return sum(1 for w in wls if (w.get("spec") or {}).get("queueName") == "provers" and not (w.get("status") or {}).get("admission")
               and not any(c.get("type") == "Finished" and c.get("status") == "True" for c in (w.get("status") or {}).get("conditions") or []))


def clear_to_write() -> bool:
    busy = lambda d: any(d.glob("*.json")) if d.is_dir() else False  # noqa: E731
    return not busy(READY) and not busy(READY.parent / "flock-netlist") and provers_pending() == 0


def tag(row: str) -> str:
    return row.split("__")[0] + "-" + hashlib.sha256(row.encode()).hexdigest()[:6]


def shapes(row: str) -> list[str]:
    return [ln.split("\t")[0] for ln in (SWEEP / row / "shapes.tsv").read_text().splitlines() if ln.strip()]


def done() -> dict[str, dict]:
    out = {}
    if DONE.exists():
        for ln in DONE.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r["key"].startswith(LANE + "/"):
                    out[r["key"].split("/", 1)[1]] = r
    return out


def written() -> set[str]:
    p = STATE / "written.txt"
    return set(p.read_text().split()) if p.exists() else set()


def write(item_id: str, item: dict) -> None:
    READY.mkdir(parents=True, exist_ok=True)
    tmp = READY / f".{item_id}.tmp"
    tmp.write_text(json.dumps(item, sort_keys=True))
    tmp.rename(READY / f"{item_id}.json")
    STATE.mkdir(parents=True, exist_ok=True)
    with open(STATE / "written.txt", "a") as f:
        f.write(item_id + "\n")


def item(row: str, shape: str, mode: str, i: int) -> dict:
    env, res = f"MODE={mode} ROW={row} SHAPE={shape}", STAGE_RES if mode == "stage" else PROVE_RES
    if mode == "shape" and i % SELFTEST_EVERY == 0:
        env, res = env + " SELFTEST=1 SELFTEST_GPU=1", SELFTEST_RES
    return {"template": "prover-bench", "tree": TREE, "queue": "provers", "priority": "dev", "resources": {"prover-bench": res},
            "env": {"CAMPAIGN": "backend-sweep", "CMD": f"bash backends/flock/pod/73-sweep-shape.sh {env}"}}


def tick(rows: list[str], limit: int | None, ahead: int, dry: bool = False) -> dict:
    d, w = done(), written()
    todo: dict[str, list[tuple[str, dict]]] = {"prove": [], "stage": []}
    counts = {}
    for row in rows:
        t, sh = tag(row), shapes(row)[:limit]
        staging = sum(1 for i in range(len(sh)) if f"{t}-s{i}" in w and f"{t}-s{i}" not in d)
        for i, s in enumerate(sh):
            sid, pid = f"{t}-s{i}", f"{t}-p{i}"
            if len(todo["prove"]) < PER_TICK and sid in d and d[sid].get("state") == "succeeded" and pid not in w:
                todo["prove"].append((pid, item(row, s, "shape", i)))
            if len(todo["stage"]) < PER_TICK and sid not in w and staging < ahead:
                todo["stage"].append((sid, item(row, s, "stage", i)))
                staging += 1
        p = [f"{t}-p{i}" for i in range(len(sh))]
        s_ = [f"{t}-s{i}" for i in range(len(sh))]
        counts[row] = {"shapes": len(sh), "queued": sum(x in w for x in p),
                       "done": sum(d.get(x, {}).get("state") == "succeeded" for x in p),
                       "failed": sum(x in d and d[x].get("state") != "succeeded" for x in p + s_)}
    if (todo["prove"] or todo["stage"]) and not dry and clear_to_write():
        for item_id, it in todo["prove"] + todo["stage"]:
            write(item_id, it)
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "loop", "status"])
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ahead", type=int, default=6, help="stage jobs written and not yet ended, per row")
    ap.add_argument("--every", type=int, default=60)
    a = ap.parse_args()
    while True:
        c = tick(a.row, a.limit, a.ahead, dry=a.cmd == "status")
        print(time.strftime("%H:%M:%SZ", time.gmtime()), json.dumps(c), flush=True)
        if a.cmd != "loop":
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
