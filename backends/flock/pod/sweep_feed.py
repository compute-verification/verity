"""The backend sweep's feeder for vy-nebius-1's dispatcher: each unit shape of a cut row (73-sweep-shape.sh MODE=cut) as two ready
files, a CPU stage job and then, once that succeeded, a one-GPU prove job, so the GPU is requested only to prove.

    sweep_feed.py tick|loop|status --row ROW[=BUILD] [--row ...] [--limit N] [--ahead 6] [--every 60] [--chunk-from ROW=N]

M0's pinned benches keep priority: a file is written only while `provers` has no pending workload, M0's ready directory
(flock-netlist) is empty and this lane's holds nothing unsubmitted, at most PER_TICK stage and prove files a tick (the dispatcher's
`provers` depth of 2 counts anyone's pending workloads, so ours never fill it ahead of a bench).
Ids are `<tag>-s<i>` (stage) and `<tag>-p<i>` (prove), `i` the shape's line in $SWEEP/<row>/shapes.tsv (smallest first), `tag` a
digest of the row; from --chunk-from on a job takes CHUNK shapes, ids `<tag>-sc<i>` and `<tag>-pc<i>`, `i` its first. Every 25th shape selftests on the GPU (the device's proofs and transcripts against the CPU's). `--limit`: only
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
CUT_RES = {"cpus": 8, "memory": 96, "gpus": 0}                   # Llama-3.2-1B's: 28 min
SAMPLED_STAGE_RES = {"cpus": 8, "memory": 64, "gpus": 0}        # the draw (137 s, 5.9 GB for Llama-3.2-1B) and 4 shapes staged at once
SAMPLED_SELFTEST = 8
PER_TICK = 3
CHUNK = 10                                                       # shapes a job: a one-shape job holds its GPU ~100-250 s to prove < 1 s


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


def item(row: str, sh: list[str], mode: str, i: int) -> dict:
    """The job for shapes i, i+1, ... (sh, their digests): each selftested whose index is a multiple of SELFTEST_EVERY."""
    env, res = f"MODE={mode} ROW={row} SHAPES={','.join(sh)}", STAGE_RES if mode == "stage" else PROVE_RES
    st = [s for j, s in enumerate(sh, i) if j % SELFTEST_EVERY == 0]
    if mode == "shape" and st:
        env, res = env + f" SELFTEST_SHAPES={','.join(st)} SELFTEST_GPU=1", SELFTEST_RES
    return {"template": "prover-bench", "tree": TREE, "queue": "provers", "priority": "dev", "resources": {"prover-bench": res},
            "env": {"CAMPAIGN": "backend-sweep", "CMD": f"bash backends/flock/pod/73-sweep-shape.sh {env}"}}


def jobs(t: str, n: int, chunk_from: int) -> list[tuple[int, int, str]]:
    """(first index, shape count, id suffix): one-shape jobs `<i>` below chunk_from, CHUNK-shape jobs `c<i>` from it."""
    one = [(i, 1, str(i)) for i in range(min(chunk_from, n))]
    return one + [(i, min(CHUNK, n - i), f"c{i}") for i in range(chunk_from, n, CHUNK)]


def cut_item(row: str, build: str) -> dict:
    return {"template": "prover-dev", "tree": TREE, "queue": "provers", "priority": "dev", "resources": {"prover-dev": CUT_RES},
            "env": {"CAMPAIGN": "backend-sweep", "CMD": f"bash backends/flock/pod/73-sweep-shape.sh MODE=cut ROW={row} BUILD={build}"}}


def tick(rows: list[str], limit: int | None, ahead: int, chunk_from: dict[str, int] | None = None, dry: bool = False,
         max_new: int | None = None) -> dict:
    """Rows are `ROW` (cut already) or `ROW=BUILD` (cut here first, one cut in flight at a time), taken in order: a row's shape
    jobs start once every earlier row's prove jobs are written. Counts are shapes: queued (prove job written), done (its job
    succeeded), failed (its stage or prove job failed)."""
    d, w = done(), written()
    todo: dict[str, list[tuple[str, dict]]] = {"prove": [], "stage": [], "cut": []}
    counts, open_ = {}, True
    cutting = any(x.endswith("-cut") and x not in d for x in w)
    for spec in rows:
        row, _, build = spec.partition("=")
        t = tag(row)
        if not (SWEEP / row / "shapes.tsv").exists():
            cid = f"{t}-cut"
            if build and cid not in w and not cutting:
                todo["cut"].append((cid, cut_item(row, build)))
                cutting = True
            failed = cid in d and d[cid].get("state") != "succeeded"
            counts[row] = {"cut": "failed" if failed else "pending"}
            open_ = open_ and failed
            continue
        sh = shapes(row)[:limit]
        js = jobs(t, len(sh), (chunk_from or {}).get(row, 0))
        staging = sum(1 for _, _, x in js if f"{t}-s{x}" in w and f"{t}-s{x}" not in d)
        c = {"shapes": len(sh), "queued": 0, "done": 0, "failed": 0}
        for i, m, x in js:
            sid, pid = f"{t}-s{x}", f"{t}-p{x}"
            if open_ and len(todo["prove"]) < PER_TICK and sid in d and d[sid].get("state") == "succeeded" and pid not in w:
                todo["prove"].append((pid, item(row, sh[i:i + m], "shape", i)))
            if open_ and len(todo["stage"]) < PER_TICK and sid not in w and staging < ahead:
                todo["stage"].append((sid, item(row, sh[i:i + m], "stage", i)))
                staging += 1
            c["queued"] += m * (pid in w)
            c["done"] += m * (d.get(pid, {}).get("state") == "succeeded")
            c["failed"] += m * any(y in d and d[y].get("state") != "succeeded" for y in (sid, pid))
        counts[row] = c
        open_ = open_ and all(f"{t}-p{x}" in w or (f"{t}-s{x}" in d and d[f"{t}-s{x}"].get("state") != "succeeded") for _, _, x in js)
    new = (todo["prove"] + todo["stage"] + todo["cut"])[:max_new]
    if new and not dry and clear_to_write():
        for item_id, it in new:
            write(item_id, it)
    return counts


def summary(name: str) -> dict | None:
    p = SWEEP / name
    return json.loads(p.read_text()) if p.exists() else None


def shape_item(env: str, res: dict) -> dict:
    return {"template": "prover-bench", "tree": TREE, "queue": "provers", "priority": "dev", "resources": {"prover-bench": res},
            "env": {"CAMPAIGN": "backend-sweep", "CMD": f"bash backends/flock/pod/73-sweep-shape.sh {env}"}}


def wholerow(b: dict, d: dict, w: set) -> tuple[list[tuple[str, dict]], dict]:
    """(b): the row's statements of one shape proved outright, in chunks of `chunk` statements (one GPU job each, `parallel` at once),
    until the measured cost agrees with the per-shape extrapolation (`extrap_s` a statement, from the shape's one-statement record)
    within `tol`, over at least `min_chunks` chunks; or the row is proved. Measured = the chunks' job seconds (the GPU held, start-up
    included) per accepted statement, times the row's statements; also the prove seconds alone."""
    row, digest, total, chunk = b["row"], b["shape"], int(b["statements"]), int(b["chunk"])
    t = f"b-{tag(row)}-{digest[:8]}"
    ids = [(f"{t}-r{i}", i, min(chunk, total - i)) for i in range(0, total, chunk)]
    ended = [x for x in ids if x[0] in d]
    sums = [summary(f"wholerow/{x[0]}.json") for x in ended]
    ok = [x for x in sums if x and x.get("rc") == 0 and x.get("accepted_statements") == x.get("statements")]
    n = sum(x["accepted_statements"] for x in ok)
    job_s, prove_s = sum(x["job_s"] for x in ok), sum(x["prove_total_s"] for x in ok)
    extrap = total * float(b["extrap_s"])
    st = {"chunks_done": len(ok), "chunks_failed": len(ended) - len(ok), "statements": n, "of": total, "gpu_h_used": round(job_s / 3600, 3),
          "extrapolated_gpu_h": round(extrap / 3600, 3),
          "measured_gpu_h": round(job_s / n * total / 3600, 3) if n else None,
          "measured_prove_only_gpu_h": round(prove_s / n * total / 3600, 3) if n else None,
          "runs": [x.get("run") for x in ok]}
    st["gap"] = round(st["measured_gpu_h"] / st["extrapolated_gpu_h"] - 1, 4) if n else None
    stopped = STATE / "b-stop.json"
    if stopped.exists():
        return [], json.loads(stopped.read_text())
    if len(ended) > len(ok):
        st["stop"] = "a chunk failed"
    elif len(ok) >= int(b["min_chunks"]) and abs(st["gap"]) <= float(b["tol"]):
        st["stop"] = f"agrees within {float(b['tol']):.0%}"
    elif len(ok) == len(ids):
        st["stop"] = "the whole row proved"
    if "stop" in st:
        st["t"] = time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())
        stopped.write_text(json.dumps(st, indent=1))
        return [], st
    inflight = sum(1 for x in ids if x[0] in w and x[0] not in d)
    todo = []
    for item_id, i, m in ids:
        if item_id not in w and inflight < int(b["parallel"]):
            env = (f"MODE=shape ROW={row} SHAPES={digest} RUNS={m} WARM=1 FLOCK_KEEP_SESSIONS=3 "
                   f"SUMMARY={SWEEP}/wholerow/{item_id}.json")
            todo.append((item_id, shape_item(env, PROVE_RES)))
            inflight += 1
    return todo, st


def sampled(deps: list[tuple[str, str]], d: dict, w: set, ahead: int = 2) -> tuple[list[tuple[str, dict]], dict]:
    """(a): each passing deployment's sampled units (73-sweep-shape.sh MODE=sampled-stage, then MODE=sampled), every unit proved. The
    first deployment's prove job selftests every SAMPLED_SELFTEST-th shape on the GPU; the rest wait until those selftests and their
    GPU cases (the device's proofs and transcripts against the CPU's) all pass."""
    todo, st = [], {"deployments": len(deps), "staged": 0, "queued": 0, "done": 0, "failed": 0, "first": None}
    gate = None
    staging = 0
    for j, (slug, ev) in enumerate(deps):
        t = f"a-{tag(slug)}"
        sid, pid = f"{t}-s", f"{t}-p"
        if sid in w and sid not in d:
            staging += 1
    for j, (slug, ev) in enumerate(deps):
        t = f"a-{tag(slug)}"
        sid, pid = f"{t}-s", f"{t}-p"
        failed = any(x in d and d[x].get("state") != "succeeded" for x in (sid, pid))
        st["failed"] += failed
        st["staged"] += d.get(sid, {}).get("state") == "succeeded"
        st["queued"] += pid in w
        st["done"] += d.get(pid, {}).get("state") == "succeeded"
        if j == 0:
            s0 = summary(f"sampled/{slug}/summary.json") if pid in d else None
            st["first"] = s0 and {k: s0.get(k) for k in ("run", "shapes", "proved", "statements", "accepted_statements", "units", "prove_total_s",
                                                          "job_s", "selftests", "selftests_pass", "gpu_cases", "gpu_cases_pass", "gpu_case_names")}
            gate = bool(s0 and s0.get("rc") == 0 and s0["selftests"] and s0["selftests_pass"] == s0["selftests"]
                        and s0["gpu_cases"] and s0["gpu_cases_pass"] == s0["gpu_cases"])
        elif not gate:
            continue
        if failed:
            continue
        if sid not in w and staging < ahead:
            todo.append((sid, shape_item(f"MODE=sampled-stage ROW={slug} EV={ev} JOBS=4", SAMPLED_STAGE_RES)))
            staging += 1
        elif d.get(sid, {}).get("state") == "succeeded" and pid not in w:
            env = f"MODE=sampled ROW={slug} EV={ev} SUMMARY={SWEEP}/sampled/{slug}/summary.json FLOCK_KEEP_SESSIONS=3"
            if j == 0:
                env += f" SELFTEST={SAMPLED_SELFTEST} SELFTEST_GPU=1"
            todo.append((pid, shape_item(env, SELFTEST_RES if j == 0 else PROVE_RES)))
    st["gate"] = gate
    return todo, st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tick", "loop", "status"])
    ap.add_argument("--row", action="append", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ahead", type=int, default=6, help="stage jobs written and not yet ended, per row")
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--chunk-from", action="append", default=[], metavar="ROW=N", help="that row's shapes below N are one-shape jobs")
    ap.add_argument("--wholerow", default=None, help="(b): a JSON file {row, shape, statements, chunk, parallel, extrap_s, tol, min_chunks}")
    ap.add_argument("--deployments", default=None, help="(a): a file of `<slug>=<evidence dir>` lines, the first selftested")
    a = ap.parse_args()
    chunk_from = {r: int(n) for r, _, n in (x.rpartition("=") for x in a.chunk_from)}
    while True:
        d, w = done(), written()
        todo, out = [], {}
        if a.wholerow:
            t, out["b"] = wholerow(json.loads(Path(a.wholerow).read_text()), d, w)
            todo += t
        if a.deployments:
            deps = [tuple(ln.split("=", 1)) for ln in Path(a.deployments).read_text().splitlines() if "=" in ln]
            t, out["a"] = sampled(deps, d, w)
            todo += t
        # (a) and (b) first; the shape sweep only fills in, one file a tick, while they have nothing to write
        if todo and a.cmd != "status" and clear_to_write():
            for item_id, it in todo:
                write(item_id, it)
        out["sweep"] = tick(a.row, a.limit, a.ahead, chunk_from, dry=a.cmd == "status" or bool(todo),
                            max_new=1 if (a.wholerow or a.deployments) else None)
        print(time.strftime("%H:%M:%SZ", time.gmtime()), json.dumps(out), flush=True)
        if a.cmd != "loop":
            return 0
        time.sleep(a.every)


if __name__ == "__main__":
    raise SystemExit(main())
