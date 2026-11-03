"""The lowrank direction's candidate queue: keeps at most `--max` candidates in flight through `launch.py`.

    python drive.py [--queue ~/.research/lowrank/night2/queue.txt] [--stop 2026-11-03T12:30Z] [--max 2]

Each queue line is `CLASS START ; NOTE ; TRAIN_ARGS` (`#` comments). A line launches once (keyed by its text), with
`launch.py --no-wait --island ISLAND --node NODE` from this checkout, whose island must be clean (the run ships HEAD's);
`--class CLASS` is added to TRAIN_ARGS when they lack it, and NODE is whichever of `--nodes` holds fewer of the queue's
runs in flight, ties going to the one launched to longer ago. An ended run is collected with `launch.py --collect`,
retried on the next passes (its slot held) when that yields no row, up to twenty tries; once collected, its points are
cached by `curve.py` and `research data custody RUN --publish` drops its local files of 1 MiB and more. Every
launch and collection goes to `state.jsonl` beside the queue, from which a restarted driver resumes. No launch at or
after `--stop`.

A tool of the agent VM, never run by the harness: only an island's `train.py` is a candidate program.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import curve  # noqa: E402

REPO = HERE.parents[3]
LAUNCH = REPO / "benchmarks" / "covert_witness" / "launch.py"
RUNS = Path.home() / ".research" / "runs"
DONE = {"done", "failed", "cancelled", "refused", "timeout", "lost"}


def parse(line: str) -> tuple[str, str, str, list[str]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # a note may hold `;`: the class and start end at the first, the train args start after the last
    head, rest = line.split(";", 1)
    note, args = rest.rsplit(";", 1)
    klass, start = head.split()
    return klass, start, note.strip(), shlex.split(args)


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", type=Path, default=Path.home() / ".research" / "lowrank" / "night2" / "queue.txt")
    ap.add_argument("--stop", type=lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")),
                    default=dt.datetime(2026, 11, 3, 12, 30, tzinfo=dt.timezone.utc))
    ap.add_argument("--max", type=int, default=2, help="candidates in flight at once")
    ap.add_argument("--nodes", default="vy-nebius-1,vy-nebius-3")
    ap.add_argument("--island", default="e1mb")
    a = ap.parse_args()
    nodes = a.nodes.split(",")
    state_path = a.queue.parent / "state.jsonl"
    # the research CLI from the lane's own bin (`~/.research/lowrank/bin/research`), else the workspace's
    env = dict(os.environ, PATH=f"{Path.home() / '.research' / 'lowrank' / 'bin'}:/workspace/.venv/bin:{os.environ['PATH']}")

    def record(rec: dict) -> None:
        with state_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def research(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["research", *args], capture_output=True, text=True, env=env, cwd=REPO)

    tries: dict[str, int] = {}
    while True:
        seen, flying, last = set(), {}, {n: "" for n in nodes}
        for x in state_path.read_text().splitlines() if state_path.exists() else []:
            s = json.loads(x)
            seen.add(s["key"])
            if s.get("run") and "row" not in s:
                flying[s["key"]] = s
                last[s.get("node", "")] = max(last.get(s.get("node", ""), ""), s["launched"])
            elif "row" in s:
                flying.pop(s["key"], None)
        for key, s in flying.items():
            research("fetch", s["run"])
            st = RUNS / s["run"] / "status.json"
            state = json.loads(st.read_text()).get("state") if st.exists() else None
            if state not in DONE:
                continue
            klass, start, note, _ = parse(key)
            r = subprocess.run([sys.executable, str(LAUNCH), klass, start, "--note", note, "--collect", s["run"]],
                               capture_output=True, text=True, env=env, cwd=REPO)
            rows = [json.loads(y) for y in r.stdout.splitlines() if y.startswith("{")]
            if not rows and tries.get(key, 0) < 19:
                # a fetch that fails on a busy machine or a full disk is retried on the next pass
                tries[key] = tries.get(key, 0) + 1
                print(f"{utc():%H:%M}Z collect failed ({tries[key]}), retrying: {s['run']}  {r.stderr[-200:]}", flush=True)
                continue
            record({"key": key, "run": s["run"], "state": state, "row": rows[-1] if rows else None,
                    "collect_rc": r.returncode, "err": r.stderr[-500:] if not rows else ""})
            print(f"{utc():%H:%M}Z ended {state}: {key}  {rows[-1] if rows else r.stderr[-300:]}", flush=True)
            if rows and curve.load(s["run"]):
                # the VM's disk is shared: once curve.py has cached its fields, drop the local copies the store holds
                research("data", "custody", s["run"], "--publish")
        flying = {k: s for k, s in flying.items()
                  if not any(json.loads(x).get("key") == k and "row" in json.loads(x)
                             for x in state_path.read_text().splitlines())}
        clean = not subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                                    f"benchmarks/covert_witness/islands/{a.island}"], capture_output=True, text=True,
                                   cwd=REPO).stdout.strip()
        lines = a.queue.read_text().splitlines() if a.queue.exists() else []
        for line in lines:
            key = line.strip()
            try:
                p = parse(line)
            except ValueError as e:
                print(f"{utc():%H:%M}Z skipped, unparsable ({e}): {key}", flush=True)
                continue
            if p is None or key in seen or utc() >= a.stop or not clean or len(flying) >= a.max:
                continue
            klass, start, note, args = p
            if "--class" not in args:
                args = [*args, "--class", klass]
            load = {n: sum(1 for s in flying.values() if s.get("node") == n) for n in nodes}
            node = min(nodes, key=lambda n: (load[n], last.get(n, "")))
            r = subprocess.run([sys.executable, str(LAUNCH), klass, start, "--note", note, "--no-wait", "--island",
                                a.island, "--node", node, "--", *args], capture_output=True, text=True, env=env, cwd=REPO)
            rows = [json.loads(y) for y in r.stdout.splitlines() if y.startswith("{")]
            run = rows[0].get("run") if rows else None
            if not run:
                # a refused launch (the VM's disk floor, the node unreachable) is retried on the next pass
                print(f"{utc():%H:%M}Z launch refused on {node}, retrying: {key}\n  {(r.stdout + r.stderr)[-300:]}",
                      flush=True)
                break
            rec = {"key": key, "launched": f"{utc():%Y-%m-%dT%H:%M:%S}Z", "run": run, "node": node}
            record(rec)
            seen.add(key)
            flying[key] = rec
            last[node] = rec["launched"]
            print(f"{utc():%H:%M}Z launched {klass} {start} on {node} {run}: {note}", flush=True)
        if not flying and utc() >= a.stop:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
