"""The lowrank direction's candidate queue: keeps at most two candidates per island in flight through `launch.py`.

    python drive.py [--queue ~/.research/lowrank/queue.txt]

Each queue line is `CLASS START ; NOTE ; TRAIN_ARGS` (`#` comments). A line launches once (keyed by its text), with
`launch.py --no-wait` from this checkout, which must be clean (the run ships HEAD); an ended run is collected with
`launch.py --collect`. Every launch and collection goes to `state.jsonl` beside the queue, from which a restarted driver
resumes. No launch after 14:30Z. The nodes are `$CW_NODES`'s (default the search's shared /tmp/te-nodes.txt).

A tool of the agent VM, never run by the harness: only `islands/CLASS/train.py` is a candidate program.
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
REPO = HERE.parents[3]
LAUNCH = REPO / "benchmarks" / "covert_witness" / "launch.py"
RUNS = Path.home() / ".research" / "runs"
DONE = {"done", "failed", "cancelled", "refused", "timeout", "lost"}
STOP = dt.time(14, 30)
PER_CLASS = 2


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
    ap.add_argument("--queue", type=Path, default=Path.home() / ".research" / "lowrank" / "queue.txt")
    a = ap.parse_args()
    state_path = a.queue.parent / "state.jsonl"
    # the research CLI from this checkout's tools (`bin/research`): /workspace's may predate the node's own pools
    env = dict(os.environ, PATH=f"{a.queue.parent / 'bin'}:/workspace/.venv/bin:{os.environ['PATH']}",
               CW_NODES=os.environ.get("CW_NODES", "/tmp/te-nodes.txt"))

    def record(rec: dict) -> None:
        with state_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def research(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["research", *args], capture_output=True, text=True, env=env, cwd=REPO)

    while True:
        seen, flying = set(), {}
        for x in state_path.read_text().splitlines() if state_path.exists() else []:
            s = json.loads(x)
            seen.add(s["key"])
            if s.get("run") and "row" not in s:
                flying[s["key"]] = s
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
            record({"key": key, "run": s["run"], "state": state, "row": rows[-1] if rows else None,
                    "collect_rc": r.returncode, "err": r.stderr[-500:] if not rows else ""})
            print(f"{utc():%H:%M}Z ended {state}: {key}  {rows[-1] if rows else r.stderr[-300:]}", flush=True)
        clean = not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                                   text=True, cwd=REPO).stdout.strip()
        lines = a.queue.read_text().splitlines() if a.queue.exists() else []
        for line in lines:
            key = line.strip()
            try:
                p = parse(line)
            except ValueError as e:
                print(f"{utc():%H:%M}Z skipped, unparsable ({e}): {key}", flush=True)
                continue
            if p is None or key in seen or utc().time() >= STOP or not clean:
                continue
            klass, start, note, args = p
            if sum(1 for k in flying if parse(k)[0] == klass) >= PER_CLASS:
                continue
            r = subprocess.run([sys.executable, str(LAUNCH), klass, start, "--note", note, "--no-wait", "--", *args],
                               capture_output=True, text=True, env=env, cwd=REPO)
            rows = [json.loads(y) for y in r.stdout.splitlines() if y.startswith("{")]
            run = rows[0].get("run") if rows else None
            if not run:
                # a refused launch (the VM's disk floor, the node unreachable) is retried on the next pass
                print(f"{utc():%H:%M}Z launch refused, retrying: {key}\n  {(r.stdout + r.stderr)[-300:]}", flush=True)
                break
            rec = {"key": key, "launched": f"{utc():%H:%M:%S}Z", "run": run}
            record(rec)
            seen.add(key)
            flying[key] = rec
            print(f"{utc():%H:%M}Z launched {klass} {start} {run}: {note}", flush=True)
        if not flying and utc().time() >= STOP:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
