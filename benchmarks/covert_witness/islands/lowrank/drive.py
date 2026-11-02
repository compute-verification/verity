"""The lowrank direction's candidate queue: keeps at most two candidates per island in flight through `launch.py`.

    python drive.py [--queue ~/.research/lowrank/queue.txt] [--launch-wt DIR] [--branch cursor/te-lowrank-4207]

Each queue line is `CLASS START ; NOTE ; TRAIN_ARGS` (`#` comments). A line launches once (keyed by its text); before each
launch the launch worktree is moved to the branch's tip, so a run ships the newest commit. Rows go to `state.jsonl` beside
the queue (and to launch.py's ledger). No launch after 14:30Z.
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
STOP = dt.time(14, 30)
PER_CLASS = 2


def parse(line: str) -> tuple[str, str, str, list[str]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    head, note, args = (x.strip() for x in line.split(";", 2))
    klass, start = head.split()
    return klass, start, note, shlex.split(args)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", type=Path, default=Path.home() / ".research" / "lowrank" / "queue.txt")
    ap.add_argument("--launch-wt", type=Path, default=Path.home() / "wt-te-lowrank-launch")
    ap.add_argument("--branch", default="cursor/te-lowrank-4207")
    a = ap.parse_args()
    state_path = a.queue.parent / "state.jsonl"
    logs = a.queue.parent / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["key"] for x in state_path.read_text().splitlines() if x.strip()} if state_path.exists() else set()
    flying: dict[str, tuple[subprocess.Popen, str, Path]] = {}
    # the research CLI from the launch worktree's tools: /workspace's may predate the node's own pools (`--pool gpu`)
    env = dict(os.environ, PATH=f"{a.queue.parent / 'bin'}:/workspace/.venv/bin:{os.environ['PATH']}")
    while True:
        for key, (proc, klass, log) in list(flying.items()):
            if proc.poll() is None:
                continue
            rows = [json.loads(x) for x in log.read_text().splitlines() if x.startswith("{")]
            with state_path.open("a") as f:
                f.write(json.dumps({"key": key, "rc": proc.returncode, "row": rows[-1] if rows else None,
                                    "run": rows[0].get("run") if rows else None}) + "\n")
            print(f"{dt.datetime.utcnow():%H:%M}Z ended rc {proc.returncode}: {key}  {rows[-1] if rows else ''}", flush=True)
            del flying[key]
        lines = [x for x in a.queue.read_text().splitlines()] if a.queue.exists() else []
        now = dt.datetime.utcnow()
        for line in lines:
            p = parse(line)
            key = line.strip()
            if p is None or key in done or key in flying or now.time() >= STOP:
                continue
            klass, start, note, args = p
            if sum(1 for _, k, _ in flying.values() if k == klass) >= PER_CLASS:
                continue
            subprocess.run(["git", "-C", str(a.launch_wt), "checkout", "-q", "--detach", a.branch], check=True)
            log = logs / f"{now:%H%M%S}-{klass}-{start}-{len(done) + len(flying)}.log"
            cmd = [sys.executable, str(a.launch_wt / "benchmarks/covert_witness/launch.py"), klass, start, "--note", note,
                   "--", *args]
            flying[key] = (subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=env), klass, log)
            done.add(key)
            with state_path.open("a") as f:
                f.write(json.dumps({"key": key, "launched": f"{now:%H:%M:%S}Z", "log": str(log)}) + "\n")
            print(f"{now:%H:%M}Z launched {klass} {start}: {note} :: {' '.join(args)}", flush=True)
            time.sleep(5)
        if not flying and now.time() >= STOP:
            return 0
        time.sleep(20)


if __name__ == "__main__":
    sys.exit(main())
