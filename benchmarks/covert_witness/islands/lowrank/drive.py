"""The lowrank direction's candidate queue: keeps at most `--max` candidates in flight through `launch.py`.

    python drive.py [--queue ~/.research/lowrank/v3/queue.txt] [--stop 2026-11-04T08:00Z] [--max 2]

Each queue line is `ISLAND START ; NOTE ; TRAIN_ARGS` (`#` comments; the note's first word is the mechanism tag). A line
launches once (keyed by its text), with `launch.py START --island ISLAND --note NOTE --node NODE --no-wait` from this
checkout, whose island must be clean (the run ships HEAD's), once START is in the starts_store.json of the pinned
commit launch.py ships (`pin.py`; a later start waits for it); NODE is whichever of `--nodes` holds fewer of the queue's
runs in flight, ties going to the one launched to longer ago. An ended run is never collected (only the scorer collects:
the coordinator's 06:57Z ruling (b)): it is fetched, its row read from `out/witness.json`, its points cached by
`curve.py`, and `research data custody RUN --publish` drops its local files of 1 MiB and more. Every launch and end
goes to `state.jsonl` beside the queue, from which a restarted driver resumes. No launch at or after `--stop`.

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
sys.path[:0] = [str(HERE), str(HERE.parents[1])]
import curve  # noqa: E402
import pin  # noqa: E402

REPO = HERE.parents[3]
LAUNCH = REPO / "benchmarks" / "covert_witness" / "launch.py"
RUNS = Path.home() / ".research" / "runs"
DONE = {"done", "failed", "cancelled", "refused", "timeout", "lost"}


def parse(line: str) -> tuple[str, str, str, list[str]] | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    # a note may hold `;`: the island and start end at the first, the train args start after the last
    head, rest = line.split(";", 1)
    note, args = rest.rsplit(";", 1)
    island, start = head.split()
    return island, start, note.strip(), shlex.split(args)


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def row(run: str, start: str, note: str) -> dict:
    """An ended run's result as its witness states it (none for a run that wrote no witness: its failure's reason)."""
    out = RUNS / run / "out"
    try:
        w = json.loads((out / "witness.json").read_text())
    except (OSError, ValueError):
        fail = RUNS / run / "failure.json"
        return {"run": run, "start": start, "note": note, "witness": None,
                "failure": fail.read_text()[-300:] if fail.exists() else None}
    return {"run": run, "start": start, "note": note, "valid": w.get("valid"), "bpb": (w.get("score") or {}).get("bpb"),
            "start_bpb": w.get("start_bpb"), "gpu_seconds": w.get("gpu_seconds"), "e_bytes": (w.get("E") or {}).get("bytes"),
            "failed_checks": [k for k, v in (w.get("checks") or {}).items() if not v]}


def pinned_starts() -> set[str]:
    """The starts the pinned measurement commit (the one launch.py ships) registers in its starts_store.json."""
    commit = pin.pinned(fetch=True)["commit"]
    r = pin.git("show", f"{commit}:benchmarks/covert_witness/starts_store.json")
    return set(json.loads(r.stdout)["starts"]) if r.returncode == 0 else set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--queue", type=Path, default=Path.home() / ".research" / "lowrank" / "v3" / "queue.txt")
    ap.add_argument("--stop", type=lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00")),
                    default=dt.datetime(2026, 11, 4, 8, 0, tzinfo=dt.timezone.utc))
    ap.add_argument("--max", type=int, default=2, help="candidates in flight at once")
    ap.add_argument("--nodes", default="vy-nebius-1,vy-nebius-3")
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
            _, start, note, _ = parse(key)
            res = row(s["run"], start, note)
            if res.get("witness", 0) is None and res["failure"] is None and tries.get(key, 0) < 19:
                # a fetch refused under the disk floor or cut short is retried on the next pass, its slot held
                tries[key] = tries.get(key, 0) + 1
                continue
            record({"key": key, "run": s["run"], "state": state, "row": res})
            print(f"{utc():%H:%M}Z ended {state}: {key}  {res}", flush=True)
            if curve.load(s["run"]):
                # the VM's disk is shared: once curve.py has cached its fields, drop the local copies the store holds
                research("data", "custody", s["run"], "--publish")
        flying = {k: s for k, s in flying.items()
                  if not any(json.loads(x).get("key") == k and "row" in json.loads(x)
                             for x in state_path.read_text().splitlines())}
        def clean(island: str) -> bool:
            return not subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--",
                                       f"benchmarks/covert_witness/islands/{island}"], capture_output=True, text=True,
                                      cwd=REPO).stdout.strip()

        lines = a.queue.read_text().splitlines() if a.queue.exists() else []
        starts = pinned_starts()
        for line in lines:
            key = line.strip()
            try:
                p = parse(line)
            except ValueError as e:
                print(f"{utc():%H:%M}Z skipped, unparsable ({e}): {key}", flush=True)
                continue
            if (p is None or key in seen or utc() >= a.stop or len(flying) >= a.max or not clean(p[0])
                    or p[1] not in starts):
                continue
            island, start, note, args = p
            load = {n: sum(1 for s in flying.values() if s.get("node") == n) for n in nodes}
            node = min(nodes, key=lambda n: (load[n], last.get(n, "")))
            r = subprocess.run([sys.executable, str(LAUNCH), start, "--island", island, "--note", note, "--node", node,
                                "--no-wait", "--", *args], capture_output=True, text=True, env=env, cwd=REPO)
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
            print(f"{utc():%H:%M}Z launched {island} {start} on {node} {run}: {note}", flush=True)
        if not flying and utc() >= a.stop:
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
