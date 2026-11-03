"""One commit's executor shadow by hand, as drive's `executor = "shadow"` records it (`research.drive.Driver._shadow_tip`):
the commit's after-landing check enqueued on infra's executor with no graph, the graph it lists read back, every task's
verdict waited for, and `research.executor.compare` against the old after-landing run's shards written as the
`executor_shadow` label on `commit:<tip>`, which `research queue shadow-report` reads.

    PYTHONPATH=<a tree with research.executor>/tools/research/src \\
      python3 shadow_by_hand.py --tip SHA [--runs RID...] --enqueue "python3.12 .../executor.py --db .../queue.db enqueue" --state DIR

Without --runs it stops once the executor's side is done; run it again with the old run's shard ids to compare (the
enqueue is idempotent, so the second call finds the graph and its verdicts at once). Each shard must have ended and be of
the tip (`research status`'s tree_sha); its step, suite and lean-audit records are fetched from its node first, since an
ended run is otherwise read from the copy here, which may hold its steps alone.

DIR keeps old.json (the run's verdicts by part), graph.json, verdicts.json, progress.log and result.json."""
from __future__ import annotations

import argparse
import json
import shlex
import time
from collections import Counter
from pathlib import Path

from research.drive import PROGRESS, SPANS, evidence
from research.executor import AFTER_LANDING_PRIORITY, PENDING, SHADOW_WAIT_S, Executor, QueueTransport, compare, old_verdicts
from research.merge import fetch_run
from research.store import vocab
from research.store.cli import _label_write_through
from research.store.local import LocalStore

#: what `old_verdicts` reads of a shard: its steps, its suite reports and the lean-audit step's log (the AUDIT lines)
OLD_RECORDS = (*PROGRESS, SPANS, "suites.json", "suites-lean.json", "lean-audit.log")


def old_side(tip: str, runs: list[str]) -> dict[str, str]:
    for rid in runs:
        if (got := fetch_run(rid, list(OLD_RECORDS), Path.cwd(), refresh=True)) is None:
            raise SystemExit(f"{rid}: `research status` can't read it")
        row = got[1]
        if not row.get("ended_utc"):
            raise SystemExit(f"{rid} hasn't ended ({row.get('state')}): compare once every shard has")
        if row.get("tree_sha") and row["tree_sha"] != tip:
            raise SystemExit(f"{rid} checked {row['tree_sha']}, not {tip}")
    if len(dirs := evidence(Path.cwd(), runs)) != len(runs):
        raise SystemExit(f"read {len(dirs)} of the {len(runs)} runs' records: {[str(d) for d in dirs]}")
    return old_verdicts(dirs)


def main() -> int:
    ap = argparse.ArgumentParser(prog="shadow_by_hand")
    ap.add_argument("--tip", required=True, help="the commit's full sha")
    ap.add_argument("--runs", nargs="*", default=[], help="the old after-landing run's shard run ids (none: the executor's side only)")
    ap.add_argument("--enqueue", required=True, help="infra's enqueue command, as drive.sh's RESEARCH_EXECUTOR_ENQUEUE")
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--every", type=float, default=120.0, help="seconds between polls")
    ap.add_argument("--wait", type=float, default=SHADOW_WAIT_S, help="seconds after the enqueue before pending is recorded")
    ap.add_argument("--by", default="top")
    ap.add_argument("--no-label", action="store_true", help="write result.json only")
    a = ap.parse_args()
    a.state.mkdir(parents=True, exist_ok=True)

    def say(s: str) -> None:
        print(s, flush=True)
        with (a.state / "progress.log").open("a") as f:
            f.write(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {s}\n")

    store, check, t0 = LocalStore(), f"after-landing:{a.tip}", time.time()
    ex = Executor(QueueTransport(store, tuple(shlex.split(a.enqueue))))
    ex.enqueue_check(check, AFTER_LANDING_PRIORITY)
    say(f"enqueued {check} at priority {AFTER_LANDING_PRIORITY}")
    while (got := ex.graph(check)) is None:
        if time.time() - t0 > a.wait:
            raise SystemExit(f"no {vocab.TASKS_GRAPH} label on commit:{a.tip} {a.wait:.0f} s after the enqueue")
        time.sleep(a.every)
    gid, keys = got
    (a.state / "graph.json").write_text(json.dumps({"graph": gid, "keys": keys}, indent=1))
    say(f"graph {gid}: {len(keys)} tasks")
    while True:
        v = ex.verdicts(keys.values())
        (a.state / "verdicts.json").write_text(json.dumps({t: v[k] for t, k in keys.items()}, indent=1, sort_keys=True))
        say(f"verdicts: {dict(Counter(v.values()))}")
        if PENDING not in v.values() or time.time() - t0 > a.wait:
            break
        time.sleep(a.every)
    if not a.runs:
        say("the executor's side is done; run again with --runs to compare")
        return 0
    old = old_side(a.tip, a.runs)
    (a.state / "old.json").write_text(json.dumps(old, indent=1, sort_keys=True))
    say(f"old: {len(old)} parts from {' '.join(a.runs)}: {dict(Counter(old.values()))}")
    value = {"check": check, "graph": gid, "run": " ".join(a.runs), **compare(keys, old, v)}
    (a.state / "result.json").write_text(json.dumps(value, indent=1))
    said = "not labelled (--no-label)"
    if not a.no_label:
        store.add_label(f"commit:{a.tip}", vocab.EXECUTOR_SHADOW, value, a.by, ref=value["run"])
        said, _ = _label_write_through(store, f"commit:{a.tip}")
    say(f"{value['agree']}/{value['units']} agree, {len(value['disagree'])} disagree, {len(value['pending'])} pending, "
        f"{len(value['missing'])} missing in executor, {len(value['executor_only'])} executor only ({said})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
