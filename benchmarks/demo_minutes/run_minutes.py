"""The minutes between research runs' records, from the first run's start to the last run's end, written under a metric
where the last run's verdict accepts. Standard library only: the onsite reader (`onsite_minutes.py`) runs it with its
verdict (``read``) and its metric.

Each run is a run id under ``--runs-dir`` (default ``$RESEARCH_RUNS``, else ``~/.research/runs``), where ``research fetch
RUN --all`` copies a run's records and its ``out/``, or a run directory. The minutes run from the first run's
``started_utc`` to the last's ``ended_utc``, each from its run's ``status.json``; nothing reads a clock.

It refuses, with no measurement and the reason (exit 1):

- a run without its record, the first without ``started_utc``, or the last without ``ended_utc``;
- what ``read`` refuses in the last run;
- a served commit (the first run's ``computation.tree_sha``, in its ``job.json``) that is not a commit of ``--main``
  (default ``origin/main``) in the tree this file is in;
- a verdict other than accept: its line gives the verdict and the minutes for the log, and the result holds no value.

Under ``research run`` it writes its own run's ``result.json``: the minutes under the metric, with the verdict, the runs
and the served commit beside them, or no measurement and the refusal as its validation's detail. It prints the same
as one ``MINUTES`` line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import settings

TARGET = 60
ROOT = Path(__file__).resolve().parents[2]
RESULT = "research/result/v0.1"


class Refused(Exception):
    status = "not_run"


class Rejected(Refused):
    status = "failed"


def run_dir(run: str, root: Path) -> Path:
    d = Path(run) if Path(run).is_dir() else root / run
    if not (d / "status.json").is_file():
        raise Refused(f"{run}: no run record (status.json) under {d}; `research fetch {run} --all` copies it")
    return d


def utc(d: Path, key: str) -> datetime:
    t = json.loads((d / "status.json").read_text()).get(key)
    if not t:
        raise Refused(f"{d.name}: its status.json has no {key}")
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def commit(d: Path) -> str:
    job = d / "job.json"
    sha = (json.loads(job.read_text()).get("computation") or {}).get("tree_sha") if job.is_file() else None
    if not sha:
        raise Refused(f"{d.name}: its job.json names no served commit (computation.tree_sha)")
    return sha


def on_main(sha: str, ref: str, repo: Path) -> None:
    p = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, ref], capture_output=True, text=True,
                       check=False)
    if p.returncode == 1:
        raise Refused(f"the served commit {sha} is not a commit of {ref}")
    if p.returncode:
        raise Refused(f"can't tell whether the served commit {sha} is a commit of {ref}: {p.stderr.strip()}")


def minutes(runs: list[str], root: Path, ref: str, repo: Path | None = None, *, read: Callable[[Path], str],
            metric: str) -> dict:
    """The measurement for ``runs``, first to last, whose last run's verdict ``read`` gives; `Refused` where it can't
    stand. ``repo`` is the tree holding ``ref`` (default this file's)."""
    ds = [run_dir(r, root) for r in runs]
    start, end = utc(ds[0], "started_utc"), utc(ds[-1], "ended_utc")
    if end < start:
        raise Refused(f"{ds[-1].name} ended before {ds[0].name} started: the runs are not in the demo's order")
    v, sha = read(ds[-1]), commit(ds[0])
    on_main(sha, ref, repo or ROOT)
    m = {"name": metric, "value": round((end - start).total_seconds() / 60, 1), "unit": "minutes", "better": "lower",
         "target": TARGET, "verdict": v, "runs": [d.name for d in ds], "commit": sha}
    if v != "accept":
        raise Rejected(f"{ds[-1].name}: the verdict is {v}, after {m['value']} minutes; only an accepting demo has "
                       f"minutes")
    return m


def result(run_id: str, runs: list[str], m: dict | None, why: Refused | None, metric: str, checker: str) -> dict:
    return {"schema": RESULT, "run_id": run_id, "workload_fingerprint": {"metric": metric, "runs": runs},
            "validation": {"status": why.status if why else "passed", "checker": checker,
                           "detail": str(why) if why else f"{m['value']} minutes to a verdict of {m['verdict']}",
                           "evidence": []},
            "measurements": [m] if m else [], "measurement_files": [], "artifacts": []}


def main(argv: list[str] | None = None, *, read: Callable[[Path], str], metric: str, checker: str,
         nargs: str | int = "+", doc: str = __doc__) -> int:
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    ap.add_argument("runs", nargs=nargs, metavar="RUN", help="the runs, first to last (module docstring)")
    ap.add_argument("--runs-dir", type=Path, default=settings.runs_root())
    ap.add_argument("--main", default="origin/main", help="the ref the served commit must be a commit of")
    a = ap.parse_args(argv)
    try:
        m, why = minutes(a.runs, a.runs_dir, a.main, read=read, metric=metric), None
    except Refused as e:
        m, why = None, e
    print(f"MINUTES {json.dumps(m, sort_keys=True)}" if m else f"MINUTES refused: {why}")
    if d := settings.run_dir():
        f = d / "result.json"
        tmp = f.with_name(f"{f.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(result(settings.run_id() or d.name, a.runs, m, why, metric, checker), indent=1) + "\n")
        os.replace(tmp, f)
    return 0 if m else 1
