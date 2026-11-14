#!/usr/bin/env python3
"""The remote demo's minutes from its first command to its verdict, read off its finished runs' records: the metric
``vllm/remote-demo-minutes`` (`demos/vllm/metrics.toml`). Standard library only; the onsite demo's reader
(`demos/onsite_served/minutes.py`) is this one with its own verdict.

    research run --cwd source -- python3 demos/vllm/minutes.py FIRST_RUN [RUN ...] VERDICT_RUN [--runs-dir DIR] [--main REF]

The arguments are the demo's research runs in the order it ran them: the first is `verity demo vllm`'s, the last the one
that composed remote's verdict on the auditor's machine (``served_verdicts.sh compose``). Each is a run id under
``--runs-dir`` (default ``$RESEARCH_RUNS``, else ``~/.research/runs``), where ``research fetch RUN --all`` copies a run's
records and its ``out/``, or a run directory. The minutes run from the first run's ``started_utc`` to the last's
``ended_utc``, each from its run's ``status.json``; nothing reads a clock.

It refuses, with no measurement and the reason (exit 1):

- a run without its record, the first without ``started_utc``, or the last without ``ended_utc``;
- no verdict: the last run has no ``out/composite/remote-v1/composite.json``, or that holds no verdict;
- a verdict covering fewer checks than the goal names (`CHECKS`: SPOT, Pearl+ and inference only), or one
  composed as a development check, on the served node;
- a served commit (the first run's ``computation.tree_sha``, in its ``job.json``) that is not a commit of ``--main``
  (default ``origin/main``) in the tree this file is in;
- a verdict other than accept: the demo's minutes are those of a demo that ended accepting, so its line gives the verdict
  and the minutes for the log, and the result holds no value.

Under ``research run`` it writes its own run's ``result.json``: the minutes under `METRIC`, with the verdict, the runs
and the served commit beside them, or no measurement and the refusal as its validation's detail. It prints the same
as one ``MINUTES`` line.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import settings

METRIC = "vllm/remote-demo-minutes"
#: remote's composite of record, whose verdict is the demo's, as `served_composite.sh` writes it under the run's out/
PRODUCT = "remote-v1"
#: the three checks the goal names, each a subprotocol of `PRODUCT`
CHECKS = ("verity/inference-only/v0", "verity/pouw/pearl-plus/v1", "verity/sampled-proofs/v1")
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


def verdict(d: Path) -> str:
    """Remote's verdict in run ``d``'s composite of `PRODUCT`; `Refused` where it is none of the goal's."""
    f = d / "out" / "composite" / PRODUCT / "composite.json"
    if not f.is_file():
        raise Refused(f"{d.name}: no verdict line, since its out/ holds no composite/{PRODUCT}/composite.json")
    c = json.loads(f.read_text())
    if not (c.get("verdict") or {}).get("verdict"):
        raise Refused(f"{d.name}: its composite of {PRODUCT} holds no verdict")
    if c.get("development"):
        raise Refused(f"{d.name}: its composite is a development check, composed on the served node: {c['development']}")
    have = {x.get("id") for x in c.get("cases") or ()}
    if gone := [i for i in CHECKS if i not in have]:
        raise Refused(f"{d.name}: its verdict covers {len(CHECKS) - len(gone)} of the goal's {len(CHECKS)} checks, "
                      f"not {', '.join(gone)}")
    return c["verdict"]["verdict"]


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


def minutes(runs: list[str], root: Path, ref: str, repo: Path | None = None, *,
            read: Callable[[Path], str] = verdict, metric: str = METRIC) -> dict:
    """The measurement for the demo's ``runs``, first to last, whose last run's verdict ``read`` gives; `Refused` where
    it can't stand. ``repo`` is the tree holding ``ref`` (default this file's)."""
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


def main(argv: list[str] | None = None, *, read: Callable[[Path], str] = verdict, metric: str = METRIC,
         checker: str = "demos/vllm/minutes.py", nargs: str | int = "+", doc: str = __doc__) -> int:
    ap = argparse.ArgumentParser(description=doc.split("\n\n")[0])
    ap.add_argument("runs", nargs=nargs, metavar="RUN", help="the demo's runs, first to last (module docstring)")
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


if __name__ == "__main__":
    sys.exit(main())
