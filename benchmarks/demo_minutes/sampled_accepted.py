#!/usr/bin/env python3
"""The share of a served run's sampled proof blocks whose proofs the Lean verifier accepted on the host that composed
its record, read off its runs' records: the metric ``vllm/served-sampled-proofs-accepted`` (`benchmarks/demo_minutes/metrics.toml`).
Standard library only; finding the runs and the served commit, and what it writes, are `minutes.py`'s.

    research run --cwd source -- python3 benchmarks/demo_minutes/sampled_accepted.py SERVED_RUN VERDICT_RUN
        [--runs-dir DIR] [--main REF]

SERVED_RUN is the vLLM served run's (`adapters/vllm/tools/served_run/run.sh`), whose ``job.json`` names the served
commit; VERDICT_RUN is the one whose ``out/record/verdicts.json`` (`served_verdicts.py record`) holds SPOT's row, its
``protocols.integrity``. The sampled proof blocks are the members its Lean answer's profile sampled at the finest level,
the checked one (`Core.Integrity`). A proof block counts as accepted only where all of these hold, and the first that doesn't is the
measurement's ``reason``, its value then 0:

- the served commit is a commit of ``--main`` (default ``origin/main``) in the tree this file is in;
- SPOT's row was recomputed by the Lean verifier on the host that composed it (its ``answer_from``, `LEAN_HERE`);
- that verdict is accept, which the Lean verifier gives only when every sampled proof block's proof passes.

So a run's share is 100 or 0 today: the answer holds one verdict over every sampled proof block, and a rejection names no count.
It refuses, with no measurement and the reason (exit 1), a run without its record, a VERDICT_RUN with no
``out/record/verdicts.json``, a record whose SPOT row is missing or empty or with no ``composed_on.served_node``, and an accepting answer that
names no sampled proof block: a partial record is no measurement, where a complete one that fails a condition measures 0.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

METRIC = "vllm/served-sampled-proofs-accepted"
TARGET = 100
UNIT = "percent of units"
RECORD = Path("out") / "record" / "verdicts.json"
#: the head of SPOT's ``answer_from`` where the composing host's Lean verifier recomputed it (served_verdicts.LEAN_HERE)
LEAN_HERE = "recomputed on this host by the Lean verifier"
HERE = Path(__file__).resolve().parent


def _reader():
    """The remote demo's reader, loaded by its path under a name of its own, with its directory first on the path for
    the `settings` it imports."""
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("remote_demo_minutes", HERE / "minutes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _reader()


def sampled(answer: dict) -> int | None:
    """The members ``answer``'s profile sampled at its finest level, None where it names none."""
    p = answer.get("profile") if isinstance(answer, dict) else None
    levels = (p or {}).get("levels") or []
    if not levels:
        return None
    n = ((p or {}).get("drawn") or {}).get(levels[-1].get("name"))
    return n if isinstance(n, int) and not isinstance(n, bool) else None


def off_main(sha: str, ref: str, repo: Path) -> bool:
    p = subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, ref], capture_output=True, text=True,
                       check=False)
    if p.returncode not in (0, 1):
        raise R.Refused(f"can't tell whether the served commit {sha} is a commit of {ref}: {p.stderr.strip()}")
    return p.returncode == 1


def share(runs: list[str], root: Path, ref: str, repo: Path | None = None) -> dict:
    """The measurement for ``runs`` (SERVED_RUN, VERDICT_RUN); `Refused` where it can't stand."""
    served, verdicts = (R.run_dir(r, root) for r in runs)
    sha = R.commit(served)
    f = verdicts / RECORD
    if not f.is_file():
        raise R.Refused(f"{verdicts.name}: no {RECORD}, the record of its verdicts")
    report = json.loads(f.read_text())
    row = (report.get("protocols") or {}).get("integrity")
    if not isinstance(row, dict) or not row:
        raise R.Refused(f"{verdicts.name}: its {RECORD} holds no SPOT row (protocols.integrity)")
    on = report.get("composed_on")
    if not isinstance(on, dict) or not isinstance(on.get("served_node"), bool):
        raise R.Refused(f"{verdicts.name}: its {RECORD} names no composed_on.served_node, so where it was composed is unknown")
    n = sampled(row.get("answer") or {})
    reasons = [
        off_main(sha, ref, repo or R.ROOT) and f"served from {sha}, not a commit of {ref}",
        not str(row.get("answer_from") or "").startswith(LEAN_HERE)
        and f"SPOT's verdict was not recomputed there by the Lean verifier: {row.get('answer_from') or 'no answer_from'}",
        row.get("verdict") != "accept" and f"the Lean verifier's verdict is {row.get('verdict')}",
    ]
    reason = next((r for r in reasons if r), None)
    if reason is None and not n:
        raise R.Refused(f"{verdicts.name}: SPOT's accepting answer names no sampled proof block at its finest level")
    accepted = n if reason is None else 0
    return {"name": METRIC, "value": 100 * accepted / n if n else 0, "unit": UNIT, "mark": "measured",
            "better": "higher", "target": TARGET, "verdict": row.get("verdict"), "sampled": n, "accepted": accepted,
            "reason": reason, "runs": [served.name, verdicts.name], "commit": sha,
            "inputs": [f"measured: SPOT's row in {verdicts.name}'s {RECORD}, of the run {served.name} served"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs=2, metavar="RUN", help="SERVED_RUN VERDICT_RUN (module docstring)")
    ap.add_argument("--runs-dir", type=Path, default=R.settings.runs_root())
    ap.add_argument("--main", default="origin/main", help="the ref the served commit must be a commit of")
    a = ap.parse_args(argv)
    try:
        m, why = share(a.runs, a.runs_dir, a.main), None
    except R.Refused as e:
        m, why = None, e
    print(f"ACCEPTED {json.dumps(m, sort_keys=True)}" if m else f"ACCEPTED refused: {why}")
    if d := R.settings.run_dir():
        doc = R.result(R.settings.run_id() or d.name, a.runs, m, why, METRIC, "benchmarks/demo_minutes/sampled_accepted.py")
        if m:
            doc["validation"]["detail"] = (f"{m['accepted']} of {m['sampled'] or 'no'} sampled proof blocks accepted"
                                           + (f": {m['reason']}" if m["reason"] else ""))
        f = d / "result.json"
        tmp = f.with_name(f"{f.name}.tmp.{os.getpid()}")
        tmp.write_text(json.dumps(doc, indent=1) + "\n")
        os.replace(tmp, f)
    return 0 if m else 1


if __name__ == "__main__":
    sys.exit(main())
