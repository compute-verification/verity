"""Overnight watcher: re-run the validation panel whenever a bound module changes and post violations.

Polls the sha1 of ``bounds/lower_coarse.py``, ``bounds/lower.py``, ``bounds/coarse.py``, ``bounds/upper.py``; when a
hash differs from the last checked set and has been stable for ``--settle`` seconds, runs

1. ``redteam.rollout_scale --reuse-brackets`` (registry dense + MoE rollout cells; brackets reused, L/U recomputed),
2. ``redteam.block_term`` (registry local-sgd / forward-nonfixed cells),
3. ``redteam.coarse_validation --reuse-istar`` (the exact panel; folds 1-2 in),

then writes ``results/REDTEAM_STATUS.md`` (top line: hashes checked, ``U < I*``, ``L > I*``, bracket violations) and,
if any soundness violation appeared, appends the cells to ``results/REDTEAM_ALERTS.md`` -- the file the lower-bound
(A) and planner (B) agents should watch.  Runs for ``--hours`` then exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
WATCHED = ("bounds/lower_coarse.py", "bounds/lower.py", "bounds/coarse.py", "bounds/upper.py")


def hashes() -> dict[str, str]:
    out = {}
    for rel in WATCHED:
        p = os.path.join(HERE, rel)
        out[rel] = hashlib.sha1(open(p, "rb").read()).hexdigest()[:10] if os.path.exists(p) else "absent"
    return out


def run(mod: str, args: list[str], log) -> int:
    cmd = [sys.executable, "-m", mod] + args
    log(f"$ {' '.join(cmd)}")
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, env={**os.environ, "PYTHONPATH": "."}, capture_output=True, text=True)
    log(f"  -> exit {p.returncode} in {time.time() - t0:.0f}s; {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''}")
    if p.returncode != 0:
        log("  stderr: " + p.stderr.strip()[-1500:])
    return p.returncode


def _load(name: str) -> dict:
    try:
        with open(os.path.join(RESULTS, name)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def summarize(h: dict[str, str]) -> tuple[str, list[str]]:
    """Status line + list of violation descriptions across the three result files."""
    viol: list[str] = []
    v = _load("validation.json")
    pn = v.get("panel") or {}
    rs = _load("rollout_scale.json")
    bt = _load("block_term.json")
    for r in v.get("rows", []):
        for x in r.get("violations") or []:
            if x.startswith("L_gt") or x.startswith("U_lt") or x == "U_illegal":
                viol.append(f"[panel] {r.get('circuit')} F={r.get('F')} G={r.get('G')} I*={r.get('Istar')} L={r.get('L')} U={r.get('U')}: {x}")
    for tag, doc in (("rollout", rs), ("block_term", bt)):
        for r in doc.get("rows", []):
            for x in r.get("violations") or []:
                viol.append(f"[{tag}] {r.get('circuit')} {r.get('regime')} F={r.get('F')} G={r.get('G')} [{r.get('I_lo')}, {r.get('I_hi')}] "
                            f"L={r.get('L')} U={r.get('U')}: {x}")
    line = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} | hashes lower_coarse `{h['bounds/lower_coarse.py']}` lower `{h['bounds/lower.py']}` "
            f"coarse `{h['bounds/coarse.py']}` upper `{h['bounds/upper.py']}` | exact panel N={pn.get('N_exact')} "
            f"U<I*={pn.get('U_lt_Istar_count')} L>I*={pn.get('L_gt_Istar_count')} | rollout cells {len(rs.get('rows', []))} "
            f"viol {sum(1 for r in rs.get('rows', []) if r.get('violations'))} | block-term cells {len(bt.get('rows', []))} "
            f"viol {sum(1 for r in bt.get('rows', []) if r.get('violations'))} | panel lower_coarse hash `{(v.get('meta') or {}).get('lower_coarse_sha1')}`")
    return line, viol


def cycle(h: dict[str, str], log, *, budget: float, jobs: int) -> None:
    run("accumulation.redteam.rollout_scale", ["--jobs", str(jobs), "--budget", str(budget), "--quiet", "--reuse-brackets"], log)
    run("accumulation.redteam.block_term", ["--jobs", str(jobs + 1), "--budget", str(budget), "--quiet"], log)
    run("accumulation.redteam.coarse_validation", ["--fuzz", "200", "--time-limit", "30", "--reuse-istar", "--quiet"], log)
    line, viol = summarize(h)
    log(line)
    with open(os.path.join(RESULTS, "REDTEAM_STATUS.md"), "w") as f:
        f.write("# Red-team status (auto-updated by `redteam/watch.py` on every bound-module hash change)\n\n")
        f.write("- " + line + "\n")
        f.write(f"- violations now: **{len(viol)}**" + ("" if not viol else " -- see `REDTEAM_ALERTS.md`") + "\n")
        f.write("- details: `validation.md` (exact panel + folded registry cells), `validation_rollout.md` (dense + MoE rollout), "
                "`validation_block_term.md` (local-sgd / forward-nonfixed block term).\n")
    if viol:
        with open(os.path.join(RESULTS, "REDTEAM_ALERTS.md"), "a") as f:
            f.write(f"\n## {time.strftime('%Y-%m-%d %H:%M:%S')} -- {len(viol)} soundness violation(s) at {h}\n\n")
            for x in viol:
                f.write(f"- {x}\n")
        log(f"!!! {len(viol)} violations -> REDTEAM_ALERTS.md")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=7.0)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--settle", type=float, default=120.0, help="a changed hash must be stable this long before a run")
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--run-now", action="store_true", help="run one cycle immediately before watching")
    a = ap.parse_args(argv)
    logf = open("/tmp/redteam/watch.log", "a")

    def log(msg: str) -> None:
        s = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    last_run = hashes()
    if a.run_now:
        cycle(last_run, log, budget=a.budget, jobs=a.jobs)
    t_end = time.time() + a.hours * 3600
    pending: dict[str, str] | None = None
    pending_since = 0.0
    log(f"watching {WATCHED} for {a.hours} h; last run at {last_run}")
    while time.time() < t_end:
        time.sleep(a.poll)
        h = hashes()
        if h == last_run:
            pending = None
            continue
        if pending != h:
            pending, pending_since = h, time.time()
            log(f"change seen: { {k: v for k, v in h.items() if last_run.get(k) != v} } -- settling")
            continue
        if time.time() - pending_since >= a.settle:
            log(f"running cycle at {h}")
            cycle(h, log, budget=a.budget, jobs=a.jobs)
            last_run = hashes()
            pending = None
    log("watch period over")
    return 0


if __name__ == "__main__":
    sys.exit(main())
