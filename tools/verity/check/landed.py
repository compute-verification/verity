#!/usr/bin/env python3
"""`tools/verity/check/landed.py COMMIT [--on MACHINE]`: what runs once a landing of main is pushed (`after_land` in `tool.py`, which
`research merge` and `research queue drive` start, detached, on main's tip).

1. `post_train.py COMMIT`, in a session of its own: the vLLM reference regression's manifest rebuilds and their label.
2. When the tree's ci.toml declares `after_landing` jobs (Daniel, 12 Oct: a part of check over about 10 min cold may run after
   the landing; the Lean audit never does), `train.sh COMMIT MACHINE --after-landing` records them on COMMIT (`check.py
   --after-landing`), and this waits for the run's end (`research status`, each POLL_S).  A run that failed is posted to the
   merge thread for the lander, who reverts the landing at once (Daniel, 12 Oct): the landing's own PRs, and any landing since
   the last after-landing check that passed, are what it tested for the first time.  `research.merge.after_landing` reads a
   commit's passing run, which citations and tables filter on.  A launch that registered no run is posted by train.sh itself.

The landing stands whatever this finds; it reports, and changes nothing on main."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MERGE_THREAD = "1790957906.278529"
POLL_S = 60
#: the after-landing jobs' slot wait and run, with the runner's publish margin: `research run`'s own bound ends a run long before
POLLS = 10 * 3600 // POLL_S
ACTIVE = ("submitted", "waiting", "running", "unknown")
#: check.py's exit for a twin that never took a slot
NOT_STARTED = 75
RUN_RE = re.compile(r"^RUN ((?:r20\d{6}-\d{6}-[0-9a-f]{4} ?)+)$", re.M)
SLEEP = time.sleep


def late_jobs(root: Path = ROOT) -> list[str]:
    sys.path.insert(0, str(root / "tools" / "verity" / "check"))
    import declaration

    jobs = declaration.load(root / "tools" / "verity" / "check" / "ci.toml")
    return [j.name for j in jobs if j.after_landing]


def research(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "research", *args], cwd=ROOT, capture_output=True, text=True)


def states(ids: list[str]) -> dict[str, dict]:
    """Each run's record from `research status RUN --json`; one it can't read is left out (its state counts as unknown)."""
    out = {}
    for rid in ids:
        try:
            out[rid] = json.loads(research("status", rid, "--json").stdout)["runs"][0]
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    return out


def outcome(runs: dict[str, dict], ids: list[str]) -> tuple[str, str]:
    """("passed" | "failed" | "none" | "running", the run that decides it) for train.sh's run ids: twins that never took a slot
    exit 75, and the one that ran decides."""
    got = [runs.get(i, {}) for i in ids]
    if any(r.get("state", "unknown") in ACTIVE for r in got):
        return "running", ""
    ran = [(i, r) for i, r in zip(ids, got) if r.get("rc") != NOT_STARTED]
    if not ran:
        return "none", ids[0]
    i, r = ran[0]
    return ("passed" if r.get("rc") == 0 else "failed"), i


def post(text: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    try:
        research("slack", "reply", "--as", os.environ.get("TRAIN_AS", "old-circuits-and-proofs"), "--thread",
                 os.environ.get("MERGE_THREAD", MERGE_THREAD), "--file", f.name)
    finally:
        os.unlink(f.name)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="landed.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("commit")
    ap.add_argument("--on", default="auto", metavar="MACHINE", help="where the after-landing jobs run (default auto)")
    ap.add_argument("--no-post-train", action="store_true", help="start only the after-landing jobs")
    a = ap.parse_args(argv)
    log = Path(tempfile.gettempdir()) / f"post-train-{a.commit[:12]}.log"
    if not a.no_post_train:
        with open(log, "ab") as f:
            subprocess.Popen([sys.executable, str(ROOT / "tools" / "verity" / "check" / "post_train.py"), a.commit], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        print(f"landed: started post_train.py {a.commit[:12]} (log {log})", flush=True)
    late = late_jobs()
    if not late:
        print("landed: ci.toml declares no after_landing job", flush=True)
        return 0
    p = subprocess.run(["bash", str(ROOT / "tools" / "verity" / "check" / "train.sh"), a.commit, a.on, "--after-landing"],
                       cwd=ROOT, capture_output=True, text=True)
    print((p.stdout + p.stderr).strip()[-2000:], flush=True)
    m = RUN_RE.search(p.stdout)
    if p.returncode or not m:
        return 1                                       # train.sh posted why
    ids = m.group(1).split()
    for _ in range(POLLS):
        got, run = outcome(states(ids), ids)
        if got != "running":
            break
        SLEEP(POLL_S)
    subject = subprocess.run(["git", "log", "-1", "--format=%s", a.commit], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    what = f"main {a.commit[:12]} ({subject})'s after-landing check ({', '.join(late)})"
    if got == "passed":
        print(f"landed: {what} passed on {run}", flush=True)
        return 0
    if got == "running":
        post(f"{what}, {' '.join(ids)}, showed no end in {POLLS * POLL_S // 3600} h: rerun "
             f"`python tools/verity/check/landed.py {a.commit[:12]} --no-post-train` (quiet)")
        return 1
    if got == "none":
        post(f"{what} never took a slot ({' '.join(ids)} all exit {NOT_STARTED}): rerun "
             f"`python tools/verity/check/landed.py {a.commit[:12]} --no-post-train` (quiet)")
        return 1
    post(f"{what} FAILED on {run} (`research inspect {run}`). Daniel's rule: revert the landing at once. The jobs ran here "
         "for the first time on this landing and on any since the last one that passed. (quiet)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
