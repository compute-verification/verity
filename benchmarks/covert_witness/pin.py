"""The pinned measurement version (fixed): the one commit of the integrator's branch whose fixed side (prepare,
evaluate, the decoder, te_core, candidate.sh, sandbox.py) measures every candidate. `pinned.json` on that branch names
it. launch.py and promote.py run from it (`source`: a clean checkout of it beside this one), launch.py taking only the
candidate's island from the candidate's commit; a record measured by any other commit than it and the earlier commits
`same_version` lists (each measuring every search run as it does) is `other-version` and stays out of the frontier and
the promotions (`te_core.frontier`). `pilot` lists an earlier version's commits, whose records te_core.panels alone
shows beside the pin's, each curve labelled by its data.

    python3 pin.py [--fetch]     # the pin, as launch.py and the frontier read it

Stdlib only.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BRANCH = "cursor/covert-witness-4207"
FILE = "benchmarks/covert_witness/pinned.json"
CHECKOUTS = Path.home() / ".research" / "cw-pinned"


def git(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def pinned(fetch: bool = False) -> dict:
    """The pin on origin's BRANCH (fetched first with `fetch`), else this checkout's pinned.json."""
    if fetch:
        r = git("fetch", "-q", "origin", BRANCH)
        if r.returncode:
            raise SystemExit(f"pin: git fetch origin {BRANCH} failed: {r.stderr.strip()[-500:]}")
    r = git("show", f"origin/{BRANCH}:{FILE}")
    if r.returncode == 0:
        return json.loads(r.stdout) | {"from": f"origin/{BRANCH}"}
    return json.loads((REPO / FILE).read_text()) | {"from": "checkout"}


def source(fetch: bool = True) -> tuple[dict, Path]:
    """The pin and a clean detached worktree of its commit (made once per repository and commit), which `research run
    --source` ships. Launches started together take turns (a lock under CHECKOUTS), so one makes the worktree."""
    CHECKOUTS.mkdir(parents=True, exist_ok=True)
    with open(CHECKOUTS / ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pin = pinned(fetch)
        common = git("rev-parse", "--path-format=absolute", "--git-common-dir").stdout.strip()
        wt = CHECKOUTS / hashlib.sha256(common.encode()).hexdigest()[:12] / pin["commit"]
        if not wt.exists():
            wt.parent.mkdir(parents=True, exist_ok=True)
            r = git("worktree", "add", "--detach", str(wt), pin["commit"])
            if r.returncode:
                raise SystemExit(f"pin: git worktree add {wt} {pin['commit']} failed: {r.stderr.strip()[-500:]}")
    head = git("rev-parse", "HEAD", cwd=wt).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=all", cwd=wt).stdout.strip()
    if head != pin["commit"] or dirty:
        raise SystemExit(f"pin: {wt} is not a clean checkout of {pin['commit']} (HEAD {head}); remove it with "
                         f"`git worktree remove --force {wt}` and launch again")
    return pin, wt


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true")
    print(json.dumps(pinned(ap.parse_args(argv).fetch)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
