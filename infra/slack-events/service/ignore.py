"""Vercel's `ignoreCommand` (vercel.json): exit 0 skips the build, 1 builds.  A Git deployment builds only for `main`, and
only when the service's inputs changed since the deployment before it (VERCEL_GIT_PREVIOUS_SHA), so a landing that touches
the service deploys itself and nothing else deploys.  No other branch builds: one that lacks the service's root fails
its build.

A deployment with no Git branch (console's `vercel deploy --prod`) always builds, and so do a branch's first deployment
(no previous SHA: the commit before may be the top of a train whose service change sits lower) and one whose diff git
can't answer (a base the clone doesn't have): skipping is the one outcome that needs proof."""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INPUTS = ("infra/slack-events/", "tools/research/src/research/slack.py", ".agents/skills/using-slack/registry.json")
BRANCH = "main"


def changed(base: str, repo: Path = REPO) -> bool | None:
    """Whether INPUTS differ between `base` and HEAD; None when git can't say."""
    r = subprocess.run(["git", "diff", "--quiet", base, "HEAD", "--", *INPUTS], cwd=repo, capture_output=True)
    return {0: False, 1: True}.get(r.returncode)


def decide(env: Mapping[str, str], diff: Callable[[str], bool | None] = changed) -> tuple[int, str]:
    ref = env.get("VERCEL_GIT_COMMIT_REF") or ""
    if not ref:
        return 1, "not a Git deployment: build"
    if ref != BRANCH:
        return 0, f"branch {ref}: only {BRANCH} deploys"
    base = env.get("VERCEL_GIT_PREVIOUS_SHA") or ""
    if not base:
        return 1, "no deployment before this one to diff against: build"
    got = diff(base)
    if got is False:
        return 0, f"no change to the service's inputs since {base[:12]}: skip"
    return 1, f"the service's inputs changed since {base[:12]}: build" if got else f"can't diff from {base[:12]}: build"


if __name__ == "__main__":
    code, why = decide(os.environ)
    print(f"ignore.py: {why}")
    sys.exit(code)
