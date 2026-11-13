"""research's settings: the environment variables it reads outside the files `tests/lint/allowlists/repo_env_reads.json` still
lists, each declared once here."""
from __future__ import annotations

import os


def train_trees() -> str | None:
    """TRAIN_TREES: where a lander's worktrees go instead of TMPDIR (train.sh's, drive's auto plans', the after-landing
    command's); the control pod's `infra/control/drive.sh` and the Nebius loop's
    `tools/research/src/research/pods/nebius/vqueue/loop.sh` always set it, to a directory on a local disk whose old trees
    they sweep.  Unset or empty, None."""
    return os.environ.get("TRAIN_TREES") or None
