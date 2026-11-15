"""The Lean tools' settings: the environment variables they read, each declared once here."""
from __future__ import annotations

import os

#: the GB the kept warm builds may take on a node before their least recently used go (`lean_audit.BUILDS_MAX_GB`)
BUILDS_MAX_GB = float(os.environ.get("VERITY_LEAN_BUILDS_MAX_GB", "120"))


def merged_in() -> dict[str, str]:
    """{commit: name} of each side `git merge` (or `git pull`) merges in, which git names to the merge drivers it runs as
    `GITHEAD_<commit>=<name>` (`merge.stale`); empty outside such a merge."""
    return {k.removeprefix("GITHEAD_"): v for k, v in os.environ.items() if k.startswith("GITHEAD_")}
