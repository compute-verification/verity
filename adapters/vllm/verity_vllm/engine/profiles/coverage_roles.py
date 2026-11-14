"""The served model of a coverage workload role, for the roles `checkpoints.json` names no repo under
(`pipeline.coverage_workloads` reads it when the manifest's lookup comes back empty)."""
from __future__ import annotations

from verity_vllm.engine.profiles.cases import CASES

__all__ = ["ROLE_REPOS"]

#: role -> HF repo, the case's own (`CASES`).  B1's manifest entry carries a compound role (``B1-B4,R*,...``), which the exact
#: role lookup misses.
ROLE_REPOS = {role: CASES[role]["repo"] for role in ("B0", "B1")}
