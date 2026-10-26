"""The served model of a coverage workload role, for the roles `checkpoints.json` names no repo under
(`pipeline.coverage_workloads` reads it when the manifest's lookup comes back empty)."""
from __future__ import annotations

#: role -> HF repo.  B1's manifest entry carries a compound role (``B1-B4,R*,...``), which the exact role lookup misses.
ROLE_REPOS = {"B0": "HuggingFaceTB/SmolLM2-135M", "B1": "Qwen/Qwen2.5-1.5B"}
