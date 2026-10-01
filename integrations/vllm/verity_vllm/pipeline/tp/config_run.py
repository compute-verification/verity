"""`tp-commit`'s config-run pieces (`--replay-k`, `--only-arm`): the k units split over the ranks, and the one-arm summary."""
from __future__ import annotations


def replay_k_per_rank(k: int | None, world: int) -> dict[str, int] | None:
    """The config run's k units split over the ranks: rank r takes k // W, one more for r < k % W."""
    if not k:
        return None
    return {str(r): k // world + (1 if r < k % world else 0) for r in range(world)}


def only_arm_summary(only_arm: str, rows: list[dict], treatment_rows: list[dict], commit_pass: list) -> dict:
    """The single-GPU Commit's only-arm summary fields: an instrumented-only run's `commit_pass` is the one decision (None for control)."""
    ok = (bool(treatment_rows) and all(bool(c) for c in commit_pass)) if only_arm == "instrumented" else None
    return {"only_arm": only_arm, "runs": len(rows), "commit_pass": ok}
