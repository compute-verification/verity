"""Pure helpers of the TP Commit's config run (`k_per_rank`) and of the single-GPU `--only-arm` summary."""
from __future__ import annotations

from pathlib import Path


def replay_k_per_rank(k: int | None, world: int) -> dict[str, int] | None:
    """The config run's k units split over the ranks: rank r takes k // W, one more for r < k % W."""
    if not k:
        return None
    return {str(r): k // world + (1 if r < k % world else 0) for r in range(world)}


def replay_host(replay_cfg: dict | None, versions: dict, out: str | Path, *, ranks_at_once: int | None, memory_map: str | None) -> None:
    """``--replay-ranks-at-once K`` and ``--memory-map`` into the TP Commit's replay config (`pipeline.tp.rank_replay`'s `host_slot`
    and `memory_map`): K host slots under the Commit's out dir, and the map request's resolved path."""
    if ranks_at_once and replay_cfg is not None:
        replay_cfg.update(ranks_at_once=int(ranks_at_once), slot_dir=str(out))
        versions["sampled_replay"]["ranks_at_once"] = int(ranks_at_once)
    if memory_map:
        if replay_cfg is None or not Path(memory_map).is_file():
            raise SystemExit("--memory-map on a TP Commit needs its request file and --required-manifest with a --build-rank per rank (each rank "
                             "writes its map from its Programs of record, in its replay)")
        replay_cfg["memory_map"] = str(Path(memory_map).resolve())


def only_arm_summary(only_arm: str, rows: list[dict], treatment_rows: list[dict], commit_pass: list) -> dict:
    """The single-GPU Commit's only-arm summary fields: an instrumented-only run's `commit_pass` is the one decision (None for control)."""
    ok = (bool(treatment_rows) and all(bool(c) for c in commit_pass)) if only_arm == "instrumented" else None
    return {"only_arm": only_arm, "runs": len(rows), "commit_pass": ok}
