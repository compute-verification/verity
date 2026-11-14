"""`verity_vllm.pipeline.commit.summarize`'s three 95% intervals over the paired deltas (end-to-end wall, decode step, prefill
step): each is the seeded bootstrap of its own list of deltas, held here by value, so a change of the generator, of its seed, of
the number of resamples or of which list an interval is taken over fails here.  The rows are `test_commit_main.py`'s.  CPU
only."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import verity_vllm.pipeline.commit as C
from tests.pipeline.test_commit_main import (
    PLUGIN,
    Clock,
    Record,
    control_row,
    instrumented_row,
)

#: a control run's times by pair, in seconds: a step of its wall, its mean decode step and its prefill step.  An instrumented
#: run's are 1.25 each.  The three lists differ, and no two means of a resample are likely to be equal, so each interval
#: moves with the seed.
WALL = (1.0, 1.1, 0.9, 1.3, 0.7, 1.7, 0.6)
DECODE = (1.05, 0.95, 1.2, 0.8, 1.45, 0.65, 1.15)
PREFILL = (0.85, 1.35, 1.02, 0.74, 1.6, 0.93, 1.21)


def paired_rows(tmp_path: Path) -> list[Record]:
    rows: list[Record] = []
    for pair, (wall, decode, prefill) in enumerate(zip(WALL, DECODE, PREFILL, strict=True)):
        rows += [control_row(tmp_path, pair, 0, step_s=wall, step_wall_decode_mean_s=decode, step_wall_prefill_s=prefill),
                 instrumented_row(tmp_path, pair, 1, f"{pair:02x}")]
    return rows


def test_each_interval_of_the_summary_is_the_seeded_bootstrap_of_its_own_paired_deltas(tmp_path: Path,
                                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(C, "time", Clock())
    made = C.summarize(paired_rows(tmp_path), [], SimpleNamespace(workload=str(tmp_path / "workload.json"), committer=PLUGIN), {}, None)
    assert made["pairs"] == 7
    assert made["delta_wall"] == [3 * 1.25 / (3 * wall) - 1 for wall in WALL]
    assert made["delta_decode"] == [1.25 / decode - 1 for decode in DECODE]
    assert made["delta_prefill"] == [1.25 / prefill - 1 for prefill in PREFILL]
    assert tuple(made["delta_wall_ci95"]) == (0.031476693241399074, 0.6704409036341811)
    assert tuple(made["delta_decode_ci95"]) == (0.057241921466764004, 0.5285807712726301)
    assert tuple(made["delta_prefill_ci95"]) == (-0.00016968449455077663, 0.4202227437521556)
