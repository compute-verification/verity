from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plan  # noqa: E402


@dataclass
class Info:
    name: str
    kind: str
    blocks: int
    block_bytes: int


class FakeStore:
    """`layout().regions` and `idle_blocks`, as ra-vllm-hooks' `Store` gives them."""

    def __init__(self, regions: list[Info], idle: dict[str, np.ndarray]) -> None:
        self.regions, self.idle = regions, idle

    def layout(self):
        return type("Layout", (), {"regions": tuple(self.regions)})()

    def idle_blocks(self, name: str) -> np.ndarray:
        return self.idle[name]


def store(free=np.arange(0, 40, 2)) -> FakeStore:
    regions = [Info("w", "weights", 100, 4096), Info("kv/0", "kv", 40, 4096), Info("kv/1", "kv", 40, 4096),
               Info("host", "filler", 30, 4096)]
    return FakeStore(regions, {"kv/0": free, "kv/1": free})


def test_a_cycle_reads_every_idle_block_once():
    s = store()
    f = plan.RotatingFiller(lambda j: 7 * 4096)
    seen: dict[str, list[int]] = {}
    for j in range(10):  # 20 + 20 + 30 = 70 idle blocks, 7 a pass
        for name, blocks in f(s, j):
            seen.setdefault(name, []).extend(int(b) for b in blocks)
    assert sorted(seen["kv/0"]) == list(range(0, 40, 2))
    assert sorted(seen["kv/1"]) == list(range(0, 40, 2))
    assert sorted(seen["host"]) == list(range(30))
    assert "w" not in seen


def test_the_budget_rounds_up_to_blocks_and_never_repeats_inside_a_pass():
    s = store()
    f = plan.RotatingFiller(lambda j: 10 ** 9)
    out = f(s, 0)
    assert sum(len(b) for _, b in out) == 70
    f = plan.RotatingFiller(lambda j: 4097)
    assert [len(b) for _, b in f(s, 0)] == [2]


def test_no_idle_blocks_or_no_budget_reads_nothing():
    assert plan.RotatingFiller(lambda j: 0)(store(), 0) == []
    empty = FakeStore([Info("kv/0", "kv", 40, 4096)], {"kv/0": np.zeros(0, np.int64)})
    assert plan.RotatingFiller(lambda j: 4096)(empty, 0) == []


def test_slack():
    assert plan.slack(1540.0, 0.010, 10e9) == int(15.4e9 - 10e9)
    assert plan.slack(1540.0, 0.001, 10e9) == 0
