"""`verity_vllm.acquire.h1_commit.RankCommit.h1rank_commit` on a rank that holds no H1 window: the Commit is refused before
the rank's model is read. And `H1Source` at a routed expert that no row was routed to: its zero rows, once an engine step."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.acquire import h1_commit as HC


class Rank(HC.RankCommit):
    """A worker whose model must not be read."""

    def _h1rank_layers(self):
        raise AssertionError("the Commit read the model of a rank with no H1 window")


def test_a_rank_whose_window_was_released_refuses_its_commit():
    """`RankExtension.h1rank_release` leaves the rank's state None."""
    rank = Rank()
    rank._h1rank = None
    with pytest.raises(RuntimeError, match="holds no H1 window"):
        rank.h1rank_commit()


class Commit:
    """What `H1Source` reads of a committer: the step being committed, its errors and its acquire."""

    def __init__(self) -> None:
        self._cur: Any = None
        self.errors: list[str] = []
        self.got: list[tuple[int, str, tuple[int, ...], str]] = []

    def _acquire(self, name: str, t: Any, job: Any) -> None:
        self.got.append((int(job.step), name, tuple(t.shape), str(t.dtype)))


def test_an_idle_expert_is_acquired_as_its_zero_rows_once_an_engine_step() -> None:
    """A routed expert no row was routed to is a call of zero rows, which runs no ``row_quant``: the source acquires its
    zero rows [0, K] and scales [0] itself. A second call of it at the same engine step is the source's error and acquires
    nothing; the next step's call is acquired, and a call while no step is committed, or after the source is taken off, is
    not."""
    torch = pytest.importorskip("torch")
    from verity_vllm.engine.int8_experts import ExpertSite

    k, n = 8, 4
    idle = ExpertSite("moe", 0, "down", torch.zeros(n, k, dtype=torch.int8), torch.ones(n, 1))
    com, x = Commit(), torch.zeros(0, k, dtype=torch.bfloat16)
    src = HC.H1Source(com, torch.nn.Module(), [], {idle.name: idle})
    src.install()
    try:
        com._cur = SimpleNamespace(step=7)
        rows = [(7, "moe.0.down/row_quant/0", (0, k), "torch.int8"), (7, "moe.0.down/row_quant/1", (0,), "torch.float32")]
        assert idle.quant_method.apply(idle, x).shape == (0, n)
        assert com.got == rows and com.errors == []
        assert idle.quant_method.apply(idle, x).shape == (0, n)
        assert com.got == rows and src.stats["errors"] == 1
        assert com.errors == ["int8: moe.0.down was called twice at engine step 7; the Program has one Int8RowQuant_v1 site"]
        com._cur = SimpleNamespace(step=8)
        idle.quant_method.apply(idle, x)
        assert [g[0] for g in com.got] == [7, 7, 8, 8] and len(com.errors) == 1
        com._cur = None
        idle.quant_method.apply(idle, x)
        assert len(com.got) == 4 and len(com.errors) == 1
    finally:
        src.uninstall()
    com._cur = SimpleNamespace(step=9)
    idle.quant_method.apply(idle, x)
    assert len(com.got) == 4 and src._scope == []
