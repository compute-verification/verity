"""`verity_vllm.pipeline.tp.warmup.run` on ranks whose finalize reports no bounded staging: a committer that reports none (None), a
finalize without the field and a committer that reports it off each leave the rank out of `bounded_warmup_<kind>.json`, which is
written only when a rank staged bounded."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline.tp import warmup as WU

STAGED = {"enabled": True, "learn_only": True, "learned_only_steps": [0], "learned_only_bytes": 1 << 20, "transient_peak_bytes": 0,
          "learn_budget_bytes": 4 << 20, "plan_mismatches": 0}
UNBOUNDED = [{"bounded_staging": None}, {}, {"bounded_staging": {}}, {"bounded_staging": {"enabled": False, "learn_only": False}}]


class _LLM:
    """An engine whose ranks answer the warm-up's calls from ``finalize``, one entry per rank."""

    def __init__(self, finalize: list[dict]) -> None:
        self.finalize = finalize
        self.rpcs: list[tuple] = []

    def collective_rpc(self, method, args=(), timeout=None):
        name = method if isinstance(method, str) else method.__name__
        self.rpcs.append((name, args))
        if name == "tp2_commit_finalize":
            return self.finalize
        if name == "warming":
            return [{"rank": f["rank"], "learn_only": False, "host_mem_kb": {"rss": 1, "hwm": 2}} for f in self.finalize]
        return [None for _ in self.finalize]


def _rank(rank: int, report: dict) -> dict:
    return {"rank": rank, "leaves": 3, "errors_n": 0, "errors": [], "pinned_pool_bytes": 1 << 20, "plans_cached": 5} | report


@pytest.mark.parametrize("report", UNBOUNDED)
def test_a_warm_up_whose_ranks_report_no_bounded_staging_writes_no_bounded_warm_up_file(tmp_path, capsys, report):
    llm = _LLM([_rank(0, report), _rank(1, report)])
    assert WU.run(llm, lambda: {"wall_s": 1.0}, tmp_path, "native_collect_v2b", 60.0) == {"wall_s": 1.0}
    assert [name for name, _args in llm.rpcs] == ["warming", "tp2_commit_install", "tp2_commit_finalize", "warming", "tp2_commit_reset"]
    assert sorted(f.name for f in tmp_path.iterdir()) == ["finalize_warmup.json"]
    assert [f["rank"] for f in json.loads((tmp_path / "finalize_warmup.json").read_text())] == [0, 1]
    assert capsys.readouterr().out == "[tp2commit] warm-up instrumented: 1.000s; ranks [(0, 3, 0, []), (1, 3, 0, [])]\n"


@pytest.mark.parametrize("report", UNBOUNDED)
def test_the_bounded_warm_up_file_holds_only_the_ranks_that_staged_bounded(tmp_path, capsys, report):
    llm = _LLM([_rank(0, report), _rank(1, {"bounded_staging": STAGED})])
    WU.run(llm, lambda: {"wall_s": 1.0}, tmp_path, "native_collect_v2b", 60.0)
    (b,) = json.loads((tmp_path / "bounded_warmup_native_collect_v2b.json").read_text())
    assert b == {"rank": 1, **STAGED, "pinned_pool_bytes": 1 << 20, "plans_cached": 5, "host_mem_kb": {"rss": 1, "hwm": 2}, "errors": []}
    assert capsys.readouterr().out.splitlines() == [
        "[tp2commit] warm-up instrumented: 1.000s; ranks [(0, 3, 0, []), (1, 3, 0, [])]",
        ("[tp2commit] rank 1: bounded staging warm-up (native_collect_v2b): learned-only steps [0] (1 MiB discarded), transient peak 0 MiB "
         "(budget 4 MiB), plan mismatches 0, plans cached 5, pinned pool 1 MiB")]
