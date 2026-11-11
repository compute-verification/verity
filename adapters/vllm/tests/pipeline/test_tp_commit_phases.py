"""`tp-commit`'s engine phase on its row's timeline (`pipeline.tp.phases.engine`), which a held-idle reading attributes a TP Commit's
lease to: `engine.build`, the engine through its probe, ended not ok when the build fails.  CPU-only: `phases.engine` runs over a
stand-in LLM, which marks the same timeline as it runs."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from verity_vllm.pipeline import timeline as T
from verity_vllm.pipeline.tp import phases as PH

from tests.pipeline import served_ranks

WORKLOAD = {"requests": [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 4}], "sampling": {"max_tokens": 4}, "sha256": "w",
            "execution": "bi-eager"}


class _LLM:
    """`vllm.LLM` over two ranks: building it and every `collective_rpc` but the CPU readings around a run mark the timeline (``LLM``,
    ``rpc.<method>``); each rpc answers one dict per rank."""

    def __init__(self, mark: T.Timeline, ranks: int = 2) -> None:
        self.mark, self.ranks = mark, ranks

    def __call__(self, **kw) -> _LLM:
        self.mark.event("LLM")
        return self

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        name = getattr(method, "__name__", method)
        if name != "thread_cpu":
            self.mark.event(f"rpc.{name}")
        if (served := served_ranks.answer(method, WORKLOAD, range(self.ranks))) is not None:
            return served
        return [{"rank": r, "leaves": 3, "errors_n": 0, "errors": [], "committer": "native_collect_v2b", "cpu_s": {}} for r in range(self.ranks)]


def _events(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines()]


@pytest.fixture
def timeline(tmp_path, monkeypatch) -> Path:
    """The row's timeline as `tp-commit` finds it (``VERITY_TIMELINE``), written through `phases.TL`'s process label and fields."""
    path = tmp_path / "row" / "timeline.jsonl"
    monkeypatch.setenv("VERITY_TIMELINE", str(path))
    assert (PH.TL.proc, PH.TL.defaults) == ("tp_commit", {"hot": False})
    monkeypatch.setattr(PH, "TL", T.Timeline.from_env(PH.TL.proc, **PH.TL.defaults))
    return path


@pytest.mark.parametrize("ranks,build_s", [(1, True), (None, False)], ids=["one-rank-answers-the-probe", "the-engine-does-not-start"])
def test_an_engine_build_that_fails_ends_its_span(timeline, ranks, build_s):
    """Fail closed when fewer ranks than ``tp`` answer the probe (SystemExit) or the engine raises: `engine.build` ended not ok,
    with the build's seconds once the engine was built."""
    llm = _LLM(T.Timeline(timeline, proc="stand-in"), ranks=ranks or 2)

    def build() -> _LLM:
        if ranks is None:
            raise RuntimeError("CUDA out of memory")
        return llm()

    with pytest.raises(SystemExit if ranks else RuntimeError, match="fail-closed: 1 ranks answered the probe, tp=2" if ranks else "out of memory"):
        PH.engine(build, 2, 60.0, model="org/model", enforce_eager=True, case="C")
    events = _events(timeline)
    end = events[-1]
    assert (end["stage"], end["ev"], end["ok"], "build_s" in end) == ("engine.build", "end", False, build_s) and T.report(events)["unclosed"] == []
