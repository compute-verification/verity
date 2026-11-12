"""`serve_remote.ServeSession.built` on CPU, over stand-ins of the engine `serve_openai.build` hands it and of the store
binder: the session's check (`remote.CommitSession.check`) runs on the engine's ``vllm.LLM`` with its in-process model
runner (`engine.handle.model_runner`), on whose device the session's filler is registered."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from tests.pipeline.test_serve_remote import VERIFIER, parse
from verity_vllm.pipeline import serve_remote as SR
from verity_vllm.protocols.remote_bandwidth import remote


def engine(runner: Any) -> NS:
    """A built `verity_vllm.LLM` as the session reads it: its ``vllm.LLM`` over vLLM's object graph, the engine core in
    this process with ``runner`` its driver worker's, scheduling synchronous, no prefix caching, one GPU."""
    config = NS(scheduler_config=NS(async_scheduling=False), cache_config=NS(enable_prefix_caching=False),
                parallel_config=NS(tensor_parallel_size=1))
    core = NS(model_executor=NS(driver_worker=NS(worker=NS(model_runner=runner))))
    return NS(vllm=NS(llm_engine=NS(vllm_config=config, engine_core=NS(engine_core=core))))


def test_built_hands_the_sessions_check_the_engines_model_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[tuple[Any, Any]] = []
    filler: list[tuple[int, Any]] = []
    check = remote.CommitSession.check

    def spy(self: Any, llm: Any, runner: Any) -> dict[str, Any]:
        checked.append((llm, runner))
        return check(self, llm, runner)

    monkeypatch.setattr(remote.CommitSession, "check", spy)
    rs = SR.session_of(parse(*VERIFIER, "--remote-filler-gib", "0.5"), tmp_path)
    assert rs is not None
    rs.cs.binder = NS(stats=lambda: {"blocks": 2}, register_filler=lambda nbytes, device: filler.append((nbytes, device)))
    runner = NS(device="cuda:0")
    llm = engine(runner)
    rs.built(llm)
    assert len(checked) == 1 and checked[0][0] is llm.vllm and checked[0][1] is runner
    assert filler == [(1 << 29, "cuda:0")]
