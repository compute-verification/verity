"""`serve_remote.ServeSession.built` on CPU, over stand-ins of the engine `serve_openai.build` hands it and of the Commit's
session (`pipeline.remote_session.commit_session`): the session's check runs on the engine's ``vllm.LLM`` with its
in-process model runner (`engine.handle.model_runner`)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from verity_vllm.pipeline import serve_remote as SR


def engine(runner: Any) -> NS:
    """A built `verity_vllm.LLM` as the session reads it: its ``vllm.LLM`` over vLLM's object graph, the engine core in
    this process with ``runner`` its driver worker's."""
    core = NS(model_executor=NS(driver_worker=NS(worker=NS(model_runner=runner))))
    return NS(vllm=NS(llm_engine=NS(engine_core=NS(engine_core=core))))


def test_built_hands_the_sessions_check_the_engines_model_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checked: list[tuple[Any, Any]] = []
    session = NS(check=lambda llm, runner: checked.append((llm, runner)))
    monkeypatch.setattr(SR.RS, "commit_session", lambda remote, out, kv_pages: session)
    rs = SR.ServeSession(NS(remote=NS()), tmp_path)
    runner = NS()
    llm = engine(runner)
    rs.built(llm)
    assert len(checked) == 1 and checked[0][0] is llm.vllm and checked[0][1] is runner
