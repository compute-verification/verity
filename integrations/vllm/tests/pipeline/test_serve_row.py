"""`serve_row` on CPU: serve-openai's engine schedules as the bench row's workload Program is specialized for, and a server
whose engine resolved another scheduler is refused by name."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_row as RW


def engine(resolved: bool | None) -> SimpleNamespace:
    """A stand-in ``vllm.LLM`` whose scheduler resolved ``async_scheduling`` to ``resolved``."""
    return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=resolved))))


def test_the_engine_schedules_as_the_rows_workload_program_is_specialized_for_and_is_refused_otherwise(tmp_path, capsys):
    """The bench row's Build declares the scheduler its workload Program models (``build_workload/workload_program.json``'s
    ``applicability.engine.async_scheduling``): the engine is built with it, not with vLLM's default (on); a row without the
    declaration is refused before the model loads, and an engine that resolved another value (or none) is refused by name."""
    (man := tmp_path / "manifest.json").write_text("{}")
    opts = cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--required-manifest", str(man)], environ={})
    with pytest.raises(cli.UsageError, match="the row's Build has no workload Program"):
        RW.of(opts)
    (wp := tmp_path / RW.WORKLOAD_PROGRAM).parent.mkdir()
    wp.write_text(json.dumps({"applicability": {"engine": {"enforce_eager": True}}}))
    with pytest.raises(cli.UsageError, match=r"declares no applicability.engine.async_scheduling \(KeyError"):
        RW.of(opts)
    wp.write_text(json.dumps({"applicability": {"engine": {"enforce_eager": True, "async_scheduling": None}}}))
    with pytest.raises(cli.UsageError, match="async_scheduling None, not a boolean"):
        RW.of(opts)
    wp.write_text(json.dumps({"applicability": {"engine": {"enforce_eager": True, "async_scheduling": False}}}))
    assert (row := RW.of(opts)) == {"async_scheduling": False, "source": str(wp)}
    assert RW.engine_args(row) == {"async_scheduling": False} and RW.engine_args(None) == {}
    assert RW.of(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--no-commit"], environ={})) is None

    assert RW.refusal(row, engine(False)) is None and RW.refusal(None, engine(True)) is None
    for resolved in (True, None):
        why = RW.refusal(row, engine(resolved))
        assert why.startswith(f"async_scheduling: the engine resolved {resolved}, and the bench row's workload Program is "
                              "specialized for False (LAG 0") and str(wp) in why
        with pytest.raises(RuntimeError, match=f"the engine resolved {resolved}"):
            RW.checked(row, SimpleNamespace(vllm=engine(resolved)))
    llm = SimpleNamespace(vllm=engine(False))
    assert RW.checked(row, llm) is llm
    assert f"SERVE-OPENAI engine resolved async_scheduling=False; the bench row declares False ({wp})" in capsys.readouterr().out
