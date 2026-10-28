"""`serve_row` on CPU: serve-openai's engine schedules as ``--async-scheduling`` says (off by default), a bench row whose
workload Program is specialized for the other scheduler is refused before the model loads, and a server whose engine
resolved another scheduler is refused by name. The flag serves with the Commit on or off."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_row as RW
from verity_vllm.pipeline import serve_seal as SS
from verity_vllm.program.frontend.target_profile import TargetProfile
from verity_vllm.program.predict.phi import Phi


def engine(resolved: bool | None) -> SimpleNamespace:
    """A stand-in ``vllm.LLM`` whose scheduler resolved ``async_scheduling`` to ``resolved``."""
    return SimpleNamespace(llm_engine=SimpleNamespace(vllm_config=SimpleNamespace(
        scheduler_config=SimpleNamespace(async_scheduling=resolved))))


def test_the_engine_schedules_as_the_flag_says_and_a_row_specialized_for_the_other_scheduler_is_refused(tmp_path, capsys):
    """The engine is built with ``--async-scheduling``'s value (off by default), never vLLM's own resolution (on). The bench
    row's Build declares the scheduler its workload Program models (``build_workload/workload_program.json``'s
    ``applicability.engine.async_scheduling``): a row without the declaration, or declaring the other scheduler than the
    flag, is refused before the model loads, and an engine that resolved another value (or none) is refused by name."""
    (man := tmp_path / "manifest.json").write_text("{}")
    base = ["--model", "m", "--revision", "r", "--required-manifest", str(man)]
    opts, on = (cli.parse("serve-openai", base + more, environ={}) for more in ([], ["--async-scheduling"]))
    assert (opts.async_scheduling, on.async_scheduling) == (False, True)
    assert so.engine_options(opts)["async_scheduling"] is False and so.engine_options(on)["async_scheduling"] is True
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
    row = RW.of(opts)
    assert row == {"async_scheduling": False, "source": str(wp)}
    with pytest.raises(cli.UsageError, match=r"--async-scheduling: the bench row's workload Program is specialized for "
                                             r"async_scheduling False \(LAG 0, .*serve the row with --no-async-scheduling, or a "
                                             r"row whose Build's target declares \"async_scheduling\": true"):
        RW.of(on)
    wp.write_text(json.dumps({"applicability": {"engine": {"enforce_eager": True, "async_scheduling": True}}}))
    row_on = RW.of(on)
    assert row_on == {"async_scheduling": True, "source": str(wp)}
    with pytest.raises(cli.UsageError, match=r"--no-async-scheduling: .* async_scheduling True \(LAG 1, .*serve the row with "
                                             r"--async-scheduling"):
        RW.of(opts)
    assert RW.of(cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--no-commit"], environ={})) is None

    assert RW.refusal(False, row, engine(False)) is None and RW.refusal(True, row_on, engine(True)) is None
    assert RW.refusal(False, None, engine(True)) == ("async_scheduling: the engine resolved True, and serve-openai built it for "
                                                     "False (--no-async-scheduling)")
    for resolved in (True, None):
        why = RW.refusal(False, row, engine(resolved))
        assert why.startswith(f"async_scheduling: the engine resolved {resolved}, and serve-openai built it for False "
                              "(--no-async-scheduling), the scheduler the bench row's workload Program is specialized for (LAG 0") and str(wp) in why
        with pytest.raises(RuntimeError, match=f"the engine resolved {resolved}"):
            RW.checked(opts, row, SimpleNamespace(vllm=engine(resolved)))
    with pytest.raises(RuntimeError, match=r"the engine resolved False, and serve-openai built it for True \(--async-scheduling\)"):
        RW.checked(on, row_on, SimpleNamespace(vllm=engine(False)))
    llm = SimpleNamespace(vllm=engine(True))
    assert RW.checked(on, row_on, llm) is llm
    assert f"SERVE-OPENAI engine resolved async_scheduling=True (--async-scheduling); the bench row declares True ({wp})" in capsys.readouterr().out


def test_async_scheduling_serves_with_the_commit_on_or_off(tmp_path):
    """``--async-scheduling`` passes the preflight with the Commit on (and so with H1, which serves on it) as without it: the
    learn-only warm-up learns the async steps' plans as it does the synchronous ones' (``serve_async_commit_gpu.py`` holds
    the two schedulers' roots equal on an H200)."""
    (man := tmp_path / "manifest.json").write_text("{}")
    base = ["--model", "m", "--revision", "r", "--required-manifest", str(man)]
    on, bare, off = (cli.parse("serve-openai", base + more, environ={})
                     for more in (["--async-scheduling"], ["--async-scheduling", "--no-commit"], []))
    assert (on.commit, on.async_scheduling, bare.commit) == (True, True, False)
    so.preflight(on, which=lambda _: "/v/bin/ninja")
    so.preflight(bare, which=lambda _: None)
    so.preflight(off, which=lambda _: "/v/bin/ninja")


def test_under_the_flag_the_served_target_declares_async_scheduling_and_lag_1(monkeypatch):
    """``serve.json``'s target (`serve_seal.target_of`), which a served window's φ is read from, declares async scheduling under
    the flag, so the Programs predicted for it are specialized for LAG 1 by declaration; without the flag it declares none,
    as every serve.json before the flag."""
    envs = SimpleNamespace(VLLM_BATCH_INVARIANT=True)
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(envs=envs))
    monkeypatch.setitem(sys.modules, "vllm.envs", envs)
    monkeypatch.setattr(TargetProfile, "from_device", classmethod(lambda cls: cls.default().with_overrides(num_sms=132)))
    off, on = SS.target_of(), SS.target_of(async_scheduling=True)
    assert "async_scheduling" not in off["target"] and on["target"]["async_scheduling"] is True
    assert {k: v for k, v in on["target"].items() if k != "async_scheduling"} == off["target"]
    assert off["execution"] == on["execution"] == "bi-eager"
    phi = Phi(model={"repo": "m", "revision": "r", "hf_config": {}}, target=on["target"], execution=on["execution"])
    assert phi.target_profile().async_scheduling_declaration()["declared_explicitly"] and phi.target_profile().lag() == 1
