"""serve-openai's Pearl+ arm (`serve_pearl`): what it refuses before the model loads, the coins it reserves before the
engine is built, which passes each Commit window ended, and the passes' document at the stop."""
from __future__ import annotations

import json
import random
import stat
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.pipeline import cli
from verity_vllm.pipeline import serve_h1 as SH
from verity_vllm.pipeline import serve_openai as so
from verity_vllm.pipeline import serve_pearl as PL
from verity_vllm.pipeline.serve_commit import CommitWindows

from tests.pipeline import predicted

ENTRY = "pouw:pearl-c-h100-v1-h3"
GATE = ("--challenger-gate", "ssh research@auditor", "--challenger-peer", "vy-dc-h200")


@pytest.fixture(autouse=True)
def _seeded_coins(monkeypatch):
    """The in-process challenger on a seeded entropy (a served one's is the Lean verifier of record's)."""
    from verity_vllm.protocols import pouw

    monkeypatch.setattr(pouw.verifier, "coins", random.Random(5).randbytes)


def _opts(tmp_path, *more: str) -> Any:
    man = predicted.row(tmp_path)
    return cli.parse("serve-openai", ["--model", "m", "--revision", "r", "--required-manifest", str(man),
                                      "--out", str(tmp_path / "out"), *more], environ={})


def test_serve_openai_refuses_a_pearl_serve_it_cant_tie_before_the_model_loads(tmp_path):
    """Pearl+ is served alone on the rows-sha512 Commit, its coins a live session reserved under --pearl-lane or none,
    and without H1's prover, receipts, window commit or draw-gated release; --pearl-lane is Pearl+'s alone."""
    def refusal(*more: str) -> str | None:
        try:
            so.preflight(_opts(tmp_path, *more), which=lambda _: "/v/bin/ninja")
        except cli.UsageError as e:
            return str(e)
        return None

    pearl = ("--protocols", ENTRY)
    assert refusal(*pearl) is None
    assert refusal(*pearl, "--protocols-config", json.dumps({"pouw": {"device": "cuda", "kernel": "k", "cubin": "c"}})) is None
    assert refusal(*pearl, *GATE, "--pearl-lane", "compute-accounting") is None
    assert "give both or neither" in refusal(*pearl, *GATE)
    assert "give both or neither" in refusal(*pearl, "--pearl-lane", "compute-accounting")
    assert "go together" in refusal(*pearl, *GATE[:2], "--pearl-lane", "x")
    assert "--pearl-lane is Pearl+'s" in refusal("--pearl-lane", "compute-accounting")
    assert "--pearl-lane is Pearl+'s" in refusal("--protocols", SH.ENTRY, "--pearl-lane", "compute-accounting")
    assert "one pouw:pearl-c-* entry alone" in refusal("--protocols", f"{ENTRY},pous:x")
    assert "--commit, --scheme rows-sha512" in refusal(*pearl, "--no-commit")
    assert "PoUW H1's, not" in refusal(*pearl, "--h1-prover", "p")
    assert f"--release-after {SH.DRAWER}" in refusal(*pearl, "--seal", "--disk-floor-gb", "0", "--retain", "device", "--release-after", SH.DRAWER)
    for own in PL.OWN:
        assert "the server's own" in refusal(*pearl, "--protocols-config", json.dumps({"pouw": {own: "x"}})), own
    assert "(pouw)" in refusal(*pearl, "--protocols-config", json.dumps({"pous": {}}))
    assert "'device' is one of" in refusal(*pearl, "--protocols-config", json.dumps({"pouw": {"device": "tpu"}}))
    assert "--protocols-config {" in refusal(*pearl, "--protocols-config", "{not json")
    assert refusal("--protocols", SH.ENTRY) is None, "H1 is still served through serve_h1"


class _Run:
    """A Pearl-C install's run as the arm reads it: the passes it ended and the caps."""

    capacity, flag_buf = 8, [0] * 4

    def __init__(self) -> None:
        self.details: dict[str, Any] = {"passes": [], "pass_ends": "step", "max_step_tokens": 64}
        self.commits = {"coins": {"source": "live", "session": "s-1"}}

    def end_pass(self, step: bool = False) -> None:
        self.details["passes"].append({"units": 0})


class _Option:
    def __init__(self) -> None:
        self.run = _Run()
        self.end_pass = self.run.end_pass
        self.closed = 0

    def commitment(self) -> dict[str, Any]:
        return {"coin_source": "in-process", "salt": "00" * 32}

    def info(self) -> dict[str, Any]:
        return {"challenger": None}


class _Llm:
    def __init__(self) -> None:
        self.option = _Option()
        self.protocols = {"pouw": self.option}
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _pass(tmp_path, i: int, units: int, tiles: int = 1) -> dict[str, Any]:
    (d := tmp_path / f"pass-{i:05d}").mkdir()
    (man := d / "manifest.json").write_text(json.dumps({"matmuls": [{"real_rows": 2, "n": 3, "k": 4}] * units}))
    return {"units": units, "tiles": tiles, "transcript_root": f"r{i}", "manifest": str(man) if units else None}


def test_an_in_process_serving_writes_its_coins_before_the_engine_and_installs_on_the_server_passes(tmp_path):
    """Without the gate the coins are an in-process challenger's (epoch 0), their commitment written now, and the engine
    is built under the server's transcript dir, an owner-only directory under the serve dir's private/."""
    s = PL.Serving(_opts(tmp_path, "--protocols", ENTRY), tmp_path / "out")
    eng = s.engine()
    assert eng["protocols"] == ENTRY and eng["protocols_config"]["pouw"]["transcript_dir"] == str(s.passes)
    assert s.passes == tmp_path / "out" / PL.PRIVATE / "passes" and stat.S_IMODE(s.dir.stat().st_mode) == 0o700
    assert eng["protocols_config"]["pouw"]["coins"].commitment.to_json() == json.loads((s.passes / "coins.json").read_text())
    assert s.reserved is None and PL.serving_of(_opts(tmp_path), tmp_path / "out") is None


def test_a_live_serving_reserves_a_salt_session_under_its_lane_before_the_engine(tmp_path):
    """With the gate the epoch's salt-only session is reserved under --pearl-lane before the engine is built, the
    install given its answer and gate, no in-process coins; its commitment is written once the install opened it."""
    asked: list[tuple[str, ...]] = []

    def call(verb: str, *words: str) -> dict[str, Any]:
        asked.append((verb, *words))
        return {"session": "s-1", "period": None}
    gate = SimpleNamespace(call=call, ledger=lambda: [])
    s = PL.Serving(_opts(tmp_path, "--protocols", ENTRY, *GATE, "--pearl-lane", "compute-accounting"), tmp_path / "out", gate)
    pouw = s.engine()["protocols_config"]["pouw"]
    assert asked[0][:3] == ("reserve", "--peer", "vy-dc-h200") and s.reserved == {"session": "s-1", "period": None}
    assert pouw["session"] == s.reserved and pouw["challenger"] == "127.0.0.1:0" and "coins" not in pouw
    assert not (s.passes / "coins.json").exists()
    llm = _Llm()
    arm = s.arm(llm)
    assert json.loads((s.passes / "coins.json").read_text()) == llm.option.run.commits["coins"]
    assert arm.describe()["coins"] == "challenger" and arm.describe()["lane"] == "compute-accounting"


def test_each_window_names_the_passes_its_steps_ended_and_the_stop_writes_them_all(tmp_path):
    """A window's passes are those ended from its open to its close; a pass outside every window (the learn warm-up's)
    is in the document, not timed. The window line and the summary key the outcome under pouw_pearl."""
    s = PL.Serving(_opts(tmp_path, "--protocols", ENTRY), tmp_path / "out")
    llm = _Llm()
    run = llm.option.run
    ticks = iter(range(100))
    arm = PL.PearlWindows(llm, s, clock=lambda: float(next(ticks)))
    run.details["passes"].append(_pass(tmp_path, 0, 1))               # the warm-up's
    arm.open(0)
    with pytest.raises(RuntimeError, match="opens while window 0's is open"):
        arm.open(1)
    run.details["passes"] += [_pass(tmp_path, 1, 2), _pass(tmp_path, 2, 1), _pass(tmp_path, 3, 1, tiles=99)]
    assert arm.close(0) == {"entry": ENTRY, "window": 0, "passes": [1, 4], "with_calls": 3, "wall_s": 1.0}
    assert arm.close(0) is None
    arm.open(1)
    run.details["passes"].append(_pass(tmp_path, 4, 0))
    arm.close(1)
    arm.drain()
    assert llm.closed == 1
    doc = json.loads((s.dir / "served.json").read_text())
    assert [(p["pass"], p["window"], p["timed"]) for p in doc["passes"]] == [(0, None, False), (1, 0, True), (2, 0, True),
                                                                              (3, 0, True), (4, 1, True)]
    assert [p["macs"] for p in doc["passes"]] == [24, 48, 24, 24, None]
    assert doc["over_caps"] == [3] and doc["passes_with_calls"] == 4 and doc["coins"]["source"] == "in-process"
    assert doc["windows"]["0"]["passes"] == [1, 4] and doc["retained"] is True and doc["passes_dir"] == str(s.passes)
    assert arm.summary() == {"windows": 2, "passes": 5, "with_calls": 4, "served": {
        "path": str(PL.PRIVATE / "served.json"), "passes": 5, "with_calls": 4, "over_caps": 1}}
    assert CommitWindows(object(), tmp_path, run_id="r", window_steps=1, arm=arm).arm_key == "pouw_pearl"
    assert CommitWindows(object(), tmp_path, run_id="r", window_steps=1, arm=object()).arm_key == "pouw_h1"


def test_an_engine_whose_pouw_option_ends_no_pass_is_refused(tmp_path):
    s = PL.Serving(_opts(tmp_path, "--protocols", ENTRY), tmp_path / "out")
    with pytest.raises(ValueError, match="no Pearl-C install"):
        PL.PearlWindows(SimpleNamespace(protocols=None), s)
