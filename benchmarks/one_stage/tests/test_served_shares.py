"""`served_shares.py` on toy serve dirs: a full-prefill step of 8,192 tokens of the served 1B declares 8.617 ms at the
tensor clock, the declared utilization is the steps' ideal seconds over the loads' busy wall (a Commit window's on-loop
close in it), term by term, the credited share is Pearl+'s credited MACs over the same wall (none for no credit or a
rejected answer), the prover's replay is a CPU term beside them (the cores it takes to keep pace), and a row with no
start position or an undeclared model is refused."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_shares as S  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)

MODEL = "verity-int8/Llama-3.2-1B-INT8"
HEADER = {"kind": "header", "schema": "verity-vllm/step-trace/v1", "config": {"max_num_batched_tokens": 8192}}
RHO = S.RHO["tensor"]


def _prefill(t0: float, step_s: float, n: int = 8, length: int = 1024, start: int | None = 0) -> list[dict]:
    """`n` requests of `length` prompt ids arriving at t0, one step at t0 + step_s prefilling all of them and emitting one
    token each, which finishes them."""
    ids = [f"r{t0}-{i}" for i in range(n)]
    return [*({"kind": "arrive", "request": r, "t": t0} for r in ids),
            {"kind": "step", "step": 0, "T": n * length, "t": t0 + step_s, "rows": [[r, start, length] for r in ids],
             "emitted": {r: 1 for r in ids}, "finished": {r: {"reason": "length"} for r in ids}}]


def _dir(d: Path, trace: list[dict], windows: list[dict] = (), model: str = MODEL) -> Path:
    d.mkdir(parents=True)
    (d / S.O.TRACE).write_text("".join(json.dumps(r) + "\n" for r in [HEADER, *trace]))
    (d / S.O.WINDOWS).write_text("".join(json.dumps(w) + "\n" for w in windows))
    (d / S.O.SERVE).write_text(json.dumps({"engine": {"pin": {"model": model}}}))
    return d


def test_full_prefill_step_declares_its_ideal_work(tmp_path):
    got = S.shares(_dir(tmp_path / "s", _prefill(0.0, 0.0274)))
    tensor = got["readings"]["tensor"]
    assert tensor["clock_mhz"] == 1830 and tensor["ideal_s"] == pytest.approx(8.617e-3, rel=1e-3)
    assert got["wall_s"] == pytest.approx(0.0274)
    assert tensor["declared"] == pytest.approx(8.617e-3 / 0.0274, rel=1e-3)
    assert got["readings"]["boost"]["declared"] < tensor["declared"]
    assert got["macs"]["linears"] == 8192 * 16 * (3072 * 2048 + 2048 * 2048 + 16384 * 2048 + 2048 * 8192)
    assert got["macs"]["lm_head"] == 8 * 2048 * 128256
    assert sum(tensor["terms"].values()) == pytest.approx(tensor["declared"]) and tensor["terms"]["linears"] > 0.9 * tensor["declared"]
    assert tensor["credited"] == 0 and got["credit"]["verdict"] is None


def test_wall_sums_loads_and_on_loop_closes(tmp_path):
    first = _prefill(0.0, 0.1)
    first.insert(8, {"kind": "window", "window": 3, "event": "open"})
    got = S.shares(_dir(tmp_path / "s", first + _prefill(10.0, 0.1), [{"window": 3, "seconds": {"on_loop": 0.3}}]))
    assert len(got["loads"]) == 2 and [ld["wall_s"] for ld in got["loads"]] == pytest.approx([0.4, 0.1])
    assert got["wall_s"] == pytest.approx(0.5)
    one = S.ideal_s(got["loads"][1]["macs"], RHO)
    assert got["readings"]["tensor"]["declared"] == pytest.approx(2 * one / 0.5)


def test_credited_share_is_pearl_plus_credited_macs(tmp_path):
    d = _dir(tmp_path / "s", _prefill(0.0, 1.0), model="verity-fp8/Llama-3.2-1B-FP8")
    answer = {"protocol": "verity/pouw/pearl-plus/v1", "verdict": "accept", "reason": None, "commitment": "00",
              "public": {"work": {"credited": 3 * 10**12, "credited_tiles": 9, "macs": 4 * 10**12}}, "profile": {"delta": "1/100"}}
    got = S.shares(d, answer)
    tensor = got["readings"]["tensor"]
    assert tensor["credited"] == pytest.approx(3e12 / RHO) and got["credit"]["of_linears"] == 3e12 / got["macs"]["linears"]
    rejected = S.shares(d, answer | {"verdict": "reject", "public": None, "profile": None, "reason": "tile 3"})
    assert rejected["readings"]["tensor"]["credited"] == 0 and rejected["credit"]["verdict"] == "reject"
    with pytest.raises(ValueError, match="credited"):
        S.shares(d, answer | {"public": {"work": [3]}})


def test_the_provers_replay_is_a_cpu_term_beside_the_gpu_shares(tmp_path):
    d = _dir(tmp_path / "s", _prefill(0.0, 2.0), model="verity-fp8/Llama-3.2-1B-FP8")
    plain = S.shares(d)
    got = S.shares(d, replay=(110.0, 32))
    assert plain["replay"] is None and got["readings"] == plain["readings"]
    assert got["replay"] == {"s": 110.0, "cpus": 32, "core_s": 3520.0, "of_wall": 55.0, "cores_to_keep_pace": 1760.0}
    with pytest.raises(ValueError, match="cpus > 0"):
        S.shares(d, replay=(110.0, 0))


def test_refuses_what_it_cannot_count(tmp_path):
    with pytest.raises(ValueError, match="no start position"):
        S.shares(_dir(tmp_path / "a", _prefill(0.0, 1.0, start=None)))
    with pytest.raises(ValueError, match="no declared shapes"):
        S.shares(_dir(tmp_path / "b", _prefill(0.0, 1.0), model="other/model"))


def test_cli(tmp_path):
    d = _dir(tmp_path / "s", _prefill(0.0, 0.0274))
    script = str(HERE.parent / "served_shares.py")
    out = subprocess.run([sys.executable, script, str(d)], capture_output=True, text=True, check=True)
    assert json.loads(out.stdout)["schema"] == S.SCHEMA
    out = subprocess.run([sys.executable, script, str(d), "--replay", "110", "--replay-cpus", "32"], capture_output=True,
                         text=True, check=True)
    assert json.loads(out.stdout)["replay"]["core_s"] == 3520.0
    half = subprocess.run([sys.executable, script, str(d), "--replay", "110"], capture_output=True, text=True)
    assert half.returncode == 2 and "go together" in half.stderr
    bad = subprocess.run([sys.executable, script, str(tmp_path / "none")], capture_output=True, text=True)
    assert bad.returncode == 2 and "served_shares.py:" in bad.stderr
