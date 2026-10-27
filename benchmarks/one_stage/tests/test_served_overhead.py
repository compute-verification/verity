"""`served_overhead.py` on toy serve dirs: a trace splits into its loads at the arrivals with nothing in flight, each load's
time per output token, time to first token and tokens per second come from its arrivals and step outputs, a Commit window's
on-loop close counts only in busy tokens per second, and `table` takes each dir's first load, refuses engines that differ
unless allowed, and puts composed over stock with a delta-method interval."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_overhead as O

HEADER = {"kind": "header", "schema": "verity-vllm/step-trace/v1", "config": {"max_num_batched_tokens": 512, "num_gpu_blocks": 7}}


def _step(t: float, emitted: dict[str, int], finished: tuple[str, ...] = ()) -> dict:
    return {"kind": "step", "t": t, "emitted": emitted, "finished": {r: {"reason": "length"} for r in finished}}


def _load(t0: float, step_s: float, *, window: int | None = None) -> list[dict]:
    """Two requests arriving at t0, a first step at t0 + 1 emitting one token each, then two steps of ``step_s`` for b."""
    recs = [{"kind": "arrive", "request": f"a{t0}", "t": t0}, {"kind": "arrive", "request": f"b{t0}", "t": t0}]
    if window is not None:
        recs.append({"kind": "window", "window": window, "event": "open"})
    return recs + [_step(t0 + 1, {f"a{t0}": 1, f"b{t0}": 1}, (f"a{t0}",)), _step(t0 + 1 + step_s, {f"b{t0}": 1}),
                   _step(t0 + 1 + 2 * step_s, {f"b{t0}": 1}, (f"b{t0}",))]


def _dir(d: Path, trace: list[dict], *, windows: list[dict] = (), engine: dict | None = None) -> Path:
    d.mkdir(parents=True)
    (d / O.TRACE).write_text("".join(json.dumps(r) + "\n" for r in [HEADER, *trace]))
    (d / O.WINDOWS).write_text("".join(json.dumps(w) + "\n" for w in windows))
    e = {"pin": {"model": "m", "revision": "r", "world": 1}, "engine_kwargs": {"max_num_batched_tokens": 512, "model": "/p"},
         "env_flags": {"VLLM_BATCH_INVARIANT": "1"}, "execution": "bi-eager"}
    (d / O.SERVE).write_text(json.dumps({"engine": engine or e, "precision": "verity_int8", "admit": "release-order",
                                         "window_requests": 2, "commit": bool(windows), "run_id": d.name}))
    return d


def test_a_trace_splits_into_loads_and_each_is_timed_from_its_arrivals_and_steps(tmp_path):
    d = _dir(tmp_path / "s", _load(100.0, 0.5) + _load(200.0, 0.25))
    got = O.serve_dir(d)
    assert len(got["loads"]) == 2 and "num_gpu_blocks" not in got["scheduler"]
    first, warm = got["loads"]
    assert (first["requests"], first["tokens"], first["steps"]) == (2, 4, 3)
    assert first["span_s"] == pytest.approx(2.0) and first["time_per_output_token_s"] == pytest.approx(0.5)
    assert first["time_to_first_token_s"] == pytest.approx(1.0) and first["tokens_per_s"] == pytest.approx(2.0)
    assert first["busy_tokens_per_s"] == first["tokens_per_s"] and first["step_s"] == pytest.approx([1.0, 0.5, 0.5])
    assert warm["span_s"] == pytest.approx(1.5)


def test_a_windows_close_on_the_loop_counts_only_in_busy_throughput(tmp_path):
    w = {"window": 0, "seconds": {"open": 0.4, "on_loop": 2.0}, "on_loop_phases_s": {"arm": 1.5}, "pouw_h1": {"tied": True}}
    ld = O.serve_dir(_dir(tmp_path / "c", _load(10.0, 0.5, window=0), windows=[w]))["loads"][0]
    assert ld["tokens_per_s"] == pytest.approx(2.0) and ld["busy_tokens_per_s"] == pytest.approx(1.0)
    assert ld["windows"] == {"n": 1, "open_s": 0.4, "close_on_loop_s": 2.0, "arm_close_s": 1.5, "h1": [True]}


def test_the_table_takes_first_loads_and_puts_composed_over_stock(tmp_path):
    c = _dir(tmp_path / "c", _load(10.0, 1.5, window=0), windows=[{"window": 0, "seconds": {"open": 0.5, "on_loop": 1.0}}])
    stock = [_dir(tmp_path / f"s{i}", _load(100.0, s) + _load(200.0, 0.1)) for i, s in enumerate((0.4, 0.5, 0.6))]
    t = O.table([c], stock)
    r = t["metrics"]["time_per_output_token_s"]
    assert r["composed"] == {"mean": pytest.approx(1.0), "n": 1}
    assert r["stock"]["mean"] == pytest.approx(0.5) and r["stock"]["n"] == 3
    half = O.t95(2) * 2.0 * (r["stock"]["sd"] / 0.5) / 3 ** 0.5
    assert r["ratio"]["mean"] == pytest.approx(2.0) and r["ratio"]["ci95"] == pytest.approx([2.0 - half, 2.0 + half])
    assert r["stock_warm"]["n"] == 3 and r["stock_warm"]["mean"] == pytest.approx(1.2 / 4)
    assert t["composed_parts"]["close_on_loop_s"]["mean"] == pytest.approx(1.0)
    md = O.markdown(t)
    assert "time per output token" in md and "2×" in md


def test_the_table_refuses_engines_that_differ_unless_allowed(tmp_path):
    c = _dir(tmp_path / "c", _load(10.0, 1.0))
    e = {"pin": {"model": "m", "revision": "r", "world": 1}, "engine_kwargs": {"max_num_batched_tokens": 2048},
         "env_flags": {"VLLM_BATCH_INVARIANT": "1"}, "execution": "bi-eager"}
    s = _dir(tmp_path / "s", _load(10.0, 0.5), engine=e)
    with pytest.raises(ValueError, match="engine_kwargs.max_num_batched_tokens"):
        O.table([c], [s])
    t = O.table([c], [s], allow=("engine_kwargs.max_num_batched_tokens",))
    assert t["allowed"] == {str(s): {"engine_kwargs.max_num_batched_tokens": [512, 2048]}}


def test_the_single_protocol_rows_are_the_serving_paths_h200_first():
    rows = [{"protocol": "Commit", "device": "NVIDIA B200", "point": "b1", "over_served": {"time_per_output_token_s": {"mean": 1.1, "n": 6}}},
            {"protocol": "Pearl-C served (cross-check)", "device": "NVIDIA H200"},
            {"protocol": "Commit", "device": "NVIDIA H200", "point": "b8", "over_served": {"time_per_output_token_s": {"mean": 1.2, "n": 6}}}]
    got = O.single_rows({"rows": rows})
    assert [(r["device"], r["time_per_output_token_ratio"]["mean"]) for r in got] == [("NVIDIA H200", 1.2), ("NVIDIA B200", 1.1)]


def test_served_stock_sh_parses():
    assert subprocess.run(["bash", "-n", str(HERE.parent / "served_stock.sh")]).returncode == 0
