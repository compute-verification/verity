"""`served_ab.py` on toy serve dirs and records: a lifetime's record keeps the aggregate, each load's step times and only the
numbers of a window's and a seal's parts; `table` takes each rep round's first load apart from its warm ones, leaves a warm
round out, gives each part as its arm's median less the arm before's, and refuses loads of different shapes."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import served_ab as AB  # noqa: E402  (a script beside the suite, importable only once its dir is on the path)
import served_overhead as O  # noqa: E402

HEADER = {"kind": "header", "schema": "verity-vllm/step-trace/v1", "config": {"max_num_batched_tokens": 512, "num_gpu_blocks": 7}}


def _load(t0: float, first: float, step: float, *, window: int | None = None) -> list[dict]:
    """Two requests arriving at t0, prefilled by a step ending ``first`` later, then two steps of ``step`` each."""
    a, b = f"a{t0}", f"b{t0}"
    recs = [{"kind": "arrive", "request": a, "t": t0}, {"kind": "arrive", "request": b, "t": t0}]
    if window is not None:
        recs.append({"kind": "window", "window": window, "event": "open"})
    step_ = lambda t, done=(): {"kind": "step", "t": t, "emitted": {a: 1, b: 1},   # noqa: E731
                                "finished": {r: {"reason": "length"} for r in done}}
    return recs + [step_(t0 + first), step_(t0 + first + step), step_(t0 + first + 2 * step, (a, b))]


def _dir(d: Path, trace: list[dict], windows: list[dict] = (), seals: list[dict] = (), protocols: dict | None = None,
         listened: float | None = None) -> Path:
    d.mkdir(parents=True)
    (d / O.TRACE).write_text("".join(json.dumps(r) + "\n" for r in [HEADER, *trace]))
    (d / O.WINDOWS).write_text("".join(json.dumps(w) + "\n" for w in windows))
    if seals:
        (d / "seals.jsonl").write_text("".join(json.dumps(s) + "\n" for s in seals))
    e = {"pin": {"model": "m", "revision": "r", "world": 1}, "engine_kwargs": {"max_num_batched_tokens": 512}}
    (d / O.SERVE).write_text(json.dumps({"engine": e, "precision": "bfloat16", "admit": "release-order", "window_requests": 2,
                                         "run_id": d.name, "protocols": protocols,
                                         **({} if listened is None else {"clocks": {"perf_counter": 3.0, "time": listened}})}))
    return d


def _record(out: Path, round_: str, arm: str, d: Path, started: float | None = None) -> None:
    (out / round_).mkdir(parents=True, exist_ok=True)
    (out / round_ / f"{arm}.json").write_text(json.dumps(AB.life(d, arm, round_, started)))


def test_numbers_keeps_numbers_alone():
    got = AB.numbers({"a": 1.5, "b": "a request's text", "c": {"d": 2, "e": ["x"]}, "f": [0.1, 0.2], "g": True, "h": {"i": "x"}})
    assert got == {"a": 1.5, "c": {"d": 2}, "f": [0.1, 0.2]}


def test_a_lifetimes_record_has_step_times_and_the_windows_timed_parts(tmp_path):
    w = {"window": 0, "steps": 3, "passed": True, "requests": ["a prompt's id"], "seconds": {"open": 0.4, "on_loop": 2.0},
         "on_loop_phases_s": {"arm": 1.5}, "pouw_pearl": {"seconds": {"install": 0.3}, "passes": 3, "calls": 9,
                                                          "coin_source": "in-process", "salt": "aa"}}
    d = _dir(tmp_path / "s", _load(10.0, 4.0, 0.5, window=0), [w], [{"window": 0, "seal_s": 1.2, "requests": ["x"],
                                                                     "phases_s": {"store": 0.7}}],
             {"entry": "pouw:x", "ready": {"seconds": {"kernel": 0.2, "lean": 1.0}}})
    doc = AB.life(d, "pearl", "rep-1")
    assert doc["schema"] == AB.LIFE and doc["aggregate"]["schema"] == O.AGGREGATE
    (ld,) = doc["loads"]
    assert ld["step_s"] == pytest.approx([4.0, 0.5, 0.5]) and (ld["requests"], ld["tokens"], ld["steps"]) == (2, 6, 3)
    assert doc["windows"] == [{"window": 0, "steps": 3, "passed": True, "failed": False,
                               "parts": {"seconds": {"open": 0.4, "on_loop": 2.0}, "on_loop_phases_s": {"arm": 1.5}},
                               "pearl": {"seconds": {"install": 0.3}, "passes": 3, "calls": 9, "coin_source": "in-process",
                                         "failed": False}}]
    assert doc["seals"] == [{"window": 0, "seal_s": 1.2, "phases_s": {"store": 0.7}}]
    assert doc["ready"] == {"seconds": {"kernel": 0.2, "lean": 1.0}} and doc["listen_s"] is None
    assert AB.measures(ld) == pytest.approx({"wall_s": 5.0, "first_token_s": 4.0, "decode_step_s": 0.5, "close_after_s": 2.0})


def test_the_table_gives_each_part_as_its_arm_less_the_arm_before(tmp_path):
    out = tmp_path / "out"
    w = {"window": 0, "passed": True, "seconds": {"open": 0.1, "on_loop": 0.0}}
    for i, jitter in enumerate((0.0, 0.2, -0.2), 1):
        _record(out, f"rep-{i}", "stock", _dir(tmp_path / f"s{i}", _load(10.0, 1.0 + jitter, 0.5) + _load(50.0, 0.5, 0.25)))
        _record(out, f"rep-{i}", "commit", _dir(tmp_path / f"c{i}", _load(10.0, 5.0 + jitter, 1.0, window=0)
                                                + _load(50.0, 4.0, 1.0, window=1), [w, {**w, "window": 1}]))
        _record(out, f"rep-{i}", "pearl", _dir(tmp_path / f"p{i}", _load(10.0, 9.0 + jitter, 2.0, window=0), [w]))
    _record(out, "warm-1", "stock", _dir(tmp_path / "sw", _load(10.0, 30.0, 0.5)))
    t = AB.table(out)
    assert t["load"] == {"requests": 2, "tokens": 6, "steps": 3} and t["rounds"] == 3 and not t["failed_windows"]
    first = t["first"]
    assert first["top"] == "pearl" and list(first["arms"]) == ["stock", "commit", "pearl"]
    assert first["arms"]["stock"]["wall_s"] == pytest.approx({"n": 3, "median": 2.0, "min": 1.8, "max": 2.2})
    assert first["arms"]["pearl"]["over_stock"] == pytest.approx(13.0 / 2.0)
    stock, commit, pearl = first["parts"]
    assert stock["share_of_top"] == pytest.approx(2.0 / 13.0)
    assert commit["wall_s"] == pytest.approx(5.0) and commit["share_of_top"] == pytest.approx(5.0 / 13.0)
    assert commit["over_stock"] == pytest.approx(2.5) and commit["first_token_s"] == pytest.approx(4.0)
    assert commit["decode_step_s"] == pytest.approx(0.5)
    assert pearl["wall_s"] == pytest.approx(6.0) and pearl["decode_step_s"] == pytest.approx(1.0)
    assert stock["share_of_top"] + commit["share_of_top"] + pearl["share_of_top"] == pytest.approx(1.0)
    warm = t["warm"]
    assert list(warm["arms"]) == ["stock", "commit"] and warm["top"] == "commit"
    assert warm["arms"]["commit"]["wall_s"]["median"] == pytest.approx(6.0) and warm["parts"][1]["wall_s"] == pytest.approx(5.0)
    md = AB.markdown(t)
    assert "| the Commit | 5 | 38.5% | 250% | 4 | 0.5 |" in md and "| pearl | 3 |" in md and "6.5×" in md


def test_the_seconds_from_the_servers_start_to_listening_are_a_lifetimes_and_a_column_beside_the_first_loads(tmp_path):
    """``listen_s`` is the server's own time as it begins to listen less the time the wrapper started it; the table
    gives it for an arm whose every lifetime recorded it, beside the first loads alone, so seconds an arm moved from
    its first load to before it listens show in the row they left and the column they reached."""
    out = tmp_path / "out"
    w = {"window": 0, "passed": True, "seconds": {"open": 0.1, "on_loop": 0.0}}
    for i, (first, listened) in enumerate(((9.0, 1040.0), (5.0, 1044.5), (5.5, 1044.0)), 1):
        d = _dir(tmp_path / f"p{i}", _load(10.0, first, 2.0, window=0) + _load(50.0, 4.0, 2.0, window=1),
                 [w, {**w, "window": 1}], listened=listened)
        assert AB.life(d, "pearl", f"rep-{i}", 1000.0)["listen_s"] == pytest.approx(listened - 1000.0)
        assert AB.life(d, "pearl", f"rep-{i}")["listen_s"] is None
        _record(out, f"rep-{i}", "pearl", d, 1000.0)
        _record(out, f"rep-{i}", "stock", _dir(tmp_path / f"s{i}", _load(10.0, 1.0, 0.5), listened=1030.0),
                1000.0 if i < 3 else None)
    t = AB.table(out)
    pearl, stock = t["first"]["arms"]["pearl"], t["first"]["arms"]["stock"]
    assert pearl["listen_s"] == pytest.approx({"n": 3, "median": 44.0, "min": 40.0, "max": 44.5}) and "listen_s" not in stock
    assert "listen_s" not in t["warm"]["arms"]["pearl"]
    first, warm = AB.markdown(t).split("**Later loads, warm**")
    assert "| × stock | server start to listening, s |" in first and "| 44 (40 to 44.5) |" in first
    assert "server start to listening" not in warm
    assert next(x for x in first.splitlines() if x.startswith("| stock |")).endswith("|  |")


def test_a_part_is_left_out_when_the_arm_before_it_did_not_run(tmp_path):
    out = tmp_path / "out"
    w = {"window": 0, "passed": False, "seconds": {"open": 0.1, "on_loop": 0.0}}
    _record(out, "rep-1", "stock", _dir(tmp_path / "s", _load(10.0, 1.0, 0.5)))
    _record(out, "rep-1", "pearl", _dir(tmp_path / "p", _load(10.0, 9.0, 2.0, window=0), [w]))
    t = AB.table(out)
    assert [p["arm"] for p in t["first"]["parts"]] == ["stock"] and t["failed_windows"] == ["rep-1 pearl window 0"]
    assert "didn't pass: rep-1 pearl window 0" in AB.markdown(t)


def test_the_table_refuses_no_record_and_loads_of_different_shapes(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="no rep-<i>/<arm>.json"):
        AB.table(out)
    _record(out, "rep-1", "stock", _dir(tmp_path / "s", _load(10.0, 1.0, 0.5)))
    short = _load(10.0, 1.0, 0.5)[:-1]
    short[-1]["finished"] = {r: {"reason": "length"} for r in short[-1]["emitted"]}
    _record(out, "rep-2", "stock", _dir(tmp_path / "t", short))
    with pytest.raises(ValueError, match="different"):
        AB.table(out)
    assert AB.main(["table", str(out)]) == 1
