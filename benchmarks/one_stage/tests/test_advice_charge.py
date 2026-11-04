"""`advice_charge.py` on a serve-openai run's Commit periods' advice. On a real served endpoint's three Commit periods
(network_e2e's endpoint fixture, `verity-vllm advice`'s unpadded output) each Commit period's bits are its advice's
encoding, which its registration binds, and its requests' prompt and output lengths and finishes are counted once
each. On toy Commit periods committed padded, as the server commits them, the bits are still the advice's own, a
Commit period that carries a request in counts only those it admits, a request carried through several one-step Commit
periods counts once, the finishes say eos, length and abort apart, and the Commit periods sum by the execution phase
their first step falls in on the engine clock. A registration that disagrees with the index or binds less than its
advice, a private report of another registration or at another τ, malformed advice, an index of nothing closed, or
advice bound to two compilers writes no record. The record is labelled as its test run's."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from verity_vllm.pipeline import window_advice as WA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import advice_charge as AC  # noqa: E402

ENDPOINT = HERE.parents[2] / "benchmarks" / "network_e2e" / "tests" / "data" / "endpoint" / "advice"
COMPILER = {"name": "toy", "digest": "ab" * 64}
PADDED = 1 << 13


def advice(requests: list[WA.RequestAdvice], before: WA.Advice | None, steps: list[list[tuple[int, int, int]]],
           outputs: int) -> WA.Advice:
    """A Commit period's advice: ``steps`` of (request, start, n) rows, carrying in what ``before`` must leave it."""
    carried = tuple(c.state for c in WA.close(before) if c.must) if before else ()
    return WA.Advice(tuple(requests), tuple(WA.StepAdvice(tuple(
        WA.Row(r, "decode" if n == 1 and s >= requests[r].prompt_len else "prefill", s, n, 0, False, ())
        for r, s, n in rows)) for rows in steps), (0,) * outputs, carried)


def toy(tmp_path: Path, periods: dict[int, dict], refused: dict | None = None, open_: list | None = None) -> Path:
    """An advice dir as the server or `verity-vllm advice` writes it: each Commit period's registration and private
    advice.json, and its report.json where the period gives ``t0``."""
    d = tmp_path / "advice"
    d.mkdir()
    windows = {}
    for k, p in periods.items():
        (d / f"w{k}" / "private").mkdir(parents=True)
        bits = p.get("registered", WA.total_bits(p["advice"]))
        reg = {"binding": {"compiler": p.get("compiler", COMPILER), "advice": {"root": "00" * 64, "bits": bits}}}
        (d / f"w{k}" / "registration.json").write_text(json.dumps(reg))
        (d / f"w{k}" / "private" / "advice.json").write_text(json.dumps(p.get("json") or WA.to_json(p["advice"])))
        if "t0" in p:
            (d / f"w{k}" / "private" / "report.json").write_text(json.dumps({
                "config": p.get("config"), "registration": p.get("report_registration", reg),
                "release": {"rule": "engine clock", "tau_s": p.get("tau", 4.0), "t0": p["t0"]}}))
        windows[str(k)] = {"registration": f"w{k}/registration.json", "root": "00" * 64,
                           "bits": p.get("indexed", bits), "commit": None}
    (d / "windows.json").write_text(json.dumps({"schema": AC.INDEX, "run": "toy", "trace": "toy", "T": 600,
                                                "windows": windows, "refused": refused or {}, "open": open_ or []}))
    return d


def req(i: int, p: int, o: int, finish: str) -> WA.RequestAdvice:
    return WA.RequestAdvice(f"varied.{i}", p, o, finish)


# varied.1 runs on from Commit period 4 into 5, which admits varied.2, which its client aborts after its 2 outputs
R0, R1, R2 = req(0, 20, 1, "eos"), req(1, 30, 3, "length"), req(2, 4, 2, "abort")
P4 = advice([R0, R1], None, [[(0, 0, 20), (1, 0, 30)]], 2)
P5 = advice([R1, R2], P4, [[(0, 30, 1), (1, 0, 4)], [(0, 31, 1), (1, 4, 1)]], 4)
# varied.3 runs through four one-step Commit periods; the second admits varied.4 too
R3, R4 = req(3, 3, 4, "length"), req(4, 5, 1, "eos")
Q0 = advice([R3], None, [[(0, 0, 3)]], 1)
Q1 = advice([R3, R4], Q0, [[(0, 3, 1), (1, 0, 5)]], 2)
Q2 = advice([R3], Q1, [[(0, 4, 1)]], 1)
Q3 = advice([R3], Q2, [[(0, 5, 1)]], 1)


def test_a_real_endpoints_commit_periods_are_its_advices_bits_and_its_requests_once(tmp_path):
    rec = AC.charge(ENDPOINT, "toy-test", 0.1)
    index = json.loads((ENDPOINT / "windows.json").read_text())
    assert [p["bits"] for p in rec["commit_periods"]] == [index["windows"][k]["bits"] for k in ("0", "1", "2")] == [
        p["registered_bits"] for p in rec["commit_periods"]]
    assert rec["bits"] == {"count": 3, "min": 480, "median": 2390, "mean": (480 + 2390 + 2392) / 3, "max": 2392,
                           "total": 480 + 2390 + 2392}
    reqs = [q for k in ("0", "1", "2") for q in json.loads((ENDPOINT / f"w{k}" / "private" / "advice.json").read_text())
            ["requests"]]
    assert rec["prompt_len"]["count"] == rec["output_len"]["count"] == len(reqs) == sum(
        p["admitted"] for p in rec["commit_periods"])
    assert rec["finishes"] == {"eos": sum(q["finish"] == "eos" for q in reqs),
                               "length": sum(q["finish"] == "length" for q in reqs), "abort": 0}
    assert rec["label"].startswith("measured on test run toy-test: honest serving of public inputs")
    assert (rec["format"], rec["T"], rec["tau"], rec["refused"], rec["open"], rec["serving"]) == (
        AC.FORMAT, 600, 0.1, {}, [], None)
    assert (rec["by_phase"], rec["phase_bits"]) == ({"missing": AC.NO_CLOCK}, None)


def test_a_carried_request_counts_once_and_finishes_say_eos_length_and_abort_apart(tmp_path):
    d = toy(tmp_path, {
        4: {"advice": P4, "registered": PADDED, "t0": 100.0},
        5: {"advice": P5, "registered": PADDED, "t0": 101.0, "config": {"max_num_batched_tokens": 512}}},
        refused={"6": "a stop string"}, open_=[7])
    out = tmp_path / "advice-charge.json"
    assert AC.main([str(d), "--run", "toy-test", "--tau", "4", "--T", "32", "--to", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert [(p["period"], p["bits"], p["registered_bits"], p["admitted"], p["carried_in"], p["engine_steps"])
            for p in rec["commit_periods"]] == [
        (4, WA.total_bits(P4), PADDED, 2, 0, 1), (5, WA.total_bits(P5), PADDED, 1, 1, 2)]
    assert rec["finishes"] == {"eos": 1, "length": 1, "abort": 1}
    assert (rec["prompt_len"]["min"], rec["prompt_len"]["max"], rec["output_len"]["total"]) == (4, 30, 6)
    assert (rec["refused"], rec["open"], rec["compiler"]) == ({"6": "a stop string"}, [7], COMPILER)
    assert rec["serving"] == {"max_num_batched_tokens": 512}


def test_a_request_carried_through_one_step_commit_periods_counts_once_and_periods_sum_by_phase(tmp_path):
    d = toy(tmp_path, {k: {"advice": a, "registered": PADDED, "t0": t}
                       for k, (a, t) in enumerate([(Q0, 1000.0), (Q1, 1000.5), (Q2, 1129.0), (Q3, 1300.0)])})
    rec = AC.charge(d, "toy-test", 4.0, 32)
    assert [(p["admitted"], p["carried_in"], p["start_s"]) for p in rec["commit_periods"]] == [
        (1, 0, 0.0), (1, 1, 0.5), (0, 1, 129.0), (0, 1, 300.0)]
    assert (rec["prompt_len"]["count"], rec["output_len"]["total"], rec["finishes"]) == (
        2, 5, {"eos": 1, "length": 1, "abort": 0})
    bits = [WA.total_bits(a) for a in (Q0, Q1, Q2, Q3)]
    assert rec["by_phase"] == [{"phase": 0, "bits": bits[0] + bits[1], "commit_periods": 2},
                               {"phase": 1, "bits": bits[2], "commit_periods": 1},
                               {"phase": 2, "bits": bits[3], "commit_periods": 1}]
    assert (rec["phase_bits"]["count"], rec["phase_bits"]["total"]) == (3, sum(bits))


def overcarried() -> dict:
    doc = WA.to_json(Q2)
    return {**doc, "carried": doc["carried"] * 2}


@pytest.mark.parametrize("periods, why", [
    ({0: {"advice": Q0, "registered": 900, "indexed": 901}},
     "Commit period 0's registration binds 900 bits, and windows.json 901"),
    ({}, "no closed Commit period's advice"),
    ({0: {"advice": Q0}, 1: {"advice": Q1, "compiler": {"name": "other", "digest": "cd" * 64}}},
     "bound to 2 compilers, not one"),
    ({0: {"advice": Q0, "registered": 100}}, f"advice is {WA.total_bits(Q0)} bits, over the 100 its registration binds"),
    ({0: {"advice": Q0, "t0": 0.0, "tau": 0.1}}, "places its outputs at τ = 0.1 s, not 4.0"),
    ({0: {"advice": Q0, "t0": 0.0, "report_registration": {"binding": {}}}},
     "Commit period 0's private report is of another registration"),
    ({0: {"advice": Q2, "json": overcarried()}}, "2 carried requests of 1"),
])
def test_advice_that_disagrees_or_holds_nothing_writes_no_record(tmp_path, capsys, periods, why):
    out = tmp_path / "advice-charge.json"
    assert AC.main([str(toy(tmp_path, periods)), "--run", "toy-test", "--tau", "4", "--to", str(out)]) == 1
    assert why in capsys.readouterr().err and not out.exists()


def test_an_index_that_isnt_vv_advices_writes_no_record(tmp_path, capsys):
    d = tmp_path / "advice"
    shutil.copytree(ENDPOINT, d)
    (d / "windows.json").write_text(json.dumps({"schema": "other"}))
    assert AC.main([str(d), "--run", "x", "--to", str(tmp_path / "o.json")]) == 1
    assert "not verity-vllm/advice-windows-index/v1" in capsys.readouterr().err
