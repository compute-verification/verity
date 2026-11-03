"""`advice_charge.py` on `verity-vllm advice`'s output. On a real served endpoint's three Commit periods (network_e2e's
endpoint fixture) each Commit period's bits are its registration's, and its requests' prompt and output lengths and
finishes are counted once each; a toy Commit period that carries a request in counts only those it admits, and the
finishes say eos, length and abort apart. A registration that disagrees with the index, an index of nothing closed, or advice bound to
two compilers writes no record. The record is labelled as its test run's."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import advice_charge as AC  # noqa: E402

ENDPOINT = HERE.parents[2] / "benchmarks" / "network_e2e" / "tests" / "data" / "endpoint" / "advice"
COMPILER = {"name": "toy", "digest": "ab" * 64}


def toy(tmp_path: Path, periods: dict[int, dict], refused: dict | None = None, open_: list | None = None) -> Path:
    """An advice dir as `verity-vllm advice` writes it: each Commit period's registration and private advice.json."""
    d = tmp_path / "advice"
    d.mkdir()
    windows = {}
    for k, p in periods.items():
        (d / f"w{k}" / "private").mkdir(parents=True)
        reg = {"binding": {"compiler": p.get("compiler", COMPILER), "advice": {"root": "00" * 64, "bits": p["bits"]}},
               "bits_by_window": {"0": p["bits"]}, "T": 600}
        (d / f"w{k}" / "registration.json").write_text(json.dumps(reg))
        (d / f"w{k}" / "private" / "advice.json").write_text(json.dumps(
            {"requests": p["requests"], "carried": p.get("carried", []), "steps": [{}] * p["steps"]}))
        if "config" in p:
            (d / f"w{k}" / "private" / "report.json").write_text(json.dumps({"config": p["config"]}))
        windows[str(k)] = {"registration": f"w{k}/registration.json", "root": "00" * 64,
                           "bits": p.get("indexed", p["bits"]), "commit": None}
    (d / "windows.json").write_text(json.dumps({"schema": AC.INDEX, "run": "toy", "trace": "toy", "T": 600,
                                                "windows": windows, "refused": refused or {}, "open": open_ or []}))
    return d


def req(i: int, p: int, o: int, finish: str) -> dict:
    return {"id": f"varied.{i}", "prompt_len": p, "output_len": o, "finish": finish}


def test_a_real_endpoints_commit_periods_are_its_registrations_bits_and_its_requests_once(tmp_path):
    rec = AC.charge(ENDPOINT, "toy-test", 0.1)
    index = json.loads((ENDPOINT / "windows.json").read_text())
    assert [p["bits"] for p in rec["commit_periods"]] == [index["windows"][k]["bits"] for k in ("0", "1", "2")]
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


def test_a_carried_request_counts_once_and_finishes_say_eos_length_and_abort_apart(tmp_path):
    d = toy(tmp_path, {
        4: {"bits": 900, "steps": 16, "requests": [req(0, 20, 7, "eos"), req(1, 300, 64, "length")]},
        5: {"bits": 700, "steps": 9, "carried": [{"C": 0, "computed": 316, "kept": 16, "made": []}],
            "requests": [req(2, 41, 3, "abort")], "config": {"max_num_batched_tokens": 512}}},
        refused={"6": "a stop string"}, open_=[7])
    out = tmp_path / "advice-charge.json"
    assert AC.main([str(d), "--run", "toy-test", "--to", str(out)]) == 0
    rec = json.loads(out.read_text())
    assert [(p["period"], p["bits"], p["admitted"], p["carried_in"], p["engine_steps"])
            for p in rec["commit_periods"]] == [
        (4, 900, 2, 0, 16), (5, 700, 1, 1, 9)]
    assert rec["finishes"] == {"eos": 1, "length": 1, "abort": 1}
    assert (rec["prompt_len"]["min"], rec["prompt_len"]["max"], rec["output_len"]["total"]) == (20, 300, 74)
    assert (rec["refused"], rec["open"], rec["compiler"]) == ({"6": "a stop string"}, [7], COMPILER)
    assert rec["serving"] == {"max_num_batched_tokens": 512}


@pytest.mark.parametrize("periods, why", [
    ({0: {"bits": 900, "indexed": 901, "steps": 1, "requests": [req(0, 1, 1, "eos")]}},
     "Commit period 0's registration binds 900 bits, and windows.json 901"),
    ({}, "no closed Commit period's advice"),
    ({0: {"bits": 9, "steps": 1, "requests": [req(0, 1, 1, "eos")]},
      1: {"bits": 9, "steps": 1, "requests": [req(1, 1, 1, "eos")], "compiler": {"name": "other", "digest": "cd" * 64}}},
     "bound to 2 compilers, not one"),
])
def test_advice_that_disagrees_or_holds_nothing_writes_no_record(tmp_path, capsys, periods, why):
    out = tmp_path / "advice-charge.json"
    assert AC.main([str(toy(tmp_path, periods)), "--run", "toy-test", "--to", str(out)]) == 1
    assert why in capsys.readouterr().err and not out.exists()


def test_an_index_that_isnt_vv_advices_writes_no_record(tmp_path, capsys):
    d = tmp_path / "advice"
    shutil.copytree(ENDPOINT, d)
    (d / "windows.json").write_text(json.dumps({"schema": "other"}))
    assert AC.main([str(d), "--run", "x", "--to", str(tmp_path / "o.json")]) == 1
    assert "not verity-vllm/advice-windows-index/v1" in capsys.readouterr().err
