"""`h200_share.py`: the number of `accounting/h200-97-percent` is the honest arm A rate over the H200 card's e4m3
ceiling at each reading of its tensor clock, times `compute-accounting/verified-compute-share`, held here to the same
arithmetic in fractions on made-up duty points and `test_compute`'s fixture record (credit three quarters of the
model's MACs). Only arm A on uniform operands at the default clocks is read; a record whose answer isn't an accept
prints no number, and a duty run with no such point can't be read. Under a research run the metric is written once a
reading, marked modeled, the default reading last."""
from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import h200_share as H
import pytest
from test_compute import BENCH, CREDITED, RUN, STAMP, TOTAL, FakeStore, Refused, a_record, stamped, store, to  # noqa: F401

import run as C

DUTY = "r20000101-000002-cccc"
#: 9.6e14 MACs over a 1.2 s host interval: 8e14 MACs a second
HONEST = {"clock": "default", "arm": "A", "class": "uniform", "x": 0, "macs": 9.6e14, "host_t0": 1000.0, "host_t1": 1001.2}
RATE = Fraction(8 * 10 ** 14)
#: points the honest rate must not read: other classes and the locked clock run faster here, and arm B moves no MACs
OTHERS = [{**HONEST, "class": "const", "macs": 1.08e15}, {**HONEST, "clock": "1980", "macs": 1.1e15},
          {**HONEST, "arm": "B", "macs": 0.0}]
CEILING = {"clock.tensor": 132 * 4096 * 1830 * 10 ** 6, "clock.sm": 132 * 4096 * 1980 * 10 ** 6}


def a_duty(tmp_path: Path, points: list[dict]) -> Path:
    record = tmp_path / "duty"
    (record / "out").mkdir(parents=True)
    (record / H.BESIDE).write_text("".join(json.dumps(p) + "\n" for p in points))
    return record


def bench(store: FakeStore, *argv: str) -> int:
    return H.main(list(argv), store_of=lambda: C.EvidenceStore(store, store.pull, Refused, "withdrawn"))


def both(store: FakeStore, tmp_path: Path, points: list[dict] = (HONEST, *OTHERS), **edits) -> None:
    store.add(RUN, a_record(tmp_path, **edits))
    store.add(DUTY, a_duty(tmp_path, list(points)))


def test_the_number_is_the_honest_rate_over_each_readings_ceiling_times_the_verified_share(store, tmp_path, capsys):
    both(store, tmp_path)
    assert bench(store, "--from", RUN, "--duty", DUTY) == 0
    out, err = capsys.readouterr()
    doc = json.loads(out.splitlines()[-1])
    share = Fraction(CREDITED, TOTAL)
    assert share == Fraction(3, 4) and err == ""
    assert list(doc["readings"]) == ["clock.tensor", "clock.sm"]
    for name, r in doc["readings"].items():
        assert r["ceiling_macs_per_s"] == CEILING[name]
        assert r["value"] == pytest.approx(float(RATE / CEILING[name] * share), rel=1e-15)
    assert doc["value"] == doc["readings"]["clock.sm"]["value"]
    assert (doc["honest_rate_macs_per_s"], doc["verified_compute_share"]) == (float(RATE), 0.75)
    assert (doc["run"], doc["duty_run"], doc["unit"], doc["mark"]) == (RUN, DUTY, H.UNIT, "modeled")
    text = out.splitlines()[:-1]
    assert text[0] == ("h200 share: an honest Pearl+ prover is credited 56.05% of an H200's capacity at clock.sm's "
                       "ceiling (modeled)")
    assert text[1] == ("  at clock.tensor (1830 MHz, 989.4 T e4m3 MACs/s): 60.64%, the honest rate being 80.85% of the "
                       "ceiling")


def test_the_honest_rate_is_arm_a_on_uniform_operands_at_the_default_clocks_alone():
    assert H.honest_rate([HONEST, *OTHERS]) == RATE
    assert H.honest_rate([HONEST, {**HONEST, "macs": 4.8e14}]) == RATE
    with pytest.raises(C.Unreadable, match="no arm A point on uniform operands at the default clocks"):
        H.honest_rate(OTHERS)


def test_an_answer_that_is_not_an_accept_prints_no_number(store, tmp_path, capsys):
    both(store, tmp_path, ANSWER=to("verdict", value="reject"))
    assert bench(store, "--from", RUN, "--duty", DUTY) == 1
    out, err = capsys.readouterr()
    assert out == "" and err.startswith("h200 share: no number: ")


def test_a_duty_run_with_no_honest_point_or_no_points_cannot_be_read(store, tmp_path, capsys):
    both(store, tmp_path, OTHERS)
    assert bench(store, "--from", RUN, "--duty", DUTY) == 2
    assert "no arm A point on uniform operands" in capsys.readouterr().err
    store.add(DUTY, a_record(tmp_path / "not-duty"))
    assert bench(store, "--from", RUN, "--duty", DUTY) == 2
    assert f"{DUTY} holds no {H.BESIDE}" in capsys.readouterr().err


def test_under_a_research_run_the_metric_is_written_once_a_reading_modeled_the_default_last(
        store, tmp_path, capsys, monkeypatch, stamped):
    out = tmp_path / "out"
    monkeypatch.setenv("OUT", str(out))
    both(store, tmp_path)
    assert bench(store, "--from", RUN, "--duty", DUTY) == 0
    doc = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert json.loads((out / "h200_share.json").read_text()) == doc
    got = json.loads((out / "result.json").read_text())["measurements"]
    assert [(m["name"], m.get("member"), m["mark"]) for m in got] == [
        (H.METRIC, "clock.tensor", "modeled"), (H.METRIC, "clock.sm", "modeled"),
        ("honest rate", None, "measured"), ("verified compute share", None, "measured")]
    assert H.METRIC == "compute-accounting/h200-capacity-credited-share"
    assert [m["value"] for m in got] == [doc["readings"]["clock.tensor"]["value"], doc["value"], float(RATE), 0.75]
    assert [m["inputs"] for m in got] == [["modeled", DUTY, RUN]] * 2 + [["measured", DUTY], ["measured", RUN]]
    assert all((m["run"], m["commit"], m["host"]) == (BENCH, STAMP["commit"], STAMP) for m in got)
