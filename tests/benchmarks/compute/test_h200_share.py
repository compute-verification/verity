"""`h200_share.py`: the number of `accounting/h200-97-percent` is the honest arm A rate over the H200 card's e4m3
ceiling at each reading of its tensor clock, times `compute-accounting/verified-compute-share`, held here to the same
arithmetic in fractions on made-up duty points and `test_compute`'s fixture record (`CREDITED` of `TOTAL`). Only arm A on uniform operands at the default clocks is read; a record whose answer isn't an accept
prints no number, and a duty run with no such point can't be read. It reads recorded runs, so it writes no
measurement."""
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
    assert err == ""
    assert list(doc["readings"]) == ["clock.tensor", "clock.sm"]
    for name, r in doc["readings"].items():
        assert r["ceiling_macs_per_s"] == CEILING[name]
        assert r["value"] == pytest.approx(float(RATE / CEILING[name] * share), rel=1e-15)
    assert doc["value"] == doc["readings"]["clock.sm"]["value"]
    assert (doc["honest_rate_macs_per_s"], doc["verified_compute_share"]) == (float(RATE), CREDITED / TOTAL)
    assert (doc["run"], doc["duty_run"], doc["unit"], doc["mark"]) == (RUN, DUTY, H.UNIT, "modeled")
    text = out.splitlines()[:-1]
    assert text[0] == ("h200 share: an honest Pearl+ prover is credited 71.49% of an H200's capacity at clock.sm's "
                       "ceiling (modeled)")
    assert text[1] == ("  at clock.tensor (1830 MHz, 989.4 T e4m3 MACs/s): 77.35%, the honest rate being 80.85% of the "
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


def test_under_a_research_run_it_writes_its_document_and_no_measurement_as_a_reprint(
        store, tmp_path, capsys, monkeypatch, stamped):
    out = tmp_path / "out"
    monkeypatch.setenv("OUT", str(out))
    monkeypatch.setenv("RESEARCH_RUN_ID", BENCH)
    both(store, tmp_path)
    assert bench(store, "--from", RUN, "--duty", DUTY) == 0
    got = capsys.readouterr()
    assert json.loads((out / "h200_share.json").read_text()) == json.loads(got.out.splitlines()[-1])
    assert H.kit.kit_report.REPRINT.format(name="h200_share", run=RUN) in got.err
    assert not (out / "result.json").exists()


def test_it_writes_its_number_through_the_kit_alone(store, tmp_path, capsys, monkeypatch):
    """`kit.report` is the one place that decides whether a command writes a measurement (a reprint of another run's
    record writes none), so the number goes through it and never through the kit's writer directly."""
    calls = []
    monkeypatch.setattr(H.kit, "report", lambda *a, **k: calls.append((a, k)) or 0)
    monkeypatch.setattr(H.kit, "writer", lambda *a, **k: pytest.fail("h200_share.py wrote past kit.report"))
    both(store, tmp_path)
    assert bench(store, "--from", RUN, "--duty", DUTY) == 0
    [((name, text, doc, measured), said)] = calls
    assert (name, text, measured, said) == ("h200_share", H.text(doc), H.measured(doc), {"from_run": RUN})
    assert [(m["name"], m.get("member"), m["mark"], m["inputs"]) for m in measured] == [
        (H.METRIC, "clock.tensor", "modeled", [DUTY, RUN]), (H.METRIC, "clock.sm", "modeled", [DUTY, RUN]),
        ("honest rate", None, "measured", [DUTY]), ("verified compute share", None, "measured", [RUN])]
    assert capsys.readouterr().out == ""
