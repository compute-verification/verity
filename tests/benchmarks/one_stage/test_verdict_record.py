"""`verdict_record.py`: a stop class's verdict records. A record holds the seven fields the reader of the stop
class's verdicts reads, the four that say what the verdict is of (``registration``, ``serve``, ``class``,
``rate``) and the two that hold its own stream to that class (``stream_registration``, ``stream_names_boundary``);
its docket, class and rate are read from the registered records, the rate never from what the verifier's
summary says of its own design; each piece's word is its command's verdict; and a summary that does not name the
class's docket, a record that is not the one the first-stage file or the summary names, a stream registered
after another record than the class's boundary record, and a second record on one piece and control are refused.
The records are written ones, in the docket's form; one test holds the writer to a stop class's first
stage as the driver writes it and to the stop audit's own summary."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.commitments import docket_digest as DD

HERE = (Path(__file__).resolve().parent)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "benchmarks" / "one_stage"))
VR = importlib.import_module("verity.protocols.proof.spot.verifier.one_stage.verdict_record")

NAMES = {"run": "r-verdict", "serve": "r-serve", "commit": "c0ffee"}
STOP_RULE = {"ids": [3], "max_model_len": 64, "outputs": 4}


def _design(kind: str, k: int | None = None, **more: object) -> dict:
    """A docket's sampling design under its key, with ``k`` where one is given."""
    return {VR.DESIGN: {VR.DESIGN: kind, **({} if k is None else {"k": k}), **more}}


def _record(k: int, n: int, *, stopped: bool | None = True, previous: str | None = None, **window: object) -> dict:
    """A docket of ``k`` of ``n`` proof blocks, in the form `registration.record` writes: a class's
    boundary record, or with ``previous`` the record of a later stream of the class. ``stopped``: None for a serve
    under no stop rule."""
    rule = {} if stopped is None else {"stop": STOP_RULE, "stopped": stopped}
    return {"format": "verity/registration/v1", "population": n, **_design("subset", k), "previous": previous,
            "window": {"kind": "served", "run": "11" * 64, "row": 7, "prompt": 256, "outputs": 64, **rule, **window}}


def _first(k: int = 2, n: int = 6, stop_k: int = 6, **boundary: object) -> dict:
    """A stop class's first-stage file as far as the writer reads it: the boundary record and the stop record after
    it."""
    b = _record(k, n, **boundary)
    s = _record(stop_k, n, previous=DD.digest(b), population="stop")
    return {"record": b, "registration": DD.digest(b), "stop": {"record": s, "registration": DD.digest(s)}}


def _stop_summary(first: dict, accepted: bool = True, complete: bool = True, **more: object) -> dict:
    return {"registration": {"boundary": first["registration"], "stop": first["stop"]["registration"]},
            "accepted": accepted, "complete": complete, **more}


def test_a_stop_records_verdict_names_its_class_serve_registration_and_registered_rate():
    first = _first()
    got = VR.record("stop_record", first, _stop_summary(first), **NAMES)
    assert got == {"piece": "stop_record", "control": None, "run": "r-verdict",
                   "verifier": {"module": "benchmarks/one_stage/stop_audit.py", "commit": "c0ffee"},
                   "verdict": "accept", "proved_by": None, "open": VR.PIECES["stop_record"].open,
                   "registration": first["registration"], "serve": "r-serve", "class": "w7/p256-o64", "rate": "1/1",
                   "stream_registration": first["stop"]["registration"], "stream_names_boundary": True}
    assert got["registration"] != got["stream_registration"], "a class's registration is its boundary record's"
    assert VR.record("stop_record", first, _stop_summary(first), stream=first["stop"]["record"], **NAMES) == got
    words = {(a, c): VR.record("stop_record", first, _stop_summary(first, a, c), **NAMES)["verdict"]
             for a in (True, False) for c in (True, False)}
    assert words == {(True, True): "accept", (True, False): "not complete", (False, True): "refuse", (False, False): "refuse"}
    assert VR.record("stop_record", first, _stop_summary(first, False), control="id", **NAMES)["control"] == "id"


def test_the_rate_is_the_registrations_and_never_the_summarys_word():
    """A stop record that registers 2 of its 6 stop blocks reads 1/3 whatever the summary says its design was, and
    every block registered reads 1/1 at any count."""
    sampled = _first(stop_k=2)
    assert VR.record("stop_record", sampled, _stop_summary(sampled, design="rate:1/1"), **NAMES)["rate"] == "1/3"
    assert [VR.rate(_record(k, n)) for k, n in ((1, 1), (6, 6), (1540, 1540), (154, 1540), (3, 4))] == \
        ["1/1", "1/1", "1/1", "1/10", "3/4"]
    assert VR.rate({"population": 8, **_design("work", 8, strata=[])}) == "work:8", "no rate, so never 1/1"
    for registered in ({"population": 6}, {"population": 6, **_design("subset")}, {"population": 6, **_design("subset", 7)},
                       {"population": 6, **_design("subset", 0)}, {"population": None, **_design("subset", 1)}):
        with pytest.raises(ValueError, match="sampling design|proof blocks"):
            VR.rate(registered)


def test_a_verdicts_own_stream_is_held_to_the_classs_boundary_record():
    """A verdict's own stream is the docket its first-stage file names: a stop audit handed another record gets no
    record."""
    first = _first()
    with pytest.raises(ValueError, match="registered by the first-stage file's stop record, not the record handed"):
        VR.record("stop_record", first, _stop_summary(first), stream=_record(8, 8, previous=first["registration"]),
                  **NAMES)


def test_a_class_is_named_as_its_serve_files_it():
    assert VR.class_of(_record(1, 1)) == "w7/p256-o64"
    assert VR.class_of(_record(1, 1, stopped=False)) == "w7/p256-o64-nostop"
    assert VR.class_of(_record(1, 1, stopped=None)) == "w7/p256-o64", "under no stop rule no class has a stop block"
    assert VR.class_of(_record(1, 1, row=0, prompt=4, outputs=4)) == "w0/p4-o4"
    for window in ({}, {"row": 7, "prompt": 256}, {"row": "7", "prompt": 256, "outputs": 64}):
        with pytest.raises(ValueError, match="names no class"):
            VR.class_of({"window": window})


def test_a_summary_or_a_record_that_is_not_this_classs_is_refused():
    first, other = _first(), _first(row=8)
    for piece, summary, why in (
            ("stop_record", _stop_summary(other), "nothing ties it to the class"),
            ("stop_record", {"accepted": False, "complete": False, "error": "ValueError: refused before it registered"},
             "nothing ties it to the class"),
            ("stop_record", {**_stop_summary(first), "registration": {"boundary": first["registration"]}},
             "nothing ties it to the class")):
        with pytest.raises(ValueError, match=why):
            VR.record(piece, first, summary, **NAMES)
    changed = {**first, "record": {**first["record"], "population": 60}}
    with pytest.raises(ValueError, match="boundary record is not the record whose digest it names"):
        VR.record("stop_record", changed, _stop_summary(first), **NAMES)
    moved = {**first, "stop": {**first["stop"], "record": {**first["stop"]["record"], **_design("subset", 1)}}}
    with pytest.raises(ValueError, match="stop record is not the record whose digest it names"):
        VR.record("stop_record", moved, _stop_summary(first), **NAMES)
    stray = _record(6, 6, previous="5a" * 64)
    elsewhere = {**first, "stop": {"record": stray, "registration": DD.digest(stray)}}
    with pytest.raises(ValueError, match="`previous` is not the class's boundary record"):
        VR.record("stop_record", elsewhere, _stop_summary(elsewhere), **NAMES)
    with pytest.raises(ValueError, match="holds no stop record"):
        VR.record("stop_record", {k: v for k, v in first.items() if k != "stop"}, _stop_summary(first), **NAMES)
    with pytest.raises(ValueError, match="holds no boundary record"):
        VR.record("stop_record", {"error": "the first stage sampled nothing"}, {}, **NAMES)


def test_a_record_names_its_piece_control_run_serve_and_commit():
    first = _first()
    summary = _stop_summary(first)
    with pytest.raises(ValueError, match="one of stop_record, not"):
        VR.record("response_tie", first, summary, **NAMES)
    with pytest.raises(ValueError, match="one of id, max_tokens"):
        VR.record("stop_record", first, summary, control="honest", **NAMES)
    for name in NAMES:
        with pytest.raises(ValueError, match=f"names its {name}"):
            VR.record("stop_record", first, summary, **{**NAMES, name: ""})


def test_the_command_appends_one_line_a_verdict_and_refuses_a_second_on_one_piece_and_control(tmp_path, monkeypatch, capsys):
    first = _first()
    (tmp_path / "first.json").write_text(json.dumps(first))
    (tmp_path / "stop.json").write_text(json.dumps(_stop_summary(first, complete=False)))
    (tmp_path / "forged.json").write_text(json.dumps(_stop_summary(first, False)))
    out = tmp_path / "out"

    def run(piece: str, summary: str, *more: str) -> int:
        return VR.main([piece, "--first", str(tmp_path / "first.json"), "--summary", str(tmp_path / summary),
                        "--serve", "r-serve", "--run", "r-verdict", "--commit", "c0ffee", "--out", str(out), *more])

    assert run("stop_record", "stop.json") == 0
    assert capsys.readouterr().out.startswith('VERDICT {"class": "w7/p256-o64", "control": null,')
    assert run("stop_record", "forged.json", "--control", "id") == 0
    lines = [json.loads(x) for x in (out / VR.FILE).read_text().splitlines()]
    assert [(x["piece"], x["control"], x["verdict"], x["run"]) for x in lines] == [
        ("stop_record", None, "not complete", "r-verdict"), ("stop_record", "id", "refuse", "r-verdict")]
    assert {x["registration"] for x in lines} == {first["registration"]} and {x["serve"] for x in lines} == {"r-serve"}
    assert [x["stream_registration"] for x in lines] == [first["stop"]["registration"]] * 2
    assert all(x["stream_names_boundary"] is True for x in lines)
    capsys.readouterr()
    assert run("stop_record", "stop.json") == 1 and run("stop_record", "forged.json", "--control", "id") == 1
    assert "already holds a verdict on stop_record" in capsys.readouterr().err
    assert len((out / VR.FILE).read_text().splitlines()) == 2, "a refused record adds no line"
    monkeypatch.setattr(VR, "_head", lambda: "")
    for given, lacks in ((("--run", "r-verdict"), "--commit"), (("--commit", "c0ffee"), "--run")):
        with pytest.raises(SystemExit):
            VR.main(["stop_record", "--first", str(tmp_path / "first.json"), "--summary", str(tmp_path / "stop.json"),
                     "--serve", "r-serve", "--out", str(out), *given])
        assert lacks in capsys.readouterr().err
