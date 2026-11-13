"""`verdict_record.py`: a stop class's verdict records. A record holds the seven fields the reader of the stop
class's verdicts reads and the four that say what the verdict is of (``registration``, ``serve``, ``class``,
``rate``); its registration, class and rate are read from the registered records, the rate never from what the
verifier's summary says of its own design; each piece's word is its command's verdict; and a summary that does not
name the class's registration, a record that is not the one ``stage1.json`` names, and a second record on one piece
and control are refused. The records are written ones, in the registration record's form; one test holds the writer to
a stop class's first stage as the driver writes it and to the stop audit's own summary."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.protocols.spot.verifier.one_stage import registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
VR = importlib.import_module("verdict_record")

NAMES = {"run": "r-verdict", "serve": "r-serve", "commit": "c0ffee"}
STOP_RULE = {"ids": [3], "max_model_len": 64, "outputs": 4}


def _record(k: int, n: int, *, stopped: bool | None = True, previous: str | None = None, **window: object) -> dict:
    """A registration record of ``k`` of ``n`` proof blocks, in the form `registration.record` writes: a class's
    boundary record, or with ``previous`` the record of a later stream of the class. ``stopped``: None for a serve
    under no stop rule."""
    rule = {} if stopped is None else {"stop": STOP_RULE, "stopped": stopped}
    return {"format": "verity/registration/v1", "population": n, "law": {"law": "subset", "k": k}, "previous": previous,
            "window": {"kind": "served", "run": "11" * 64, "row": 7, "prompt": 256, "outputs": 64, **rule, **window}}


def _stage1(k: int = 2, n: int = 6, stop_k: int = 6, **boundary: object) -> dict:
    """A stop class's ``stage1.json`` as far as the writer reads it: the boundary record and the stop record after it."""
    b = _record(k, n, **boundary)
    s = _record(stop_k, n, previous=R.digest(b), population="stop")
    return {"record": b, "registration": R.digest(b), "stop": {"record": s, "registration": R.digest(s)}}


def _stop_summary(stage1: dict, accepted: bool = True, complete: bool = True, **more: object) -> dict:
    return {"registration": {"boundary": stage1["registration"], "stop": stage1["stop"]["registration"]},
            "accepted": accepted, "complete": complete, **more}


def _second_summary(stage1: dict, held: bool = True, **outcome: object) -> dict:
    return {"first_stage": {"registration": stage1["registration"]}, "held": held, "outcome": outcome}


def test_a_stop_records_verdict_names_its_class_serve_registration_and_registered_rate():
    stage1 = _stage1()
    got = VR.record("stop_record", stage1, _stop_summary(stage1), **NAMES)
    assert got == {"piece": "stop_record", "control": None, "run": "r-verdict",
                   "verifier": {"module": "benchmarks/one_stage/stop_audit.py", "commit": "c0ffee"},
                   "verdict": "accept", "proved_by": None, "open": VR.PIECES["stop_record"].open,
                   "registration": stage1["registration"], "serve": "r-serve", "class": "w7/p256-o64", "rate": "1/1"}
    assert got["registration"] != stage1["stop"]["registration"], "a class's registration is its boundary record's"
    words = {(a, c): VR.record("stop_record", stage1, _stop_summary(stage1, a, c), **NAMES)["verdict"]
             for a in (True, False) for c in (True, False)}
    assert words == {(True, True): "accept", (True, False): "not complete", (False, True): "refuse", (False, False): "refuse"}
    assert VR.record("stop_record", stage1, _stop_summary(stage1, False), control="id", **NAMES)["control"] == "id"


def test_the_rate_is_the_registrations_and_never_the_summarys_word():
    """A stop record that registers 2 of its 6 stop blocks reads 1/3 whatever the summary says its design was, and
    every block registered reads 1/1 at any count."""
    sampled = _stage1(stop_k=2)
    assert VR.record("stop_record", sampled, _stop_summary(sampled, design="rate:1/1"), **NAMES)["rate"] == "1/3"
    assert [VR.rate(_record(k, n)) for k, n in ((1, 1), (6, 6), (1540, 1540), (154, 1540), (3, 4))] == \
        ["1/1", "1/1", "1/1", "1/10", "3/4"]
    assert VR.rate({"population": 8, "law": {"law": "work", "k": 8, "strata": []}}) == "work:8", "no rate, so never 1/1"
    for registered in ({"population": 6}, {"population": 6, "law": {"law": "subset"}},
                       {"population": 6, "law": {"law": "subset", "k": 7}}, {"population": 6, "law": {"law": "subset", "k": 0}},
                       {"population": None, "law": {"law": "subset", "k": 1}}):
        with pytest.raises(ValueError, match="sampling design|proof blocks"):
            VR.rate(registered)


def test_the_second_stages_word_is_its_commands_and_its_rate_the_boundary_records():
    stage1 = _stage1(k=2, n=6)
    words = {name: VR.record("second_stage", stage1, summary, **NAMES) for name, summary in {
        "complete": _second_summary(stage1, accepted=True, complete=True),
        "accepted, complete false": _second_summary(stage1, accepted=True, complete=False),
        "accepted, no complete": _second_summary(stage1, accepted=True),
        "held, its form refused": _second_summary(stage1, accepted=False, statements=["two-stage-form"]),
        "not held": _second_summary(stage1, False, accepted=True, complete=True),
        "no outcome": {"first_stage": {"registration": stage1["registration"]}, "held": False}}.items()}
    assert {name: got["verdict"] for name, got in words.items()} == {
        "complete": "accept", "accepted, complete false": "not complete", "accepted, no complete": "not complete",
        "held, its form refused": "withheld", "not held": "refuse", "no outcome": "refuse"}
    for got in words.values():
        assert (got["rate"], got["class"], got["registration"]) == ("1/3", "w7/p256-o64", stage1["registration"])
        assert got["verifier"]["module"] == "benchmarks/one_stage/two_stage.py" and got["proved_by"] is None
        assert got["open"] == "the two-stage form's soundness is not proved"


def test_a_class_is_named_as_its_serve_files_it():
    assert VR.class_of(_record(1, 1)) == "w7/p256-o64"
    assert VR.class_of(_record(1, 1, stopped=False)) == "w7/p256-o64-nostop"
    assert VR.class_of(_record(1, 1, stopped=None)) == "w7/p256-o64", "under no stop rule no class has a stop block"
    assert VR.class_of(_record(1, 1, row=0, prompt=4, outputs=4)) == "w0/p4-o4"
    for window in ({}, {"row": 7, "prompt": 256}, {"row": "7", "prompt": 256, "outputs": 64}):
        with pytest.raises(ValueError, match="names no class"):
            VR.class_of({"window": window})


def test_a_summary_or_a_record_that_is_not_this_classs_is_refused():
    stage1, other = _stage1(), _stage1(row=8)
    for piece, summary, why in (
            ("stop_record", _stop_summary(other), "nothing ties it to the class"),
            ("stop_record", {"accepted": False, "complete": False, "error": "ValueError: refused before it registered"},
             "nothing ties it to the class"),
            ("stop_record", {**_stop_summary(stage1), "registration": {"boundary": stage1["registration"]}},
             "nothing ties it to the class"),
            ("second_stage", _second_summary(other, accepted=True), "nothing ties it to the class"),
            ("second_stage", {"held": False, "refused": "the first stage sampled nothing"}, "nothing ties it to the class")):
        with pytest.raises(ValueError, match=why):
            VR.record(piece, stage1, summary, **NAMES)
    changed = {**stage1, "record": {**stage1["record"], "population": 60}}
    with pytest.raises(ValueError, match="boundary record is not the record whose digest it names"):
        VR.record("second_stage", changed, _second_summary(stage1, accepted=True), **NAMES)
    moved = {**stage1, "stop": {**stage1["stop"], "record": {**stage1["stop"]["record"], "law": {"law": "subset", "k": 1}}}}
    with pytest.raises(ValueError, match="stop record is not the record whose digest it names"):
        VR.record("stop_record", moved, _stop_summary(stage1), **NAMES)
    stray = _record(6, 6, previous="5a" * 64)
    elsewhere = {**stage1, "stop": {"record": stray, "registration": R.digest(stray)}}
    with pytest.raises(ValueError, match="`previous` is not the class's boundary record"):
        VR.record("stop_record", elsewhere, _stop_summary(elsewhere), **NAMES)
    with pytest.raises(ValueError, match="holds no stop record"):
        VR.record("stop_record", {k: v for k, v in stage1.items() if k != "stop"}, _stop_summary(stage1), **NAMES)
    with pytest.raises(ValueError, match="holds no boundary record"):
        VR.record("second_stage", {"error": "the first stage sampled nothing"}, {}, **NAMES)


def test_a_record_names_its_piece_control_run_serve_and_commit():
    stage1 = _stage1()
    summary = _stop_summary(stage1)
    with pytest.raises(ValueError, match="one of stop_record, second_stage"):
        VR.record("response_tie", stage1, summary, **NAMES)
    with pytest.raises(ValueError, match="one of id, max_tokens"):
        VR.record("stop_record", stage1, summary, control="honest", **NAMES)
    for name in NAMES:
        with pytest.raises(ValueError, match=f"names its {name}"):
            VR.record("stop_record", stage1, summary, **{**NAMES, name: ""})


def test_the_command_appends_one_line_a_verdict_and_refuses_a_second_on_one_piece_and_control(tmp_path, monkeypatch, capsys):
    stage1 = _stage1()
    (tmp_path / "stage1.json").write_text(json.dumps(stage1))
    (tmp_path / "stop.json").write_text(json.dumps(_stop_summary(stage1, complete=False)))
    (tmp_path / "forged.json").write_text(json.dumps(_stop_summary(stage1, False)))
    (tmp_path / "second.json").write_text(json.dumps(_second_summary(stage1, accepted=False)))
    out = tmp_path / "out"
    monkeypatch.setenv(VR.RUN_ENV, "r-env")

    def run(piece: str, summary: str, *more: str) -> int:
        return VR.main([piece, "--stage1", str(tmp_path / "stage1.json"), "--summary", str(tmp_path / summary),
                        "--serve", "r-serve", "--commit", "c0ffee", "--out", str(out), *more])

    assert run("stop_record", "stop.json") == 0
    assert capsys.readouterr().out.startswith('VERDICT {"class": "w7/p256-o64", "control": null,')
    assert run("stop_record", "forged.json", "--control", "id") == 0 and run("second_stage", "second.json", "--run", "r-given") == 0
    lines = [json.loads(x) for x in (out / VR.FILE).read_text().splitlines()]
    assert [(x["piece"], x["control"], x["verdict"], x["run"]) for x in lines] == [
        ("stop_record", None, "not complete", "r-env"), ("stop_record", "id", "refuse", "r-env"),
        ("second_stage", None, "withheld", "r-given")]
    assert {x["registration"] for x in lines} == {stage1["registration"]} and {x["serve"] for x in lines} == {"r-serve"}
    capsys.readouterr()
    assert run("stop_record", "stop.json") == 1 and run("stop_record", "forged.json", "--control", "id") == 1
    assert "already holds a verdict on stop_record" in capsys.readouterr().err
    assert len((out / VR.FILE).read_text().splitlines()) == 3, "a refused record adds no line"
    monkeypatch.delenv(VR.RUN_ENV)
    monkeypatch.setattr(VR, "_head", lambda: "")
    with pytest.raises(SystemExit):
        VR.main(["stop_record", "--stage1", str(tmp_path / "stage1.json"), "--summary", str(tmp_path / "stop.json"),
                 "--serve", "r-serve", "--out", str(out)])
    err = capsys.readouterr().err
    assert "--run" in err and "--commit" in err


@pytest.mark.slow
def test_the_writer_reads_a_stop_class_as_the_driver_and_the_stop_audit_write_it(tmp_path, monkeypatch):
    """The fixtures above are written in the form the writer expects, so they can't show that the form is the
    drivers'. This registers a stop class's first stage on a test's challenger as `two_stage.py first --stop` does,
    builds its stop stream again as the stop audit does, and takes `stop_audit.summary_of`'s own summary: the record
    names the class's boundary record, the directory the class is filed under and the rate its stop record registers.
    No session is proved and Lean is not called."""
    T = importlib.import_module("test_two_stage_driver")
    SA = importlib.import_module("stop_audit")
    pubs = [json.loads(json.dumps(T._stop_class().public()))]
    monkeypatch.setattr(T.AC, "start", T._start(T.OneSession("verdict-record")))
    (first,) = T.TS._first_all(T._args(tmp_path), pubs, T.PAIR, T.RULE, None, T.STOP)
    stage1 = json.loads(json.dumps(first))
    design = f"rate:{T.Q}/{T.N}"
    r = SA.resume(stage1, T.PAIR, T.STOP, design, proof_deadline=T.RULE.proof_deadline)
    audit = {"accepted": True, "complete": False, "verdicts": {}, "checks": {}}
    summary = json.loads(json.dumps(SA.summary_of(r, T.STOP, design, {"accepted": True, "audit": audit})))
    got = VR.record("stop_record", stage1, summary, **NAMES)
    layout = r.layout
    assert got["class"] == f"w{pubs[0]['window']}/p{layout.prompt}-o{layout.outputs}"
    assert got["registration"] == r.boundary == stage1["registration"] != r.registration.digest
    assert got["rate"] == VR.rate({"population": layout.spans, "law": {"law": "subset", "k": len(r.sample.units)}})
    assert (got["verdict"], got["verifier"]["module"]) == ("not complete", SA.TOOL)
    second = {"first_stage": {"registration": r.first.registration.digest}, "held": True, "outcome": {"accepted": False}}
    assert VR.record("second_stage", stage1, second, **NAMES)["verdict"] == "withheld"
