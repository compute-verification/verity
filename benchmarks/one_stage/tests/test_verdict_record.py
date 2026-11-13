"""`verdict_record.py`: a stop class's verdict records. A record holds the seven fields the reader of the stop
class's verdicts reads, the four that say what the verdict is of (``registration``, ``serve``, ``class``,
``rate``) and the two that hold its own stream to that class (``stream_registration``, ``stream_names_boundary``);
its registration, class and rate are read from the registered records, the rate never from what the verifier's
summary says of its own design; each piece's word is its command's verdict; and a summary that does not name the
class's registration, a record that is not the one the first-stage file or the summary names, a stream registered
after another record than the class's boundary record, and a second record on one piece and control are refused.
The records are written ones, in the registration record's form; one test holds the writer to a stop class's first
stage as the driver writes it and to the stop audit's own summary."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from verity.protocols.proofs.spot.verifier.one_stage import registration as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
VR = importlib.import_module("verdict_record")

NAMES = {"run": "r-verdict", "serve": "r-serve", "commit": "c0ffee"}
STOP_RULE = {"ids": [3], "max_model_len": 64, "outputs": 4}


def _design(kind: str, k: int | None = None, **more: object) -> dict:
    """A registration record's sampling design under its key, with ``k`` where one is given."""
    return {VR.DESIGN: {VR.DESIGN: kind, **({} if k is None else {"k": k}), **more}}


def _record(k: int, n: int, *, stopped: bool | None = True, previous: str | None = None, **window: object) -> dict:
    """A registration record of ``k`` of ``n`` proof blocks, in the form `registration.record` writes: a class's
    boundary record, or with ``previous`` the record of a later stream of the class. ``stopped``: None for a serve
    under no stop rule."""
    rule = {} if stopped is None else {"stop": STOP_RULE, "stopped": stopped}
    return {"format": "verity/registration/v1", "population": n, **_design("subset", k), "previous": previous,
            "window": {"kind": "served", "run": "11" * 64, "row": 7, "prompt": 256, "outputs": 64, **rule, **window}}


def _first(k: int = 2, n: int = 6, stop_k: int = 6, **boundary: object) -> dict:
    """A stop class's first-stage file as far as the writer reads it: the boundary record and the stop record after
    it."""
    b = _record(k, n, **boundary)
    s = _record(stop_k, n, previous=R.digest(b), population="stop")
    return {"record": b, "registration": R.digest(b), "stop": {"record": s, "registration": R.digest(s)}}


def _stop_summary(first: dict, accepted: bool = True, complete: bool = True, **more: object) -> dict:
    return {"registration": {"boundary": first["registration"], "stop": first["stop"]["registration"]},
            "accepted": accepted, "complete": complete, **more}


def _interior(first: dict, n: int = 8) -> dict:
    """The interior registration a second stage registers after the boundary record of ``first``: every one of its
    ``n`` proof blocks."""
    return _record(n, n, previous=first["registration"], interior=True)


def _second_summary(first: dict, held: bool = True, interior: dict | None = None, **outcome: object) -> dict:
    """A second stage's summary on ``first``, naming the interior registration ``interior`` where it registered one."""
    named = {} if interior is None else {"second_stage": {"registration": R.digest(interior)}}
    return {"first_stage": {"registration": first["registration"]}, "held": held, "outcome": outcome, **named}


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


def test_the_second_stages_word_is_its_commands_and_its_rate_the_boundary_records():
    first = _first(k=2, n=6)
    interior = _interior(first)
    words = {name: VR.record("second_stage", first, summary, stream=stream, **NAMES) for name, (summary, stream) in {
        "complete": (_second_summary(first, interior=interior, accepted=True, complete=True), interior),
        "accepted, complete false": (_second_summary(first, interior=interior, accepted=True, complete=False), interior),
        "accepted, no complete": (_second_summary(first, interior=interior, accepted=True), interior),
        "held, its form refused": (_second_summary(first, interior=interior, accepted=False,
                                                   statements=["two-stage-form"]), interior),
        "not held": (_second_summary(first, False, interior, accepted=True, complete=True), interior),
        "not held, no interior": (_second_summary(first, False, accepted=True, complete=True), None),
        "no outcome": ({"first_stage": {"registration": first["registration"]}, "held": False}, None)}.items()}
    assert {name: got["verdict"] for name, got in words.items()} == {
        "complete": "accept", "accepted, complete false": "not complete", "accepted, no complete": "not complete",
        "held, its form refused": "withheld", "not held": "refuse", "not held, no interior": "refuse",
        "no outcome": "refuse"}
    for name, got in words.items():
        assert (got["rate"], got["class"], got["registration"]) == ("1/3", "w7/p256-o64", first["registration"])
        assert got["verifier"]["module"] == "benchmarks/one_stage/two_stage.py" and got["proved_by"] is None
        assert got["open"] == "the two-stage form's soundness is not proved"
        assert (got["stream_registration"], got["stream_names_boundary"]) == (
            (None, None) if name in ("not held, no interior", "no outcome") else (R.digest(interior), True))


def test_a_verdicts_own_stream_is_held_to_the_classs_boundary_record():
    """A verdict names its own stream's registration, the record handed is the one its summary names, and that
    record's ``previous`` is the class's boundary record: a second stage whose interior was registered after another
    record, however right the boundary record its summary names, gets no record."""
    first, other = _first(), _first(row=8)
    interior, strayed = _interior(first), _interior(other)
    held = _second_summary(first, interior=interior, accepted=True, complete=True)
    for summary, stream, why in (
            (held, None, "the record handed .* is none"),
            (held, _interior(first, n=4), "not the record with that digest"),
            (held, first["stop"]["record"], "not the record with that digest"),
            (_second_summary(first, interior=strayed, accepted=True, complete=True), strayed,
             "interior registration's `previous` is not the class's boundary record"),
            (_second_summary(first, accepted=True, complete=True), None, "a second stage that held registered one"),
            (_second_summary(first, False), interior, "names no interior registration .* for the record handed")):
        with pytest.raises(ValueError, match=why):
            VR.record("second_stage", first, summary, stream=stream, **NAMES)
    with pytest.raises(ValueError, match="registered by the first-stage file's stop record, not the record handed"):
        VR.record("stop_record", first, _stop_summary(first), stream=interior, **NAMES)


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
             "nothing ties it to the class"),
            ("second_stage", _second_summary(other, interior=_interior(other), accepted=True),
             "nothing ties it to the class"),
            ("second_stage", {"held": False, "refused": "the first stage sampled nothing"}, "nothing ties it to the class")):
        with pytest.raises(ValueError, match=why):
            VR.record(piece, first, summary, **NAMES)
    changed = {**first, "record": {**first["record"], "population": 60}}
    with pytest.raises(ValueError, match="boundary record is not the record whose digest it names"):
        VR.record("second_stage", changed, _second_summary(first, accepted=True), **NAMES)
    moved = {**first, "stop": {**first["stop"], "record": {**first["stop"]["record"], **_design("subset", 1)}}}
    with pytest.raises(ValueError, match="stop record is not the record whose digest it names"):
        VR.record("stop_record", moved, _stop_summary(first), **NAMES)
    stray = _record(6, 6, previous="5a" * 64)
    elsewhere = {**first, "stop": {"record": stray, "registration": R.digest(stray)}}
    with pytest.raises(ValueError, match="`previous` is not the class's boundary record"):
        VR.record("stop_record", elsewhere, _stop_summary(elsewhere), **NAMES)
    with pytest.raises(ValueError, match="holds no stop record"):
        VR.record("stop_record", {k: v for k, v in first.items() if k != "stop"}, _stop_summary(first), **NAMES)
    with pytest.raises(ValueError, match="holds no boundary record"):
        VR.record("second_stage", {"error": "the first stage sampled nothing"}, {}, **NAMES)


def test_a_record_names_its_piece_control_run_serve_and_commit():
    first = _first()
    summary = _stop_summary(first)
    with pytest.raises(ValueError, match="one of stop_record, second_stage"):
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
    (tmp_path / "interior.json").write_text(json.dumps(_interior(first)))
    (tmp_path / "second.json").write_text(json.dumps(_second_summary(first, interior=_interior(first), accepted=False)))
    out = tmp_path / "out"

    def run(piece: str, summary: str, *more: str) -> int:
        return VR.main([piece, "--first", str(tmp_path / "first.json"), "--summary", str(tmp_path / summary),
                        "--serve", "r-serve", "--run", "r-verdict", "--commit", "c0ffee", "--out", str(out), *more])

    assert run("stop_record", "stop.json") == 0
    assert capsys.readouterr().out.startswith('VERDICT {"class": "w7/p256-o64", "control": null,')
    assert run("stop_record", "forged.json", "--control", "id") == 0
    assert run("second_stage", "second.json") == 1
    assert "`--stream`: its `interior/registration.json`" in capsys.readouterr().err
    assert run("second_stage", "second.json", "--stream", str(tmp_path / "interior.json")) == 0
    lines = [json.loads(x) for x in (out / VR.FILE).read_text().splitlines()]
    assert [(x["piece"], x["control"], x["verdict"], x["run"]) for x in lines] == [
        ("stop_record", None, "not complete", "r-verdict"), ("stop_record", "id", "refuse", "r-verdict"),
        ("second_stage", None, "withheld", "r-verdict")]
    assert {x["registration"] for x in lines} == {first["registration"]} and {x["serve"] for x in lines} == {"r-serve"}
    assert [x["stream_registration"] for x in lines] == [first["stop"]["registration"]] * 2 + [R.digest(_interior(first))]
    assert all(x["stream_names_boundary"] is True for x in lines)
    capsys.readouterr()
    assert run("stop_record", "stop.json") == 1 and run("stop_record", "forged.json", "--control", "id") == 1
    assert "already holds a verdict on stop_record" in capsys.readouterr().err
    assert len((out / VR.FILE).read_text().splitlines()) == 3, "a refused record adds no line"
    monkeypatch.setattr(VR, "_head", lambda: "")
    for given, lacks in ((("--run", "r-verdict"), "--commit"), (("--commit", "c0ffee"), "--run")):
        with pytest.raises(SystemExit):
            VR.main(["stop_record", "--first", str(tmp_path / "first.json"), "--summary", str(tmp_path / "stop.json"),
                     "--serve", "r-serve", "--out", str(out), *given])
        assert lacks in capsys.readouterr().err


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
    (wrote,) = T.TS._first_all(T._args(tmp_path), pubs, T.PAIR, T.RULE, None, T.STOP)
    first = json.loads(json.dumps(wrote))
    design = f"rate:{T.Q}/{T.N}"
    r = SA.resume(first, T.PAIR, T.STOP, design, proof_deadline=T.RULE.proof_deadline)
    audit = {"accepted": True, "complete": False, "verdicts": {}, "checks": {}}
    summary = json.loads(json.dumps(SA.summary_of(r, T.STOP, design, {"accepted": True, "audit": audit})))
    got = VR.record("stop_record", first, summary, **NAMES)
    layout = r.layout
    assert got["class"] == f"w{pubs[0]['window']}/p{layout.prompt}-o{layout.outputs}"
    assert got["registration"] == r.boundary == first["registration"] != r.registration.digest
    assert got["rate"] == VR.rate({"population": layout.spans, **_design("subset", len(r.sample.units))})
    assert (got["verdict"], got["verifier"]["module"]) == ("not complete", SA.TOOL)
    assert (got["stream_registration"], got["stream_names_boundary"]) == (r.registration.digest, True)
    assert r.registration.digest == first["stop"]["registration"] and first["stop"]["record"]["previous"] == r.boundary
    interior = _record(4, 4, previous=r.boundary, interior=True)
    second = {"first_stage": {"registration": r.first.registration.digest}, "held": True, "outcome": {"accepted": False},
              "second_stage": {"registration": R.digest(interior)}}
    assert VR.record("second_stage", first, second, stream=interior, **NAMES)["verdict"] == "withheld"
