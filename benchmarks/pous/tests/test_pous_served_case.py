"""`served_case.py`, PoUS's and the wipe's pieces for the composite, built from what a served window kept.

PoUS's window is `served_piece.py`'s, run on the CPU over the tiny int8 checkpoint at the small point with the PoUS tests'
signer (`test_pous_served_piece`'s fixture); the builder encodes ``C*`` again from the same snapshot on the CPU. The wipe's
is one wipe in this process at a small fill (`wipe_period.run`), laid out as `streamed_wipe/served_wipe.sh` leaves it:
``answer.json`` with the run's placement in its profile, ``served.json`` and the leg's ``wipe.case.json``. Each piece is
held to the composite's form (`benchmarks/one_stage/composite.py`'s ``piece``: a case of commitment, evidence and params,
an answer whose commitment is the digest of the case's commitment), and a kept output that was changed after the run is
refused with its reason before anything is written.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pous import memory_challenger as mc
from verity.core.protocols.pous import streamed_wipe as sw
from verity.core.protocols.pous import wipe_period as W
from verity.ml.commitments.identity import canonical_json_bytes

pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[3]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SPT = _load("pous_served_piece_tests", ROOT / "benchmarks" / "pous" / "tests" / "test_pous_served_piece.py")
SC = _load("pous_served_case", ROOT / "benchmarks" / "pous" / "served_case.py")
ts, served = SPT.ts, SPT.served
WIPE_ID = mc.TEST_PREFIX + W.PROTOCOL
SMALL_WIPE = sw.WipeParams(64, W.READS, 1, 1_000, (("gpu", 64 * sw.LABEL_BYTES),))


def _mode(p: Path) -> int:
    return stat.S_IMODE(p.stat().st_mode)


def _listing(d: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(d.iterdir())}


def _pous(s, **kw):
    return SC.pous_piece(s.out, schemes=ts.SCHEMES, hf_home=s.hf, device="cpu", pins=s.pins, **kw)


def _edit(path: Path, change) -> None:
    doc = json.loads(path.read_text())
    change(doc)
    path.write_text(json.dumps(doc))


def _hexjson(h: str) -> dict:
    return json.loads(bytes.fromhex(h))


def _rehex(doc) -> str:
    return canonical_json_bytes(doc).hex()


def test_pous_case_is_the_composites_piece_and_its_answer_the_runs(served, tmp_path, monkeypatch):
    """The case is the run protocol's (`period.case`, setup's own inputs encoded again at the kept salt) and its answer
    the run's own, about the digest of setup's published outputs; both in the private directory, which stays 0700 with
    its files 0600, the public window untouched."""
    answer, _ = SPT._run(served, tmp_path, monkeypatch)
    assert answer["verdict"] == "accept", answer["reason"]
    public = _listing(served.out)
    private, got = _pous(served)
    assert private == served.private and _listing(served.out) == public
    case = json.loads((private / "case.json").read_text())
    assert sorted(case) == ["commitment", "evidence", "params"]
    assert json.loads((private / "answer.json").read_text()) == got
    run = json.loads((private / "run.json").read_text())
    assert got == run["answer"] and got["protocol"] == SPT.TEST_PROTOCOL and got["verdict"] == "accept"
    assert _hexjson(case["commitment"]) == run["published"]
    assert got["commitment"] == SC.digest(case) == hashlib.sha512(canonical_json_bytes(run["published"])).hexdigest()
    assert got["commitment"] != answer["commitment"]                              # the public one is the hiding root
    assert case["params"] == {"point": run["point"], "session_id": run["session_id"], "trusted": run["trusted"]}
    ev = _hexjson(case["evidence"])
    assert sorted(ev) == ["digests", "session", "setup"] and ev["session"] == run["session"]
    assert bytes.fromhex(ev["digests"]) == (private / "digests.bin").read_bytes()
    assert ev["setup"]["commitment"] == run["published"]["commitment"] + run["published"]["salt"]
    q = SC.period.point_of(run["point"]).params
    assert len(bytes.fromhex(ev["setup"]["evidence"])) == 4 + 4 * q.segments + q.blocks * (4 + q.block_bytes)
    assert SC.pous_answer(case, schemes=ts.SCHEMES, protocol=SPT.TEST_PROTOCOL) == got
    assert _mode(private) == 0o700 and _mode(private / "case.json") == _mode(private / "answer.json") == 0o600
    for secret in (run["published"]["salt"], run["published"]["root"], got["commitment"], case["evidence"][:64]):
        assert secret not in "".join(p.decode() for p in public.values())


def test_after_the_run_the_builder_writes_the_piece_the_run_wrote(served, tmp_path, monkeypatch):
    """The served piece writes its case through this builder; built again after the run, from the kept outputs and
    ``C*`` encoded again, it is the same bytes."""
    SPT._run(served, tmp_path, monkeypatch)
    kept = {f: (served.private / f).read_bytes() for f in ("case.json", "answer.json")}
    _pous(served)
    assert {f: (served.private / f).read_bytes() for f in kept} == kept


def test_a_tampered_pous_case_is_answered_otherwise(served, tmp_path, monkeypatch):
    """The component answers the case from its own bytes: another commitment is another digest, a session without one of
    its coin openings is rejected, and a case not in canonical form is refused before it is read."""
    SPT._run(served, tmp_path, monkeypatch)
    _, got = _pous(served)
    case = json.loads((served.private / "case.json").read_text())
    published = _hexjson(case["commitment"]) | {"salt": "00" * 32}
    other = SC.pous_answer(case | {"commitment": _rehex(published)}, schemes=ts.SCHEMES, protocol=SPT.TEST_PROTOCOL)
    assert other["commitment"] != got["commitment"] and other["verdict"] == "reject"
    ev = _hexjson(case["evidence"])
    ev["session"]["openings"].pop()
    lost = SC.pous_answer(case | {"evidence": _rehex(ev)}, schemes=ts.SCHEMES, protocol=SPT.TEST_PROTOCOL)
    assert lost["verdict"] == "reject" and lost["commitment"] == got["commitment"]
    loose = json.dumps(_hexjson(case["commitment"]), indent=1).encode().hex()
    with pytest.raises(SC.Refused, match="commitment is not canonical JSON"):
        SC.pous_answer(case | {"commitment": loose}, schemes=ts.SCHEMES)
    with pytest.raises(SC.Refused, match="a case holds exactly"):
        SC.pous_answer(case | {"blobs": []}, schemes=ts.SCHEMES)


@pytest.mark.parametrize("tamper, reason", [
    ("public commitment", "doesn't open the public answer's commitment"),
    ("published", "the opened row is not the digest of setup's published outputs"),
    ("session", "the run protocol answers the case reject"),
    ("table", "is not the memory challenger's table"),
    ("verdict", "is not the run's own answer under a hiding commitment"),
])
def test_a_pous_window_changed_after_the_run_is_refused(served, tmp_path, monkeypatch, tamper, reason):
    """Each kept output is held to the others: the public commitment to the opening, the opened row to setup's published
    outputs, the session and the table to the run's answer and to ``C*`` encoded again, the public verdict to the run's.
    A refusal writes nothing."""
    SPT._run(served, tmp_path, monkeypatch)
    p = served.private
    kept = {f: (p / f).read_bytes() for f in ("case.json", "answer.json")}
    if tamper == "public commitment":
        _edit(served.out / "answer.json", lambda d: d.update(commitment="11" * 64))
    elif tamper == "published":
        _edit(p / "run.json", lambda d: d["published"].update(commitment="22" * 64))
    elif tamper == "session":
        _edit(p / "run.json", lambda d: d["session"]["openings"].pop())
    elif tamper == "table":
        t = bytearray((p / "digests.bin").read_bytes())
        t[5] ^= 1
        (p / "digests.bin").write_bytes(bytes(t))
    else:
        _edit(served.out / "answer.json", lambda d: d.update(verdict="reject"))
    with pytest.raises(SC.Refused, match=reason):
        _pous(served)
    assert {f: (p / f).read_bytes() for f in kept} == kept


def test_weights_that_are_not_the_pins_make_no_pous_case(served, tmp_path, monkeypatch):
    SPT._run(served, tmp_path, monkeypatch)
    _edit(served.pins, lambda d: d["checkpoints"][0]["files"]["model.safetensors"].update(sha256="33" * 32))
    with pytest.raises(SC.Refused, match="not the pinned checkpoint's"):
        _pous(served)


def test_a_pous_window_whose_session_did_not_complete_has_no_case(served, tmp_path):
    SPT.SP.run(served.serve, 3, served.out, hf_home=served.hf, device="cpu", node=SPT.NODE, private=served.private,
               party=SPT.CP.Party(SPT.AUDITOR, (sys.executable, "-c", "import sys; sys.exit(3)"), ("127.0.0.1", 9)),
               pins=served.pins, schemes=ts.SCHEMES, protocol=SPT.TEST_PROTOCOL)
    with pytest.raises(SC.Refused, match="names no private directory: the run's session didn't complete"):
        SC.pous_piece(served.out, schemes=ts.SCHEMES, hf_home=served.hf, device="cpu", pins=served.pins)


def _wipe_window(tmp_path: Path, *, protocol: str = WIPE_ID, egress_bits: int = W.UNCOUNTED) -> tuple[Path, Path, W.Wipe]:
    """A served wipe's public directory as `served_wipe.sh` leaves it, and the serve dir its ``served.json`` names."""
    w = W.run("n0", SMALL_WIPE, signer=ts.TestSigner(b"served case wipe", 2, directory=tmp_path),
              log=ts.session_log(tmp_path), schemes=ts.SCHEMES,
              protocol=protocol, clock=iter(range(10 ** 9, 10 ** 12)).__next__, coin_secret=b"coins",
              egress_bits=egress_bits)
    assert w.verdict == "accept", w.answer
    serve, window = tmp_path / "serve", tmp_path / "pieces" / "wipe"
    window.mkdir(parents=True)
    serve.mkdir()
    answer = w.answer | {"profile": w.answer["profile"] | {"placement": {"challenger": "its own host"}}}
    (window / "answer.json").write_text(json.dumps(answer, sort_keys=True) + "\n")
    (window / "wipe.case.json").write_text(json.dumps(w.case, sort_keys=True))
    (window / "served.json").write_text(json.dumps({"serve_dir": str(serve), "window": "all", "server": "n0"}))
    return window, serve, w


def test_the_wipe_case_is_the_legs_and_its_answer_the_legs_own(tmp_path):
    """The case is the one the leg wrote, its answer the public one without the run's placement; both go to the serve
    dir's private wipe directory, and the public directory gains nothing."""
    window, serve, w = _wipe_window(tmp_path)
    public = _listing(window)
    private, got = SC.wipe_piece(window, schemes=ts.SCHEMES)
    assert private == serve / "private" / "wipe" and _listing(window) == public
    assert json.loads((private / "case.json").read_text()) == w.case
    assert got == w.answer == json.loads((private / "answer.json").read_text())
    assert got["commitment"] == SC.digest(w.case) == W.digest(SC.wipe_party.commitment_of(_hexjson(w.case["commitment"]))).hex()
    assert _mode(private) == 0o700 and _mode(private / "case.json") == 0o600


def test_the_wipe_case_kept_privately_is_the_one_read(tmp_path):
    window, serve, w = _wipe_window(tmp_path)
    private = serve / "private" / "wipe"
    private.mkdir(parents=True)
    (private / "wipe.case.json").write_text(json.dumps(w.case))
    (window / "wipe.case.json").unlink()
    assert SC.wipe_piece(window, schemes=ts.SCHEMES)[1] == w.answer


def test_a_tampered_wipe_case_is_refused(tmp_path):
    """Another commitment is another digest than the answer's; a signed session changed after the challenger signed it
    is rejected, not as the run answered; a case that is not what its own parts make is refused."""
    window, serve, w = _wipe_window(tmp_path)
    c = _hexjson(w.case["commitment"])
    for name, case, reason in [
        ("commitment", w.case | {"commitment": _rehex(c | {"session": "44" * 32})},
         "the answer's commitment is not the digest"),
        ("evidence", w.case | {"evidence": _rehex({"signed": _hexjson(w.case["evidence"])["signed"] | {"egress_bits": 7}})},
         "the wipe protocol answers the case reject"),
        ("params", w.case | {"params": w.case["params"] | {"extra": 1}}, "is not the one its decoded commitment"),
    ]:
        (window / "wipe.case.json").write_text(json.dumps(case))
        with pytest.raises(SC.Refused, match=reason):
            SC.wipe_piece(window, schemes=ts.SCHEMES)
        assert not (serve / "private" / "wipe" / "case.json").exists(), name


def test_a_wipe_of_another_server_is_refused(tmp_path):
    window, _, _ = _wipe_window(tmp_path)
    _edit(window / "served.json", lambda d: d.update(server="n1"))
    with pytest.raises(SC.Refused, match="a wipe of another server than n1"):
        SC.wipe_piece(window, schemes=ts.SCHEMES)


def _fake_lean(tmp_path: Path, answer: dict) -> Path:
    exe = tmp_path / "lean"
    (tmp_path / "lean.json").write_text(json.dumps(answer))
    exe.write_text(f"#!/bin/sh\ncat > /dev/null\ncat {tmp_path / 'lean.json'}\n")
    exe.chmod(0o755)
    return exe


def test_with_the_lean_its_answer_is_written_when_it_agrees(tmp_path):
    window, serve, w = _wipe_window(tmp_path)
    lean = w.answer | {"reason": None if w.answer["reason"] else "-"}
    _, got = SC.wipe_piece(window, schemes=ts.SCHEMES, exe=_fake_lean(tmp_path, lean))
    assert got == lean == json.loads((serve / "private" / "wipe" / "answer.json").read_text())
    other = w.answer | {"verdict": "reject", "public": None, "profile": None, "reason": "no"}
    with pytest.raises(SC.Refused, match="answers the case reject"):
        SC.wipe_piece(window, schemes=ts.SCHEMES, exe=_fake_lean(tmp_path, other))


def test_a_case_is_written_as_canonical_json_whatever_its_size(tmp_path, monkeypatch):
    """A string past `CHUNK` goes out in slices, the file the same bytes as one canonical dump."""
    window, serve, w = _wipe_window(tmp_path)
    monkeypatch.setattr(SC, "CHUNK", 7)
    private, got = SC.wipe_piece(window, schemes=ts.SCHEMES)
    for name, doc in (("case.json", w.case), ("answer.json", got)):
        assert (private / name).read_text() == json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n"
    SC.write(tmp_path / "empty", {}, {"a": "0123456789abcdef", "b": [1, {"d": 2, "c": None}]})
    assert (tmp_path / "empty" / "case.json").read_text() == "{}\n"
    assert (tmp_path / "empty" / "answer.json").read_text() == '{"a":"0123456789abcdef","b":[1,{"c":null,"d":2}]}\n'


def test_the_cli_refuses_a_test_signed_session_and_what_is_no_piece(tmp_path, capsys):
    """Under the deployed schemes a test signer's session is no accept, so the run's answer stands refused (exit 1);
    an answer of another protocol is no piece (exit 2)."""
    window, serve, _ = _wipe_window(tmp_path, protocol=W.PROTOCOL)
    assert SC.main(["--window", str(window)]) == 1
    assert "CASE: refused: the wipe protocol answers the case reject" in capsys.readouterr().out
    assert not (serve / "private" / "wipe" / "case.json").exists()
    _edit(window / "answer.json", lambda d: d.update(protocol="verity/other/v0"))
    assert SC.main(["--window", str(window)]) == 2
    assert "no PoUS or wipe answer" in capsys.readouterr().err
