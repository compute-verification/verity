"""a5's case for the audit on the proofs themselves, `SampledProofs.OneStageZk`'s: v1's case with the evidence's verdicts
replaced by the proofs' files named by SHA-512, the params naming the auditor's circuit and public files' digests, and
blobs the paths of those files and the pinned tables; and the audits v2 can't verify say why."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5  # noqa: E402
from verity.core.protocols.sampled_proofs.one_stage import proofs as PF  # noqa: E402

_spec = importlib.util.spec_from_file_location("one_stage_replay", HERE.parent / "replay.py")
REPLAY = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = REPLAY
_spec.loader.exec_module(REPLAY)


@pytest.fixture
def RP(monkeypatch):
    """This suite's `replay`, which a5 imports by that name, under that name for one test only: `network_traces` has a
    `replay` of its own, which `active_replay` (imported by this suite's warden tests) reads by the same name."""
    monkeypatch.setitem(sys.modules, "replay", REPLAY)
    return REPLAY


def sha(p: Path) -> str:
    return hashlib.sha512(p.read_bytes()).hexdigest()


def test_case_v2_names_the_proofs_files_in_place_of_verdicts(tmp_path, RP):
    (tmp_path / "circuit.txt").write_bytes(b"circuit")
    (tmp_path / "pub-4.bin").write_bytes(b"public")
    s = tmp_path / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_bytes(b"{}")
    (s / "rep-0.bin").write_bytes(b"rep")
    table = tmp_path / "tables" / "t.bin"
    table.parent.mkdir()
    table.write_bytes(b"table")
    evidence = {"draw": {"units": [1]}, "sessions": [sha(s / "session.json")], "verdicts": {"m0": {"accepted": True}}}
    case = {"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(), "params": {"delta": "1/100"}}
    got = a5.case_v2(case, evidence["sessions"], proof=PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub-4.bin", (s,)),
                     circuit="cc" * 64, public=sha(tmp_path / "pub-4.bin"), registered=None, tables=[table])
    assert got["commitment"] == "ab"
    assert got["params"] == {"delta": "1/100", "proofs": {"circuit": "cc" * 64, "public": sha(tmp_path / "pub-4.bin"),
                                                          "registered": None}}
    e = json.loads(bytes.fromhex(got["evidence"]))
    assert "verdicts" not in e and e["draw"] == evidence["draw"] and e["sessions"] == evidence["sessions"]
    assert e["proofs"] == {"sessions": [{"name": evidence["sessions"][0], "record": sha(s / "session.json"),
                                         "reps": {"rep-0.bin": sha(s / "rep-0.bin")}}]}
    assert got["evidence"] == RP.DT.canonical(e).encode().hex()
    assert got["blobs"] == [str(p) for p in (tmp_path / "circuit.txt", tmp_path / "pub-4.bin", s / "session.json",
                                             s / "rep-0.bin", table)]


EV = {"key": {"W": 4, "H": 5, "public": "aa" * 32}, "ledger": ["{}"], "main": ["cc" * 20], "pins": ["{}"],
      "segment": {"label": "draw", "start": 1, "after": "dd" * 64, "openings": [{"coin": "ee" * 64}]},
      "session": {"close": {}, "hellos": [], "log": [], "record": {}}}


def test_the_draws_release_names_the_challenger_and_carries_the_coins():
    """`drawn_by`: the params name the challenger's key under the draw's family and the window root, and the evidence
    carries the coins in `Challenger.drawCoins`'s form, the evidence less its key; no evidence, no release."""
    got = a5.drawn_by(EV, "ab" * 64)
    assert got == {"params": {"challenger": {"key": EV["key"], "label": a5.AC.FAMILY}, "window_root": "ab" * 64},
                   "evidence": {"coins": {k: v for k, v in EV.items() if k != "key"}}}
    assert a5.drawn_by(None, "ab" * 64) is None


def test_case_v2_with_the_release_names_the_challenger_v2_requires(tmp_path, RP):
    (tmp_path / "circuit.txt").write_bytes(b"circuit")
    (tmp_path / "pub-4.bin").write_bytes(b"public")
    s = tmp_path / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_bytes(b"{}")
    evidence = {"draw": {"units": [1]}, "sessions": [sha(s / "session.json")], "verdicts": {}}
    case = {"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(), "params": {"delta": "1/100"}}
    proof = PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub-4.bin", (s,))
    drawn = a5.drawn_by(EV, "ab" * 64)
    got = a5.case_v2(case, evidence["sessions"], proof=proof, circuit="cc" * 64, public="ee" * 64, registered=None,
                     tables=[], drawn=drawn)
    assert got["params"] == {"delta": "1/100", "challenger": drawn["params"]["challenger"], "window_root": "ab" * 64,
                             "proofs": {"circuit": "cc" * 64, "public": "ee" * 64, "registered": None}}
    assert json.loads(bytes.fromhex(got["evidence"]))["coins"] == drawn["evidence"]["coins"]
    none = a5.case_v2(case, evidence["sessions"], proof=proof, circuit="cc" * 64, public="ee" * 64, registered=None,
                      tables=[])
    assert "challenger" not in none["params"] and "coins" not in json.loads(bytes.fromhex(none["evidence"]))


def test_v2_names_the_public_file_as_the_audit_pinned_it(tmp_path):
    """The case's public digest is the member's pin (``st["public"]``: the staged file's, then the completed file's once
    the tie holds, `pin_completed`), never the file's digest read when the case is written: a public file changed since
    names another file, which v2 finds in no blob."""
    (tmp_path / "pub-4.bin").write_bytes(b"public, changed after the draw")
    s = tmp_path / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_bytes(b"{}")
    mm = {"st": {"dir": tmp_path, "public": "ee" * 64}, "m": types.SimpleNamespace(n=4), "pin": "cc" * 64}
    drawn = a5.drawn_by(EV, "ab" * 64)
    got = a5.v2_proof([{"mm": mm, "sdir": tmp_path}], zk=True, drawn=drawn)
    assert got["public"] == "ee" * 64 and got["proof"].public == tmp_path / "pub-4.bin" and got["proof"].sessions == (s,)
    assert got["drawn"] == drawn


@pytest.mark.parametrize("completed_body", [b"body 1", b"body 1, re-laid port-major"])
def test_a_drawn_members_completed_public_file_is_pinned_once_the_tie_holds_and_is_the_one_v2_names(tmp_path,
                                                                                                    completed_body):
    """`complete_drawn` rewrites a drawn member's public file after the draw under the completed header, an
    instance-major member's body re-laid too, and the member's session proves that file. Once the tie holds,
    `pin_completed` pins it as the member's ``completed`` entry, and v2's case names it: by SHA-512 the blob the case
    carries, not the staged file, which no blob holds (OneStageZk's ``public file: no blob has the SHA-512``). The
    staged entries and their leaf layer, which R4 holds to the registration, are as written before the draw."""
    staged = {"m0-key": b'{"header": "staged"}\nbody 0', "m1-key": b'{"header": "staged"}\nbody 1'}
    members = []
    for k, (key, data) in enumerate(staged.items()):
        (tmp_path / f"m{k}").mkdir()
        (tmp_path / f"m{k}" / "pub-4.bin").write_bytes(data)
        members.append({"k": k, "m": types.SimpleNamespace(key=key, n=4), "pin": "cc" * 64,
                        "st": {"dir": tmp_path / f"m{k}", "public": hashlib.sha512(data).hexdigest()}})
    pin = a5.public_pin(members, {key: data.partition(b"\n")[2] for key, data in staged.items()})
    (tmp_path / a5.PIN).write_text(json.dumps(pin))
    mm, completed = members[1], b'{"header": "completed"}\n' + completed_body
    (mm["st"]["dir"] / "pub-4.bin").write_bytes(completed)
    a5.pin_completed(tmp_path, [(mm, [0])])
    after = json.loads((tmp_path / a5.PIN).read_text())
    assert after["leaf_layer"] == pin["leaf_layer"]
    assert {k: {x: m[x] for x in ("k", "file", "body")} for k, m in after["members"].items()} == pin["members"]
    assert "completed" not in after["members"]["m0-key"]
    assert after["members"]["m1-key"]["completed"] == {"file": hashlib.sha512(completed).hexdigest(),
                                                      "body": hashlib.sha512(completed_body).hexdigest()}
    s = tmp_path / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_bytes(b"{}")
    got = a5.v2_proof([{"mm": mm, "sdir": tmp_path}], zk=True, drawn=a5.drawn_by(EV, "ab" * 64))
    assert got["public"] == PF.file_sha512(got["proof"].public) == after["members"]["m1-key"]["completed"]["file"]
    assert got["public"] != pin["members"]["m1-key"]["file"]


def test_v2_says_why_an_audit_has_no_case():
    assert "plain" in a5.v2_proof([{}], zk=False)
    assert "drew 2 members" in a5.v2_proof([{}, {}], zk=True)
    assert "NO_CHALLENGER" in a5.v2_proof([{}], zk=True)


def test_a_v2_refusal_is_written_beside_the_v1_case(tmp_path, monkeypatch, RP):
    class C:
        def __init__(self, *a):
            pass

        def case(self):
            return {"commitment": "ab", "evidence": "7b7d", "params": {}}

    monkeypatch.setattr(RP, "params", lambda *a, **k: {})
    monkeypatch.setattr(RP, "evidence", lambda *a, **k: {})
    monkeypatch.setattr(RP, "Case", C)
    monkeypatch.setattr(RP, "python", lambda c: {"verdict": "accept"})
    a5.write_case(tmp_path, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], v2="v2 verifies --zk sessions; this audit's are plain")
    assert json.loads((tmp_path / "case.json").read_text())["commitment"] == "ab"
    assert not (tmp_path / "case-v2.json").exists()
    assert (tmp_path / "case-v2-error.txt").read_text().startswith("ValueError: v2 verifies --zk")


def test_a_tied_case_names_its_values_parameters_and_row_formats_as_the_lean_reads_them(tmp_path, monkeypatch, RP):
    """`OneStage.paramsOfJson` takes exactly its keys: a case tied to W (a5's ``tie``) names the program function's
    ``parameters`` beside its registered values and their row formats as ``row_schemas``."""
    class T:
        def to_json(self):
            return {"bits": 32}

    program = type("Program", (), {"fn": type("Fn", (), {"signature": staticmethod(lambda: ([("v", T())], None))})})()
    monkeypatch.setattr(RP, "descriptor", lambda program: {"program": "p"})
    monkeypatch.setattr(RP, "evidence", lambda *a, **k: {})
    monkeypatch.setattr(RP, "python", lambda c: {"verdict": "accept"})
    tied = {"log": [{"previous": None}], "registered": {"v": 4}, "outputs": {"v": {"record": "dd" * 64, "value": "v"}},
            "schemas": {"v": "row-seg/v1"}}
    a5.write_case(tmp_path, program, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], tied=tied)
    p = json.loads((tmp_path / "case.json").read_text())["params"]
    assert sorted(p) == ["anchors", "delta", "law", "leaf_layer", "log", "outputs", "parameters", "partition", "ports",
                         "program", "query", "registered", "row_schemas", "scheme", "work_table"]
    assert (p["parameters"], p["row_schemas"]) == ([["v", {"bits": 32}]], {"v": "row-seg/v1"})


def test_the_case_carries_the_verifiers_partition_which_v2_verifies_the_sessions_against(tmp_path, monkeypatch, RP):
    monkeypatch.setattr(RP, "descriptor", lambda program: {"program": "p"})
    monkeypatch.setattr(RP, "evidence", lambda *a, **k: {})
    monkeypatch.setattr(RP, "python", lambda c: {"verdict": "accept"})
    part = {"program": "p", "rule": "r"}
    a5.write_case(tmp_path, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], partition=part)
    assert json.loads((tmp_path / "case.json").read_text())["params"]["partition"] == part
