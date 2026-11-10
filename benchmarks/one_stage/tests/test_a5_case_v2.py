"""a5's case for the audit on the proofs themselves, `Spot.OneStageZk`'s: v1's case with the evidence's verdicts
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
from verity.core.protocols.spot.one_stage import proofs as PF  # noqa: E402

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


def test_v2_names_the_public_file_as_it_was_staged_before_the_draw(tmp_path):
    """The case's public digest is the one a5 took when it staged the member, before the sample, never the file's digest
    read after it: a public file changed since names another file, which v2 finds in no blob."""
    (tmp_path / "pub-4.bin").write_bytes(b"public, changed after the draw")
    s = tmp_path / "sessions" / "a"
    s.mkdir(parents=True)
    (s / "session.json").write_bytes(b"{}")
    mm = {"st": {"dir": tmp_path, "public": "ee" * 64}, "m": types.SimpleNamespace(n=4), "pin": "cc" * 64}
    got = a5.v2_proof([{"mm": mm, "sdir": tmp_path}], zk=True)
    assert got["public"] == "ee" * 64 and got["proof"].public == tmp_path / "pub-4.bin" and got["proof"].sessions == (s,)


def test_v2_hands_v2_the_tables_its_circuit_looks_up_alone(tmp_path):
    """The stage holds every MUFU table a circuit with lookup slots may read, and v2 holds the blobs' pinned tables to the
    circuit's META lookups (`OneStageZk.tablesHeld`): the case names those tables and no other."""
    (tmp_path / "circuit.txt").write_text('flock-hm 1\nMETA {"lookups": {"ex2": {}, "rcp": {}}}\ng 1 2\n')
    (tmp_path / "tables").mkdir()
    for n in ("rsq", "sqrt", "rcp", "ex2"):
        (tmp_path / "tables" / f"{n}.u32").write_bytes(n.encode())
    mm = {"st": {"dir": tmp_path, "public": "ee" * 64}, "m": types.SimpleNamespace(n=4), "pin": "cc" * 64}
    got = a5.v2_proof([{"mm": mm, "sdir": tmp_path}], zk=True)
    assert got["tables"] == [tmp_path / "tables" / "ex2.u32", tmp_path / "tables" / "rcp.u32"]


def test_v2_says_why_an_audit_has_no_case():
    assert "plain" in a5.v2_proof([{}], zk=False)
    assert "drew 2 members" in a5.v2_proof([{}, {}], zk=True)


@pytest.fixture
def stub_case(monkeypatch, RP):
    class C:
        def __init__(self, *a):
            pass

        def case(self):
            return {"commitment": "ab", "evidence": "7b7d", "params": {}}

    monkeypatch.setattr(RP, "params", lambda *a, **k: {})
    monkeypatch.setattr(RP, "evidence", lambda *a, **k: {})
    monkeypatch.setattr(RP, "Case", C)
    monkeypatch.setattr(RP, "python", lambda c: {"verdict": "accept"})
    return RP


def write(out: Path, v2) -> None:
    a5.write_case(out, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], v2=v2)


def test_a_v2_refusal_is_written_beside_the_v1_case_and_leaves_spot_no_case(tmp_path, stub_case):
    write(tmp_path, "v2 verifies --zk sessions; this audit's are plain")
    assert json.loads((tmp_path / "case.json").read_text())["commitment"] == "ab"
    assert not (tmp_path / "case-v2.json").exists()
    assert not (tmp_path / "case-sampled.json").exists()
    assert (tmp_path / "case-v2-error.txt").read_text().startswith("ValueError: v2 verifies --zk")


def test_spot_case_is_the_v2_case_in_its_one_stage_form(tmp_path, monkeypatch, stub_case):
    """Sampled proofs refuses one-stage v1's case, so its case is v2's with the form, never v1's."""
    zk = {"blobs": ["b"], "commitment": "ab", "evidence": stub_case.DT.canonical({"proofs": {}}).encode().hex(),
          "params": {"proofs": {"circuit": "cc" * 64, "registered": None}}}
    monkeypatch.setattr(a5, "case_v2", lambda case, sessions, **k: zk)
    write(tmp_path, {})
    assert json.loads((tmp_path / "case-v2.json").read_text()) == zk
    got = json.loads((tmp_path / "case-sampled.json").read_text())
    assert got == a5.case_with_form(zk)
    assert got["params"]["form"] == "one-stage" and got["params"]["proofs"] == zk["params"]["proofs"]
    assert json.loads(bytes.fromhex(got["evidence"])) == {"form": "one-stage", "proofs": {}}
    assert not (tmp_path / "case-v2-error.txt").exists()


def test_the_case_carries_the_verifiers_partition_which_v2_verifies_the_sessions_against(tmp_path, monkeypatch, RP):
    monkeypatch.setattr(RP, "descriptor", lambda program: {"program": "p"})
    monkeypatch.setattr(RP, "evidence", lambda *a, **k: {})
    monkeypatch.setattr(RP, "python", lambda c: {"verdict": "accept"})
    part = {"program": "p", "rule": "r"}
    a5.write_case(tmp_path, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], partition=part)
    assert json.loads((tmp_path / "case.json").read_text())["params"]["partition"] == part
