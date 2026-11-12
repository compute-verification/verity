"""a5's case for the audit on the proofs themselves, `Spot.OneStageZk`'s: v1's case with the evidence's verdicts
replaced by the proofs' files named by SHA-512, the params naming the auditor's circuit and public files' digests, and
blobs the paths of those files and the pinned tables; and the audits v2 can't verify say why. A window of several
members has its case in v2's per-member form beside those, in a file of its own."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import a5  # noqa: E402
from verity.protocols.spot.verifier.one_stage import proofs as PF  # noqa: E402


@pytest.fixture
def RP():
    """The `replay` a5 reads: the file beside it, loaded by its path (`a5.own_replay`)."""
    return a5.own_replay()


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


def test_two_sampled_members_statements_make_no_v2_case_and_each_alone_names_its_own_files(tmp_path, RP):
    """v2 verifies one table's one session, so an audit whose sample reached two members has no v2 case and says so;
    either member's statement alone is a case of its own files, a circuit two members share (a template's, hard-linked
    into each sampled member's directory) named at that member's path."""
    m0, m1 = tmp_path / "m0", tmp_path / "m1"
    for d in (m0, m1):
        d.mkdir()
        (d / "pub-2.bin").write_bytes(d.name.encode())
    (m0 / "circuit.txt").write_bytes(b"circuit")
    os.link(m0 / "circuit.txt", m1 / "circuit.txt")
    s0, s1 = (tmp_path / f"session-m{k}" for k in (0, 1))
    for k, s in enumerate((s0, s1)):
        (s / "sessions" / "a").mkdir(parents=True)
        (s / "sessions" / "a" / "session.json").write_text(json.dumps({"k": k}))
    names = [sha(s / "sessions" / "a" / "session.json") for s in (s0, s1)]
    done = [{"mm": {"k": k, "m": SimpleNamespace(base=2 * k, n=2), "pin": "cc" * 64, "st": {"dir": d, "public": sha(d / "pub-2.bin")}},
             "sdir": s} for k, (d, s) in enumerate(((m0, s0), (m1, s1)))]
    assert a5.v2_proof(done, zk=True) == ("v2 verifies one table's session on its circuit and public file; this audit "
                                          "drew 2 members")
    evidence = {"draw": {"units": [1, 2]}, "sessions": names[1:], "verdicts": {"m1": {"accepted": True}}}
    case = {"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(), "params": {"delta": "1/100"}}
    got = a5.case_v2(case, names[1:], **a5.v2_proof(done[1:], zk=True))
    assert got["params"] == {"delta": "1/100", "proofs": {"circuit": "cc" * 64, "public": sha(m1 / "pub-2.bin"),
                                                          "registered": None}}
    e = json.loads(bytes.fromhex(got["evidence"]))
    assert "verdicts" not in e and e["sessions"] == names[1:]
    assert [x["name"] for x in e["proofs"]["sessions"]] == names[1:]
    assert got["blobs"] == [str(p) for p in (m1 / "circuit.txt", m1 / "pub-2.bin", s1 / "sessions" / "a" / "session.json")]
    with pytest.raises(ValueError, match="2 names for 1 sessions"):
        a5.case_v2(case, names, **a5.v2_proof(done[1:], zk=True))


def test_v2_case_is_the_sampled_members_one_statement_whatever_the_windows_other_members(tmp_path):
    """`v2_proof` names the one sampled member's statement, its circuit pin, the public file's digest it was staged
    with and its registered file, read; the window's other members and its registered circuit record add nothing to it:
    v2 holds that public file's units to the registration's population, so a window of several members is refused
    there, never given a form of its own."""
    def member(k: int, base: int, n: int, reads: dict | None = None) -> dict:
        d = tmp_path / f"m{k}"
        d.mkdir()
        st = {"dir": d, "meta": {"vus_per_block": k + 1}, "public": str(k) * 2 * 64}
        if reads is not None:
            st["registered_file"] = d / "registered.json"
            st["registered_file"].write_text(json.dumps(reads))
        return {"k": k, "m": SimpleNamespace(base=base, n=n), "pin": str(k) * 128, "st": st}

    ms = [member(0, 0, 2), member(1, 2, 3, {"ports": {}})]
    sdir = tmp_path / "session-m1"
    (sdir / "sessions" / "a").mkdir(parents=True)
    (sdir / "sessions" / "a" / "session.json").write_text("{}")
    done = [{"mm": ms[1], "sdir": sdir}]
    got = a5.v2_proof(done, True)
    assert {k: v for k, v in got.items() if k != "proof"} == {
        "circuit": "1" * 128, "public": "1" * 128, "registered": {"ports": {}}, "tables": []}
    assert (got["proof"].circuit, got["proof"].public, got["proof"].sessions) == (
        tmp_path / "m1" / "circuit.txt", tmp_path / "m1" / "pub-3.bin", (sdir / "sessions" / "a",))
    iface = tmp_path / "circuit.json"
    iface.write_bytes(b'{"format": "verity/window-circuit/v1"}')
    for mm in ms:
        mm["st"]["interface_file"] = iface
    held = a5.v2_proof(done, True)
    assert {k: v for k, v in held.items() if k != "proof"} == {k: v for k, v in got.items() if k != "proof"}
    assert held["proof"] == got["proof"]


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
    assert "drew 0 members" in a5.v2_proof([], zk=True)


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


def test_a5_records_the_members_whose_link_points_the_registration_refuses():
    """`a5.link_points` over the staged members, each its units at its circuit's units a proof block: under `subset:20`
    over 40 units a draw that ends holds 8 to 20 of a member of 28, whose link points differ at one unit a proof block,
    so it is recorded refused, and not at four; one of 8, or of 4 at two a proof block, is not. Under `subset:2` a draw
    that ends holds at most 2 of it, so it is not refused (one that runs out of coins fails the window:
    `a5.draw_ran_out`). A law that doesn't apply refuses every member, recorded with why."""
    def member(k: int, template: str, base: int, n: int, g: int) -> dict:
        return {"k": k, "m": SimpleNamespace(template=template, base=base, n=n), "st": {"meta": {"vus_per_block": g}}}

    ms = [member(0, "t", 0, 28, 1), member(1, "u", 28, 8, 1), member(2, "v", 36, 4, 2)]
    assert a5.link_points({"k": 20, "law": "subset"}, 40, ms) == {"members": 3, "refused": [
        {"member": "m0", "template": "t", "n": 28, "vus_per_block": 1, "fewest": 8, "most": 20}]}
    assert a5.link_points({"k": 20, "law": "subset"}, 40, [member(0, "t", 0, 28, 4)]) == {"members": 1, "refused": []}
    assert a5.link_points({"k": 2, "law": "subset"}, 40, ms) == {"members": 3, "refused": []}
    got = a5.link_points({"k": 41, "law": "subset"}, 40, ms)
    assert got == {"members": 3, "error": "a subset draw of 41 exceeds the population of 40"}


def test_a5_records_a_law_none_of_whose_draws_ends(monkeypatch):
    """`a5.link_points` records the registration's refusal of a law no draw of which ends (`link_points.NEVER_ENDS`),
    which `coins.law_problem` refuses first for every law it reads: past it, `subset:41` over 40 units is refused for
    every member."""
    from verity.protocols.spot.verifier.one_stage import link_points as LP

    monkeypatch.setattr(a5.CC, "law_problem", lambda law, population: None)
    m = {"k": 0, "m": SimpleNamespace(template="t", base=0, n=40), "st": {"meta": {"vus_per_block": 1}}}
    assert a5.link_points({"k": 41, "law": "subset"}, 40, [m]) == {"members": 1, "error": LP.NEVER_ENDS}


def test_a5_records_whether_the_draw_ran_out_of_the_challengers_coins():
    """`a5.draw_ran_out`, which fails the window: `subset:9` over 10 units first reads a stream
    of 226 bytes (`2 k · width(n) + 64`), so it runs out of one coin's 64 bytes and not of 256 zero bytes, which reject
    no word; a law that doesn't apply records why."""
    law = {"k": 9, "law": "subset"}
    assert a5.draw_ran_out(law, 10, bytes(64)) is True
    assert a5.draw_ran_out(law, 10, bytes(256)) is False
    assert a5.draw_ran_out({"k": 11, "law": "subset"}, 10, bytes(256)) == "a subset draw of 11 exceeds the population of 10"


def staged(tmp_path: Path, k: int, key: str, base: int, n: int, *, lookups: str = "{}", reads: dict | None = None) -> dict:
    """A member as a5 stages it before the sample: its directory with its circuit and public files, its pin, its public
    file's digest and its circuit's META."""
    d = tmp_path / f"m{k}"
    (d / "tables").mkdir(parents=True)
    (d / "circuit.txt").write_text(f'flock-hm 1\nMETA {{"lookups": {lookups}}}\ng 1 2\n')
    (d / f"pub-{n}.bin").write_bytes(f"public {k}".encode())
    for t in ("rcp", "ex2"):
        (d / "tables" / f"{t}.u32").write_bytes(t.encode())
    st = {"dir": d, "public": sha(d / f"pub-{n}.bin"), "meta": {"vus_per_block": 4}}
    if reads is not None:
        st["registered_file"] = d / "registered.json"
        st["registered_file"].write_text(json.dumps(reads))
    return {"k": k, "m": types.SimpleNamespace(n=n, base=base, key=key), "st": st, "pin": f"{k:02x}" * 64}


def session(tmp_path: Path, mm: dict, *names: str) -> dict:
    """A sampled member's entry of a5's ``done``: its session directories, each a record and a rep file."""
    sdir = tmp_path / f"session-m{mm['k']}"
    for name in names:
        (sdir / "sessions" / name).mkdir(parents=True)
        (sdir / "sessions" / name / "session.json").write_text(json.dumps({"session": [mm["k"], name]}))
        (sdir / "sessions" / name / "rep-0.bin").write_bytes(f"rep {mm['k']} {name}".encode())
    return {"mm": mm, "sdir": sdir}


def test_the_per_member_proof_holds_every_member_and_the_sampled_ones_sessions(tmp_path):
    """The parameters name every member of the window, sampled or not, each by what a5 staged before the sample; a
    sampled member brings its session and the tables its circuit looks up, one path a table."""
    reads = {"ports": {"a": {"value": "W"}}}
    mms = [staged(tmp_path, 0, "t/c0", 0, 3, lookups='{"rcp": {}}'), staged(tmp_path, 1, "t/c1", 3, 2, reads=reads),
           staged(tmp_path, 2, "u/c0", 5, 4, lookups='{"ex2": {}, "rcp": {}}')]
    done = [session(tmp_path, mms[0], "a"), session(tmp_path, mms[2], "b")]
    got = a5.members_proof(mms, done, zk=True)
    assert [(m.name, m.units, m.vus_per_block, m.circuit_sha512, m.public_sha512, m.registered) for m in got["members"]] == [
        ("t/c0", ((0, 3),), 4, "00" * 64, mms[0]["st"]["public"], None),
        ("t/c1", ((3, 5),), 4, "01" * 64, mms[1]["st"]["public"], reads),
        ("u/c0", ((5, 9),), 4, "02" * 64, mms[2]["st"]["public"], None)]
    assert [m.proof.sessions for m in got["members"]] == [
        (tmp_path / "session-m0" / "sessions" / "a",), (), (tmp_path / "session-m2" / "sessions" / "b",)]
    assert got["members"][2].proof.circuit == tmp_path / "m2" / "circuit.txt"
    assert got["members"][2].proof.public == tmp_path / "m2" / "pub-4.bin"
    assert got["tables"] == [tmp_path / "m2" / "tables" / "ex2.u32", tmp_path / "m0" / "tables" / "rcp.u32"]


def test_the_per_member_proof_says_why_an_audit_has_none(tmp_path):
    mms = [staged(tmp_path, 0, "t/c0", 0, 3), staged(tmp_path, 1, "t/c1", 3, 2)]
    assert a5.members_proof(mms[:1], [], zk=True) is None
    assert "plain" in a5.members_proof(mms, [], zk=False)
    mms[1]["st"]["meta"] = {}
    assert a5.members_proof(mms, [], zk=True).startswith("member t/c1's circuit names no vus_per_block")


def members_case(tmp_path: Path, RP, order: tuple[int, int] = (0, 2)) -> tuple[dict, dict, list[dict], list[str]]:
    mms = [staged(tmp_path, 0, "t/c0", 0, 3), staged(tmp_path, 1, "t/c1", 3, 2), staged(tmp_path, 2, "u/c0", 5, 4)]
    done = [session(tmp_path, mms[0], "a"), session(tmp_path, mms[2], "b")]
    names = [sha(done[i > 0]["sdir"] / "sessions" / "ab"[i > 0] / "session.json") for i in order]
    evidence = {"proved_units": [1, 6], "sessions": names, "verdicts": {"m0": {"accepted": True}}}
    case = {"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(), "params": {"delta": "1/100"}}
    return case, a5.members_proof(mms, done, zk=True), mms, names


def test_case_members_names_every_member_and_a_statement_for_each_sampled_one(tmp_path, RP):
    case, proof, mms, names = members_case(tmp_path, RP)
    got = a5.case_members(case, **proof)
    assert got["commitment"] == "ab"
    assert got["params"] == {"delta": "1/100", "proofs": {"members": {
        "t/c0": {"circuit": "00" * 64, "public": mms[0]["st"]["public"], "registered": None, "units": [[0, 3]],
                 "vus_per_block": 4},
        "t/c1": {"circuit": "01" * 64, "public": mms[1]["st"]["public"], "registered": None, "units": [[3, 5]],
                 "vus_per_block": 4},
        "u/c0": {"circuit": "02" * 64, "public": mms[2]["st"]["public"], "registered": None, "units": [[5, 9]],
                 "vus_per_block": 4}}}}
    e = json.loads(bytes.fromhex(got["evidence"]))
    assert "verdicts" not in e and e["sessions"] == names and e["proved_units"] == [1, 6]
    s0, s2 = (tmp_path / f"session-m{k}" / "sessions" / x for k, x in ((0, "a"), (2, "b")))
    assert e["proofs"] == {"statements": [
        {"member": "t/c0", "sessions": [{"name": names[0], "record": names[0], "reps": {"rep-0.bin": sha(s0 / "rep-0.bin")}}]},
        {"member": "u/c0", "sessions": [{"name": names[1], "record": names[1], "reps": {"rep-0.bin": sha(s2 / "rep-0.bin")}}]}]}
    assert got["evidence"] == RP.DT.canonical(e).encode().hex()
    assert got["blobs"] == [str(f) for f in (
        tmp_path / "m0" / "circuit.txt", tmp_path / "m0" / "pub-3.bin", s0 / "session.json", s0 / "rep-0.bin",
        tmp_path / "m2" / "circuit.txt", tmp_path / "m2" / "pub-4.bin", s2 / "session.json", s2 / "rep-0.bin")]


def test_case_members_refuses_sessions_out_of_the_members_order(tmp_path, RP):
    """v2 reads the statements' sessions as v1's ``sessions`` in order, and the statements in the members' order: an
    audit whose sessions ran in another order has no per-member case."""
    case, proof, _, _ = members_case(tmp_path, RP, order=(2, 0))
    with pytest.raises(ValueError, match="are not the evidence's sessions in order"):
        a5.case_members(case, **proof)


def test_the_per_member_case_is_written_beside_the_others_and_changes_none(tmp_path, monkeypatch, stub_case):
    """A window of several members that sampled two still has no case in the one statement's form, v2's or the one
    onsite and remote read; its per-member case is a file of its own, or its error is."""
    monkeypatch.setattr(a5, "case_members", lambda case, **k: {"params": {"proofs": {"members": k["members"]}}})
    a5.write_case(tmp_path, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], v2=a5.v2_proof([{}, {}], zk=True), per_member={"members": {}, "tables": []})
    assert json.loads((tmp_path / "case-v2-members.json").read_text()) == {"params": {"proofs": {"members": {}}}}
    assert (tmp_path / "case-v2-error.txt").read_text().startswith("ValueError: v2 verifies one table's session")
    assert not (tmp_path / "case-v2.json").exists() and not (tmp_path / "case-sampled.json").exists()
    assert not (tmp_path / "case-v2-members-error.txt").exists()
    other = tmp_path / "plain"
    other.mkdir()
    a5.write_case(other, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], per_member="v2 verifies --zk sessions; this audit's are plain")
    assert (other / "case-v2-members-error.txt").read_text().startswith("ValueError: v2 verifies --zk")
    assert sorted(f.name for f in other.iterdir()) == ["case-python.json", "case-v2-members-error.txt", "case.json"]
