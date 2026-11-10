<<<<<<< 99c8914848
"""a5's case for the audit on the proofs themselves, `SampledProofs.OneStageZk`'s: v1's case with the evidence's verdicts
replaced by the proofs' files named by SHA-512, the params naming the auditor's circuit digest (per member, every
member's units, circuit digest and registered file), and blobs the paths of those files and the pinned tables; and the
audits v2 can't verify say why."""
||||||| merged common ancestors
"""a5's case for the audit on the proofs themselves, `SampledProofs.OneStageZk`'s: v1's case with the evidence's verdicts
replaced by the proofs' files named by SHA-512, the params naming the auditor's circuit digest, and blobs the paths of
those files and the pinned tables; and the audits v2 can't verify say why."""
=======
"""a5's case for the audit on the proofs themselves, `Spot.OneStageZk`'s: v1's case with the evidence's verdicts
replaced by the proofs' files named by SHA-512, the params naming the auditor's circuit and public files' digests, and
blobs the paths of those files and the pinned tables; and the audits v2 can't verify say why."""
>>>>>>> 7af4d24fca3e

from __future__ import annotations

import hashlib
import importlib.util
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
<<<<<<< 99c8914848
    got = a5.case_v2(case, evidence["sessions"], proofs={"circuit": "cc" * 64, "registered": None},
                     statements=[(None, PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub-4.bin", (s,)))], tables=[table])
||||||| merged common ancestors
    got = a5.case_v2(case, evidence["sessions"], proof=PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub-4.bin", (s,)),
                     circuit="cc" * 64, registered=None, tables=[table])
=======
    got = a5.case_v2(case, evidence["sessions"], proof=PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub-4.bin", (s,)),
                     circuit="cc" * 64, public=sha(tmp_path / "pub-4.bin"), registered=None, tables=[table])
>>>>>>> 7af4d24fca3e
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
    iface = tmp_path / "circuit.json"
    iface.write_bytes(b'{"format": "verity/window-circuit/v1"}')
    proofs = {"circuit": "cc" * 64, "registered": {"ports": {}}, "interface": sha(iface)}
    held = a5.case_v2(case, evidence["sessions"], proofs=proofs,
                      statements=[(None, PF.Proof(tmp_path / "circuit.txt", tmp_path / "pub-4.bin", (s,)))],
                      tables=[table], interfaces=[iface, iface])
    assert held["params"]["proofs"] == proofs
    assert held["blobs"] == [*got["blobs"], str(iface)]


def test_the_per_member_case_names_each_statements_member_and_sessions_and_each_file_once(tmp_path, RP):
    """In the per-member form each statement names its member and the next of the evidence's sessions, and a file two
    members' statements share (a template's circuit, hard-linked into each drawn member's directory) is one blob."""
    m0, m1 = tmp_path / "m0", tmp_path / "m1"
    for d in (m0, m1):
        d.mkdir()
        (d / "pub-2.bin").write_bytes(d.name.encode())
    (m0 / "circuit.txt").write_bytes(b"circuit")
    os.link(m0 / "circuit.txt", m1 / "circuit.txt")
    s0, s1 = (tmp_path / f"session-m{k}" / "sessions" / "a" for k in (0, 1))
    for k, s in enumerate((s0, s1)):
        s.mkdir(parents=True)
        (s / "session.json").write_text(json.dumps({"k": k}))
    names = [sha(s0 / "session.json"), sha(s1 / "session.json")]
    evidence = {"draw": {"units": [1, 2]}, "sessions": names, "verdicts": {"m0": {"accepted": True}}}
    case = {"commitment": "ab", "evidence": RP.DT.canonical(evidence).encode().hex(), "params": {"delta": "1/100"}}
    own = [{"circuit": "cc" * 64, "registered": None}, {"circuit": "cc" * 64, "registered": {"ports": {}}}]
    proofs = a5.v2_members([(0, 0, 2, 1, own[0]), (1, 2, 2, 1, own[1])])
    statements = [(f"m{k}", PF.Proof(d / "circuit.txt", d / "pub-2.bin", (s,))) for k, (d, s) in enumerate(((m0, s0), (m1, s1)))]
    got = a5.case_v2(case, names, proofs=proofs, statements=statements, tables=[])
    assert got["params"] == {"delta": "1/100", "proofs": {"members": {
        "m0": {**own[0], "units": [[0, 2]], "vus_per_block": 1}, "m1": {**own[1], "units": [[2, 4]], "vus_per_block": 1}}}}
    e = json.loads(bytes.fromhex(got["evidence"]))
    assert "verdicts" not in e and e["sessions"] == names
    assert [(st["member"], st["public"], [x["name"] for x in st["sessions"]]) for st in e["proofs"]["statements"]] == [
        ("m0", sha(m0 / "pub-2.bin"), names[:1]), ("m1", sha(m1 / "pub-2.bin"), names[1:])]
    assert got["blobs"] == [str(p) for p in (m0 / "circuit.txt", m0 / "pub-2.bin", s0 / "session.json",
                                             m1 / "pub-2.bin", s1 / "session.json")]
    with pytest.raises(ValueError, match="1 names left"):
        a5.case_v2(case, [*names, "x"], proofs=proofs, statements=statements, tables=[])


def test_v2_fixes_its_form_by_the_windows_members_never_by_the_draw(tmp_path):
    """`v2_proof`'s parameters are the window's, fixed before the draw: a window of one member is the one-statement
    form; a window of several is the per-member form, naming every member, the undrawn too, with its units, pin, units
    a proof block and registered file, and a statement a drawn member, even when only one was drawn."""
    def member(k: int, base: int, n: int, reads: dict | None = None) -> dict:
        d = tmp_path / f"m{k}"
        d.mkdir()
        st = {"dir": d, "meta": {"vus_per_block": k + 1}}
        if reads is not None:
            st["registered_file"] = d / "registered.json"
            st["registered_file"].write_text(json.dumps(reads))
        return {"k": k, "m": SimpleNamespace(base=base, n=n), "pin": str(k) * 128, "st": st}

    ms = [member(0, 0, 2), member(1, 2, 3, {"ports": {}})]
    sdir = tmp_path / "session-m1"
    (sdir / "sessions" / "a").mkdir(parents=True)
    (sdir / "sessions" / "a" / "session.json").write_text("{}")
    done = [{"mm": ms[1], "sdir": sdir}]
    got = a5.v2_proof(done, True, ms)
    assert got["proofs"] == {"members": {
        "m0": {"circuit": "0" * 128, "registered": None, "units": [[0, 2]], "vus_per_block": 1},
        "m1": {"circuit": "1" * 128, "registered": {"ports": {}}, "units": [[2, 5]], "vus_per_block": 2}}}
    assert [(m, p.circuit, p.public, p.sessions) for m, p in got["statements"]] == [
        ("m1", tmp_path / "m1" / "circuit.txt", tmp_path / "m1" / "pub-3.bin", (sdir / "sessions" / "a",))]
    one = a5.v2_proof(done, True, ms[1:])
    assert one["proofs"] == {"circuit": "1" * 128, "registered": {"ports": {}}}
    assert [m for m, _ in one["statements"]] == [None]
    iface = tmp_path / "circuit.json"
    iface.write_bytes(b'{"format": "verity/window-circuit/v1"}')
    for mm in ms:
        mm["st"]["interface_file"] = iface
    got = a5.v2_proof(done, True, ms)
    assert got["proofs"]["members"]["m1"]["interface"] == sha(iface) and "interface" not in got["proofs"]["members"]["m0"]
    assert got["interfaces"] == [iface]


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


def test_v2_says_why_an_audit_has_no_case():
    assert "plain" in a5.v2_proof([{}], zk=False, members=[{}])
    assert "drew none" in a5.v2_proof([], zk=True, members=[{}])


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


<<<<<<< 99c8914848
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
||||||| merged common ancestors
=======
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
>>>>>>> 7af4d24fca3e


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
    from verity.core.protocols.sampled_proofs.one_stage import link_points as LP

    monkeypatch.setattr(a5.CC, "law_problem", lambda law, population: None)
    m = {"k": 0, "m": SimpleNamespace(template="t", base=0, n=40), "st": {"meta": {"vus_per_block": 1}}}
    assert a5.link_points({"k": 41, "law": "subset"}, 40, [m]) == {"members": 1, "error": LP.NEVER_ENDS}


def test_a5_records_whether_the_draw_ran_out_of_the_challengers_coins():
    """`a5.draw_ran_out`, which fails the window in v2's per-member form: `subset:9` over 10 units first reads a stream
    of 226 bytes (`2 k · width(n) + 64`), so it runs out of one coin's 64 bytes and not of 256 zero bytes, which reject
    no word; a law that doesn't apply records why."""
    law = {"k": 9, "law": "subset"}
    assert a5.draw_ran_out(law, 10, bytes(64)) is True
    assert a5.draw_ran_out(law, 10, bytes(256)) is False
    assert a5.draw_ran_out({"k": 11, "law": "subset"}, 10, bytes(256)) == "a subset draw of 11 exceeds the population of 10"
