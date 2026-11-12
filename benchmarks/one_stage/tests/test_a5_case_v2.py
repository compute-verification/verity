"""a5's case for the audit on the proofs themselves, `Spot.OneStageZk`'s: v1's case with the evidence's verdicts
replaced by the proofs' files named by SHA-512, the params naming the auditor's circuit and public files' digests, and
blobs the paths of those files and the pinned tables; and the audits v2 can't verify say why. A window of several
members has its case in v2's per-member form beside those, in a file of its own."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

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


def write(out: Path, v2, **more) -> None:
    a5.write_case(out, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], v2=v2, **more)


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


#: a sample's coin evidence in `a5_coins.evidence`'s form: the challenger's key and what its session published
COINS = {"key": {"H": 20, "W": 1, "public": "ab" * 80}, "ledger": ["opened"], "main": ["c0"], "pins": ["pin"],
         "segment": {"label": "draw", "openings": []}, "session": {"log": ["head"]}}
WINDOW_ROOT = "ee" * 64


@pytest.fixture
def zk(monkeypatch, stub_case):
    """`case_v2` answering one fixed case, whose params and evidence name no challenger."""
    case = {"blobs": ["b"], "commitment": "ab", "evidence": stub_case.DT.canonical({"proofs": {}}).encode().hex(),
            "params": {"proofs": {"circuit": "cc" * 64, "registered": None}}}
    monkeypatch.setattr(a5, "case_v2", lambda c, sessions, **k: case)
    return case


def test_a_case_sampled_on_the_challengers_coins_names_the_challenger_and_carries_its_coins(tmp_path, zk, stub_case):
    """v2 reads a sample's coins under the challenger its parameters name, on the window's root, from the evidence's
    coins: the case of an audit sampled on the challenger's coins holds all three, the key in the parameters alone, and
    its sampled-proofs form is that case's. Its proofs, commitment and blobs are the unnamed case's."""
    write(tmp_path, {}, coins=COINS, window_root=WINDOW_ROOT)
    got = json.loads((tmp_path / "case-v2.json").read_text())
    assert got["params"] == {**zk["params"], "challenger": {"key": COINS["key"], "label": "draw"}, "window_root": WINDOW_ROOT}
    evidence = {"coins": {k: v for k, v in COINS.items() if k != "key"}, "proofs": {}}
    assert sorted(evidence["coins"]) == ["ledger", "main", "pins", "segment", "session"]
    assert got["evidence"] == stub_case.DT.canonical(evidence).encode().hex()
    assert (got["blobs"], got["commitment"]) == (zk["blobs"], zk["commitment"])
    sampled = json.loads((tmp_path / "case-sampled.json").read_text())
    assert sampled == a5.case_with_form(got) and sampled["params"]["challenger"] == got["params"]["challenger"]
    assert not (tmp_path / "case-v2-error.txt").exists()


def test_an_audit_on_the_challengers_coins_with_no_window_root_has_no_v2_case(tmp_path, zk):
    """v2 reads the challenger's session as the one its ledger opens on the window's root: with no root to name, no
    case is written, the error file says why, and the v1 case stays."""
    write(tmp_path, {}, coins=COINS, window_root=None)
    assert not (tmp_path / "case-v2.json").exists() and not (tmp_path / "case-sampled.json").exists()
    assert "no window root" in (tmp_path / "case-v2-error.txt").read_text()
    assert (tmp_path / "case.json").exists()


@pytest.mark.parametrize("trees", [False, True])
def test_the_per_member_case_names_the_challenger_only_where_its_sessions_ran_on_the_windows_session(tmp_path, monkeypatch,
                                                                                                     zk, trees):
    """The one statement's case names the challenger its sample's coins are from wherever the audit has their evidence.
    The per-member case reads its sessions' trees from that challenger's session too, so it names the challenger only
    where they ran there."""
    pm = {**zk, "params": {"proofs": {"members": []}}}
    monkeypatch.setattr(a5, "case_members", lambda case, **k: pm)
    write(tmp_path, {}, per_member={}, coins=COINS, window_root=WINDOW_ROOT, member_trees=trees)
    named = {"challenger": {"key": COINS["key"], "label": "draw"}, "window_root": WINDOW_ROOT}
    assert json.loads((tmp_path / "case-v2.json").read_text())["params"] == {**zk["params"], **named}
    assert json.loads((tmp_path / "case-v2-members.json").read_text())["params"] == {**pm["params"], **(named if trees else {})}


@pytest.mark.parametrize("params,evidence", [({"challenger": {}}, {}), ({}, {"coins": {}})])
def test_a_case_that_names_a_challenger_already_is_not_named_again(stub_case, params, evidence):
    case = {"blobs": [], "commitment": "ab", "evidence": stub_case.DT.canonical(evidence).encode().hex(), "params": params}
    with pytest.raises(ValueError, match="already"):
        a5.case_coins(case, COINS, "draw", WINDOW_ROOT)


def test_the_case_is_named_where_replay_names_another_module(monkeypatch, stub_case):
    """A process may hold another directory's `replay` under that name (`a5.own_replay`): the case is named with the
    `replay.py` beside a5 all the same."""
    monkeypatch.setitem(sys.modules, "replay", types.ModuleType("replay"))
    case = {"blobs": [], "commitment": "ab", "evidence": stub_case.DT.canonical({}).encode().hex(), "params": {}}
    got = a5.case_coins(case, COINS, "draw", WINDOW_ROOT)
    assert got["params"]["window_root"] == WINDOW_ROOT and "coins" in json.loads(bytes.fromhex(got["evidence"]))


def test_the_case_carries_the_verifiers_partition_which_v2_verifies_the_sessions_against(tmp_path, monkeypatch, RP):
    monkeypatch.setattr(RP, "descriptor", lambda program: {"program": "p"})
    monkeypatch.setattr(RP, "evidence", lambda *a, **k: {})
    monkeypatch.setattr(RP, "python", lambda c: {"verdict": "accept"})
    part = {"program": "p", "rule": "r"}
    a5.write_case(tmp_path, None, {}, {}, {}, None, 0.01, None, {"format": "f"}, {}, instances=0, sessions=[],
                  session_roots={}, served=[], partition=part)
    assert json.loads((tmp_path / "case.json").read_text())["params"]["partition"] == part


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
