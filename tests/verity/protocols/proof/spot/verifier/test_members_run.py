"""`one_stage.members.members_run` on what the stand-in files of its difftest never reach: members whose files are
held and sessions the verifier accepts. A window of two members on a challenger's own release log, each session closed
after its evidence; the verifier's verdict and its check of a member's files' form are handed in (`verdict`, `form`),
and the read of the auditor's ledger (`challenger.trusted_session_of`, which its own tests hold) is taken as passed.
The run refuses a statement of a member the sample holds no unit of, then holds every member's files before any
session, a member with no statement as one with a session, and refuses a member whose public file is not its slice of
the docket; it credits each accepted session its member's units, refuses proved units other than the evidence's
and a verdict naming another program, and is no verdict: a run of sessions that all accepted is what v1's check at the
coins then reads."""

from __future__ import annotations

import functools
import hashlib
import json
import shutil

import pytest

from verity.circuits import partition_object as PO, population as PN
from verity.circuits import codec
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.types import Array, Tuple, Value
from verity.commitments import coin_tree as CT, registered_value as RV
from verity.protocols.proof.spot.verifier.challenger import challenger as C
from verity.protocols.proof.spot.verifier.one_stage import closing as K, docket as DK, members as M, one_circuit as OC
from verity.protocols.proof.spot.verifier.one_stage import partition as P, proofs as PF

from test_challenger import entropy

XOR16 = PrimitiveDefinition("MembersRunXor16", 1, (("a", Value(16)), ("b", Value(16))), Value(16), lambda a, b: a ^ b,
                           register=False)
#: each template's ``x`` a 128-byte row, past a compression before its last block, which a Flock block's pin needs
PAIR = bind(CompositeDefinition("MembersRunPair", 0, (), lambda S: ((("x", Array(64, Value(16))),), Value(16)),
                                lambda B, S, x: B.call(XOR16, x[0], x[1]), register=False))
TRIPLE = bind(CompositeDefinition("MembersRunTriple", 0, (), lambda S: ((("x", Array(64, Value(16))),), Value(16)),
                                  lambda B, S, x: B.call(XOR16, B.call(XOR16, x[0], x[1]), x[2]), register=False))
PROGRAM = PN.mixed_population_program([(PAIR, 2), (TRIPLE, 3)])
QUERY = P.template_instances_query([PAIR.id, TRIPLE.id])
A, B = "a/c0", "b/c0"
UNITS = {A: [[0, 2]], B: [[2, 5]]}
PARTITION = PO.build(PROGRAM, QUERY, [[UNITS[A][0], UNITS[B][0]]])
SCHEME = "frame-v3-sha512/hm96-sha512"
SPEC = {"block": 8, "rounds_log": 1, "scheme": CT.SCHEME, "streams": [CT.SESSION, "circuit/rep0", "circuit/rep1"]}
ROOTS = {A: ("11" * 64, "aa" * 64), B: ("22" * 64, "bb" * 64)}
#: each template's one output field's tree, ``<template>/o0``, its leaves the template's units
TREES = {A: ("a/o0", 2, "66" * 64, "88" * 64), B: ("b/o0", 3, "77" * 64, "99" * 64)}
WINDOW_ROOT = "ee" * 64
#: each template's run input node's one tree, ``input/n<w>``, its rows the node's 64-leaf rows, its domain under the
#: window's open root (`members.run_tree_held`)
INPUTS = {m: (f"input/n{w}", n, r, RV.domain(P.program_digest(PROGRAM), f"input/n{w}", 64, n, 16,
                                              window=WINDOW_ROOT).domain_id.hex())
          for m, w, n, r in ((A, 0, 2, "33" * 64), (B, 1, 3, "44" * 64))}
RECORD = {"population": 5, "scheme": SCHEME,
          "roots": [{"domain": ROOTS[m][1], "leaves": 4, "port": f"{m}/out", "root": ROOTS[m][0]} for m in (A, B)] +
                   [{"domain": d, "leaves": n, "port": t, "root": r} for t, n, r, d in TREES.values()] +
                   [{"domain": d, "leaves": n, "port": t, "root": r} for t, n, r, d in INPUTS.values()]}
SAMPLED = {A: [1], B: [0, 2]}


def slices(units: dict, record: dict) -> dict:
    """`members.slices` under the window's program and query."""
    return M.slices(units, record, program=PROGRAM, query=QUERY, window_root=WINDOW_ROOT)
DESCRIPTOR = json.dumps(json.loads(json.dumps(codec.encode_program(PROGRAM))), sort_keys=True,
                        separators=(",", ":")).encode()
lean = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")


@functools.cache
def lowered(member: str) -> bytes:
    """The member's circuit file: its block's C_S, lowered by Lean (`one_circuit.lower`)."""
    return OC.lower(OC.restrict(DESCRIPTOR, PARTITION, UNITS[member]), PARTITION, PF.lower_executable,
                    cut=OC.cuts(DESCRIPTOR, PARTITION), formed=OC.formed_in(DESCRIPTOR, PARTITION, UNITS[member]))


def reads(member: str, **changes) -> dict:
    """The member's registered file: its port ``x`` opening its template's run input node's one tree."""
    t, n, r, d = INPUTS[member]
    return {"format": "verity/registered-reads/v2",
            "ports": {"x": {"domain": d, "leaves": n, "root": r, "value": t, **changes}}}


def frame(member: str, out: tuple[str, str] | None = None, **more: tuple[str, str]) -> dict:
    """A header's ``frame_v3``: the member's ``out`` and its template's tree at ``o0``, ``out`` and ``more`` each a
    root and a domain."""
    ports = {"out": out or ROOTS[member], "o0": TREES[member][2:], "x": INPUTS[member][2:], **more}
    return {"domain_ids": {p: d for p, (_, d) in ports.items()}, "roots": {p: r for p, (r, _) in ports.items()}}


def input_entry(member: str) -> dict:
    """The slice's entry of the member's port ``x`` at its run input node's one tree (`OneStageZk.runInputRootsAt`)."""
    t, n, r, d = INPUTS[member]
    return {"domain": d, "leaves": n, "port": "x", "root": r, "tree": t}


def tree_entry(member: str, **changes) -> dict:
    """The slice's entry of the member's template's tree at ``o0`` (`OneStageZk.ownFieldsOf`)."""
    t, n, r, d = TREES[member]
    return {"domain": d, "leaves": n, "port": "o0", "root": r, "tree": t, **changes}


def header(member: str, **changes) -> dict:
    return {"frame_v3": frame(member),
            "instances": sum(hi - lo for lo, hi in UNITS[member]), "scheme": SCHEME,
            "units": {"indices": [u for lo, hi in UNITS[member] for u in range(lo, hi)]}, **changes}


class Window:
    """The two members' files under `at`, the sessions of the members of `sessions` on one challenger's release log,
    each tree's link, one round and its closing round before the next tree's first, and the per-member parameters and
    evidence over them. `headers` are the members' public files' headers (`header`'s by default). A member's circuit
    file is its block's C_S lowered (`lowered`), among the files once it is derived."""

    def __init__(self, at, headers: dict | None = None, sessions: tuple[str, ...] = (A, B)) -> None:
        self.at, self.files, self.underived = at, [], {}
        program = json.loads(DESCRIPTOR)
        self.descriptor = DESCRIPTOR
        challenger = C.LiveChallenger("verity/test/coins/v1", 0, [("sample", 1)], entropy("members-run"))
        log = [json.dumps({"event": "commitment", "root": challenger.commitment.root.hex()})]
        statements, members = [], {}
        for k, name in enumerate((A, B)):
            public = json.dumps((headers or {}).get(name) or header(name)).encode() + b"\nthe rows"
            try:
                self.put(lowered(name))
            except (OC.NotDerived, OC.Refused) as e:
                self.underived[name] = str(e)
            members[name] = {"public": self.put(public), "registered": reads(name), "block": k, "vus_per_block": 1}
            if name not in sessions:
                continue
            tree, line = challenger.sample_tree(SPEC, bytes([k]) * CT.NONCE_BYTES)
            log.append(line)
            root, root_f, root_b, message = tree.root().hex(), bytes([7]) * 8, bytes([8]) * 8, bytes([9]) * 64

            def opened(stream: str, rnd: int, n: int, after: bytes) -> list[int]:
                coins, _, line = challenger.open_tree(root, stream, rnd, n, after)
                log.append(line)
                return coins
            points = opened(CT.SESSION, 0, 2, C.link_after(root_f, {"circuit": root_b}))
            coins = opened("circuit/rep0", 0, 3, message)
            record = json.dumps({
                "coin_tree": tree.record(), "unit_draw": {"units": SAMPLED[name]},
                "link": {"points": [[f"{c:032x}"] for c in points], "root_b": {"circuit": root_b.hex()},
                         "root_f": root_f.hex()},
                "streams": [{"stream": "circuit/rep0", "rounds": [{"coins": [f"{c:032x}" for c in coins], "k": 0,
                                                                  "msg_sha512": message.hex(), "n": 3}]}]}).encode()
            rep = f"proof of {name}".encode()
            opened(CT.SESSION, C.CLOSING_ROUND, 0,
                   hashlib.sha512(K.session_evidence(record, [("circuit.rep0.bin", rep)])).digest())
            statements.append({"member": name, "sessions": [{"name": f"s{k}", "record": self.put(record),
                                                             "reps": {"circuit.rep0.bin": self.put(rep)}}]})
        self.params = {"challenger": {"key": {"W": 8, "H": 5, "public": "00" * C.PUBLIC_BYTES}, "label": "sample"},
                       "partition": PARTITION, "program": program,
                       "query": QUERY, "window_root": WINDOW_ROOT, "proofs": {"members": members}}
        self.evidence = {"coins": {"ledger": [], "main": [], "pins": [], "segment": {}, "session": {"log": log}},
                         "proved_units": [u + UNITS[m][0][0] for m in (A, B) if m in sessions for u in SAMPLED[m]],
                         "sessions": [s["name"] for st in statements for s in st["sessions"]],
                         "proofs": {"statements": statements}}

    def circuit(self, member: str) -> str:
        """The SHA-512 of the member's circuit file."""
        return hashlib.sha512(lowered(member)).hexdigest()

    def put(self, b: bytes) -> str:
        d = hashlib.sha512(b).hexdigest()
        (self.at / d[:16]).write_bytes(b)
        self.files.append(self.at / d[:16])
        return d

    def run(self, verdict=None, record: dict = RECORD, evidence: dict | None = None, params: dict | None = None,
            program=PROGRAM, form=None, tables: dict | None = None):
        """`members_run` on the window's files and, under their pinned SHA-512s whatever their bytes, the `tables`."""
        return M.members_run(self.params if params is None else params, record,
                             self.evidence if evidence is None else evidence, M.blobs_of(self.files) | (tables or {}),
                             program=program, verify=lambda key, message, signature: False,
                             verdict=verdict or accepting, form=form or formed)


def accepting(st: M.Statement) -> dict:
    return {"accepted": True, "why": "", "program_sha512": st.partition["program"]}


def formed(circuit, public, tables) -> None:
    return None


def never(st: M.Statement) -> dict:
    raise AssertionError("a session was verified before the refusal")


@pytest.fixture(autouse=True)
def ledger(monkeypatch):
    monkeypatch.setattr(C, "trusted_session_of", lambda *a: None)


@lean
def test_a_run_of_accepted_sessions_credits_each_members_units_and_names_no_refusal(tmp_path):
    """Each session is verified on its member's own files, with the program and partition every member reads, and
    credited the units of its member's public file its record drew; the run is no verdict, and it names no refusal
    for the audit above it."""
    w = Window(tmp_path)
    seen, circuits = [], []

    def verdict(st: M.Statement) -> dict:
        seen.append(st)
        circuits.append(PF.file_sha512(st.circuit))
        return accepting(st)
    run, refused = w.run(verdict)
    assert refused is None and run.all and run.proved == (1, 2, 4), refused
    assert [(f.session, f.accepted, f.units, f.program) for f in run.findings] == [
        ("s0", True, (1,), PARTITION["program"]), ("s1", True, (2, 4), PARTITION["program"])]
    assert run.all and M.run_reason(run, refused) is None
    ms = w.params["proofs"]["members"]
    # C_S is derived under a directory the run holds only while it runs
    assert circuits == [w.circuit(A), w.circuit(B)]
    assert [PF.file_sha512(st.public) for st in seen] == [ms[A]["public"], ms[B]["public"]]
    assert all(json.loads(st.program) == w.params["program"] and st.partition == PARTITION
               and [n for n, _ in st.reps] == ["circuit.rep0.bin"] for st in seen)


@lean
def test_proved_units_other_than_the_evidences_and_a_verdict_of_another_program_are_refused(tmp_path):
    w = Window(tmp_path)
    assert w.run(evidence=w.evidence | {"proved_units": [1, 2]}) == (None, M.PROVED_UNITS)
    assert w.run(lambda st: accepting(st) | {"program_sha512": "ab"}) == (
        None, f"member {A}: session s0's verdict names program ab, not the partition object's "
              f"\"{PARTITION['program']}\"")
    got = w.run(lambda st: {"accepted": True, "why": ""})
    assert got == (None, f"member {A}: session s0's verdict names program none, not the partition object's "
                         f"\"{PARTITION['program']}\"")


def test_a_statement_of_a_member_the_sample_missed_is_refused_before_any_session(tmp_path):
    """A session proves the units of the sample that are its member's, so a statement of a member the sample holds no
    unit of is a session that proves none: the run refuses it by the member's name before it reads a member's file or
    verifies a session, the first such statement in the statements' order."""
    w = Window(tmp_path)
    missed = "the sample holds none of its units, and a member the sample missed has no statement"
    assert w.run(never, evidence=w.evidence | {"proved_units": [0, 1]}) == (None, f"member {B}: {missed}")
    assert w.run(never, evidence=w.evidence | {"proved_units": [3]}) == (None, f"member {A}: {missed}")
    assert w.run(never, evidence=w.evidence | {"proved_units": []}) == (None, f"member {A}: {missed}")


@lean
def test_a_session_the_verifier_rejects_is_a_finding_and_the_run_proves_only_the_others_units(tmp_path):
    w = Window(tmp_path)
    run, refused = w.run(lambda st: accepting(st) if st.public.read_bytes().startswith(
        json.dumps(header(B)).encode()) else {"accepted": False, "why": "S16"})
    assert refused is None and not run.all and run.proved == (2, 4) and M.run_reason(run, refused) is None, refused
    assert [(f.accepted, f.why, f.units) for f in run.findings] == [(False, "S16", ()), (True, "", (2, 4))]


@pytest.mark.parametrize("change, why", [
    ({"units": {"indices": [0, 1, 2]}}, M.MEMBER_UNITS),
    ({"units": {"indices": [1, 0]}}, M.MEMBER_UNITS),
    ({"units": {}}, M.MEMBER_UNITS),
    ({"scheme": "another"}, "public file: not the registration's: scheme"),
    ({"frame_v3": frame(A, ("33" * 64, "aa" * 64))}, "public file: not the registration's: roots"),
    ({"frame_v3": frame(A, ("11" * 64, "cc" * 64))}, "public file: not the registration's: domains"),
    ({"frame_v3": frame(A, y=("11" * 64, "aa" * 64))}, M.unwired_why("y")),
    ({"instances": 5}, "public file: not the registration's: population"),
    ({"instances": 3, "scheme": 5}, M.POPULATION_WHY),
])
@lean
def test_a_member_is_refused_where_its_public_file_is_not_its_slice_with_or_without_a_session(tmp_path, change, why):
    """The first member's public file names other units than the member's, in another order or none, or another scheme,
    root, domain or count than the member's roots of the docket, ports bare: the run refuses the member with the
    reason before any session is verified, and as it does where the member has no statement."""
    for k, sessions in enumerate(((A, B), (B,))):
        (tmp_path / str(k)).mkdir()
        w = Window(tmp_path / str(k), {A: header(A, **change)}, sessions)
        assert w.run(never) == (None, f"member {A}: {why}")


@lean
def test_every_members_files_are_held_in_the_members_order_before_any_session(tmp_path):
    """A window whose first member has no statement: its files are read and held as the second's are, both before the
    second's session is verified, and the run credits that session's units. A member whose circuit no blob holds, or
    whose files the verifier's form check refuses, is refused by its name, with no statement as with one. Its
    circuit is its block's C_S lowered (`lowered`)."""
    w = Window(tmp_path, sessions=(B,))
    ms, seen = w.params["proofs"]["members"], []

    def form(circuit, public, tables):
        seen.append((PF.file_sha512(circuit), PF.file_sha512(public), tables))

    def verdict(st: M.Statement) -> dict:
        seen.append("session")
        return accepting(st)
    run, refused = w.run(verdict, form=form)
    assert refused is None and run.all and run.proved == (2, 4) and [f.session for f in run.findings] == ["s1"], refused
    assert seen == [(w.circuit(A), ms[A]["public"], ()), (w.circuit(B), ms[B]["public"], ()), "session"]

    def refusing(circuit, public, tables):
        return "refused: x" if PF.file_sha512(circuit) == w.circuit(A) else None
    for sessions in ((B,), (A, B)):
        w = Window(tmp_path, sessions=sessions)
        assert w.run(never, form=refusing) == (None, f"member {A}: refused: x")
        w.files = [f for f in w.files if PF.file_sha512(f) != w.circuit(A)]
        assert w.run(never, form=refusing) == (None, f"member {A}: refused: x")


def test_a_member_is_held_to_its_own_circuits_tables_among_the_blobs(tmp_path):
    """The tables a statement is handed are its circuit's lookups among the blobs' pinned tables (`own_tables`), and
    are refused where they are not, by name, its circuit's lookups: one it does not look up as one it lacks."""
    rcp, ex2 = tmp_path / "rcp.u32", tmp_path / "ex2.u32"
    held = {PF.MUFU_SHA512["rcp"]: rcp, PF.MUFU_SHA512["ex2"]: ex2}
    assert PF.own_tables(["rcp"], held) == ["rcp"] and PF.own_tables([], held) == []
    assert PF.own_tables(["sqrt", "rcp", "ex2"], held) == ["rcp", "ex2"]
    assert PF.tables_match_why(["rcp"], ["rcp"]) is None and PF.tables_match_why([], []) is None
    assert PF.tables_match_why(["rcp"], []) == f"{PF.NO_TABLES}: the circuit looks up none, the blobs hold rcp"
    assert PF.tables_match_why(["rcp"], ["rcp", "ex2"]) == (
        f"{PF.NO_TABLES}: the circuit looks up rcp, ex2, the blobs hold rcp")
    assert PF.tables_why([], held) == f"{PF.NO_TABLES}: the circuit looks up none, the blobs hold rcp, ex2"


def test_a_members_files_are_refused_by_the_first_of_leans_checks_that_applies(tmp_path):
    """`files_why` in Lean's order: a circuit that does not parse or a public file that does not load (the form check's
    refusal, the scope's word for word), the template's refusal, the rest of the form, then the population check."""
    strata = P.strata(PROGRAM, QUERY)
    slice_ = slices(UNITS, RECORD)[B]
    plain, empty, public, other = (tmp_path / n for n in ("plain", "empty", "public", "other"))
    plain.write_bytes(b"flock-circuit/v1 u\nMETA {}\n")
    empty.write_bytes(b"flock-circuit/v1 u\nMETA {\"units_per_vu\": 0}\nCIRCUIT unit 0\nflock-ir-unit/v2 unit 0 0 1\n")
    public.write_bytes(json.dumps(header(B)).encode() + b"\nthe rows")
    other.write_bytes(json.dumps(header(B, instances=5)).encode() + b"\nthe rows")

    def why(circuit, public, formed):
        return M.files_why(circuit, public, (), UNITS[B], strata, slice_, lambda *a: formed)
    assert PF.template_why(empty) == PF.NO_ROW_WHY and PF.template_why(plain) is None
    for unread in M.UNREAD:
        assert why(empty, other, f"{unread}x)") == f"{unread}x)"
    assert why(empty, other, "refused: a form") == PF.NO_ROW_WHY
    assert why(plain, other, "refused: a form") == "refused: a form"
    assert why(plain, other, None) == "public file: not the registration's: population"
    assert why(plain, public, None) is None
    with pytest.raises(M.Outside, match="strata"):
        M.files_why(plain, public, (), UNITS[B], ValueError("no"), slice_, formed)


def test_the_members_population_check_reads_the_registrations_strata(tmp_path):
    strata = P.strata(PROGRAM, QUERY)
    slice_ = slices(UNITS, RECORD)[B]
    assert M.population_why(UNITS[B], strata, header(B), slice_) is None
    assert M.population_why([[2, 6]], strata, header(B), slice_) == M.MEMBER_STRATA
    assert M.population_why(UNITS[B], strata[::-1], header(B), slice_) == M.MEMBER_STRATA
    assert M.population_why(UNITS[B], strata, "no header", slice_) == M.MEMBER_UNITS


def test_under_a_subset_design_the_strata_need_only_cover_the_units():
    """PAIR, TRIPLE, PAIR: the strata cover the units out of order, which a subset design's member check takes and a
    stratified one's refuses; an overlap, a gap and a range past the units are no cover."""
    program = PN.mixed_population_program([(PAIR, 2), (TRIPLE, 3), (PAIR, 2)])
    strata = P.strata(program, QUERY)
    assert [(t, [tuple(r) for r in rs]) for t, rs in strata] == [(PAIR.id, [(0, 2), (5, 7)]), (TRIPLE.id, [(2, 5)])]
    assert not P.in_order(program, QUERY, 7) and P.covers(program, QUERY, 7) and not P.covers(program, QUERY, 6)
    slice_ = slices(UNITS, RECORD)[B]
    assert M.population_why(UNITS[B], strata, header(B), slice_, subset=True) is None
    assert M.population_why(UNITS[B], strata, header(B), slice_) == M.MEMBER_STRATA
    assert M.population_why(UNITS[B], strata, "no header", slice_, subset=True) == M.MEMBER_UNITS
    assert P.cover([("p", [[0, 2], [5, 7]]), ("t", [[2, 5], [5, 5]])]) == 7
    assert P.cover_ranges([[0, 2], [5, 7], [2, 5], [5, 4]]) == 7
    for bad in ([("p", [[0, 3], [5, 7]]), ("t", [[2, 5]])], [("p", [[0, 2], [6, 7]]), ("t", [[2, 5]])],
                [("p", [[0, 2], [5, 7], [7, 6]]), ("t", [[2, 5]])]):
        assert P.cover(bad) is None
        assert M.population_why(UNITS[B], bad, header(B), slice_, subset=True) == M.MEMBER_COVER
    assert M.population_why([[2, 8]], strata, header(B), slice_, subset=True) == M.MEMBER_COVER
    in_order = P.strata(PROGRAM, QUERY)
    assert M.population_why([[2, 8]], in_order, header(B), slice_, subset=True) == M.MEMBER_STRATA
    assert P.cover(in_order) == 5 and P.in_order(PROGRAM, QUERY, 5) and P.covers(PROGRAM, QUERY, 5)
    assert all(M.population_why(UNITS[B], in_order, header(B), slice_, subset=s) is None for s in (False, True))
    assert M.subset_design({"law": "subset", "k": 1})
    assert not any(map(M.subset_design, ({"law": "stratified", "k": 1}, {"law": "work", "k": 1}, "rate:1/7",
                                         "stratified-rate:1/2")))
    rate, strat = (M.resolved({"law": law, "query": QUERY}, program) for law in ("rate:2/7", "stratified-rate:1/2"))
    assert M.subset_design(rate) and not M.subset_design(strat) and M._kind(strat) == "stratified"
    assert M.resolved({"law": "rate:9/7", "query": QUERY}, program) is None
    assert "stratified" in M.strata_design("stratified")


def test_the_registrations_roots_are_each_one_members_and_none_a_row_tables(tmp_path):
    w = Window(tmp_path)
    roots = RECORD["roots"]

    def refused(*more: dict, params: dict | None = None) -> str:
        run, why = w.run(record=RECORD | {"roots": [*roots, *more]}, params=params)
        assert run is None
        return why
    assert refused({"port": M.TABLE_PORT, "root": "00" * 64}) == M.ROW_TABLES
    assert refused({"port": "zzz/out"}) == ("roots: the registration's root zzz/out is no member's: a member's roots "
                                            "are named <member>/<port>, and a window's tree of a template's field "
                                            "<template>/<field>")
    assert refused({"port": A}).startswith(f"roots: the registration's root {A} is no member's")
    assert refused({"root": "00" * 64}) == "roots: a root of the registration names no port"
    ms = w.params["proofs"]["members"]
    sts = w.evidence["proofs"]["statements"]
    for top in (A, "a"):
        nested = w.params | {"proofs": {"members": {top: ms[A] | {"registered": None},
                                                    f"{A}/x": ms[B] | {"registered": None}}}}
        statements = [sts[0] | {"member": top}, sts[1] | {"member": f"{A}/x"}]
        run, why = w.run(record=RECORD | {"roots": [{"port": f"{A}/x/out"}]}, params=nested,
                         evidence=w.evidence | {"proofs": {"statements": statements}})
        assert run is None and why == (DK.name_template(A) if top == A else
                                       f"roots: the registration's root {A}/x/out is two members': a member's roots "
                                       "are named <member>/<port>, and a root is one member's")
    bare = w.params | {"proofs": {"members": {n: m | {"registered": None} for n, m in ms.items()}}}
    assert w.run(record=RECORD | {"roots": {}}, params=bare) == (None, "roots: the registration's roots are not a list")
    assert w.run(record={"scheme": SCHEME, "roots": roots}) == (None, "the record names no population")


def test_the_registrations_field_trees_are_the_ones_c_and_the_partition_give(tmp_path):
    """The run derives from C and the partition the trees a registration holds, one ``<template>/o<j>`` per output
    field of each member's template, its leaves the template's units (`OneStageZk.derivedFieldTrees`), and refuses a
    registration that omits one, splits one in two (two roots of the tree, or the members' own roots in its place),
    holds one at another leaf count, or holds a tree of no output field, by name, before any session."""
    w = Window(tmp_path)
    own, (ta, tb) = [*RECORD["roots"][:2], *RECORD["roots"][4:]], RECORD["roots"][2:4]

    def why(*trees: dict) -> tuple:
        return w.run(never, record=RECORD | {"roots": [*own, *trees]})
    assert why(ta) == (None, M.field_trees_why("it omits b/o0"))
    assert why(ta, {**tb, "leaves": 2}, {**tb, "leaves": 1}) == (None, M.field_trees_why("it holds b/o0 more than once"))
    assert why(ta, {**tb, "port": f"{B}/o0"}) == (None, M.field_trees_why("it omits b/o0"))
    assert why(ta, {**tb, "leaves": 4}) == (None, M.field_trees_why("it holds b/o0 at 4 leaves, not 3"))
    assert why(ta, tb, {**tb, "port": "b/o1"}) == (None, M.field_trees_why("it holds b/o1, no output field's tree"))


def test_a_members_slice_outside_the_run_is_the_runs_own():
    """`slices`, which an audit outside the run reads a member's slice by: each member's count of units, its roots of
    the docket with its name taken off, and the scheme, at which its honest header holds; a root the record names
    under a member that its header does not is refused there (a port wired to another member's tree, named under its
    reader, is in the reader's slice and so must be in its header); and where the run refuses the record, so does it,
    with the run's reason."""
    got = slices(UNITS, RECORD)
    assert got == {m: {"population": n, "scheme": SCHEME,
                       "roots": [{"domain": ROOTS[m][1], "leaves": 4, "port": "out", "root": ROOTS[m][0]},
                                 tree_entry(m), input_entry(m)]}
                   for m, n in ((A, 2), (B, 3))}
    strata = P.strata(PROGRAM, QUERY)
    assert all(M.population_why(UNITS[m], strata, header(m), got[m]) is None for m in (A, B))
    wired = RECORD | {"roots": [*RECORD["roots"], {**RECORD["roots"][0], "port": f"{B}/y"}]}
    both = slices(UNITS, wired)
    assert [r["port"] for r in both[B]["roots"]] == ["out", "y", "o0", "x"] and both[A] == got[A]
    assert M.population_why(UNITS[B], strata, header(B), both[B]) == "public file: not the registration's: roots, domains"
    named = header(B, frame_v3=frame(B, y=ROOTS[A]))
    assert M.population_why(UNITS[B], strata, named, both[B]) is None
    for record, why in ((RECORD | {"roots": [*RECORD["roots"], {"port": M.TABLE_PORT, "root": "00" * 64}]}, M.ROW_TABLES),
                        (RECORD | {"population": 6}, M.MEMBERS_LAID),
                        (RECORD | {"roots": {}}, "roots: the registration's roots are not a list")):
        with pytest.raises(ValueError) as e:
            slices(UNITS, record)
        assert str(e.value) == why


def test_a_members_slice_that_names_a_port_twice_is_refused_whatever_its_roots(tmp_path):
    """A record naming a member's port twice, at its true root and at another, gives a slice naming the port twice,
    which the run and `slices` refuse by name before any session (`OneStageZk.Member.portsOnce`); the same roots with
    the other under the other member, so each slice names the port once, pass. A root naming no port as a string
    refuses by name."""
    w = Window(tmp_path)
    other = {**RECORD["roots"][0], "root": "33" * 64, "domain": "44" * 64}
    twice = RECORD | {"roots": [*RECORD["roots"], other]}
    assert w.run(never, record=twice) == (None, M.port_twice_why(A, "out"))
    with pytest.raises(ValueError) as e:
        slices(UNITS, twice)
    assert str(e.value) == M.port_twice_why(A, "out")
    once = slices(UNITS, RECORD | {"roots": [*RECORD["roots"], {**other, "port": f"{B}/out2"}]})
    assert [r["port"] for r in once[A]["roots"]] == ["out", "o0", "x"]
    assert [r["port"] for r in once[B]["roots"]] == ["out", "out2", "o0", "x"]
    assert M.port_twice_why(A, "out") == (f"member {A}: its slice names port out twice: a member's slice holds one root "
                                          "of each port, the one its public file's root there is held to")
    with pytest.raises(M._Refused) as e:
        M._ports_once({A: {}}, {A: [{"port": 1, "root": "aa"}]})
    assert str(e.value) == M.slice_no_port_why(A)


def test_a_members_own_output_port_reads_its_templates_field_tree():
    """A root ``<template>/<field>`` of a member's template is the window's tree of the field, no member's own: a
    member of the template with no root at that field has the tree in its slice under the field's port
    (`OneStageZk.ownFieldsOf`), at which a header naming it holds; a member with its own root at a field its template
    has the tree of is refused, in either order of the two roots (`OneStageZk.ownFieldsHeld`), in `slices` and in the
    one statement's ports (`public_ports`): a wire from the field would read the window's tree, the member its own."""
    tree = {"domain": "cc" * 64, "leaves": 3, "port": "b/o0", "root": "33" * 64}
    got = slices(UNITS, RECORD | {"roots": [*RECORD["roots"][:3], tree, *RECORD["roots"][4:]]})
    assert got[B]["roots"][1] == {"domain": "cc" * 64, "leaves": 3, "port": "o0", "root": "33" * 64, "tree": "b/o0"}
    assert got[A] == slices(UNITS, RECORD)[A]
    for roots in ([*RECORD["roots"], {**tree, "port": "a/out"}], [{**tree, "port": "a/out"}, *RECORD["roots"]]):
        with pytest.raises(ValueError) as e:
            slices(UNITS, RECORD | {"roots": roots})
        assert str(e.value) == DK.own_field(A, "out")
    with pytest.raises(ValueError) as e:
        DK.public_ports({"roots": [RECORD["roots"][0], {**tree, "port": "a/out"}]})
    assert str(e.value) == DK.own_field(A, "out")
    assert set(DK.public_ports({"roots": [RECORD["roots"][0], {**tree, "port": "a/o0"}]})) == {"out", "o0"}
    strata = P.strata(PROGRAM, QUERY)
    named = header(B, frame_v3=frame(B, o0=("33" * 64, "cc" * 64)))
    assert M.population_why(UNITS[B], strata, named, got[B]) is None


def test_a_members_rows_of_a_shared_tree_are_its_registered_reads_there():
    """A member's port the slice holds at its template's field tree (``o0`` at ``b/o0``) is held by the member's
    registered reads there, not by its header's own root (`OneStageZk.headerHeldAt`): with a read ``(o0, b/o0)`` the
    header's own root at ``o0`` is not compared, and the header's other ports still are; with no read there, or a read
    of ``o0`` at another tree, a header root not the registration's is refused by name (`shared_unread`), and one that
    is the registration's still holds."""
    tree = {"domain": "cc" * 64, "leaves": 3, "port": "b/o0", "root": "33" * 64}
    got = slices(UNITS, RECORD | {"roots": [*RECORD["roots"][:3], tree, *RECORD["roots"][4:]]})[B]
    strata = P.strata(PROGRAM, QUERY)

    def own(o0: str, out: str = ROOTS[B][0]) -> dict:
        return header(B, frame_v3={"domain_ids": {"out": ROOTS[B][1], "o0": "dd" * 64, "x": INPUTS[B][3]},
                                   "roots": {"out": out, "o0": o0, "x": INPUTS[B][2]}})
    assert M.population_why(UNITS[B], strata, own("44" * 64), got, {("o0", "b/o0")}) is None
    assert M.population_why(UNITS[B], strata, own("44" * 64), got) == M.shared_unread("o0", "b/o0")
    assert M.population_why(UNITS[B], strata, own("44" * 64), got, {("o0", "a/o0")}) == M.shared_unread("o0", "b/o0")
    assert M.population_why(UNITS[B], strata, own("44" * 64, out="55" * 64), got, {("o0", "b/o0")}) == (
        "public file: not the registration's: roots")
    assert M.population_why(UNITS[B], strata, own("44" * 64) | {"instances": 99}, got, {("o0", "b/o0")}) == (
        M.POPULATION_WHY)
    named = header(B, frame_v3=frame(B, o0=("33" * 64, "cc" * 64)))
    assert M.population_why(UNITS[B], strata, named, got) is None


def registered_file(ports: dict[str, str]) -> dict:
    """A `verity/registered-reads/v2` file of the ports given, each its name and value."""
    return {"format": "verity/registered-reads/v2",
            "ports": {p: {"domain": "dd" * 64, "leaves": 2, "root": "aa" * 64, "value": v} for p, v in ports.items()}}


def test_a_members_port_is_held_by_its_registered_reads_only_at_a_registered_value_its_file_holds():
    """A port the slice holds no root of (``w``) is not compared where the verifier's registered file holds it at the
    value the reads name, as a registered value (`OneStageZk.headerHeldAt`): `Flock.Registered.checkPort` holds its
    rows to that file's root. Read at any other value, it is refused by name (`unregistered_why`), through the file
    as `_files` reads it (`registered_reads`) and with the reads and the file apart; with no read of it, as before
    (`unwired_why`); a port the slice holds is compared whatever the reads name."""
    strata = P.strata(PROGRAM, QUERY)
    got = slices(UNITS, RECORD)[B]

    def own(out: str = ROOTS[B][0]) -> dict:
        return header(B, frame_v3=frame(B, (out, ROOTS[B][1]), w=("77" * 64, "dd" * 64)))

    def why(ports: dict[str, str]) -> str | None:
        return M.population_why(UNITS[B], strata, own(), got, *M.registered_reads(registered_file(ports)))
    assert why({"w": "h1/w"}) is None
    for v in ("b/o0", "input/n3", "committed/w"):
        assert why({"w": v}) == M.unregistered_why("w", v)
    assert why({"v": "h1/w"}) == M.unwired_why("w")
    assert M.population_why(UNITS[B], strata, own(), got) == M.unwired_why("w")
    held = registered_file({"w": "h1/w"})["ports"]
    assert M.population_why(UNITS[B], strata, own(), got, {("w", "h1/z")}, held) == M.unregistered_why("w", "h1/z")
    assert M.population_why(UNITS[B], strata, own(), got, {("w", "h1/w")}) == M.unregistered_why("w", "h1/w")
    assert M.population_why(UNITS[B], strata, own(out="55" * 64), got,
                            *M.registered_reads(registered_file({"w": "h1/w", "out": "h1/w"}))) == (
        "public file: not the registration's: roots")


def test_a_member_named_as_another_members_template_is_refused():
    """Members ``a`` and ``a/c0``: a root ``a/out`` would be the member ``a``'s own port and template ``a``'s tree of
    ``out`` at once, so the window is refused by name in either order of the members (`OneStageZk.NAME_TEMPLATE`)."""
    for units in ({"a": [[0, 2]], A: [[2, 5]]}, {A: [[0, 2]], "a": [[2, 5]]}):
        with pytest.raises(ValueError) as e:
            slices(units, RECORD | {"roots": [{**RECORD["roots"][0], "port": "a/out"}]})
        assert str(e.value) == DK.name_template("a")


def test_a_registered_files_committed_trees_are_its_windows(tmp_path):
    """A member's registered port that opens a committed tree, a value ``committed/<port>``, is held to the record's
    root of it (`OneStageZk.treePortsHeld`) before any session: a genuine tree at another root (the last window's), the
    window's root under another domain, another leaf count, another member's own tree, and a tree the record holds no
    root of are each refused by name, and a port whose value is a writer's field tree's name is refused by name."""
    w = Window(tmp_path)
    ms = w.params["proofs"]["members"]
    tree = {"domain": "cc" * 64, "leaves": 4, "port": "a/out", "root": "33" * 64}
    record = RECORD | {"roots": [*RECORD["roots"], tree]}

    def why(root: str = tree["root"], domain: str = tree["domain"], leaves: int = 4,
            value: str = "committed/a/out") -> str | None:
        port = {"value": value, "root": root, "domain": domain, "leaves": leaves}
        reg = {"format": "verity/registered-reads/v2", "ports": {"x": port}}
        return w.run(record=record, params=w.params | {"proofs": {"members": {A: ms[A], B: ms[B] | {"registered": reg}}}})[1]

    def other(t: str, what: str) -> str:
        return (f"registered: member {B}'s port x opens the committed tree {t} at another {what} than the "
                "registration's: a statement opens its window's registered tree, never another window's or another "
                "member's")
    assert "opens the committed tree" not in (why() or "")
    assert why(root="44" * 64) == other("a/out", "root")
    assert why(domain="dd" * 64) == other("a/out", "domain")
    assert why(leaves=5) == other("a/out", "leaf count")
    assert why(root=ROOTS[A][0], domain=ROOTS[A][1]) == other("a/out", "root")
    assert why(value="committed/z/out") == (
        f"registered: member {B}'s port x opens the committed tree z/out, and the registration holds no root z/out: "
        "a committed tree a statement opens is one its window registered before any coin")
    assert "opens the committed tree" not in (why(value=f"committed/{A}/out", root=ROOTS[A][0], domain=ROOTS[A][1])
                                              or "")
    assert why(value=f"committed/{A}/out") == other(f"{A}/out", "root")
    assert why(value="a/o0") == M.wired_tree_why(B, "x")
    assert why(value="b/o0") == M.wired_tree_why(B, "x")


def test_a_members_own_output_opens_its_templates_field_tree(tmp_path):
    """A member's registered output port ``o0`` whose value is its own template's field tree, ``b/o0``, is held to the
    registration's root of that tree (`OneStageZk.portTreeHeld`): at the tree's root it passes this check, at another
    root it is refused by name, and the same value at another member's template, or at a wired input, is still refused
    by `wired_tree_why`."""
    w = Window(tmp_path)
    ms = w.params["proofs"]["members"]
    tree, record = RECORD["roots"][3], RECORD

    def why(who: str = B, port: str = "o0", root: str = tree["root"]) -> str | None:
        reg = {"format": "verity/registered-reads/v2",
               "ports": {port: {"value": "b/o0", "root": root, "domain": tree["domain"], "leaves": 3}}}
        return w.run(record=record, params=w.params | {"proofs": {"members": {**ms, who: ms[who] | {"registered": reg}}}})[1]
    assert "b/o0" not in (why() or "") and M.wired_tree_why(B, "o0") != why()
    assert why(root="44" * 64) == (f"registered: member {B}'s port o0 opens the committed tree b/o0 at another root "
                                   "than the registration's: a statement opens its window's registered tree, never "
                                   "another window's or another member's")
    assert why(who=A) == M.wired_tree_why(A, "o0")
    assert why(port="x") == M.wired_tree_why(B, "x")


def test_a_wired_row_opens_only_the_writers_field_tree_c_wires_into_it_at_the_records_root():
    """`OneStageZk.portTreeHeld` at C's wired rows (`wiredRowsOf`, `members._wired_rows`), as Lean's guards on
    `wiredAtT`: a reader's row ``x`` C wires from ``W``'s ``o0`` opens ``W/o0`` at the record's root; at another root it
    is refused by name, and another writer's tree, a field of ``W`` C doesn't wire, the other cut row and another
    member each by `wired_tree_why`."""
    tree = {"port": "W/o0", "root": "aa" * 64, "domain": "dd" * 64, "leaves": 4}
    record = {"roots": [tree, {**tree, "port": "W/o1"}, {**tree, "port": "V/o0"}]}

    def why(port: str = "x", value: str = "W/o0", root: str = tree["root"], who: str = "R/c0",
            wired: frozenset = frozenset({("x", "W/o0")})) -> str | None:
        reg = {"format": "verity/registered-reads/v2",
               "ports": {port: {"value": value, "root": root, "domain": tree["domain"], "leaves": 4}}}
        try:
            M._tree_ports_held({who: {"registered": reg}}, record, {"R/c0": set(wired)})
        except M._Refused as e:
            return str(e)
        return None
    assert why() is None
    assert why(root="44" * 64) == (
        "registered: member R/c0's port x opens the committed tree W/o0 at another root than the registration's: a "
        "statement opens its window's registered tree, never another window's or another member's")
    assert why(value="V/o0") == M.wired_tree_why("R/c0", "x")
    assert why(value="W/o1") == M.wired_tree_why("R/c0", "x")
    assert why(who="R/c1") == M.wired_tree_why("R/c1", "x")
    cut = frozenset({("x#1", "W/o0")})
    assert why(port="x#1", wired=cut) is None
    assert why(port="x#0", wired=cut) == M.wired_tree_why("R/c0", "x#0")


@lean
def test_a_fresh_input_opens_the_windows_one_root_of_its_node(tmp_path):
    """A registered port of run input node 1, its value ``input/n1``, opens the record's one root of the node, the
    window's and no member's (`OneStageZk.rootsOf`, `public_ports`): the fresh row changed after the coin (another
    root), a node the record holds no root of, and two members opening node 1 under two roots are each refused by name
    before any session, and a member named ``input`` is refused."""
    w = Window(tmp_path)
    ms = w.params["proofs"]["members"]
    tree = next(r for r in RECORD["roots"] if r["port"] == "input/n1")

    def why(a: dict | None = None, b: dict | None = None) -> str | None:
        members = {A: ms[A] | {"registered": a or reads(A)}, B: ms[B] | {"registered": b or reads(B)}}
        return w.run(params=w.params | {"proofs": {"members": members}})[1]
    assert why() is None
    assert set(DK.public_ports({"roots": [RECORD["roots"][0], tree]})) == {"out"}
    assert why(b=reads(B, root="55" * 64)) == (
        f"registered: member {B}'s port x opens the committed tree input/n1 at another root than the registration's: a "
        "statement opens its window's registered tree, never another window's or another member's")
    assert why(b=reads(B, value="input/n4")) == (
        f"registered: member {B}'s port x opens the committed tree input/n4, and the registration holds no root "
        "input/n4: a committed tree a statement opens is one its window registered before any coin")
    assert why(reads(B, root="55" * 64)).startswith(
        f"registered: members {A} and {B} read the registered value input/n1 as two (root")
    with pytest.raises(ValueError) as e:
        slices({"input": [[0, 2]], B: [[2, 5]]}, RECORD)
    assert str(e.value) == M.INPUT_MEMBER


def test_what_the_run_does_not_decide_is_outside_and_never_a_pass(tmp_path):
    """Parameters or evidence the per-member parsers refuse (among them ``proved_units`` that are no list of natural
    numbers, which the run compares with the members' units, a member handed units of its own, and a query other than
    the partition object's), a program other than the parameters', a registered
    file whose port is no registered port's, and parameters naming a wiring beside the members."""
    w = Window(tmp_path)
    ms = w.params["proofs"]["members"]
    for params, evidence in ((w.params | {"proofs": {"circuit": "", "public": "", "registered": None}}, None),
                             (None, w.evidence | {"proofs": {"sessions": []}}),
                             (None, {k: v for k, v in w.evidence.items() if k != "sessions"}),
                             (None, w.evidence | {"proved_units": [1, "2", 4]}),
                             (None, w.evidence | {"proved_units": [True, 2, 4]})):
        with pytest.raises(M.Outside, match="per-member form"):
            w.run(params=params, evidence=evidence)
    with pytest.raises(M.Outside, match=f"^params: proofs: member {A}: {M.UNITS_HANDED}$"):
        w.run(params=w.params | {"proofs": {"members": {A: ms[A] | {"units": [[2, 1]]}, B: ms[B]}}})
    with pytest.raises(M.Outside, match="partition object"):
        w.run(program=PN.population_program(PAIR, 5))
    with pytest.raises(M.Outside, match=f"^{M.PARTITION_QUERY}$"):
        w.run(params=w.params | {"query": w.params["query"] | {"params": {"template": "Other_v1"}}})
    with pytest.raises(M.Outside, match="registered port"):
        w.run(params=w.params | {"proofs": {"members": {A: ms[A] | {"registered": {"ports": {"p": {}}}}, B: ms[B]}}})
    with pytest.raises(M.Outside, match="per-member form"):
        w.run(params=w.params | {"proofs": {"members": ms, "wiring": {"templates": {}, "cuts": {}, "wires": []}}})


def test_a_return_of_no_fields_has_no_field_tree():
    """A return's fields count as Lean's `Circuits.Descriptor.fieldsOf` counts them: the empty tuple's none, a
    tuple's each, any other type one."""
    types = (Tuple(), Tuple(Value(8), Value(16)), Value(8), Array(2, Value(8)))
    assert [M.return_fields(t) for t in types] == [0, 2, 1, 1]


def _periods_interface_tests():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[6] / "tests/verity/commitments/periods/test_interface.py"
    spec = importlib.util.spec_from_file_location("periods_test_interface", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_members_run_input_ports_are_held_at_their_nodes_trees_derived_from_c(tmp_path):
    """`OneStageZk.runInputRoots`' mirror on a window whose C reads ``x``, ``sx``, ``wrow`` and ``sw`` from run input
    nodes (`Window` with ``run_inputs``): from C and the partition alone, each member's slice holds each such port at
    the record's ``input/n<w>`` root, with that tree, at C's leaf count and the domain under the window's open root
    (`run_tree_held`), which the writer commits under its run root; under another window's root the domain is refused
    by name; a root of a member's own at one refuses by name, and so does a record of no run input's tree. Inside a tied
    parameter's gates, two members reading the node under two roots are refused by name (`tied_roots_why`), under one
    root they are not."""
    TI = _periods_interface_tests()
    got = {}

    def record(*a, **_k):
        got["roots"] = a[4]
        return {"leaf_layer": "l"}

    reg = type("R", (), {"KINDS": ("served",), "record": staticmethod(record), "digest": staticmethod(lambda rec: "d")})
    win = TI.I.Window(TI.I.Circuit("test", ((TI.COORD, 10),)), TI.RUN, TI.SAMPLING_DESIGN, 4)
    ins, outs = TI.values(10)
    win.record(TI.COORD, ins, outs)
    win.write(win.close(), tmp_path, {"by": "test"}, registrar=reg)
    program, query = win.window.program, win.window.query
    members = {m.name: [[m.base, m.base + m.n]] for m in win.window.members}
    runs = {r.port: r.param for r in win.runs}
    root = TI.RUN
    from verity.circuits import codec, edges as EG
    Rt = EG._Root(json.loads(json.dumps(codec.encode_program(program))))
    ranges = P._template(program, query, "q").ranges
    lens = {w: n for name, us in members.items() for _p, w, n in M.member_inputs(Rt, ranges, name, us)}

    def bound(r: dict) -> dict:
        w = int(r["port"].removeprefix("input/n"))
        rows = Rt.nodes[w]["out_leaves"] * Rt.nodes[w]["members"] // lens[w]
        words = lens[w] * (EG._bits(Rt.nodes[w]["ret"]) // Rt.nodes[w]["out_leaves"]) // 16
        return {**r, "domain": RV.domain(P.program_digest(program), r["port"], words, rows, 16,
                                         window=root).domain_id.hex()}
    roots = got["roots"]
    assert roots == [bound(r) if r["port"].startswith("input/") else r for r in roots]
    rec = {"population": 10, "scheme": "frame-v3-sha512/hm96-sha512", "roots": roots}
    by_port = {r["port"]: r for r in roots}
    sl = M.slices(members, rec, program=program, query=query, window_root=root)
    for name in members:
        held = {r["port"]: r for r in sl[name]["roots"] if r.get("tree", "").startswith("input/")}
        assert set(held) == set(runs), (name, held)
        for p, t in runs.items():
            assert held[p] == {"domain": by_port[t]["domain"], "leaves": by_port[t]["leaves"], "port": p,
                               "root": by_port[t]["root"], "tree": t}
    first = next(iter(members))
    w = int(runs["x"].removeprefix("input/n"))
    with pytest.raises(ValueError) as e:
        M.slices(members, rec, program=program, query=query, window_root="ee" * 64)
    assert str(e.value).startswith("registered: the record's input/n") and str(e.value).endswith(
        "has another domain than C's run input node " + str(e.value).split("node ")[1].split(" ")[0] +
        " and the window's open root give it: a run input node's one tree binds its window")
    own = {"domain": "dd", "leaves": 4, "port": f"{first}/x", "root": "aa" * 64}
    with pytest.raises(ValueError) as e:
        M.slices(members, {**rec, "roots": [*roots, own]}, program=program, query=query, window_root=root)
    assert str(e.value) == M.run_own_root_why(first, "x", w)
    old = {**rec, "roots": [r for r in roots if not r["port"].startswith("input/")]}
    with pytest.raises(ValueError) as e:
        M.slices(members, old, program=program, query=query, window_root=root)
    assert str(e.value).startswith(f"member {first}: port ")
    offs = program.root.body.offsets
    tied = [range(offs[w], offs[w + 1])]
    two = list(members)[:2]

    def mine(*rs: str) -> list[dict]:
        return [{"domain": "dd", "leaves": 4, "port": f"{n}/x", "root": r} for n, r in zip(two, rs)]
    with pytest.raises(ValueError) as e:
        M.slices(members, {**rec, "roots": [*roots, *mine("aa" * 64, "bb" * 64)]}, program=program, query=query,
                 window_root=root, tied=tied)
    assert str(e.value) == M.tied_roots_why(w)
    sl = M.slices(members, {**rec, "roots": [*roots, *mine("aa" * 64, "aa" * 64)]}, program=program,
                  query=query, window_root=root, tied=tied)
    assert [next(r["root"] for r in sl[n]["roots"] if r["port"] == "x") for n in two] == ["aa" * 64] * 2


def test_the_registered_reads_refusals_are_leans_word_for_word():
    """The texts of `pinned_read_why`, `unpinned_read_why`, `two_reads_why`, `read_form_why` and the run input tree's
    refusals, which Lean's guards (`OneStageZk`, `Flock.OneCircuit.READ_FORM_WHY`) hold too."""
    assert M.tied_roots_why(3) == ("members: C's tied run input node 3 is read under two roots: every reader of a tied "
                                   "node opens one tree")
    assert M.pinned_read_why("a", "p", "w", "root") == (
        "registered: member a's port p reads the registered input w at another root than its pin of record: a "
        "registered input the weights' registration of record pins is read at the pin's root, leaf count and domain")
    assert M.unpinned_read_why("a", "p", "u") == (
        "registered: member a's port p reads the registered input u, which the program marks, R8 accepted no pin of "
        "and the verifier holds no root of: a marked input is read only at a pin R8 accepted or a root the verifier "
        "holds")
    assert M.range_node_why("a", 9) == (
        "member a: its units meet a range the partition gives at C's root node 9, which C does not have: a range is of "
        "C's root nodes")
    assert M.two_reads_why("a", "p", 1, 2) == (
        "registered reads: member a's units read parameter p at C's input nodes 1 and 2: a parameter that reads an "
        "input reads one node, each port one tree")
    assert M.read_form_why(2, "p", "x") == (
        "registered reads: parameter p of root node 2: x, and its argument reads an input node: a parameter that reads "
        "an input reads one node, each row in one lane's run")
    assert M.run_tree_missing_why("a", "p", 3) == (
        "member a: port p reads C's run input node 3, and the record holds no root of input/n3: a run input node's rows "
        "are committed once, in its one tree, before any coin")
    assert M.run_tree_why(3, "domain") == (
        "registered: the record's input/n3 has another domain than C's run input node 3 and the window's open root give "
        "it: a run input node's one tree binds its window")
    assert M.run_tree_unbound_why(3) == (
        "registered: C's run input node 3 is read from its one tree input/n3, and the verifier names no window open root "
        "(window_root) for its domain to bind")
    assert DK.WEIGHTS_QUERY_WHY == (
        "registered: the program marks registered inputs the verifier does not expect, and the partition is no template "
        "query that applies to the program, whose ranges its members' reads of them are read off: a marked input is read "
        "only at a pin R8 accepted")
    assert DK.WEIGHTS_UNREGISTERED_WHY == (
        "registered: the program marks registered inputs the verifier does not expect, and the parameters carry no "
        "weights' registration of record R8 accepts: a marked input is read only at a pin R8 accepted")
