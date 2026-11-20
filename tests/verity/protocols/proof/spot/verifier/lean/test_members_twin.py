"""`one_stage.members.members_run`, the Python of `OneStageZk.membersRun`, held to Lean's check of the per-member
form (`members.check`, the executable's ``check``) on a window of two members whose sessions' trees the challenger
session opens one after another, each closed after its session's evidence.

The public files are stand-ins, so no member's files are held: an audit with nothing else wrong ends at its first
member's public file, which does not load, on both sides and before any session (its circuit is C_S, derived), and each other row changes one thing and reads
Lean's reason off the twin. An audit's sample holds a unit of each member (`Audit`'s `sampled`), so no statement is of
a member it missed; where it misses a member that has a statement, both refuse that statement before any member's file
is read (`OneStageZk.sampledHeld`). A case every session accepts, which Lean accepts, is a whole audit's, which
`benchmarks/one_stage` runs with the twin beside it.

The twin decides the sessions' trees by `challenger.members_tree_reason` (with `member_tree_reason`, `closes` and
`members_order`), and the rows hold it to Lean's `Challenger.membersTreeReason` on each refusal of its own: a tree
closed after other evidence or never closed, a tree the log does not show, one tree for two sessions, a tree no
session records, and trees opened in another order or before the one before is closed. What it refuses of a session's
tree as any session's (`session_tree_reason`) is `test_challenger_difftest.py`'s.

Three rows are where the twin is not Lean's step for step (the module's docstring); none is run here. A session whose
link's ``root_B`` is not the length its statement fixes, which Lean refuses by `challenger.LINK_ROOT_LEN` and the twin's session's own verdict rejects,
needs a statement Lean sets up, on real files, under a release log this file's challenger signs, and
`test_challenger_difftest.py` holds the two sides of the trees' check with and without the length. A member whose
circuit parses and is outside the scope's rules, or whose ports' roots are, which Lean refuses with its files and the
twin leaves to the member's session, needs real files too.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import pytest

import verity_difftest as D
from verity.circuits import partition_object as PO, population as PN
from verity.commitments import coin_tree as CT
from verity.protocols.proof.spot.verifier.challenger import challenger as Ch
from verity.protocols.proof.spot.verifier.one_stage import (
    closing as CL,
    draw as DR,
    members as M,
    partition as P,
    proofs as PF,
    docket as R,
)
from verity.protocols.proof.spot.prover import docket as DK

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_challenger_difftest import Ledger, TreeSession, _pinned, lms_verify
from test_header_held_at_difftest import HELD
from test_difftest import (LEAF, PAIR16 as PAIR, RECEIVED, SCHEME, TRIPLE16 as TRIPLE, V2_TABLES, WINDOW, Setup,
                           _challenger_sample, _json)

#: slow: the first of them builds the executable, and each row runs it
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]

A, B = f"{PAIR.id}/c0", f"{TRIPLE.id}/c0"
UNITS = {A: [[0, 2]], B: [[2, 5]]}
DESIGN = {"k": 2, "law": "subset"}
_TI = importlib.util.spec_from_file_location(
    "periods_test_interface", Path(__file__).resolve().parents[7] / "tests/verity/commitments/periods/test_interface.py")
TI = importlib.util.module_from_spec(_TI)
_TI.loader.exec_module(TI)
#: `TI.chain`'s window of four members, a block of two units each: units from 2 on read unit u - 2, one wire
WIRED = {f"{TI.GID}/c{k}": [[2 * k, 2 * k + 2]] for k in range(4)}


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _never(_):
    raise AssertionError("a session was verified before the refusal")


class Audit:
    """A window of the members `A` and `B` (two instances of `PAIR16`, three of `TRIPLE16`, the door's u16 form), its docket's two roots
    named `ports` beside its templates' field trees `<template>/o0`, audited on a challenger session that releases
    the sample and then opens one coin tree a member of `opened`, in that order (a member it leaves out has no session and no statement), each tree's link, its rounds and
    its closing round after its session's evidence (its record and one proof file) before the next tree's first round:
    the per-member parameters and evidence, and the files. Only
    the members of `closed` have their tree's closing round opened, and with `together` those rounds are opened after
    every tree's other rounds. The sample holds a unit of exactly the members of `sampled`: the challenger session is
    one whose coins give such a sample. With `setup`, `units` and `trees`, a window of that program's members, each
    its units, and its templates' field trees `<template>/o0`, each its leaves; `extra` are roots of members' own
    beside `ports`, and each port of `twice` is named by a second root, at another root."""

    def __init__(self, seed: str, ports: tuple[str, str] = (f"{A}/out", f"{B}/out"), opened: tuple[str, ...] = (A, B),
                 closed: tuple[str, ...] = (A, B), together: bool = False, sampled: tuple[str, ...] = (A, B),
                 setup: Setup | None = None, units: dict | None = None, trees: dict | None = None,
                 extra: tuple[str, ...] = (), twice: tuple[str, ...] = ()):
        rng = random.Random(seed)
        units = UNITS if units is None else units
        names = tuple(units)
        s = self.s = setup or Setup(PN.mixed_population_program([(PAIR, 2), (TRIPLE, 3)]),
                                    P.template_instances_query([PAIR.id, TRIPLE.id]), names=ports)
        trees = {PAIR.id: 2, TRIPLE.id: 3} if trees is None else trees
        s.ports |= {f"{t}/o0": {"domain": rng.randbytes(64).hex(), "leaves": n} for t, n in trees.items()}
        s.ports |= {p: s.ports[ports[1]] for p in extra}
        roots = [{"port": p, "root": rng.randbytes(64).hex(), **s.ports[p]} for p in s.ports]
        roots += [{**r, "root": rng.randbytes(64).hex()} for r in roots if r["port"] in twice]
        self.partition = PO.build(s.program, s.query, [[units[n][0] for n in names]])
        self.rec = DK.record(s.program, s.query, DESIGN, SCHEME, roots, WINDOW, leaf_layer=LEAF,
                             partition=self.partition)
        held = None
        while held != set(sampled):
            ch, sample, _ = _challenger_sample(rng, s, DESIGN, self.rec, trees=0)
            held = {n for n in names if any(lo <= u < hi for u in DR.proved(sample) for lo, hi in units[n])}
        self.files: dict[str, bytes] = {}
        sessions = {}

        def close(t: TreeSession, after: bytes) -> list[str]:
            t.open(CT.SESSION, 0, after)
            return t.lines[-1:]

        late = []
        for name in opened:
            t = TreeSession(rng, ch.s.ch, V2_TABLES)
            while t.spec["rounds_log"] < 1:
                t = TreeSession(rng, ch.s.ch, V2_TABLES)
            record, rep = D.canonical(t.record).encode(), f"proof of {name}".encode()
            after = hashlib.sha512(CL.session_evidence(record, [("circuit.rep0.bin", rep)])).digest()
            ch.s.lines += t.lines
            if name in closed and together:
                late.append((t, after))
            elif name in closed:
                ch.s.lines += close(t, after)
            sessions[name] = {"name": f"s{names.index(name)}", "record": self.put(record),
                              "reps": {"circuit.rep0.bin": self.put(rep)}}
        for t, after in late:
            ch.s.lines += close(t, after)
        ch.s.close = ch.s.signed(None)
        ch.ledger = Ledger(rng, ch.s, withheld=True)
        ch.pins, ch.main = _pinned(rng, ch.ledger, "honest")
        proved = DR.proved(sample)
        statements = [{"member": name, "sessions": [sessions[name]]} for name in names if name in sessions]
        self.evidence = {"coins": {k: v for k, v in ch.evidence(ch.segment.to_json()).items() if k != "key"},
                         "draw": sample, "instances": len(proved), "proved_units": proved,
                         "receipt": R.receipt(self.rec, "v", received=RECEIVED), "served": sorted(sample["units"]),
                         "session_roots": {r["port"]: r["root"] for r in self.rec["roots"]},
                         "statement": {}, "proofs": {"statements": statements},
                         "sessions": [s["name"] for st in statements for s in st["sessions"]]}
        members = {name: {"public": self.put(json.dumps({"member": name}).encode() + b"\nnot a public file"),
                          "registered": None, "block": k, "vus_per_block": 1} for k, name in enumerate(names)}
        self.params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                       "law": DESIGN, "leaf_layer": LEAF, "log": [], "partition": self.partition,
                       "ports": s.ports, "program": s.desc, "query": s.query, "registered": None, "scheme": SCHEME,
                       "window_root": ch.ledger.window["root"], "work_table": None, "proofs": {"members": members}}

    def put(self, b: bytes) -> str:
        self.files[_sha(b)] = b
        return _sha(b)

    def ask(self, at: Path, params: dict | None = None, evidence: dict | None = None, without: tuple[str, ...] = (),
            verdict=_never) -> tuple[str, tuple]:
        """Lean's reason for the case of `params` and `evidence` (the audit's own when None) on the audit's files
        without those of the digests `without`, written under `at`; and the twin's answer on the same."""
        at.mkdir(parents=True, exist_ok=True)
        paths = []
        for d, b in self.files.items():
            if d not in without:
                (at / d[:16]).write_bytes(b)
                paths.append(at / d[:16])
        p, e = _json(self.params if params is None else params), _json(self.evidence if evidence is None else evidence)
        verdict_, why = M.check(p, D.canonical(self.rec).encode(), D.canonical(e).encode(), paths)
        assert verdict_ == "reject", (verdict_, why)
        return why, M.members_run(p, _json(self.rec), e, M.blobs_of(paths), program=self.s.program, verify=lms_verify,
                                  verdict=verdict)

    def proofs(self, statements: list[dict]) -> dict:
        return self.evidence | {"proofs": {"statements": statements},
                                "sessions": [s["name"] for st in statements for s in st["sessions"]]}

    def members(self, name: str, **changes) -> dict:
        ms = self.params["proofs"]["members"]
        return self.params | {"proofs": {"members": ms | {name: ms[name] | changes}}}

    def without(self, name: str) -> dict:
        return self.params | {"proofs": {"members": {k: m for k, m in self.params["proofs"]["members"].items()
                                                     if k != name}}}

    def refused(self, at: Path, params: dict) -> str:
        """Lean's refusal of the parameters `params` at its parse, which the twin raises as `Outside` with the same
        reason."""
        at.mkdir(parents=True, exist_ok=True)
        p = _json(params)
        verdict, why = M.check(p, D.canonical(self.rec).encode(), D.canonical(self.evidence).encode())
        assert verdict == "reject" and why.startswith("params: "), (verdict, why)
        with pytest.raises(M.Outside) as e:
            M.members_run(p, _json(self.rec), _json(self.evidence), {}, program=self.s.program, verify=lms_verify,
                          verdict=_never)
        assert str(e.value) == why, (str(e.value), why)
        return why


def test_the_twin_answers_leans_reason_on_each_refusal_of_the_per_member_run(tmp_path):
    """On an audit with nothing wrong but its stand-in files, Lean's check ends at the first member's files, its
    public file one that does not load, and the twin answers that reason, the Flock verifier's own (`members.flock_form`);
    neither verifies a session. Each change below is refused by both with one reason, before any session is verified:
    the statements out of the members' order, a statement of a
    member the parameters don't name, a second statement of one member, a statement of two sessions, a record or a proof
    file no blob holds, a proof file changed after its session's closing round (`NO_CLOSING`), a tree no statement's
    session records (`UNUSED_TREE`), members that are not the population laid range by range (`MEMBERS_LAID`), two
    members reading one registered value at two roots, a registered port stating positions. A member's circuit is C_S (`one_circuit.restrict`), read
    last."""
    a = Audit("members-twin")
    first, last = a.evidence["proofs"]["statements"]
    s0 = first["sessions"][0]

    def same(k: str, starts: str, **ask) -> None:
        why, (run, refused) = a.ask(tmp_path / k, **ask)
        assert run is None and why == M.run_reason(run, refused) and why.startswith(f"proofs: {starts}"), (k, why, refused)

    same("order", f"member {A}: its statement follows a member's whose units start after its own",
         evidence=a.proofs([last, first]))
    same("unnamed", "member x/c0: the parameters name no such member",
         evidence=a.proofs([{**first, "member": "x/c0"}, last]))
    same("twice", f"member {A}: a second statement: a member's units are one statement's",
         evidence=a.proofs([first, {**last, "member": A}]))
    same("retried", f"member {A}: its statement has 2 sessions",
         evidence=a.proofs([{**first, "sessions": [s0, {**s0, "name": "s2"}]}, last]))
    same("no-record", "session s0's record: no blob has the SHA-512", without=(s0["record"],))
    same("no-rep", "session s0's files: circuit.rep0.bin: no blob has the SHA-512",
         without=(s0["reps"]["circuit.rep0.bin"],))
    changed = a.put(b"another proof")
    same("changed", f"session s0's coins are not the challenger's: {Ch.NO_CLOSING}",
         evidence=a.proofs([{**first, "sessions": [{**s0, "reps": {"circuit.rep0.bin": changed}}]}, last]))
    same("unused", f"the challenger's release log shows 2 coin trees and the sessions use 1: {Ch.UNUSED_TREE}",
         evidence=a.proofs([first]))
    laid = Audit("members-twin-laid", opened=(A,))
    why, (run, refused) = laid.ask(tmp_path / "laid", params=laid.without(B))
    assert run is None and why == M.run_reason(run, refused) == f"proofs: {M.MEMBERS_LAID}", (why, refused)
    port = {"value": "v", "leaves": 4, "domain": "dd" * 64}

    def registered(root: str, form: str = "verity/registered-reads/v2") -> dict:
        return {"format": form, "ports": {"p": {**port, "root": root * 64}}}
    same("v1", f"member {A}: registered reads: not a verity/registered-reads/v2 file",
         params=a.members(A, registered=registered("aa", "verity/registered-reads/v1")))
    placed = registered("aa")
    placed["ports"]["p"]["reads"] = [0]
    same("positions", f"member {A}: registered reads: port p: a verity/registered-reads/v2 port holds keys of [value, "
                      "root, leaves, words, word_bits, domain, row_schema, known, param] and no other (no position: C "
                      "and the partition give them)", params=a.members(A, registered=placed))
    two = a.members(A, registered=registered("aa"))
    two = two | {"proofs": {"members": two["proofs"]["members"] | {
        B: a.params["proofs"]["members"][B] | {"registered": registered("bb")}}}}
    same("class", f"registered: members {A} and {B} read the registered value v as two (root {'aa' * 64}, against "
                  f"{'bb' * 64}): a value the window registers is one tree", params=two)
    why, (run, refused) = a.ask(tmp_path / "honest")
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why.startswith(f"proofs: member {A}: {M.UNREAD[1]}"), why


def test_a_member_with_no_statement_has_its_files_held_by_both_before_any_session(tmp_path):
    """A window audited with a statement of its second member alone, that statement's tree logged and closed: Lean
    holds every member's files, sampled or not (`OneStageZk.membersFiles`), so it refuses the first member, which has
    no statement, for its circuit, and the twin answers the same before any session."""
    a = Audit("members-twin-no-statement", opened=(B,))
    assert [st["member"] for st in a.evidence["proofs"]["statements"]] == [B] and a.evidence["sessions"] == ["s1"]
    why, (run, refused) = a.ask(tmp_path / "no-statement")
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why.startswith(f"proofs: member {A}: {M.UNREAD[1]}"), why


def test_a_member_no_statement_names_is_held_to_its_own_circuits_lookup_tables(tmp_path):
    """A window audited with a statement of its second member alone: the first member, which no statement names, is
    held to its own circuit's lookup tables among the blobs as a sampled one is (`OneStageZk.ownTables`, the first read
    of every member's files), before any session. With its circuit looking up ``rcp`` where the blobs hold no table,
    both refuse it for the tables. The other way, blobs holding a file under ``rcp``'s pinned digest where its circuit
    looks up none: the twin does not refuse it for the table, and answers as it does without it, at the member's
    public file, a stand-in that does not load. Lean's check finds a blob by the SHA-512 of its bytes, so its side of that
    row needs the table's own bytes, which this suite does not load: both sides on the table itself are
    `benchmarks/one_stage`'s (``tests/test_members_tables.py``)."""
    a = Audit("members-twin-no-statement-tables", opened=(B,))
    assert [st["member"] for st in a.evidence["proofs"]["statements"]] == [B]
    _, plain = a.ask(tmp_path / "holds")
    blobs = M.blobs_of(sorted((tmp_path / "holds").iterdir()))
    held = M.members_run(_json(a.params), _json(a.rec), _json(a.evidence),
                         blobs | {PF.MUFU_SHA512["rcp"]: next(iter(blobs.values()))}, program=a.s.program,
                         verify=lms_verify, verdict=_never)
    assert held == plain and held[0] is None and held[1].startswith(f"member {A}: {M.UNREAD[1]}"), (held, plain)


def test_the_twin_answers_leans_reason_on_the_logs_order_and_the_registrations_roots(tmp_path):
    """A release log that opens the second statement's tree first is refused by both as `MEMBERS_ORDER`, the statements
    being in the members' order; and a docket with a root that is no member's is refused by both, the trees'
    check passed."""
    late = Audit("members-twin-order", opened=(B, A))
    why, (run, refused) = late.ask(tmp_path / "order")
    assert run is None and why == M.run_reason(run, refused) == f"proofs: {Ch.MEMBERS_ORDER}", (why, refused)

    other = Audit("members-twin-roots", ports=(f"{A}/out", "zzz/out"))
    why, (run, refused) = other.ask(tmp_path / "roots")
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why == ("proofs: roots: the registration's root zzz/out is no member's: a member's roots are named "
                   "<member>/<port>, and a window's tree of a template's field <template>/<field>")


def test_a_members_slice_that_names_a_port_twice_is_refused_by_both(tmp_path):
    """A docket naming the first member's port ``out`` twice, at its true root and at another: both refuse its slice by
    name (`PORT_TWICE_WHY`), after the roots and before any member's file; with ``out`` named once, both end at the
    first member's stand-in public file."""
    why, (run, refused) = Audit("members-twin-port-twice", twice=(f"{A}/out",)).ask(tmp_path / "twice")
    assert run is None and why == M.run_reason(run, refused) == f"proofs: {M.port_twice_why(A, 'out')}", (why, refused)
    why, (run, refused) = Audit("members-twin-port-twice").ask(tmp_path / "once")
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why.startswith(f"proofs: member {A}: {M.UNREAD[1]}"), why


def test_the_twin_answers_leans_reason_on_each_refusal_of_the_sessions_trees(tmp_path):
    """The trees' check the twin decides by (`challenger.members_tree_reason`, with `member_tree_reason`, `closes` and
    `members_order`) against Lean's (`Challenger.membersTreeReason`), on the refusals the other rows leave: a session
    whose tree is never closed (`NO_CLOSING`), a session whose record names a tree the log does not show (`NO_TREE`),
    two sessions on one tree (`SHARED_TREE`), and a log that opens the second tree before the first is closed
    (`MEMBERS_ORDER`). With the other rows' changed proof file, unused tree and trees opened in the other order, every
    refusal of its own is run on both sides."""
    def same(x: Audit, k: str, reason: str, **ask) -> None:
        why, (run, refused) = x.ask(tmp_path / k, **ask)
        assert run is None and why == M.run_reason(run, refused) == f"proofs: {reason}", (k, why, refused)

    same(Audit("members-twin-open", closed=(A,)), "open", f"session s1's coins are not the challenger's: {Ch.NO_CLOSING}")
    same(Audit("members-twin-together", together=True), "together", Ch.MEMBERS_ORDER)
    a = Audit("members-twin-trees")
    first, last = a.evidence["proofs"]["statements"]
    s0, s1 = first["sessions"][0], last["sessions"][0]
    record = json.loads(a.files[s1["record"]])
    elsewhere = a.put(D.canonical(record | {"coin_tree": record["coin_tree"] | {"root": "00" * 64}}).encode())
    same(a, "no-tree", f"session s1's coins are not the challenger's: {Ch.NO_TREE}",
         evidence=a.proofs([first, {**last, "sessions": [{**s1, "record": elsewhere}]}]))
    same(a, "shared", f"two sessions' coins are one tree's: {Ch.SHARED_TREE}",
         evidence=a.proofs([first, {**last, "sessions": [{**s1, "record": s0["record"], "reps": s0["reps"]}]}]))


def test_a_statement_of_a_member_the_sample_missed_is_refused_by_both_before_any_session(tmp_path):
    """A window whose sample holds units of one member alone, audited with a statement of each member, each statement's
    tree logged and closed: Lean refuses the other member's statement by its member's name (`OneStageZk.sampledHeld`),
    after the trees, the members laid and the roots and before any member's file, and the twin answers the same,
    whichever member the sample missed."""
    for k, held, missed in (("second", (A,), B), ("first", (B,), A)):
        a = Audit(f"members-twin-missed-{k}", sampled=held)
        why, (run, refused) = a.ask(tmp_path / k)
        assert run is None and why == M.run_reason(run, refused), (k, why, refused)
        assert why == (f"proofs: member {missed}: the sample holds none of its units, and a member the sample missed "
                       "has no statement")


def test_leans_parse_and_the_twin_take_a_members_units_from_its_block_and_refuse_the_same_parameters(tmp_path):
    """A member names its block of the partition object's ``blocks`` (`one_circuit.block`, `OneStageZk.membersOf`):
    Lean's parse (`proofs.members_parsed`, the executable's ``members``) reads each member's units as the twin's
    `members._block` does. Lean refuses at its parse, and the twin raises `Outside` with the same ``params:`` reason:
    a member handed ``units`` (`UNITS_HANDED`) beside its block, a block the partition lacks, a partition naming no
    blocks, and two members naming one block, whose units are then one unit in two members'."""
    a = Audit("members-twin-blocks")
    got, refused = PF.members_parsed(a.params, a.evidence)
    assert refused is None, refused
    descriptor = json.dumps(a.params["program"], sort_keys=True, separators=(",", ":")).encode()
    assert {m["name"]: m["units"] for m in got["members"]} == {
        name: [list(u) for u in M._block(name, m, descriptor, a.params["partition"])]
        for name, m in a.params["proofs"]["members"].items()} == UNITS
    assert a.refused(tmp_path / "handed", a.members(A, units=UNITS[A])) == (
        f"params: proofs: member {A}: {M.UNITS_HANDED}")
    assert a.refused(tmp_path / "range", a.members(B, block=2)) == (
        f"params: proofs: member {B}: partition: E_BLOCKS: no block 2 among the partition's 2")
    none = a.params | {"partition": {k: v for k, v in a.params["partition"].items() if k != "blocks"}}
    assert a.refused(tmp_path / "none", none) == (
        f"params: proofs: member {min(A, B)}: partition: E_BLOCKS: the partition object names no blocks")
    assert a.refused(tmp_path / "one-block", a.members(B, block=0)) == (
        "params: proofs: unit 0 is in two members' units: a unit is one member's")


def test_a_members_own_root_at_a_port_c_wires_into_it_is_refused_by_both(tmp_path):
    """`WIRED`'s window, its second member's port ``x`` read off its writer's output field's tree: with each wired port
    at that tree (`OneStageZk.fieldRoots`, `members._wired_roots`), both end at the first member's stand-in public file;
    a root of the second member's own at ``x`` is refused by both by name (`OWN_ROOT_WHY`) before any member's file."""
    c0, c1 = list(WIRED)[:2]
    program, query = TI.chain().program()

    def audit(seed: str, **kw) -> Audit:
        return Audit(seed, ports=(f"{c0}/out", f"{c1}/out"), opened=(c0,), closed=(c0,), sampled=(c0, c1),
                     setup=Setup(program, query, names=(f"{c0}/out", f"{c1}/out")), units=WIRED, trees={TI.GID: 8}, **kw)
    why, (run, refused) = audit("members-twin-wired").ask(tmp_path / "honest")
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why.startswith(f"proofs: member {c0}: "), why
    why, (run, refused) = audit("members-twin-wired-own", extra=(f"{c1}/x",)).ask(tmp_path / "own")
    assert run is None and why == M.run_reason(run, refused) == f"proofs: {M.own_root_why(c1, 'x')}", (why, refused)


#: `HELD`'s piece, `OneStageZk.headerHeldAt` on a slice and a header
pieces = D.server_fixture(HELD)


def test_a_wired_port_whose_writers_tree_the_record_names_none_of_is_refused_at_its_readers_header(tmp_path, pieces):
    """`WIRED`'s window with its members named under a template ``g`` that is not C's Definition id: the record's
    tree of the field is ``g/o0``, so `fieldRoots` (`members._wired_roots`) finds no tree of the writer's field and
    gives the second member's wired port ``x`` no root, which neither run refuses there (both end at the first member's
    stand-in public file). The reader's public file must name ``x`` (`Flock.HmRow.checkPublic` reads a root of every
    port of the block circuit), and its header held to that slice is refused by both by name (`UNWIRED_WHY`:
    `OneStageZk.headerHeld`'s `unwired`, `members.population_why`)."""
    units = {f"g/c{k}": u for k, u in enumerate(WIRED.values())}
    g0, g1 = list(units)[:2]
    program, query = TI.chain().program()
    a = Audit("members-twin-wired-unrooted", ports=(f"{g0}/out", f"{g1}/out"), opened=(g0,), closed=(g0,),
              sampled=(g0, g1), setup=Setup(program, query, names=(f"{g0}/out", f"{g1}/out")), units=units,
              trees={"g": 8})
    why, (run, refused) = a.ask(tmp_path / "unrooted")
    assert run is None and why == M.run_reason(run, refused) and why.startswith(f"proofs: member {g0}: "), (why, refused)
    got = M.slices(units, a.rec, program=program, query=query)[g1]
    assert sorted(r["port"] for r in got["roots"]) == ["o0", "out"], got
    roots = {r["port"]: r["root"] for r in got["roots"]} | {"x": "aa" * 64}
    domains = {r["port"]: r["domain"] for r in got["roots"]} | {"x": "dd" * 64}
    h = {"frame_v3": {"domain_ids": domains, "roots": roots}, "instances": 2, "scheme": SCHEME}
    assert M.population_why(units[g1], P.strata(program, query), h | {"units": {"indices": [2, 3]}}, got) == (
        M.unwired_why("x"))
    j = {"reads": [], "registered": None, "slice": got, "header": h}
    assert pieces.call("header_held_at", j) == ("output", D.canonical(M.unwired_why("x")))
