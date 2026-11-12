"""`one_stage.members.members_run`, the Python of `OneStageZk.membersRun`, held to Lean's refusal of the per-member
form (`members.refusal`, the executable's ``refusal``) on a window of two members whose sessions' trees the challenger
session opens one after another, each closed after its session's evidence.

The files are stand-ins, so no session is accepted: an audit with nothing else wrong ends at its first session's
verdict on both sides, and each other row changes one thing and reads Lean's reason off the twin. A case every session
accepts, which Lean refuses for its form alone (`members.ALL_PASSED`), is a whole audit's, which
`benchmarks/one_stage` runs with the twin beside it.

The twin decides the sessions' trees by `challenger.members_tree_reason` (with `member_tree_reason`, `closes` and
`members_order`), and the rows hold it to Lean's `Challenger.membersTreeReason` on each of its refusals: a tree closed
after other evidence or never closed, a tree the log does not show, one tree for two sessions, a tree no session
records, and trees opened in another order or before the one before is closed.

Two rows are where the twin is not Lean's step for step (the module's docstring). One is here: a circuit file whose
META names another ``vus_per_block`` than the parameters' and that does not parse, which Lean leaves to its session's
reject and the twin refuses. The other is not run: a session whose link's ``root_B`` is not the length its statement
fixes, which Lean refuses by `challenger.LINK_ROOT_LEN` and the twin's session's own verdict rejects. It needs a
statement Lean sets up, on real files, under a release log this file's challenger signs, and
`test_challenger_difftest.py` holds the two sides of the trees' check with and without the length.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

import pytest

import verity_difftest as D
from verity.commitments import coin_tree as CT
from verity.protocols.spot.device import challenger as Ch
from verity.protocols.spot.verifier.one_stage import (closing as CL, draw as DR, members as M, partition as P,
                                                      proofs as PF, registration as R)

from test_challenger_difftest import Ledger, TreeSession, _pinned, lms_verify
from test_difftest import (LEAF, PAIR, RECEIVED, SCHEME, STAND_IN_CIRCUIT, TRIPLE, V2_TABLES, WINDOW, Setup,
                           _challenger_draw, _json)

#: slow: the first of them builds the executable, and each row runs it
pytestmark = [pytest.mark.slow, pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")]

A, B = f"{PAIR.id}/c0", f"{TRIPLE.id}/c0"
UNITS = {A: [[0, 2]], B: [[2, 5]]}
DESIGN = {"k": 2, "law": "subset"}


def _sha(b: bytes) -> str:
    return hashlib.sha512(b).hexdigest()


def _never(_):
    raise AssertionError("a session was verified before the refusal")


class Audit:
    """A window of the members `A` and `B` (two instances of `PAIR`, three of `TRIPLE`), its registration's two roots
    named `ports`, audited on a challenger session that releases the sample and then opens one coin tree a member, in
    the order `opened`, each tree's link, its rounds and its closing round after its session's evidence (its record and
    one proof file) before the next tree's first round: the per-member parameters and evidence, and the files. Only
    the members of `closed` have their tree's closing round opened, and with `together` those rounds are opened after
    every tree's other rounds."""

    def __init__(self, seed: str, ports: tuple[str, str] = (f"{A}/out", f"{B}/out"), opened: tuple[str, str] = (A, B),
                 closed: tuple[str, ...] = (A, B), together: bool = False):
        rng = random.Random(seed)
        s = self.s = Setup(P.mixed_population_program([(PAIR, 2), (TRIPLE, 3)]),
                           P.template_instances_query([PAIR.id, TRIPLE.id]), names=ports)
        roots = [{"port": p, "root": rng.randbytes(64).hex(), **s.ports[p]} for p in s.ports]
        self.rec = R.record(s.program, s.query, DESIGN, SCHEME, roots, WINDOW, leaf_layer=LEAF)
        ch, sample, _ = _challenger_draw(rng, s, DESIGN, self.rec, trees=0)
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
            sessions[name] = {"name": f"s{(A, B).index(name)}", "record": self.put(record),
                              "reps": {"circuit.rep0.bin": self.put(rep)}}
        for t, after in late:
            ch.s.lines += close(t, after)
        ch.s.close = ch.s.signed(None)
        ch.ledger = Ledger(rng, ch.s, withheld=True)
        ch.pins, ch.main = _pinned(rng, ch.ledger, "honest")
        proved = DR.proved(sample)
        statements = [{"member": name, "sessions": [sessions[name]]} for name in (A, B)]
        self.evidence = {"coins": {k: v for k, v in ch.evidence(ch.segment.to_json()).items() if k != "key"},
                         "draw": sample, "instances": len(proved), "proved_units": proved,
                         "receipt": R.receipt(self.rec, "v", received=RECEIVED), "served": None, "session_roots": None,
                         "statement": {}, "proofs": {"statements": statements}, "sessions": ["s0", "s1"]}
        members = {name: {"circuit": self.put(STAND_IN_CIRCUIT + name.encode()),
                          "public": self.put(json.dumps({"member": name}).encode() + b"\nnot a public file"),
                          "registered": None, "units": UNITS[name], "vus_per_block": 4} for name in (A, B)}
        self.params = {"anchors": [], "challenger": {"key": ch.s.key.to_json(), "label": "draw"}, "delta": "1/20",
                       "law": DESIGN, "leaf_layer": LEAF, "log": [], "partition": P.partition(s.program, s.query),
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
        why, decided = M.refusal(p, D.canonical(self.rec).encode(), D.canonical(e).encode(), paths)
        assert decided, why
        return why, M.members_run(p, _json(self.rec), e, M.blobs_of(paths), program=self.s.program, verify=lms_verify,
                                  verdict=verdict)

    def proofs(self, statements: list[dict]) -> dict:
        return self.evidence | {"proofs": {"statements": statements},
                                "sessions": [s["name"] for st in statements for s in st["sessions"]]}

    def members(self, name: str, **changes) -> dict:
        ms = self.params["proofs"]["members"]
        return self.params | {"proofs": {"members": ms | {name: ms[name] | changes}}}


def _rejected(run, refused) -> str:
    """What Lean answers where the twin's run holds a session that did not accept: v1's check, by its verdict's name."""
    assert refused is None and run is not None and not run.all and M.run_reason(run, refused) is None
    return f"verdict flock-verify/{next(f.session for f in run.findings if not f.accepted)} did not accept"


def test_the_twin_answers_leans_reason_on_each_refusal_of_the_per_member_run(tmp_path):
    """On an audit with nothing wrong but its stand-in files, Lean's check ends at the first session's verdict and the
    twin's run holds both sessions rejected by the verifier's own run, proving nothing. Each change below is refused by
    both with one reason, before any session is verified: the statements out of the members' order, a statement of a
    member the parameters don't name, a second statement of one member, a statement of two sessions, a record or a proof
    file no blob holds, a proof file changed after its session's closing round (`NO_CLOSING`), a tree no statement's
    session records (`UNUSED_TREE`), members that are not the population laid range by range (`MEMBERS_LAID`), two
    members reading one registered value at two roots, a circuit no blob holds, and blobs without the lookup table a
    member's circuit names."""
    a = Audit("members-twin")
    first, last = a.evidence["proofs"]["statements"]
    s0 = first["sessions"][0]

    why, got = a.ask(tmp_path / "honest", verdict=M.flock_verdict)
    assert why == _rejected(*got) == "verdict flock-verify/s0 did not accept"
    assert [(f.session, f.accepted, f.units) for f in got[0].findings] == [("s0", False, ()), ("s1", False, ())]
    assert got[0].proved == ()

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
    same("laid", M.MEMBERS_LAID, params=a.members(B, units=[[2, 4]]))
    port = {"value": "v", "leaves": 4, "domain": "dd" * 64, "reads": [0]}

    def registered(root: str) -> dict:
        return {"format": "verity/registered-reads/v1", "ports": {"p": {**port, "root": root * 64}}}
    two = a.members(A, registered=registered("aa"))
    two = two | {"proofs": {"members": two["proofs"]["members"] | {
        B: a.params["proofs"]["members"][B] | {"registered": registered("bb")}}}}
    same("class", f"registered: members {A} and {B} read the registered value v as two (root {'aa' * 64}, against "
                  f"{'bb' * 64}): a value the window registers is one tree", params=two)
    same("no-circuit", f"member {A}: circuit: no blob has the SHA-512", without=(a.params["proofs"]["members"][A]["circuit"],))
    looks = a.put(b"a circuit\nMETA " + json.dumps({"lookups": {"rcp": {}}}).encode() + b"\n")
    same("tables", f"member {A}: {PF.NO_TABLES}: the circuit looks up rcp, the blobs hold none",
         params=a.members(A, circuit=looks))


def test_the_twin_answers_leans_reason_on_the_logs_order_and_the_registrations_roots(tmp_path):
    """A release log that opens the second statement's tree first is refused by both as `MEMBERS_ORDER`, the statements
    being in the members' order; and a registration with a root that is no member's is refused by both, the trees'
    check passed."""
    late = Audit("members-twin-order", opened=(B, A))
    why, (run, refused) = late.ask(tmp_path / "order")
    assert run is None and why == M.run_reason(run, refused) == f"proofs: {Ch.MEMBERS_ORDER}", (why, refused)

    other = Audit("members-twin-roots", ports=(f"{A}/out", "zzz/out"))
    why, (run, refused) = other.ask(tmp_path / "roots")
    assert run is None and why == M.run_reason(run, refused), (why, refused)
    assert why == ("proofs: roots: the registration's root zzz/out is no member's: a member's roots are named "
                   "<member>/<port>")


def test_the_twin_answers_leans_reason_on_each_refusal_of_the_sessions_trees(tmp_path):
    """The trees' check the twin decides by (`challenger.members_tree_reason`, with `member_tree_reason`, `closes` and
    `members_order`) against Lean's (`Challenger.membersTreeReason`), on the refusals the other rows leave: a session
    whose tree is never closed (`NO_CLOSING`), a session whose record names a tree the log does not show (`NO_TREE`),
    two sessions on one tree (`SHARED_TREE`), and a log that opens the second tree before the first is closed
    (`MEMBERS_ORDER`). With the other rows' changed proof file, unused tree and trees opened in the other order, every
    refusal of that check is run on both sides."""
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


def test_the_one_row_run_here_where_the_twin_is_not_leans(tmp_path):
    """A circuit file whose META names 8 units a proof block, under parameters naming 4, that is no circuit file: Lean
    reads no ``vus_per_block`` off a file it cannot parse and its session rejects; the twin reads the META alone and
    refuses. Neither accepts."""
    a = Audit("members-twin-block")
    odd = a.put(b"not a circuit file\nMETA " + json.dumps({"vus_per_block": 8}).encode() + b"\n")
    why, (run, refused) = a.ask(tmp_path / "block", params=a.members(A, circuit=odd))
    assert why == "verdict flock-verify/s0 did not accept"
    assert run is None and refused == (f"member {A}: vus_per_block: the member's circuit has 8 units a proof block, "
                                       "not the 4 the parameters name")
