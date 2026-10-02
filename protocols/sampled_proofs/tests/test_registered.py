"""Registered values: a program's table committed once under hiding leaves (`one_stage.registered`), the registration's
R7, the verifier's read map from its own program, the read check, and the audit record's link to the registered roots."""
from __future__ import annotations

import hashlib

import pytest

import verity.ml.scalar  # noqa: F401
from verity.commitments import hm96
from verity.commitments.merkle import merkle_depth
from verity.commitments.rowleaf import ROLE_X, sha512_row_digest
from verity.ir.codec import canonical_json
from verity.ir.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ir.program import Program
from verity.ir.refs import coll_concat
from verity.ir.types import Array, Value
from verity_sampled_proofs.one_stage import audit, draw as D, partition as P, registered as G, registration as R

XOR8 = PrimitiveDefinition("RegisteredTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
MIX = bind(CompositeDefinition("RegisteredTestMix", 0, (), lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8)))), Value(8)),
                               lambda B, S, x, w: B.call(XOR8, B.call(XOR8, x[0], w[0]), B.call(XOR8, x[1], w[1])), register=False))
#: the row of the table W each unit reads, fixed by the program
POS = (3, 3, 0, 2, 1, 3)
TABLE = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 2}
SERVED = {"kind": "served", "row": 7, "run": "r-registered"}


def _program(pos=POS, rows: int = len(TABLE), body=None) -> Program:
    def sig(S):
        return (("x", Array(len(pos), Array(2, Value(8)))), ("W", Array(rows, Array(2, Value(8))))), Array(len(pos), Value(8))

    def default(B, S, x, W):
        return coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p]) for i, p in enumerate(pos)])

    return Program(bind(CompositeDefinition("RegisteredTestProgram", 0, (), sig, body or default, register=False)))


Q = P.template_instance_query(MIX.id)


def _registered(prog: Program) -> G.Registered:
    return G.Registered.commit(P.program_sha512(prog), "W", TABLE)


def _record(prog: Program, reg: G.Registered, **kw):
    part = P.partition(prog, Q)
    n = P.units(prog, Q)
    ports = {p: {"leaves": leaves, "domain": R.served_domain(part["program"], P.digest(part), p, schema, leaves, "r-registered")}
             for p, schema, leaves in (("x", "hm96-sha512/row/v1", n), ("out", "u16", n))}
    roots = [{"port": p, "root": "ab" * 64, **ports[p]} for p in ports]
    return R.record(prog, Q, LAW, SCHEME, roots, SERVED, registered=[reg.entry()], **kw), ports


def test_the_registered_root_is_the_hm96_rows_tree_under_its_own_domain():
    prog = _program()
    reg = _registered(prog)
    dom = G.domain(P.program_sha512(prog), "W", 2, len(TABLE))
    inst = hm96.SHA512
    for r, row in enumerate(TABLE):
        assert reg.bcs[r] == inst.commit_string(inst.default_key, sha512_row_digest(row, 16, ROLE_X), reg.salts[r])
        assert G.verify_read(dom, reg.root, r, reg.bcs[r], reg.path(r))
        assert len(reg.path(r)) == 64 * merkle_depth(len(TABLE))
    assert reg.entry() == {"value": "W", "root": reg.root.hex(), "leaves": 5, "words": 2, "domain": dom.domain_id.hex()}
    # fresh salts: the same table registered again is another root, whose rows are unlinkable to these
    again = _registered(prog)
    assert again.root != reg.root and set(again.bcs).isdisjoint(reg.bcs)


def test_a_read_opens_only_its_own_row_under_its_own_value():
    prog = _program()
    reg = _registered(prog)
    dom = reg.domain
    assert not G.verify_read(dom, reg.root, 2, reg.bcs[3], reg.path(3))
    assert not G.verify_read(dom, reg.root, 3, reg.bcs[2], reg.path(3))
    other = G.domain(P.program_sha512(prog), "V", 2, len(TABLE))
    assert not G.verify_read(other, reg.root, 3, reg.bcs[3], reg.path(3))
    flipped = bytes([reg.path(3)[0] ^ 1]) + reg.path(3)[1:]
    assert not G.verify_read(dom, reg.root, 3, reg.bcs[3], flipped)
    assert not G.verify_read(dom, reg.root, 3, reg.bcs[3], reg.path(3)[:-64])
    # the row with a fresh salt is another b || c: it opens nothing
    inst = hm96.SHA512
    fresh = inst.commit_string(inst.default_key, sha512_row_digest(TABLE[3], 16, ROLE_X), inst.fresh_salt())
    assert not G.verify_read(dom, reg.root, 3, fresh, reg.path(3))


def test_salts_are_one_per_row_and_never_reused():
    prog = _program()
    salt = hm96.SHA512.fresh_salt()
    with pytest.raises(ValueError, match="never reused"):
        G.Registered.commit(P.program_sha512(prog), "W", TABLE[:2], [salt, salt])
    with pytest.raises(ValueError):
        G.Registered.commit(P.program_sha512(prog), "W", TABLE[:2], [salt])
    with pytest.raises(ValueError):
        G.Registered.commit(P.program_sha512(prog), "W", [[1, 2], [3]])


def test_a_statement_table_carries_the_registered_salts_and_paths():
    prog = _program()
    reg = _registered(prog)
    positions, refs = G.table(POS)
    assert positions == [3, 0, 2, 1] and refs == [0, 0, 1, 2, 3, 0]
    rows, salts, entry = reg.statement(positions)
    assert rows == [tuple(TABLE[p]) for p in positions] and salts == [reg.salts[p] for p in positions]
    assert entry == {"value": "W", "positions": positions, "paths": [reg.path(p).hex() for p in positions]}
    with pytest.raises(ValueError):
        reg.statement([5])


def test_reads_come_from_the_verifiers_own_program():
    prog = _program()
    assert G.reads(prog, Q, {"W": 2}) == {"w": {"value": "W", "rows": list(POS)}}
    assert G.reads(prog, Q, {"W": 2, "x": 2}) == {"x": {"value": "x", "rows": list(range(6))}, "w": {"value": "W", "rows": list(POS)}}
    assert G.reads(prog, Q, {}) == {}
    assert G.shape(prog, "W", 2) == 5
    for bad in ({"W": 1}, {"W": 3}, {"V": 2}):
        with pytest.raises(ValueError):
            G.reads(prog, Q, bad)


def test_a_read_that_is_not_one_whole_row_is_refused():
    """A unit reading across two rows, or a parameter that reads the table in one unit and not in another, has no public
    position."""
    def straddle(B, S, x, W):
        flat = [W[r][j] for r in range(len(TABLE)) for j in range(2)]
        return coll_concat(Array(6, Value(8)), [B.call(MIX, x[i], coll_concat(Array(2, Value(8)), flat[2 * p + 1:2 * p + 3]))
                                                 for i, p in enumerate((0, 1, 1, 2, 0, 3))])

    with pytest.raises(ValueError, match="one whole row"):
        G.reads(_program(body=straddle), Q, {"W": 2})

    def mixed(B, S, x, W):
        return coll_concat(Array(6, Value(8)), [B.call(MIX, x[i], W[POS[i]] if i % 2 else x[i]) for i in range(6)])

    with pytest.raises(ValueError, match="in another unit"):
        G.reads(_program(body=mixed), Q, {"W": 2})


def test_the_verifiers_reads_file():
    """Everything in the file but the root is the verifier's own, from its program: a registration that named another leaf
    count or domain would change nothing here."""
    prog = _program()
    reg = _registered(prog)
    doc = G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()})
    assert doc == {"format": "verity/registered-reads/v0",
                   "ports": {"w": {"value": "W", "root": reg.root.hex(), "leaves": 5, "domain": reg.domain.domain_id.hex(),
                                   "reads": list(POS)}}}
    assert {k: doc["ports"]["w"][k] for k in ("leaves", "domain")} == {k: G.registered_entries(prog, {"W": 2})["W"][k]
                                                                       for k in ("leaves", "domain")}
    assert G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()}, ports={"w": "cs"})["ports"].keys() == {"cs"}
    with pytest.raises(ValueError, match="does not hold"):
        G.reads_file(prog, Q, {"W": 2}, {})
    with pytest.raises(ValueError):
        G.reads_file(prog, Q, {"W": 3}, {"W": reg.root.hex()})


def test_registration_v2_names_the_registered_values_and_r7_holds_them():
    prog = _program()
    reg = _registered(prog)
    rec, ports = _record(prog, reg)
    assert rec["format"] == "verity/registration/v2" and rec["registered"] == [reg.entry()]
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports}
    assert R.check(rec, prog, Q, **kw, registered={"W": 2}).ok
    assert R.digest(rec) != R.digest({**rec, "format": "verity/registration/v1"})
    e = reg.entry()
    cases = [({"registered": [{**e, "leaves": 4}]}, "R7-leaves"), ({"registered": [{**e, "words": 1}]}, "R7-words"),
             ({"registered": [{**e, "domain": "00" * 64}]}, "R7-domain"), ({"registered": [{**e, "root": "xy"}]}, "R7-root"),
             ({"registered": []}, "R7-values"), ({"registered": [e, e]}, "R7-values"), ({"registered": [{**e, "value": "x"}]}, "R7-values"),
             ({"registered": [{**e, "extra": 1}]}, "R7-registered"), ({"format": "verity/registration/v1"}, "R1-protocol")]
    for change, code in cases:
        assert code in R.check({**rec, **change}, prog, Q, **kw, registered={"W": 2}).codes, code
    assert "R7-registered" in R.check(rec, prog, Q, **kw, registered={"W": 3}).codes
    # a v2 record where the verifier registers nothing, and a v1 record where it registers a value, are refused
    assert R.check(rec, prog, Q, **kw).codes == ("keys",)
    v1 = {k: v for k, v in rec.items() if k != "registered"} | {"format": "verity/registration/v1"}
    assert R.check(v1, prog, Q, **kw).ok and R.check(v1, prog, Q, **kw, registered={"W": 2}).codes == ("keys",)


def test_a_value_is_committed_once_and_the_log_holds_its_root():
    """R7 holds a later record's root to the one an earlier record of the log registered: the verifier holds no roots
    beside its log, so no caller can leave them out."""
    prog = _program()
    reg = _registered(prog)
    rec, ports = _record(prog, reg)
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"W": 2}}

    def later(window, previous, entry):
        return R.record(prog, Q, LAW, SCHEME, rec["roots"], {**SERVED, "row": window}, previous=R.digest(previous),
                        registered=[entry] if entry else ())

    assert R.check(later(8, rec, reg.entry()), prog, Q, **kw, log=[rec]).ok
    assert R.check(later(8, rec, _registered(prog).entry()), prog, Q, **kw, log=[rec]).codes == ("R7-held",)
    v1 = later(8, rec, None)
    assert R.held_roots([rec, v1]) == {"W": reg.root.hex()} and R.held_roots([v1]) == {}
    assert R.check(later(9, v1, _registered(prog).entry()), prog, Q, **kw, log=[rec, v1]).codes == ("R7-held",)


def test_a_record_without_registered_values_keeps_v1s_bytes():
    prog = _program()
    rec, _ = _record(prog, _registered(prog))
    v1 = R.record(prog, Q, LAW, SCHEME, rec["roots"], SERVED)
    assert v1["format"] == "verity/registration/v1" and "registered" not in v1
    assert R.digest(v1) == hashlib.sha512(b"verity/registration/v1\0" + canonical_json(v1)).hexdigest()


def test_the_audit_record_holds_the_roots_the_reads_were_opened_against():
    prog = _program()
    reg = _registered(prog)
    rec, _ = _record(prog, reg)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    args = (rec, R.receipt(rec, "verifier-test"), d, D.parse_law("subset:2"), [1, 4], 2, {}, {"lean": lean}, 0.25)
    ok = audit.audit_record(*args, session_registered={"W": reg.root.hex()})
    assert ok["accepted"] and "registered" not in ok["checks"]
    assert ok["integrity_profile"]["inputs"]["registered"] == [{k: reg.entry()[k] for k in ("value", "root", "leaves", "domain")}]
    for opened in (None, {}, {"W": "cd" * 64}, {"W": reg.root.hex(), "V": "cd" * 64}):
        r = audit.audit_record(*args, session_registered=opened)
        assert not r["accepted"] and r["checks"]["registered"]["codes"] == ["registered"] and r["integrity_profile"] is None
    v1 = R.record(prog, Q, LAW, SCHEME, rec["roots"], SERVED)
    plain = audit.audit_record(v1, R.receipt(v1, "verifier-test"), *args[2:])
    assert plain["accepted"] and "registered" not in plain["integrity_profile"]["inputs"]
    assert not audit.audit_record(v1, R.receipt(v1, "verifier-test"), *args[2:], session_registered={"W": reg.root.hex()})["accepted"]


def test_the_opened_roots_come_from_the_verifier_of_records_accepted_verdicts():
    """`session_registered` reads the Lean verdict's `registered` finding, only on an accepted verdict of record: a v2 audit
    passes on what the verifier of record opened, and fails closed on anything else."""
    prog = _program()
    reg = _registered(prog)
    rec, _ = _record(prog, reg)
    d = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64, "registered": {"w": {"value": "W", "root": reg.root.hex()}}}
    assert audit.session_registered({"lean": lean}) == {"W": reg.root.hex()}
    for v in ({**lean, "accepted": False}, {**lean, "accepted": None}, {**lean, "of_record": False}, {**lean, "registered": None},
              {k: x for k, x in lean.items() if k != "registered"}):
        assert audit.session_registered({"lean": v}) is None, v
    other = {**lean, "registered": {"w": {"value": "W", "root": "cd" * 64}}}
    assert audit.session_registered({"lean": lean, "lean-2": other}) == {"W": None}
    assert audit.session_registered({"lean": lean, "lean-2": lean}) == {"W": reg.root.hex()}

    def decide(verdicts):
        return audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, D.parse_law("subset:2"), [1, 4], 2, {}, verdicts,
                                  0.25, session_registered=audit.session_registered(verdicts))

    assert decide({"lean": lean})["accepted"]
    for verdicts in ({"lean": other}, {"lean": lean, "lean-2": other}, {"lean": {k: x for k, x in lean.items() if k != "registered"}}):
        r = decide(verdicts)
        assert not r["accepted"] and r["checks"]["registered"]["codes"] == ["registered"], verdicts
