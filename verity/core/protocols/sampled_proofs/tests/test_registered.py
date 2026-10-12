"""Registered values: a program's table committed once under hiding leaves (`one_stage.registered`), the registration's
R7, the verifier's read map from its own program, the read check, and the audit record's link to the registered roots."""
from __future__ import annotations

import hashlib

import pytest

import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import coll_concat
from verity.ml.circuits.types import Array, Value
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P, registered as G, registration as R

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
#: a 16-bit registration's root as `registered.py` computed it before `word_bits`: three 64-word rows, salts 01.., 02.., 03..
ROOT_16 = ("17b12eea66ca473baf95f834bcdef8bbb94001549fa85061f3b0166d795d2812"
           "7d5db86a105ab3ceb1821cf9f57a66323a873f55355370fdeff6bd7e2cfd8a07")
#: a 32-bit registration's root: one row of eps and the softmax scale at D = 64, salt 09..
ROOT_32 = ("111850c621f50137b50e36ee9e22fe8cefd28761e8b7ad98b175eaf0e8c7413d"
           "27364f184783f7680bf638c9b039154a53ef8908faba261a4eb208c22552c40f")


def _program(pos=POS, rows: int = len(TABLE), body=None, w: int = 8) -> Program:
    def sig(S):
        return (("x", Array(len(pos), Array(2, Value(8)))), ("W", Array(rows, Array(2, Value(w))))), Array(len(pos), Value(8))

    def default(B, S, x, W):
        return coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p] if w == 8 else x[i]) for i, p in enumerate(pos)])

    return Program(bind(CompositeDefinition("RegisteredTestProgram", 0, (), sig, body or default, register=False)))


Q = P.template_instance_query(MIX.id)


def _registered(prog: Program) -> G.Registered:
    return G.Registered.commit(P.program_digest(prog), "W", TABLE)


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
    dom = G.domain(P.program_digest(prog), "W", 2, len(TABLE))
    inst = hm96.SHA512
    for r, row in enumerate(TABLE):
        # the statement's port row: the two 16-bit words, zero-padded to one SHA-512 block
        assert G.row_words(row) == [*row, *[0] * 62]
        assert reg.bcs[r] == inst.commit_string(inst.default_key, sha512_row_digest([*row, *[0] * 62], 16, ROLE_X), reg.salts[r])
        assert G.verify_read(dom, reg.root, r, reg.bcs[r], reg.path(r))
        assert len(reg.path(r)) == 64 * merkle_depth(len(TABLE))
    assert reg.entry() == {"value": "W", "root": reg.root.hex(), "leaves": 5, "words": 2, "domain": dom.domain_id.hex()}
    # fresh salts: the same table registered again is another root, whose rows are unlinkable to these
    again = _registered(prog)
    assert again.root != reg.root and set(again.bcs).isdisjoint(reg.bcs)


def test_a_16_bit_value_keeps_the_bytes_it_had_before_word_bits():
    """16 is the default width: its domain binds the five keys it always did, and a whole-block row is the port's row
    as it stands, so a 16-bit value's domain, leaves and root are the ones registered before `word_bits` (`ROOT_16`)."""
    from verity.ml.commitments.identity import identity_digest

    prog = P.program_digest(_program())
    binding = identity_digest(G.REGISTERED_DOMAIN, {"program": prog, "value": "W", "schema": G.SCHEMA, "words": 2, "leaves": 5})
    assert G.domain(prog, "W", 2, 5).domain_id == G.domain(prog, "W", 2, 5, 16).domain_id
    assert G.domain(prog, "W", 2, 5).binding == bytes.fromhex(binding)
    row = list(range(1, 65))
    assert G.row_words(row) == row and G.row_digest(row) == sha512_row_digest(row, 16, ROLE_X)
    rows = [[(r * 64 + j) * 1009 % 65536 for j in range(64)] for r in range(3)]
    reg = G.Registered.commit("ab" * 64, "cs_table", rows, [bytes([r + 1]) * 192 for r in range(3)])
    assert reg.root.hex() == ROOT_16 and "word_bits" not in reg.entry()



def test_a_32_bit_value_is_its_u16_words_low_first_under_its_own_domain():
    """A 32-bit leaf is two u16 words, low first (the statement's layout of a port row, and C-Flock's `values` row): its
    domain names `word_bits`, so a 32-bit registration is never a 16-bit one's, and the entry carries it."""
    prog = P.program_digest(_program(w=32))
    assert G.row_words([0x3727C5AC, 0x3E38AA3B], 32) == [0xC5AC, 0x3727, 0xAA3B, 0x3E38, *[0] * 60]
    assert G.domain(prog, "W", 2, 5, 32).domain_id != G.domain(prog, "W", 2, 5).domain_id
    reg = G.Registered.commit(prog, "W", [[0x3727C5AC, 0x3E38AA3B]] * 2, word_bits=32)
    assert reg.entry() == {"value": "W", "root": reg.root.hex(), "leaves": 2, "words": 2, "word_bits": 32,
                           "domain": G.domain(prog, "W", 2, 2, 32).domain_id.hex()}
    inst = hm96.SHA512
    assert reg.bcs[0] == inst.commit_string(inst.default_key, G.row_digest([0x3727C5AC, 0x3E38AA3B], 32), reg.salts[0])
    assert G.verify_read(reg.domain, reg.root, 1, reg.bcs[1], reg.path(1))
    assert not G.verify_read(G.domain(prog, "W", 2, 2), reg.root, 1, reg.bcs[1], reg.path(1))
    pinned = G.Registered.commit("ab" * 64, "values", [[0x3727C5AC, 0x3E38AA3B]], [bytes([9]) * 192], word_bits=32)
    assert pinned.root.hex() == ROOT_32
    for bad in (dict(rows=[[1 << 32, 0]], word_bits=32), dict(rows=[[1 << 16, 0]]), dict(rows=[[1, 2]], word_bits=8),
                dict(rows=[[1, 2]], word_bits=64)):
        with pytest.raises(ValueError):
            G.Registered.commit(prog, "W", **bad)


def test_the_verifier_derives_word_bits_from_its_own_program():
    for w, want in ((1, 16), (8, 16), (16, 16), (17, 32), (32, 32)):
        assert G.width(_program(w=w), "W") == want
        assert G.registered_entries(_program(w=w), {"W": 2})["W"]["word_bits"] == want
    with pytest.raises(ValueError, match="at most 32 bits"):
        G.width(_program(w=64), "W")
    with pytest.raises(ValueError, match="not a parameter"):
        G.width(_program(), "V")


def test_a_read_opens_only_its_own_row_under_its_own_value():
    prog = _program()
    reg = _registered(prog)
    dom = reg.domain
    assert not G.verify_read(dom, reg.root, 2, reg.bcs[3], reg.path(3))
    assert not G.verify_read(dom, reg.root, 3, reg.bcs[2], reg.path(3))
    other = G.domain(P.program_digest(prog), "V", 2, len(TABLE))
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
        G.Registered.commit(P.program_digest(prog), "W", TABLE[:2], [salt, salt])
    with pytest.raises(ValueError):
        G.Registered.commit(P.program_digest(prog), "W", TABLE[:2], [salt])
    with pytest.raises(ValueError):
        G.Registered.commit(P.program_digest(prog), "W", [[1, 2], [3]])


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
                   "ports": {"w": {"value": "W", "root": reg.root.hex(), "leaves": 5, "words": 2, "word_bits": 16,
                                   "domain": reg.domain.domain_id.hex(), "reads": list(POS)}}}
    keys = ("leaves", "words", "word_bits", "domain")
    assert {k: doc["ports"]["w"][k] for k in keys} == {k: G.registered_entries(prog, {"W": 2})["W"][k] for k in keys}
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
             ({"registered": [{**e, "extra": 1}]}, "R7-registered"), ({"format": "verity/registration/v1"}, "R1-protocol"),
             ({"registered": [{**e, "word_bits": 32}]}, "R7-word_bits"), ({"registered": [{**e, "word_bits": 16}]}, "R7-word_bits"),
             ({"registered": [{**e, "leaves": 5.0}]}, "R7-leaves"), ({"registered": [{**e, "words": 2.0}]}, "R7-words")]
    for change, code in cases:
        assert code in R.check({**rec, **change}, prog, Q, **kw, registered={"W": 2}).codes, code
    assert "R7-registered" in R.check(rec, prog, Q, **kw, registered={"W": 3}).codes
    # a count equal to the verifier's is still refused as a bool: True would give the registration a second digest
    one = _program(pos=(0, 0), rows=1)
    rec1, ports1 = _record(one, G.Registered.commit(P.program_digest(one), "W", TABLE[:1]))
    kw1 = {"law": LAW, "scheme": SCHEME, "ports": ports1}
    assert rec1["registered"][0]["leaves"] == 1 and R.check(rec1, one, Q, **kw1, registered={"W": 2}).ok
    true = {**rec1, "registered": [{**rec1["registered"][0], "leaves": True}]}
    assert R.digest(true) != R.digest(rec1) and R.check(true, one, Q, **kw1, registered={"W": 2}).codes == ("R7-leaves",)
    # the record names 16 by leaving it out, whatever the entry it was given
    assert R.record(prog, Q, LAW, SCHEME, rec["roots"], SERVED, registered=[{**e, "word_bits": 16}]) == rec


def test_an_r7_root_of_127_hex_digits_and_a_newline_is_refused():
    """128 characters, all hex but the last: `$` would match before the newline, so the root is held to hex by `fullmatch`,
    as Lean's `isHex` holds it."""
    prog = _program()
    reg = _registered(prog)
    rec, ports = _record(prog, reg)
    root = reg.entry()["root"][:127] + "\n"
    assert len(root) == 128
    got = R.check({**rec, "registered": [{**reg.entry(), "root": root}]}, prog, Q, law=LAW, scheme=SCHEME, ports=ports,
                  registered={"W": 2})
    assert got.codes == ("R7-root",)


def test_r7_holds_a_32_bit_value_to_the_verifiers_width():
    """The verifier's program says the table's leaves are 32 bits: a 32-bit registration of it passes R7, and one that
    names another width, or none (16), is refused, as is a 32-bit registration of a 16-bit table."""
    prog = _program(w=32)
    reg = G.Registered.commit(P.program_digest(prog), "W", [[r * 0x10001, r] for r in range(5)], word_bits=32)
    rec, ports = _record(prog, reg)
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports}
    assert rec["registered"] == [reg.entry()] and reg.entry()["word_bits"] == 32
    assert R.check(rec, prog, Q, **kw, registered={"W": 2}).ok
    no_width = {k: x for k, x in reg.entry().items() if k != "word_bits"}
    assert R.check({**rec, "registered": [no_width]}, prog, Q, **kw, registered={"W": 2}).codes == ("R7-word_bits",)
    as_float = {**rec, "registered": [{**reg.entry(), "word_bits": 32.0}]}
    assert R.digest(as_float) != R.digest(rec)
    assert R.check(as_float, prog, Q, **kw, registered={"W": 2}).codes == ("R7-word_bits",)
    sixteen = G.Registered.commit(P.program_digest(prog), "W", TABLE)
    assert set(R.check({**rec, "registered": [sixteen.entry()]}, prog, Q, **kw, registered={"W": 2}).codes) == {"R7-domain",
                                                                                                              "R7-word_bits"}
    narrow = _program()
    rec16, ports16 = _record(narrow, G.Registered.commit(P.program_digest(narrow), "W", TABLE, word_bits=32))
    assert set(R.check(rec16, narrow, Q, law=LAW, scheme=SCHEME, ports=ports16, registered={"W": 2}).codes) == {"R7-domain",
                                                                                                              "R7-word_bits"}
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
