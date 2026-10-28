"""Registered values: a program's table committed once under hiding leaves (`one_stage.registered`), the registration's
R7, the verifier's read map from its own program, the read check, and the audit record's link to the registered roots."""
from __future__ import annotations

import hashlib
import re
import tracemalloc

import pytest

import verity.ml.circuits.ops.scalar  # noqa: F401
from verity.ml.commitments import hm96
from verity.ml.commitments.merkle import merkle_depth
from verity.ml.commitments.rowleaf import ROLE_X, sha512_row_digest
from verity.ml.circuits.codec import canonical_json
from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Affine, Concat, Explicit, Strided, coll_concat
from verity.ml.circuits.types import Array, Value
from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P, registered as G, registration as R

XOR8 = PrimitiveDefinition("RegisteredTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
MIX = bind(CompositeDefinition("RegisteredTestMix", 0, (), lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8)))), Value(8)),
                               lambda B, S, x, w: B.call(XOR8, B.call(XOR8, x[0], w[0]), B.call(XOR8, x[1], w[1])), register=False))
#: the row of the table W each proof block reads, fixed by the program
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


#: each proof block's seed group: the row of `pouw.roots` it reads
GROUPS = (0, 0, 1, 1, 1, 0)
SEEDED = bind(CompositeDefinition(
    "RegisteredTestSeeded", 0, ("BITS",),
    lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8))), ("roots", Array(2, _digest(S.BITS)))), Value(8)),
    lambda B, S, x, w, roots: B.call(XOR8, B.call(XOR8, x[0], w[0]), B.call(XOR8, x[1], w[1])), register=False))
#: the verifier's sources of each seed group's roots: the served port x's root and W's registered root
SUPPLIED = {"pouw.roots": [[{"port": "x"}, {"value": "W"}], [{"value": "W"}, {"port": "x"}]]}
#: `supplied_salts("ab" * 64, "pouw.roots", 2)`' row 1, its first 32 bytes
SUPPLIED_SALT_1 = "09c18346b8d4a7e4a3f0cf7eb02a7460033e89c2e5733eaa24b2b2b53f036839"


def _digest(bits: int) -> Array:
    """A SHA-512 digest as PoUW's seeds read it: 16 leaves of 32 bits, or 512 of one."""
    return Array(16, Value(32)) if bits == 32 else Array(512, Value(1))


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
    ports = {p: {"leaves": leaves, "domain": R.window_domain(part["program"], P.digest(part), p, schema, leaves, "r-registered")}
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
    """A 32-bit leaf is two u16 words, low first (the statement's layout of a port row, and Flock's `values` row): its
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
    # rows of one leaf: w is an array of two whole rows, each read at its own key (a circuit without ports w[0] and w[1]
    # refuses the reads file that names them)
    assert G.reads(prog, Q, {"W": 1}) == {f"w[{j}]": {"value": "W", "rows": [2 * p + j for p in POS]} for j in range(2)}
    for bad in ({"W": 3}, {"V": 2}):
        with pytest.raises(ValueError):
            G.reads(prog, Q, bad)


def test_a_read_that_is_not_one_whole_row_is_refused():
    """A proof block reading across two rows, or a parameter that reads the table in one proof block and not in another, has no public
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


@pytest.mark.parametrize("schema", [G.SCHEMA_V2, G.SCHEMA_SEG])
@pytest.mark.parametrize("word_bits", [16, 32])
def test_a_bit_row_value_is_its_leaves_bit_row_under_its_ports_schema(schema, word_bits):
    """A value a bit-row port reads is committed in that port's format: each row the bit row of its leaves (`row_words`
    unpadded, little-endian), `x` its `sha512/row/v2` or `sha512/row-seg/v1` digest, each leaf framed under the port's
    schema, so the root differs from the `row/v1` root of the same rows and salts, and a read opens it only in its format.
    The domain is the same in every format."""
    from verity.ml.commitments.rowleaf import bit_row, row_leaf, sha512_row_seg_digest, sha512_row_v2_digest

    prog = P.program_digest(_program(w=word_bits))
    rows = [[0x3727C5AC & ((1 << word_bits) - 1), 0x3E38AA3B & ((1 << word_bits) - 1)], [5, 6], [7, 8]]
    salts = [bytes([r + 1]) * 192 for r in range(3)]
    reg = G.Registered.commit(prog, "W", rows, salts, word_bits, schema)
    v1 = G.Registered.commit(prog, "W", rows, salts, word_bits)
    inst = hm96.SHA512
    words = G.row_words(rows[0], word_bits, schema)
    assert len(words) == 2 * word_bits // 16, "a bit row is its leaves' bits, no whole-block padding"
    data, n = bit_row((words, 16))
    assert n == 2 * word_bits and data == b"".join(int(v).to_bytes(word_bits // 8, "little") for v in rows[0])
    digest = sha512_row_v2_digest(data, n) if schema == G.SCHEMA_V2 else sha512_row_seg_digest(data, n)
    assert G.row_digest(rows[0], word_bits, schema) == digest
    assert reg.bcs[0] == inst.commit_string(inst.default_key, digest, salts[0])
    assert reg.domain.domain_id == v1.domain.domain_id and reg.root != v1.root
    assert reg.levels[0][1] == row_leaf(reg.domain, 1, inst.tree_leaf(inst.default_key, reg.bcs[1]), schema)
    for r in range(3):
        assert G.verify_read(reg.domain, reg.root, r, reg.bcs[r], reg.path(r), schema)
        assert not G.verify_read(reg.domain, reg.root, r, reg.bcs[r], reg.path(r))
        other = G.SCHEMA_SEG if schema == G.SCHEMA_V2 else G.SCHEMA_V2
        assert not G.verify_read(reg.domain, reg.root, r, reg.bcs[r], reg.path(r), other)
    assert reg.schema == schema and "row_schema" not in reg.entry(), "the registration's entry is the same in every format"
    with pytest.raises(ValueError, match="not 'hm96-sha512/row/v3'"):
        G.Registered.commit(prog, "W", rows, salts, word_bits, "hm96-sha512/row/v3")


def test_a_row_seg_table_one_token_a_row_is_what_the_embedding_read_opens():
    """`gates.embedding`'s read at a hidden position (`EmbeddingRead_v1`) reads a table registered here under
    ``hm96-sha512/row-seg/v1`` with one token a row (``words = H``): each row opens at its token's position under the
    registration's root and domain id, and at no other token's."""
    from verity.ml.commitments.gates import embedding as E

    V, H = 5, 3
    rows = [[(7919 * t + 31 * j) & 0xFFFF for j in range(H)] for t in range(V)]
    salts = [bytes([t + 1]) * 192 for t in range(V)]
    reg = G.Registered.commit(P.program_digest(_program()), "W", rows, salts, schema=G.SCHEMA_SEG)
    dom, D = reg.domain.domain_id, merkle_depth(V)
    for t in range(V):
        path = [reg.path(t)[64 * k:64 * (k + 1)] for k in range(D)]
        assert E.reference(V, t, reg.root, dom, rows[t], salts[t], path) == (rows[t], 1)
        assert E.reference(V, (t + 1) % V, reg.root, dom, rows[t], salts[t], path)[1] == 0


def test_the_verifiers_reads_file_names_a_bit_row_values_format():
    """`row_schema` is the reads file's key for a value read by a bit-row port (`schemas`), absent for `row/v1`, so a v1
    file is the bytes it was. A bit row is its leaves' bits: W's two 8-bit leaves are one u16 word of a `row/v2` row, where
    a `row/v1` row gives each its own word."""
    prog = _program()
    reg = _registered(prog)
    v1 = G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()})
    for schema in (G.SCHEMA_V2, G.SCHEMA_SEG):
        doc = G.reads_file(prog, Q, {"W": 1}, {"W": reg.root.hex()}, schemas={"W": schema})
        assert doc["ports"]["w"] == {**v1["ports"]["w"], "words": 1, "row_schema": schema,
                                     "domain": G.domain(P.program_digest(prog), "W", 1, 5).domain_id.hex()}
        assert list(doc["ports"]["w"])[-2:] == ["row_schema", "reads"]
        with pytest.raises(ValueError, match="10 leaves are not rows of 2 .4 of its leaves in a"):
            G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()}, schemas={"W": schema})
    assert G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()}, schemas={"W": G.SCHEMA}) == v1
    with pytest.raises(ValueError, match="not 'hm96-sha512/row/v3'"):
        G.reads_file(prog, Q, {"W": 2}, {"W": reg.root.hex()}, schemas={"W": "hm96-sha512/row/v3"})


XOR1 = PrimitiveDefinition("RegisteredTestXor1", 1, (("a", Value(1)), ("b", Value(1))), Value(1), lambda a, b: a ^ b,
                           register=False)
#: an A row of K = 64 int8 codes, its bits one-bit leaves: a `row/v2` row of K/2 = 32 u16 words
K = 64
A_ROW = Array(8 * K, Value(1))
#: a tile reading R = 2 rows of A (H1's tile proof block: its parameter `a`, the circuit's ports `a0` and `a1`)
TILE = bind(CompositeDefinition("RegisteredTestTile", 0, (), lambda S: ((("a", Array(2, A_ROW)),), Value(1)),
                                lambda B, S, a: B.call(XOR1, a[0][0], a[1][0]), register=False))
QT = P.template_instance_query(TILE.id)


def _tiles(rows=lambda A, B, u: A[2 * u:2 * u + 2]) -> Program:
    """Two tiles over A's four rows, proof block u reading `rows(A, B, u)` (default rows 2u and 2u + 1); B is another value of A's
    shape."""
    def sig(S):
        return (("A", Array(4, A_ROW)), ("B", Array(4, A_ROW))), Array(2, Value(1))

    return Program(bind(CompositeDefinition("RegisteredTestTiles", 0, (), sig,
                                            lambda B_, S, A, B: coll_concat(sig(S)[1], [B_.call(TILE, rows(A, B, u))
                                                                                        for u in range(2)]),
                                            register=False)))


def test_a_bit_row_value_counts_its_parameters_leaves_in_its_bits():
    """A `row/v2` row of `words` leaves of `word_bits` holds `words * word_bits` bits, as many of the parameter's leaves as
    they fill (`row_leaves`): K int8 codes as one-bit leaves are K/2 = 32 u16 words, 8K leaves a row, so A is 4 rows
    under its domain of 32 words, the one H1's served tree commits under; a `row/v1` row keeps one parameter leaf a word."""
    prog = _tiles()
    assert G.row_leaves(prog, "A", K // 2, G.SCHEMA_V2) == G.row_leaves(prog, "A", K // 2, G.SCHEMA_SEG) == 8 * K
    assert G.row_leaves(prog, "A", K // 2) == K // 2 and G.shape(prog, "A", K // 2) == 4 * 8 * K // (K // 2)
    assert G.shape(prog, "A", K // 2, G.SCHEMA_V2) == 4
    assert G.registered_entries(prog, {"A": K // 2}, {"A": G.SCHEMA_V2})["A"] == {
        "leaves": 4, "words": K // 2, "word_bits": 16, "domain": G.domain(P.program_digest(prog), "A", K // 2, 4).domain_id.hex()}
    assert G.registered_entries(prog, {"A": K // 2})["A"]["leaves"] == 64, "row/v1 is counted as it was"
    with pytest.raises(ValueError, match="leaves are not rows of 33"):
        G.shape(prog, "A", 33, G.SCHEMA_V2)


def test_a_parameter_of_several_rows_reads_each_at_its_own_port():
    """A tile's parameter of R whole rows reads each at its own key `a[i]` (the circuit's port `a{i}`): proof block u's rows 2u
    and 2u + 1. The reads file names each as a port of A, in A's row format, with its own positions; a parameter that is
    exactly one row keeps its name."""
    prog = _tiles()
    want = {"a[0]": {"value": "A", "rows": [0, 2]}, "a[1]": {"value": "A", "rows": [1, 3]}}
    assert G.reads(prog, QT, {"A": K // 2}, {"A": G.SCHEMA_V2}) == want
    assert G.reads(prog, QT, {"A": 8 * K}) == want, "row/v1 rows of one word a leaf, the same rows"
    rows = [[(17 * r + j) & 0xFFFF for j in range(K // 2)] for r in range(4)]
    reg = G.Registered.commit(P.program_digest(prog), "A", rows, schema=G.SCHEMA_V2)
    doc = G.reads_file(prog, QT, {"A": K // 2}, {"A": reg.root.hex()}, ports={"a[0]": "a0", "a[1]": "a1"},
                       schemas={"A": G.SCHEMA_V2})
    port = {"value": "A", "root": reg.root.hex(), "leaves": 4, "words": K // 2, "word_bits": 16,
            "domain": reg.domain.domain_id.hex(), "row_schema": G.SCHEMA_V2}
    assert doc == {"format": G.READS, "ports": {"a0": {**port, "reads": [0, 2], "param": "a[0]"},
                                                "a1": {**port, "reads": [1, 3], "param": "a[1]"}}}
    for p, reads in (("a0", [0, 2]), ("a1", [1, 3])):
        for u, r in enumerate(reads):
            assert G.verify_read(reg.domain, reg.root, r, reg.bcs[r], reg.path(r), G.SCHEMA_V2)
            assert not G.verify_read(reg.domain, reg.root, reads[1 - u], reg.bcs[r], reg.path(r), G.SCHEMA_V2)
    crossed = _tiles(lambda A, B, u: coll_concat(Array(2, A_ROW), [A[3 - u], A[u]]))
    assert G.reads(crossed, QT, {"A": K // 2}, {"A": G.SCHEMA_V2}) == {"a[0]": {"value": "A", "rows": [3, 2]},
                                                                  "a[1]": {"value": "A", "rows": [0, 1]}}


def test_two_reads_under_one_port_name_are_refused():
    """A port is one read: a parameter named `a[0]` beside row 0 of the parameter `a`, or a `ports` map naming two reads
    one port, is refused, naming the port and both reads, never merged into one port's rows."""
    tile = bind(CompositeDefinition("RegisteredTestTileNamed", 0, (), lambda S: ((("a", Array(2, A_ROW)), ("a[0]", A_ROW)), Value(1)),
                                    lambda B, S, a, a0: B.call(XOR1, a[0][0], a0[0]), register=False))

    def sig(S):
        return (("A", Array(4, A_ROW)), ("B", Array(4, A_ROW))), Array(2, Value(1))

    prog = Program(bind(CompositeDefinition("RegisteredTestTilesNamed", 0, (), sig,
                                            lambda B_, S, A, B: coll_concat(sig(S)[1], [B_.call(tile, A[2 * u:2 * u + 2], B[u])
                                                                                        for u in range(2)]),
                                            register=False)))
    with pytest.raises(ValueError, match=r"port 'a\[0\]' names two reads: row 0 of parameter 'a' of template .* and "
                                         r"parameter 'a\[0\]' of template "):
        G.reads(prog, P.template_instance_query(tile.id), {"A": K // 2, "B": K // 2}, {"A": G.SCHEMA_V2, "B": G.SCHEMA_V2})
    tiles = _tiles()
    roots = {"A": "ab" * 64}
    for ports, want in (({"a[0]": "a", "a[1]": "a"}, "port 'a' names two reads: 'a[0]' of 'A' and 'a[1]' of 'A'"),
                        ({"a[0]": "a[1]"}, "port 'a[1]' names two reads: 'a[0]' of 'A' and 'a[1]' of 'A'")):
        with pytest.raises(ValueError, match=re.escape(want)):
            G.reads_file(tiles, QT, {"A": K // 2}, roots, ports=ports, schemas={"A": G.SCHEMA_V2})
    with pytest.raises(ValueError, match=re.escape("port 'w' names two reads: 'x' of 'x' and 'w' of 'W'")):
        G.reads_file(_program(), Q, {"W": 2, "x": 2}, {"W": "ab" * 64, "x": "cd" * 64}, ports={"x": "w"})
    assert G.reads_file(tiles, QT, {"A": K // 2}, roots, ports={"a[0]": "a0", "a[1]": "a1"},
                        schemas={"A": G.SCHEMA_V2})["ports"].keys() == {"a0", "a1"}


def test_a_parameter_of_several_rows_that_are_not_whole_rows_of_one_value_is_refused():
    """Each element is one whole row of the one value: elements straddling two rows, or a row of A beside a row of B,
    are no public read."""
    def straddle(A, B, u):
        flat = [A[r][j] for r in range(4) for j in range(8 * K)]
        return coll_concat(Array(2, A_ROW), flat[8 * K * u + 4 * K:8 * K * u + 4 * K + 16 * K])

    with pytest.raises(ValueError, match="nor an array of whole rows"):
        G.reads(_tiles(straddle), QT, {"A": K // 2}, {"A": G.SCHEMA_V2})
    mixed = _tiles(lambda A, B, u: coll_concat(Array(2, A_ROW), [A[u], B[u]]))
    with pytest.raises(ValueError, match="nor an array of whole rows"):
        G.reads(mixed, QT, {"A": K // 2, "B": K // 2}, {"A": G.SCHEMA_V2, "B": G.SCHEMA_V2})
    with pytest.raises(ValueError, match="nor an array of whole rows"):
        G.reads(_tiles(), QT, {"A": K // 2})


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
    """R7 holds a later record's root of a value the verifier holds across windows (`held`) to the one an earlier record
    of the log registered: the verifier holds no roots beside its log, so no caller can leave them out. A verifier that
    names neither that nor the window's root refuses its own inputs, naming both."""
    prog = _program()
    reg = _registered(prog)
    rec, ports = _record(prog, reg)
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"W": 2}, "held": ["W"]}

    def later(window, previous, entry):
        return R.record(prog, Q, LAW, SCHEME, rec["roots"], {**SERVED, "row": window}, previous=R.digest(previous),
                        registered=[entry] if entry else ())

    assert R.check(later(8, rec, reg.entry()), prog, Q, **kw, log=[rec]).ok
    assert R.check(later(8, rec, _registered(prog).entry()), prog, Q, **kw, log=[rec]).codes == ("R7-held",)
    v1 = later(8, rec, None)
    assert R.held_roots([rec, v1]) == {("W", reg.entry()["domain"]): reg.root.hex()} and R.held_roots([v1]) == {}
    assert R.check(later(9, v1, _registered(prog).entry()), prog, Q, **kw, log=[rec, v1]).codes == ("R7-held",)
    with pytest.raises(ValueError, match=r"registered values \['W'\].*`window_root`.*`held`"):
        R.check(later(8, rec, reg.entry()), prog, Q, **{**kw, "held": None}, log=[rec])


#: two windows' open roots, as the challenger's signed opens name them
WINDOWS = ("a1" * 64, "b2" * 64)
#: the window's own rows: x, which a window-independent program registers under the same name every window
X_ROWS = [[i, 2 * i] for i in range(len(POS))]


def _windows(prog: Program, xs, w: G.Registered):
    """A record per window, each the next's `previous`, registering x's commitment of that window and the weights W."""
    base, ports = _record(prog, w)
    out = []
    for i, x in enumerate(xs):
        out.append(R.record(prog, Q, LAW, SCHEME, base["roots"], {**SERVED, "row": 7 + i},
                            previous=R.digest(out[-1]) if out else None, registered=[x.entry(), w.entry()]))
    return out, {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"x": 2, "W": 2}}


def test_two_honest_windows_register_the_same_names_under_their_own_roots():
    """A window-independent program registers x afresh every window under the same name. Without the
    window's open root R, x's domain is the same in both windows, so R7 would hold the second window's x to the first's
    root (`R7-held`, and a replay of the first window's root would pass): the verifier refuses its own inputs instead,
    naming `window_root`, whether or not it holds W. With R (`window_root`, from the challenger's signed open), x's domain
    binds its window and both windows are accepted, while W, which the verifier holds across windows (`held`), keeps one
    window-independent domain and its first root."""
    prog = _program()
    d = P.program_digest(prog)
    w = _registered(prog)
    plain, kw = _windows(prog, [G.Registered.commit(d, "x", X_ROWS) for _ in WINDOWS], w)
    assert R.check(plain[0], prog, Q, **kw).ok
    for held in (None, ["W"]):
        with pytest.raises(ValueError, match=r"registered values \[.*'x'\].*`window_root`"):
            R.check(plain[1], prog, Q, **kw, log=plain[:1], held=held)
        with pytest.raises(ValueError, match=r"`window_root`"):
            R.check({**plain[1], "registered": plain[0]["registered"]}, prog, Q, **kw, log=plain[:1], held=held)
    xs = [G.Registered.commit(d, "x", X_ROWS, window=r) for r in WINDOWS]
    recs, kw = _windows(prog, xs, w)
    for i, (r, rec) in enumerate(zip(WINDOWS, recs)):
        assert R.check(rec, prog, Q, **kw, log=recs[:i], window_root=r, held=["W"]).ok
        assert rec["registered"][0]["domain"] == G.domain(d, "x", 2, len(POS), window=r).domain_id.hex()
        assert rec["registered"][1]["domain"] == w.entry()["domain"] == G.domain(d, "W", 2, len(TABLE)).domain_id.hex()
    assert xs[0].entry()["domain"] != xs[1].entry()["domain"] and xs[0].root != xs[1].root
    # (c) the weights are held across the windows: a second window's W under another root, or under its window's
    # domain, is refused
    other = G.Registered.commit(d, "W", TABLE)
    moved = {**recs[1], "registered": [xs[1].entry(), other.entry()]}
    assert R.check(moved, prog, Q, **kw, log=recs[:1], window_root=WINDOWS[1], held=["W"]).codes == ("R7-held",)
    windowed = {**recs[1], "registered": [xs[1].entry(), G.Registered.commit(d, "W", TABLE, window=WINDOWS[1]).entry()]}
    assert R.check(windowed, prog, Q, **kw, log=recs[:1], window_root=WINDOWS[1], held=["W"]).codes == ("R7-domain",
                                                                                                         "R7-held")
    # a verifier that doesn't hold W takes it per window too: then the weights' window-independent entry is refused
    assert R.check(recs[1], prog, Q, **kw, log=recs[:1], window_root=WINDOWS[1]).codes == ("R7-domain",)
    # the reads file opens each value under the domain R7 took
    reads = G.reads_file(prog, Q, {"x": 2, "W": 2}, {"x": xs[1].root.hex(), "W": w.root.hex()}, window=WINDOWS[1],
                         fixed=["W"])["ports"]
    assert {p: e["domain"] for p, e in reads.items() if p in ("x", "w")} == {"x": xs[1].entry()["domain"],
                                                                           "w": w.entry()["domain"]}


def test_a_window_replaying_an_earlier_windows_root_is_refused():
    """A second window that names the first window's x root opens it under no domain R7 takes. Under the
    first window's domain R7 refuses the entry (`R7-domain`); under its own window's domain the entry passes R7, which
    holds a per-window value to nothing, but no row of the first window's tree opens that root under the second
    window's domain, so every registered read of the replayed root is refused (`verify_read`, the Lean verifier's
    `Flock.Registered` check of each table row against its `--registered` port)."""
    prog = _program()
    d = P.program_digest(prog)
    w = _registered(prog)
    xs = [G.Registered.commit(d, "x", X_ROWS, window=r) for r in WINDOWS]
    recs, kw = _windows(prog, xs, w)
    win = {"window_root": WINDOWS[1], "held": ["W"], "log": recs[:1]}
    replay = {**recs[1], "registered": [xs[0].entry(), w.entry()]}
    assert R.check(replay, prog, Q, **kw, **win).codes == ("R7-domain",)
    renamed = {**recs[1], "registered": [{**xs[0].entry(), "domain": xs[1].entry()["domain"]}, w.entry()]}
    assert R.check(renamed, prog, Q, **kw, **win).ok
    first, second = (G.domain(d, "x", 2, len(POS), window=r) for r in WINDOWS)
    for r in range(len(POS)):
        assert G.verify_read(first, xs[0].root, r, xs[0].bcs[r], xs[0].path(r))
        assert not G.verify_read(second, xs[0].root, r, xs[0].bcs[r], xs[0].path(r))
    port = G.reads_file(prog, Q, {"x": 2, "W": 2}, {"x": xs[0].root.hex(), "W": w.root.hex()}, window=WINDOWS[1],
                        fixed=["W"])["ports"]["x"]
    assert port["domain"] == second.domain_id.hex()


def test_the_window_root_and_held_values_are_the_verifiers_never_the_records():
    """R7 reads the window's open root and the values held across windows from the verifier alone (the challenger's
    signed open, and the window Program the period commits), never from the record. A record whose window names another
    root or holds x gets the verdict it gets without them; a record or entry carrying them as keys of its own is refused
    by its form."""
    prog = _program()
    d = P.program_digest(prog)
    w = _registered(prog)
    plain, kw = _windows(prog, [G.Registered.commit(d, "x", X_ROWS) for _ in WINDOWS], w)
    xs = [G.Registered.commit(d, "x", X_ROWS, window=r) for r in WINDOWS]
    recs, _ = _windows(prog, xs, w)
    claims = {"window_root": WINDOWS[0], "held": ["x", "W"]}
    claimed = lambda rec: {**rec, "window": {**rec["window"], **claims}}  # noqa: E731
    with pytest.raises(ValueError, match=r"registered values \['x'\].*`window_root`"):
        R.check(claimed(plain[1]), prog, Q, **kw, log=plain[:1], held=["W"])
    win = {"window_root": WINDOWS[1], "held": ["W"], "log": recs[:1]}
    for rec in (recs[1], {**recs[1], "registered": [xs[0].entry(), w.entry()]},
                {**recs[1], "registered": [G.Registered.commit(d, "x", X_ROWS).entry(), w.entry()]}):
        assert R.check(claimed(rec), prog, Q, **kw, **win).codes == R.check(rec, prog, Q, **kw, **win).codes
    assert R.check(recs[1], prog, Q, **kw, **win).ok
    for k, v in claims.items():
        assert R.check({**recs[1], k: v}, prog, Q, **kw, **win).codes == ("keys",)
        entries = [{**xs[1].entry(), k: v}, w.entry()]
        assert R.check({**recs[1], "registered": entries}, prog, Q, **kw, **win).codes == ("R7-registered",)


def test_a_session_bound_domain_binds_the_session_alone():
    """A value whose root the window's open root contains (PoUW's round-0 input-ids table) binds the window's session in
    the open root's place: its domain differs from the unbound one, the window-bound one and another session's, and a
    domain binds at most one of a window, a registration and a session."""
    prog = P.program_digest(_program())
    rec = G.Registered.commit(prog, "x", X_ROWS, session="s-1")
    doms = {G.domain(prog, "x", 2, len(X_ROWS), **kw).domain_id for kw in
            ({}, {"window": WINDOWS[0]}, {"session": "s-1"}, {"session": "s-2"})}
    assert len(doms) == 4 and rec.domain.domain_id == G.domain(prog, "x", 2, len(X_ROWS), session="s-1").domain_id
    for kw in ({"window": WINDOWS[0], "session": "s-1"}, {"produced": "0" * 128, "session": "s-1"}, {"session": ""}):
        with pytest.raises(ValueError):
            G.domain(prog, "x", 2, len(X_ROWS), **kw)


def _prefixed(prefix: str) -> Program:
    """The program as a window-dependent path wrote it before the window's root was in the domain: every parameter's name
    carries its window's prefix."""
    def sig(S):
        return ((f"{prefix}x", Array(len(POS), Array(2, Value(8)))),
                (f"{prefix}W", Array(len(TABLE), Array(2, Value(8))))), Array(len(POS), Value(8))

    return Program(bind(CompositeDefinition("RegisteredTestProgram", 0, (), sig,
                                            lambda B, S, x, W: coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p])
                                                                                       for i, p in enumerate(POS)]),
                                            register=False)))


def test_the_window_prefixed_form_still_verifies():
    """A registration whose names carry their window's prefix is still accepted, with or without the window's
    root: each window's values are other names, so nothing in the log holds them. (It holds nothing across windows
    either, the weights included, which is what the window's root in the domain gives back.)"""
    for windowed in (False, True):
        log: list = []
        for i, r in enumerate(WINDOWS):
            prog, x, w, window_root = _prefixed(f"w{i}/"), f"w{i}/x", f"w{i}/W", r if windowed else None
            d = P.program_digest(prog)
            base, ports = _record(prog, G.Registered.commit(d, w, TABLE))
            rec = R.record(prog, Q, LAW, SCHEME, base["roots"], {**SERVED, "row": 7 + i},
                           previous=R.digest(log[-1]) if log else None,
                           registered=[G.Registered.commit(d, x, X_ROWS, window=window_root).entry(),
                                       G.Registered.commit(d, w, TABLE).entry()])
            assert R.check(rec, prog, Q, LAW, SCHEME, ports, log=log, registered={x: 2, w: 2}, window_root=window_root,
                           held=[w] if windowed else None).ok, (i, windowed)
            log.append(rec)


def test_a_window_root_and_held_values_are_the_verifiers_in_one_form():
    prog = _program()
    w = _registered(prog)
    recs, kw = _windows(prog, [G.Registered.commit(P.program_digest(prog), "x", X_ROWS, window=WINDOWS[0])], w)
    for bad in ("A1" * 64, "a1" * 63, "a1" * 63 + "a\n", 1, b"\xa1" * 64):
        with pytest.raises(ValueError, match="window root"):
            R.check(recs[0], prog, Q, **kw, window_root=bad)
    for bad in (["V"], "W", [1], {"W": 1}):
        with pytest.raises(ValueError, match="held values"):
            R.check(recs[0], prog, Q, **kw, window_root=WINDOWS[0], held=bad)


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
    passes on what the verifier opened, and fails closed on anything else."""
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


# -- supplied values ---------------------------------------------------------------------------------------------------------

def _seeded(bits: int = 32) -> Program:
    """Units that read a row of x, a row of the registered table W and their seed group's row of `pouw.roots`, a pair of
    digests, as PoUW's compiled proof blocks read theirs."""
    unit = bind(SEEDED, BITS=bits)

    def sig(S):
        return (("x", Array(len(POS), Array(2, Value(8)))), ("W", Array(len(TABLE), Array(2, Value(8)))),
                ("pouw.roots", Array(max(GROUPS) + 1, Array(2, _digest(bits))))), Array(len(POS), Value(8))

    def body(B, S, x, W, roots):
        return coll_concat(sig(S)[1], [B.call(unit, x[i], W[p], roots[g]) for i, (p, g) in enumerate(zip(POS, GROUPS))])

    return Program(bind(CompositeDefinition(f"RegisteredTestSeededProgram{bits}", 0, (), sig, body, register=False)))


def _seeded_case(bits: int = 32, x_root: str = "ab" * 64, roots=None, **kw):
    """A v2 record of `_seeded(bits)` registering W and `pouw.roots`, the latter as the verifier fills it from the record
    (`supplied_value`) unless `roots` (a `Registered`) says otherwise, with R7's arguments."""
    prog = _seeded(bits)
    q = P.template_instance_query(bind(SEEDED, BITS=bits).id)
    part, n = P.partition(prog, q), P.units(prog, q)
    ports = {p: {"leaves": n, "domain": R.window_domain(part["program"], P.digest(part), p, s, n, "r-registered")}
             for p, s in (("x", "hm96-sha512/row/v1"), ("out", "u16"))}
    served = [{"port": p, "root": x_root if p == "x" else "cd" * 64, **ports[p]} for p in ports]
    w = G.Registered.commit(P.program_digest(prog), "W", TABLE)
    base = R.record(prog, q, LAW, SCHEME, served, SERVED, registered=[w.entry()], **kw)
    sup = G.sources(SUPPLIED)["pouw.roots"]
    roots = roots or G.supplied_value(prog, "pouw.roots", sup, base, ["pouw.roots"])
    rec = R.record(prog, q, LAW, SCHEME, served, SERVED, registered=[w.entry(), roots.entry()], **kw)
    words = 2 * _digest(bits).leaves
    check = {"program": prog, "query": q, "law": LAW, "scheme": SCHEME, "ports": ports,
             "registered": {"W": 2, "pouw.roots": words}, "supplied": SUPPLIED}
    return rec, check, w, roots


def test_a_supplied_value_is_the_records_roots_as_its_leaves():
    """The verifier fills each seed group's row with the bytes of the roots its sources name, read as the parameter's leaves
    (each big-endian 32-bit half, least significant leaf first: PoUW's `seeds.leaves`), and commits the rows under public
    salts: a function of the record alone, which the prover reproduces."""
    assert G.bytes_leaves(bytes.fromhex("12345678" "01020304"), 32) == [0x12345678, 0x01020304]
    assert G.bytes_leaves(bytes.fromhex("12345678"), 16) == [0x5678, 0x1234]
    assert G.bytes_leaves(bytes.fromhex("80000001"), 1) == [1] + [0] * 30 + [1]
    for bad in ((b"\0" * 3, 32), (b"\0" * 4, 3), (b"\0" * 4, 64)):
        with pytest.raises(ValueError):
            G.bytes_leaves(*bad)
    salts = G.supplied_salts("ab" * 64, "pouw.roots", 2)
    assert salts[1][:32].hex() == SUPPLIED_SALT_1
    assert salts[1] == hashlib.shake_256(b"verity/one-stage/supplied-salt/v0\0" + b"ab" * 64 + b"\0pouw.roots\0"
                                         + (1).to_bytes(4, "little")).digest(192)
    for bits in (32, 1):
        rec, check, w, roots = _seeded_case(bits)
        x, wr = bytes.fromhex("ab" * 64), w.root
        rows = G.supplied_rows(check["program"], "pouw.roots", G.sources(SUPPLIED)["pouw.roots"], rec, ["pouw.roots"])
        assert rows == [G.bytes_leaves(x + wr, bits), G.bytes_leaves(wr + x, bits)] == [list(r) for r in roots.rows]
        assert roots.salts == tuple(G.supplied_salts(P.program_digest(check["program"]), "pouw.roots", 2))
        assert roots.entry()["words"] == check["registered"]["pouw.roots"] and roots.word_bits == (32 if bits == 32 else 16)
        assert G.supplied_root(check["program"], "pouw.roots", G.sources(SUPPLIED)["pouw.roots"], rec,
                               G.registered_entries(check["program"], check["registered"])["pouw.roots"],
                               ["pouw.roots"]) == roots.root.hex()
        assert R.check(rec, **check).ok, bits
        # its reads are a registered value's: each proof block's row is its seed group's
        assert G.reads(check["program"], check["query"], check["registered"])["roots"] == {"value": "pouw.roots",
                                                                                         "rows": list(GROUPS)}


def test_r7_refuses_a_supplied_value_the_prover_chose():
    """A record whose `pouw.roots` is any other root than the one the verifier fills from the record is refused,
    whether its rows are other roots (all zero, as the witness fixes them), the right rows under fresh salts, or the right
    commitment of a record whose x root moved since."""
    rec, check, w, roots = _seeded_case()
    prog, words = check["program"], check["registered"]["pouw.roots"]
    zero = G.Registered.commit(P.program_digest(prog), "pouw.roots", [[0] * words] * 2, word_bits=32)
    zero_public = G.Registered.commit(P.program_digest(prog), "pouw.roots", [[0] * words] * 2,
                                      G.supplied_salts(P.program_digest(prog), "pouw.roots", 2), word_bits=32)
    fresh = G.Registered.commit(P.program_digest(prog), "pouw.roots", roots.rows, word_bits=32)
    for other in (zero, zero_public, fresh):
        bad, _, _, _ = _seeded_case(roots=other)
        assert R.check(bad, **check).codes == ("R7-supplied",)
    moved, _, _, _ = _seeded_case(x_root="ee" * 64, roots=roots)
    assert R.check(moved, **check).codes == ("R7-supplied",)
    assert R.check(_seeded_case(x_root="ee" * 64)[0], **check).ok


def test_r7_refuses_a_supplied_value_it_cannot_fill():
    """A source naming nothing the record holds, a supplied value's own root, bytes that are not lowercase hex, rows of the
    wrong count or width, or a supplied value the verifier does not expect registered: each is `R7-supplied`."""
    rec, check, _, _ = _seeded_case()
    for sup in ({"pouw.roots": [[{"port": "y"}, {"value": "W"}]] * 2},
                {"pouw.roots": [[{"port": "x"}, {"value": "pouw.roots"}]] * 2},
                {"pouw.roots": [[{"port": "x"}, {"bytes": "AB" * 64}]] * 2},
                {"pouw.roots": [[{"port": "x"}, {"bytes": ""}]] * 2},
                {"pouw.roots": [[{"port": "x"}, {"bytes": "abc"}]] * 2},
                {"pouw.roots": [[{"port": "x"}, {"value": "W"}]]},
                {"pouw.roots": [[{"port": "x"}]] * 2},
                {**SUPPLIED, "V": [[{"port": "x"}]]}):
        assert R.check(rec, **{**check, "supplied": sup}).codes == ("R7-supplied",), sup
    # the verifier's own bytes are a source like any root: the right ones pass
    x = {"pouw.roots": [[{"bytes": "ab" * 64}, {"value": "W"}], [{"value": "W"}, {"bytes": "ab" * 64}]]}
    assert R.check(rec, **{**check, "supplied": x}).ok


def test_a_supplied_value_is_the_records_own_not_the_logs():
    """R7-held is for a value committed once: a supplied value moves with the record's roots, so a later record whose x
    root moved registers another `pouw.roots`, held to the root it fills from its own record."""
    first, check, w, _ = _seeded_case()
    served = [{**r, "root": "ee" * 64} if r["port"] == "x" else r for r in first["roots"]]
    later = {**first, "roots": served, "window": {**SERVED, "row": 8}, "previous": R.digest(first),
             "registered": [w.entry()]}
    roots = G.supplied_value(check["program"], "pouw.roots", G.sources(SUPPLIED)["pouw.roots"], later, ["pouw.roots"])
    later["registered"] = [w.entry(), roots.entry()]
    assert later["registered"][1]["root"] != first["registered"][1]["root"]
    assert R.check(later, **check, log=[first], held=["W"]).ok
    stale = {**later, "registered": first["registered"]}
    assert R.check(stale, **check, log=[first], held=["W"]).codes == ("R7-supplied",)


def test_supplied_values_are_a_v2_verifiers_in_one_form():
    rec, check, _, _ = _seeded_case()
    for bad in ([], {"pouw.roots": [{"port": "x"}]}, {"pouw.roots": [[{"port": "x", "value": "W"}]]},
                {"pouw.roots": [[{"root": "x"}]]}, {"pouw.roots": [[{"port": 1}]]}, {"pouw.roots": [["x"]]}):
        with pytest.raises(ValueError):
            R.check(rec, **{**check, "supplied": bad})
    v1 = {k: x for k, x in rec.items() if k != "registered"} | {"format": "verity/registration/v1"}
    with pytest.raises(ValueError, match="none registered"):
        R.check(v1, **{**check, "registered": None})


#: the verifier's coin, and each seed group's row of `pouw.roots` filled from the window salt and positions, the bytes
#: PoUW's `pouw.salt` and `pouw.positions` take
COIN = "c0" * 64
MIXED = {"pouw.roots": [[{"window": COIN}] + [{"position": "0"}] * 8, [{"position": "5"}] * 8 + [{"window": COIN}]]}
#: the verifier's binding (its stream's digest and the identity it registered the prover under), and a window naming it
BINDING = {"stream": "5a" * 64, "prover": "b0" * 32}
BOUND = {**SERVED, **BINDING}
#: `window_salt(COIN, BOUND, BINDING)`, its first 32 bytes
WINDOW_SALT_0 = "4b8a2d020b9e146946769ecd04fb9ba6bd3b689b9a625e8334287acce768ef37"


def _u64(p: int) -> bytes:
    return p.to_bytes(8, "big")


def _mixed_case(base, sup=MIXED, log=(), window=BOUND, binding=BINDING):
    """`base` (a `_seeded_case`) as a record after `log` for `window`, its `pouw.roots` filled from `sup` as a verifier
    bound to `binding` fills it, with R7's arguments."""
    rec, check, w, _ = base
    rec = {**rec, "window": dict(window), "previous": R.digest(log[-1]) if log else None, "registered": [w.entry()]}
    roots = G.supplied_value(check["program"], "pouw.roots", G.sources(sup)["pouw.roots"], rec, ["pouw.roots"], list(log),
                             binding=binding)
    return ({**rec, "registered": [w.entry(), roots.entry()]},
            {**check, "supplied": sup, "log": list(log), "binding": binding}, roots)


def test_a_position_follows_the_logs_units():
    """W-a: a position source is K after every proof block of the log's earlier records, as a u64 big-endian, so a proof block served
    again in a later record takes another position and so another seed (the red team's replay, two calls of one A and B
    at one position under one salt, gets no second receipt); a record that names the positions it would have had first,
    or a source past its own proof blocks, is `R7-supplied`."""
    assert G.log_base([]) == 0 and G.log_base([{"population": 6}, {"population": 3}]) == 9
    for bad in ([{"population": -1}], [{"population": True}], [{}], [{"population": 1.0}], ["r"]):
        with pytest.raises(ValueError):
            G.log_base(bad)
    base = _seeded_case()
    first, check, roots = _mixed_case(base)
    salt = G.window_salt(bytes.fromhex(COIN), BOUND, BINDING)
    assert [list(r) for r in roots.rows] == [G.bytes_leaves(salt + _u64(0) * 8, 32), G.bytes_leaves(_u64(5) * 8 + salt, 32)]
    assert R.check(first, **check).ok
    window = {**BOUND, "row": 8}
    later, check, again = _mixed_case(base, log=[first], window=window)
    check["held"] = ["W"]
    salt = G.window_salt(bytes.fromhex(COIN), window, BINDING)
    assert [list(r) for r in again.rows] == [G.bytes_leaves(salt + _u64(6) * 8, 32), G.bytes_leaves(_u64(11) * 8 + salt, 32)]
    assert R.check(later, **check).ok
    replay = G.supplied_value(check["program"], "pouw.roots", G.sources(MIXED)["pouw.roots"], later, ["pouw.roots"],
                              binding=BINDING)
    assert R.check({**later, "registered": [later["registered"][0], replay.entry()]}, **check).codes == ("R7-supplied",)
    for k in ("6", "01", "-1", "", "x"):
        sup = {"pouw.roots": [[{"window": COIN}] + [{"position": k}] * 8] * 2}
        assert R.check(later, **{**check, "supplied": sup}).codes == ("R7-supplied",), k
    assert R.check(later, **{**check, "log": [{**first, "population": "6"}]}).codes == ("R6-previous", "R7-supplied")


def test_a_window_salt_is_the_coin_bound_to_the_stream_prover_and_window():
    """W-c: a window source is SHA-512 of `WINDOW_SALT`, the verifier's coin, its binding (its stream's digest and its
    prover's identity) and the record's window as canonical JSON: one salt a prover, stream and window, as R6 takes one
    record a window, and never the prover's. A record filled under another window or coin is `R7-supplied`, a window
    outside the canonical domain fills nothing, and neither does a verifier with no binding."""
    coin = bytes.fromhex(COIN)
    salt = G.window_salt(coin, BOUND, BINDING)
    assert salt == hashlib.sha512(b"verity/one-stage/window-salt/v1\0" + coin + b"\0" + b"5a" * 64 + b"\0" + b"b0" * 32
                                  + b"\0" + canonical_json(BOUND)).digest()
    assert salt.hex()[:64] == WINDOW_SALT_0
    assert len({salt, G.window_salt(coin, {**BOUND, "row": 8}, BINDING), G.window_salt(b"\1", BOUND, BINDING),
                G.window_salt(coin, BOUND, {**BINDING, "stream": "5b" * 64}),
                G.window_salt(coin, BOUND, {**BINDING, "prover": "b1" * 32})}) == 5
    for bad in ([], "served", {**BOUND, "row": 1.5}, {**BOUND, "run": "r\u00e9"}, {**BOUND, "row": 2**53}):
        with pytest.raises(ValueError):
            G.window_salt(coin, bad, BINDING)
    with pytest.raises(ValueError, match="no binding"):
        G.window_salt(coin, BOUND, None)
    rec, check, _ = _mixed_case(_seeded_case())
    assert R.check(rec, **check).ok
    assert R.check({**rec, "window": {**BOUND, "row": 9}}, **check).codes == ("R7-supplied",)
    other = {"pouw.roots": [[{"window": "c1" * 64} if "window" in s else s for s in r] for r in MIXED["pouw.roots"]]}
    assert R.check(rec, **{**check, "supplied": other}).codes == ("R7-supplied",)
    for bad in ({**BOUND, "row": 1.5}, {**BOUND, "run": "r\u00e9"}):
        assert R.check({**rec, "window": bad}, **check).codes == ("R7-supplied",)
    for coin in ("", "C0" * 64, "c0c"):
        sup = {"pouw.roots": [[{"window": coin}, {"window": COIN}]] * 2}
        assert R.check(rec, **{**check, "supplied": sup}).codes == ("R7-supplied",), coin


def test_a_tile_replayed_across_streams_in_one_window_is_refused():
    """Two streams, or two provers, in one window under one coin: each one's positions run from its own log, so a window
    salt of the coin and the window alone would seed a tile alike in both, and a tile proved for one could be registered
    again in the other. The salt binds the verifier's stream and prover, and R6 holds the record's window to name them:
    stream A's record (or prover P's) is refused by stream B's verifier (or prover Q's) as it stands (`R6-stream` or
    `R6-prover`, and `R7-supplied`) and relabeled to B's binding (`R7-supplied`: its seeds are A's), while B's own record
    of the same window passes with other seeds. A verifier with no binding fills no window salt, and a binding in any
    other form is refused before the record is read."""
    base = _seeded_case()
    rec, check, roots = _mixed_case(base)
    assert R.check(rec, **check).ok
    for k, theirs in (("stream", "5b" * 64), ("prover", "b1" * 32)):
        other = {**BINDING, k: theirs}
        assert R.check(rec, **{**check, "binding": other}).codes == (f"R6-{k}", "R7-supplied"), k
        relabeled = {**rec, "window": {**rec["window"], k: theirs}}
        assert R.check(relabeled, **{**check, "binding": other}).codes == ("R7-supplied",), k
        own, own_check, own_roots = _mixed_case(base, window={**BOUND, k: theirs}, binding=other)
        assert R.check(own, **own_check).ok
        assert own_roots.root != roots.root and own_roots.rows != roots.rows
    assert R.check(rec, **{**check, "binding": None}).codes == ("R7-supplied",)
    unbound = {**rec, "window": SERVED}
    assert R.check(unbound, **check).codes == ("R6-stream", "R6-prover", "R7-supplied")
    for bad in ({"stream": "5a" * 64}, {**BINDING, "run": "r"}, {**BINDING, "stream": "5A" * 64},
                {**BINDING, "stream": "5a" * 63}, {**BINDING, "prover": ""}, {**BINDING, "prover": "b0b"},
                {**BINDING, "prover": 1}, ["stream", "prover"]):
        with pytest.raises(ValueError):
            R.check(rec, **{**check, "binding": bad})


def test_r7_holds_a_pinned_value_to_its_pin():
    """W-d: a value the verifier pinned before the record (the weights' root recorded before the salt, which root(B)
    must be) is held to that root, ahead of the log's; a pin of a value it doesn't expect registered or fills itself, of
    no 64-byte lowercase root, or with no value registered is refused before the record is read."""
    rec, check, w, _ = _seeded_case()
    assert R.check(rec, **check, pinned={"W": w.root.hex()}).ok
    assert R.check(rec, **check, pinned={"W": "ee" * 64}).codes == ("R7-held",)
    later = {**rec, "window": {**SERVED, "row": 8}, "previous": R.digest(rec)}
    assert R.check(later, **check, log=[rec], held=["W"]).ok
    assert R.check(later, **check, log=[rec], pinned={"W": "ee" * 64}).codes == ("R7-held",)
    for bad in ({"V": "ee" * 64}, {"pouw.roots": "ee" * 64}, {"W": "EE" * 64}, {"W": "ee"}, {"W": 1}, ["W"]):
        with pytest.raises(ValueError):
            R.check(rec, **check, pinned=bad)
    v1 = {k: x for k, x in rec.items() if k != "registered"} | {"format": "verity/registration/v1"}
    with pytest.raises(ValueError):
        R.check(v1, **{**check, "registered": None, "supplied": None}, pinned={})


def test_r7_reuses_an_earlier_outputs_commitment():
    """A value that is an output of an earlier run (`outputs`; never NCI's baseline θ) takes that run's port: its entry
    has the port's domain and root, and the port's leaf count is the value's rows, so its rows stay under the earlier
    run's hiding leaves. The output overrides what the log or nothing would hold it to; a record or port the log doesn't
    hold, or a port of other leaves, is `R7-output`, and a malformed `outputs` is refused before the record is read."""
    prog = _program()
    reg = _registered(prog)
    first, ports = _record(prog, reg)
    part = P.partition(prog, Q)
    dom = R.window_domain(part["program"], P.digest(part), "prev.out", "hm96-sha512/row/v1", len(TABLE), "r-earlier")
    earlier = R.record(prog, Q, LAW, SCHEME, [*first["roots"], {"port": "prev.out", "root": "cd" * 64, "leaves": len(TABLE),
                                                               "domain": dom},
                                              {"port": "short", "root": "cd" * 64, "leaves": 4, "domain": dom}],
                       {**SERVED, "row": 6}, registered=[reg.entry()])
    entry = {**reg.entry(), "root": "cd" * 64, "domain": dom}
    rec = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 8}, previous=R.digest(earlier),
                   registered=[entry])
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"W": 2}, "log": [earlier]}
    out = {"W": {"record": R.digest(earlier), "port": "prev.out"}}
    assert R.check(rec, prog, Q, **kw, outputs=out).ok
    assert R.check(rec, prog, Q, **kw, held=["W"]).codes == ("R7-domain", "R7-held")
    assert R.check({**rec, "registered": [{**entry, "root": "ce" * 64}]}, prog, Q, **kw, outputs=out).codes == ("R7-held",)
    assert R.check({**rec, "registered": [reg.entry()]}, prog, Q, **kw, outputs=out).codes == ("R7-domain", "R7-held")
    for bad in ({"record": "ee" * 64, "port": "prev.out"}, {"record": R.digest(earlier), "port": "nope"},
                {"record": R.digest(earlier), "port": "short"}, {"record": R.digest(rec), "port": "prev.out"}):
        assert R.check(rec, prog, Q, **kw, outputs={"W": bad}).codes[0] == "R7-output", bad
    for bad in ({"x": out["W"]}, {"W": {**out["W"], "record": "EE" * 64}}, {"W": {**out["W"], "extra": 1}},
                {"W": {"record": R.digest(earlier)}}, {"W": {**out["W"], "port": 1}}, {"W": "prev.out"}, ["W"]):
        with pytest.raises(ValueError):
            R.check(rec, prog, Q, **kw, outputs=bad)
    with pytest.raises(ValueError):
        R.check(rec, prog, Q, **kw, outputs=out, pinned={"W": "cd" * 64})
    reads = G.reads_file(prog, Q, {"W": 2}, {"W": "cd" * 64}, domains={"W": dom})["ports"]
    assert {e["domain"] for e in reads.values()} == {dom}
    assert {e["domain"] for e in G.reads_file(prog, Q, {"W": 2}, {"W": "cd" * 64})["ports"].values()} == {reg.entry()["domain"]}


def test_r7_reuses_an_earlier_records_registered_value():
    """A value may reuse an earlier record's registered value (`outputs: {value: {"record", "value"}}`, H1's `a/<K>@<s>` as
    a Commit's AROW, say): its entry takes that entry's domain and root, with no tree of its own, under another name or the
    same. A record entry of another root is `R7-held`, of another domain `R7-domain` too; a record or value the log
    doesn't hold, or an earlier entry of other leaves, words or word bits, is `R7-output`; and a malformed `outputs` is
    refused before the record is read."""
    prog = _program()
    reg = _registered(prog)
    first, ports = _record(prog, reg)
    dom = "da" * 64
    theirs = {**reg.entry(), "value": "a/16@0", "root": "cd" * 64, "domain": dom}
    earlier = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 6},
                       registered=[{**theirs, "root": "ce" * 64}, theirs])
    entry = {**reg.entry(), "root": "cd" * 64, "domain": dom}
    rec = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 8}, previous=R.digest(earlier),
                   registered=[entry])
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"W": 2}, "log": [earlier]}
    out = {"W": {"record": R.digest(earlier), "value": "a/16@0"}}
    assert R.check(rec, prog, Q, **kw, outputs=out).ok
    assert R.check(rec, prog, Q, **kw).codes == ("R7-domain",)
    assert R.check({**rec, "registered": [{**entry, "root": "ef" * 64}]}, prog, Q, **kw, outputs=out).codes == ("R7-held",)
    assert R.check({**rec, "registered": [{**entry, "domain": "db" * 64}]}, prog, Q, **kw,
                   outputs=out).codes == ("R7-domain",)
    assert R.check({**rec, "registered": [reg.entry()]}, prog, Q, **kw, outputs=out).codes == ("R7-domain", "R7-held")
    same = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 6}, registered=[{**theirs, "value": "W"}])
    assert R.check({**rec, "previous": R.digest(same)}, prog, Q, **{**kw, "log": [same]},
                   outputs={"W": {"record": R.digest(same), "value": "W"}}).ok
    for e in ({**theirs, "leaves": 4}, {**theirs, "words": 1}, {**theirs, G.WIDTH_KEY: 32}, {**theirs, "domain": 1},
              {**theirs, "root": None}, {**theirs, "leaves": True}):
        other = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 6}, registered=[e])
        assert R.check({**rec, "previous": R.digest(other)}, prog, Q, **{**kw, "log": [other]},
                       outputs={"W": {"record": R.digest(other), "value": "a/16@0"}}).codes[0] == "R7-output", e
    for bad in ({"record": "ee" * 64, "value": "a/16@0"}, {"record": R.digest(earlier), "value": "nope"},
                {"record": R.digest(earlier), "value": "W"}, {"record": R.digest(rec), "value": "a/16@0"}):
        assert R.check(rec, prog, Q, **kw, outputs={"W": bad}).codes[0] == "R7-output", bad
    for bad in ({"W": {**out["W"], "port": "prev.out"}}, {"W": {**out["W"], "value": 1}}, {"W": {"value": "a/16@0"}},
                {"W": {**out["W"], "record": "EE" * 64}}):
        with pytest.raises(ValueError):
            R.check(rec, prog, Q, **kw, outputs=bad)


#: an AROW row table's port (`verity.ml.windows.rows.RowTable.port`: `row-table/<engine step>/<op_path>`)
ROW_TABLE = "row-table/3/model.layers.0.mlp.down_proj"


def test_pouw_roots_reads_an_arow_row_table_root():
    """AROW: a row table the record registers as a port (one `hm96-sha512/row-seg/v1` leaf per token row) is read as any
    port's root, so the verifier fills `pouw.roots` from it, and a record whose row table moved after `pouw.roots` was
    fixed is `R7-supplied`."""
    _, check, w, _ = _seeded_case()
    prog, q = check["program"], check["query"]
    part = P.partition(prog, q)
    dom = R.window_domain(part["program"], P.digest(part), ROW_TABLE, "hm96-sha512/row-seg/v1", 5, "r-registered")
    ports = {**check["ports"], ROW_TABLE: {"leaves": 5, "domain": dom}}
    sup = {"pouw.roots": [[{"port": ROW_TABLE}, {"value": "W"}]] * 2}
    check = {**check, "ports": ports, "supplied": sup}

    def record(table_root, roots=None):
        served = [{"port": p, "root": table_root if p == ROW_TABLE else "ab" * 64, **ports[p]} for p in ports]
        base = R.record(prog, q, LAW, SCHEME, served, SERVED, registered=[w.entry()])
        roots = roots or G.supplied_value(prog, "pouw.roots", G.sources(sup)["pouw.roots"], base, ["pouw.roots"])
        return R.record(prog, q, LAW, SCHEME, served, SERVED, registered=[w.entry(), roots.entry()]), roots

    rec, roots = record("5e" * 64)
    assert [list(r) for r in roots.rows] == [G.bytes_leaves(bytes.fromhex("5e" * 64) + w.root, 32)] * 2
    assert R.check(rec, **check).ok
    assert R.check(record("5f" * 64, roots)[0], **check).codes == ("R7-supplied",)


class _Program:
    """A stand-in with what `parameter_gates` and `shape` read: a parameter `W` of `n` 16-bit leaves over `refs`."""

    def __init__(self, refs, n: int | None = None) -> None:
        params = [("x", Value(8)), ("W", Array(len(refs) if n is None else n, Value(16)))]
        self.fn = type("Fn", (), {"signature": staticmethod(lambda: (params, Value(8)))})()
        self._refs = refs

    def input_coll(self, i: int):
        return type("Coll", (), {"refs": self._refs})() if i == 1 else None


def _gates_by_list(program, value: str) -> range:
    """`parameter_gates` as it was, one Python int per leaf: the oracle the run scan must equal."""
    names = [name for name, _ in program.fn.signature()[0]]
    if value not in names:
        raise ValueError(f"{value!r} is not a parameter of the program ({names})")
    gates = [ref[2] for ref in program.input_coll(names.index(value)).refs]
    if not gates or gates != list(range(gates[0], gates[0] + len(gates))):
        raise ValueError(f"parameter {value!r}: its gates are not one interval")
    return range(gates[0], gates[0] + len(gates))


def _outcome(f, program, value):
    try:
        return f(program, value)
    except ValueError as e:
        return str(e)


@pytest.mark.parametrize("refs", [
    Affine("g", 0, 7, 12),
    Concat([Affine("g", 0, 3, 4), Affine("g", 0, 7, 5)]),
    Concat([Affine("g", 0, 3, 4), Explicit((("g", 0, 7), ("g", 0, 8))), Strided("g", 0, 9, 2, 2, 1, 6)]),
    Strided("g", 0, 9, 3, 3, 1, 9),
    Explicit((("g", 0, 4), ("g", 0, 5), ("g", 0, 6))),
    Concat([Affine("g", 0, 3, 4), Affine("h", 0, 7, 2)]),
    Concat([Affine("g", 0, 3, 4), Affine("g", 1, 7, 2)]),
    Concat([Affine("g", 0, 3, 4), Affine("g", 0, 8, 2)]),
    Concat([Affine("g", 0, 7, 2), Affine("g", 0, 3, 4)]),
    Concat([Affine("g", 0, 3, 4), Affine("g", 0, 6, 2)]),
    Explicit((("g", 0, 4), ("g", 0, 6), ("g", 0, 5))),
    Explicit((("g", 0, 4), ("g", 0, 4))),
    Strided("g", 0, 9, 3, 1, 3, 9),
    Strided("g", 0, 5, 1, 0, 0, 3),
    Strided("g", 0, 9, 2, 3, 1, 6),
    Explicit(()),
], ids=["affine", "two-runs", "mixed", "strided-contiguous", "explicit", "other-space-contiguous",
        "other-idx-contiguous", "gap", "out-of-order", "overlap", "explicit-swap", "repeat", "transpose", "broadcast",
        "strided-gaps", "empty"])
def test_parameter_gates_reads_runs_as_the_leaf_list_did(refs):
    """The run scan returns the range, or refuses with the message, that the per-leaf list did, on every refs form; the
    list read only each ref's leaf, so a run in another space or idx that continues the interval is accepted by both."""
    prog = _Program(refs)
    assert _outcome(G.parameter_gates, prog, "W") == _outcome(_gates_by_list, prog, "W")
    assert _outcome(G.parameter_gates, prog, "V") == _outcome(_gates_by_list, prog, "V")


def test_parameter_gates_of_an_h200_weight_is_its_range_in_constant_memory():
    """A weight tensor's parameter at H200 width (2^30 leaves, one `Affine`) is its range and its rows without a Python int
    per leaf: the per-leaf list held ~2 x 43 GB here and was OOM-killed."""
    n = 1 << 30
    prog = _Program(Affine("g", 0, 5, n), n)
    tracemalloc.start()
    try:
        gates, rows = G.parameter_gates(prog, "W"), G.shape(prog, "W", 1024)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert gates == range(5, 5 + n) and rows == n // 1024
    assert peak < 1 << 16, peak
