"""Registered values: a program's table committed once under hiding leaves (`one_stage.registered`), the registration's
R7, the verifier's read map from its own program, the read check, and the audit record's link to the registered roots."""
from __future__ import annotations

import hashlib
import random
import re
import tracemalloc

import pytest

import verity.definitions.ops.scalar  # noqa: F401
from verity.commitments import CommitmentDomain, RangeIndexedDomain, hm96
from verity.commitments import history as HT
from verity.commitments.merkle import merkle_depth, tree_levels
from verity.commitments.rowleaf import ROLE_X, sha512_row_digest
from verity.circuits.codec import canonical_json
from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity.circuits.program import Program
from verity.circuits.refs import Affine, Concat, Explicit, Strided, coll_concat
from verity.circuits.types import Array, Value
from verity.protocols.pouw.verifier import audit as PA
from verity.protocols.pouw.verifier.protocol import tagged
from verity.protocols.spot.verifier.one_stage import (audit, draw as D, partition as P, registered as G, registration as R,
                                                 verifier as V)

XOR8 = PrimitiveDefinition("RegisteredTestXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                           register=False)
MIX = bind(CompositeDefinition("RegisteredTestMix", 0, (), lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8)))), Value(8)),
                               lambda B, S, x, w: B.call(XOR8, B.call(XOR8, x[0], w[0]), B.call(XOR8, x[1], w[1])), register=False))
#: the row of the table W each proof block reads, fixed by the program
POS = (3, 3, 0, 2, 1, 3)
TABLE = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
SCHEME = "frame-v3-sha512/hm96-sha512"
LAW = {"law": "subset", "k": 2}
#: the sample the audit record's tests hand it: units 1 and 4 of six
SAMPLE = {"law": "subset", "population": 6, "k": 2, "units": [1, 4]}
SERVED = {"kind": "served", "row": 7, "run": "r-registered"}
#: a 16-bit registration's root: three 64-word rows, salts 01.., 02.., 03..
ROOT_16 = ("511a71aa555f05f2b992997f7d6fe2af8e225c2a378ace51d54c18642f73193e"
           "5f4a2e5773666a0fc342aa1911db7806cdbcb3d144a6fd8ffd0d9c6b0426d557")
#: a 32-bit registration's root: one row of eps and the softmax scale at D = 64, salt 09..
ROOT_32 = ("c0263747d89040d789185af524ea46e535071948299eb60851799b78fd3a9168"
           "c5e79be456e606dd9eaa5516e6d5b6013536c24f1fe17ba13dd09a37150a62b0")


#: each proof block's seed group: the row of `pouw.roots` it reads
GROUPS = (0, 0, 1, 1, 1, 0)
SEEDED = bind(CompositeDefinition(
    "RegisteredTestSeeded", 0, ("BITS",),
    lambda S: ((("x", Array(2, Value(8))), ("w", Array(2, Value(8))), ("roots", Array(2, _digest(S.BITS)))), Value(8)),
    lambda B, S, x, w, roots: B.call(XOR8, B.call(XOR8, x[0], w[0]), B.call(XOR8, x[1], w[1])), register=False))
#: the verifier's sources of each seed group's roots: the served port x's root and W's registered root
SUPPLIED = {"pouw.roots": [[{"port": "x"}, {"value": "W"}], [{"value": "W"}, {"port": "x"}]]}
#: `supplied_salts("ab" * 64, "pouw.roots", 2)`' row 1, its first 32 bytes
SUPPLIED_SALT_1 = "95efb95ff108c49d1fac7d5f65d112446bf004701d688364ab226995805a68dc"


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


def test_a_16_bit_value_binds_only_the_five_keys():
    """16 is the default width: its domain binds the five keys alone, under the SHA-512 identity frame (64 bytes), and a
    whole-block row is the port's row as it stands (`ROOT_16`)."""
    from verity.commitments.identity import identity_digest_sha512

    prog = P.program_digest(_program())
    binding = identity_digest_sha512(G.REGISTERED_DOMAIN, {"program": prog, "value": "W", "schema": G.SCHEMA, "words": 2,
                                                           "leaves": 5})
    assert G.domain(prog, "W", 2, 5).domain_id == G.domain(prog, "W", 2, 5, 16).domain_id
    assert G.domain(prog, "W", 2, 5).binding == binding and len(binding) == 64
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
    from verity.commitments.rowleaf import bit_row, row_leaf, sha512_row_seg_digest, sha512_row_v2_digest

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
    from verity.commitments.gates import embedding as E

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


def test_bind_ports_binds_each_port_the_file_leaves_out_at_the_registrations_root():
    """`bind_ports` (`Flock.RegBound`): a v1 file's ports the statement has and the file names no registered port for are
    bound at the registration's root and domain of their rows, its registered ports left as they are; a file with no
    other port, or of v0, is returned as it is (Lean holds it to name every port); a port the registration holds no root
    for is refused."""
    w = {"value": "W", "root": "aa" * 64, "leaves": 2, "domain": "bb" * 64, "reads": [0, 1]}
    roots = {p: {"root": f"{i}" * 128, "domain": f"{i + 4}" * 128, "leaves": 2} for i, p in enumerate(("w", "x", "out"))}
    v1 = {"format": G.READS_V1, "ports": {"w": w}}
    got = G.bind_ports(v1, ["w", "x", "out"], roots)
    assert got == {**v1, G.BOUND: {"out": {"root": "2" * 128, "domain": "6" * 128}, "x": {"root": "1" * 128, "domain": "5" * 128}}}
    assert G.bind_ports(v1, ["w"], roots) == v1 and G.BOUND not in G.bind_ports(v1, ["w"], roots)
    v0 = {"format": G.READS, "ports": {"w": w}}
    assert G.bind_ports(v0, ["w", "x"], roots) == v0
    with pytest.raises(ValueError, match=r"ports \['y'\] are neither registered ports of the file nor"):
        G.bind_ports(v1, ["w", "x", "y"], roots)


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


def test_a_seed_salt_the_server_chose_is_refused_by_name():
    """The window's seed salt is the verifier's, read from its challenger's log (`coins.seed_salt_of`), never the
    record's: a verifier that read it accepts a window naming it and refuses by name (`R6-seed-salt`) one naming another
    or none; one that read none refuses a window naming any, a salt the server chose, and the detail names both. A salt
    is read on the window's session in one form: in another, or without the window's open root, the verifier refuses its
    own inputs."""
    prog = _program()
    w = _registered(prog)
    x = G.Registered.commit(P.program_digest(prog), "x", X_ROWS, window=WINDOWS[0])
    (rec,), kw = _windows(prog, [x], w)
    kw = {**kw, "window_root": WINDOWS[0], "held": ["W"]}
    salt, other = "5a" * 64, "5b" * 64
    named = lambda s: {**rec, "window": {**rec["window"], R.SEED_SALT: s}}  # noqa: E731
    assert R.check(named(salt), prog, Q, **kw, seed_salt=salt).ok
    assert R.check(rec, prog, Q, **kw).ok
    for s in (other, salt.upper(), None, 5):
        assert R.check(named(s), prog, Q, **kw, seed_salt=salt).codes == ("R6-seed-salt",), s
    assert R.check(rec, prog, Q, **kw, seed_salt=salt).codes == ("R6-seed-salt",)
    chosen = R.check(named(salt), prog, Q, **kw)
    assert chosen.codes == ("R6-seed-salt",)
    assert chosen.detail == {"R6-seed-salt": {"registered": salt, "verifier": None}}
    for bad in (salt.upper(), salt[:-2], salt + "\n", "", 5, bytes.fromhex(salt)):
        with pytest.raises(ValueError, match="seed salt is lowercase hex"):
            R.check(named(salt), prog, Q, **kw, seed_salt=bad)
    with pytest.raises(ValueError, match="window's open root"):
        R.check(named(salt), prog, Q, **{**kw, "window_root": None, "held": None}, seed_salt=salt)


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
    d = SAMPLE
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    args = (rec, R.receipt(rec, "verifier-test"), d, D.parse_design("subset:2"), [1, 4], 2, {}, {"lean": lean}, 0.25)
    ok = audit.audit_record(*args, session_registered={"W": reg.root.hex()})
    assert ok["accepted"] and "registered" not in ok["checks"]
    assert ok["integrity_profile"]["inputs"]["registered"] == [{k: reg.entry()[k] for k in ("value", "root", "leaves", "domain")}]
    for opened in (None, {}):
        unread = audit.audit_record(*args, session_registered=opened)
        assert unread["accepted"] and "registered" not in unread["checks"], "a registered value needs no reported read"
        assert unread["integrity_profile"] == ok["integrity_profile"]
    for opened, wrong in (({"W": "cd" * 64}, ["W"]), ({"W": reg.root.hex(), "V": "cd" * 64}, ["V"]), ({"W": None}, ["W"])):
        r = audit.audit_record(*args, session_registered=opened)
        assert not r["accepted"] and r["checks"]["registered"]["codes"] == ["registered"] and r["integrity_profile"] is None
        assert r["checks"]["registered"]["detail"] == {"registered": ["W"], "opened": sorted(opened), "wrong": wrong}
    v1 = R.record(prog, Q, LAW, SCHEME, rec["roots"], SERVED)
    plain = audit.audit_record(v1, R.receipt(v1, "verifier-test"), *args[2:])
    assert plain["accepted"] and "registered" not in plain["integrity_profile"]["inputs"]
    assert not audit.audit_record(v1, R.receipt(v1, "verifier-test"), *args[2:], session_registered={"W": reg.root.hex()})["accepted"]


def test_the_opened_roots_come_from_the_verifier_of_records_accepted_verdicts():
    """`session_registered` reads the Lean verdict's `registered` finding, only on an accepted verdict of record: a v2 audit
    passes on what the verifier opened and on a verdict that reports no read, and fails on a read reported at another
    root, by one verdict or by one of two."""
    prog = _program()
    reg = _registered(prog)
    rec, _ = _record(prog, reg)
    d = SAMPLE
    lean = {"accepted": True, "of_record": True, "digest": "dd" * 64, "registered": {"w": {"value": "W", "root": reg.root.hex()}}}
    assert audit.session_registered({"lean": lean}) == {"W": reg.root.hex()}
    for v in ({**lean, "accepted": False}, {**lean, "accepted": None}, {**lean, "of_record": False}, {**lean, "registered": None},
              {k: x for k, x in lean.items() if k != "registered"}):
        assert audit.session_registered({"lean": v}) is None, v
    other = {**lean, "registered": {"w": {"value": "W", "root": "cd" * 64}}}
    assert audit.session_registered({"lean": lean, "lean-2": other}) == {"W": None}
    assert audit.session_registered({"lean": lean, "lean-2": lean}) == {"W": reg.root.hex()}

    def decide(verdicts):
        return audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, D.parse_design("subset:2"), [1, 4], 2, {}, verdicts,
                                  0.25, session_registered=audit.session_registered(verdicts))

    assert decide({"lean": lean})["accepted"]
    assert decide({"lean": {k: x for k, x in lean.items() if k != "registered"}})["accepted"]
    for verdicts in ({"lean": other}, {"lean": lean, "lean-2": other}):
        r = decide(verdicts)
        assert not r["accepted"] and r["checks"]["registered"]["codes"] == ["registered"], verdicts
        assert r["checks"]["registered"]["detail"]["wrong"] == ["W"], verdicts


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
    head = b"verity/one-stage/supplied-salt/v1\0" + b"ab" * 64 + b"\0pouw.roots\0" + (1).to_bytes(4, "little")
    assert salts[1] == b"".join(hashlib.sha512(head + c.to_bytes(4, "big")).digest() for c in range(3))
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


def test_a_supplied_value_in_a_bit_row_format_is_its_leaves_packed():
    """Where the verifier names a supplied value's row format (`schemas`: `row/v2` or `row-seg/v1`, the format of the
    port that reads it), R7 commits each row's leaves packed into that format's words, each leaf's bits least significant
    first (`bit_words`): a `Digest` row in `row/v2` is 32 u16 words, each big-endian 64-bit word's u16 pieces low first.
    Both verifiers accept the record registering it so; the same rows in `row/v1` or in the other bit-row format are
    refused."""
    d = hashlib.sha512(b"a table's root").digest()
    assert G.bit_words(G.digest_leaves(d), 1, 16) == [(int.from_bytes(d[i:i + 8], "big") >> (16 * j)) & 0xFFFF
                                                      for i in range(0, 64, 8) for j in range(4)]
    assert G.bit_words([0x5, 0xA], 4, 8) == [0xA5] and G.bit_words([0x12345678], 32, 16) == [0x5678, 0x1234]
    for bad in (([1, 0, 1], 1, 16), ([2], 1, 1)):
        with pytest.raises(ValueError):
            G.bit_words(*bad)
    base, check, w, flat = _seeded_case(1)
    prog, sup = check["program"], G.sources(SUPPLIED)["pouw.roots"]
    bare = {**base, "registered": [w.entry()]}
    leaves = G.supplied_rows(prog, "pouw.roots", sup, bare, ["pouw.roots"])
    words = {**check["registered"], "pouw.roots": 64}
    for schema, other in ((G.SCHEMA_V2, G.SCHEMA_SEG), (G.SCHEMA_SEG, G.SCHEMA_V2)):
        roots = G.supplied_value(prog, "pouw.roots", sup, bare, ["pouw.roots"], schema=schema)
        assert roots.schema == schema and (roots.words, roots.word_bits, roots.leaves) == (64, 16, 2)
        assert [list(r) for r in roots.rows] == [G.bit_words(r, 1, 16) for r in leaves]
        rec = {**base, "registered": [w.entry(), roots.entry()]}
        mine = {**check, "registered": words, "schemas": {"pouw.roots": schema}}
        got = V.check(rec, **mine)
        assert got.ok and got.codes == (), (schema, got)
        assert R.check(rec, **{**mine, "schemas": {"pouw.roots": other}}).codes == ("R7-supplied",)
        assert not R.check(rec, **{**check, "registered": words}).ok
        assert R.check(base, **mine).codes == ("R7-words", "R7-domain", "R7-supplied")
    assert flat.schema == G.SCHEMA and R.check(base, **check).ok


def _history(body_flip: bool = False) -> tuple[dict, str, str]:
    """A history table `k_hist` of writer `W`'s three `o` strings, its one member's public body under the leaf layer
    of that body and a reader's, the window registering its root: the verifier's `history` parameter, the leaf layer
    and the root (hex)."""
    bcs = [hashlib.sha512(b"k_hist/%d" % i).digest() * 2 for i in range(3)]
    body = bytearray(b"".join(bcs))
    publics = {"W/0": bytes(body), "R/0": b"the reader's body"}
    if body_flip:
        body[-1] ^= 1
    digests = {k: hashlib.sha512(b).hexdigest() for k, b in publics.items()}
    root = HT.root(HT.domain("p", "k_hist", 64, 16), bcs, HT.placeholder(64)).hex()
    table = {"name": "k_hist", "program": "p", "writer": "W", "out": "o", "words": 64, "capacity": 16, "first": 0,
             "units": 3, "runs": [{"start": 0, "count": 3, "writer": "W", "out": "o", "radix": [3], "base": 0, "step": [1]}],
             "members": [{"key": "W/0", "base": 0, "n": 3, "ports": ["o"], "body": bytes(body).hex()}]}
    return ({"digests": digests, "tables": [table], "held": {}, "registered": {"k_hist": root}},
            HT.leaf_layer(digests), root)


@pytest.mark.slow
def test_a_history_source_is_the_root_the_verifier_computes():
    """A supplied value's source `{"history": "H/root"}` is the root the verifier computes for table H from its writer's
    public bodies the leaf layer binds (`history.roots_of`, the Lean's `HistoryTable.rootsOf`): where the served port x's
    root is that root, it fills the rows `{"port": "x"}` fills, in both verifiers. A body the leaf layer doesn't bind, a
    registered root that is not the one computed, a table or field the verifier computed none of, or a `history`
    without the verifier's leaf layer refuses the parameters in both, and the verifier's verdict is `R7-history`."""
    hist, leaf, root = _history()
    rec, check, _, _ = _seeded_case(x_root=root, leaf_layer=leaf)
    check = {**check, "leaf_layer": leaf}
    sup = {"pouw.roots": [[{"history": "k_hist/root"}, {"value": "W"}], [{"value": "W"}, {"history": "k_hist/root"}]]}
    assert R.check(rec, **check).ok
    assert R.check(rec, **check, history=hist).ok
    assert HT.resolve(HT.roots_of(leaf, hist), sup) == {
        "pouw.roots": [[{"bytes": root}, {"value": "W"}], [{"value": "W"}, {"bytes": root}]]}
    got = V.check(rec, **{**check, "supplied": sup}, history=hist)
    assert got.ok and got.codes == ()
    domain = HT.domain("p", "k_hist", 64, 16).domain_id.hex()
    assert HT.resolve(HT.roots_of(leaf, hist), {"v": [[{"history": "k_hist/domain"}]]}) == {"v": [[{"bytes": domain}]]}

    def lean(**change) -> dict:
        args = {**check, "supplied": sup, "history": hist, **change}
        params = V._params(args["program"], args["query"], args["law"], args["scheme"], args["ports"], args["leaf_layer"],
                           (), [], None, None, args["registered"], args["supplied"], history=args["history"])
        return V.lean("registration", {"params": params, "record": rec})

    assert lean().get("codes") == []
    tampered = _history(body_flip=True)[0]
    for change in ({"history": tampered},
                   {"history": {**hist, "registered": {"k_hist": "00" * 64}}},
                   {"supplied": {"pouw.roots": [[{"history": "k/root"}, {"value": "W"}]] * 2}},
                   {"supplied": {"pouw.roots": [[{"history": "k_hist/rows"}, {"value": "W"}]] * 2}},
                   {"history": {**hist, "extra": 1}},
                   {"leaf_layer": None}):
        args = {**check, "supplied": sup, "history": hist, **change}
        with pytest.raises(R.HistoryRefused):
            R.check(rec, **args)
        assert "refused" in lean(**change), change
        got = V.check(rec, **args)
        assert not got.ok and got.codes == (R.HISTORY,), (change, got)
        assert got.verifiers["lean"].keys() == {"refused"} and got.verifiers["python"].keys() == {"refused"}


def test_the_run_of_record_takes_each_registrations_history_tables(monkeypatch):
    """`run` hands the Lean's run (`Run.run`) the history tables of each registration it parses, the boundary's and,
    under a two-stage design, the interior's, as the parameters `check` hands the Lean's registration, so a reader's
    `{"history": ...}` source resolves in the run as it does at registration."""
    hist, leaf, root = _history()
    rec, check, _, _ = _seeded_case(x_root=root, leaf_layer=leaf)
    other = {**hist, "registered": {"k_hist": "11" * 64}}
    seen = []
    monkeypatch.setattr(V, "lean", lambda piece, args: seen.append((piece, args)) or {"refused": "the spy's"})
    p, q, design, ports = check["program"], check["query"], check["law"], check["ports"]
    interior = {"program": p, "query": q, "law": design, "copies": (ports, leaf, (), [], None, None, None, None, None, None,
                                                                 None),
                "history": other, "replay": p, "values": [], "record": rec, "received": "r", "read": b"", "coins": None,
                "proved_units": None, "instances": None, "served": None, "session_roots": None}
    for it, want in ((None, None), (interior, other)):
        got = V.run({}, rec, p, q, design, check["scheme"], ports, leaf, (), [], None, None, {}, [], [], "c", "r", b"", None,
                    [], {}, [], {}, 0.5, history=hist, interior=it)
        assert not got.ok and got.codes == (V.DISAGREE,)
        [(piece, args)] = seen
        seen.clear()
        assert piece == "run" and args["params"]["history"] == hist
        params = V._params(p, q, design, check["scheme"], ports, leaf, (), [], None, None, None, history=hist)
        assert args["params"] == params
        assert (args.get("interior") or {}).get("params", {}).get("history") == want


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
    """A value that is an output of an earlier run (`outputs`; never the inference-only check's θ) takes that run's port: its entry
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


#: The entry and root a verifier built from a registration it read under the deployment's key: a link's certified rows,
#: a value that is none of the program's parameters
CERTIFIED = {"link-0/in/7": {"root": "cd" * 64, "leaves": 3, "words": 64, "domain": "ef" * 64}}


def test_r7_takes_the_verifiers_certified_entries():
    """A value the verifier holds a certified entry for (`certified`) is one of the record's registered values beside
    the ones it expects registered: the record names it once, its entry is the verifier's on every key, its root
    included (`R7-certified`), and a verifier that holds no entry for it refuses the record (`R7-values`). With an entry
    and no parameter registered the verifier expects a v2 record, whose registered values are then the certified ones
    alone, and without the entry it refuses that record by its keys. A certified value that is one the verifier
    expects a parameter registered under or fills, pins, holds or takes from an earlier run, or an entry in another
    form, is refused before the record is read."""
    prog = _program()
    reg = _registered(prog)
    rec, ports = _record(prog, reg)
    seeded, check, _, _ = _seeded_case()
    design = {k: x for k, x in check.items() if k not in ("program", "query", "ports", "registered", "supplied")}
    kw = {**design, "ports": ports, "registered": {"W": 2}}
    (value, mine), = CERTIFIED.items()
    e = {"value": value, **mine}
    both = {**rec, "registered": [reg.entry(), e]}
    assert R.check(both, prog, Q, **kw, certified=CERTIFIED).ok
    assert R.check({**both, "registered": [e, reg.entry()]}, prog, Q, **kw, certified=CERTIFIED).ok
    assert R.check(both, prog, Q, **kw, certified={value: {**mine, "word_bits": 16}}).ok
    assert R.check(both, prog, Q, **kw).codes == ("R7-values",)
    assert R.check(both, prog, Q, **kw, certified={}).codes == ("R7-values",)
    assert R.check(rec, prog, Q, **kw, certified=CERTIFIED).codes == ("R7-values", "R7-certified")
    assert R.check({**rec, "registered": [reg.entry(), e, e]}, prog, Q, **kw, certified=CERTIFIED).codes == ("R7-values",)
    cases = [({"root": "ab" * 64}, ("R7-certified",)), ({"root": "xy"}, ("R7-root", "R7-certified")),
             ({"leaves": 4}, ("R7-leaves",)), ({"leaves": 3.0}, ("R7-leaves",)), ({"words": 32}, ("R7-words",)),
             ({"domain": "00" * 64}, ("R7-domain",)), ({"word_bits": 32}, ("R7-word_bits",)),
             ({"word_bits": 16}, ("R7-word_bits",)), ({"extra": 1}, ("R7-registered",))]
    for change, codes in cases:
        got = R.check({**rec, "registered": [reg.entry(), {**e, **change}]}, prog, Q, **kw, certified=CERTIFIED)
        assert got.codes == codes, change
    # the verifier's own value is held as it was, the certified one beside it
    own = {**both, "registered": [{**reg.entry(), "leaves": 4}, e]}
    assert R.check(own, prog, Q, **kw, certified=CERTIFIED).codes == ("R7-leaves",)
    assert R.check(both, prog, Q, **{**kw, "registered": {"W": 3}}, certified=CERTIFIED).codes == ("R7-registered",)
    assert R.check({**both, "registered": [reg.entry(), {**e, "root": "ab" * 64}]}, prog, Q,
                   **{**kw, "registered": {"W": 3}}, certified=CERTIFIED).codes == ("R7-registered", "R7-certified")
    # leaves of 32 bits name their width in the verifier's entry and in the record's
    wide = {value: {**mine, "word_bits": 32}}
    rec32 = {**rec, "registered": [reg.entry(), {**e, "word_bits": 32}]}
    assert R.check(rec32, prog, Q, **kw, certified=wide).ok
    assert R.check(rec32, prog, Q, **kw, certified=CERTIFIED).codes == ("R7-word_bits",)
    assert R.check(both, prog, Q, **kw, certified=wide).codes == ("R7-word_bits",)
    # a program none of whose parameters is registered: the record's registered values are the certified ones alone
    lone = _program(w=16)
    served, ports1 = _record(lone, reg)
    kw1 = {**design, "ports": ports1}
    v1 = {**{k: x for k, x in served.items() if k != "registered"}, "format": R.FORMAT}
    only = {**served, "registered": [e]}
    assert only["format"] == R.FORMAT_V2
    assert R.check(only, lone, Q, **kw1, certified=CERTIFIED).ok
    assert R.check(only, lone, Q, **kw1, registered={}, certified=CERTIFIED).ok
    assert R.check(only, lone, Q, **kw1).codes == ("keys",)
    assert R.check(v1, lone, Q, **kw1).ok and R.check(v1, lone, Q, **kw1, certified=CERTIFIED).codes == ("keys",)
    assert R.check({**only, "registered": []}, lone, Q, **kw1, certified=CERTIFIED).codes == ("R7-values", "R7-certified")
    assert R.check({**only, "registered": [{**e, "root": "ab" * 64}]}, lone, Q, **kw1,
                   certified=CERTIFIED).codes == ("R7-certified",)
    assert R.check({**only, "registered": [e, reg.entry()]}, lone, Q, **kw1, certified=CERTIFIED).codes == ("R7-values",)
    for bad in ({"W": mine}, {value: {**mine, "root": "CD" * 64}}, {value: {**mine, "root": "cd"}},
                {value: {**mine, "leaves": True}}, {value: {**mine, "words": -1}}, {value: {**mine, "words": 64.0}},
                {value: {**mine, "domain": 5}}, {value: {**mine, "word_bits": 8}}, {value: {**mine, "word_bits": True}},
                {value: {**mine, "extra": 1}}, {value: {k: x for k, x in mine.items() if k != "root"}}, {value: None},
                {value: [mine]}, [value], {7: mine}, "x"):
        with pytest.raises(ValueError):
            R.check(both, prog, Q, **kw, certified=bad)
    # a certified value is the verifier's by its entry alone: it names it nowhere else
    with pytest.raises(ValueError):
        R.check(seeded, **check, certified={"pouw.roots": mine})
    for named in ({"supplied": {value: [[{"port": "x"}]]}}, {"pinned": {value: "cd" * 64}}, {"held": [value]},
                  {"schemas": {value: G.SCHEMA_V2}}, {"outputs": {value: {"record": "ee" * 64, "port": "x"}}}):
        with pytest.raises(ValueError):
            R.check(both, prog, Q, **kw, **named, certified=CERTIFIED)


def test_the_audit_record_holds_a_certified_value_to_a_reported_read():
    """A value the verifier holds a certified entry for needs a read the accepting verdicts of record report. The record
    accepts when they report one at the registered root. It fails `certified`, naming the value, when they report none:
    no read at all, or the value's port named under a value the record does not list and at another root, which is
    none of the record's own values, so the registered roots' check never sees it. A read at another root fails
    `registered` as any registered value's does. Without the certified names the record asks no read of the value, and
    each of the unread records is accepted."""
    _, check, _, _ = _seeded_case()
    design = {k: x for k, x in check.items() if k not in ("program", "query", "ports", "registered", "supplied")}
    lone = _program(w=16)
    served, ports = _record(lone, _registered(_program()))
    (value, mine), = CERTIFIED.items()
    rec = {**served, "registered": [{"value": value, **mine}]}
    assert R.check(rec, lone, Q, **design, ports=ports, certified=CERTIFIED).ok
    d = SAMPLE
    base = {"accepted": True, "of_record": True, "digest": "dd" * 64}
    read = {**base, "registered": {"rows.0": {"value": value, "root": mine["root"]}}}
    renamed = {**base, "registered": {"rows.0": {"value": "unlisted", "root": "ab" * 64}}}

    def decide(verdicts, certified=CERTIFIED):
        return audit.audit_record(rec, R.receipt(rec, "verifier-test"), d, D.parse_design("subset:2"), [1, 4], 2, {},
                                  verdicts, 0.25, session_registered=audit.session_registered(verdicts, rec),
                                  certified=sorted(certified))

    ok = decide({"lean": read})
    assert ok["accepted"] and "certified" not in ok["checks"] and "registered" not in ok["checks"]
    assert audit.session_registered({"lean": renamed}, rec) == {}, "the record lists no such value: the read is dropped"
    for verdicts in ({"lean": base}, {"lean": {**base, "registered": {}}}, {"lean": renamed}):
        r = decide(verdicts)
        assert not r["accepted"] and r["integrity_profile"] is None and "registered" not in r["checks"], verdicts
        assert r["checks"]["certified"] == {"ok": False, "codes": ["certified"],
                                            "detail": {"certified": [value], "unread": [value]}}, verdicts
        assert decide(verdicts, certified={})["accepted"], "without the certified names nothing asks the read"
    assert decide({"lean": read, "lean-2": renamed})["accepted"], "another session's read of the value itself"
    moved = decide({"lean": {**base, "registered": {"rows.0": {"value": value, "root": "ab" * 64}}}})
    assert not moved["accepted"] and moved["checks"]["registered"]["detail"]["wrong"] == [value]
    assert "certified" not in moved["checks"]
    two = decide({"lean": renamed}, certified=[value, "other", value])
    names = sorted([value, "other", value])
    assert two["checks"]["certified"]["detail"] == {"certified": names, "unread": names}


def test_r7_holds_a_windows_history_to_the_previous_records_port():
    """A window's history (`outputs: {value: {"port": P}}`, the earlier windows' outputs it reads) is port P of the log's
    last record, the one R6 holds `previous` to, whose digest the verifier takes from the log (`last_digest`): the entry
    takes that port's domain and root, so a fresh root is `R7-held` and the value's own domain `R7-domain` too; a port of
    other leaves, a port the last record lacks, one only an older record holds, and no previous record are `R7-output`;
    and a port that isn't a string, another key, a pinned value, or no declaration at all for a value the last record
    registered under the same name and domain (`unwindowed`) is refused before the record is read."""
    prog = _program()
    reg = _registered(prog)
    first, ports = _record(prog, reg)
    dom = "da" * 64
    port = {"port": "W", "root": "cd" * 64, "leaves": len(TABLE), "domain": dom}
    last = R.record(prog, Q, LAW, SCHEME, [*first["roots"], port], {**SERVED, "row": 6}, registered=[reg.entry()])
    entry = {**reg.entry(), "root": "cd" * 64, "domain": dom}
    rec = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 8}, previous=R.digest(last),
                   registered=[entry])
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"W": 2}, "log": [last]}
    hist = {"W": {"port": "W"}}
    assert R.last_digest([first, last]) == R.digest(last) and R.last_digest([]) == ""
    assert R.check(rec, prog, Q, **kw, outputs=hist).ok
    assert R.check(rec, prog, Q, **kw, outputs={"W": {"record": R.digest(last), "port": "W"}}).ok
    with pytest.raises(ValueError):
        R.check(rec, prog, Q, **kw)
    assert R.check({**rec, "registered": [{**entry, "root": "ce" * 64}]}, prog, Q, **kw, outputs=hist).codes == ("R7-held",)
    assert R.check({**rec, "registered": [reg.entry()]}, prog, Q, **kw, outputs=hist).codes == ("R7-domain", "R7-held")
    assert R.check(rec, prog, Q, **kw, outputs={"W": {"port": "nope"}}).codes[0] == "R7-output"
    short = R.record(prog, Q, LAW, SCHEME, [*first["roots"], {**port, "leaves": 4}], {**SERVED, "row": 6},
                     registered=[reg.entry()])
    assert R.check({**rec, "previous": R.digest(short)}, prog, Q, **{**kw, "log": [short]},
                   outputs=hist).codes[0] == "R7-output"
    after = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 7}, previous=R.digest(last),
                     registered=[reg.entry()])
    assert R.check({**rec, "previous": R.digest(after)}, prog, Q, **{**kw, "log": [last, after]},
                   outputs=hist).codes[0] == "R7-output"
    assert R.check({**rec, "previous": R.digest(after)}, prog, Q, **{**kw, "log": [last, after]},
                   outputs={"W": {"record": R.digest(last), "port": "W"}}).ok
    assert R.check({**rec, "previous": None}, prog, Q, **{**kw, "log": []}, outputs=hist).codes[0] == "R7-output"
    for bad in ({"W": {"port": 1}}, {"W": {"port": "W", "value": "W"}}, {"W": {}}):
        with pytest.raises(ValueError):
            R.check(rec, prog, Q, **kw, outputs=bad)
    with pytest.raises(ValueError):
        R.check(rec, prog, Q, **kw, outputs=hist, pinned={"W": "cd" * 64})


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


# -- the pool chain ----------------------------------------------------------------------------------------------------

#: a pool's root as PoolWindow registers its `pin` and `pout`: one row, a SHA-512 digest of 32-bit leaves
POOL_ROOT = Array(1, _digest(32))
#: the empty pool's root, which a chain restarts from
GENESIS = hashlib.sha512(b"the empty pool").hexdigest()
#: each window's closing pool, a row of 16 words, and its open root
POOLS = [[(0x01000193 * (w + 1) + i) & 0xFFFFFFFF for i in range(16)] for w in range(3)]
POOL_WINDOWS = (*WINDOWS, "c3" * 64)


def _pooled() -> Program:
    """Units that read a row of x and of the weights W, in a program whose window opens a pool from `pin` and leaves it
    in `pout`, which no unit here reads."""
    def sig(S):
        return (("x", Array(len(POS), Array(2, Value(8)))), ("W", Array(len(TABLE), Array(2, Value(8)))),
                ("pin", POOL_ROOT), ("pout", POOL_ROOT)), Array(len(POS), Value(8))

    def body(B, S, x, W, pin, pout):
        return coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p]) for i, p in enumerate(POS)])

    return Program(bind(CompositeDefinition("RegisteredTestPooledProgram", 0, (), sig, body, register=False)))


def _pool(log, reset=False, week=0, verdicts=None, lag=1) -> tuple[dict, dict]:
    """The verifier's pool, of the lag `lag`, the verdict log `verdicts` (none by default), the reset `reset` and the
    week `week`, and the pool fields a window after `log` names under it: the same reset and week, and the spent mark
    R6 holds it to."""
    pool = {"pin": "pin", "pout": "pout", "genesis": GENESIS, "lag": lag, "verdicts": verdicts or {}, "reset": reset,
            "week": week}
    return pool, {R.POOL_RESET: reset, R.POOL_WEEK: week, R.POOL_SPENT: reset or R.pool_spent_before(pool, log)}


def _pool_window(prog: Program, w: G.Registered, i: int, log, pin=None, reset=False, week=0, verdicts=None, lag=1,
                 **window):
    """Window `i`'s record of `_pooled`, the next after `log`: x and `pout` (`POOLS[i]`) afresh under its window's open
    root, the weights `w` held, `pin` the entry `pin`, the empty pool filled from `genesis` by default, and its window
    naming the pool fields `_pool` gives, then the fields `window`; with the verifier's R7 arguments and its pool
    (`_pool`)."""
    d = P.program_digest(prog)
    base, ports = _record(prog, w)
    x = G.Registered.commit(d, "x", X_ROWS, window=POOL_WINDOWS[i])
    pout = G.Registered.commit(d, "pout", [POOLS[i]], word_bits=32, window=POOL_WINDOWS[i])
    pool, marks = _pool(log, reset, week, verdicts, lag)

    def record(pin_entry):
        return R.record(prog, Q, LAW, SCHEME, base["roots"], {**SERVED, "row": 7 + i, **marks, **window},
                        previous=R.digest(log[-1]) if log else None,
                        registered=[x.entry(), w.entry(), pout.entry(), *([pin_entry] if pin_entry else [])])

    if pin is None:
        pin = G.supplied_value(prog, "pin", [[("bytes", GENESIS)]], record(None), ["pin"]).entry()
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"x": 2, "W": 2, "pin": 16, "pout": 16},
          "log": list(log), "window_root": POOL_WINDOWS[i], "held": ["W"], "pool": pool}
    return record(pin), kw


def _entry(rec, value: str) -> dict:
    return next(e for e in rec["registered"] if e["value"] == value)


def test_a_window_opens_the_pool_the_previous_window_of_record_left():
    """Pool chain: a registration's first window opens the empty pool by the public rule, with no reset of its own,
    `pin` filled from `genesis` under a fixed domain, and each next window the pool the previous window of record left,
    `pin` the previous record's `pout` entry, its domain and root (the window's open root binds `pout`'s domain, so no
    other window's pool opens under it), whatever the verifier made of that window. A wrong `pin` is refused by name
    (`R7-pool-chain`): another pool under the window's own domain, the empty pool, or the window's own `pout`."""
    prog = _pooled()
    w = _registered(prog)
    first, kw = _pool_window(prog, w, 0, [])
    assert R.check(first, prog, Q, **kw).ok
    fixed = G.Registered.commit(P.program_digest(prog), "pin", [POOLS[0]], word_bits=32).entry()
    assert _entry(first, "pin")["domain"] == fixed["domain"]
    carried = {**_entry(first, "pout"), "value": "pin"}
    second, kw2 = _pool_window(prog, w, 1, [first], carried)
    assert R.pool_chains(kw2["pool"], [first], prog)
    assert R.check(second, prog, Q, **kw2).ok
    third, kw3 = _pool_window(prog, w, 2, [first, second], {**_entry(second, "pout"), "value": "pin"})
    assert R.check(third, prog, Q, **kw3).ok
    other = G.Registered.commit(P.program_digest(prog), "pin", [POOLS[2]], word_bits=32, window=POOL_WINDOWS[1])
    for pin, codes in (({**carried, "root": other.root.hex()}, ("R7-held", "R7-pool-chain")),
                       (other.entry(), ("R7-domain", "R7-held", "R7-pool-chain")),
                       (_entry(first, "pin"), ("R7-domain", "R7-held", "R7-pool-chain")),
                       ({**_entry(second, "pout"), "value": "pin"}, ("R7-domain", "R7-held", "R7-pool-chain")),
                       ({**carried, "domain": _entry(second, "pout")["domain"]}, ("R7-domain", "R7-pool-chain"))):
        wrong, _ = _pool_window(prog, w, 1, [first], pin)
        assert R.check(wrong, prog, Q, **kw2).codes == codes, pin
    empty, _ = _pool_window(prog, w, 1, [first])
    assert R.check(empty, prog, Q, **kw2).codes == ("R7-domain", "R7-held", "R7-pool-chain")


def test_the_public_rule_restarts_the_pool_with_no_advice():
    """The public rule opens the empty pool, `pin` filled from `genesis`, where there is no pool to carry, at a
    registration's first window and past a registration boundary (a previous record of another program), and after the
    verifier refused the window the public lag back, as its own verdict log names it (`verdicts`, `pool_refused`), with
    no reset of serving's: the records name none, so the rule costs no advice. Where it restarts, the previous window's
    pool is refused by name (`R7-pool-chain`). Where the verdict log names that window's run accepted, or holds no
    verdict for it, or the log is shorter than the lag, the window carries the pool. A malformed pool refuses the
    verifier's own inputs."""
    prog = _pooled()
    w = _registered(prog)
    first, kw = _pool_window(prog, w, 0, [])
    elsewhere, _ = _record(_program(), _registered(_program()))
    for log in ([], [elsewhere]):
        opened, kw2 = _pool_window(prog, w, 1, log)
        assert not R.pool_chains(kw2["pool"], log, prog)
        assert R.check(opened, prog, Q, **kw2).ok
    refused = {R.digest(first): False}
    after, kw3 = _pool_window(prog, w, 1, [first], verdicts=refused)
    assert R.pool_refused(kw3["pool"], [first]) and not R.pool_chains(kw3["pool"], [first], prog)
    assert R.check(after, prog, Q, **kw3).ok
    carry = {**_entry(first, "pout"), "value": "pin"}
    carried, _ = _pool_window(prog, w, 1, [first], carry, verdicts=refused)
    assert R.check(carried, prog, Q, **kw3).codes == ("R7-domain", "R7-supplied", "R7-pool-chain")
    for verdicts, lag in (({R.digest(first): True}, 1), ({GENESIS: False}, 1), (refused, 2)):
        on, kw4 = _pool_window(prog, w, 1, [first], carry, verdicts=verdicts, lag=lag)
        assert R.pool_chains(kw4["pool"], [first], prog), (verdicts, lag)
        assert R.check(on, prog, Q, **kw4).ok, (verdicts, lag)
    second, _ = _pool_window(prog, w, 1, [first], carry)
    late, kw5 = _pool_window(prog, w, 2, [first, second], verdicts=refused, lag=2)
    assert not R.pool_chains(kw5["pool"], [first, second], prog)
    assert R.check(late, prog, Q, **kw5).ok
    pool = kw["pool"]
    missing = ({k: v for k, v in pool.items() if k != b} for b in ("lag", "reset", "verdicts", "week"))
    for bad, change in (*((m, {}) for m in missing), ({**pool, "pin": "pout"}, {}), ({**pool, "pin": "V"}, {}),
                        ({**pool, "genesis": GENESIS.upper()}, {}), ({**pool, "genesis": GENESIS[:-2]}, {}),
                        ({**pool, "lag": 0}, {}), ({**pool, "lag": True}, {}), ({**pool, "week": -1}, {}),
                        ({**pool, "week": False}, {}), ({**pool, "verdicts": []}, {}),
                        ({**pool, "verdicts": {GENESIS[:-2]: False}}, {}), ({**pool, "verdicts": {GENESIS: 0}}, {}),
                        ({**pool, "reset": 1}, {}), ({**pool, "refused": False}, {}), ({**pool, "extra": False}, {}),
                        (pool, {"supplied": {"pin": [[{"bytes": GENESIS}]]}}),
                        (pool, {"pinned": {"pout": "ab" * 64}}), (pool, {"schemas": {"pin": G.SCHEMA_SEG}})):
        with pytest.raises(ValueError):
            R.check(first, prog, Q, **{**kw, "pool": bad, **change})


def test_serving_resets_its_pool_at_most_once_a_week():
    """Serving's own reset, as an engine crash needs, opens the empty pool at any window, and the record's window names
    it; the previous window's pool is then refused by name (`R7-pool-chain`). The reset spends the week's: it and each
    window after it in the week name it spent (`pool_spent`), and R6 refuses a reset where the week's is spent before
    it, by name (`R6-pool-reset-week`), whatever the verdicts. The next week starts unspent and may reset again. So
    the records name at most one window of a week's n, log2(n + 1) bits a week of advice. A window that doesn't reset
    carries the pool whether or not the week's reset is spent."""
    prog = _pooled()
    w = _registered(prog)
    first, _ = _pool_window(prog, w, 0, [])
    restart, kw = _pool_window(prog, w, 1, [first], reset=True)
    assert not R.pool_chains(kw["pool"], [first], prog) and restart["window"][R.POOL_SPENT] is True
    assert R.check(restart, prog, Q, **kw).ok
    stale, _ = _pool_window(prog, w, 1, [first], {**_entry(first, "pout"), "value": "pin"}, reset=True)
    assert R.check(stale, prog, Q, **kw).codes == ("R7-domain", "R7-supplied", "R7-pool-chain")
    log = [first, restart]
    again, kw2 = _pool_window(prog, w, 2, log, reset=True)
    assert R.check(again, prog, Q, **kw2).codes == ("R6-pool-reset-week",)
    after, kw3 = _pool_window(prog, w, 2, log, reset=True, verdicts={R.digest(restart): False})
    assert R.check(after, prog, Q, **kw3).codes == ("R6-pool-reset-week",)
    kept, kw4 = _pool_window(prog, w, 2, log, {**_entry(restart, "pout"), "value": "pin"})
    assert R.pool_chains(kw4["pool"], log, prog) and kept["window"][R.POOL_SPENT] is True
    assert R.check(kept, prog, Q, **kw4).ok
    next_week, kw5 = _pool_window(prog, w, 2, log, reset=True, week=1)
    assert R.check(next_week, prog, Q, **kw5).ok


#: the window fields a verifier with no pool holds a record to name none of
UNPOOLED = {R.POOL_RESET: None, R.POOL_WEEK: None, R.POOL_SPENT: None}


def test_a_window_names_its_pools_reset():
    """R6 holds the record's window to name the reset the verifier's pool holds (`pool_reset`), a bool, and none for a
    verifier with no pool: a window naming the other, none or another value is refused by name (`R6-pool-reset`),
    beside whatever the verifier's own reset gives R7."""
    prog = _pooled()
    w = _registered(prog)
    first, kw = _pool_window(prog, w, 0, [])
    carried = {**_entry(first, "pout"), "value": "pin"}
    _, kw2 = _pool_window(prog, w, 1, [first], carried)
    _, kw3 = _pool_window(prog, w, 1, [first], reset=True)
    for named in (True, None, 1, "true"):
        rec, _ = _pool_window(prog, w, 0, [], pool_reset=named)
        assert R.check(rec, prog, Q, **kw).codes == ("R6-pool-reset",), named
        rec, _ = _pool_window(prog, w, 1, [first], carried, pool_reset=named)
        assert R.check(rec, prog, Q, **kw2).codes == ("R6-pool-reset",), named
    for named in (False, None, 0, "false"):
        rec, _ = _pool_window(prog, w, 1, [first], reset=True, pool_reset=named)
        assert R.check(rec, prog, Q, **kw3).codes == ("R6-pool-reset",), named
    fresh = G.Registered.commit(P.program_digest(prog), "pin", [POOLS[1]], word_bits=32, window=POOL_WINDOWS[1]).entry()
    plain = {k: v for k, v in kw2.items() if k != "pool"}
    unnamed, _ = _pool_window(prog, w, 1, [first], fresh, **UNPOOLED)
    assert R.check(unnamed, prog, Q, **plain).ok
    for named in (False, True):
        rec, _ = _pool_window(prog, w, 1, [first], fresh, **{**UNPOOLED, R.POOL_RESET: named})
        assert R.check(rec, prog, Q, **plain).codes == ("R6-pool-reset",), named


def test_a_window_names_its_week_and_whether_its_weeks_reset_is_spent():
    """R6 holds the record's window to name its week on the verifier's clock (`pool_week`, `R6-pool-week`) and whether
    the week's reset is spent at it (`pool_spent`, `R6-pool-spent`): that it resets, or that the previous record, of
    its week, named it spent. So a week's windows from its reset on name it spent, the next week's first window names
    it unspent, and a window naming another mark, another week or none is refused by name. A verifier with no pool
    holds the window to name neither."""
    prog = _pooled()
    w = _registered(prog)
    first, kw = _pool_window(prog, w, 0, [], reset=True)
    assert first["window"][R.POOL_SPENT] is True and R.check(first, prog, Q, **kw).ok
    carry = {**_entry(first, "pout"), "value": "pin"}
    second, kw2 = _pool_window(prog, w, 1, [first], carry)
    assert second["window"][R.POOL_SPENT] is True and R.check(second, prog, Q, **kw2).ok
    carry2 = {**_entry(second, "pout"), "value": "pin"}
    third, kw3 = _pool_window(prog, w, 2, [first, second], carry2, week=1)
    assert third["window"][R.POOL_SPENT] is False and R.check(third, prog, Q, **kw3).ok
    for named in (False, None, 1, "true"):
        rec, _ = _pool_window(prog, w, 1, [first], carry, pool_spent=named)
        assert R.check(rec, prog, Q, **kw2).codes == ("R6-pool-spent",), named
    for named in (True, None, 0, "false"):
        rec, _ = _pool_window(prog, w, 2, [first, second], carry2, week=1, pool_spent=named)
        assert R.check(rec, prog, Q, **kw3).codes == ("R6-pool-spent",), named
    for named in (1, None, False, "0"):
        rec, _ = _pool_window(prog, w, 1, [first], carry, pool_week=named)
        assert R.check(rec, prog, Q, **kw2).codes == ("R6-pool-week",), named
    fresh = G.Registered.commit(P.program_digest(prog), "pin", [POOLS[1]], word_bits=32, window=POOL_WINDOWS[1]).entry()
    plain = {k: v for k, v in kw2.items() if k != "pool"}
    for field, code in ((R.POOL_WEEK, "R6-pool-week"), (R.POOL_SPENT, "R6-pool-spent")):
        for named in (0, False, True):
            rec, _ = _pool_window(prog, w, 1, [first], fresh, **{**UNPOOLED, field: named})
            assert R.check(rec, prog, Q, **plain).codes == (code,), (field, named)


def test_a_hit_on_a_page_an_older_window_wrote_reads_it_through_the_carried_copies():
    """The carried pages' copy chain: each window recommits at its edge the pages it carries on, those it wrote and
    those it copied from the previous window, as its port `W`, and reads the previous window's as its history
    (`outputs: {"W": {"port": "W"}}`), so a hit in window w + 2 on a page window w wrote reads it through window
    w + 1's copy, under the rules R7 already has. Window w's own port, two windows back, is no history of window
    w + 2's (`R7-domain`, `R7-held`), and where window w + 1 kept no copy there is no port to read (`R7-output`)."""
    prog = _program()
    reg = _registered(prog)
    first, ports = _record(prog, reg)
    hist = {"W": {"port": "W"}}

    def window(i, log, pages=None, copies=True):
        port = {"port": "W", "root": f"c{i}" * 64, "leaves": len(TABLE), "domain": f"d{i}" * 64}
        read = {**reg.entry(), "root": pages["root"], "domain": pages["domain"]} if pages else reg.entry()
        rec = R.record(prog, Q, LAW, SCHEME, [*first["roots"], *([port] if copies else [])], {**SERVED, "row": 6 + i},
                       previous=R.digest(log[-1]) if log else None, registered=[read])
        own = {"W": {"leaves": port["leaves"], "domain": port["domain"]}} if copies else {}
        return rec, port, {"law": LAW, "scheme": SCHEME, "ports": {**ports, **own}, "registered": {"W": 2},
                           "log": list(log)}

    w0, p0, kw0 = window(0, [])
    assert R.check(w0, prog, Q, **kw0).ok
    w1, p1, kw1 = window(1, [w0], p0)
    assert R.check(w1, prog, Q, **kw1, outputs=hist).ok
    w2, _, kw2 = window(2, [w0, w1], p1)
    assert R.check(w2, prog, Q, **kw2, outputs=hist).ok
    skip, _, _ = window(2, [w0, w1], p0)
    assert R.check(skip, prog, Q, **kw2, outputs=hist).codes == ("R7-domain", "R7-held")
    bare, _, _ = window(1, [w0], p0, copies=False)
    w2b, _, kw2b = window(2, [w0, bare], p0)
    assert R.check(w2b, prog, Q, **kw2b, outputs=hist).codes[0] == "R7-output"


# -- cut and continue ---------------------------------------------------------------------------------------------------

#: three windows' arrivals as the network certifier saw them: each one's id, the values it arrives with, the window it
#: arrives in, and the outputs the certifier released of it in each window from then on. It closes in its last window,
#: and at the edge of each window before that it is open: it is cut there and continues in the next window.
ARRIVALS = ((7, (11, 12, 13, 14, 15), 0, ((21, 22, 23), (24, 25))), (8, (31, 32), 0, ((41,),)),
            (9, (51, 52, 53), 0, ((), (61,), (62,))), (10, (71,), 1, ((), (72, 73))))
#: a window's arrival slots, and the values of an arrival's row: those it arrived with, then its outputs, zero past them
SLOTS, VALUES = 4, 10


def _cut_rows(w: int, served=None) -> tuple[list[list[int]], list[list[int]]]:
    """Window `w`'s certified arrivals (`ing`: each slot's valid bit, id and input length) and its rows (`req`: each
    slot's outputs in the window, its cut bit, then its values), from `ARRIVALS`. The arrivals open at window w − 1's
    edge come first, in release order, each a fresh arrival whose input is the cut one's and the e outputs the
    certifier released of it, so its length is P + e. `served`: for an id, how many outputs serving had computed past
    those released when it was cut, which a continuation counted from serving (P + n) takes into its input."""
    continued, arriving = [], []
    for ident, given, at, outs in ARRIVALS:
        k = w - at
        if not 0 <= k < len(outs):
            continue
        extra = (served or {}).get(ident, 0) if k else 0
        head = [*given, *(t for o in outs[:k] for t in o), *outs[k][:extra]]
        values = [*head, *outs[k][extra:]]
        row = [len(outs[k]) - extra, int(k + 1 < len(outs)), *values, *[0] * (VALUES - len(values))]
        (continued if k else arriving).append(([1, ident, len(head)], row))
    rows = continued + arriving
    rows += [([0, 0, 0], [0] * (2 + VALUES))] * (SLOTS - len(rows))
    return [a for a, _ in rows], [r for _, r in rows]


def _continued() -> Program:
    """`_pooled` with the window's certified arrivals `ing` and its rows `req`, which no unit here reads."""
    def sig(S):
        return (("x", Array(len(POS), Array(2, Value(8)))), ("W", Array(len(TABLE), Array(2, Value(8)))),
                ("ing", Array(SLOTS, Array(3, Value(16)))), ("req", Array(SLOTS, Array(2 + VALUES, Value(16)))),
                ("pin", POOL_ROOT), ("pout", POOL_ROOT)), Array(len(POS), Value(8))

    def body(B, S, x, W, ing, req, pin, pout):
        return coll_concat(sig(S)[1], [B.call(MIX, x[i], W[p]) for i, p in enumerate(POS)])

    return Program(bind(CompositeDefinition("RegisteredTestContinuedProgram", 0, (), sig, body, register=False)))


def _cut_window(prog: Program, w: G.Registered, i: int, log, pin=None, ing=None, verdicts=None, reset=False):
    """Window `i`'s record of `_continued`, the next after `log`: x, `req` (`_cut_rows(i)`'s) and `pout` (`POOLS[i]`)
    afresh under its window's open root, the weights `w` held, `ing` the certifier's commitment of `_cut_rows(i)`'s
    arrivals (or the entry `ing`), `pin` the entry `pin`, the empty pool filled from `genesis` by default, and its
    window naming the pool fields `_pool` gives for `verdicts` and `reset`. With the verifier's R7 arguments: the
    certifier's root of the window's arrivals pinned, and its pool (`_pool`)."""
    d = P.program_digest(prog)
    pool, marks = _pool(log, reset, verdicts=verdicts)
    base, ports = _record(prog, w)
    arrivals, rows = _cut_rows(i)
    stamped = G.Registered.commit(d, "ing", arrivals)
    entries = [G.Registered.commit(d, "x", X_ROWS, window=POOL_WINDOWS[i]).entry(), w.entry(), ing or stamped.entry(),
               G.Registered.commit(d, "req", rows, window=POOL_WINDOWS[i]).entry(),
               G.Registered.commit(d, "pout", [POOLS[i]], word_bits=32, window=POOL_WINDOWS[i]).entry()]

    def record(pin_entry):
        return R.record(prog, Q, LAW, SCHEME, base["roots"], {**SERVED, "row": 7 + i, **marks},
                        previous=R.digest(log[-1]) if log else None,
                        registered=[*entries, *([pin_entry] if pin_entry else [])])

    if pin is None:
        pin = G.supplied_value(prog, "pin", [[("bytes", GENESIS)]], record(None), ["pin"]).entry()
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports,
          "registered": {"x": 2, "W": 2, "ing": 3, "req": 2 + VALUES, "pin": 16, "pout": 16},
          "pinned": {"ing": stamped.root.hex()}, "log": list(log), "window_root": POOL_WINDOWS[i], "held": ["W"],
          "pool": pool}
    return record(pin), kw


def test_an_arrival_open_at_a_windows_edge_continues_in_the_next_from_the_pool_it_left():
    """Cut and continue: an arrival open at a window's edge is cut there, at the e outputs the network certifier released
    of it, and continues in the next window as a fresh arrival, (its id, P + e), whose row is the values it arrived with
    then those outputs, first among that window's arrivals in release order. The record needs nothing new for it. The
    window's arrivals are a certified input, the root the certifier signed for the window, which the verifier pins
    (`ing`); its rows (`req`: each arrival's outputs in the window and its cut bit) are its own values, under its open
    root; and its pool chains, `pin` the previous window's `pout`. So window 0 cuts two arrivals, windows 1 and 2 each
    open the pool the one before left and serve the continuations, and a continuation counted from the outputs serving
    computed (P + n) instead of those released, or the previous window's arrivals again, is not the certifier's root
    (`R7-held`). A window opening the empty pool where it carries the pool is refused by name (`R7-pool-chain`)."""
    prog = _continued()
    w = _registered(prog)
    first, kw = _cut_window(prog, w, 0, [])
    assert R.check(first, prog, Q, **kw).ok
    second, kw2 = _cut_window(prog, w, 1, [first], {**_entry(first, "pout"), "value": "pin"})
    assert R.check(second, prog, Q, **kw2).ok
    third, kw3 = _cut_window(prog, w, 2, [first, second], {**_entry(second, "pout"), "value": "pin"})
    assert R.check(third, prog, Q, **kw3).ok
    (ing0, req0), (ing1, req1), (ing2, req2) = (_cut_rows(i) for i in range(3))
    assert ing0[:3] == [[1, 7, 5], [1, 8, 2], [1, 9, 3]] and [r[:2] for r in req0[:3]] == [[3, 1], [1, 0], [0, 1]]
    assert ing1[:3] == [[1, 7, 8], [1, 9, 3], [1, 10, 1]] and [r[:2] for r in req1[:3]] == [[2, 0], [1, 1], [0, 1]]
    assert ing2[:2] == [[1, 9, 4], [1, 10, 1]] and [r[:2] for r in req2[:2]] == [[1, 0], [2, 0]]
    assert req1[0][2:10] == req0[0][2:10] and req2[0][2:6] == req1[1][2:6]
    served = _cut_rows(1, {7: 1})[0]
    assert served[0] == [1, 7, 9]
    for ing in (_entry(first, "ing"), G.Registered.commit(P.program_digest(prog), "ing", served).entry()):
        wrong, kw_wrong = _cut_window(prog, w, 1, [first], {**_entry(first, "pout"), "value": "pin"}, ing)
        assert R.check(wrong, prog, Q, **kw_wrong).codes == ("R7-held",)
    empty, kw_empty = _cut_window(prog, w, 1, [first])
    assert R.check(empty, prog, Q, **kw_empty).codes == ("R7-domain", "R7-held", "R7-pool-chain")


def test_a_window_that_restarts_its_pool_still_serves_the_continuations():
    """A window that restarts its pool, by the public rule after the verifier refused the window the public lag back or
    by serving's own reset, opens the empty pool (`genesis`) and still serves the continuations of the arrivals open at
    the previous window's edge: its certified arrivals list them first, at the outputs released through that window, as
    when it carries the pool. They find the pool empty and recompute what it held of them, which costs more and is
    correct. The previous window's pool is then refused by name (`R7-pool-chain`), and the arrivals are held to the
    certifier's root as in any window."""
    prog = _continued()
    w = _registered(prog)
    first, _ = _cut_window(prog, w, 0, [])
    second, _ = _cut_window(prog, w, 1, [first], {**_entry(first, "pout"), "value": "pin"})
    assert _cut_rows(2)[0][:2] == [[1, 9, 4], [1, 10, 1]]
    for how in ({"verdicts": {R.digest(second): False}}, {"reset": True}):
        restart, kw = _cut_window(prog, w, 2, [first, second], **how)
        assert R.check(restart, prog, Q, **kw).ok, how
        carried, kw2 = _cut_window(prog, w, 2, [first, second], {**_entry(second, "pout"), "value": "pin"}, **how)
        assert R.check(carried, prog, Q, **kw2).codes == ("R7-domain", "R7-supplied", "R7-pool-chain"), how
        stale, kw3 = _cut_window(prog, w, 2, [first, second], ing=_entry(second, "ing"), **how)
        assert R.check(stale, prog, Q, **kw3).codes == ("R7-held",), how


#: Pearl+'s y trees' binding tag: PoUW's rows domain over a call's served y, at salt nonce 2^61 + the call's index
PEARL_Y_ROWS = "verity/pouw/pearl-plus/y-rows/v1"
PEARL_SALT, PEARL_KEY, PEARL_CALL = bytes(range(64)), bytes(range(32)), 7


def _pearl_plus(rows, y: bool = False):
    """A Pearl+ call's tree of `rows` (each a row of FP32 words) as PoUW commits it, `audit.commit_rows` under
    hm96-sha512: A's under `activation_binding` at salt nonce u, y's under `PEARL_Y_ROWS` at 2^61 + u. With the domain
    as a verifier rebuilds it from that binding, owner and rows, and `bc(i, row)`: one-stage's `b || c` of `row` as a
    `row-seg/v1` row of 32-bit words under row i's salt."""
    u = PEARL_CALL
    binding = tagged(PEARL_Y_ROWS, PEARL_SALT, u) if y else PA.activation_binding(PEARL_SALT, u)
    nonce = (1 << 61) + u if y else u
    tree = PA.commit_rows(binding, u, [b"".join(w.to_bytes(4, "little") for w in r) for r in rows], PA.HM96,
                          salt_key=PEARL_KEY, nonce=nonce)
    dom = CommitmentDomain(binding, u, RangeIndexedDomain(0, len(rows)), hash="sha512")
    assert dom.domain_id == tree.domain.domain_id

    def bc(i: int, row) -> bytes:
        return hm96.SHA512.commit_string(hm96.SHA512.default_key, G.row_digest(row, 32, G.SCHEMA_SEG),
                                         PA.hm96_salt(PEARL_KEY, nonce, i))
    return tree, dom, bc


def test_one_stage_reads_open_a_pearl_plus_tree_under_its_own_domain():
    """A value committed outside one-stage, under PoUW's rows domain: Pearl+'s A (a call's x widened to FP32, K words a
    row, zero rows up to 16) and y (its served y widened to FP32, N words a row). One-stage's `row-seg/v1` leaves of
    32-bit words under that domain are Pearl+'s leaves, so its reads open Pearl+'s root at every row. A row with one
    word flipped opens nothing, nor does a row read as `row/v2`, nor any row under one-stage's own domain for the value
    (`registered.domain`, whichever binding it takes)."""
    rnd = random.Random(7)

    def rows(n: int, words: int, real: int) -> list[list[int]]:
        return [[rnd.getrandbits(16) << 16 for _ in range(words)] if i < real else [0] * words for i in range(n)]

    for y, value, rs in ((False, "pearl-a/0/0", rows(16, 2048, 5)), (True, "pearl-y/0/0", rows(5, 3072, 5))):
        tree, dom, bc = _pearl_plus(rs, y)
        root, bcs = tree.commitment.root, [bc(i, r) for i, r in enumerate(rs)]
        paths = [b"".join(tree.open(i).path) for i in range(len(rs))]
        assert tree_levels(dom, [G.leaf(dom, i, b, G.SCHEMA_SEG) for i, b in enumerate(bcs)])[-1][0] == root, value
        assert all(G.verify_read(dom, root, i, b, paths[i], G.SCHEMA_SEG) for i, b in enumerate(bcs)), value
        flipped = [rs[0][0] ^ 1 << 16, *rs[0][1:]]
        assert not G.verify_read(dom, root, 0, bc(0, flipped), paths[0], G.SCHEMA_SEG), value
        assert not G.verify_read(dom, root, 0, bcs[0], paths[0], G.SCHEMA_V2), value
        own = G.domain("ab" * 64, value, len(rs[0]), len(rs), 32)
        assert own.domain_id != dom.domain_id
        assert not any(G.verify_read(own, root, i, b, paths[i], G.SCHEMA_SEG) for i, b in enumerate(bcs)), value


def test_r7_reuses_a_pearl_plus_entry_under_its_own_domain():
    """Pearl+'s window record W registers each call's A rows and y under PoUW's rows domain, not one-stage's. A value
    reusing W's entry (`outputs`) as `row-seg/v1` rows of 32-bit words is held to that domain and root verbatim, A's and
    y's alike. The record's entry under one-stage's own domain for the value is `R7-domain` (`registered.domain`,
    whichever binding it takes), under another root `R7-held`; W's entry of 16-bit words, or of the call's real rows
    rather than its 16, is `R7-output`. R7 counts the value's rows alike under `row/v2`, so the row format binds where a
    row is read (`test_one_stage_reads_open_a_pearl_plus_tree_under_its_own_domain`)."""
    prog = _program(rows=16, w=32)
    first, ports = _record(prog, _registered(prog))
    own = G.domain(P.program_digest(prog), "W", 2, 16, 32).domain_id.hex()
    kw = {"law": LAW, "scheme": SCHEME, "ports": ports, "registered": {"W": 2}, "schemas": {"W": G.SCHEMA_SEG}}
    for y, name in ((False, "pearl-a/0/0"), (True, "pearl-y/0/0")):
        rows = [[(2 * r + j + 1) << 16 for j in range(2)] if r < 5 else [0, 0] for r in range(16)]
        tree, dom, _ = _pearl_plus(rows, y)
        theirs = {"value": name, "root": tree.commitment.root.hex(), "leaves": 16, "words": 2, G.WIDTH_KEY: 32,
                  "domain": dom.domain_id.hex()}
        assert theirs["domain"] != own

        def earlier(e):
            return R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 6}, registered=[e])

        def codes(entry, w=theirs, **change):
            rec = R.record(prog, Q, LAW, SCHEME, first["roots"], {**SERVED, "row": 8}, previous=R.digest(earlier(w)),
                           registered=[{**theirs, "value": "W", **entry}])
            return R.check(rec, prog, Q, **{**kw, "log": [earlier(w)], **change},
                           outputs={"W": {"record": R.digest(earlier(w)), "value": name}}).codes

        assert codes({}) == ()
        assert codes({}, schemas={"W": G.SCHEMA_V2}) == ()
        assert codes({"domain": own}) == ("R7-domain",)
        assert codes({"root": "ef" * 64}) == ("R7-held",)
        for w in ({k: v for k, v in theirs.items() if k != G.WIDTH_KEY}, {**theirs, "leaves": 5}):
            assert codes({}, w)[0] == "R7-output", (name, w)


#: an AROW row table's port (`verity.commitments.windows.rows.RowTable.port`: `row-table/<engine step>/<op_path>`)
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
