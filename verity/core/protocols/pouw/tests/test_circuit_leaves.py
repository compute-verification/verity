"""The tile checks' committed rows and tile digest (`verity.ml.protocols.pouw.circuit.leaves`): the row schemas' words and commit strings
are the pinned vectors (``pouw_rows_vectors``), each row's bytes are the tensor's memory, a row's commit string is core's
over `sha512/row-seg/v1` or `sha512/row/v2` at every padding and segment edge, the pinned row-seg rows are rebuilt from the
spec and core's own row-seg vectors are PoUW's rows, each unit's statement rows are its ports' schemas, and the tile digest
in gates is the pinned one and `pearl_c.leaf`.  C-Flock's tests recompute every row's vector through its own ports
(``verity/ml/flock/tests/test_pouw_rows.py``)."""
from __future__ import annotations

import hashlib
import json
import random
import types

import pytest

import pouw_rows_vectors as V
from verity.ml.commitments.frame_v3 import vectors_sha512 as FV
from verity.ml.commitments.rowleaf import (ROLE_W, ROLE_X, SCHEMA_HM96_SHA512_ROW_SEG, SCHEMA_HM96_SHA512_ROW_V2, bit_row,
                                        nvfp4_row_bytes, row_bytes, sha512_row_seg_digest, sha512_row_seg_word_digest,
                                        sha512_row_v2_digest)
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.scalar import BitAnd, SelectBf16

from verity.ml.protocols.pouw.circuit import leaves as L
from verity.ml.protocols.pouw.circuit import pc4, pc8, rowk
from verity.ml.protocols.pouw.schemes import pearl_c as C

PINNED = json.loads(V.OUT.read_text())


def test_pinned_row_vectors():
    assert V.vectors() == PINNED


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row"] == "tile"], ids=lambda c: c["case"])
def test_the_gates_compute_the_pinned_tile_digests(case):
    """`PearlTileDigest` gives the pinned tile digest from the case's C̃ and U."""
    tm, tn = case["k"]
    x = V.inputs(case["case"], "tile", (tm, tn))
    got = evaluate(L.tile_digest(tm, tn), [v for r in x["c"] for v in r], [v for r in x["u"] for v in r])
    assert L.digest_bytes(got).hex() == case["tile_digest"]


def test_the_rows_are_the_tensors_bytes():
    """Each row's bytes are the tensor's memory: bf16 and fp32 rows their words little-endian, nvfp4-p's codes and scales
    `rowleaf.nvfp4_row_bytes`' and fp8-p's codes the codes' bytes, each then P's BF16 words; and each packed row's bytes
    and bit count are `rowleaf.bit_row`'s of its elements (what `sha512/row/v2` hashes)."""
    rng, k = random.Random(3), 128
    x16, x32 = [rng.getrandbits(16) for _ in range(k)], [rng.getrandbits(32) for _ in range(k)]
    c4, c8 = [rng.getrandbits(4) for _ in range(k)], [rng.getrandbits(8) for _ in range(k)]
    scales, p = [rng.getrandbits(8) for _ in range(k // 16)], [rng.getrandbits(16) for _ in range(L.P_WORDS)]
    le = lambda vs, n: b"".join(v.to_bytes(n, "little") for v in vs)                 # noqa: E731
    rows = {"bf16": (L.words_bf16(x16), le(x16, 2)), "fp32": (L.words_fp32(x32), le(x32, 4)),
            "nvfp4-p": (L.words_nvfp4_p(c4, scales, p), nvfp4_row_bytes(c4, scales) + le(p, 2)),
            "fp8-p": (L.words_fp8_p(c8, p), bytes(c8) + le(p, 2))}
    for row, (words, memory) in rows.items():
        assert row_bytes(words, 16) == memory, row
        assert len(words) == L.ROW_WORDS[row](k), row
    assert bit_row((c4, 4), (scales, 8), (p, 16)) == (rows["nvfp4-p"][1], 16 * L.ROW_WORDS["nvfp4-p"](k))
    assert bit_row((c8, 8), (p, 16)) == (rows["fp8-p"][1], 16 * L.ROW_WORDS["fp8-p"](k))


@pytest.mark.parametrize("row, role, sizes", [("bf16", ROLE_W, (1, 63, 64, 65)), ("nvfp4-p", ROLE_W, (2, 16, 54, 56, 118, 120)),
                                              ("tile", ROLE_X, (16,)), ("bf16", ROLE_X, (1, 64, 511, 512, 513, 1024, 1025)),
                                              ("nvfp4-p", ROLE_X, (82, 1216))],
                         ids=["v2", "v2-packed", "tile", "seg", "seg-packed"])
def test_a_rows_commit_string_is_cores_at_every_padding_edge(row, role, sizes):
    """`commit` is `hm96.SHA512.commit_string` of the row's `sha512/row/v2` or `sha512/row-seg/v1` digest of the words'
    bit row: a registered bf16 row of part of a block, a block and either side of it; v2 rows that pad into one block
    after the prefix (54 words), into two (56), and across the next edge (118, 120); row-seg rows of one short segment, one
    full one (512 words), a full one and one word (513), two full and two and a word, and nvfp4-p's at k = 64 and 4,096."""
    rng = random.Random(len(row) + role)
    for n in sizes:
        words, y = [rng.getrandbits(16) for _ in range(n)], bytes(rng.getrandbits(8) for _ in range(192))
        schema = L.ROW_SCHEMA[row, role]
        x = (sha512_row_seg_digest(row_bytes(words, 16), 16 * n) if schema == SCHEMA_HM96_SHA512_ROW_SEG
             else sha512_row_v2_digest(row_bytes(words, 16), 16 * n))
        assert L.row_digest(row, role, words) == x, n
        assert L.commit(row, role, words, y) == L.INST.commit_string(L.KEY, x, y), n


def test_every_per_call_row_but_ok_the_tile_the_caps_and_the_cuts_is_a_row_seg_row():
    """The units' committed rows (`pc4.PORTS`, `pc8.PORTS`, the window tile's `pc8.SIDE_PORTS`, `cap.PORTS`, and the
    row_k cut's `rowk.PORTS4`, `rowk.PORTS8`) are exactly the rows with a schema, and the row-seg ones are the per-call
    rows (role X) but ok, the tile
    digest, the cap's rows and the cut's six: A_i, A′_i ‖ P_A,i of both checks and A's salt-dead counts.  Every other row,
    each registered row (role W) and the cap's chunk digests and strip row (64 words, one whole block, so a unit that reads
    or writes one hashes a block before its last) included, is a `sha512/row/v2` row: no row is `sha512/row/v1`."""
    from verity.ml.protocols.pouw.circuit import cap

    used = {(row, role) for ports in (pc4.PORTS, pc8.PORTS, {"side": pc8.SIDE_PORTS}, cap.PORTS, rowk.PORTS4,
                                      rowk.PORTS8)
            for unit in ports.values() for row, role, _ in unit["rows"]}
    assert used == set(L.ROW_SCHEMA)
    seg = {(row, ROLE_X) for row in ("bf16", "fp32", "nvfp4-p", "fp8-p", "fp8-pi", "sd")}
    assert {k for k, s in L.ROW_SCHEMA.items() if s == SCHEMA_HM96_SHA512_ROW_SEG} == seg
    assert {k for k, s in L.ROW_SCHEMA.items() if s == SCHEMA_HM96_SHA512_ROW_V2} == set(L.ROW_SCHEMA) - seg


#: each unit's statement rows: its input rows' (row, role) by read, then each output row's (row, role) and run of outputs
STATEMENTS = {
    (pc4, "row_a"): ([("bf16", ROLE_X)], [(("nvfp4-p", ROLE_X), 3), (("ok", ROLE_X), 1)]),
    (pc4, "row_b"): ([("bf16", ROLE_W)], [(("nvfp4-p", ROLE_W), 3), (("ok", ROLE_W), 1)]),
    (pc4, "tile"): ([("nvfp4-p", ROLE_X)] * 2 + [("nvfp4-p", ROLE_W)] * 3, [(("tile", ROLE_X), 1)]),
    (pc8, "row_a"): ([("fp32", ROLE_X)], [(("fp8-p", ROLE_X), 2), (("ok", ROLE_X), 1)]),
    (pc8, "row_b"): ([("fp32", ROLE_W)], [(("fp8-p", ROLE_W), 2), (("ok", ROLE_W), 1)]),
    (pc8, "tile"): ([("fp8-p", ROLE_X)] * 2 + [("fp8-p", ROLE_W)] * 3, [(("tile", ROLE_X), 1)]),
}


@pytest.mark.parametrize("mod, role", list(STATEMENTS), ids=lambda x: getattr(x, "__name__", x))
def test_a_units_statement_rows_are_its_ports_schemas(mod, role):
    """`statement_rows` of a unit's ports is its C-Flock META's ``row_schemas`` and ``out_row_ports``: each input row once
    per row a tile reads of it (a 2 × 3 tile: two A rows, three B rows), then each output row, every one under its
    (row, role)'s schema (`ROW_SCHEMA`, so the reader's and the writer's agree) and of its words' bits; the row a row unit
    writes as its codes, scales and P (A′_i ‖ P_A,i, B̃_j ‖ P_B,j, the fp8-p rows) is one run of output ports, and META
    names the runs only then: a tile's one output row, its digest, is one port's.  Every schema is one META's
    ``row_schemas`` accepts, v2 or row-seg."""
    k, (tm, tn) = 128, (2, 3)
    reads = {p: tm for p in ("ac", "as", "pa")} | {p: tn for p in ("bc", "bs", "pb")}
    got = L.statement_rows(mod.PORTS[role], k, reads)
    ins, outs = STATEMENTS[mod, role]
    want = [[L.ROW_SCHEMA[r], 16 * L.ROW_WORDS[r[0]](k)] for r in ins + [r for r, _n in outs]]
    runs = {"out_row_ports": [n for _r, n in outs]} if role != "tile" else {}
    assert got == {"row_schemas": want, **runs}
    assert {s for s, _bits in got["row_schemas"]} <= {SCHEMA_HM96_SHA512_ROW_V2, SCHEMA_HM96_SHA512_ROW_SEG}


def test_statement_rows_refuse_ports_that_are_no_statement():
    """A row of input and output ports, a row whose ports a tile reads different numbers of, and output rows that miss,
    repeat or reorder an output are refused."""
    for rows, why in (((("ok", ROLE_X, ("x", 0)),), "input ports or output ports"),
                      ((("nvfp4-p", ROLE_X, ("ac", "pa")),), "the same number of rows"),
                      ((("nvfp4-p", ROLE_X, (0, 2)), ("ok", ROLE_X, (3,))), "once each and in order"),
                      ((("nvfp4-p", ROLE_X, (1, 0, 2)),), "once each and in order"),
                      ((("nvfp4-p", ROLE_X, (0, 1)), ("ok", ROLE_X, (1,))), "once each and in order")):
        with pytest.raises(ValueError, match=why):
            L.statement_rows({"rows": rows}, 64, {"ac": 2, "pa": 3})


def test_advice_is_in_no_read_set():
    """Advice (`pc8.PORTS`: the cap tile's certificates) is plain witness only while no row or region reads it: each of
    `pc8`'s units names its advice apart from its rows and regions; a tile with the cap's advice has the same statement
    rows as without, so no committed row (and no link, which reads one) holds a bit of it; and advice that is also a
    row's port or a public input region, or that names an output, is refused."""
    from verity.ml.protocols.pouw.circuit import cap

    for role, unit in pc8.PORTS.items():
        bound = {p for _r, _rl, ps in unit["rows"] for p in ps} | set(unit["regions"])
        assert not set(unit["advice"]) & bound and all(isinstance(p, str) for p in unit["advice"]), role
    reads = {"ac": 2, "pa": 2, "bc": 3, "pb": 3}
    tile = pc8.PORTS["tile"]
    assert L.statement_rows({**tile, "advice": cap.ADVICE}, 128, reads) == L.statement_rows(tile, 128, reads)
    for unit in ({**tile, "advice": ("pa",)}, {**pc8.PORTS["row_a"], "advice": ("fa",)}, {**tile, "advice": (0,)}):
        with pytest.raises(ValueError, match="no committed row and no public input region"):
            L.statement_rows(unit, 128, reads)


SEG_TAG, SEGMENT_TAG = b"verity/sha512-row-seg/v1\0", b"verity/sha512-row-seg/v1/segment\0"


def _block(tag: bytes, *fields: tuple[int, int]) -> bytes:
    return (tag + b"".join(v.to_bytes(n, "big") for v, n in fields)).ljust(128, b"\0")


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row_schema"] == SCHEMA_HM96_SHA512_ROW_SEG],
                         ids=lambda c: c["case"])
def test_the_pinned_row_seg_rows_are_the_spec_by_hand(case):
    """A per-call row's pinned vector is frame-v3's `sha512/row-seg/v1` (its PROTOCOL.md) built by hand: the words' bytes
    in 1 KiB segments, each segment's length and d_j = SHA-512(S(j, s, m_j) ‖ segment j) as pinned, x = SHA-512(Q(ℓ) ‖
    d_0 ‖ … ‖ d_(s-1)), which is core's digest of the words as a word row (an fp32 row's also of its 32-bit words), and
    the commit string and tree leaf over it."""
    x = V.inputs(case["case"], case["row"], case["k"])
    words = V.words(case["row"], x)
    raw, n_bits = row_bytes(words, 16), 16 * len(words)
    s = -(-n_bits // 8192)
    ms = [min(8192, n_bits - 8192 * j) for j in range(s)]
    assert [g["n_bits"] for g in case["segments"]] == ms
    ds = [hashlib.sha512(_block(SEGMENT_TAG, (j, 8), (s, 8), (m, 8)) + raw[1024 * j:1024 * j + m // 8]).digest()
          for j, m in enumerate(ms)]
    assert [d.hex() for d in ds] == [g["digest"] for g in case["segments"]]
    want = hashlib.sha512(_block(SEG_TAG, (n_bits, 8), (s, 8), (1024, 4)) + b"".join(ds)).digest()
    assert case["x"] == want.hex() == sha512_row_seg_word_digest(words, 16).hex()
    if case["row"] == "fp32":
        assert want == sha512_row_seg_word_digest(x["x"], 32)
    bc = L.INST.commit_string(L.KEY, want, V.shake(case["case"], "salt", L.INST.salt_bytes))
    assert (bc.hex(), L.tree_leaf(bc).hex()) == (case["bc"], case["tree_leaf"])


CORE_SEG = FV.load()["row_seg_leaves"]


@pytest.mark.parametrize("case", [c for c in CORE_SEG if "words" in c or "hiding" in c], ids=lambda c: c["name"])
def test_cores_row_seg_vectors_are_pouws_per_call_rows(case):
    """Core's pinned `sha512/row-seg/v1` word rows (`row_seg_leaves` in `vectors_sha512.json`: an A_i BF16 row, a Pearl-C
    FP32 row and a short odd BF16 row) are PoUW's per-call bf16 and fp32 rows of the same elements, and core's 8 KiB hiding
    row's commit string and hm96 leaf are PoUW's of the row's words."""
    if "words" in case:
        w = case["words"]
        elems = FV.elements(w["label"], w["bits"], w["n"])
        row, words = ("bf16", L.words_bf16(elems)) if w["bits"] == 16 else ("fp32", L.words_fp32(elems))
    else:
        raw = bytes.fromhex(case["row_bytes"])
        row, words = "bf16", [int.from_bytes(raw[i:i + 2], "little") for i in range(0, len(raw), 2)]
    assert L.row_digest(row, ROLE_X, words).hex() == case["digest"]
    if "hiding" in case:
        h = case["hiding"]
        assert h["key_sha512"] == hashlib.sha512(L.KEY).hexdigest()
        bc = L.commit(row, ROLE_X, words, bytes.fromhex(h["salt"]))
        assert (bc.hex(), L.tree_leaf(bc).hex()) == (h["bc"], h["hm96_leaf"])


@pytest.mark.parametrize("tm, tn", [(1, 1), (2, 3), (3, 33), (2, 64)])
def test_the_tile_digest_is_pearl_cs_leaf(tm, tn):
    """`PearlTileDigest` is `pearl_c.leaf` (h0): a partial chunk, a row of two chunks, and one message per row."""
    rng = random.Random(tm * 100 + tn)
    c, u = ([[rng.getrandbits(32) for _ in range(tn)] for _ in range(tm)] for _ in range(2))
    got = evaluate(L.tile_digest(tm, tn), [v for r in c for v in r], [v for r in u for v in r])
    assert L.digest_bytes(got) == C.leaf(types.SimpleNamespace(c=c, u=u))


def test_out_of_schema_rows_are_refused():
    """An empty row and a row in a tree it never sits in are refused; a registered row of part of a block is a v2 row and a
    per-call row of the same words a row-seg row, each of any length."""
    for row, words in (("bf16", 32), ("fp32", 96)):
        assert L.row_digest(row, ROLE_W, [0] * words) == sha512_row_v2_digest(bytes(2 * words), 16 * words)
        assert L.row_digest(row, ROLE_X, [0] * words) == sha512_row_seg_word_digest([0] * words, 16)
    with pytest.raises(ValueError, match="positive number of words"):
        L.commit("tile", ROLE_X, [], bytes(192))
    with pytest.raises(ValueError, match="no tile row sits in a tree of role"):
        L.commit("tile", ROLE_W, [0] * 16, bytes(192))
    with pytest.raises(ValueError):
        L.words_nvfp4_p([0] * 64, [0] * 3, [0] * L.P_WORDS)
    with pytest.raises(ValueError):
        L.words_fp8_p([0] * 64, [0] * 63)
    with pytest.raises(ValueError):
        L.words_tile(bytes(31))
    with pytest.raises(ValueError):
        L.words_ok(2)


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row"] == "ok"], ids=lambda c: c["case"])
def test_an_ok_row_with_a_bit_set_above_bit_0_is_refused(case):
    """An ok row is one word with bits 1-15 zero, so one admission value has one leaf: the reference reads the pinned
    vector's word (`ok_of`) and refuses it with any one of bits 1-15 set, and so does `commit`, as it refuses a row of two
    words or none."""
    words, salt = V.words("ok", V.inputs(case["case"], "ok", case["k"])), V.shake(case["case"], "salt", L.INST.salt_bytes)
    role = V.ROLES[case["role"]]
    assert L.ok_of(words) == words[0] and L.commit("ok", role, words, salt).hex() == case["bc"]
    for bad in [[words[0] | 1 << b] for b in range(1, 16)] + [[words[0], 0], []]:
        with pytest.raises(ValueError, match="an ok row is one word"):
            L.ok_of(bad)
        with pytest.raises(ValueError, match="an ok row is one word"):
            L.commit("ok", role, bad, salt)


ROW_UNITS = ([pc4.templates(64)[r] for r in ("row_a", "row_b")]
             + [pc8.templates(128, dev=d)[r] for d in ("h100", "sm120") for r in ("row_a", "row_b")])


@pytest.mark.parametrize("fn", ROW_UNITS, ids=lambda fn: fn.id)
def test_a_row_unit_fixes_its_ok_words_bits_1_to_15_in_gates(fn):
    """The circuit can't write an ok row `ok_of` refuses: a row unit's last output, its ok row's one word (`leaves.OK`), is
    `SelectBf16` of its admission bit between the constants 1 and 0 (`pc4.ok_word`), so every bit of the word is a gate's
    and bits 1-15 are zero whatever the bit, not free bits of the committed row."""
    b = fn.body
    _space, k, _leaf = b.ret[len(b.ret) - 1]
    gate = b.nodes[k]
    bit, one, zero = (b.nodes[a.refs[0][1]] for a in gate.args)
    assert fn.signature()[1].fields[-1] == L.OK
    assert gate.fn.id == SelectBf16.id and bit.fn.id == BitAnd.id
    assert [(c.fn.family, c.fn.evaluate()) for c in (one, zero)] == [("Const", 1), ("Const", 0)]
    assert [L.ok_of([gate.fn.evaluate(v, 1, 0)]) for v in (0, 1)] == [0, 1]
