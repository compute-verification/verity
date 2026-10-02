"""The tile checks' hiding leaves in gates (`verity_pouw.circuit.leaves`): the row schemas' words and leaves are the pinned
vectors (``pouw_rows_vectors``), the gates compute them, hm96's mask is core's `hm96.SHA512.mask`, a row's leaf is core's
commit string at every padding edge, and the tile digest is `pearl_c.leaf`."""
from __future__ import annotations

import json
import random
import types

import pytest

import pouw_rows_vectors as V
from verity.commitments.rowleaf import ROLE_W, ROLE_X, nvfp4_row_bytes, row_bytes
from verity.evaluation import evaluate
from verity.ir.defs import CompositeDefinition, bind
from verity.ir.types import Array

from verity_pouw.circuit import hashes as H
from verity_pouw.circuit import leaves as L
from verity_pouw.circuit import words as W
from verity_pouw.schemes import pearl_c as C

PINNED = json.loads(V.OUT.read_text())


def _definition(name, params, n_out, body):
    return bind(CompositeDefinition(name, 0, (), lambda S: (params, Array(n_out, W.U32)), body, register=False))


def _be(halves) -> bytes:
    return b"".join(int(v).to_bytes(4, "big") for v in halves)


def _bits(rng: random.Random, n: int) -> bytes:
    return bytes(rng.randrange(256) for _ in range(n))


def test_pinned_row_vectors():
    assert V.vectors() == PINNED


# the gates' encoder of each row and the row's parameters, by row
_ROWS = {
    "bf16": (lambda e, x, k: L.bf16_row(e, x["x"], k), lambda k: (("x", Array(k, W.BF16)),)),
    "fp32": (lambda e, x, k: L.fp32_row(e, x["x"], k), lambda k: (("x", Array(k, W.F32)),)),
    "nvfp4-p": (lambda e, x, k: L.nvfp4_p_row(e, x["codes"], x["scales"], x["p"], k),
                lambda k: (("codes", Array(k, W.E2M1)), ("scales", Array(k // 16, W.U8)), ("p", Array(L.P_WORDS, W.BF16)))),
    "fp8-p": (lambda e, x, k: L.fp8_p_row(e, x["codes"], x["p"], k),
              lambda k: (("codes", Array(k, W.U8)), ("p", Array(L.P_WORDS, W.BF16)))),
}


@pytest.mark.parametrize("case", [c for c in PINNED["cases"] if c["row"] == "tile" or c["k"] == 64], ids=lambda c: c["case"])
def test_the_gates_compute_the_pinned_leaves(case):
    """Each schema's encoder and `Hm96Sha512` (a tile's: `PearlTileDigest` then `Hm96Sha512`) give the pinned b ‖ c from
    the case's elements and salt."""
    row, k, role = case["row"], case["k"], case["role"]
    x = V.inputs(case["case"], row, tuple(k) if row == "tile" else k)
    salt = V.shake(case["case"], "salt", L.INST.salt_bytes)
    if row == "tile":
        tm, tn = k
        params = (("c", Array(tm, Array(tn, W.F32))), ("u", Array(tm, Array(tn, W.F32))), ("salt", L.SALT))
        args = [[v for r in x["c"] for v in r], [v for r in x["u"] for v in r]]
        body = lambda B, S, c, u, y: L.tile_leaf(H.Emit(B), tm, tn, c, u, y)           # noqa: E731
    else:
        encode, sig = _ROWS[row]
        params, names = sig(k) + (("salt", L.SALT),), [n for n, _t in sig(k)]
        args = [x[n] for n in names]

        def body(B, S, *a):
            e = H.Emit(B)
            return L.leaf(e, role, encode(e, dict(zip(names, a)), k), a[-1])
    got = _be(evaluate(_definition(f"TLeaf_{case['case']}", params, 32, body), *args, L.halves(salt)))
    assert got.hex() == case["bc"]


def test_the_rows_are_the_tensors_bytes():
    """Packed in gates, each row's bytes are `rowleaf.row_bytes` of its reference words, nvfp4-p's codes and scales are
    `rowleaf.nvfp4_row_bytes`', and fp8-p's codes are the codes' bytes."""
    rng, k = random.Random(3), 128
    x = {"x": [rng.getrandbits(16) for _ in range(k)], "codes": [rng.getrandbits(4) for _ in range(k)],
         "scales": [rng.getrandbits(8) for _ in range(k // 16)], "p": [rng.getrandbits(16) for _ in range(L.P_WORDS)]}
    x32, c8 = [rng.getrandbits(32) for _ in range(k)], [rng.getrandbits(8) for _ in range(k)]
    for row, inputs, ref in (("bf16", {"x": x["x"]}, L.words_bf16(x["x"])), ("fp32", {"x": x32}, L.words_fp32(x32)),
                             ("nvfp4-p", x, L.words_nvfp4_p(x["codes"], x["scales"], x["p"])),
                             ("fp8-p", {"codes": c8, "p": x["p"]}, L.words_fp8_p(c8, x["p"]))):
        encode, sig = _ROWS[row]
        names = [n for n, _t in sig(k)]
        fn = _definition(f"TRow_{row}", sig(k), L.ROW_WORDS[row](k) // 2,
                         lambda B, S, *a: (lambda e: e.array(encode(e, dict(zip(names, a)), k)))(H.Emit(B)))
        assert _be(evaluate(fn, *(inputs[n] for n in names))) == row_bytes(ref, 16), row
        assert len(ref) == L.ROW_WORDS[row](k)
    assert row_bytes(L.words_nvfp4_p(x["codes"], x["scales"], x["p"]), 16) == \
        nvfp4_row_bytes(x["codes"], x["scales"]) + row_bytes(x["p"], 16)
    assert row_bytes(L.words_fp8_p(c8, x["p"]), 16) == bytes(c8) + row_bytes(x["p"], 16)


def test_the_mask_is_hm96s():
    """`Hm96Mask` is `hm96.SHA512.mask` under the pinned key: random salts, zero, all ones, and single bits at the ends of
    the salt's halves and bytes."""
    rng = random.Random(5)
    salts = [_bits(rng, 192) for _ in range(6)] + [bytes(192), b"\xff" * 192]
    salts += [(1 << t).to_bytes(192, "little") for t in (0, 7, 8, 31, 32, 1535)]
    for y in salts:
        assert _be(evaluate(L.Hm96Mask, L.halves(y))) == L.INST.mask(L.KEY, y)


def test_the_hankel_block_is_the_gf2_product():
    """Output bit a of `PouwHm96Hankel` is the parity of y_b · κ_(a+b) over b < 32, every operand's bit t bit t mod 8 of its
    byte t div 8 (hm96's bit order)."""
    rng = random.Random(6)
    bit = lambda data, t: data[t // 8] >> (t % 8) & 1                                   # noqa: E731
    for _ in range(200):
        y, k0, k1 = (rng.getrandbits(32) for _ in range(3))
        yb, kb = y.to_bytes(4, "big"), k0.to_bytes(4, "big") + k1.to_bytes(4, "big")
        want = bytearray(4)
        for a in range(32):
            if sum(bit(yb, b) & bit(kb, a + b) for b in range(32)) & 1:
                want[a // 8] |= 1 << (a % 8)
        assert W.PouwHm96Hankel.evaluate(y, k0, k1) == int.from_bytes(want, "big")


@pytest.mark.parametrize("role", [ROLE_X, ROLE_W])
def test_a_rows_leaf_is_cores_commit_string_at_every_padding_edge(role):
    """`Hm96Sha512{ROLE,WORDS}` is `hm96.SHA512.commit_string` of the row's `sha512/row/v1` digest: rows that pad into one
    block after the prefix (54 words), into two (56), and across the next edge (118, 120)."""
    rng = random.Random(role)
    for n in (2, 16, 54, 56, 118, 120):
        words, y = [rng.getrandbits(16) for _ in range(n)], _bits(rng, 192)
        got = _be(evaluate(L.hm96_leaf(role, n), L.halves(row_bytes(words, 16)), L.halves(y)))
        assert got == L.commit(role, words, y), n


@pytest.mark.parametrize("tm, tn", [(1, 1), (2, 3), (3, 33), (2, 64)])
def test_the_tile_digest_is_pearl_cs_leaf(tm, tn):
    """`PearlTileDigest` is `pearl_c.leaf` (h0): a partial chunk, a row of two chunks, and one message per row."""
    rng = random.Random(tm * 100 + tn)
    c, u = ([[rng.getrandbits(32) for _ in range(tn)] for _ in range(tm)] for _ in range(2))
    got = _be(evaluate(L.tile_digest(tm, tn), [v for r in c for v in r], [v for r in u for v in r]))
    assert got == C.leaf(types.SimpleNamespace(c=c, u=u))


def test_out_of_schema_rows_are_refused():
    for role, words in ((3, 16), (ROLE_X, 15), (ROLE_W, 0)):
        with pytest.raises(ValueError):
            L.hm96_leaf(role, words)
    with pytest.raises(ValueError):
        L.words_nvfp4_p([0] * 64, [0] * 3, [0] * L.P_WORDS)
    with pytest.raises(ValueError):
        L.words_fp8_p([0] * 64, [0] * 63)
    with pytest.raises(ValueError):
        L.words_tile(bytes(31))
