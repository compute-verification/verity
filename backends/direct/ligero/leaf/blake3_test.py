"""Conformance of the BLAKE3 leaf: native == the `blake3` package, gadget == native through the witness program.

The gadget tests run the system's program with a small numpy interpreter (the ligero witness path needs torch, absent on
the laptop); `test_gadget_torch_witness_path` repeats the check through `witness.build_witness` where torch exists."""
from __future__ import annotations

import numpy as np
import pytest

from ..compile import Linear, LigeroCtx, P, Quadratic
from ..relations import FP8_ADA_RELATION
from . import blake3 as L


def _eval(e, W: np.ndarray) -> np.ndarray:
    acc = np.full(W.shape[1], e.c % P, dtype=np.int64)
    for j, v in e.lin.items():
        acc = (acc + (v % P) * W[j]) % P
    return acc


def run_program_np(sys, pub: dict, hints: dict) -> np.ndarray:
    """W (m, l) from ``sys.program`` for the op kinds the leaf uses (pin | hint | prod | bits)."""
    l = len(next(iter(hints.values())))
    W = np.zeros((sys.m, l), dtype=np.int64)
    for op in sys.program:
        if op[0] == "pin":
            W[op[1]] = np.asarray(pub[op[2]], dtype=np.int64) % P
        elif op[0] == "hint":
            W[op[1]] = np.asarray(hints[op[2]], dtype=np.int64) % P
        elif op[0] == "prod":
            W[op[1]] = (_eval(op[2], W) * _eval(op[3], W)) % P
        elif op[0] == "bits":
            _, row0, nbits, e = op
            x = _eval(e, W)
            for i in range(nbits):
                W[row0 + i] = (x >> i) & 1
        else:
            raise ValueError(op[0])
    return W


def failing_constraints(sys, W: np.ndarray, pub: dict) -> list[str]:
    bad = []
    for c in sys.linear:
        lhs = _eval(type("E", (), {"lin": c.lin, "c": c.c})(), W)
        rhs = np.asarray(pub[c.pub], dtype=np.int64) % P if c.pub else 0
        if ((lhs - rhs) % P).any():
            bad.append(c.name)
    for q in sys.quadratic:
        if ((_eval(q.a, W) * _eval(q.b, W) - _eval(q.c, W)) % P).any():
            bad.append(q.name)
    return bad


@pytest.mark.parametrize("word_bits", [8, 16])
@pytest.mark.parametrize("role", [1, 2])
def test_native_is_blake3_of_the_row_bytes(word_bits, role):
    rng = np.random.default_rng(7 + word_bits + role)
    rows = rng.integers(0, 1 << word_bits, size=(6, L.K_VU), dtype=np.int64)
    rows[0] = 0                                       # the all-zero row too
    leaf = L.Blake3Leaf()
    d = leaf.native(rows, word_bits=word_bits, role=role)
    nch = L.n_chunks(word_bits)
    assert d.shape == (6, L.DIGEST_ELEMS) and (d[:, 0] == nch + 4 * role).all() and d[:, 1:].min() >= 0 and d[:, 1:].max() < 1 << 16
    assert not d[:, 1 + 16 * nch:].any()
    ref = leaf.reference_digest(rows, word_bits=word_bits, role=role)
    for r in range(6):
        assert leaf.leaf_bytes(d[r]) == ref[r] and len(ref[r]) == 32
    # the role is in the key (red-team-leaf NIT): the other role's digest and leaf differ
    other = 3 - role
    d2 = leaf.native(rows, word_bits=word_bits, role=other)
    assert not np.array_equal(d[:, 1:], d2[:, 1:]) and leaf.leaf_bytes(d2[1]) != ref[1]
    assert leaf.leaf_bytes(d2[1]) == leaf.reference_digest(rows, word_bits=word_bits, role=other)[1]
    assert L.key_for(1) != L.key_for(2) and L.key_for(role).startswith(b"verity/blake3-leaf/v1/")


def test_leaf_bytes_is_total_on_field_rows_and_tags_malformed_frames():
    """The contract's conformance suite feeds random field rows: a malformed frame maps to the tagged SHA-256 domain
    (never a BLAKE3 root), the frame parser itself rejects it, and only a wrong length / non-field element raises."""
    import hashlib
    malformed = [[6, 1 << 16] + [0] * (L.DIGEST_ELEMS - 2),        # limb out of range
                 [0] * L.DIGEST_ELEMS,                              # no chunks / no role
                 [6] + [1] * (L.DIGEST_ELEMS - 1),                  # unused slot not 0
                 [2] + [0] * (L.DIGEST_ELEMS - 1),                  # role 0
                 [3 + 12] + [0] * (L.DIGEST_ELEMS - 1)]             # role 3
    seen = set()
    for m in malformed:
        with pytest.raises(ValueError):
            L.elems_to_cvs(m)
        b = L.leaf_bytes(m)
        assert b == hashlib.sha256(L.MALFORMED_TAG + np.asarray(m, dtype="<u4").tobytes()).digest() and len(b) == 32
        seen.add(b)
    assert len(seen) == len(malformed)
    good = L.Blake3Leaf().native(np.zeros((1, L.K_VU), dtype=np.int64), word_bits=8, role=1)[0]
    assert L.leaf_bytes(good) not in seen and L.elems_to_cvs(good)[1] == 1
    with pytest.raises(ValueError):
        L.leaf_bytes([6] + [0] * 32)                                   # wrong length
    with pytest.raises(ValueError):
        L.leaf_bytes([P] + [0] * (L.DIGEST_ELEMS - 1))                 # not a field element


def _column_system(word_bits: int, role: int, nblk: float = 1):
    """One column system with ``nblk`` blocks per column: 0.5 (the unfolded relations), 1 (x2), 2 (x4)."""
    leaf = L.Blake3Leaf()
    ctx = LigeroCtx(FP8_ADA_RELATION.params)
    ctx.scope = "t"
    k = int(nblk * 8 * L.BLOCK_LEN) // word_bits
    words = [ctx.hint(f"w[{i}]") for i in range(k)]
    carry = [ctx.hint(f"carry[{i}]") for i in range(L.CARRY_ELEMS)]
    digest, carry_out = leaf.gadget(ctx, words, word_bits=word_bits, role=role, row_words=L.K_VU, carry_in=carry)
    assert len(digest) == leaf.digest_elems and len(carry_out) == leaf.carry_elems
    return leaf, ctx.sys, words, carry, digest, carry_out


def _one_row_inputs(leaf, sys, word_bits: int, role: int, seed: int, nblk: float = 1):
    k = int(nblk * 8 * L.BLOCK_LEN) // word_bits
    steps = L.K_VU // k
    rng = np.random.default_rng(seed)
    rows = rng.integers(0, 1 << word_bits, size=(1, L.K_VU), dtype=np.int64)
    car = leaf.carry_values(rows, word_bits=word_bits, role=role, steps=steps)          # (1, steps, CARRY_ELEMS)
    pub: dict = {}
    hints = {}
    for i in range(k):
        hints[f"w[{i}]"] = rows[0].reshape(steps, k)[:, i]
    for i in range(L.CARRY_ELEMS):
        hints[f"carry[{i}]"] = car[0, :, i]
    return rows, steps, car, pub, hints


@pytest.mark.parametrize("nblk", [0.5, 1, 2])
@pytest.mark.parametrize("word_bits", [8, 16])
def test_gadget_equals_native_along_one_row(word_bits, nblk):
    """One VU row over its `steps` columns (l = steps): the program's rows satisfy every constraint, the carried rows
    chain (carry_out[j] == carry_in[j + 1]), and the digest expressions at the last column are the native chunk CVs."""
    role = 1
    leaf, sys, words, carry, digest, carry_out = _column_system(word_bits, role, nblk)
    rows, steps, car, pub, hints = _one_row_inputs(leaf, sys, word_bits, role, 11 + word_bits + int(2 * nblk), nblk)
    W = run_program_np(sys, pub, hints)
    assert failing_constraints(sys, W, pub) == []
    outs = np.stack([_eval(e, W) for e in carry_out])                                    # (CARRY_ELEMS, steps)
    assert np.array_equal(outs[:, :-1], car[0, 1:, :].T), "carry_out[j] must be the next column's carry_in"
    dg = np.array([int(_eval(e, W)[steps - 1]) for e in digest])
    assert np.array_equal(dg, leaf.native(rows, word_bits=word_bits, role=role)[0])
    assert leaf.leaf_bytes(dg) == leaf.reference_digest(rows, word_bits=word_bits, role=role)[0]


def test_tampered_rows_are_rejected():
    """A flipped product row (one XOR bit), a flipped add bit and a wrong carried CV each break a constraint."""
    word_bits, role = 8, 1
    leaf, sys, words, carry, digest, carry_out = _column_system(word_bits, role)
    rows, steps, car, pub, hints = _one_row_inputs(leaf, sys, word_bits, role, 3)
    W0 = run_program_np(sys, pub, hints)
    assert failing_constraints(sys, W0, pub) == []
    prod_rows = [r.idx for r in sys.rows if r.kind == "prod"]
    bit_rows = [r.idx for r in sys.rows if r.kind == "bit"]
    for rows_ in (prod_rows, bit_rows):
        W = W0.copy()
        j = rows_[len(rows_) // 2]
        W[j, 5] = (W[j, 5] + 1) % P
        assert failing_constraints(sys, W, pub) != []
    # a wrong carried CV in a chunk-start column is masked by the key selection (cs = 1) -- but not elsewhere
    hints2 = dict(hints)
    v = hints2["carry[0]"].copy()
    v[7] ^= 1
    hints2["carry[0]"] = v
    W = run_program_np(sys, pub, hints2)
    assert failing_constraints(sys, W, pub) == []                     # the program follows the (wrong) hint consistently
    outs = np.stack([_eval(e, W) for e in carry_out])
    assert not np.array_equal(outs[:, :-1], car[0, 1:, :].T)         # ... and the chain constraint is what catches it


PER_COLUMN_ROLE_ROWS = 15331      # one 64-byte column, one role: compression + layout + selects + holds + copies


def test_census_numbers_are_the_documented_ones():
    c = L.census(8)
    assert c["per_block_bit_design"] == 15136 and c["compression_rows"] == 14880
    assert c["lookup_xor_rows_per_u32"] == 2112 and c["bit_xor_rows_per_u32"] == 32
    assert c["per_column_role_rows"] == PER_COLUMN_ROLE_ROWS
    assert L.DIGEST_ELEMS == 49 and L.CARRY_ELEMS == 65 and L.n_chunks(8) == 2 and L.n_chunks(16) == 3


def test_gadget_torch_witness_path():
    torch = pytest.importorskip("torch", exc_type=ImportError)   # the laptop carries a raising torch stub
    from ..witness import build_witness, check_constraints, eval_expr, tables
    word_bits, role = 8, 2
    leaf, sys, words, carry, digest, carry_out = _column_system(word_bits, role)
    rows, steps, car, pub, hints = _one_row_inputs(leaf, sys, word_bits, role, 5)
    pub_t = {n: torch.as_tensor(v, dtype=torch.int64) for n, v in pub.items()}
    ht = torch.stack([torch.as_tensor(hints[n], dtype=torch.int64) for _, n in sys.hints])
    W = build_witness(sys, pub_t, ht, "cpu")
    assert check_constraints(sys, tables(sys, "cpu"), W, pub_t) == []
    dg = np.array([int(eval_expr(e, W)[steps - 1]) for e in digest])
    assert np.array_equal(dg, leaf.native(rows, word_bits=word_bits, role=role)[0])
    # chain_witness (the compose layer's entry point): two VUs + 3 pad columns, columns laid out VU-major
    k = 8 * L.BLOCK_LEN // word_bits
    rows2 = np.random.default_rng(9).integers(0, 1 << word_bits, size=(2, L.K_VU), dtype=np.int64)
    l = 2 * steps + 3
    wt = torch.zeros((l, k), dtype=torch.int64)
    wt[:2 * steps] = torch.as_tensor(rows2.reshape(2 * steps, k))
    carry_t, dg_t = leaf.chain_witness(wt, word_bits=word_bits, role=role, row_words=L.K_VU, n_vus=2, steps=steps, l=l)
    assert carry_t.shape == (L.CARRY_ELEMS, l) and dg_t.shape == (2, L.DIGEST_ELEMS)
    car2 = leaf.carry_values(rows2, word_bits=word_bits, role=role, steps=steps)
    assert np.array_equal(carry_t.numpy()[:, :2 * steps], car2.reshape(2 * steps, -1).T) and not carry_t[:, 2 * steps:].any()
    assert np.array_equal(dg_t.numpy(), leaf.native(rows2, word_bits=word_bits, role=role))
