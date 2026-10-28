"""Dither's served unit (`circuit.dither`): the coins and the filler's per-call pad, a call's 16-bit words and O, the leaves
a kernel's sampled cells are checked against, the operand roots, the Definitions on bits against the reference, and the
dot step's binding on both cards. O at 16 bits with A′: the layout, A′'s map against the served Definitions, an honest
gate-up call and its consumer, and the forged O, the forged A′, A = −V − d_A and the guessed A′ refused."""
from __future__ import annotations

import functools
import hashlib
import random

import numpy as np
import pytest

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate
from verity.ml.circuits.ops.prims import BlackwellS8ImmaDot32  # noqa: F401  (registers the s8 steps' families)
from verity.ml.circuits.silicon.integer import HOPPER_SM90_S8_K32
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import segment as SG

SALT = hashlib.sha512(b"window salt").digest()
COIN = hashlib.sha512(b"call coin").digest()
RECEIPT = hashlib.sha512(b"operand-root receipt").digest()
SEEDS = D.seeds(COIN, RECEIPT, 0)
#: K = 160 at L = 64 (t = 2): three segments, the last of 32; 1 KiB rows
SH = D.Shape(128, 128, 160, lp=32, L=64, row=1024)
KS = hashlib.sha512(b"prover's salt key").digest()
OFF = 1 << 33


def _operands(sh: D.Shape, seed: int = 7) -> tuple[list[list[int]], list[list[int]]]:
    rng = np.random.default_rng(seed)
    return rng.integers(-128, 128, size=(sh.m, sh.k)).tolist(), rng.integers(-128, 128, size=(sh.n, sh.k)).tolist()


@functools.cache
def _call(sh: D.Shape = SH):
    A, B = _operands(sh)
    T, O = D.values(sh, SEEDS, A, B)
    return A, B, T, O, D.commit(sh, D.values_buffer(sh, T, O), D.salts(KS, OFF))


def _np(sh: D.Shape, A, B):
    """X, W′, A + d_A and B + d_B in numpy, from the reference's coins."""
    na, nb = np.array(D.noise_a(sh, SEEDS, range(sh.m))), np.array(D.noise_b(sh, SEEDS, range(sh.n)))
    da, db = np.array(D.dithers(sh, SEEDS, "A", range(sh.m))), np.array(D.dithers(sh, SEEDS, "B", range(sh.n)))
    A, B = np.array(A), np.array(B)
    return D.wrap8(A + na), D.wrap8(B + nb), A + da, B + db


def test_served_seeds_are_the_coin_after_the_receipt_and_pinned():
    """seed_x = SHA-512(SEEDS_TAG ‖ coin ‖ operand-root receipt ‖ be64(call) ‖ x): each of the coin, the receipt and the
    call's index moves every stream's seed, and the streams' seeds differ."""
    assert SEEDS["dA"] == hashlib.sha512(b"verity/pouw/dither/seeds/v1\0" + COIN + RECEIPT + bytes(8) + b"dA\0\0\0\0\0\0").digest()
    other = hashlib.sha512(b"other").digest()
    for moved in (D.seeds(other, RECEIPT, 0), D.seeds(COIN, other, 0), D.seeds(COIN, RECEIPT, 1)):
        assert all(moved[x] != SEEDS[x] for x in D.STREAMS)
    assert len(set(SEEDS.values())) == len(D.STREAMS) and set(SEEDS) == set(D.STREAMS)
    assert hashlib.sha512(b"".join(SEEDS[x] for x in D.STREAMS)).hexdigest()[:32] == "2b38647082f0dfdcf3a814882f55b92c"
    with pytest.raises(ValueError, match="64-byte"):
        D.seeds(COIN, RECEIPT[:32], 0)


def test_pad_is_fixed_by_the_coin_and_changes_with_it():
    rows = range(8)
    pad = np.array(D.pad_rows(D.fill_seeds(SALT, COIN), 4096, rows))
    assert D.pad_rows(D.fill_seeds(SALT, COIN), 4096, rows) == pad.tolist()
    other_coin = hashlib.sha512(b"another call's coin").digest()
    other_salt = hashlib.sha512(b"another window's salt").digest()
    for seeds in (D.fill_seeds(SALT, other_coin), D.fill_seeds(other_salt, COIN)):
        moved = np.array(D.pad_rows(seeds, 4096, rows))
        assert (moved != pad).mean() > 0.98
        assert all(seeds[x] != D.fill_seeds(SALT, COIN)[x] for x in (D.PAD, *D.STREAMS))
    assert pad.min() == -128 and pad.max() == 127


def test_pad_and_seeds_pinned():
    """The derivation a kernel reproduces: seed_x = SHA-512(FILL_SEEDS_TAG ‖ salt ‖ coin ‖ x), then the pad stream's
    blocks."""
    seeds = D.fill_seeds(SALT, COIN)
    assert seeds["pad"] == hashlib.sha512(D.FILL_SEEDS_TAG + SALT + COIN + b"pad\0\0\0\0\0").digest()
    k, i = 4096, 3
    blocks = b"".join(hashlib.sha512(seeds["pad"] + b"pad\0\0\0\0\0" + (i * 64 + b).to_bytes(8, "big")).digest()
                      for b in range(64))
    assert D.pad_rows(seeds, k, [i])[0] == np.frombuffer(blocks, dtype=np.int8).tolist()
    assert hashlib.sha512(b"".join(seeds[x] for x in (D.PAD, *D.STREAMS))).hexdigest()[:32] == "db8610d22ea9dc75058e0b5122c54691"
    pad = np.array(D.pad_rows(seeds, k, range(4)), dtype=np.int8).tobytes()
    assert hashlib.sha512(pad).hexdigest()[:32] == "c9259c0840d4afcace2b430c740baa8d"


def test_seeds_refuse_short_digests():
    with pytest.raises(ValueError, match="64-byte"):
        D.fill_seeds(SALT[:32], COIN)


def test_noise_is_dithers_plus_hyb():
    na, nb = np.array(D.noise_a(SH, SEEDS, range(4))), np.array(D.noise_b(SH, SEEDS, range(4)))
    for n in (na, nb):
        assert n.min() >= -D.NOISE and n.max() <= D.NOISE + 1
    assert all(sorted(p) == list(range(SH.lp)) for p in D.perms(SEEDS, SH.k, SH.lp))
    lp = SH.lp
    assert set(np.unique(na[:, :lp] - na[:, lp:2 * lp]).tolist()) <= {-1, 0, 1}
    assert not set(np.unique(nb[:, :lp] - nb[:, lp:2 * lp]).tolist()) <= {-1, 0, 1}


def test_values_are_16_bit_prefix_sums_and_o_at_32_bits():
    A, B, T, O, _ = _call()
    X, W, Ad, Bd = _np(SH, A, B)
    T, O = np.array(T), np.array(O)
    assert T.shape == (SH.S, SH.m, SH.n) and SH.S == 3 and SH.seg(3) == (128, 160)
    rng = random.Random(1)
    for _ in range(20):
        i, j = rng.randrange(SH.m), rng.randrange(SH.n)
        full = [int(np.dot(X[i, :b], W[j, :b])) % (1 << 16) for b in (64, 128, 160)]
        o = int(np.dot(Ad[i], Bd[j])) % (1 << 32)
        assert T[:, i, j].tolist() == full and int(O[i, j]) == o
        assert D.cell_values(SH, SEEDS, A[i], B[j], i, j) == (full, o)


def test_default_shape_is_nofold_opts_pick_and_its_layout_is_4993s():
    """L = 1,024 and L′ = 128 (t = 8) with 16 KiB rows: per tile P = 2(S − 1) + 4 rows, the words of s = 1 .. S − 1 and
    then O (T_S is implied by O); every value's leaf and offset is the one `leaf_of` lists it at. A filler's tile commits
    all S segments' words and no O, and a call of one segment commits O alone."""
    sh = D.Shape(256, 128, 4096)
    assert (sh.L, sh.lp, sh.t, sh.row, sh.S, sh.W, sh.row_w, sh.row_o, sh.lpt, sh.leaves) == \
        (1024, 128, 8, 16384, 4, 3, 2, 4, 10, 20)
    for pos in range(sh.leaves):
        s, cells = sh.leaf_of(pos)
        assert s is None if pos % 10 >= 6 else s == pos % 10 // 2 + 1
        for x, (i, j) in enumerate(cells[::997]):
            got = sh.o_leaf(i, j) if s is None else sh.word_leaf(i, j, s)
            assert got == (pos, (4 if s is None else 2) * cells.index((i, j)))
    assert sh.opened(5, 9, 4) == [sh.word_leaf(5, 9, 3)[0], sh.o_leaf(5, 9)[0]]
    assert sh.opened(5, 9, 2) == [sh.word_leaf(5, 9, 2)[0], sh.word_leaf(5, 9, 1)[0]]
    with pytest.raises(ValueError, match="implied by O"):
        sh.word_leaf(5, 9, 4)
    fill = D.Shape(128, 128, 4096, o=False)
    assert (fill.W, fill.row_o, fill.lpt) == (4, 0, 8) and fill.domain() != D.Shape(128, 128, 4096).domain()
    assert fill.opened(5, 9, 4) == [fill.word_leaf(5, 9, 4)[0], fill.word_leaf(5, 9, 3)[0]]
    one = D.Shape(128, 128, 1024)
    assert (one.S, one.W, one.lpt, one.opened(5, 9, 1)) == (1, 0, 4, [one.o_leaf(5, 9)[0]])


def test_leaves_from_sampled_cells_and_openings():
    A, B, T, O, cm = _call()
    assert len(cm.rows) == SH.leaves == SH.tiles * SH.lpt and (SH.W, SH.row_w, SH.row_o) == (2, 32, 64)
    rng = random.Random(2)
    for _ in range(6):
        i, j, s = rng.randrange(SH.m), rng.randrange(SH.n), rng.randrange(1, SH.W + 1)
        for pos, off, width, want in ((*SH.word_leaf(i, j, s), 2, T[s - 1][i][j]), (*SH.o_leaf(i, j), 4, O[i][j])):
            row, salt, path = cm.open(pos)
            assert salt == D.prf_salt(KS, OFF + pos)
            assert D.leaf_row(SH, SEEDS, A, B, pos) == row
            assert int.from_bytes(row[off:off + width], "little") == want
            assert D.opens(cm.dom, cm.root, SH.leaves, pos, row, salt, path)
            assert not D.opens(cm.dom, cm.root, SH.leaves, pos, bytes([row[0] ^ 1]) + row[1:], salt, path)
            assert not D.opens(cm.dom, cm.root, SH.leaves, pos, row, D.prf_salt(KS, pos), path)


def test_operand_roots_and_the_checkpoint_pin():
    """A opens from the call's operand root and B from the weight root; registration's pin is B's checkpoint digest, which
    any one changed code moves."""
    A, B, *_ = _call()
    salt = lambda p: D.prf_salt(KS, 1 << 40 | p)  # noqa: E731
    ta, tb = D.operand_tree("A", A, salt), D.operand_tree("B", B, salt)
    assert ta.root != tb.root and ta.dom != tb.dom
    for t, M, r in ((ta, A, 77), (tb, B, 3)):
        row, s, path = t.open(r)
        assert D.operand_row(row) == M[r] and D.opens(t.dom, t.root, len(M), r, row, s, path)
        assert not D.opens(tb.dom if t is ta else ta.dom, t.root, len(M), r, row, s, path)
    pin = D.checkpoint_digest(B)
    moved = [list(r) for r in B]
    moved[5][17] = moved[5][17] + 1 if moved[5][17] < 127 else 0
    assert D.checkpoint_digest(B) == pin != D.checkpoint_digest(moved)
    with pytest.raises(ValueError, match="int8"):
        D.checkpoint_digest([[128] * SH.k])


def _opened_rows(sh: D.Shape, cm: D.Tree, i: int, j: int, s: int) -> dict[int, bytes]:
    out = {}
    for p in sh.opened(i, j, s):
        row, salt, path = cm.open(p)
        assert D.opens(cm.dom, cm.root, sh.leaves, p, row, salt, path)
        out[p] = row
    return out


def _inputs(sh: D.Shape, A, B, cm: D.Tree, i: int, j: int, s: int) -> list[list[int]]:
    rows = _opened_rows(sh, cm, i, j, s)
    t0 = D.word_at(sh, rows, i, j, s - 1)
    if s == sh.S:
        return D.block_inputs(sh, SEEDS, A[i], B[j], i, j, s, t0, o=D.out_at(sh, rows, i, j))
    return D.block_inputs(sh, SEEDS, A[i], B[j], i, j, s, t0, D.word_at(sh, rows, i, j, s))


@pytest.mark.slow
def test_definitions_pass_an_honest_call_on_bits():
    """Every segment of sampled cells, A's and B's rows and the values opened from their roots, through the Definitions
    on bits: `PouwDitherBlock_v1` at s < S and `PouwDitherLast_v1` (whole rows, T_{S−1} and O) at s = S, whose last
    segment is short."""
    A, B, T, O, cm = _call()
    for i, j in ((0, 0), (77, 100), (127, 127)):
        for s in range(1, SH.S + 1):
            ins = _inputs(SH, A, B, cm, i, j, s)
            last = s == SH.S
            assert (D.check_last(SH, SEEDS, i, j, ins) if last else D.check_block(ins)) == 1
            if not last or (i, j) == (77, 100):
                assert list(evaluate(D.block_definition(SH, s), *D.block_bits(ins, last))) == [1]


@pytest.mark.slow
@pytest.mark.parametrize("k, L", [(64, 32), (32, 32)])
def test_last_refuses_what_v32s_checks_refuse(k, L):
    """One bit of each port flipped in a last segment's block, at S = 2 (K = 64, L = 32) and S = 1: the Definition agrees
    with v3.2's checks in the clear (T_S from O by the literal binding identity, the segment and O), and a flipped t0 or
    o fails."""
    sh = D.Shape(128, 128, k, lp=32, L=L, row=1024)
    A, B = _operands(sh, 11)
    i, j = 5, 9
    T, o = D.cell_values(sh, SEEDS, A[i], B[j], i, j)
    honest = D.block_inputs(sh, SEEDS, A[i], B[j], i, j, sh.S, T[-2] if sh.S > 1 else 0, o=o)
    assert len(honest) == (8 if sh.S > 1 else 5)
    d = D.block_definition(sh, sh.S)
    assert D.check_last(sh, SEEDS, i, j, honest) == 1 and list(evaluate(d, *D.block_bits(honest, True))) == [1]
    rng = random.Random(3)
    widths = D.block_widths(honest, True)
    for port, width in enumerate(widths):
        bad = [list(v) for v in honest]
        bad[port][rng.randrange(len(bad[port]))] ^= 1 << rng.randrange(width)
        want = D.check_last(sh, SEEDS, i, j, bad)
        assert list(evaluate(d, *D.block_bits(bad, True))) == [want]
        if width >= 16:
            assert want == 0


def test_block_refuses_what_the_reference_refuses():
    sh = D.Shape(128, 128, 96, lp=32, L=32, row=1024)
    A, B = _operands(sh, 12)
    i, j = 3, 4
    T, _ = D.cell_values(sh, SEEDS, A[i], B[j], i, j)
    rng = random.Random(4)
    for s in (1, 2):
        honest = D.block_inputs(sh, SEEDS, A[i], B[j], i, j, s, T[0] if s == 2 else 0, T[s - 1])
        d = D.block_definition(sh, s)
        for port, width in enumerate(D.block_widths(honest, False)):
            bad = [list(v) for v in honest]
            bad[port][rng.randrange(len(bad[port]))] ^= 1 << rng.randrange(width)
            want = D.check_block(bad)
            assert list(evaluate(d, *D.block_bits(bad)) ) == [want]
            if port >= 4:
                assert want == 0


def test_a_skipped_noise_tile_and_a_forged_o_are_refused():
    """A prover that skips one L′ tile of segment 1 and commits O honestly: segment 1's block and the last segment's (its
    T_2 against the T_3 O implies) refuse almost every cell, and segment 2's (T_2 − T_1 still right) passes. A forged O
    fails the last block alone."""
    A, B, T, O, _ = _call()
    X, W, *_ = _np(SH, A, B)
    Xs = X.copy()
    Xs[:, 32:64] = 0
    Ts = [(np.cumsum([Xs[:, a:b] @ W[:, a:b].T for a, b in map(SH.seg, (1, 2, 3))], axis=0) & D.M16)[s] for s in range(3)]
    rng = random.Random(5)
    fails = {1: 0, 2: 0, 3: 0, "O": 0}
    for _ in range(12):
        i, j = rng.randrange(SH.m), rng.randrange(SH.n)
        for s in (1, 2, 3):
            t0 = int(Ts[s - 2][i, j]) if s > 1 else 0
            ins = D.block_inputs(SH, SEEDS, A[i], B[j], i, j, s, t0, int(Ts[s - 1][i, j]), O[i][j])
            fails[s] += 1 - (D.check_last(SH, SEEDS, i, j, ins) if s == 3 else D.check_block(ins))
        ins = D.block_inputs(SH, SEEDS, A[i], B[j], i, j, 3, T[1][i][j], o=O[i][j] ^ 1 << 20)
        fails["O"] += 1 - D.check_last(SH, SEEDS, i, j, ins)
    assert fails[1] >= 11 and fails[2] == 0 and fails[3] >= 11 and fails["O"] == 12


def test_out_dot_on_bits_is_its_word_at_the_extremes():
    """`PouwDitherOutDot_v1`'s 9-bit operands: a + da reaches 128 and -128, and the sum wraps at 2^32."""
    rng = random.Random(6)
    cases = [([127] * 32, [1] * 32, [127] * 32, [1] * 32, 0), ([0x80] * 32, [0] * 32, [127] * 32, [1] * 32, D.M32)]
    cases += [([rng.randrange(256) for _ in range(32)], [rng.randrange(2) for _ in range(32)],
                [rng.randrange(256) for _ in range(32)], [rng.randrange(2) for _ in range(32)], rng.randrange(1 << 32))
              for _ in range(4)]
    for a, da, b, db, acc in cases:
        bits = lambda ws, n: [(w >> e) & 1 for w in ws for e in range(n)]  # noqa: E731
        want = D.out_dot(acc, a, da, b, db)
        got = evaluate(D.PouwDitherOutDot, bits([acc], 32), bits(a, 8), bits(da, 1), bits(b, 8), bits(db, 1))
        assert int("".join(map(str, reversed(list(got)))), 2) == want
        word = lambda ds: sum(d << k for k, d in enumerate(ds))  # noqa: E731
        assert D.PouwDitherOutDotWord.evaluate(acc, *a, word(da), *b, word(db)) == want
        assert D.PouwDitherOutDot0Word.evaluate(*a, word(da), *b, word(db)) == (want - acc) & D.M32
    assert D.out_dot(0, [127] * 32, [1] * 32, [127] * 32, [1] * 32) == 32 * 128 * 128


def test_each_word_primitive_is_its_definitions_word_view():
    """A word primitive's operands are one word per leaf, which C-Flock's lowering hands its Boolean Definition's builder as
    that Definition's elements (`ir_lower._boolean_build`): so its ports are the word view of the Definition's
    (`boolean.word_type`), a dither row `Array<32, Value<1>>` being one `Value<32>`."""
    from verity.ml.circuits import boolean as BL

    for name in D.__all__:
        d = getattr(D, name)
        w = getattr(d, "word", None)
        if w is not None:
            params, ret = d.signature({})
            assert ([BL.word_type(t) for _, t in params], BL.word_type(ret)) == ([t for _, t in w.params], w.ret), name


def test_the_dot_step_is_each_cards_below_the_limit():
    """sm_120's wrapping IMMA (`PouwSegDot_v2`'s binding) and the H200's clamping step agree on every chain below
    `K_LIMIT` products, the extreme one included, and part at it: so every Definition refuses K_LIMIT."""
    assert [d.name for d in D.dot_steps("sm_120")] == [d.name for d in D.dot_steps("sm_90")] == ["PouwSegDot0", "PouwSegDot"]
    with pytest.raises(ValueError, match="no device record"):
        D.dot_steps("sm_80")
    rng = random.Random(7)
    for k, gen in ((4096, lambda: rng.randrange(256)), (D.K_LIMIT - 32, lambda: 0x80), (D.K_LIMIT, lambda: 0x80)):
        acc_w = acc_h = 0
        for _ in range(k // 32):
            a, b = [gen() for _ in range(32)], [gen() for _ in range(32)]
            acc_w = SG.seg_dot(acc_w, a, b)
            acc_h = HOPPER_SM90_S8_K32.step(acc_h, a, b)
        assert (acc_w == acc_h) == (k < D.K_LIMIT)
    with pytest.raises(ValueError, match="below"):
        D.Shape(128, 128, D.K_LIMIT, lp=128)


@pytest.mark.parametrize("bad", [dict(m=100), dict(lp=48), dict(L=48), dict(row=1000)])
def test_shape_refuses(bad):
    with pytest.raises(ValueError):
        D.Shape(**{**dict(m=128, n=128, k=256, lp=32, L=64), **bad})


def test_shape_refuses_t_past_8():
    """t = min(L, K) / L′ is at most 8: L 8,192 at L′ 256 is refused at K 16,384, and L′ 1,024 passes, as do the default
    and L ≥ K with L′ ≥ K / 8."""
    with pytest.raises(ValueError, match="at most 8"):
        D.Shape(128, 128, 16384, lp=256, L=8192)
    assert D.Shape(128, 128, 16384, lp=1024, L=8192).t == 8
    assert D.Shape(128, 128, 4096).t == 8
    assert D.Shape(128, 128, 4096, lp=512, L=8192).S == 1


@pytest.mark.parametrize("defn, params", [(D.PouwDitherBlock, dict(L=48, FIRST=True)),
                                          (D.PouwDitherBlock, dict(L=D.K_LIMIT, FIRST=True)),
                                          (D.PouwDitherLast, dict(K=96, L=40)),
                                          (D.PouwDitherLast, dict(K=D.K_LIMIT, L=1024))])
def test_definitions_refuse_lengths_off_32_or_past_the_limit(defn, params):
    with pytest.raises(ValueError, match="multiple of 32 below"):
        bind(defn, **params).signature()


# -- O at 16 bits and A′ ----------------------------------------------------------------------------------------------------------

#: a gate-up call ([gate ‖ up], I = 128) at K = 160 (S = 3) committing O at 16 bits and A′, and its consumer at K = I (S = 2)
PSH = D.Shape(128, 256, 160, lp=32, L=64, row=1024, ow=16, act=True)
CSH = D.Shape(128, 128, PSH.I, lp=32, L=64, row=1024)
CCOIN = hashlib.sha512(b"the consumer's coin").digest()


def _f32(xs) -> list[int]:
    return np.asarray(xs, dtype=np.float32).view(np.uint32).astype(int).tolist()


def _scales(sh: D.Shape, seed: int = 8) -> tuple[list[int], list[int]]:
    """s_a a row and s_w a column at a served W8A8 linear's sizes, so that the gate is a few units."""
    rng = np.random.default_rng(seed)
    return _f32(rng.uniform(0.006, 0.012, sh.m)), _f32(rng.uniform(0.0015, 0.003, sh.n))


def _receipt(*roots: bytes) -> bytes:
    """A stand-in for the digest of the certifier's receipt of the roots a call's operands are bound in."""
    return hashlib.sha512(b"".join(roots)).digest()


@functools.cache
def _producer():
    A, B = _operands(PSH, 13)
    T, O = D.values(PSH, SEEDS, A, B)
    sc = D.act_scales(PSH, *_scales(PSH))
    Ap, recs = D.acts(PSH, O, sc)
    return A, B, T, O, sc, Ap, recs, D.commit(PSH, D.values_buffer(PSH, T, O, Ap, recs), D.salts(KS, OFF))


def _amax(rec: bytes) -> int:
    return int.from_bytes(rec[:2], "little")


def _block(sh: D.Shape, seeds, A, B, cm: D.Tree, sc, i: int, j: int, s: int, o=None, amax=None,
           ap=None) -> list[list[int]]:
    """Block (i, j, s)'s inputs from its rows opened against `cm`'s root, with `o`, `amax` or `ap` in place of the opened
    O, record's amax or A′."""
    rows = _opened_rows(sh, cm, i, j, s)
    t0 = D.word_at(sh, rows, i, j, s - 1)
    if s < sh.S:
        return D.block_inputs(sh, seeds, A[i], B[j], i, j, s, t0, D.word_at(sh, rows, i, j, s))
    o = D.out_at(sh, rows, i, j) if o is None else o
    if not sh.gate(j):
        return D.block_inputs(sh, seeds, A[i], B[j], i, j, s, t0, o=o)
    amax = _amax(D.rec_at(sh, rows, i)) if amax is None else amax
    ap = D.act_at(sh, rows, i, j) if ap is None else ap
    return D.block_inputs(sh, seeds, A[i], B[j], i, j, s, t0, o=o, up=B[j + sh.I], sc=sc, amax=amax, ap=ap)


def _check(sh: D.Shape, seeds, i: int, j: int, s: int, ins) -> int:
    return D.check_last(sh, seeds, i, j, ins) if s == sh.S else D.check_block(ins)


def _codes(hs, amax: int) -> list[int]:
    return [D.wrap8(D.requant(h, amax)) for h in hs]


def test_o16_and_act_layout():
    """O at 16 bits takes a tile-segment's rows, and A′'s rows follow each tile-row's tiles, a row's I codes and its
    record in rows of `row` bytes with the last short: every value's leaf and offset is the one `leaf_of` lists it at, and
    the leaves' rows tile the values end to end. At 8B's gate-up (one segment, 16 KiB rows) a tile is O's two leaves and an
    A′ row one of 14,340 bytes. A row whose codes fill its leaves puts its record in a leaf of its own. A gate cell's last
    block opens its A′ and its row's record too; no Definition checks that block yet; today's shapes keep their layout
    and domain."""
    big = D.Shape(16384, 28672, 4096, lp=512, L=4096, row=16384, ow=16, act=True)
    assert (big.S, big.W, big.row_o, big.lpt, big.I, big.arow, big.apr, big.stride, big.leaves) == \
        (1, 0, 2, 2, 14336, 14340, 1, 576, 73728)
    assert big.leaf_of(448)[0] == D.ACT and big.leaf_span(448) == (448 * 16384, 14340)
    assert big.rec_leaf(0) == (448, 14336) and big.leaf_of(448)[1][-1] == (0, 14336)
    assert big.leaf_span(576) == (448 * 16384 + 128 * 14340, 16384) and big.leaf_of(576)[0] is None
    assert D.Shape(16384, 65536, 16384, lp=512, L=4096, row=16384, ow=16, act=True).apr == 3
    assert D.Shape(256, 128, 4096).label() == {"m": 256, "n": 128, "k": 4096, "lp": 128, "t": 8, "row": 16384, "leaves": 20}
    assert D.Shape(256, 128, 4096, ow=16).domain() != D.Shape(256, 128, 4096).domain()
    for sh in (D.Shape(256, 384, 256, lp=32, L=128, row=128, ow=16, act=True),
               D.Shape(128, 256, 64, lp=32, L=64, row=128, ow=16, act=True)):
        end = 0
        for pos in range(sh.leaves):
            kind, cells = sh.leaf_of(pos)
            off, n = sh.leaf_span(pos)
            width, rec = {D.ACT: 1, None: 2}.get(kind, 2), kind == D.ACT and cells[-1][1] == sh.I
            assert off == end and n == (len(cells) - rec) * width + rec * D.REC
            end += n
            for x, (i, j) in enumerate(cells[:len(cells) - rec:7]):
                got = sh.act_leaf(i, j) if kind == D.ACT else sh.o_leaf(i, j) if kind is None else sh.word_leaf(i, j, kind)
                assert got == (pos, width * 7 * x)
            if rec:
                assert sh.rec_leaf(cells[-1][0]) == (pos, n - D.REC)
        assert end == sh.nbytes == sh.tiles * sh.lpt * 128 + sh.m * (sh.I + D.REC)
    small = D.Shape(128, 256, 64, lp=32, L=64, row=128, ow=16, act=True)
    assert small.apr == 2 and small.leaf_of(small.act_leaves(3)[1]) == (D.ACT, [(3, 128)])
    sh = D.Shape(256, 384, 256, lp=32, L=128, row=128, ow=16, act=True)
    assert (sh.S, sh.W, sh.row_w, sh.row_o, sh.lpt, sh.I, sh.apr, sh.stride) == (2, 1, 256, 256, 512, 192, 2, 1792)
    assert [sh.leaf_span(p)[1] for p in sh.act_leaves(200)] == [128, 64 + D.REC]
    assert sh.opened(5, 9, 2) == [sh.word_leaf(5, 9, 1)[0], sh.o_leaf(5, 9)[0], sh.act_leaf(5, 9)[0], sh.rec_leaf(5)[0]]
    assert sh.opened(5, 200, 2) == [sh.word_leaf(5, 200, 1)[0], sh.o_leaf(5, 200)[0]]
    assert D.block_definition(sh, 1).name.startswith("PouwDitherBlock")
    for bad in (sh, D.Shape(256, 384, 256, lp=32, L=128, row=128, ow=16)):
        with pytest.raises(ValueError, match="no Definition"):
            D.block_definition(bad, bad.S)
    with pytest.raises(ValueError, match="32 or 16"):
        D.Shape(128, 128, 256, ow=8)
    with pytest.raises(ValueError, match="commits O"):
        D.Shape(128, 128, 256, o=False, act=True)
    with pytest.raises(ValueError, match="u16 below"):
        D.Shape(128, 131072, 256, act=True)


def test_act_map_is_the_served_definitions():
    """A′'s map is the served words: `dequant` is `PouwZyDequant_v1`'s (h1zy's), `silu_mul` is `SiluMulBf16_v2` on random
    and edge words (NaN, ±inf, ±0, subnormals, gates past expf's overflow), and `abs_max`, `act_scale` and `requant` are
    the served quantizer's, `Int8RowQuant_v1`'s scale and codes, on rows with NaN, ±inf, zeros, rows under the scale's
    floor and rows whose maximum sits at a binade's edge."""
    from verity.core.protocols.pouw.circuit import h1zy
    from verity.ml.circuits import evaluation as E
    from verity.ml.definitions.registry import act
    from verity.ml.definitions.registry import int8_exact as IE

    rng = np.random.default_rng(25)
    z = rng.integers(-(1 << 31), 1 << 31, 64).tolist() + [0, -1, (1 << 31) - 1, -(1 << 31)]
    sa, sw = _f32(rng.uniform(1e-4, 1e-1, len(z))), _f32(rng.uniform(1e-4, 1e-1, len(z)))
    sw[:3] = [0x7F800000, 0x7FC00000, 0x80000000]
    assert [D.dequant(*v) for v in zip(z, sa, sw)] == [h1zy.dequant(*v) for v in zip(z, sa, sw)]
    edges = [0x0000, 0x8000, 0x7F80, 0xFF80, 0x7FC0, 0xFFC1, 0x0001, 0x8001, 0x42B2, 0xC2B2, 0xC2B1, 0xC2A0, 0xC2A1, 0x3F80,
             0xBF80, 0x7F7F, 0xFF7F]
    words = edges + rng.integers(0, 1 << 16, 400).tolist()
    pairs = [(g, u) for g in edges for u in edges] + list(zip(words, rng.permutation(words).tolist()))
    assert act.SiluMulBf16.id == "SiluMulBf16_v2"
    assert [D.silu_mul(g, u) for g, u in pairs] == [act.SiluMulBf16.evaluate(g, u) for g, u in pairs]
    n = 16
    xb = rng.integers(0, 1 << 16, (14, n)).astype(np.uint16)
    xb[0, :] = 0
    xb[1, 3] = 0x7FC0
    xb[2, 5] = 0xFF80
    xb[3, :] = rng.integers(0, 0x0D00, n)
    xb[4, :2] = [0x4000, 0xBFFF]
    xb[5, :2] = [0x3FFF, 0x3FFE]
    xb[6:, :] = (rng.standard_normal((8, n)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    got = np.asarray(E.evaluate_batch(bind(IE.Int8RowQuant, K=n), xb)).astype(int)
    for r, g in zip(xb.tolist(), got):
        amax = D.abs_max(r)
        assert D.act_scale(amax) == g[n] and [D.requant(h, amax) for h in r] == (g[:n] & 0xFF).tolist(), r
    assert D.abs_max([0x7FC0, 0x3F80, 0xC000]) == 0x4000 and D.requant(0x7FC0, 0x3F80) == 0x81


def test_o16_act_honest_producer_and_consumer():
    """An honest gate-up call at O 16 bits with A′, and its consumer reading A′: every segment's block of the producer
    passes, its gate cells' last blocks with A′; its O and A′ rows are the reference's from only the cells they hold and
    the row's record; and the consumer, its seeds after a receipt of the producer's root, opens A′'s rows and records
    from that root as its A, passes, and its amax check passes on every row."""
    A, B, T, O, sc, Ap, recs, cm = _producer()
    assert len(cm.rows) == PSH.leaves == 320 and len({x for r in Ap for x in r}) > 64
    assert all(-127 <= x <= 127 for r in Ap for x in r) and {abs(max(r, key=abs)) for r in Ap} == {127}
    H = D.act_hs(PSH, O, sc)
    assert all(rec.amax == D.abs_max(h) and rec.k < PSH.I and h[rec.k] & 0x7FFF == rec.amax
               and all(h[j] & 0x7FFF != rec.amax for j in range(rec.k)) for rec, h in zip(recs, H))
    rng = random.Random(21)
    cells = [(0, 0), (127, 127), (5, 200), (64, 255), *((rng.randrange(PSH.m), rng.randrange(PSH.n)) for _ in range(6))]
    for i, j in cells:
        for s in range(1, PSH.S + 1):
            ins = _block(PSH, SEEDS, A, B, cm, sc, i, j, s)
            assert _check(PSH, SEEDS, i, j, s, ins) == 1, (i, j, s)
            if s == PSH.S:
                assert ins[7] == [O[i][j] & D.M16] and (len(ins) == 12) == (j < PSH.I)
    for pos in (PSH.o_leaf(3, 4)[0], PSH.word_leaf(9, 130, 2)[0]):
        assert D.leaf_row(PSH, SEEDS, A, B, pos) == cm.rows[pos]
    assert D.leaf_row(PSH, SEEDS, A, B, PSH.act_leaf(77, 5)[0], sc, recs[77]) == cm.rows[PSH.act_leaf(77, 5)[0]]
    cs = D.seeds(CCOIN, _receipt(cm.root), 0)
    Bc = _operands(CSH, 14)[1]
    Tc, Oc = D.values(CSH, cs, Ap, Bc)
    cc = D.commit(CSH, D.values_buffer(CSH, Tc, Oc), D.salts(KS, OFF + PSH.leaves))
    for i, j in ((0, 0), (127, 5), (40, 99)):
        rows = {p: cm.open(p) for p in PSH.act_leaves(i)}
        assert all(D.opens(cm.dom, cm.root, PSH.leaves, p, *r) for p, r in rows.items())
        a_i, rec = D.act_row(PSH, {p: r[0] for p, r in rows.items()}, i), D.rec_at(PSH, {p: r[0] for p, r in rows.items()}, i)
        assert a_i == Ap[i] and rec == recs[i].pack()
        for s in range(1, CSH.S + 1):
            assert _check(CSH, cs, i, j, s, _block(CSH, cs, {i: a_i}, Bc, cc, None, i, j, s)) == 1
    assert all(D.check_amax(PSH, SEEDS, A[i], B, i, recs[i].pack(), sc) == 1 for i in range(PSH.m))


def test_o16_a_forged_o_is_refused():
    """O at 16 bits forged in any one bit fails the last block of a gate cell and of an up cell, and a forged O row
    fails its opening against the honest root."""
    A, B, T, O, sc, Ap, recs, cm = _producer()
    rng = random.Random(22)
    for i, j in [(3, 4), (3, 132), *((rng.randrange(PSH.m), rng.randrange(PSH.n)) for _ in range(8))]:
        o = O[i][j] & D.M16 ^ 1 << rng.randrange(16)
        assert D.check_last(PSH, SEEDS, i, j, _block(PSH, SEEDS, A, B, cm, sc, i, j, PSH.S, o=o)) == 0, (i, j)
    p, off = PSH.o_leaf(3, 4)
    row, salt, path = cm.open(p)
    assert not D.opens(cm.dom, cm.root, PSH.leaves, p, row[:off] + bytes([row[off] ^ 1]) + row[off + 1:], salt, path)


def test_o16_a_forged_act_is_refused():
    """A′ forged at a gate cell fails that cell's last block: any other byte, and A′ mapped from an O whose committed 16
    bits are right and whose high bits aren't (O + 2^16, which the block's 32-bit dot refuses wherever the map moves).
    A forged A′ byte or record byte in a row the consumer opens fails its opening against the producer's root."""
    A, B, T, O, sc, Ap, recs, cm = _producer()
    rng = random.Random(23)
    cells = [(0, 0), (127, 127), *((rng.randrange(PSH.m), rng.randrange(PSH.I)) for _ in range(10))]
    moved = 0
    for i, j in cells:
        bad = (Ap[i][j] + rng.randrange(1, 256)) & 0xFF
        assert D.check_last(PSH, SEEDS, i, j, _block(PSH, SEEDS, A, B, cm, sc, i, j, PSH.S, ap=bad)) == 0
        hi = D.act_word(O[i][j] + (1 << 16), O[i][j + PSH.I], sc.sa[i], sc.sw[j], sc.sw[j + PSH.I], recs[i].amax)
        ins = _block(PSH, SEEDS, A, B, cm, sc, i, j, PSH.S, ap=hi)
        assert D.check_last(PSH, SEEDS, i, j, ins) == int(hi == Ap[i][j] & 0xFF) and ins[7] == [O[i][j] & D.M16]
        moved += hi != Ap[i][j] & 0xFF
    assert moved >= len(cells) - 2
    for p, off in (PSH.act_leaf(77, 5), PSH.rec_leaf(77)):
        row, salt, path = cm.open(p)
        assert not D.opens(cm.dom, cm.root, PSH.leaves, p, row[:off] + bytes([row[off] ^ 1]) + row[off + 1:], salt, path)


def test_o16_a_forged_amax_is_refused():
    """A prover that sets row i's amax to the row's second-largest |h| and names that cell passes the consumer's amax
    check, and with A′ formed under that amax every gate cell's last block but the maximum's passes: the clamp codes the
    maximum ±127 as before, and its block refuses |h| > amax. The consumer refuses an amax one step above the maximum
    (attained by no cell, whichever column the record names), a record naming a column off the maximum, and one naming a
    column ≥ I."""
    A, B, T, O, sc, Ap, recs, cm = _producer()
    H = D.act_hs(PSH, O, sc)
    rng = random.Random(27)
    for i in [0, 127, *rng.sample(range(1, 127), 6)]:
        top, second = sorted({h & 0x7FFF for h in H[i]}, reverse=True)[:2]
        codes = _codes(H[i], second)
        forged = D.ActRecord(second, D.argmax(H[i], second))
        kmax = [j for j, h in enumerate(H[i]) if h & 0x7FFF == top]
        assert codes != Ap[i] and all(abs(codes[j]) == 127 for j in kmax)
        assert D.check_amax(PSH, SEEDS, A[i], B, i, forged.pack(), sc) == 1
        for j in kmax[:1] + rng.sample([j for j in range(PSH.I) if j not in kmax], 3):
            ins = _block(PSH, SEEDS, A, B, cm, sc, i, j, PSH.S, amax=second, ap=codes[j])
            assert D.check_last(PSH, SEEDS, i, j, ins) == int(j not in kmax), (i, j)
        rec = recs[i]
        for k in (rec.k, forged.k, rng.randrange(PSH.I)):
            assert D.check_amax(PSH, SEEDS, A[i], B, i, D.ActRecord(top + 1, k).pack(), sc) == 0
        for k in (forged.k, PSH.I, D.NO_INDEX):
            assert D.check_amax(PSH, SEEDS, A[i], B, i, D.ActRecord(rec.amax, k).pack(), sc) == 0
        assert D.check_amax(PSH, SEEDS, A[i], B, i, rec.pack(), sc) == 1


def test_o16_a_negative_maximum_is_bound():
    """A row whose maximum is negative: the served map's [−128, 127] codes a forged amax's clipped maximum −128, outside a
    set of columns at |A′| = 127, and the quantizer here codes it −127. Under the row's second-largest |h| the maximum's
    gate cell refuses |h| > amax and its −127, and a −128 code at the honest amax is refused as a forged A′ byte."""
    A, B, T, O, sc, Ap, recs, cm = _producer()
    H = D.act_hs(PSH, O, sc)
    neg = [i for i in range(PSH.m) if H[i][recs[i].k] & 0x8000]
    assert len(neg) > 16
    for i in neg[:4]:
        k, top = recs[i].k, recs[i].amax
        second = max(h & 0x7FFF for h in H[i] if h & 0x7FFF < top)
        assert Ap[i][k] == -127 and D.requant(H[i][k], second) == 0x81
        assert D.check_amax(PSH, SEEDS, A[i], B, i, D.ActRecord(second, D.argmax(H[i], second)).pack(), sc) == 1
        assert D.check_last(PSH, SEEDS, i, k, _block(PSH, SEEDS, A, B, cm, sc, i, k, PSH.S, amax=second, ap=0x81)) == 0
        assert D.check_last(PSH, SEEDS, i, k, _block(PSH, SEEDS, A, B, cm, sc, i, k, PSH.S, ap=0x80)) == 0
        assert D.check_last(PSH, SEEDS, i, k, _block(PSH, SEEDS, A, B, cm, sc, i, k, PSH.S)) == 1


def test_amax_check_bounds_the_maximum_by_magnitude():
    """At the top of a binade an amax one bf16 step below the row's maximum still rounds the maximum to 127 (2 / RN(1.9921875
    / 127) ≈ 127.498), so a rounding check alone passes it: `bounded` refuses it on |h| > amax, and the consumer passes the
    record naming the second cell. Zero rows and rows under the scale's floor take no special case; the first of tied
    maxima is named; a row of NaN alone has no column to name. A NaN h, which amax skips and the quantizer codes −127,
    passes the bound. An amax that is NaN or negative is refused, and +inf, a row's honest amax beside an infinite h,
    passes."""
    from verity.ml.circuits.ops import fp32 as F

    def ok(row, rec):
        return D.amax_ok(len(row), rec.pack(), lambda k: row[k])

    top, below = 0x4000, 0x3FFF
    row = [top, below] + [0x3F00] * 30
    q = float(F.f32_of(F.f32_div_bits(F.bf16_to_f32_bits(top), D.act_scale(below))))
    assert 127.49 < q < 127.5 and D.requant(top, below) == 127
    assert _codes(row, below)[:2] == [127, 127] and not D.bounded(top, below) and D.bounded(below, below)
    assert ok(row, D.ActRecord(below, 1)) == 1
    honest, rec = D.act_codes(row)
    assert rec == D.ActRecord(top, 0) and ok(row, rec) == 1 and all(D.bounded(h, top) for h in row)
    for r, want in (([0x0001, 0x0080, 0x8100] * 4, D.ActRecord(0x0100, 2)), ([0x8000, 0] * 4, D.ActRecord(0, 0)),
                    ([0x3F00] + [0xBF80, 0x3F80] * 9, D.ActRecord(0x3F80, 1))):
        codes, rec = D.act_codes(r)
        assert rec == want and ok(r, rec) == 1 and all(D.bounded(h, rec.amax) for h in r), r
    assert set(D.act_codes([0x0001, 0x0080, 0x8100])[0]) == {0}
    assert D.act_codes([0x7FC0, 0xFFC1])[1] == D.ActRecord(0, D.NO_INDEX)
    nan = [0x3F80, 0x7FC0, 0xFFC0, 0x3F00]
    codes, rec = D.act_codes(nan)
    assert rec == D.ActRecord(0x3F80, 0) and codes[1:3] == [-127, -127] and ok(nan, rec) == 1
    assert all(D.bounded(h, 0x3F80) for h in nan) and not D.bounded(0x3F80, 0x3F00)
    assert ok([0x7FC0], D.ActRecord(0x7FC0, 0)) == 0 and not D.bounded(0, 0x7FC0)
    assert ok([0xBF80], D.ActRecord(0xBF80, 0)) == 0 and not D.bounded(0, 0x8000)
    inf = [0x3F80, 0xFF80]
    assert D.act_codes(inf)[1] == D.ActRecord(0x7F80, 1) and ok(inf, D.ActRecord(0x7F80, 1)) == 1
    assert D.amax_ok(4, D.ActRecord(0x3F80, 4).pack(), lambda k: 0x3F80) == 0


@pytest.mark.parametrize("k", [64, 128])
def test_o16_a_guessed_act_passes_only_with_the_o16_it_guesses(k):
    """Requant is many-to-one, so A′ alone is guessed: at A = B = 0 a prover that skips the work commits A′ = 0 under an
    amax past every |h|, which every gate cell's A′ check alone (`act_ok`) passes and the consumer's amax check refuses
    (no column's |h| is that amax). The last block also checks O mod 2^16, which it has to guess: with T_{S−1}
    honest (S = 1 at K 64, 2 at K 128), exactly the cells whose O mod 2^16 is the guess pass (O = Σ d_A d_B ~ Bin(K, 1/4),
    the guess its mode ⌊(K + 1)/4⌋), about a tenth of them."""
    sh = D.Shape(128, 256, k, lp=32, L=64, row=1024, ow=16, act=True)
    A, B = [[0] * k] * sh.m, [[0] * k] * sh.n
    sc = D.ActScales(*_scales(sh))
    g = (k + 1) // 4
    rng = random.Random(24)
    cells = sorted({(rng.randrange(sh.m), rng.randrange(sh.I)) for _ in range(200)})
    a_only = passed = 0
    for i, j in cells:
        Tc, o = D.cell_values(sh, SEEDS, A[i], B[j], i, j)
        ins = D.block_inputs(sh, SEEDS, A[i], B[j], i, j, sh.S, Tc[-2] if sh.S > 1 else 0, o=g, up=B[j + sh.I], sc=sc,
                             amax=0x7F7F, ap=0)
        a_only += D.act_ok(sh, ins)
        got = D.check_last(sh, SEEDS, i, j, ins)
        assert got == int(o & D.M16 == g), (i, j, o)
        passed += got
    assert a_only == len(cells) and 0 < passed < len(cells) / 5
    assert not any(D.check_amax(sh, SEEDS, A[0], B, 0, D.ActRecord(0x7F7F, k).pack(), sc) for k in range(0, sh.I, 17))
