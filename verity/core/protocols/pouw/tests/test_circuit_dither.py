"""Dither's served unit (`circuit.dither`): the coins and the filler's per-call pad, a call's 16-bit words and O, the leaves
a kernel's sampled cells are checked against, the operand roots, the Definitions on bits against the reference, and the
dot step's binding on both cards."""
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
    all S segments' words and no O, and a call of one segment commits O alone, which earns no credit."""
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
    and L ≥ K with L′ ≥ K / 8, a shape that commits O alone and earns no credit."""
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
