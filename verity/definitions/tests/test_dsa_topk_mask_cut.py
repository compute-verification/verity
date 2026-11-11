"""`DsaTopKMaskCounted_v1{N,TOPK}` cut into units of at most a Flock block each (`registry.dsa_topk_mask_cut`), held bit for bit
to the whole mask:

* the twin (`values`) is `DsaTopKMaskCounted_v1`'s mask, as the circuit evaluator computes it;
* each unit's Boolean Definition, evaluated on what the units before it computed (its block of the row, n, its writers'
  outputs), computes the twin's outputs, and the mask units' outputs in block order are the Call's mask: on rows that tie at the
  boundary, straddle signed zero and hold NaNs and infinities, at n from 0 past N, at small statics (blocks of a few keys, a
  short last block joined to the one before, TOPK 1 and TOPK a whole block) and, with ``VERITY_ML_DEFINITIONS_SLOW`` set, at the
  served ones (TOPK 2,048, the first class past `WHOLE_MAX` and 131,072).  Every port is whole u16 words: each writer's bits
  past its value are 0, and each reader's outputs are the same with them random;
* `DsaTopKMaskWhole_v1`, the mask up to `WHOLE_MAX` a unit, is the counted mask in u16 words;
* `DsaTopKDigitPick_v1`, block b's counts its own port ``counts_b``, is gate for gate the pick over one port of the M blocks'
  counts, at small statics and at the served ones (M = 8, each round, TOPK 2,048);
* `plan` wires each parameter to one earlier unit's output of its width and `cut` counts its units, None up to `WHOLE_MAX`.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from verity import settings
from verity.circuits import boolean as BL
from verity.circuits import evaluation as E
from verity.circuits.boolean.trace import Traced, trace
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import resolve
from verity.circuits.types import Array
from verity.definitions.registry import boolean_dsa_topk_mask as BM
from verity.definitions.registry import dsa_topk_mask as M
from verity.definitions.registry import dsa_topk_mask_cut as X

POOL = [0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x7FC00000, 0xFFC00000, 0x7F800000, 0xFF800000, 0x00000001, 0x80000001]
SERVED_TOPK = 2048


def _rows(rng, N: int, R: int, TOPK: int) -> tuple[list[list[int]], list[int]]:
    """R random rows with repeats and specials, rows of a few values (ties at the boundary) and constant ones (every key tied),
    each with an n: none, fewer than TOPK, TOPK, a few past it, half, all, past N, u32's top bit and top."""
    rows = [[rng.choice(POOL) if rng.random() < 0.5 else rng.getrandbits(32) for _ in range(N)] for _ in range(R)]
    rows += [[rng.choice(POOL[:3]) for _ in range(N)] for _ in range(max(1, R // 8))] + [[w] * N for w in (POOL[0], POOL[4])]
    ns = [0, 1, TOPK - 1, TOPK, TOPK + 1, N // 2, N - 1, N, N + 1, 1 << 31, (1 << 32) - 1]
    return rows, [ns[i % len(ns)] if i < len(ns) else rng.choice(ns + [rng.randrange(N + 1)]) for i in range(len(rows))]


def _bits(words, width: int) -> np.ndarray:
    """Words as `evaluate_bits` takes them: each split into its ``width`` bits, LSB first, on a last axis."""
    w = np.asarray(words, dtype=np.uint64)
    return ((w[..., None] >> np.arange(width, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def _words(bits: np.ndarray) -> list[int]:
    """Each row of bits (LSB first) as a word."""
    return [int(x) for x in (bits.astype(np.uint64) << np.arange(bits.shape[-1], dtype=np.uint64)).sum(axis=-1)]


def _fields(defn) -> list[int]:
    """The bits of each field of ``defn``'s return."""
    _, ret = resolve(defn).signature()
    return [sum(BL.elements(f[1] if isinstance(f, tuple) else f)) for f in (getattr(ret, "fields", None) or (ret,))]


def _read(u: X.Unit, param: str, TOPK: int) -> int | None:
    """The bits of a wired parameter its unit reads: a prefix its low 8K, an r its low RB; None for all of them."""
    return {"prefix": X.DIGIT * u.defn.bindings.get("K", 0), "r": X._rb(TOPK)}.get(param)


def _chain(rows, ns, TOPK: int, block: int, rng: random.Random | None = None) -> dict[tuple, list[np.ndarray]]:
    """Every unit of the plan evaluated on bits (`evaluate_bits`, a row an instance) on its block of the rows, n and its
    writers' outputs as the units before it computed them: each unit's output fields, by key.  With ``rng``, the bits of a
    wired input past those its unit reads (`_read`) are random in place of its writer's."""
    R, N = len(rows), len(rows[0])
    scores, out = np.asarray(rows, dtype=np.uint64), {}
    for u in X.plan(N, TOPK, block):
        args = []
        if u.keys != (0, 0):
            args += [_bits(scores[:, u.keys[0]:u.keys[1]], 32).reshape(R, -1), _bits(ns, 32)]
        for p, w, f in u.wires:
            a, lo = out[w][f].copy(), _read(u, p, TOPK)
            if rng is not None and lo is not None:
                a[:, lo:] = np.asarray([[rng.getrandbits(1) for _ in range(a.shape[1] - lo)] for _ in range(R)], dtype=a.dtype)
            args.append(a)
        got = np.asarray(BS.evaluate_bits(u.defn, *args)).reshape(R, -1)
        at = np.cumsum([0, *_fields(u.defn)])
        out[u.key] = [got[:, a:b] for a, b in zip(at, at[1:])]
    return out


def _check(N: int, TOPK: int, block: int, R: int, seed: int) -> None:
    rows, ns = _rows(random.Random(seed), N, R, TOPK)
    out = _chain(rows, ns, TOPK, block)
    junk = _chain(rows, ns, TOPK, block, random.Random(seed + 1))
    assert all(np.array_equal(a, b) for k in out for a, b in zip(out[k], junk[k])), (N, TOPK, block)
    whole = np.asarray(E.evaluate_batch(bind(M.DsaTopKMaskCounted, N=N, TOPK=TOPK), rows, ns)).reshape(len(rows), -1)
    bl = X.blocks(N, TOPK, block)
    assert all(not out[("mask", b)][0][:, e - a:].any() for b, (a, e) in enumerate(bl)), (N, TOPK, block)
    mask = np.concatenate([out[("mask", b)][0][:, :e - a] for b, (a, e) in enumerate(bl)], axis=1)
    assert np.array_equal(mask, whole), (N, TOPK, block)
    for i, (row, n) in enumerate(zip(rows, ns)):
        for key, want in X.values(row, n, TOPK, block).items():
            if key[0] == "count":
                got = (tuple(_words(out[key][0][i].reshape(X.BINS, 32))),)
            elif key[0] == "mask":
                got = (tuple(int(x) for x in out[key][0][i][:len(want[0])]),)
            else:
                got = tuple(_words(f[i:i + 1])[0] for f in out[key])
            assert got == want, (N, TOPK, block, key, i)


@pytest.mark.parametrize("N,TOPK,block", [(20, 4, 8), (21, 4, 8), (18, 4, 8), (40, 5, 16), (9, 1, 4), (30, 16, 16), (16, 8, 8)])
def test_the_twin_is_the_counted_mask(N, TOPK, block) -> None:
    rows, ns = _rows(random.Random(N * 100 + TOPK), N, 64, TOPK)
    whole = np.asarray(E.evaluate_batch(bind(M.DsaTopKMaskCounted, N=N, TOPK=TOPK), rows, ns)).reshape(len(rows), -1)
    for row, n, want in zip(rows, ns, whole):
        v = X.values(row, n, TOPK, block)
        assert [x for b in range(len(X.blocks(N, TOPK, block))) for x in v[("mask", b)][0]] == want.tolist()


@pytest.mark.slow
@pytest.mark.parametrize("N,TOPK,block", [(20, 4, 8), (18, 4, 8), (9, 1, 4), (30, 16, 16)])
def test_each_unit_is_its_twin_and_the_units_compose_to_the_counted_mask(N, TOPK, block) -> None:
    """Blocks of 8 keys over 20 (a last block of 4) and 18 (its 2 keys joined to the block before), TOPK 1, and one block.
    Each writer's bits past its value are 0, and each reader's outputs are the same with them random."""
    _check(N, TOPK, block, 24, N * 7 + TOPK)


@pytest.mark.parametrize("N,TOPK", [(20, 4), (16, 4), (9, 1), (30, 16), (40, 5)])
def test_the_whole_mask_is_the_counted_mask_in_words(N, TOPK) -> None:
    """`DsaTopKMaskWhole_v1`'s output is `DsaTopKMaskCounted_v1`'s mask, then 0 to its last u16 word."""
    rows, ns = _rows(random.Random(N * 31 + TOPK), N, 24, TOPK)
    whole = np.asarray(E.evaluate_batch(bind(M.DsaTopKMaskCounted, N=N, TOPK=TOPK), rows, ns)).reshape(len(rows), -1)
    t = bind(X.DsaTopKMaskWhole, N=N, TOPK=TOPK)
    got = np.asarray(BS.evaluate_bits(t, _bits(rows, 32).reshape(len(rows), -1), _bits(ns, 32))).reshape(len(rows), -1)
    assert _fields(t) == [X.wide(N)] and t.signature()[1] == X.words(N)
    assert np.array_equal(got[:, :N], whole) and not got[:, N:].any()


@pytest.mark.slow
@pytest.mark.skipif(not settings.VERITY_ML_DEFINITIONS_SLOW,
                    reason="VERITY_ML_DEFINITIONS_SLOW=1 runs it (the served statics' units, built and evaluated whole)")
@pytest.mark.parametrize("N", [X.WHOLE_MAX + 2048, 131072])
def test_the_units_compose_to_the_counted_mask_at_served_statics(N) -> None:
    """At TOPK 2,048 and `BLOCK`: the first class the cut serves and the longest row at 128k context, 8 blocks.  The small
    statics' test holds the same units' logic; this one holds the served Definitions themselves (the 16,384-key blocks' trie
    from AT, the popcount and tie-count widths, 8 blocks' counts in a pick), which no smaller binding builds."""
    _check(N, SERVED_TOPK, X.BLOCK, 54, N)


@pytest.mark.parametrize("N,TOPK,block", [(20, 4, 8), (18, 4, 8), (X.WHOLE_MAX + 2048, 2048, X.BLOCK), (131072, 2048, X.BLOCK)])
def test_plan_wires_each_parameter_to_one_earlier_output_of_its_width(N, TOPK, block) -> None:
    units = X.plan(N, TOPK, block)
    bl = X.blocks(N, TOPK, block)
    written: dict[tuple, list[int]] = {}
    for u in units:
        params, _ = resolve(u.defn).signature()
        width = {name: sum(BL.elements(t)) for name, t in params}
        got = {p: written[w][f] for p, w, f in u.wires}
        assert len(got) == len(u.wires), u.key
        operands = {"scores", "n"} if u.keys != (0, 0) else set()
        assert set(width) == operands | set(got), u.key
        assert all(got[p] == width[p] for p in got), u.key
        if u.key[0] == "pick":
            assert [n for n, _ in params[:len(bl)]] == [f"counts_{b}" for b in range(len(bl))]
            assert [w for p, w, _ in u.wires if p.startswith("counts_")] == [("count", u.key[1], b) for b in range(len(bl))]
        if operands:
            assert width["scores"] == 32 * (u.keys[1] - u.keys[0]) and u.keys == bl[u.key[-1]]
        written[u.key] = _fields(u.defn)
    assert [u.keys for u in units if u.key[0] == "mask"] == bl and bl[0][0] == 0 and bl[-1][1] == N
    assert all(a[1] == b[0] for a, b in zip(bl, bl[1:])) and all(e - a >= TOPK for a, e in bl)


def _one_port_pick(S) -> Traced:
    """The pick's circuit with the M blocks' counts one parameter, Array<M, Array<BINS, I32>>, block b's at b."""
    params, ret = X._pick_sig(S)
    rest = [t for name, t in params if not name.startswith("counts_")]
    return trace(X._pick(S), [Array(S.M, Array(X.BINS, X.I32)), *rest], ret)


@pytest.mark.parametrize("M_,K,TOPK", [(m, k, t) for m, t in ((1, 4), (2, 4), (3, 16)) for k in range(X.ROUNDS)]
                         + [pytest.param(8, k, SERVED_TOPK, marks=pytest.mark.slow) for k in range(X.ROUNDS)])
def test_the_pick_with_a_port_a_block_is_the_one_port_pick_gate_for_gate(M_, K, TOPK) -> None:
    """The served bindings (8 blocks, TOPK 2,048's r) are the ones the cut calls, and no smaller binding's trace is theirs."""
    S = SimpleNamespace(M=M_, K=K, TOPK=TOPK)
    split, one = X.DsaTopKDigitPick.traced(S), _one_port_pick(S)
    assert (split.n_inputs, split.outputs, split.gates) == (one.n_inputs, one.outputs, one.gates)
    params, _ = X._pick_sig(S)
    assert [name for name, _ in params] == [f"counts_{b}" for b in range(M_)] + (["prefix", "r"] if K else [])
    assert all(t == Array(X.BINS, X.I32) for _, t in params[:M_])


@pytest.mark.parametrize("family", [M.DsaTopKMaskCounted, BM.DsaTopKMaskCountedV2])
def test_the_cut_counts_its_units_past_whole_max(family) -> None:
    assert X.cut(bind(family, N=X.WHOLE_MAX, TOPK=SERVED_TOPK)) is None
    for N in (X.WHOLE_MAX + 2048, 65536, 131072):
        M_ = -(-N // X.BLOCK)
        cut = X.cut(bind(family, N=N, TOPK=SERVED_TOPK))
        assert sum(n for _, n in cut) == 5 * M_ + 4 == len(X.plan(N, SERVED_TOPK))
    assert M.CUTS == {"DsaTopKMaskCounted_v1": X.__name__} and BM.CUTS == {"DsaTopKMaskCounted_v2": X.__name__}
