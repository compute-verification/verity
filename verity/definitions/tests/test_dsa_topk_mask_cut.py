"""`DsaTopKMaskCounted_v1{N,TOPK}` cut into units of at most a Flock block each (`registry.dsa_topk_mask_cut`), held bit for bit
to the whole mask:

* the twin (`values`) is `DsaTopKMaskCounted_v1`'s mask, as the circuit evaluator computes it;
* each unit's Boolean Definition, evaluated on what the units before it computed (its block of the row, n, its writers'
  outputs), computes the twin's outputs, and the mask units' outputs in block order are the Call's mask: on rows that tie at the
  boundary, straddle signed zero and hold NaNs and infinities, at n from 0 past N, at small statics (blocks of a few keys, a
  short last block joined to the one before, TOPK 1 and TOPK a whole block) and, with ``VERITY_ML_DEFINITIONS_SLOW`` set, at the
  served ones (TOPK 2,048, the first class past `WHOLE_MAX` and 131,072);
* `plan` wires each parameter to earlier units' outputs of its width and `cut` counts its units, None up to `WHOLE_MAX`.
"""
from __future__ import annotations

import random

import numpy as np
import pytest

from verity import settings
from verity.circuits import boolean as BL
from verity.circuits import evaluation as E
from verity.circuits.defs import bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import resolve
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


def _chain(rows, ns, TOPK: int, block: int) -> dict[tuple, list[np.ndarray]]:
    """Every unit of the plan evaluated on bits (`evaluate_bits`, a row an instance) on its block of the rows, n and its
    writers' outputs as the units before it computed them: each unit's output fields, by key."""
    R, N = len(rows), len(rows[0])
    scores, out = np.asarray(rows, dtype=np.uint64), {}
    for u in X.plan(N, TOPK, block):
        args = []
        if u.keys != (0, 0):
            args += [_bits(scores[:, u.keys[0]:u.keys[1]], 32).reshape(R, -1), _bits(ns, 32)]
        wired: list[tuple[str, list]] = []
        for p, w, f in u.wires:
            if not wired or wired[-1][0] != p:
                wired.append((p, []))
            wired[-1][1].append(out[w][f])
        args += [np.concatenate(parts, axis=1) for _, parts in wired]
        got = np.asarray(BS.evaluate_bits(u.defn, *args)).reshape(R, -1)
        at = np.cumsum([0, *_fields(u.defn)])
        out[u.key] = [got[:, a:b] for a, b in zip(at, at[1:])]
    return out


def _check(N: int, TOPK: int, block: int, R: int, seed: int) -> None:
    rows, ns = _rows(random.Random(seed), N, R, TOPK)
    out = _chain(rows, ns, TOPK, block)
    whole = np.asarray(E.evaluate_batch(bind(M.DsaTopKMaskCounted, N=N, TOPK=TOPK), rows, ns)).reshape(len(rows), -1)
    mask = np.concatenate([out[("mask", b)][0] for b in range(len(X.blocks(N, TOPK, block)))], axis=1)
    assert np.array_equal(mask, whole), (N, TOPK, block)
    for i, (row, n) in enumerate(zip(rows, ns)):
        for key, want in X.values(row, n, TOPK, block).items():
            if key[0] == "count":
                got = (tuple(_words(out[key][0][i].reshape(X.BINS, 32))),)
            elif key[0] == "mask":
                got = (tuple(int(x) for x in out[key][0][i]),)
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
    """Blocks of 8 keys over 20 (a last block of 4) and 18 (its 2 keys joined to the block before), TOPK 1, and one block."""
    _check(N, TOPK, block, 24, N * 7 + TOPK)


@pytest.mark.slow
@pytest.mark.skipif(not settings.VERITY_ML_DEFINITIONS_SLOW,
                    reason="VERITY_ML_DEFINITIONS_SLOW=1 runs it: about 2 min at 43,008 keys and 3.5 at 131,072 on a CPU node, 4 GB")
@pytest.mark.parametrize("N", [X.WHOLE_MAX + 2048, 131072])
def test_the_units_compose_to_the_counted_mask_at_served_statics(N) -> None:
    """At TOPK 2,048 and `BLOCK`: the first class the cut serves and the longest row at 128k context, 8 blocks.  The small
    statics' test holds the same units' logic; this one holds the served Definitions themselves (the 16,384-key blocks' trie
    from AT, the popcount and tie-count widths, 8 blocks' counts in a pick), which no smaller binding builds."""
    _check(N, SERVED_TOPK, X.BLOCK, 54, N)


@pytest.mark.parametrize("N,TOPK,block", [(20, 4, 8), (18, 4, 8), (X.WHOLE_MAX + 2048, 2048, X.BLOCK), (131072, 2048, X.BLOCK)])
def test_plan_wires_each_parameter_to_earlier_outputs_of_its_width(N, TOPK, block) -> None:
    units = X.plan(N, TOPK, block)
    bl = X.blocks(N, TOPK, block)
    written: dict[tuple, list[int]] = {}
    for u in units:
        params, _ = resolve(u.defn).signature()
        width = {name: sum(BL.elements(t)) for name, t in params}
        got: dict[str, int] = {}
        for p, w, f in u.wires:
            got[p] = got.get(p, 0) + written[w][f]
        operands = {"scores", "n"} if u.keys != (0, 0) else set()
        assert set(width) == operands | set(got), u.key
        assert all(got[p] == width[p] for p in got), u.key
        if operands:
            assert width["scores"] == 32 * (u.keys[1] - u.keys[0]) and u.keys == bl[u.key[-1]]
        written[u.key] = _fields(u.defn)
    assert [u.keys for u in units if u.key[0] == "mask"] == bl and bl[0][0] == 0 and bl[-1][1] == N
    assert all(a[1] == b[0] for a, b in zip(bl, bl[1:])) and all(e - a >= TOPK for a, e in bl)


@pytest.mark.parametrize("family", [M.DsaTopKMaskCounted, BM.DsaTopKMaskCountedV2])
def test_the_cut_counts_its_units_past_whole_max(family) -> None:
    assert X.cut(bind(family, N=X.WHOLE_MAX, TOPK=SERVED_TOPK)) is None
    for N in (X.WHOLE_MAX + 2048, 65536, 131072):
        M_ = -(-N // X.BLOCK)
        cut = X.cut(bind(family, N=N, TOPK=SERVED_TOPK))
        assert sum(n for _, n in cut) == 5 * M_ + 4 == len(X.plan(N, SERVED_TOPK))
    assert M.CUTS == {"DsaTopKMaskCounted_v1": X.__name__} and BM.CUTS == {"DsaTopKMaskCounted_v2": X.__name__}
