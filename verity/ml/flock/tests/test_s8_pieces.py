"""Flock's pieces for the exact s8 family's word primitives: the sm_120 IMMA steps (`ops.prims`' satfinite s8 x s8 and wrapping
u8 x s8) and `LimbSplit_v1`'s four limb steps (`ops.s8`).  Each lowers by its Boolean version's builder (`ir_lower._boolean_build`:
`ops.boolean.gemm`, `ops.boolean.s8`).  Lowered by `ir_lower.lower` into a Flock circuit and evaluated there, each agrees with
its Definition's reference on uniform words and on the edges: s32's ends against the operand corners (the clamp, the wrap), the
clamp's thresholds at +-127, the carry's at each byte.  The limb splits and an s8 dot over the u8 x s8 step lower piece by piece
and agree too; the dot's nonzeros grow by one step's own per step, so a K-long chain stays linear in K.  The pieces resolve in
a process that never imported their Boolean modules."""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops import s8 as S8
from verity.ml.circuits.silicon.integer import S32_MAX, S32_MIN, decode_int8, s32_value
from verity_flock import ir_lower as IL
from verity_flock.gf2 import Circuit

STEPS = [P.BlackwellS8ImmaDot32, P.BlackwellU8S8ImmaDot32]
LIMB_STEPS = [S8.S32LowByte, S8.S32ByteCarry, S8.S32ClampS8, S8.S32ClampRest]
SPLITS = [bind(S8.LimbSplit, FORM="bytes", N=3, Y=(-127 * 65793, 127 * 65793)),
          bind(S8.LimbSplit, FORM="bytes", N=4, Y=(-(1 << 30), 1 << 30)),
          bind(S8.LimbSplit, FORM="byteclamp", N=3, Y=(-65152, 65151))]
U8S8 = {"DOT": P.BlackwellU8S8ImmaDot32, "A": (0, 255), "B": (-128, 127)}
CODES = (0x00, 0x01, 0x7F, 0x80, 0x81, 0xFF)
ACCS = (S32_MIN, S32_MIN + 1, -(1 << 20), -1, 0, 1, 1 << 20, S32_MAX - 1, S32_MAX)


def _lower(target) -> tuple[Circuit, list[int], int]:
    params, _ = E.batch.signature_of(target)
    widths = [w for pt in params for w in pt.leaf_widths()]
    C = Circuit()
    ins = [C.input(f"in{j}", w) for j, w in enumerate(widths)]
    outs = IL.lower(C, target, ins)
    for k, o in enumerate(outs):
        C.output(f"out{k}", o)
    return C, widths, len(outs)


def _agree(target, *inputs: np.ndarray) -> np.ndarray:
    """The lowering's outputs on ``inputs`` (one array a parameter, a row an instance), held to the reference's; the reference."""
    C, widths, n_out = _lower(target)
    cols = [c.astype(np.uint64) for x in inputs for c in (x.reshape(len(x), -1).T)]
    assert len(cols) == len(widths)
    T = len(cols[0])
    got_s, fail = C.evaluate({f"in{j}": IL.slice_words(c, w) for j, (c, w) in enumerate(zip(cols, widths))}, T)
    assert not fail
    got = np.stack([IL.unslice_words(got_s[f"out{k}"], T) for k in range(n_out)], axis=1)
    ref = E.evaluate_batch(target, *inputs).astype(np.uint64).reshape(T, -1)
    bad = np.flatnonzero((got != ref).any(axis=1))
    assert not bad.size, f"{target.id}: {bad.size} of {T} instances differ from the reference, first {bad[:5].tolist()}"
    return ref


def _words(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).astype(np.uint32)


@pytest.mark.parametrize("prim", STEPS, ids=lambda d: d.name)
def test_the_imma_step_pieces_compute_their_steps(prim) -> None:
    """Uniform words, and every operand corner (0, 1, 127, 128, 129, 255 as codes, each lane alike) against every s32 edge of
    the accumulator: the satfinite step clamps there and the u8 x s8 step wraps, in both directions."""
    rng = np.random.default_rng(41)
    n = 3000
    acc = rng.integers(0, 1 << 32, n, dtype=np.uint64).astype(np.uint32)
    a = rng.integers(0, 256, (n, 32), dtype=np.uint64).astype(np.uint8)
    b = rng.integers(0, 256, (n, 32), dtype=np.uint64).astype(np.uint8)
    corners = [(ca, cb, c) for ca in CODES for cb in CODES for c in ACCS]
    acc = np.concatenate([acc, _words([c for _, _, c in corners])])
    a = np.concatenate([a, np.repeat(np.array([ca for ca, _, _ in corners], np.uint8)[:, None], 32, axis=1)])
    b = np.concatenate([b, np.repeat(np.array([cb for _, cb, _ in corners], np.uint8)[:, None], 32, axis=1)])
    ref = _agree(prim, acc, a, b)[:, 0]
    fa = "u8" if prim is P.BlackwellU8S8ImmaDot32 else "s8"
    exact = np.array([s32_value(int(c)) + 32 * decode_int8(fa, int(x[0])) * decode_int8("s8", int(y[0]))
                      for c, x, y in zip(acc[n:], a[n:], b[n:])])
    out = np.array([s32_value(int(r)) for r in ref[n:]])
    assert (exact > S32_MAX).any() and (exact < S32_MIN).any() and (out != exact).sum() == ((exact > S32_MAX) | (exact < S32_MIN)).sum()


@pytest.mark.parametrize("prim", LIMB_STEPS, ids=lambda d: d.name)
def test_the_limb_step_pieces_compute_their_steps(prim) -> None:
    """Every word within 1024 of 0 (the clamp's thresholds at -128 .. -126 and 126 .. 128 among them), each byte's carry edge
    and its neighbours, s32's ends, and uniform words."""
    edges = [s * ((1 << k) + d) for k in range(32) for d in (-1, 0, 1) for s in (1, -1)]
    edges += [S32_MIN, S32_MIN + 1, S32_MAX - 1, S32_MAX, 32639, 32640, -32640, -32641, 65151, -65152]
    rng = np.random.default_rng(43)
    y = np.concatenate([_words(range(-1024, 1025)), _words([e for e in edges if S32_MIN <= e <= S32_MAX]),
                        rng.integers(0, 1 << 32, 8192, dtype=np.uint64).astype(np.uint32)])
    _agree(prim, y)


@pytest.mark.parametrize("target", SPLITS, ids=lambda t: t.id)
def test_the_limb_splits_lower_piece_by_piece(target) -> None:
    """Y's ends, 0 and every word within 300 of them, and uniform words within Y."""
    lo, hi = target.bindings["Y"]
    near = [v for c in (lo, 0, hi) for v in range(c - 300, c + 301) if lo <= v <= hi]
    rng = np.random.default_rng(47)
    _agree(target, _words(near + rng.integers(lo, hi + 1, 4000).tolist()))


def test_an_s8_dot_over_the_u8s8_step_lowers_piece_by_piece() -> None:
    """`S8Dot_v1`'s plain arm over the u8 x s8 step at K = 64: uniform operands in A and B, and A's top against B's ends."""
    target = bind(S8.S8Dot, **S8.SMOKE_PLAIN | U8S8)
    rng = np.random.default_rng(53)
    n = 1000
    a = rng.integers(0, 256, (n, 64)).astype(np.uint8)
    b = rng.integers(-128, 128, (n, 64)).astype(np.int64).astype(np.uint8)
    a[:2], b[0], b[1] = 255, 0x80, 0x7F
    ref = _agree(target, a, b)[:, 0]
    assert [s32_value(int(r)) for r in ref[:2]] == [-255 * 128 * 64, 255 * 127 * 64]


@pytest.mark.parametrize("dot", STEPS, ids=lambda d: d.name)
def test_an_s8_dots_nonzeros_grow_by_one_steps_own_per_step(dot) -> None:
    """Each step commits its accumulator (`forms.Circuit.commit`), so a step past the first adds exactly the step piece's own
    nonzeros: uncommitted, each accumulator bit would be a form over every earlier step's, and the chain quadratic in K."""
    step = _lower(dot)[0].counts()["nnz_AB"]
    extra = U8S8 if dot is P.BlackwellU8S8ImmaDot32 else {}
    nnz = {K: _lower(bind(S8.S8Dot, **S8.SMOKE_PLAIN | {"K": K, "L": K, "DOT": dot} | extra))[0].counts()["nnz_AB"]
           for K in (64, 128, 256)}
    assert nnz[128] - nnz[64] == 2 * step and nnz[256] - nnz[128] == 4 * step


def test_the_pieces_resolve_without_their_boolean_modules() -> None:
    """A process that imports `ir_lower` and the word modules alone, as a Program of word primitives does, finds every piece."""
    ids = sorted(d.id for d in STEPS + LIMB_STEPS)
    code = ("import sys\n"
            "from verity_flock import ir_lower as IL\n"
            "from verity.ml.circuits.ops import prims, s8\n"
            "assert not {'verity.ml.circuits.ops.boolean.gemm', 'verity.ml.circuits.ops.boolean.s8'} & set(sys.modules)\n"
            f"print(sorted(k for k in {ids!r} if k in IL.PIECES))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == repr(ids)
