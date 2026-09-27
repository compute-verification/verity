"""`ScaledMmFp8BlockSharedScale_v1` (registry.fp8): each weight block's scale products `FMUL(sx[kb], sw[kb])` computed once and read by
its G coordinates.  The same function and signature as `ScaledMmFp8Block_v1` -- the same circuit once the G - 1 copies of each product are
merged, bit-equal under the reference evaluator on e4m3 / f32 edge words (NaN, +-inf, +-0, subnormals, saturation, products that
overflow or underflow) -- and under `Q_word_v1{16,32,no-recompute}` no violation: the partition commits the N/G * K/G products."""
from __future__ import annotations

import random
import re

import pytest

from verity.evaluation import evaluate
from verity.ir.defs import bind
from verity.ir.layout import resolve
from verity_vllm.program.registry import fp8
from verity_vllm.query import word as W

SHAPES = [(256, 256), (384, 128), (128, 512)]
E4M3_FINITE = [0x00, 0x80, 0x01, 0x81, 0x07, 0x87, 0x08, 0x88, 0x38, 0xB8, 0x7E, 0xFE, 0x77, 0xF7]
E4M3_NAN = [0x7F, 0xFF]
F32_FINITE = [0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x00800000, 0x3F800000, 0xBF800000,
              0x7F7FFFFF, 0xFF7FFFFF, 0x3B124925, 0x1F800000, 0x5F800000, 0x0DA24260, 0x72000000, 0x00400000, 0x2B8CBCCC]
F32_SPECIAL = [0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000, 0x7F800001, 0x7FBFFFFF, 0xFFFFFFFF]


def _pair(K: int, N: int):
    return bind(fp8.ScaledMmFp8Block, K=K, N=N, G=128), bind(fp8.ScaledMmFp8BlockSharedScale, K=K, N=N, G=128)


def _canon(fn, table: dict) -> tuple[int, ...]:
    """Each returned leaf as an interned gate DAG (primitive, operands; inputs by position): equal ids, the same function."""
    c, call = W._call_scope(fn)
    lo, n = call.offset, fn.gates
    ids = [0] * n
    for j in range(n):
        g = c.gate(lo + j)
        ops = tuple(ids[o - lo] if lo <= o < lo + n else table.setdefault(("in", o), len(table)) for o in g.operands)
        ids[j] = table.setdefault((g.prim.id, ops), len(table))
    return tuple(ids[resolve(call, r) - lo] for r in fn.body.ret)


@pytest.mark.parametrize("K,N", SHAPES)
def test_the_same_circuit_with_the_products_merged(K, N):
    old, new = _pair(K, N)
    assert new.signature() == old.signature()
    table: dict = {}
    assert _canon(old, table) == _canon(new, table)
    assert old.gates - new.gates == (128 - 1) * (N // 128) * (K // 128)      # G - 1 F32Mul per (weight block, kb)


def _byte(rng, p_edge: float, p_nan: float) -> int:
    r = rng.random()
    if r < p_nan:
        return rng.choice(E4M3_NAN)
    if r < p_nan + p_edge:
        return rng.choice(E4M3_FINITE)
    b = rng.getrandbits(8)
    return 0x38 if b & 0x7F == 0x7F else b


def _scale(rng, mode: str) -> int:
    if mode == "special":
        return rng.choice(F32_SPECIAL + F32_FINITE)
    if mode == "tiny":                                                       # the product underflows: subnormal or zero
        return rng.choice([0x00000001, 0x007FFFFF, 0x00800000, 0x00400000, 0x1F800000, 0x0DA24260]) if rng.random() < 0.5 else \
            rng.randrange(1, 60) << 23 | rng.getrandbits(23)
    return rng.randrange(100, 127) << 23 | rng.getrandbits(23)             # 2^-27 .. 1: the checkpoint's and the quant's scales


def _bf16_class(b: int) -> str:
    e, m = (b >> 7) & 0xFF, b & 0x7F
    return "nan" if e == 0xFF and m else "inf" if e == 0xFF else "zero" if not b & 0x7FFF else "subnormal" if e == 0 else "normal"


def test_bit_equal_on_edge_vectors():
    K, N, G = 256, 256, 128
    old, new = _pair(K, N)
    rng = random.Random(0xF8B10C)
    seen = set()
    for k in range(32):
        mode = ("normal", "tiny", "special", "normal")[k % 4]
        p_edge, p_nan = (0.4, 0.0) if k % 8 != 7 else (0.2, 0.0005)
        a = ([_byte(rng, p_edge, p_nan) for _ in range(K)], [_scale(rng, mode) for _ in range(K // G)],
             [_byte(rng, p_edge, p_nan) for _ in range(N * K)], [_scale(rng, mode) for _ in range(N // G * (K // G))])
        yo, yn = evaluate(old, *a), evaluate(new, *a)
        assert yo == yn, (k, mode, [i for i, (p, q) in enumerate(zip(yo, yn)) if p != q][:8])
        seen.update(_bf16_class(y) for y in yo)
    assert seen == {"nan", "inf", "zero", "subnormal", "normal"}, seen


@pytest.mark.parametrize("K,N", SHAPES)
def test_the_partition_commits_each_product_once(K, N):
    _old, new = _pair(K, N)
    nb, kb = N // 128, K // 128
    r = W.unit_rule(new)
    assert r["violations"] == [] and r["redundant_gates"] == 0
    assert r["by_kind"] == {"output": N, "committed": nb * kb} and r["committed_interior_words"] == nb * kb
    (c,) = r["committed_interior"]
    assert c["head"] == "F32Mul_v1" and c["scope"] == f"ScaledMmFp8BlockSharedScaleTile_v1{{K={K},G=128}}" and c["bits"] == 32 * nb * kb
    tile = bind(fp8.ScaledMmFp8BlockSharedScaleTile, K=K, G=128)
    assert W.Graph(tile).recomputed == []


def test_new_ids_carry_no_version_in_their_names():
    for d in (fp8.ScaledMmFp8BlockSharedScale, fp8.ScaledMmFp8BlockSharedScaleTile, fp8.ScaledMmFp8BlockCoordinateGivenScale):
        name = d.id.rsplit("_v", 1)[0]
        assert d.id == f"{name}_v1" and not re.search(r"[Vv]\d", name) and name.startswith("ScaledMmFp8Block")
    assert fp8.SHARED_SCALE == {"ScaledMmFp8Block_v1": fp8.ScaledMmFp8BlockSharedScale}
