"""DeepSeek-V4-Pro's FlashMLA sparse decode on the H200 (`flashmla`, `boolean_flashmla`, `kernels.flashmla_twin`).

* the twin (the arithmetic written again over numpy words) word for word against the reference on every binding it registers;
* the primitives on their edge words, and the libm pieces against float64 and at their special words;
* the decode against float64 softmax with the sink at DeepSeek-V4-Pro's statics, split or not;
* the scheduler port on the cases whose splits are known, and on random batches: every request's splits tile its blocks;
* the small Boolean Definitions against their word views; the large ones (a head is about a million ANDs, compiled flat) are
  circuit-check's (`boolean_flashmla.circuit-check.py`).
"""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.boolean.trace import word_of
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import evaluate, evaluate_batch, self_check
from verity.ml.circuits.evaluation.bits import evaluate_words
from verity.ml.kernels import tc as CK
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import flashmla_twin as T
from verity_vllm.program.registry import boolean_flashmla as BF
from verity_vllm.program.registry import flashmla as F

U = np.uint32
DEFS = {d.id: d for d in F.FLASHMLA_COMPOSITES}
TWIN_TARGETS = [t for family in T.KERNELS for t in T._targets(DEFS, family)]
NAN, PINF, NINF = 0x7FFFFFFF, 0x7F800000, 0xFF800000
SPECIAL = [0, 0x80000000, PINF, NINF, 0x7FC00000, NAN, 0xFFFFFFFF, 1, 0x80000001, 0x00800000, 0x3F800000, 0xBF800000,
           0xC2FC0000, 0xC3000000, 0x42FE0000, 0x7F7FFFFF, 0xFF7FFFFF]


def _f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(U)


def _bf16_f64(w) -> np.ndarray:
    return (np.asarray(w, dtype=np.uint16).astype(U) << U(16)).view(np.float32).astype(np.float64)


# --- the twin --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("target", TWIN_TARGETS, ids=[t.id for t in TWIN_TARGETS])
def test_the_twin_is_the_reference(target):
    sc = self_check(target, kernel=T.KERNEL, n=24, seed=0)
    assert sc.ok, str(sc)
    assert sc.declined == 0, str(sc)


def test_the_twin_claims_only_the_hopper_step():
    from verity.ml.circuits.ops.prims import AmpereBF16TcDot16
    from verity_vllm.program.kernels import kernel_registry as KR

    hopper = bind(F.FlashMlaSplit, NB=1, D=32, BN=16, PARTIAL=True, DOT=F.HopperBF16WgmmaDot16, SCALE=0.25)
    ampere = bind(F.FlashMlaSplit, NB=1, D=32, BN=16, PARTIAL=True, DOT=AmpereBF16TcDot16, SCALE=0.25)
    assert KR.covers(hopper, T.KERNEL) is None
    assert KR.covers(ampere, T.KERNEL) == f"DOT={AmpereBF16TcDot16.id}"


def test_one_ulp_of_the_scale_changes_the_words():
    """The twin reads `sm_scale_div_log2` as the host narrows it (double product, one rounding to f32): the neighbouring f32
    gives other output words, so the self-check above pins the rounding."""
    t = bind(F.FlashMlaSplit, NB=1, D=32, BN=16, PARTIAL=True, DOT=F.HopperBF16WgmmaDot16, SCALE=32 ** -0.5)
    args = T._sample_split(np.random.default_rng(3), 8, None, **dict(t.bindings))
    k = np.asarray(args[1]).reshape(8, 16, 33)
    q = np.asarray(args[0])
    sc = T.scale_log2(32 ** -0.5)
    assert sc == F.scale_log2_bits(32 ** -0.5) == int(np.float32(32 ** -0.5 * float(np.float32(1.44269504))).view(U))
    same = [np.array_equal(np.concatenate(T.split(q[i:i + 1], k[i, :, :32], k[i, :, 32], sc, 16)[::-1], axis=None),
                           np.concatenate(T.split(q[i:i + 1], k[i, :, :32], k[i, :, 32], sc + 1, 16)[::-1], axis=None))
            for i in range(8)]
    assert not all(same)


# --- the primitives and the libm pieces -----------------------------------------------------------------------------------------

def test_the_primitives_on_their_edge_words():
    for a in SPECIAL:
        for b in SPECIAL:
            assert F.F32Geu.evaluate(a, b) == int(T.geu(a, b).item()), (hex(a), hex(b))
            assert F.U32Ge.evaluate(a, b) == int(a >= b)
    assert [F.F32Geu.evaluate(*p) for p in ((0x7FC00000, PINF), (0x80000000, 0), (0x3F800000, 0x40000000), (NINF, NINF))] \
        == [1, 1, 0, 1]
    assert [F.E8m0ToBf16.evaluate(x) for x in (0, 1, 127, 254, 255)] == [0x0040, 0x0080, 0x3F80, 0x7F00, 0x7FFF]
    assert [int(v) for v in T.e8m0_to_bf16(np.arange(256))] == [F.E8m0ToBf16.evaluate(x) for x in range(256)]


def _ulps(got: np.ndarray, want: np.ndarray) -> np.ndarray:
    w32 = want.astype(np.float32)
    ok = np.isfinite(w32) & (np.abs(w32) >= np.float32(2.0 ** -126))
    return (np.abs(got.view(np.float32).astype(np.float64) - want) / np.spacing(np.abs(w32)).astype(np.float64))[ok]


def test_the_libm_pieces_against_float64():
    """exp2f within 2.5 ulp, log2f within 1.25, __fdividef(1, x) within 1.25 of float64 where the result is normal (a sweep over
    seeds 0-19, 200,000 inputs each: worst 2.05, 0.88 and 0.97 ulp).  The twin stands in for the reference, which
    `test_the_twin_is_the_reference` holds it to."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-125.9, 127.9, 20_000).astype(np.float32)
    assert _ulps(T.exp2f(x.view(U)), np.exp2(x.astype(np.float64))).max() <= 2.5
    x = (2.0 ** rng.uniform(-149, 128, 20_000)).astype(np.float32)
    x = x[np.isfinite(x) & (x > 0)]
    assert _ulps(T.log2f(x.view(U)), np.log2(x.astype(np.float64))).max() <= 1.25
    x = (2.0 ** rng.uniform(-149, 126, 20_000) * rng.choice([-1, 1], 20_000)).astype(np.float32)
    assert _ulps(T.rcp1(x.view(U)), 1.0 / x.astype(np.float64)).max() <= 1.25


@pytest.mark.parametrize("fn,cases", [
    (F.NvExp2f, {0: 0x3F800000, 0x80000000: 0x3F800000, 0x3F800000: 0x40000000, 0xBF800000: 0x3F000000, 0x43000000: PINF,
                 0xC2FC0000: 0x00800000, 0xC3150000: 1, 0xC3160000: 0, PINF: PINF, NINF: 0, 0x7FC00000: NAN}),
    (F.NvLog2f, {0: NINF, 0x80000000: NINF, 0x3F800000: 0, 0x40000000: 0x3F800000, 0x3F000000: 0xBF800000, 1: 0xC3150000,
                 0x00800000: 0xC2FC0000, 0xBF800000: NAN, PINF: PINF, NINF: NAN, 0x7FC00000: NAN}),
    (F.NvFdividefRcp, {0: PINF, 0x80000000: NINF, 0x3F800000: 0x3F800000, 0x00800000: 0x7E800000, 0x00400000: 0x7F000000,
                       1: PINF, 0x7E800000: 0x00800000, 0x7F000000: 0, PINF: 0, NINF: 0x80000000, 0x7FC00000: NAN}),
], ids=["exp2f", "log2f", "fdividef"])
def test_the_libm_pieces_at_their_special_words(fn, cases):
    got = {x: evaluate(bind(fn), x)[0] for x in cases}
    assert got == cases, {hex(x): (hex(g), hex(cases[x])) for x, g in got.items() if g != cases[x]}


def test_a_row_is_valid_when_its_index_is_not_minus_one_and_it_lies_within_its_segment():
    m = lambda v: v & 0xFFFFFFFF  # noqa: E731
    cases = {(5, 10, 9): 1, (5, 10, 10): 0, (m(-1), 10, 0): 0, (5, 0, 0): 0, (5, m(-1), 0): 0, (0x7FFFFFFF, 1, 0): 1,
             (m(-2), 1, 0): 1, (0, 0x7FFFFFFF, 1 << 30): 1, (0, 0x80000000, 0): 0}
    assert {c: evaluate(bind(F.FlashMlaRowValid), *c)[0] for c in cases} == cases


# --- the decode against float64 ---------------------------------------------------------------------------------------------------

DSV4 = F.FLASHMLA_DSV4
E4M3 = CK.e4m3_to_f32_batch(np.arange(127, dtype=np.uint8))[0].astype(np.float64)


def _to_e4m3(x: np.ndarray) -> np.ndarray:
    idx = np.abs(np.clip(np.abs(x), 0, 448)[..., None] - E4M3).argmin(axis=-1).astype(np.uint8)
    return np.where(x < 0, idx | 0x80, idx).astype(np.uint8)


def _operands(rng, N: int, NW: int, swa: int, extra: int, NH: int = 16):
    """A token's q and N cache rows as the cache writer leaves them: each 64-element NoPE block e4m3 under the UE8M0 scale
    2^ceil(log2(amax / 448)), the RoPE bf16; 1 in 10 indices -1."""
    DN, DR, QT = DSV4["DN"], DSV4["DR"], DSV4["QT"]
    k = rng.standard_normal((N, DN + DR))
    e = np.ceil(np.log2(np.abs(k[:, :DN]).reshape(N, DN // QT, QT).max(axis=2) / 448)).astype(int)
    nope = _to_e4m3(k[:, :DN] / np.repeat(2.0 ** e, QT, axis=1))
    rope = (k[:, DN:].astype(np.float32).view(U) >> U(16)).astype(np.uint16)
    idx = np.where(rng.random(N) < 0.9, rng.integers(0, 1 << 20, N), -1).astype(np.int64).astype(U)
    q = ((rng.standard_normal((NH, DN + DR)) * 2).astype(np.float32).view(U) >> U(16)).astype(np.uint16)
    lens = [U(swa)] + ([U(extra)] if N > NW else [])
    return q, nope, (e + 127).astype(np.uint8), rope, idx, lens, rng.standard_normal(NH).astype(np.float32).view(U)


def _softmax_with_sink(q, nope, scale, rope, idx, lens, sink, NW: int) -> np.ndarray:
    valid = T.row_valid(idx, lens, NW)
    k = _bf16_f64(T.kv_rows(nope, scale, rope, valid, DSV4["QT"]))
    s = np.where(valid[None, :], _bf16_f64(q) @ k.T * DSV4["SCALE"], -np.inf)
    sk = sink.view(np.float32).astype(np.float64)
    m = np.maximum(s.max(axis=1), sk)
    w = np.exp(s - m[:, None])
    return (w @ k) / (w.sum(axis=1) + np.exp(sk - m))[:, None]


@pytest.mark.parametrize("seed", range(3))
def test_the_decode_is_softmax_with_the_sink(seed):
    """A token's 16 heads over the 128-row window and 512 compressed rows, in one split and in the 10 a lone request gets: each
    head within 0.01 of its largest output from float64 softmax with the sink, and the two within 0.015 of each other (a sweep
    over seeds 0-19: worst 0.0043 and 0.0071; P is rounded to bf16 before P V)."""
    rng = np.random.default_rng(seed)
    ops = _operands(rng, 640, 128, swa=int(rng.integers(1, 129)), extra=int(rng.integers(1, 513)))
    want = _softmax_with_sink(*ops, NW=128)
    kw = dict(NW=128, QT=DSV4["QT"], BN=DSV4["BN"], SCALE=DSV4["SCALE"])
    one = _bf16_f64(T.decode(*ops, SPLITS=(0, 10), **kw))
    ten = _bf16_f64(T.decode(*ops, SPLITS=tuple(range(11)), **kw))
    top = np.abs(want).max(axis=1, keepdims=True)
    assert (np.abs(one - want) / top).max() <= 0.01
    assert (np.abs(ten - want) / top).max() <= 0.01
    assert (np.abs(one - ten) / top).max() <= 0.015


def test_a_token_with_no_valid_row_writes_zeros():
    rng = np.random.default_rng(7)
    q, nope, scale, rope, idx, _, sink = _operands(rng, 128, 64, swa=0, extra=0)
    kw = dict(NW=64, QT=DSV4["QT"], BN=DSV4["BN"], SCALE=DSV4["SCALE"])
    for sp in ((0, 2), (0, 1, 2)):
        out = T.decode(q, nope, scale, rope, idx, [U(0), U(0)], sink, SPLITS=sp, **kw)
        assert not out.any()


# --- the scheduler ---------------------------------------------------------------------------------------------------------------

def test_a_lone_request_is_one_split_per_block():
    """b = 1 over the window and 512 compressed rows (10 blocks, 15 units of work over 132 parts): each part takes one block."""
    metas, ns = T.sched_meta([128], [512], topk=128, extra_topk=512)
    assert ns == [0, 10] and len(metas) == 132
    assert T.request_splits(metas, ns, 0, 10) == tuple(range(11))
    assert metas[0].words() == (0, 0, 0, 1, 0, 1, 1) and metas[9].words() == (0, 0, 9, 10, 9, 1, 1)
    assert all(m.begin_req_idx == 1 for m in metas[10:])


def test_the_scheduler_tiles_every_request():
    rng = np.random.default_rng(0)
    for _ in range(200):
        b = int(rng.integers(1, 300))
        extra_topk = int(rng.choice([0, 512, 2048]))
        swa = rng.integers(0, 129, b)
        extra = rng.integers(0, extra_topk + 1, b) if extra_topk else None
        metas, ns = T.sched_meta(swa, extra, topk=128, extra_topk=extra_topk, parts=T.num_sm_parts(64))
        assert ns[0] == 0 and all(x < y for x, y in zip(ns, ns[1:]))
        for r in range(b):
            nb = T.request_blocks(int(swa[r]), None if extra is None else int(extra[r]))[1] // T.BLOCK
            sp = T.request_splits(metas, ns, r, nb)
            assert len(sp) - 1 == ns[r + 1] - ns[r]


def test_the_scheduler_constants():
    assert (T.num_sm_parts(64), T.num_sm_parts(128), T.num_sm_parts(64, s_q=2)) == (132, 66, 66)
    assert [T.request_blocks(*a) for a in ((128, None), (0, None), (128, 512), (1, 1), (65, 0))] == \
        [(128, 128), (64, 64), (128, 640), (64, 128), (128, 128)]


# --- the statics --------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn,kw,needle", [
    (F.FlashMlaDecodeHead, dict(SPLITS=(0, 0), D=32, BN=16), "SPLITS=(0, 0)"),
    (F.FlashMlaDecode, dict(NH=1, SPLITS=(0, 2), NW=24, DN=16, DR=16, QT=16, BN=16), "NW=24"),
    (F.FlashMlaKvRow, dict(DN=448, DR=64, QT=48), "QT=48"),
    (F.FlashMlaBlock, dict(D=40, BN=16, FIRST=True), "D=40"),
    (F.FlashMlaCombine, dict(NS=1, D=16), "NS=1"),
], ids=["splits", "window", "scale-block", "block", "combine"])
def test_bad_statics_are_refused_by_name(fn, kw, needle):
    extra = {k: v for k, v in (("DOT", F.HopperBF16WgmmaDot16), ("SCALE", 0.25)) if k in fn.statics}
    with pytest.raises(ValueError, match=needle.replace("(", r"\(").replace(")", r"\)")):
        bind(fn, **kw, **extra).body


# --- the Boolean Definitions --------------------------------------------------------------------------------------------------------

def _scalar_vectors(params, n: int, rng) -> list[np.ndarray]:
    out = []
    for _, pt in params:
        w = pt.w
        if w == 32:
            v = np.where(rng.random(n) < 0.5, rng.choice(np.array(SPECIAL, dtype=U), n), T._f32(rng, (n,), scale=64.0, p=0.0))
        else:
            v = rng.integers(0, 1 << w, n)
        out.append(np.asarray(v, dtype=np.uint64))
    return out


SMALL = [BF.F32GeuV2, BF.U32GeV2, BF.E8m0ToBf16V2, BF.FlashMlaRowValidV2, bind(BF.NvExp2fV2), bind(BF.NvLog2fV2),
         bind(BF.NvFdividefRcpV2)]


@pytest.mark.parametrize("target", SMALL, ids=[t.id for t in SMALL])
def test_the_small_boolean_definitions_are_their_words(target):
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    args = _scalar_vectors(word.signature()[0], 512, np.random.default_rng(1))
    if target.id.startswith("FlashMlaRowValid"):
        args = [np.where(np.random.default_rng(2).random(512) < 0.3, 0xFFFFFFFF, a % 40).astype(np.uint64) for a in args]
    got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(512, -1)
    want = np.asarray(evaluate_batch(word, *args, kernel="reference"), dtype=np.uint64).reshape(512, -1)
    assert np.array_equal(got, want)


def test_the_boolean_row_is_its_word():
    target = bind(BF.FlashMlaKvRowV2, DN=16, DR=16, QT=8)
    word = word_of(target)
    assert SW.boolean_version(word).id == target.id
    args = T._sample_row(np.random.default_rng(4), 128, None, DN=16, DR=16, QT=8)
    got = np.asarray(evaluate_words(target, *args), dtype=np.uint64).reshape(128, -1)
    want = np.asarray(evaluate_batch(word, *args, kernel="reference"), dtype=np.uint64).reshape(128, -1)
    assert np.array_equal(got, want)
