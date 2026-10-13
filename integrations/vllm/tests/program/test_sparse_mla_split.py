"""`registry.sparse_mla_split` (FA3's sparse MLA with the key range split and recombined), `boolean_sparse_mla_split` and the
numpy twins `kernels.sparse_mla_split_twins`: MUFU.LG2's model reproduces the sm_120 measurement at either end of C0's range and
leaves only the powers of two below 1 open; the split counts and boundaries are the kernel's; one split is `SparseMlaHead_v1`; the
twins are the word Definitions word for word; a split head is softmax attention within bf16 rounding; the Boolean versions agree
with their word views."""
from __future__ import annotations

import hashlib
import math
import re

import numpy as np
import pytest

from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import evaluate
from verity_vllm.program import boolean as SW
from verity_vllm.program.kernels import sparse_mla_split_twins as TW
from verity_vllm.program.registry import boolean_sparse_mla_split as BX
from verity_vllm.program.registry import mla_capture as MC
from verity_vllm.program.registry import sparse_mla as M
from verity_vllm.program.registry import sparse_mla_split as W
from tests.program.test_boolean_fp8 import F32_SPECIALS, _flat, _inputs
from tests.program.test_sparse_mla import FA3, _bf16, _case, _run_head, _val

GLM = M.SPARSE_MLA_GLM53
U32 = np.uint32


def _digests() -> tuple[str, str, str]:
    """(mantissa outputs, sweep inputs, sweep outputs): the sha256s `mufu_lg2.json`'s provenance records."""
    m = W.lg2_table()["provenance"]["measured"]
    mant = re.fullmatch(r".*sha256 of the outputs ([0-9a-f]{64})", m["mantissas"]).group(1)
    sweep = re.fullmatch(r".*sha256 of the inputs ([0-9a-f]{64}), of the outputs ([0-9a-f]{64})", m["sweep"])
    return mant, sweep.group(1), sweep.group(2)


def _lg2_word(x: int) -> int:
    return evaluate(W.MufuLg2Ftz, x)[0]


@pytest.mark.parametrize("c0", ["C0", "C0_max"])
def test_lg2_reproduces_the_sm120_measurement(c0: str) -> None:
    """Every mantissa at [1, 2) and the 2^20-word sweep of run r20261004-060203-ba52, by the digests the table records, at
    either end of C0's measured range."""
    mant, sweep_in, sweep_out = _digests()
    h = hashlib.sha256()
    for lo in range(0, 1 << 23, 1 << 21):
        h.update(TW.mufu_lg2_words(U32(0x3F800000) | np.arange(lo, lo + (1 << 21), dtype=U32), c0).astype("<u4").tobytes())
    assert h.hexdigest() == mant
    x = MC._splitmix(np.arange(1 << 20, dtype=np.uint64) + (np.uint64(0xABC) << np.uint64(40))).astype(U32)
    assert hashlib.sha256(x.astype("<u4").tobytes()).hexdigest() == sweep_in
    assert hashlib.sha256(TW.mufu_lg2_words(x, c0).astype("<u4").tobytes()).hexdigest() == sweep_out


@pytest.mark.parametrize("x,want", [
    (0x3F800000, 0x00000000),            # lg2(1) = +0
    (0x40000000, 0x3F800000),            # lg2(2) = 1
    (0x40800000, 0x40000000),            # lg2(4) = 2
    (0x00000000, W.NEG_INF), (0x80000000, W.NEG_INF), (0x00000001, W.NEG_INF), (0x807FFFFF, W.NEG_INF),  # zeros, subnormals
    (0xBF800000, W.QNAN), (0xFF800000, W.QNAN), (0x7FC00000, W.QNAN), (0x7F800001, W.QNAN), (0xFFC00000, W.QNAN),
    (0x7F800000, W.INF),
])
def test_lg2_known_words(x: int, want: int) -> None:
    assert _lg2_word(x) == want
    assert int(TW.mufu_lg2_words([x])[0]) == want


def test_lg2_is_log2_within_its_accuracy() -> None:
    """|lg2(x) - log2(x)| <= 2^-21.5 max(1, |log2 x|) on random positive normals: the worst over seeds 0..19 of 2^18 words each
    (this draw) is 2^-22.1."""
    worst = 0.0
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = ((rng.integers(1, 255, 1 << 18, dtype=np.uint32) << 23) | rng.integers(0, 1 << 23, 1 << 18, dtype=np.uint32))
        y = TW.mufu_lg2_words(x).view(np.float32).astype(np.float64)
        ref = np.log2(x.view(np.float32).astype(np.float64))
        worst = max(worst, float(np.max(np.abs(y - ref) / np.maximum(1.0, np.abs(ref)))))
    assert worst <= 2.0 ** -21.5


def _lg2_sample() -> np.ndarray:
    rng = np.random.default_rng(5)
    E_ = np.arange(1, 255, dtype=U32)
    band = (rng.integers(126, 129, 3000, dtype=np.uint32) << 23) | rng.integers(0, 1 << 23, 3000, dtype=np.uint32)
    return np.concatenate([F32_SPECIALS.astype(U32), E_ << 23, (E_ << 23) | 1, (E_ << 23) | 0x7FFFFF, U32(1 << 31) | (E_ << 23),
                           band, rng.integers(0, 1 << 32, 3000, dtype=np.uint64).astype(U32),
                           np.array([0x00800000, 0x007FFFFF, 0x7F7FFFFF, 0x3F7FFFFF, 0x3F800001, 0xFF7FFFFF], U32)])


def test_lg2_word_twin_and_bits_agree() -> None:
    """`MufuLg2Ftz_v1`, its twin and `MufuLg2Ftz_v2` on the specials, every exponent at a few mantissas (the powers of two
    among them), the band E = 126..128 around 1 and random words; the comparisons with 0 likewise."""
    x = _lg2_sample()
    word = np.array([_lg2_word(int(v)) for v in x], U32)
    assert np.array_equal(TW.mufu_lg2_words(x), word)
    assert np.array_equal(np.asarray(BS.evaluate_words(BX.MufuLg2Ftz, x)).astype(U32).reshape(-1), word)
    for wd, bits, twin in ((W.F32NeZeroFtz, BX.F32NeZeroFtz, TW.nez), (W.F32GtZeroFtz, BX.F32GtZeroFtz, TW.gtz)):
        want = np.array([evaluate(wd, int(v))[0] for v in x])
        assert np.array_equal(twin(x).astype(int), want), wd.name
        assert np.array_equal(np.asarray(BS.evaluate_words(bits, x)).astype(int).reshape(-1), want), bits.name


def test_lg2_boolean_versions_link_to_their_words() -> None:
    for word, bits in ((W.MufuLg2Ftz, BX.MufuLg2Ftz), (W.F32NeZeroFtz, BX.F32NeZeroFtz), (W.F32GtZeroFtz, BX.F32GtZeroFtz)):
        assert TR.word_of(bits).id == word.id
        assert SW.boolean_version(word).id == bits.id


def test_the_comparisons_with_zero() -> None:
    nez = {0: 0, 0x80000000: 0, 1: 0, 0x807FFFFF: 0, 0x00800000: 1, 0x80800000: 1, 0x7F800000: 1, 0xFF800000: 1, 0x7FC00000: 0}
    for x, want in nez.items():
        assert evaluate(W.F32NeZeroFtz, x)[0] == want, hex(x)
        assert evaluate(W.F32GtZeroFtz, x)[0] == (want and not x >> 31), hex(x)


def test_the_lg2_circuit_reads_its_table_in_one_sum() -> None:
    """The table's sums are nonnegative and below 2^37 at every mantissa but 0 (`_lg2_plan` checks it): C1 in 17 digits, C2 in
    11, 2S in 39 bits."""
    assert BX._lg2_plan()[0] == (17, 11, 39)


def _sensitive_mantissas() -> np.ndarray:
    """The mantissas whose fraction v the two ends of C0's range give differently."""
    lo, hi = W.lg2_fit("C0"), W.lg2_fit("C0_max")
    u = np.arange(1 << lo.nb, dtype=np.int64)
    out = []
    for g in np.flatnonzero(np.asarray(lo.C0) != np.asarray(hi.C0)):
        s = lo.C1[g] * u + ((lo.C2[g] * lo.square(u)) << lo.square_shift)
        out.append((int(g) << lo.nb) + np.flatnonzero((lo.C0[g] + s) >> lo.shift != (hi.C0[g] + s) >> hi.shift))
    return np.concatenate(out).astype(U32)


@pytest.mark.slow
def test_c0s_range_changes_no_word() -> None:
    """About 45 s (10^8 words): C0's range moves v at the mantissas of three segments, and at no exponent does that change a word,
    so the measurement determines lg2 on every encoding but the 126 powers of two below 1 (v(0) = 0 is measured at E = 127
    only).  The fast tests sample exponents; this one is the claim the docstring and `mufu_lg2_undetermined` make."""
    d = _sensitive_mantissas()
    assert sorted(set((d >> 17).tolist())) == [23, 26, 54] and d.size
    for e in range(1, 255):
        x = (U32(e) << U32(23)) | d
        assert np.array_equal(TW.mufu_lg2_words(x), TW.mufu_lg2_words(x, "C0_max")), e


def test_the_undetermined_words_are_the_powers_of_two_below_one() -> None:
    E_ = np.arange(0, 256, dtype=U32)
    powers = E_ << 23
    assert np.flatnonzero(TW.undetermined_words(powers)).tolist() == list(range(1, 127))
    assert not TW.undetermined_words(U32(1 << 31) | powers).any()
    x = _lg2_sample()
    x = x[(x & 0x7FFFFF) != 0]
    assert not TW.undetermined_words(x).any()
    assert [W.mufu_lg2_undetermined(int(v)) for v in powers] == TW.undetermined_words(powers).tolist()


# --- the split counts and boundaries --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("max_seqlen_k,batch,want", [(2048, 1, 28), (2048, 14, 28), (2048, 992, 28), (2048, 993, 1),
                                                     (128, 4, 1), (256, 1, 1), (320, 1, 5)])
def test_static_num_splits(max_seqlen_k: int, batch: int, want: int) -> None:
    """At 132 SMs: top-k 2048 is 32 blocks, the first count within 0.85 of the best wave efficiency 28; four blocks or fewer, or
    a batch past `MAX_SPLIT_BATCH`, never split; five blocks split five ways."""
    assert W.static_num_splits(132, max_seqlen_k, batch) == want


def test_blocks_per_sm_is_the_exact_ceiling() -> None:
    """The device's f32 `ceilf(total * 1.1f / 132)` is ceil(11 total / 1320) for every batch of up to 5000 blocks."""
    assert [W.blocks_per_sm(t, 132) for t in range(1, 5001)] == [-(-11 * t // 1320) for t in range(1, 5001)]


@pytest.mark.parametrize("valid,want", [
    # the DSA capture's batch of 14 tokens: 181 blocks, 2 per SM
    ([1, 2, 5, 64, 65, 66, 128, 129, 130, 1001, 2047, 2048, 2048, 2048], [1, 1, 1, 1, 1, 1, 1, 2, 2, 8, 16, 16, 16, 16]),
    # the attention probe's tokens 128, 129, 1000, 2047 and 2048: 86 blocks, 1 per SM, so each token its block count up to 28
    ([129, 130, 1001, 2048, 2048], [3, 3, 16, 28, 28]),
    ([2048], [28]), ([130], [3]), ([64], [1]),
])
def test_dynamic_num_splits(valid: list[int], want: list[int]) -> None:
    assert W.dynamic_num_splits(valid, 132, W.static_num_splits(132, 2048, len(valid))) == want


def test_a_token_alone_splits_as_in_the_probes_batch() -> None:
    """The probe found that a batch of its five tokens and each token alone give the same words: the model gives each token the
    same count both ways."""
    batch = [129, 130, 1001, 2048, 2048]
    assert W.dynamic_num_splits(batch, 132, 28) == [W.dynamic_num_splits([t], 132, 28)[0] for t in batch]


def test_no_split_at_bound_one_or_past_the_batch_limit() -> None:
    assert W.dynamic_num_splits([2048] * 5, 132, 1) == [1] * 5
    assert W.dynamic_num_splits([2048] * (W.MAX_SPLIT_BATCH + 1), 132, 28) == [1] * (W.MAX_SPLIT_BATCH + 1)


def test_a_batch_without_keys_is_refused() -> None:
    with pytest.raises(ValueError, match="blocks_per_sm = 0"):
        W.dynamic_num_splits([0, 0], 132, 28)


@pytest.mark.parametrize("T,NS,want", [
    (130, 2, [(0, 2), (2, 3)]),
    (130, 3, [(0, 1), (1, 2), (2, 3)]),
    (130, 28, [(0, 1), (1, 2), (2, 3)]),          # more splits than blocks: one block each, the rest empty
    (1001, 8, [(2 * s, 2 * s + 2) for s in range(8)]),
    (2048, 28, [(2 * s, 2 * s + 2) for s in range(16)]),   # per = ceil(32 / 28) = 2: splits 16..27 are empty
    (2048, 16, [(2 * s, 2 * s + 2) for s in range(16)]),
    (64, 4, [(0, 1)]),
    (2047, 5, [(0, 7), (7, 14), (14, 21), (21, 28), (28, 32)]),
])
def test_split_blocks(T: int, NS: int, want: list) -> None:
    assert W.split_blocks(T, NS) == want


def test_statics_are_checked() -> None:
    for T, NS in ((0, 1), (5, 0), (5, W.LANES + 1)):
        with pytest.raises(ValueError, match="selected keys in NS"):
            W.check_split_statics(T, NS)
        with pytest.raises(ValueError, match="selected keys in NS"):
            bind(W.SparseMlaHeadSplit, T=T, NS=NS, DR=16, DV=16, BN=16, SCALE=0.0625, **FA3).body
    for n in (0, W.LANES + 1):
        with pytest.raises(ValueError, match="the warp holds 1 to 32"):
            bind(W.SparseMlaCombine, NSPLIT=n, DV=16, INV=FA3["INV"]).body
        with pytest.raises(ValueError, match="the warp holds 1 to 32"):
            TW.combine_rows(np.zeros((1, n), U32), np.zeros((1, n, 4), U32))


# --- the head -------------------------------------------------------------------------------------------------------------------

def _run_split(T, NS, DR, DV, BN, scale, qn, qr, kv) -> np.ndarray:
    head = bind(W.SparseMlaHeadSplit, T=T, NS=NS, DR=DR, DV=DV, BN=BN, SCALE=scale, **FA3)
    return np.asarray(evaluate(head, qn.tolist(), qr.tolist(), kv.reshape(-1).tolist()), dtype=U32)


@pytest.mark.parametrize("T", [5, 37, 130])
def test_one_split_is_the_unsplit_head(T: int) -> None:
    DR, DV, BN = (16, 16, 16) if T < 100 else (GLM["DR"], GLM["DV"], GLM["BN"])
    qn, qr, kv = _case(T + 1, T, DR, DV, kv_scale=0.5)
    want = _run_head(T, DR, DV, BN, 0.0625, qn, qr, kv)
    assert np.array_equal(_run_split(T, 1, DR, DV, BN, 0.0625, qn, qr, kv), want)
    got, nf = TW.head_split_rows(qn[None], qr[None], kv, T=T, NS=1, DR_=DR, DV=DV, BN=BN, scale=0.0625)
    assert not nf.any() and np.array_equal(got[0].astype(U32), want)


def test_more_splits_than_blocks_is_one_block_each() -> None:
    T, DR, DV, BN = 37, 16, 16, 16
    qn, qr, kv = _case(2, T, DR, DV)
    want = _run_split(T, 3, DR, DV, BN, 0.0625, qn, qr, kv)
    assert np.array_equal(_run_split(T, W.LANES, DR, DV, BN, 0.0625, qn, qr, kv), want)


@pytest.mark.parametrize("T,NS", [(17, 2), (37, 2), (37, 3), (64, 4), (65, 3), (100, 7)])
def test_the_twin_is_the_head_word_for_word(T: int, NS: int) -> None:
    """Splits of one block and several, the token's last block partial and zero-padded or full, an uneven last split."""
    DR, DV, BN, R = 16, 16, 16, 3
    rng = np.random.default_rng(T * 31 + NS)
    qn, qr, kv = _bf16(rng, (R, DV)), _bf16(rng, (R, DR)), _bf16(rng, (T, DV + DR), 0.5)
    got, nf = TW.head_split_rows(qn, qr, kv, T=T, NS=NS, DR_=DR, DV=DV, BN=BN, scale=0.0625)
    want = np.stack([_run_split(T, NS, DR, DV, BN, 0.0625, qn[r], qr[r], kv) for r in range(R)])
    assert not nf.any() and np.array_equal(got.astype(U32), want)


def test_the_twin_is_the_head_at_the_served_statics() -> None:
    """T = 130 in two splits (blocks [0, 2) and [2, 3), the token's last block of 2 keys), DR = 64, DV = 512, BN = 64, the scale
    1/16: the first disagreement the H200 capture found was at this token."""
    T, NS, DR, DV, BN = 130, 2, GLM["DR"], GLM["DV"], GLM["BN"]
    rng = np.random.default_rng(130)
    qn, qr, kv = _bf16(rng, (2, DV)), _bf16(rng, (2, DR)), _bf16(rng, (T, DV + DR), 0.5)
    got, nf = TW.head_split_rows(qn, qr, kv, T=T, NS=NS, scale=GLM["SCALE"])
    want = np.stack([_run_split(T, NS, DR, DV, BN, GLM["SCALE"], qn[r], qr[r], kv) for r in range(2)])
    assert not nf.any() and np.array_equal(got.astype(U32), want)


def test_a_token_is_its_split_heads() -> None:
    T, NS, NH, DR, DV, BN = 40, 2, 3, 16, 16, 16
    rng = np.random.default_rng(4)
    qn, qr, kv = _bf16(rng, (NH, DV)), _bf16(rng, (NH, DR)), _bf16(rng, (T, DV + DR))
    tok = bind(W.SparseMlaSplit, T=T, NS=NS, NH=NH, DR=DR, DV=DV, BN=BN, SCALE=0.0625, **FA3)
    got = np.asarray(evaluate(tok, qn.reshape(-1).tolist(), qr.reshape(-1).tolist(), kv.reshape(-1).tolist()), dtype=U32)
    heads = [_run_split(T, NS, DR, DV, BN, 0.0625, qn[h], qr[h], kv) for h in range(NH)]
    assert np.array_equal(got.reshape(NH, DV), np.stack(heads))


@pytest.mark.parametrize("T,NS", [(1001, 8), (2048, 16)])
def test_a_split_head_is_softmax_attention_within_bf16_rounding(T: int, NS: int) -> None:
    """At GLM-5.3's statics, through the twin: softmax(q . k / 16) kv_c within the kernel's roundings, as the unsplit head is."""
    DR, DV, scale = GLM["DR"], GLM["DV"], GLM["SCALE"]
    qn, qr, kv = _case(T + NS, T, DR, DV, kv_scale=0.25)
    got, nf = TW.head_split_rows(qn[None], qr[None], kv, T=T, NS=NS, scale=scale)
    s = (_val(kv[:, :DV]) @ _val(qn) + _val(kv[:, DV:]) @ _val(qr)) * scale
    p = np.exp(s - s.max())
    ref = (p / p.sum()) @ _val(kv[:, :DV])
    assert not nf.any()
    assert np.max(np.abs(_val(got[0]) - ref)) <= 2 ** -7 * np.max(np.abs(_val(kv[:, :DV])))


# --- a split's end and the combine ----------------------------------------------------------------------------------------------

def _end_word(DV: int, scale: float):
    return bind(W.SparseMlaSplitEnd, DV=DV, INV=FA3["INV"], SCALE=scale)


def _combine_word(NSPLIT: int, DV: int):
    return bind(W.SparseMlaCombine, NSPLIT=NSPLIT, DV=DV, INV=FA3["INV"])


@pytest.mark.parametrize("scale", [0.0625, 576 ** -0.5])
def test_the_split_end_twin_is_the_word(scale: float) -> None:
    word = _end_word(16, scale)
    m, l, o = _inputs(word, 400, 17)
    rng = np.random.default_rng(18)
    m[:100] = (rng.standard_normal(100).astype(np.float32) * np.float32(40)).view(U32)    # the running max: any sign
    l[:50] = 0                                                                              # an empty sum
    part, lse = TW.split_end_rows(m, l, o, scale=scale)
    want = _flat(E.evaluate_batch(word, m, l, o))[0]
    assert np.array_equal(np.concatenate([part, lse[:, None]], axis=1).astype(np.uint64), want)
    assert (lse[:50] == W.NEG_INF).all()


def _lse_rows(rng, R: int, NSPLIT: int) -> np.ndarray:
    lse = (rng.standard_normal((R, NSPLIT)).astype(np.float32) * np.float32(8)).view(U32)
    lse[rng.random((R, NSPLIT)) < 0.1] = W.NEG_INF
    return lse


@pytest.mark.parametrize("NSPLIT", [1, 2, 3, 16, 28, 32])
def test_the_combine_twin_is_the_word(NSPLIT: int) -> None:
    """Gaussian lse at several scales with empty (-inf) splits planted, and `_inputs`' positive words with specials."""
    DV, rng = 8, np.random.default_rng(NSPLIT)
    word = _combine_word(NSPLIT, DV)
    lse0, o0 = _inputs(word, 96, NSPLIT)
    lse = np.concatenate([lse0, _lse_rows(rng, 96, NSPLIT)])
    o = np.concatenate([o0, (rng.standard_normal((96, NSPLIT, DV)).astype(np.float32)).view(U32)])
    want = _flat(E.evaluate_batch(word, lse, o))[0]
    assert np.array_equal(TW.combine_rows(lse, o).astype(np.uint64).reshape(len(lse), -1), want)


def _combine(lse, o) -> np.ndarray:
    lse, o = np.asarray(lse, U32), np.asarray(o, U32)
    return np.asarray(evaluate(_combine_word(len(lse), o.shape[1]), lse.tolist(), o.reshape(-1).tolist()), dtype=U32)


def test_one_split_combines_to_its_bf16() -> None:
    rng = np.random.default_rng(9)
    o = rng.standard_normal((1, 16)).astype(np.float32)
    for lse in (0.0, -3.5, 17.25):
        w = o[0].view(U32)
        assert np.array_equal(_combine(np.float32([lse]).view(U32), o.view(U32)), (w + 0x7FFF + ((w >> 16) & 1)) >> 16)


def test_an_empty_split_changes_nothing_and_none_gives_zeros() -> None:
    rng = np.random.default_rng(10)
    lse = (rng.standard_normal(3).astype(np.float32) * np.float32(4)).view(U32)
    o = rng.standard_normal((4, 16)).astype(np.float32).view(U32)
    with_empty = np.array([lse[0], W.NEG_INF, lse[1], lse[2]], U32)
    assert np.array_equal(_combine(with_empty, o), _combine(lse, o[[0, 2, 3]]))
    assert not _combine(np.full(4, W.NEG_INF, U32), o).any()


# --- the Boolean versions -------------------------------------------------------------------------------------------------------

BOOL = {"DOT": BX.SPARSE_MLA_GLM53["DOT"], "INV": BX.SPARSE_MLA_GLM53["INV"]}
AGREEMENT = [bind(BX.SparseMlaSplitEndV2, DV=16, INV=BOOL["INV"], SCALE=0.0625),
             bind(BX.SparseMlaCombineV2, NSPLIT=3, DV=16, INV=BOOL["INV"]),
             bind(BX.SparseMlaHeadSplitV2, T=17, NS=2, DR=16, DV=16, BN=16, SCALE=0.0625, **BOOL)]


@pytest.mark.slow
@pytest.mark.parametrize("target", AGREEMENT, ids=lambda t: t.id)
def test_boolean_agrees_with_its_word_view(target) -> None:
    """5 s for the end, 20 s for the combine, minutes for the head (two blocks, two ends and a combine traced): what the
    word-level tests can't see is the Boolean pieces' agreement on special words (zeros, infinities, NaNs, subnormals) through
    a split's end and the combine."""
    word = TR.word_of(target)
    lifted = {k: (TR.word_of(x) if k in ("DOT", "INV") else x) for k, x in target.bindings.items()}
    assert (word.defn.name, word.defn.version, word.bindings) == (target.defn.name, 1, lifted)
    assert SW.boolean_version(word).id == target.id
    inputs = _inputs(word, 96, sum(map(ord, target.id)))
    for g, r in zip(_flat(BS.evaluate_words(target, *inputs)), _flat(E.evaluate_batch(word, *inputs))):
        bad = np.flatnonzero((g != r).any(axis=1))
        assert not bad.size, f"{target.id}: {bad.size} of {len(r)} instances differ from {word.id}, first {bad[:5]}"
