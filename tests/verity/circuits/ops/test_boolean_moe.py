"""`verity.circuits.ops.moe` and `ops.boolean.moe`: the padded static MoE pair GEMMs. The Boolean select and pair forms agree
with their word Definitions (`word_of`) on sampled words, on every expert id and on the ids outside [0, E); the word kernels
equal the reference; each composite is its word body (the selected row's GEMM coordinate); and the Boolean forms are pinned by
gate count and program digest."""
from __future__ import annotations

import re

import numpy as np
import pytest

from verity.circuits import codec
from verity.circuits import evaluation as E
from verity.circuits.boolean import trace as TR
from verity.circuits.defs import REGISTRY, bind
from verity.circuits.evaluation import bits as BS
from verity.circuits.evaluation.reference import statics_of
from verity.circuits.ops import gemm as G
from verity.circuits.ops import moe as M
from verity.circuits.ops import prims as P
from verity.circuits.ops.boolean import gemm as BG
from verity.circuits.ops.boolean import moe as BM

from _boolean_checks import assert_agrees, assert_pinned

SELECTS = [bind(BM.select_bf16(n)) for n in (1, 3, 5, 128, 256)]
PAIRS = [bind(BM.MoePairCoordinate, E=3, K=32, **BG.GEMM_AMPERE), bind(BM.MoePairCoordinate, E=3, K=32, **BG.GEMM_HOPPER),
         bind(BM.MoePairCoordinate, E=128, K=16, **BG.GEMM_HOPPER), bind(BM.MoePairCoordinateW, E=3, K=32, **BG.GEMM_HOPPER),
         bind(BM.MoePairCoordinateW, E=128, K=16, **BG.GEMM_HOPPER), bind(BM.MoePairGemm, E=3, K=16, N=3, **BG.GEMM_HOPPER),
         bind(BM.MoePairGemmW, E=3, K=16, N=3, **BG.GEMM_AMPERE)]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MoeSelectBf16x1_v2": (47, 78, 15, "6d3922d0d25fd16356c41c7452dac726cb700c9d71437afbc1eb1700f22328a83625cd8e91aa587409c51153776956567d005a3160ba4da1ced05384ab6d5cd7"),
    "MoeSelectBf16x3_v2": (93, 154, 30, "f352243325f060a47d82bf28d8065bcfbb95a5e755a3405dc92acd017c445857cb7eba9e92536c988c2cf8b9af697e20a8b090d483a086f1156cb3980b7dd544"),
    "MoeSelectBf16x5_v2": (140, 232, 45, "9b53d858fefd32eccf8ba1fb0ef9be8da238b8a4d8c758b09cb8d78ed82850a7efd8bf5e35cf0b40d780d7692d12be5edf942faa5f8ea2c993788a00d7e9838c"),
    "MoeSelectBf16x128_v2": (2072, 4128, 15, "e85875920aaf80b812339accd91cae930807120b8c113c746c26b8e56c9ef7890f5ba921c0e30f8ee8089014494cf7da15e8e8d4dfe5a7f15dca31a15a72053b"),
    "MoeSelectBf16x256_v2": (4119, 8222, 15, "3c0df0b7b483f00daade633fe98f5ff5adc87e3a4fa13a91155053c320fcd42c156b99f5b4adcd97b45a16cffdfe35da3abb86551e1c867e01e8c1eff0671ee2"),
    PAIRS[0].id: (19562, 36460, 2695, "289e7eec00b1bfacc077296489cbf58ec43689c517ac4a6183d11f9e4386dae4747ff610b874ce88b81404ef3155f9d063b6e341dbc33f05402f92f328cf321e"),
    PAIRS[1].id: (18786, 35124, 2497, "67c1764d58d4593624314f7c20d10cef5cf3edc396950054caaff0e9ce1bb13b1589219d765b10f1eef821e508e5bec26225d9bc853c2e0a9bcf9e3f3ae0efae"),
    PAIRS[2].id: (41090, 81195, 1016, "6a797f5e4f4bd72abcc4e5a39b3200694e281f76a8973ba0549680e096c05e4d6d2b57d10ae07640467324d3c7cb22435d11e381b4b7407ced9f20852d6abfdd"),
    PAIRS[3].id: (21184, 39686, 2637, "ba2237c613659b0378a8604bd08811eda923a17f22df9d67118f7a938977b192f20e87afcaf8c4ef77b661d8b28291bc654908393052f83b95be4b20516b0e2b"),
    PAIRS[4].id: (43488, 85757, 1156, "78835d396d20d5842898f99d32c4b7339f77e09b7a93c9cce4d60356c66b28e69400e23dd1d2a48ea6c615ab63096d887c575e9ce77d09ffca9f0d43513d26c4"),
    PAIRS[5].id: (28278, 52833, 3768, "5db2e22e6e717115ff0e391e52241a6cfeb33785ac1203b26b1f31b71a71005c8cbaea833150f757ab32b72999ea542f04635c51675f7395b7ffd158a7f45e70"),
    PAIRS[6].id: (36636, 68523, 4485, "2c8843dfebb792bd15aeaf97f193366f84eba3f042fcfb68b1ed4410063a47b7bc09314092deb58befc3748f7d139a2969e1f6f0dc2a2ad7249357e4fb582a0e"),
}


@pytest.mark.parametrize("target", SELECTS + PAIRS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    assert_pinned(target, PINS)


def test_the_select_costs_one_mux_per_candidate_bit_and_the_range_bit() -> None:
    """(E - 1) * 16 ANDs of the tree at a power of two, 16 for the fill and 31 - b for the OR of the id's high bits."""
    for n in (2, 4, 128, 256, 1024):
        b = (n - 1).bit_length()
        assert BS.compile(bind(BM.select_bf16(n))).counts["and"] == (n - 1) * 16 + 16 + 31 - b


def _ids(n: int) -> np.ndarray:
    """Every id in [0, E), then the edges past it: E, the next powers of two, 2^31 - 1, 2^31 (the most negative), -1."""
    edges = [n, n + 1, 1 << (n - 1).bit_length(), 1 << ((n - 1).bit_length() + 1), (1 << 31) - 1, 1 << 31, (1 << 32) - 1]
    return np.asarray(list(range(n)) + edges, np.uint64).astype(np.uint32)


@pytest.mark.parametrize("target", SELECTS, ids=lambda t: t.id)
def test_the_select_agrees_with_its_word_on_every_id_and_the_edges(target) -> None:
    n = int(re.match(r"MoeSelectBf16x(\d+)_v2", target.id)[1])
    rng = np.random.default_rng(23)
    ids = np.concatenate([_ids(n), rng.integers(0, 1 << 32, 500, dtype=np.uint64).astype(np.uint32)])
    reps = 8
    e = np.repeat(ids, reps)
    column = rng.integers(0, 1 << 16, (len(e), n), dtype=np.uint64).astype(np.uint16)
    column[::reps] = 0x7FFF ^ np.uint16(1)
    assert_agrees(target, [e, column])
    want = np.where(e < n, column[np.arange(len(e)), np.minimum(e, n - 1)], 0x7FFF)
    assert (BS.evaluate_words(target, e, column).reshape(-1) == want).all()


@pytest.mark.parametrize("target", PAIRS, ids=lambda t: t.id)
def test_the_pair_forms_agree_with_their_word_definitions(target) -> None:
    """On the kernel's sampler (`moe._sample`: tensor-core operand words, specials, an id past E one time in eight), and on
    every id of `_ids` with uniform words."""
    word = TR.word_of(target)
    params, _ = E.batch.signature_of(word)
    S = statics_of(word)
    rng = np.random.default_rng(29)
    sampled = E.kernels(word)["numpy"].sample(rng, 2_000, params, **S)
    assert_agrees(target, sampled)
    ids = _ids(S["E"])
    uniform = [rng.integers(0, 1 << (32 if a.ndim == 1 else 16), (len(ids),) + a.shape[1:], dtype=np.uint64).astype(a.dtype)
               for a in sampled]
    uniform[1] = ids
    assert_agrees(target, uniform)


@pytest.mark.parametrize("target", PAIRS, ids=lambda t: t.id)
def test_the_word_kernels_equal_the_reference(target) -> None:
    check = E.self_check(TR.word_of(target), n=256)
    assert check.ok and check.declined == 0, str(check)


def test_each_pair_coordinate_is_the_selected_rows_gemm_coordinate() -> None:
    """`MoePairCoordinate_v1`(x, e, slab) = `GemmCoordinate_v2`(x, slab[e]) and the routed form `F2fpBf16(F32Mul_v2(DotBf16_v2(x,
    slab[e]), g))` for every e in [0, E); an e outside it reads the 0x7FFF row."""
    rng = np.random.default_rng(31)
    Ex, K, H = 4, 32, P.HopperBF16WgmmaDot16
    pair, pair_w = bind(M.MoePairCoordinate, E=Ex, K=K, DOT=H), bind(M.MoePairCoordinateW, E=Ex, K=K, DOT=H)
    x = rng.integers(0, 1 << 16, (1, K), dtype=np.uint64).astype(np.uint16)
    slab = (rng.standard_normal((1, Ex, K)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
    g = np.asarray([0x3E9A0000], np.uint32)
    for e in [*range(Ex), Ex, 1 << 31]:
        row = slab[:, e] if e < Ex else np.full((1, K), M.FILL, np.uint16)
        ee = np.asarray([e], np.uint32)
        want = E.evaluate_batch(bind(G.GemmCoordinate, K=K, DOT=H), x, row, kernel="reference")
        acc = E.evaluate_batch(bind(G.DotBf16, K=K, DOT=H), x, row, kernel="reference")
        want_w = E.evaluate_batch(P.F2fpBf16, E.evaluate_batch(M.F.F32MulV2, acc, g), kernel="reference")
        assert (E.evaluate_batch(pair, x, ee, slab, kernel="reference") == want).all()
        assert (E.evaluate_batch(pair_w, x, ee, g, slab, kernel="reference") == want_w).all()


@pytest.mark.parametrize("defn,rest", [(M.MoePairCoordinate, {}), (M.MoePairGemmW, {"N": 2}), (BM.MoePairCoordinateW, {}),
                                       (BM.MoePairGemm, {"N": 2})], ids=lambda d: getattr(d, "id", ""))
def test_each_composite_refuses_the_k_its_gemm_refuses(defn, rest) -> None:
    dot = BG.HopperBF16WgmmaDot16 if defn in (BM.MoePairCoordinateW, BM.MoePairGemm) else P.HopperBF16WgmmaDot16
    for K in (17, 24):
        with pytest.raises(ValueError, match=re.escape(f"{defn.id} refuses this binding: K={K} is not a multiple of the 16-wide step")):
            bind(defn, E=4, K=K, DOT=dot, **rest)


def test_a_select_on_bits_is_bounded_and_the_word_select_decodes_by_id() -> None:
    with pytest.raises(ValueError, match="1 to 1024 experts, not 1025"):
        BM.select_bf16(1025)
    with pytest.raises(ValueError, match="at least one expert"):
        M.select_bf16(0)
    assert "MoeSelectBf16x96_v1" not in REGISTRY
    assert codec.lookup_primitive(REGISTRY, "MoeSelectBf16x96_v1") is M.select_bf16(96)
