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
    "MoeSelectBf16x1_v2": (47, 78, 15, "a32dc50a20736318663c59fb583e829ccec7e0b1a03c755eb666f17a49fbf66b54e95d0846db71f54f21167d2b19f5b6576012b249bc063fb9c649562f5c7dc9"),
    "MoeSelectBf16x3_v2": (93, 154, 30, "98063ef302db600f0286a9889f469279a426993679a2fd6e3f738b4ad3221e6004e1ebaa271720bcde6027303aa16042d2e8b3742fe318c826290fbe3a5029c5"),
    "MoeSelectBf16x5_v2": (140, 232, 45, "acd9a40136790284d6240990464031f84f74d1bfc37e449a844b494832c8aae0a9bd456c6c823e0c8f08e040f043ac223b28122e2316c0495338dde6dd47b775"),
    "MoeSelectBf16x128_v2": (2072, 4128, 15, "b6f0098abbb3ebdb580c3d731a0e00c22968e9039ec355ad8fbec500cec6a2fb227dc16219f75f1c2af2e49026090fd435a4c11b51b82e2cf6c4e41ad863af2d"),
    "MoeSelectBf16x256_v2": (4119, 8222, 15, "290d8ee06644f0c77ce83286a30be3ae88aadef2f179f5e847c17d0e03edb73f71eedf8e7bb442a102ac3f1fbe1be42989f34098bc7e4e42ef55a19f1fee0f4d"),
    PAIRS[0].id: (19562, 36460, 2695, "2f47e086f36e62467515b07f3aa8bfcb323128db6da4922e6776966b088b10422fc7a2df2baa1bfbb9986c8460fed6807c22070e0d8543a295c6eefbd1d490e5"),
    PAIRS[1].id: (18786, 35124, 2497, "95625ff343912f6f16ce6006f67c6a46bea52f5dcc17c4175b58350d590a5ba45fdfaf741978158e9f4cbd79dd13bbda9e6aab531162e1e11eb4e04bdeeb7b2c"),
    PAIRS[2].id: (41090, 81195, 1016, "36dfc709c6ff261510015928849acf6bfd965e68262a78c82383de7bfab8a9f61051d3afe6c13b818becb16d91bd6fbbaada73a35cf0fbd9a215531205aab566"),
    PAIRS[3].id: (21184, 39686, 2637, "c8bf4593e341c158e6baf3177205eff029426510fc810c6fc6281bc77f8ea4704dae6942b9c49ab30599930a28d79f460dfd4893f8f42c170ba6525be66a8df6"),
    PAIRS[4].id: (43488, 85757, 1156, "29cb5b210c15e462332b0303b30bba160fc320b6e8a240d8cc3c807fdfb3cc628be9b5be8d4ecdd317edc66a3feed0566f5487b42bc6d28054bdf426ba48e8f2"),
    PAIRS[5].id: (28278, 52833, 3768, "03f9f6340f0efa5622d3d70d6df49ec1d5c592ac537d520740eebb96a380bc1e448ae5cc6c4d83804631eb924bf7b1c351170cbb96a286d20a540337b48172a6"),
    PAIRS[6].id: (36636, 68523, 4485, "1343a65bb890bcbdd013463f4f2bfc61f44cd773ae432a4f4acd21184ab025d5f00c12828a015b68e401153900a699b70c801531550ca42f0e158301bc9be80f"),
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
