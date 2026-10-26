"""`verity.ml.circuits.ops.moe` and `ops.boolean.moe`: the padded static MoE pair GEMMs. The Boolean select and pair forms agree
with their word Definitions (`word_of`) on sampled words, on every expert id and on the ids outside [0, E); the word kernels
equal the reference; each composite is its word body (the selected row's GEMM coordinate); and the Boolean forms are pinned by
gate count and program digest."""
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from verity.ml.circuits import boolean as BL
from verity.ml.circuits import codec
from verity.ml.circuits import evaluation as E
from verity.ml.circuits.boolean import trace as TR
from verity.ml.circuits.defs import REGISTRY, bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.evaluation.reference import standalone_call, statics_of
from verity.ml.circuits.ops import gemm as G
from verity.ml.circuits.ops import moe as M
from verity.ml.circuits.ops import prims as P
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.circuits.ops.boolean import moe as BM

SELECTS = [bind(BM.select_bf16(n)) for n in (1, 3, 5, 128, 256)]
PAIRS = [bind(BM.MoePairCoordinate, E=3, K=32, **BG.GEMM_AMPERE), bind(BM.MoePairCoordinate, E=3, K=32, **BG.GEMM_HOPPER),
         bind(BM.MoePairCoordinate, E=128, K=16, **BG.GEMM_HOPPER), bind(BM.MoePairCoordinateW, E=3, K=32, **BG.GEMM_HOPPER),
         bind(BM.MoePairCoordinateW, E=128, K=16, **BG.GEMM_HOPPER), bind(BM.MoePairGemm, E=3, K=16, N=3, **BG.GEMM_HOPPER),
         bind(BM.MoePairGemmW, E=3, K=16, N=3, **BG.GEMM_AMPERE)]

#: id -> (And, Xor, Not gates, standalone program digest)
PINS = {
    "MoeSelectBf16x1_v2": (47, 78, 15, "8b57316b664dc4ab25beb6d8e5b551f718ae7f013a6c36baadbb94183c4df3fd7062c83dbf43247e49488aac87a180a74dcf6bf806228a2adfe6a1359b79bbe4"),
    "MoeSelectBf16x3_v2": (93, 154, 30, "fe51a7e201339801f5f97d97148d21e52d4b6ba4c2548cf4dc73dd800bf780241895a44db1b9990c09c61649a83e2a2885a0e6b21e29d3cfdb875a05cd4eb8c8"),
    "MoeSelectBf16x5_v2": (140, 232, 45, "9e25d19c7ac6a45668bf059d2ebe52798a593bde2c86aa459320c52a87f23d94ec68283c644599aeedf55545309f423b104f70ea7cc57a71aa72fd427fdbb490"),
    "MoeSelectBf16x128_v2": (2072, 4128, 15, "c858cf1ede90a7396d4e9db81d6ceae98feb9ab4ca3778a28eb2109a3c0e2fbd735b8987f66a3b386523d5cd9c8e7be9d29bf3764c02f4a912fb8011de5fc8fb"),
    "MoeSelectBf16x256_v2": (4119, 8222, 15, "57eb3ae12d851498a7792da244318e9051cd86db002004532aa81ccb919561276127ad5221f4ff0ca9d2ef504c9c88e82c056ead70b1b32dc43503ca75e9b089"),
    PAIRS[0].id: (19562, 36460, 2695, "3c12bec7f3752b00b0228d410d0073e50fa6a51189d8826da3920550169093d36c436726bb32534d36a0087199d2125d036ef8cfee307979bc0c603d0e834739"),
    PAIRS[1].id: (18786, 35124, 2497, "877f48b0071f3cbd2b34832faa79f2e3c2397f45b2f2a05a1e57553aa62559b26783029e365aecd39c41d68746dc6d24dacbd1fd8fb539aaace2ab240ad149f1"),
    PAIRS[2].id: (41090, 81195, 1016, "9cbe63a20be772bb9a0830cea5b1f7de46dc8fdf44a34fa59c6fdd184eb0e6722508f0fb34c4c44f6efff8410f3f43646a5b24f7190d47d55ba3ba125964aab0"),
    PAIRS[3].id: (21184, 39686, 2637, "57bf3b55b853a94a4ca32427b81d33e296d501f6449f587d14e2dc5c7d511db697e441413b3db5ff260044f4b9196725f4e61545bcf8f2a68ae1ee81b34d48be"),
    PAIRS[4].id: (43488, 85757, 1156, "3a5571586bb768c90a61e4101f09e2ec7cf2f61ae20d2fc4283de94b0078350e378e6988b7d94ebbeded24514d03addc74d1a6783ae0c5e78228e1da4aac486b"),
    PAIRS[5].id: (28278, 52833, 3768, "1919a7f013962f7c3ca1f5e5dfd393f42f01a30be6d7b5aa4f5c290856baff9a0692aa547c590d4c07532690ceca785bf49a506891bdbb7f2ef2ca9b8baf3f25"),
    PAIRS[6].id: (36636, 68523, 4485, "4f7dbb4b2c6e598e386b5f20279e9c3e30819b34cd15cd5c369538a8bc7061af1927dac53456f90afef8aa5d9122b1c01698b75e83f8ed579901f26da40bcc4b"),
}


@pytest.mark.parametrize("target", SELECTS + PAIRS, ids=lambda t: t.id)
def test_pinned_counts_and_digests(target) -> None:
    ands, xors, nots, digest = PINS[target.id]
    assert BL.is_boolean(target)
    assert BS.compile(target).counts == {"and": ands, "xor": xors, "not": nots}
    prog, _ = standalone_call(target)
    desc = codec.encode_program(prog)
    assert codec.program_digest(desc) == digest
    D = codec.decode_program(json.loads(codec.canonical_json(desc)), REGISTRY)
    assert D.gates == prog.gates and codec.program_digest(codec.encode_program(D)) == digest


def test_the_select_costs_one_mux_per_candidate_bit_and_the_range_bit() -> None:
    """(E - 1) * 16 ANDs of the tree at a power of two, 16 for the fill and 31 - b for the OR of the id's high bits."""
    for n in (2, 4, 128, 256, 1024):
        b = (n - 1).bit_length()
        assert BS.compile(bind(BM.select_bf16(n))).counts["and"] == (n - 1) * 16 + 16 + 31 - b


def _agree(target, inputs) -> None:
    word = TR.word_of(target)
    got = BS.evaluate_words(target, *inputs).astype(np.uint64)
    ref = E.evaluate_batch(word, *inputs).astype(np.uint64)
    assert got.shape == ref.shape
    bad = np.flatnonzero(got.reshape(len(got), -1) != ref.reshape(len(ref), -1))
    assert not bad.size, f"{target.id}: {bad.size} instances differ from {word.id}, first {bad[:5]}"


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
    _agree(target, [e, column])
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
    _agree(target, sampled)
    ids = _ids(S["E"])
    uniform = [rng.integers(0, 1 << (32 if a.ndim == 1 else 16), (len(ids),) + a.shape[1:], dtype=np.uint64).astype(a.dtype)
               for a in sampled]
    uniform[1] = ids
    _agree(target, uniform)


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
