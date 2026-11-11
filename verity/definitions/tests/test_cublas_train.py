"""`verity.definitions.registry.cublas_bf16`'s training bindings: `train_route` is the route table the capture fitted
(`TRAIN_CAPTURE`) and refuses what it did not capture, `route_gemm` binds each route's Definition on bits, which, and whose word,
equals that route's numpy model (`bf16_gemm_check.cublas_route_rows`) on spread and cancelling words (an unsplit one at a K that
is not a multiple of 16 on the operands padded with +0 words), `aligned_bounds` makes the B 8 x S 512 dW's grid.z slices and
the m and xl steps', and `attention_bmm` is `BatchedGemm_v2` at each bmm's reduction and output lengths.  Each model's agreement
with sm_120's words is the capture's (`cublas_bf16.TRAIN_CAPTURE`); the grid.z dW's is also held here on words of a full-tensor
dump of that step that tell its k-tile layouts apart (`fixtures/cublas_dw_grid_z_sm120.npz`), and the lf model's s step's routes
on the capture's words that tell each from every other GEMM route (`fixtures/cublas_train_lf_sm120.npz`).  `train_tf32` binds
`GemmTf32_v1` at each shape the TF32 capture covers and refuses the rest, and the TF32 GEMM, on words and on bits, gives the
capture's words that tell it from each alternative (`fixtures/cublas_tf32_sm120.npz`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.circuits.evaluation import bits as BS
from verity.definitions.registry import bf16_gemm_check as BC
from verity.definitions.registry import cublas_bf16 as CB

HOP = "DOT=HopperBF16WgmmaDot16_v3"
FIXTURES = Path(__file__).parent / "fixtures"
#: the capture's routes other than the unsplit one
SPLIT = {
    (256, 576, 1536, "dX"): ("CublasSplitKGemmBf16Partials_v2", 1536, 576, 8),
    (256, 576, 49152, "dX"): ("CublasSplitKGemmBf16Partials_v2", 49152, 576, 17),
    (4096, 576, 49152, "dX"): ("CublasSerialSplitKGemm_v2", 49152, 576, 7),
    (4096, 576, 49152, "dW"): ("CublasSerialSplitKGemm_v2", 4096, 576, 2),
    (2048, 512, 512, "dW"): ("CublasSplitKGemmBf16Partials_v2", 2048, 512, 8),
    (2048, 512, 1408, "dW"): ("CublasSplitKGemmBf16Partials_v2", 2048, 512, 4),
    (2048, 1408, 512, "dW"): ("CublasSplitKGemmBf16Partials_v2", 2048, 1408, 4),
    (2048, 512, 32000, "dX"): ("CublasSerialSplitKGemm_v2", 32000, 512, 5),
}
#: the grid.z dW routes: (grid.z, the launched cutlass kernel's k-tile)
ALIGNED = {(4096, 576, 576, "dW"): (7, 32), (4096, 576, 192, "dW"): (12, 64), (4096, 576, 1536, "dW"): (3, 32),
           (4096, 1536, 576, "dW"): (3, 32)}


def test_train_routes_are_the_capture() -> None:
    assert set(CB.TRAIN_ROUTES) == {(M, K, N, r) for Ms, linears in ((CB.TRAIN_M, CB.TRAIN_LINEARS), (CB.LF_M, CB.LF_LINEARS))
                                    for M in Ms for K, N in linears for r in ("fwd", "dX", "dW")}
    for key, route in CB.TRAIN_ROUTES.items():
        if key in SPLIT:
            name, K, N, z = SPLIT[key]
            assert CB.train_linear(key[3], *key[:3]).id == f"{name}{{K={K},N={N},SPLIT_K={z},{HOP}}}"
        elif key in ALIGNED:
            z, t = ALIGNED[key]
            assert route == {"kernel": "splitk", "SPLIT_K": z, "partial": "bf16", "align": 8, "tile_k": t}
            assert CB.train_linear(key[3], *key[:3]).id == (f"CublasSplitKGemmBf16PartialsAligned_v2{{K=4096,N={key[1]},"
                                                            f"SPLIT_K={z},ALIGN=8,TILE_K={t},{HOP}}}")
        else:
            M, K, N, role = key
            k, n = {"fwd": (K, N), "dX": (N, K), "dW": (M, K)}[role]
            assert route == {"kernel": "nosplit"} and CB.train_linear(role, M, K, N).id == f"Gemm_v3{{K={k},N={n},{HOP}}}"


def test_train_route_refuses_what_the_capture_does_not_cover() -> None:
    for args in (("fwd", 512, 576, 576), ("dX", 256, 576, 320), ("dW", 16384, 576, 576), ("fwd", 2048, 576, 576),
                 ("dW", 4096, 512, 512)):
        with pytest.raises(ValueError, match="is not captured"):
            CB.train_linear(*args)
    with pytest.raises(ValueError, match="role 'dY'"):
        CB.train_route("dY", 256, 576, 576)
    for route in ({"kernel": "splitk", "SPLIT_K": 4}, {"kernel": "splitk", "SPLIT_K": 3, "align": 8, "tile_k": 16},
                  {"kernel": "serialk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16}):
        with pytest.raises(ValueError, match="no Definition on bits"):
            CB.route_gemm(route, 64, 3)
    with pytest.raises(ValueError, match="gemv"):
        CB.route_gemm({"kernel": "gemv", "LANES": 32}, 64, 3)
    with pytest.raises(ValueError, match="SPLIT_K_in_2_to_K_over_16"):
        CB.route_gemm({"kernel": "serialk", "SPLIT_K": 5}, 64, 3)
    with pytest.raises(ValueError, match="SPLIT_K_chunks_at_ALIGN"):
        CB.route_gemm({"kernel": "splitk", "SPLIT_K": 5, "partial": "bf16", "align": 64, "tile_k": 64}, 256, 3)
    with pytest.raises(ValueError, match="no tile_k"):
        CB.route_gemm({"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8}, 64, 3)
    with pytest.raises(ValueError, match="TILE_K_whole_k16_steps"):
        CB.route_gemm({"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 24}, 64, 3)


def test_aligned_bounds_are_the_capture_slices() -> None:
    assert [hi - lo for lo, hi in CB.aligned_bounds(4096, 7, 8)] == [592] * 6 + [544]
    assert [hi - lo for lo, hi in CB.aligned_bounds(4096, 12, 8)] == [344] * 11 + [312]
    assert [hi - lo for lo, hi in CB.aligned_bounds(4096, 3, 8)] == [1368, 1368, 1360]
    assert CB.aligned_bounds(1536, 8, 64) == CB.split_bounds(1536, 8)
    assert [CB.residue_first(0, n, t) for n, t in ((1368, 32), (1360, 32), (592, 32), (544, 32), (344, 64), (312, 64))] == [
        (24, 8), (16, 16), (16, 16), (0, 0), (24, 40), (56, 8)]
    for z in (8, 4):
        assert CB.aligned_bounds(2048, z, 8) == CB.split_bounds(2048, z), "the s step's dW is the plain bf16-workspace route"
        assert all(CB.residue_first(lo, hi, 32) == (0, 0) for lo, hi in CB.split_bounds(2048, z))
    # the m and xl steps' grid.z dW at K 2048: 3 slices with no reduction kernel, whose first two take a residue tile of 16
    # words, 5 with a bf16 workspace in whole tiles, and an f32 workspace's 12, each a residue tile first
    for z, sizes, tiles in ((3, [688, 688, 672], [(16, 16), (16, 16), (0, 0)]), (5, [416] * 4 + [384], [(0, 0)] * 5),
                            (12, [176] * 11 + [112], [(16, 16)] * 12)):
        bounds = CB.aligned_bounds(2048, z, 8)
        assert [hi - lo for lo, hi in bounds] == sizes and [CB.residue_first(lo, hi, 32) for lo, hi in bounds] == tiles, z
        assert bounds != CB.split_bounds(2048, z)


def test_the_grid_z_dw_is_sm120s_on_the_dump_words() -> None:
    """Five dW words of a full-tensor dump of the B 8 x S 512 step on sm_120, each one row of dY^T against one of X^T at K 4096:
    two over 3 grid.z slices at k-tile 32 and three over 12 at k-tile 64.  The residue-first k-tiles give each word; a chain
    with the zeros at each slice's back (a tile as long as the slice) misses the first four, and the fifth is one a chain with
    them at the front misses."""
    d = np.load(FIXTURES / "cublas_dw_grid_z_sm120.npz")
    for i in range(len(d["gpu"])):
        x, w, z, t = d["x"][i:i + 1], d["w"][i:i + 1], int(d["split"][i]), int(d["tile_k"][i])
        assert int(BC.cublas_aligned_splitk_rows(x, w, z, 8, t)[0, 0]) == int(d["gpu"][i]), i
        whole = -(-max(hi - lo for lo, hi in CB.aligned_bounds(4096, z, 8)) // 16) * 16
        back = int(BC.cublas_aligned_splitk_rows(x, w, z, 8, whole)[0, 0])
        assert back == int(d["back"][i]) and (back != int(d["gpu"][i])) == (i < 4), i


def test_the_s_step_routes_are_sm120s_on_its_words() -> None:
    """Words of the lf model's s step's linears on sm_120 at B 2 x S 1024, per F.linear role: count-probe words (the all-ones row
    against `splitk_count_probe`'s rows) and the fewest random words that rule out what the capture's random words rule out.  Each
    role's route gives every word, and every other GEMM route at 2 to 16 slices misses one; the gemv routes are left out, since
    cuBLAS launches cutlass GEMMs at M 2048.  A split route's model chunks K as its Definition on bits does (`split_bounds`)."""
    d = np.load(FIXTURES / "cublas_train_lf_sm120.npz")
    for K, N in CB.LF_LINEARS:
        for role, (_, _, row) in CB.TRAIN_LINEAR_ROLES.items():
            route = CB.train_route(role, 2048, K, N)
            words = [tuple(d[f"{K}x{N}/{role}/{kind}/{a}"] for a in "xwy") for kind in ("random", "probe")]
            assert all(np.array_equal(BC.cublas_route_rows(route, x, w), y) for x, w, y in words), (K, N, role)
            k = row(2048, K, N)[0]
            if "SPLIT_K" in route:
                chunks = BC._split_cols(k, route["SPLIT_K"], "contiguous")
                assert [(int(c[0]), int(c[-1]) + 1) for c in chunks] == CB.split_bounds(k, route["SPLIT_K"]), (K, N, role)
            others = [r for r in BC.cublas_routes_at(list(range(2, min(16, k // 16) + 1))) if r["kernel"] != "gemv" and r != route]
            fits = [r for r in others if all(np.array_equal(BC.cublas_route_rows(r, x, w), y) for x, w, y in words)]
            assert not fits, (K, N, role, fits)


def test_attention_bmm_is_batched_gemm_at_each_op() -> None:
    want = {"qk": ("D", "S"), "pv": ("S", "D"), "dp": ("D", "S"), "dv": ("S", "D"), "dq": ("S", "D"), "dk": ("S", "D")}
    assert set(CB.TRAIN_BMM) == set(want)
    for S, D in CB.TRAIN_BMM_SHAPES:
        for op, (k, n) in want.items():
            K, N = ({"S": S, "D": D}[k], {"S": S, "D": D}[n])
            assert CB.attention_bmm(op, S, D, B=9).id == f"BatchedGemm_v2{{B=9,K={K},N={N},{HOP}}}"
    assert CB.attention_bmm("qk", 256, 64).id == f"BatchedGemm_v2{{B=1,K=64,N=256,{HOP}}}"
    for args, match in ((("qk", 128, 64), "is not captured"), (("pv", 512, 128), "is not captured"), (("qv", 256, 64), "op 'qv'"),
                        (("dq", 256, 64, 0), "B = 0")):
        with pytest.raises(ValueError, match=match):
            CB.attention_bmm(*args)


def test_train_tf32_is_the_tf32_capture() -> None:
    assert CB.TRAIN_TF32 == ((4096, 64, 64), (11264, 64, 64), (301056, 64, 64), (2048, 512, 512))
    for M, K, N in CB.TRAIN_TF32:
        assert CB.train_tf32(M, K, N).id == f"GemmTf32_v1{{K={K},N={N}}}"
    for args in ((4096, 64, 32), (2048, 512, 1408), (512, 512, 512)):
        with pytest.raises(ValueError, match="is not captured"):
            CB.train_tf32(*args)
    assert "whole k8 TF32 steps" in CB.GemmTf32.refuse({"K": 12}) and CB.GemmTf32.refuse({"K": 16}) is None


def test_the_tf32_gemm_is_sm120s_on_the_capture_words() -> None:
    """Words of the cc 12.0 capture's f32 mm's under allow_tf32 at each `TRAIN_TF32` shape in each of the four operand layouts
    (an unsplit `cutlass_80_tensorop_s1688gemm` kernel in each): `GemmTf32_v1` on words and its σ `GemmTf32_v2` on bits give
    every word, and every alternative the fit tried misses one: the k8 step's products in groups of 8, 4 + 4 or 2 + 2 + 2 + 2
    at adder widths 24 to 28, each operand rounded to TF32 by RNE, RNA or truncation (all but the step itself), and an f32
    FFMA chain."""
    import dataclasses

    from verity.circuits.defs import bind
    from verity.circuits.evaluation.reference import evaluate
    from verity.definitions.numerics import mhc_twin as MT
    from verity.definitions.numerics import torch_kernels as TK
    from verity.definitions.registry import boolean_sigma as SG
    from verity.definitions.registry import mhc as MHC

    d = np.load(FIXTURES / "cublas_tf32_sm120.npz")
    cases = sorted({k.rsplit("/", 1)[0] for k in d.files})
    assert cases == sorted(f"{M}x{K}x{N}/{la}{lb}" for M, K, N in CB.TRAIN_TF32 for la in "nt" for lb in "nt"), cases
    base = MHC.HOPPER_TF32_WGMMA_K8_CANDIDATE
    alts = {(g, w, r): dataclasses.replace(base, groups=g, width=w) for g in ((8,), (4, 4), (2, 2, 2, 2)) for w in range(24, 29)
            for r in ("rne", "rna", "trunc") if (g, w, r) != (tuple(base.groups), base.width, "rne")}
    missed = dict.fromkeys([*alts, "ffma"], False)
    for case in cases:
        x, w, y = (d[f"{case}/{a}"] for a in "xwy")
        K = x.shape[1]
        gemm = bind(CB.GemmTf32, K=K, N=1)
        words = [int(np.ravel(evaluate(gemm, [int(v) for v in xi], [int(v) for v in wi]))[0]) for xi, wi in zip(x, w)]
        assert words == y.tolist(), case
        bits = np.asarray(BS.evaluate_words(bind(SG.GemmTf32, K=K, N=1), x, w[:, None, :])).reshape(-1).astype(np.uint32)
        assert np.array_equal(bits, y), case
        for (g, width, r), model in alts.items():
            acc = np.zeros(len(y), np.uint32)
            for k in range(0, K, 8):
                acc = MT.tf32_dot8(acc, x[:, k:k + 8], w[:, k:k + 8], model, r, r)
            missed[(g, width, r)] |= not np.array_equal(acc, y)
        acc = np.zeros(len(y), np.uint32)
        for k in range(K):
            acc = TK.fma(x[:, k], w[:, k], acc)
        missed["ffma"] |= not np.array_equal(acc, y)
    assert all(missed.values()), [h for h, m in missed.items() if not m]


#: one route of each kind at K 64, N 3; the grid.z ones over chunks of [24, 24, 16] at k-tile 16, each 24-word chunk's residue
#: tile 8 words and 8 zeros
GRID_Z = [{"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16},
          {"kernel": "splitk", "SPLIT_K": 3, "partial": "f32", "align": 8, "tile_k": 16},
          {"kernel": "serialk", "SPLIT_K": 3, "align": 8, "tile_k": 16}]
ROUTES = [{"kernel": "nosplit"}, {"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"}, {"kernel": "serialk", "SPLIT_K": 4},
          *GRID_Z]
ROUTE_IDS = ["nosplit", "splitk-bf16", "serialk", "splitk-bf16-align8", "splitk-f32-align8", "serialk-align8"]


def test_the_training_routes_bind_the_opt_in_step_and_the_served_programs_keep_v2() -> None:
    """`TRAIN_DOT` is `HopperBF16WgmmaDot16_v3`, ``opt_in``: each route binds it and names the word the served step names, and
    `boolean_version` of that word, and of a route's word, is still on `HopperBF16WgmmaDot16_v2`, so no served Program moves."""
    from verity.circuits.boolean.trace import word_of
    from verity.definitions.ops import prims as P
    from verity.definitions.ops.boolean import gemm as BG
    from verity.definitions.boolean import boolean_version

    assert CB.TRAIN_DOT is BG.HopperBF16WgmmaDot16V3 and CB.TRAIN_DOT.opt_in
    assert word_of(CB.TRAIN_DOT) is word_of(BG.HopperBF16WgmmaDot16) is P.HopperBF16WgmmaDot16
    assert boolean_version(P.HopperBF16WgmmaDot16).id == "HopperBF16WgmmaDot16_v2"
    for route in ROUTES:
        target = CB.route_gemm(route, 64, 3)
        assert target.bindings["DOT"].id == "HopperBF16WgmmaDot16_v3"
        assert boolean_version(word_of(target)).id == target.id.replace("_v3}", "_v2}")
    bmm = CB.attention_bmm("qk", 256, 64)
    assert boolean_version(word_of(bmm)).id == bmm.id.replace("_v3}", "_v2}")


def test_the_training_steps_chain_parts_give_the_served_steps_words() -> None:
    """`HOPPER_STEP_V3`'s four parts, chained by `GemmF32Out_v2` at K 48 over 3 rows (the first step from +0, then two later
    ones, x's own gates shared), give `HOPPER_STEP`'s words: on rows of near-1 words, and on rows of uniform or near-1 words
    with special words (NaNs of several payloads, the infinities, the zeros, subnormals, the largest finite) at a rate of 1,
    5 or 30%."""
    from verity.circuits.defs import bind
    from verity.definitions.ops.boolean import gemm as BG

    rng = np.random.default_rng(118)
    special = np.array([0x7FC0, 0x7F81, 0xFFFF, 0x7F80, 0xFF80, 0x0000, 0x8000, 0x0001, 0x807F, 0x7F7F, 0xFF7F], np.uint16)
    n, K, N = 4000, 48, 3
    rate = np.repeat([0.0, 0.01, 0.05, 0.3], n // 4)

    def words(shape):
        r = rate.reshape(-1, *([1] * (len(shape) - 1)))
        u = rng.integers(0, 1 << 16, shape).astype(np.uint16)
        u = np.where(rng.random(shape) < np.where(r == 0, 1.0, 0.5), (u & 0x80FF) | 0x3F00, u).astype(np.uint16)
        return np.where(rng.random(shape) < r, rng.choice(special, shape), u)

    x, w = words((n, K)), words((n, N, K))
    got, want = (np.asarray(BS.evaluate_words(bind(CB.GemmF32Out, K=K, N=N, DOT=d), x, w)).astype(np.uint32)
                 for d in (CB.TRAIN_DOT, BG.HopperBF16WgmmaDot16))
    assert np.array_equal(got, want)
    special_out = (got & 0x7F800000) == 0x7F800000
    assert special_out.sum() > 1000 and (~special_out).sum() > 1000


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_route_gemm_on_bits_is_the_route_model(route) -> None:
    """Each route's Definition on bits is its numpy model on the spread and cancelling words, and no other route's model fits the
    cancelling words: the closest two, serial split-K over 4 chunks and over the 3 grid.z ones, differ at 32 of the 48 words."""
    rng = np.random.default_rng(117)
    target = CB.route_gemm(route, 64, 3)
    for x, w in BC.route_probe_rows(16, 64, 3, rng):
        got = np.asarray(BS.evaluate_words(target, x, np.broadcast_to(w, (len(x), *w.shape)))).astype(np.uint16)
        assert np.array_equal(got, BC.cublas_route_rows(route, x, w)), route
    assert all((got != BC.cublas_route_rows(r, x, w)).sum() >= 32 for r in ROUTES if r != route), "the cancelling words tell routes apart"


@pytest.mark.parametrize("route", ROUTES, ids=ROUTE_IDS)
def test_route_gemm_on_words_is_the_route_model(route) -> None:
    """Each route's Definition on words, what the grader prices and the training rules bind, is its numpy model on the spread
    and cancelling words, evaluated by the reference evaluator."""
    from verity.circuits.boolean.trace import word_of
    from verity.circuits.evaluation.reference import evaluate

    word = word_of(CB.route_gemm(route, 64, 3))
    for x, w in BC.route_probe_rows(4, 64, 3, np.random.default_rng(119)):
        got = np.array([evaluate(word, [int(v) for v in row], [int(v) for v in w.reshape(-1)]) for row in x], np.uint16)
        assert np.array_equal(got, BC.cublas_route_rows(route, x, w)), route


def test_an_unsplit_route_pads_k_to_whole_k16_steps() -> None:
    """At a K that is not a multiple of 16 an unsplit route is `GemmPad_v2`: on bits, the route's model on x and W's rows padded
    with +0 words at the back to the next multiple of 16.  A split route there is refused."""
    from verity.circuits.boolean.trace import word_of
    from verity.definitions.boolean import boolean_version

    target = CB.route_gemm({"kernel": "nosplit"}, 40, 3)
    assert target.id == f"GemmPad_v2{{T=40,K=48,N=3,{HOP}}}"
    assert boolean_version(word_of(target)).id == target.id.replace("_v3}", "_v2}")
    for x, w in BC.route_probe_rows(16, 40, 3, np.random.default_rng(117)):
        got = np.asarray(BS.evaluate_words(target, x, np.broadcast_to(w, (len(x), *w.shape)))).astype(np.uint16)
        assert np.array_equal(got, BC.cublas_route_rows({"kernel": "nosplit"}, *(np.pad(a, ((0, 0), (0, 8))) for a in (x, w))))
    with pytest.raises(ValueError, match="K_positive_multiple_of_16"):
        CB.route_gemm({"kernel": "serialk", "SPLIT_K": 2}, 40, 3)


@pytest.mark.parametrize("route", GRID_Z, ids=ROUTE_IDS[3:])
def test_aligned_route_on_bits_takes_each_residue_tile_first(route) -> None:
    """At K 64 over 3 slices at k-tile 16, a +-2^13 pair at words 0 and 12 sits in one k16 step of a chain with the zeros at each
    slice's back, and in two of the residue-first chain (8 words and 8 zeros, then 16 words), so the small words each step
    absorbs differ and the bits tell the two layouts apart, whatever the slices' partials meet in."""
    x, w = BC.route_probe_rows(16, 64, 3, np.random.default_rng(117))[1]
    wf = np.where(np.abs(BC.DR.bf16_to_f32_words(w).view(np.float32)) >= 256, np.float32(2.0 ** -8),
                  BC.DR.bf16_to_f32_words(w).view(np.float32))
    wf[:, 0], wf[:, 12] = 2.0 ** 13, -2.0 ** 13
    w = (wf.view(np.uint32) >> 16).astype(np.uint16)
    got = np.asarray(BS.evaluate_words(CB.route_gemm(route, 64, 3), x, np.broadcast_to(w, (len(x), *w.shape)))).astype(np.uint16)
    assert np.array_equal(got, BC.cublas_route_rows(route, x, w))
    assert (got != BC.cublas_route_rows({**route, "tile_k": 32}, x, w)).all(), "a whole-slice tile is the zeros at the back"
