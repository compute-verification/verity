"""`verity.ml.definitions.registry.cublas_bf16`'s training bindings: `train_route` is the route table the capture fitted
(`TRAIN_CAPTURE`) and refuses what it did not capture, `route_gemm` binds each route's Definition on bits, which equals that
route's numpy model (`bf16_gemm_check.cublas_route_rows`) on spread and cancelling words, `aligned_bounds` makes the B 8 x S 512
dW's grid.z slices, and `attention_bmm` is `BatchedGemm_v2` at each bmm's reduction and output lengths.  Each model's agreement
with sm_120's words is the capture's (`cublas_bf16.TRAIN_CAPTURE`); the grid.z dW's is also held here on words of a full-tensor
dump of that step that tell its k-tile layouts apart (`fixtures/cublas_dw_grid_z_sm120.npz`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from verity.ml.circuits.evaluation import bits as BS
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity.ml.definitions.registry import cublas_bf16 as CB

HOP = "DOT=HopperBF16WgmmaDot16_v2"
FIXTURES = Path(__file__).parent / "fixtures"
#: the capture's routes other than the unsplit one
SPLIT = {
    (256, 576, 1536, "dX"): ("CublasSplitKGemmBf16Partials_v2", 1536, 576, 8),
    (256, 576, 49152, "dX"): ("CublasSplitKGemmBf16Partials_v2", 49152, 576, 17),
    (4096, 576, 49152, "dX"): ("CublasSerialSplitKGemm_v2", 49152, 576, 7),
    (4096, 576, 49152, "dW"): ("CublasSerialSplitKGemm_v2", 4096, 576, 2),
}
#: the grid.z dW routes: (grid.z, the launched cutlass kernel's k-tile)
ALIGNED = {(4096, 576, 576, "dW"): (7, 32), (4096, 576, 192, "dW"): (12, 64), (4096, 576, 1536, "dW"): (3, 32),
           (4096, 1536, 576, "dW"): (3, 32)}


def test_train_routes_are_the_capture() -> None:
    assert set(CB.TRAIN_ROUTES) == {(M, K, N, r) for M in CB.TRAIN_M for K, N in CB.TRAIN_LINEARS for r in ("fwd", "dX", "dW")}
    for key, route in CB.TRAIN_ROUTES.items():
        if key in SPLIT:
            name, K, N, z = SPLIT[key]
            assert CB.train_linear(key[3], *key[:3]).id == f"{name}{{K={K},N={N},SPLIT_K={z},{HOP}}}"
        elif key in ALIGNED:
            z, t = ALIGNED[key]
            assert route == {"kernel": "splitk", "SPLIT_K": z, "partial": "bf16", "align": 8, "tile_k": t}
            assert CB.train_linear(key[3], *key[:3]).id == (f"CublasSplitKGemmBf16PartialsAligned_v1{{K=4096,N={key[1]},"
                                                            f"SPLIT_K={z},ALIGN=8,TILE_K={t},{HOP}}}")
        else:
            M, K, N, role = key
            k, n = {"fwd": (K, N), "dX": (N, K), "dW": (M, K)}[role]
            assert route == {"kernel": "nosplit"} and CB.train_linear(role, M, K, N).id == f"Gemm_v3{{K={k},N={n},{HOP}}}"


def test_train_route_refuses_what_the_capture_does_not_cover() -> None:
    for args in (("fwd", 512, 576, 576), ("dX", 256, 576, 320), ("dW", 16384, 576, 576)):
        with pytest.raises(ValueError, match="is not captured"):
            CB.train_linear(*args)
    with pytest.raises(ValueError, match="role 'dY'"):
        CB.train_route("dY", 256, 576, 576)
    with pytest.raises(ValueError, match="no Definition on bits"):
        CB.route_gemm({"kernel": "splitk", "SPLIT_K": 4}, 64, 3)
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


ROUTES = [{"kernel": "nosplit"}, {"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"}, {"kernel": "serialk", "SPLIT_K": 4},
          {"kernel": "splitk", "SPLIT_K": 3, "partial": "bf16", "align": 8, "tile_k": 16}]


@pytest.mark.parametrize("route", ROUTES, ids=["nosplit", "splitk-bf16", "serialk", "splitk-bf16-align8"])
def test_route_gemm_on_bits_is_the_route_model(route) -> None:
    rng = np.random.default_rng(117)
    target = CB.route_gemm(route, 64, 3)
    for x, w in BC.route_probe_rows(16, 64, 3, rng):
        got = np.asarray(BS.evaluate_words(target, x, np.broadcast_to(w, (len(x), *w.shape)))).astype(np.uint16)
        assert np.array_equal(got, BC.cublas_route_rows(route, x, w)), route
    assert all((got != BC.cublas_route_rows(r, x, w)).all() for r in ROUTES if r != route), "the cancelling words tell routes apart"


def test_aligned_route_on_bits_takes_each_residue_tile_first() -> None:
    """At K 64 over 3 slices at k-tile 16, a +-2^13 pair at words 0 and 12 sits in one k16 step of a chain with the zeros at each
    slice's back, and in two of the residue-first chain (8 words and 8 zeros, then 16 words), so the small words each step
    absorbs differ and the bits tell the two layouts apart."""
    route = ROUTES[-1]
    x, w = BC.route_probe_rows(16, 64, 3, np.random.default_rng(117))[1]
    wf = np.where(np.abs(BC.DR.bf16_to_f32_words(w).view(np.float32)) >= 256, np.float32(2.0 ** -8),
                  BC.DR.bf16_to_f32_words(w).view(np.float32))
    wf[:, 0], wf[:, 12] = 2.0 ** 13, -2.0 ** 13
    w = (wf.view(np.uint32) >> 16).astype(np.uint16)
    got = np.asarray(BS.evaluate_words(CB.route_gemm(route, 64, 3), x, np.broadcast_to(w, (len(x), *w.shape)))).astype(np.uint16)
    assert np.array_equal(got, BC.cublas_route_rows(route, x, w))
    assert (got != BC.cublas_route_rows({**route, "tile_k": 32}, x, w)).all(), "a whole-slice tile is the zeros at the back"
