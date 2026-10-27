"""`verity.ml.definitions.registry.cublas_bf16`'s training bindings: `train_route` is the route table the capture fitted
(`TRAIN_CAPTURE`) and refuses what it did not capture, `route_gemm` binds each route's Definition on bits, which equals that
route's numpy model (`bf16_gemm_check.cublas_route_rows`) on spread and cancelling words, and `attention_bmm` is
`BatchedGemm_v2` at each bmm's reduction and output lengths."""
from __future__ import annotations

import numpy as np
import pytest

from verity.ml.circuits.evaluation import bits as BS
from verity.ml.definitions.registry import bf16_gemm_check as BC
from verity.ml.definitions.registry import cublas_bf16 as CB

HOP = "DOT=HopperBF16WgmmaDot16_v2"
#: the capture's routes other than the unsplit one
SPLIT = {
    (256, 576, 1536, "dX"): ("CublasSplitKGemmBf16Partials_v2", 1536, 576, 8),
    (256, 576, 49152, "dX"): ("CublasSplitKGemmBf16Partials_v2", 49152, 576, 17),
    (4096, 576, 49152, "dX"): ("CublasSerialSplitKGemm_v2", 49152, 576, 7),
    (4096, 576, 49152, "dW"): ("CublasSerialSplitKGemm_v2", 4096, 576, 2),
}
ALIGNED = {(4096, 576, 576, "dW"): 7, (4096, 576, 192, "dW"): 12, (4096, 576, 1536, "dW"): 3, (4096, 1536, 576, "dW"): 3}


def test_train_routes_are_the_capture() -> None:
    assert set(CB.TRAIN_ROUTES) == {(M, K, N, r) for M in CB.TRAIN_M for K, N in CB.TRAIN_LINEARS for r in ("fwd", "dX", "dW")}
    for key, route in CB.TRAIN_ROUTES.items():
        if key in SPLIT:
            name, K, N, z = SPLIT[key]
            assert CB.train_linear(key[3], *key[:3]).id == f"{name}{{K={K},N={N},SPLIT_K={z},{HOP}}}"
        elif key in ALIGNED:
            assert route == {"kernel": "splitk", "SPLIT_K": ALIGNED[key], "partial": "bf16", "align": 8}
            with pytest.raises(ValueError, match="rounded up to 8"):
                CB.train_linear(key[3], *key[:3])
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


ROUTES = [{"kernel": "nosplit"}, {"kernel": "splitk", "SPLIT_K": 4, "partial": "bf16"}, {"kernel": "serialk", "SPLIT_K": 4}]


@pytest.mark.parametrize("route", ROUTES, ids=["nosplit", "splitk-bf16", "serialk"])
def test_route_gemm_on_bits_is_the_route_model(route) -> None:
    rng = np.random.default_rng(117)
    target = CB.route_gemm(route, 64, 3)
    for x, w in BC.route_probe_rows(16, 64, 3, rng):
        got = np.asarray(BS.evaluate_words(target, x, np.broadcast_to(w, (len(x), *w.shape)))).astype(np.uint16)
        assert np.array_equal(got, BC.cublas_route_rows(route, x, w)), route
    assert all((got != BC.cublas_route_rows(r, x, w)).all() for r in ROUTES if r != route), "the cancelling words tell routes apart"
