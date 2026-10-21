"""The replay row kernels of DeepSeek-V4-Pro's served rows that no other module evaluates, one per Definition family, each the numpy
twin its Definition is held to: `mhc_twin` (the six mHC families), `dsv4_compressor_twin` (the head-512 compressed rows), and the
hash router's `GatherI32x{V}` lookup, which is wiring only.  A row is one token: the twins' leading token axis is 1, and each
Tuple's fields are ports "0", "1", ... in the Definition's return order.  The twins are checked word for word against their
Definitions (tests/program/test_mhc.py, the kernel self-check), and on the H200 by the captures `EVALUATOR_PROVENANCE` names.
The row kernels register from `rows` (kernel "rows").
"""
from __future__ import annotations

__models__ = ("deepseek-ai/DeepSeek-V4-Pro",)

from typing import Any, Callable, Sequence

import numpy as np

from verity_vllm.program.kernels.ir_rows import f64_statics
from verity_vllm.program.kernels.kernel_registry import Declined

#: the vocabularies of the served `GatherI32x{V}_v1` rows (DeepSeek-V4-Pro's hash router reads tid2eid at its vocabulary)
GATHER_VOCABS = (129280,)

_MHC = ("verity_vllm.program.kernels.mhc_twin.{fn} on one token ({sig}), word for word the Definition's operations in its order "
        "(the TF32 k8 step HopperTf32WgmmaDot8_v1's candidate group sum, B rounded RNE, the squared sum's lo contraction)")

EVALUATORS = {
    "MhcPost_v1": _MHC.format(fn="post", sig="MhcPost_v1{HC,N}: new_residual's FMA chain per stream, RNE to bf16"),
    "MhcFusedPostPre_v1": _MHC.format(fn="fused_post_pre", sig="MhcFusedPostPre_v1{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}: NS "
                                      "MhcFusedSplit_v1s, then MhcPreBigFuse_v1") + " -> residual', layer input, post', comb'",
    "MhcPostPreTf32_v1": _MHC.format(fn="post_pre_tf32", sig="MhcPostPreTf32_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}: MhcPost_v1, "
                                     "MhcPrenormTf32_v1 over tf32_ranges(HC*H, NS), MhcPreBigFuse_v1") + " -> residual', layer input, post', comb'",
    "MhcPreBroadcast_v1": _MHC.format(fn="pre_broadcast", sig="MhcPreBroadcast_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}: MhcPrenormTf32_v1 "
                                      "of the embedding, MhcPreBigFuseBroadcast_v1") + " -> layer input, post, comb",
    "MhcResidualBroadcast_v1": "the embedding's H bf16 words copied to each of the HC streams (MhcResidualBroadcast_v1{HC,H}: no arithmetic)",
    "HcHead_v1": _MHC.format(fn="hc_head", sig="HcHead_v1{HC,H,TH,EPS,HC_EPS}: the threads' sqr / mix chains, halving trees, rsqrt, "
                             "sigmoid + HC_EPS, the streams' weighted sum") + " -> H bf16",
    "Dsv4MlaCompressC4_v1": "verity_vllm.program.kernels.dsv4_compressor_twin.compress_c4 (Dsv4MlaCompressC4_v1{ROWS,D,QB,EPS}: "
                            "Dsv4MlaC4Column_v1 per column, Dsv4MlaNormRopeQuant_v1{4,D,QB,EPS}) -> NoPE e4m3 bytes, RoPE bf16, e8m0 scales",
    "Dsv4MlaCompressC128_v1": "verity_vllm.program.kernels.dsv4_compressor_twin.compress_c128 (Dsv4MlaCompressC128_v1{W,R,D,QB,EPS}: "
                              "Dsv4MlaC128Column_v1 per column, Dsv4MlaNormRopeQuant_v1{128,D,QB,EPS}) -> NoPE e4m3 bytes, RoPE bf16, e8m0 scales",
    **{f"GatherI32x{v}_v1": f"column[token] of the {v} committed i32 words, 0xFFFFFFFF for a token outside [0, {v}) "
                            f"(GatherI32x{v}_v1, registry.dsv4_decode_indices.gather_i32: a lookup, no arithmetic)" for v in GATHER_VOCABS},
}

_MHC_PROVENANCE = {
    "property": "dsv4_mhc_capture (verity-vllm/dsv4-mhc-capture/v1), tests/properties/dsv4_mhc_capture_gpu.py",
    "run": "r20261013-205619-36e8",
    "art": "art:b5b410ee8690cc67d2bb638073b2528ef4d66b12a76d12949efc7f8f41983cd8",
    "step_fit": "r20261013-115825-580e (the probe that fitted HopperTf32WgmmaDot8_v1's k8 step alone)",
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80, DeepGEMM on (VLLM_USE_DEEP_GEMM=1, vLLM's default)",
    "captured_utc": "2026-10-13",
    "cases": "the fused post + pre at 1, 2, 7, 8, 9, 15 and 16 tokens; the TF32 post + pre at 17, 64, 65, 128 and 129 (MhcPost_v1 "
             "among its pieces); the layer-0 broadcast at 1, 16, 64 and 65 with every residual_out copy; hc_head at 1, 7, 16, 64 and 129",
    "tests": ["tests/program/test_mhc.py (every mHC word Definition against mhc_twin, word for word)",
              "tests/program/test_kernel_self_check.py (each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity_vllm/program/registry/mhc.py (the reference; the twin is checked against it, never the other way round)",
}

_COMPRESSOR_PROVENANCE = {
    "property": "dsv4_compressor_capture (verity-vllm/dsv4-compressor-capture/v1), tests/properties/dsv4_compressor_capture_gpu.py",
    "run": "r20261013-211655-2a70",
    "art": "art:914dd281ad207ce67db3affd9378d3fa2088a27e00508046ab9f9dce09922048",
    "device": "cc 9.0 with 132 SMs (the capture's gate), vLLM d9105ea80, nvidia-cutlass-dsl 4.6.2",
    "captured_utc": "2026-10-13",
    "cases": "a ratio-4 layer (state width 1024, the 8-row window) and a ratio-128 layer (W 8, R 16) into fp8_ds_mla caches: every "
             "byte of every compressed cache row",
    "tests": ["tests/program/test_kernel_self_check.py (dsv4-compressor-twin and each row kernel against its Definition)",
              "tests/check/test_sampled_replay_dsv4_rows.py (each row kernel on spec-text statics against its Definition)"],
    "definitions": "verity_vllm/program/registry/dsv4_compressor.py (the reference; the twin is checked against it, never the other way round)",
}

#: family -> what pinned its twin to the H200 (a family absent here has no hardware-specific semantics)
PROVENANCE = {
    **{f: _MHC_PROVENANCE for f in ("MhcPost_v1", "MhcFusedPostPre_v1", "MhcPostPreTf32_v1", "MhcPreBroadcast_v1",
                                     "MhcResidualBroadcast_v1", "HcHead_v1")},
    "Dsv4MlaCompressC4_v1": _COMPRESSOR_PROVENANCE, "Dsv4MlaCompressC128_v1": _COMPRESSOR_PROVENANCE,
}


def _words(a: np.ndarray, n: int, dtype) -> np.ndarray:
    a = np.asarray(a)
    if a.size != n:
        raise Declined(f"an operand of {a.size} words, the signature has {n}")
    return a.astype(dtype, copy=False).reshape(n)


def _bf16(a: np.ndarray, *shape: int) -> np.ndarray:
    return _words(a, int(np.prod(shape)), np.uint16).reshape(shape)


def _f32(a: np.ndarray, *shape: int) -> np.ndarray:
    return _words(a, int(np.prod(shape)), np.uint32).reshape(shape)


def _ports(*arrays: np.ndarray) -> dict[str, np.ndarray]:
    return {str(j): np.asarray(a).reshape(-1) for j, a in enumerate(arrays)}


def _mix(HC: int) -> int:
    return HC * (2 + HC)


def _fuse(S: dict[str, Any]) -> dict[str, Any]:
    return {k: (float(S[k]) if k in ("EPS", "HC_EPS") else int(S[k])) for k in ("NS", "TN", "REPEAT", "EPS", "HC_EPS")}


def mhc_post(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcPost_v1{HC,N}(x, residual, post, comb)` -> HC x N bf16 words."""
    from verity_vllm.program.kernels import mhc_twin as W
    HC, N = int(statics["HC"]), int(statics["N"])
    out = W.post(_bf16(ops[0], 1, N), _bf16(ops[1], 1, HC, N), _f32(ops[2], 1, HC), _f32(ops[3], 1, HC, HC))
    return _ports(out.astype(np.uint16))


def _layer_operands(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> tuple:
    HC, H = int(statics["HC"]), int(statics["H"])
    OUT = _mix(HC)
    return (_bf16(ops[0], 1, H), _bf16(ops[1], 1, HC, H), _f32(ops[2], 1, HC), _f32(ops[3], 1, HC, HC), _f32(ops[4], OUT, HC * H),
            _f32(ops[5], 3), _f32(ops[6], OUT), _bf16(ops[7], H))


def _layer_ports(res, out, post, comb) -> dict[str, np.ndarray]:
    return _ports(res.astype(np.uint16), out.astype(np.uint16), post.astype(np.uint32), comb.astype(np.uint32))


def mhc_fused_post_pre(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcFusedPostPre_v1{HC,H,NS,TH,TN,REPEAT,EPS,HC_EPS}(x, residual, post, comb, fn, scale, base, w)` -> ports 0 (residual', HC x H
    bf16), 1 (the layer input, H bf16), 2 (post', HC f32) and 3 (comb', HC x HC f32)."""
    from verity_vllm.program.kernels import mhc_twin as W
    S = f64_statics(statics)
    f = _fuse(S)
    return _layer_ports(*W.fused_post_pre(*_layer_operands(S, ops), NS=f["NS"], TH=int(S["TH"]), TN=f["TN"], REPEAT=f["REPEAT"],
                                          EPS=f["EPS"], HC_EPS=f["HC_EPS"]))


def mhc_post_pre_tf32(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcPostPreTf32_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}(x, residual, post, comb, fn, scale, base, w)` -> MhcFusedPostPre_v1's ports."""
    from verity_vllm.program.kernels import mhc_twin as W
    S = f64_statics(statics)
    return _layer_ports(*W.post_pre_tf32(*_layer_operands(S, ops), **_fuse(S)))


def mhc_pre_broadcast(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcPreBroadcast_v1{HC,H,NS,TN,REPEAT,EPS,HC_EPS}(x, fn, scale, base, w)` -> ports 0 (the layer input, H bf16), 1 (post, HC
    f32) and 2 (comb, HC x HC f32)."""
    from verity_vllm.program.kernels import mhc_twin as W
    S = f64_statics(statics)
    HC, H = int(S["HC"]), int(S["H"])
    out, post, comb = W.pre_broadcast(_bf16(ops[0], 1, H), _f32(ops[1], _mix(HC), H), _f32(ops[2], 3), _f32(ops[3], _mix(HC)),
                                      _bf16(ops[4], H), **_fuse(S))
    return _ports(out.astype(np.uint16), post.astype(np.uint32), comb.astype(np.uint32))


def mhc_residual_broadcast(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`MhcResidualBroadcast_v1{HC,H}(x)` -> HC x H bf16 words, each stream the embedding's."""
    HC, H = int(statics["HC"]), int(statics["H"])
    return _ports(np.tile(_bf16(ops[0], H), HC))


def hc_head(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`HcHead_v1{HC,H,TH,EPS,HC_EPS}(residual, fn, scale, base)` -> H bf16 words."""
    from verity_vllm.program.kernels import mhc_twin as W
    S = f64_statics(statics)
    HC, H = int(S["HC"]), int(S["H"])
    out = W.hc_head(_bf16(ops[0], 1, HC, H), _f32(ops[1], HC, HC * H), _f32(ops[2], 1), _f32(ops[3], HC), int(S["TH"]), float(S["EPS"]),
                    float(S["HC_EPS"]))
    return _ports(out.astype(np.uint16))


def _compress(rows: Callable[[dict[str, Any]], int], run: Callable[..., tuple], statics: dict[str, Any],
              ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    S = f64_statics(statics)
    D, QB, R = int(S["D"]), int(S["QB"]), rows(S)
    q, rope, e = run(_f32(ops[0], R, D), _f32(ops[1], R, D), _bf16(ops[2], D), _f32(ops[3], QB), S)
    return _ports(np.asarray(q).astype(np.uint8), np.asarray(rope).astype(np.uint16), np.asarray(e).astype(np.uint8))


def dsv4_mla_compress_c4(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4MlaCompressC4_v1{ROWS,D,QB,EPS}(kv, score, w, cs)` -> ports 0 (D - QB e4m3 bytes), 1 (QB bf16) and 2 (D / QB e8m0 bytes)."""
    from verity_vllm.program.kernels import dsv4_compressor_twin as C
    return _compress(lambda S: int(S["ROWS"]), lambda kv, sc, w, cs, S: C.compress_c4(kv, sc, w, cs, int(S["QB"]), float(S["EPS"])),
                     statics, ops)


def dsv4_mla_compress_c128(statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`Dsv4MlaCompressC128_v1{W,R,D,QB,EPS}(kv, score, w, cs)` -> Dsv4MlaCompressC4_v1's ports."""
    from verity_vllm.program.kernels import dsv4_compressor_twin as C
    return _compress(lambda S: int(S["W"]) * int(S["R"]),
                     lambda kv, sc, w, cs, S: C.compress_c128(kv, sc, w, cs, int(S["W"]), int(S["R"]), int(S["QB"]), float(S["EPS"])),
                     statics, ops)


def gather_i32(V: int, statics: dict[str, Any], ops: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    """`GatherI32x{V}_v1(token, column)` -> column[token] (one i32 word), 0xFFFFFFFF for a token outside [0, V)."""
    t = int(_words(ops[0], 1, np.int64)[0]) & 0xFFFFFFFF
    t = t - (1 << 32) if t & 0x80000000 else t
    col = _words(ops[1], V, np.uint32)
    return {"0": np.array([col[t] if 0 <= t < V else 0xFFFFFFFF], dtype=np.uint32)}


#: family -> (row kernel, the ports it returns in the Definition's return order)
ROWS: dict[str, tuple[Callable[..., dict[str, np.ndarray]], tuple[str, ...]]] = {
    "MhcPost_v1": (mhc_post, ("0",)),
    "MhcFusedPostPre_v1": (mhc_fused_post_pre, ("0", "1", "2", "3")),
    "MhcPostPreTf32_v1": (mhc_post_pre_tf32, ("0", "1", "2", "3")),
    "MhcPreBroadcast_v1": (mhc_pre_broadcast, ("0", "1", "2")),
    "MhcResidualBroadcast_v1": (mhc_residual_broadcast, ("0",)),
    "HcHead_v1": (hc_head, ("0",)),
    "Dsv4MlaCompressC4_v1": (dsv4_mla_compress_c4, ("0", "1", "2")),
    "Dsv4MlaCompressC128_v1": (dsv4_mla_compress_c128, ("0", "1", "2")),
    **{f"GatherI32x{v}_v1": (lambda statics, ops, _v=v: gather_i32(_v, statics, ops), ("0",)) for v in GATHER_VOCABS},
}


def _sample_compress(family: str) -> Callable[..., list[np.ndarray]]:
    from verity_vllm.program.kernels import dsv4_compressor_twin as C
    return C.SAMPLES[family]


def _sample_gather(V: int) -> Callable[..., list[np.ndarray]]:
    """Tokens in [0, V) but for one in eight (negative, V and past it), and random column words."""
    def sample(rng: np.random.Generator, n: int, params: Sequence[Any], **statics: Any) -> list[np.ndarray]:
        tok = rng.integers(0, V, n)
        out = rng.random(n) < 1 / 8
        tok = np.where(out, rng.choice(np.array([-1, -(1 << 31), V, V + 1, (1 << 31) - 1]), n), tok)
        return [tok.astype(np.int64).astype(np.uint32), rng.integers(0, 1 << 32, (n, V), dtype=np.uint64).astype(np.uint32)]
    return sample


SAMPLES: dict[str, Callable[[], Callable[..., list[np.ndarray]]]] = {
    "Dsv4MlaCompressC4_v1": lambda: _sample_compress("Dsv4MlaCompressC4_v1"),
    "Dsv4MlaCompressC128_v1": lambda: _sample_compress("Dsv4MlaCompressC128_v1"),
    **{f"GatherI32x{v}_v1": (lambda _v=v: _sample_gather(_v)) for v in GATHER_VOCABS},
}


def check_targets(family: str) -> list:
    """The small specialisations `family`'s self-check binds (the served statics are too wide for the reference)."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import dsv4_compressor as C
    from verity_vllm.program.registry import dsv4_decode_indices as DI
    from verity_vllm.program.registry import mhc as M
    eps = 1e-6
    fuse = dict(REPEAT=2, EPS=eps, HC_EPS=eps)
    if family.startswith("GatherI32x"):
        return [DI.gather_i32(int(family[len("GatherI32x"):-len("_v1")]))]
    return {
        "MhcPost_v1": [bind(M.MhcPost, HC=4, N=8), bind(M.MhcPost, HC=2, N=4)],
        "MhcFusedPostPre_v1": [bind(M.MhcFusedPostPre, HC=2, H=32, NS=2, TH=4, TN=1, **fuse)],
        "MhcPostPreTf32_v1": [bind(M.MhcPostPreTf32, HC=2, H=64, NS=2, TN=2, **fuse)],
        "MhcPreBroadcast_v1": [bind(M.MhcPreBroadcast, HC=2, H=64, NS=1, TN=2, **fuse)],
        "MhcResidualBroadcast_v1": [bind(M.MhcResidualBroadcast, HC=4, H=8)],
        "HcHead_v1": [bind(M.HcHead, HC=2, H=16, TH=th, EPS=eps, HC_EPS=eps) for th in (1, 2)],
        "Dsv4MlaCompressC4_v1": [bind(C.Dsv4MlaCompressC4, ROWS=rows, D=32, QB=16, EPS=eps) for rows in (8, 4)],
        "Dsv4MlaCompressC128_v1": [bind(C.Dsv4MlaCompressC128, W=2, R=4, D=32, QB=16, EPS=eps)],
    }[family]


def register_rows(evaluators: dict, row: Callable, provenance: dict) -> None:
    """Register this module's row evaluators without growing the shared rows catalog."""
    evaluators |= EVALUATORS
    provenance |= PROVENANCE
    for family, (fn, returns) in ROWS.items():
        sample = SAMPLES.get(family)
        row(family, fn, returns, sample=sample() if sample else None, check_targets=lambda rng, f=family: check_targets(f))


__all__ = ["EVALUATORS", "GATHER_VOCABS", "PROVENANCE", "ROWS", "check_targets", "register_rows"]
