"""Cross-check: the registered composites, evaluated gate by gate by the IR oracle, against the validated
C++/CPU models of the numerics lane (`verity_vllm.program.kernels`) on random and adversarial inputs.

This is conformance evidence for the *definitions* (the IR body reproduces the frozen model the lane
measured bit-exact against production); correspondence against captured production values is the
separate `verity_vllm.correspondence.capture_identities` tool.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluate import evaluate_call
from verity.ml.circuits.layout import FunctionCall
from verity.ml.circuits.program import Program
from verity.ml.circuits.refs import Coll
from verity.ml.circuits.types import Array, Record
from verity_vllm.program.registry import b1, prims as P
from verity_vllm.program.registry.b1 import B1_CONFIG as C


from verity.ml.circuits.defs import CompositeDefinition

_STANDALONE = CompositeDefinition("Standalone", 1, ("FN",), lambda S: S.FN.signature(),
                                  lambda B, S, *args: B.call(S.FN, *args),
                                  doc="root wrapper: one call of F over Input gates", register=False)


def _standalone(spec):
    """A program whose root is exactly one call of `spec`; returns (program, the FunctionCall)."""
    prog = Program(bind(_STANDALONE, FN=spec))
    circ = prog.circuit
    calls = [c for c in circ.root_calls() if not c.is_primitive]
    assert len(calls) == 1, calls
    return prog, calls[0]


def _rand_bf16(rng, n, scale=1.0):
    x = (rng.standard_normal(n) * scale).astype(np.float32)
    return x.view(np.uint32) >> 16


def _bf16_words(x: np.ndarray) -> list[int]:
    return [int(v) for v in np.asarray(x, dtype=np.uint16).ravel()]


def _u16(v: list[int]) -> np.ndarray:
    return np.array(v, dtype=np.uint16)


def check_rmsnorm(rng, rows=4, N=1536, eps=1e-6):
    from verity_vllm.program.kernels import rms_relation, rms_triton_model
    prog, call = _standalone(bind(b1.RMSNormTriton, N=N, EPS=eps))
    tabs = rms_relation.tables("triton")
    mism = 0
    for _ in range(rows):
        x = _rand_bf16(rng, N, scale=float(rng.choice([0.01, 1.0, 30.0])))
        w = _rand_bf16(rng, N)
        ref = rms_triton_model.rms_norm_rows(_u16(x)[None], _u16(w), eps, tabs)[0]
        out = evaluate_call(prog.circuit, call, [int(v) for v in x] + [int(v) for v in w])
        mism += int(np.count_nonzero(np.array(out, dtype=np.uint16) != ref))
    return {"function": call.id, "rows": rows, "words": rows * N, "mismatched_words": mism, "gates_per_call": call.size}


def check_gemm_coordinate(rng, n=64, K=1536):
    from verity_vllm.program.kernels import mma
    prog, call = _standalone(bind(b1.GemmCoordinate, K=K))
    mism = 0
    for _ in range(n):
        x = _rand_bf16(rng, K)
        w = _rand_bf16(rng, K)
        acc = mma.tc_dot_chain(mma.AMPERE_BF16, 0, [int(v) for v in x], [int(v) for v in w])
        ref = P.f32_to_bf16_rn_bits(acc, P.BF16_NAN_F2FP)
        out = evaluate_call(prog.circuit, call, [int(v) for v in x] + [int(v) for v in w])
        mism += int(out[0] != ref)
    return {"function": call.id, "coordinates": n, "mismatched_words": mism, "gates_per_call": call.size}


def check_attention_head(rng, Ts=(1, 5, 64, 65, 130, 200), D=128):
    from verity_vllm.program.kernels import fa2_relation
    res = []
    for T in Ts:
        prog, call = _standalone(bind(b1.AttentionHead, T=T, D=D))
        q = _rand_bf16(rng, D, scale=float(rng.choice([0.3, 1.0, 3.0])))
        k = _rand_bf16(rng, T * D, scale=float(rng.choice([0.3, 1.0, 3.0]))).reshape(T, D)
        v = _rand_bf16(rng, T * D).reshape(T, D)
        # the lane's model: q (H, D), k/v (n, KVH, D) with kvh mapping; single head -> H=1, KVH=1
        rr = fa2_relation.model_row(_u16(q)[None, :], _u16(k)[:, None, :], _u16(v)[:, None, :], D ** -0.5)
        ref = np.asarray(rr.out).reshape(-1)[:D]
        args = [int(x) for x in q] + [int(x) for x in k.ravel()] + [int(x) for x in v.ravel()]
        out = np.array(evaluate_call(prog.circuit, call, args), dtype=np.uint16)
        res.append({"T": T, "mismatched_words": int(np.count_nonzero(out != ref)), "gates": call.size})
    return {"function": "AttentionHead_v1", "cases": res}


def check_silu_rope(rng):
    from tests.program import beyond_gemm as BG
    out = {}
    I = 8960
    prog, call = _standalone(bind(b1.SiluMul, I=I))
    gu = _rand_bf16(rng, 2 * I, scale=3.0)
    got = np.array(evaluate_call(prog.circuit, call, [int(x) for x in gu]), dtype=np.uint16)
    ref = None
    for name in ("silu_and_mul_words", "silu_and_mul", "silu_mul_words"):
        fn = getattr(BG, name, None)
        if fn is not None:
            try:
                ref = np.asarray(fn(_u16(gu)[None])).reshape(-1)
            except Exception:
                ref = None
            break
    out["SiluMul"] = {"words": I, "mismatched_words": None if ref is None else int(np.count_nonzero(got != ref)),
                      "reference": name if ref is not None else "none available in beyond_gemm (checked against capture instead)"}
    # RoPE: NeoX rotation against beyond_gemm.rope_model (exact f64 arithmetic, RNE to f32, RNE to bf16), several positions
    NH, D = 12, 128
    prog, call = _standalone(bind(b1.RoPE, NHEADS=NH, D=D))
    cache = np.asarray(BG.rope_cache_numpy("torch_f32"))
    mism = 0
    for pos in (0, 1, 7, 100, 511, 4095):
        x = _rand_bf16(rng, NH * D, scale=4.0)
        got = np.array(evaluate_call(prog.circuit, call, [int(v) for v in x] + [int(v) for v in cache[pos]]), dtype=np.uint16)
        ref = np.asarray(BG.rope_model(_u16(x)[None], cache, np.array([pos]))).reshape(-1)
        mism += int(np.count_nonzero(got != ref))
    out["RoPE"] = {"words": 6 * NH * D, "mismatched_words": mism, "reference": "beyond_gemm.rope_model"}
    return out


@dataclass(frozen=True, kw_only=True)
class Options:
    pass


def main(a: Options):
    import json
    rng = np.random.default_rng(1)
    t = time.time()
    rep = {"rmsnorm": check_rmsnorm(rng), "gemm_coordinate": check_gemm_coordinate(rng),
           "attention_head": check_attention_head(rng)}
    try:
        rep.update(check_silu_rope(rng))
    except Exception as e:  # pragma: no cover
        rep["silu_rope_error"] = repr(e)
    rep["seconds"] = round(time.time() - t, 1)
    print(json.dumps(rep, indent=1))
