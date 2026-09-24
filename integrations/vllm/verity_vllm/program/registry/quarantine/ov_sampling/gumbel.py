"""ov-sampling/GumbelArgmaxBf16@v1 — vLLM d9105ea80 `_gumbel_sample_kernel` (temperature 1, no top-k/top-p) as an exact
Verity definition: the served token of a sampled row is

    key   = philox4x32_10(counter=(pos_lo, pos_hi, 0, 0), key=(seed_lo, seed_hi))[0]       # tl.randint(seed, pos)
    u[v]  = max(uniform(philox4x32_10(counter=(v, 0, 0, 0), key=(key, 0))[0]), 2^-31')     # tl.rand(key, v), _TL_RAND_MIN
    g[v]  = -logf(-log1pf(-u[v]))                                                          # libdevice, FTZ, bit-exact
    token = argmax_v (f32(logits[v]) + g[v])                                               # first index among the maxima

with `seed` the request's `SamplingParams.seed` (int64) and `pos` the absolute position of the sampled token, both
STATICS of the specialization (`GumbelArgmaxBf16_v1{V, SEED, POS}`), the bf16 logits row the only operand.  The noise
is therefore a pure function of (SEED, POS, v) and never a root, a snapshot or a tolerance: `logf` / `log1pf` are the
libdevice bodies (`__nv_logf`, `__nv_log1pf` of libdevice.10.bc, transcribed instruction by instruction: fma.rn.ftz,
add.rz.ftz, mul.ftz, integer exponent extraction), validated bit-exact against the RTX 4090 on 8 (seed, pos) rows x
151936 lanes (out/gen/lanes/ov-sampling/probe/compare.json), on the 176 sampled tokens of the nine B1 captures
(`ov_recompute.json`, REPORT.md §6) and by the G4 differential tests (`gumbel_difftest.py`, `gumbel_temp_difftest.py`:
64/64 cases against vLLM's kernels on the RTX 4090, including all-equal logits, ±max-finite, subnormal and alternating rows).

Primitives (leaf semantics on encodings, pure Python integers / IEEE doubles only):
  * `GumbelStreamKey_v1(seed: Value<64>, pos: Value<64>) -> Value<32>`
  * `GumbelNoiseLane_v1(key: Value<32>, v: I32) -> F32`
Composites:
  * `GumbelArgmaxStep_v1` — the scan step `(best, best_i, i), x, key -> (best', best_i', i+1), best_i'`
  * `GumbelArgmaxBf16_v1{V, SEED, POS}(logits: Array<V, bf16>) -> I32`
  * `GumbelArgmaxTempStep_v1{T}`, `GumbelArgmaxTempBf16_v1{V, SEED, POS, T}` — the same at temperature T not in {0, 1}: the
    `_temperature_kernel` quotient `div.full.f32(f32(x), T)` = `F32Mul(DivFullScaleA(f32(x), T), DivFullRcp(T))` (the
    registry's div.full model) is folded into the lane, T the fp32 bit pattern of the request's temperature.

Not covered (Unsupported by the recognizer, by design): APPLY_TEMPERATURE=True (in-kernel `div.full.f32`), IS_DRAFTING,
PER_TOKEN_COL, USE_FP64, rows masked by top-k / top-p (`-inf` lanes written by `_topk_topp_kernel` / `_topp_sb_*`: a checked
witness, not a function of the row — REPORT.md §6.3, "RNG/root/state contract").
"""

from __future__ import annotations

from verity.ir.defs import bind, composite
from verity.ir.refs import tuple_of
from verity_vllm.program.registry import prims as P
from verity_vllm.program.registry import ref_prims as RP
from verity_vllm.program.registry.b1 import const
from verity_vllm.program.registry.prims import BF16, F32, I32
from verity.ir.types import Array, Tuple

SEMANTIC_ID = "ov-sampling/GumbelArgmaxBf16@v1"

# [X-06] the exact fp32 helpers, Philox / tl.rand, libdevice transcriptions and the two primitives now live in the main registry
# (`verity_vllm.program.registry.sampling`, seed/pos/temp as OPERANDS); this quarantine module keeps the static-seed composites and their
# G4 difftest adapters (64/64 on the RTX 4090) and re-exports the shared pieces so the existing tests keep their names.
from verity_vllm.program.registry.sampling import (  # noqa: E402,F401
    I64,
    M32,
    M64,
    U32,
    GumbelNoiseLane,
    GumbelStreamKey,
    add_rz_ftz,
    fma_ftz,
    noise_lane,
    nv_log1pf,
    nv_logf,
    philox4x32_10_word0,
    stream_key,
)


@composite("GumbelArgmaxStep", 1, [], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", BF16), ("key", U32)),
                                                  Tuple(Tuple(F32, I32, I32), I32)), conformance="model-untested")
def GumbelArgmaxStep(B, S, carry, x, key):
    """One vocabulary lane: y = f32(x) + g(key)[i] (add.ftz.f32); take (y, i) iff y > best (strict: first index wins,
    NaN never wins); i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    g = B.call(GumbelNoiseLane, key, i)
    y = B.call(P.F32AddFtz, B.call(P.Bf16ToF32, x), g)
    gt = B.call(RP.F32GtStrict, y, best)
    nb = B.call(P.SelectF32, gt, y, best)
    ni = B.call(P.SelectI32, gt, i, best_i)
    nxt = B.call(P.I32Add, i, B.call(const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("GumbelArgmaxBf16", 1, ["V", "SEED", "POS"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def GumbelArgmaxBf16(B, S, logits):
    """Gumbel-max token selection at temperature 1: argmax_v(f32(logits[v]) + g(SEED, POS)[v]), lowest index on ties."""
    key = B.call(GumbelStreamKey, B.call(const(64, int(S.SEED) & M64)), B.call(const(64, int(S.POS) & M64)))
    zero = B.call(const(32, 0))
    y0 = B.call(P.F32AddFtz, B.call(P.Bf16ToF32, logits[0]), B.call(GumbelNoiseLane, key, zero))
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    step = B.scan(GumbelArgmaxStep, init, xs=(logits[1:S.V],), shared=(key,))
    return step[0][1]


def _tempered(B, x, t_bits: int, rcp):
    """`_temperature_kernel`: `f32(x) / T` with Triton's fp32 `/` = `div.full.f32` (registry model, b1.RMSNormTriton):
    F32Mul(DivFullScaleA(f32(x), T), DivFullRcp(T)).  Verified bit-exact on 16 x 151936 lanes of the B1 temp07 capture
    (REPORT.md: IEEE division reproduces only ~90.7% of the lanes)."""
    xf = B.call(P.Bf16ToF32, x)
    return B.call(P.F32Mul, B.call(P.DivFullScaleA, xf, B.call(const(32, t_bits))), rcp)


@composite("GumbelArgmaxTempStep", 1, ["T"], lambda S: ((("carry", Tuple(F32, I32, I32)), ("x", BF16), ("key", U32), ("rcp", F32)),
                                                          Tuple(Tuple(F32, I32, I32), I32)), conformance="model-untested")
def GumbelArgmaxTempStep(B, S, carry, x, key, rcp):
    """One vocabulary lane at temperature T: y = div.full(f32(x), T) + g(key)[i]; take (y, i) iff y > best; i += 1."""
    best, best_i, i = carry[0], carry[1], carry[2]
    g = B.call(GumbelNoiseLane, key, i)
    y = B.call(P.F32AddFtz, _tempered(B, x, int(S.T) & M32, rcp), g)
    gt = B.call(RP.F32GtStrict, y, best)
    nb = B.call(P.SelectF32, gt, y, best)
    ni = B.call(P.SelectI32, gt, i, best_i)
    nxt = B.call(P.I32Add, i, B.call(const(32, 1)))
    return tuple_of(tuple_of(nb, ni, nxt), ni)


@composite("GumbelArgmaxTempBf16", 1, ["V", "SEED", "POS", "T"], lambda S: ((("logits", Array(S.V, BF16)),), I32), conformance="model-untested")
def GumbelArgmaxTempBf16(B, S, logits):
    """Gumbel-max token selection at temperature T (static, fp32 bits, T not in {0, 1}):
    argmax_v(div.full(f32(logits[v]), T) + g(SEED, POS)[v]), lowest index on ties.  vLLM: `_temperature_kernel` (in place,
    div.full.f32) then `_gumbel_sample_kernel` (APPLY_TEMPERATURE=False)."""
    t_bits = int(S.T) & M32
    key = B.call(GumbelStreamKey, B.call(const(64, int(S.SEED) & M64)), B.call(const(64, int(S.POS) & M64)))
    rcp = B.call(P.DivFullRcp, B.call(const(32, t_bits)))
    zero = B.call(const(32, 0))
    y0 = B.call(P.F32AddFtz, _tempered(B, logits[0], t_bits, rcp), B.call(GumbelNoiseLane, key, zero))
    init = tuple_of(y0, zero, B.call(const(32, 1)))
    step = B.scan(bind(GumbelArgmaxTempStep, T=t_bits), init, xs=(logits[1:S.V],), shared=(key, rcp))
    return step[0][1]


__all__ = ["SEMANTIC_ID", "GumbelStreamKey", "GumbelNoiseLane", "GumbelArgmaxStep", "GumbelArgmaxBf16", "GumbelArgmaxTempStep",
           "GumbelArgmaxTempBf16", "stream_key", "noise_lane", "nv_logf", "nv_log1pf", "fma_ftz", "add_rz_ftz"]
