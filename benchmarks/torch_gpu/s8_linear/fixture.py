"""The exact W8A8 linear's fixture, on the CPU: the large-model linears it runs at, its inputs' edges, and the twin of
`ScaledMmS8Coordinate_v1{K, BlackwellS8ImmaDot32_v1}` the GPU's words are checked against. numpy and stdlib only.

The twin, on the s32 sums z = X W^T (exact for K < 2^17 in any order, so any exact s32 GEMM gives the Definition's z):

    y[i, n] = F2fpBf16(F32MulV2(F32MulV2(S32ToF32Rn(z[i, n]), s_a[i]), s_w[n]))

numpy's int32 -> float32 cast and float32 multiply round to nearest even and keep subnormals (`epilogue` refuses a process
where something set flush-to-zero or denormals-are-zero), and every NaN ends as 0x7FFF (`tc.f2fp_bf16_batch`), so a NaN's
payload in between doesn't change the word. `tests/test_twin.py` holds the twin to the Definition's evaluator.

Inputs (`run.py` builds them on the GPU; the codes from a torch generator, the scales from `scales`):
    codes    uniform over all of s8, -128 included (the Definition's domain; `int8_exact`'s quantizer only reaches +-127).
             Row 0 is random, so M = 1 is a random row; rows 1-3 and columns 0-3 are `STRUCTURED` (all -128, all 127,
             zero, alternating), so their products reach +-16384 K, the s32 edge at K = 131040.
    scales   positive and spread over 2^-24..2^2 (s_a) and 2^-20..2^-2 (s_w); rows 4.. take `SPECIAL_SA` and columns 4.. take
             `SPECIAL_SW` in turn: infinities, NaNs of either sign and payload, signed zeros, the largest finite and an
             overflow, the least normal, subnormals down to 2^-149, and pairs where the first product is subnormal and the
             second normal again (a flush-to-zero on either multiply or on a subnormal input changes those words).
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass

import numpy as np

#: the output head's rows (once a model, not a layer)
HEAD_N = 256


@dataclass(frozen=True)
class Model:
    """A decoder-only shape: width d, its layers, its linears (name, N, K, the quantizer whose codes it reads: ``attn``,
    ``o``, ``mlp`` or ``down``), its attention (query heads of head_dim, at a context), and the sequences of a decode step."""
    name: str
    d: int
    layers: int
    linears: tuple[tuple[str, int, int, str], ...]
    heads: int
    head_dim: int
    context: int
    sequences: int

    @property
    def mlp(self) -> int:
        return next(n for name, n, _k, _q in self.linears if name == "up")

    @property
    def weight_bytes(self) -> int:
        return self.layers * sum(n * k for _, n, k, _ in self.linears) + HEAD_N * self.d


def _mha(d: int, layers: int, heads: int, head_dim: int, context: int, sequences: int, name: str) -> Model:
    return Model(name, d, layers, (("qkv", 3 * d, d, "attn"), ("o", d, d, "o"), ("up", 4 * d, d, "mlp"),
                                   ("down", d, 4 * d, "down")), heads, head_dim, context, sequences)


#: the models' shapes (estimated, frozen with the circuits), each a non-gated 4x MLP and int8 weights: mqa16k52 for one
#: H200 (MQA at width 16384, 52 layers, context 32, 20,480 sequences a step), and for the RTX PRO 6000 MQA at width 16384
#: and MHA at width 8192
_MQA16K = (("q", 16384, 16384, "attn"), ("kv", 128, 16384, "attn"), ("o", 16384, 16384, "o"), ("up", 65536, 16384, "mlp"),
           ("down", 16384, 65536, "down"))
MODELS = {
    "mqa16k52": Model("mqa16k52", 16384, 52, _MQA16K, 256, 64, 32, 20480),
    "mqa16k": Model("mqa16k", 16384, 35, _MQA16K, 256, 64, 96, 4096),
    "mha8k": _mha(8192, 70, 128, 64, 32, 1023, "mha8k"),
}
DEFAULT_MODELS = ("mqa16k52",)


def model(name: str) -> Model:
    """One of `MODELS`, or ``d<width>``: the fused-QKV 4x MLP layer at that width, 80 GB of int8 weights' layers, heads of
    128, a context of 0 and a decode step of 256."""
    if name in MODELS:
        return MODELS[name]
    hit = re.fullmatch(r"d(\d+)", name)
    if not hit:
        raise ValueError(f"{name}: not one of {sorted(MODELS)} or d<width>")
    d = int(hit.group(1))
    return _mha(d, round(80e9 / (12 * d * d)), d // 128, 128, 0, 256, name)


def linears(d: int) -> dict[str, tuple[int, int]]:
    """The fused-QKV 4x MLP layer's linears at width d, (N, K)."""
    return {name: (n, k) for name, n, k, _ in model(f"d{d}").linears}


#: what a layer commits a token row (commit.py): the served commit's quantizer tables, the fused cuts, every op
TABLE_SETS = ("quantize", "fused", "every")


def tables(md: Model, table_set: str) -> list[tuple[str, int, bool]]:
    """One layer's tables, (op under ``model.layers.L.``, bytes a token row, whether it ends in an f32 scale): the four
    quantizers' s8 codes; fused adds each linear's bf16 output, attention's output and its probabilities (heads x the full
    context), and the two residual adds; every adds the two RMSNorms', the ReLU's and attention's scores."""
    d, up, att = md.d, md.mlp, md.heads * md.head_dim
    rows = [("self_attn.input_quant", d, True), ("self_attn.o_proj.input_quant", att, True), ("mlp.input_quant", d, True),
            ("mlp.down_proj.input_quant", up, True)]
    if table_set == "quantize":
        return rows
    rows += [(f"{'mlp' if q in ('mlp', 'down') else 'self_attn'}.{name}_proj", 2 * n, False) for name, n, _k, q in md.linears]
    rows += [("self_attn.attn_out", 2 * att, False), ("self_attn.residual", 2 * d, False), ("mlp.residual", 2 * d, False)]
    if md.context:
        rows.append(("self_attn.probs", 2 * md.heads * md.context, False))
    if table_set == "every":
        rows += [("input_layernorm", 2 * d, False), ("post_attention_layernorm", 2 * d, False), ("mlp.act", 2 * up, False)]
        if md.context:
            rows.append(("self_attn.scores", 2 * md.heads * md.context, False))
    if table_set not in TABLE_SETS:
        raise ValueError(f"{table_set}: not one of {TABLE_SETS}")
    return rows
#: the largest K the s8 Definitions take that is a multiple of 32 (S8_K_LIMIT = 2^17 refuses 2^17)
EDGE_K = 131040
S32_MAX = 2**31 - 1

#: Ms the gate runs: decode (a masked tile at 7, 33, 257 and 1023, the models' steps at 1023, 4096 and 20480, the e2e demo's
#: batches at 4096 and 16384) and prefill
GATE_MS = (1, 7, 16, 32, 33, 64, 128, 256, 257, 512, 1023, 2048, 4096, 8192, 16384, 20480)
STRUCTURED = ("min", "max", "zero", "alt")
STRUCT_ROWS = {1: "min", 2: "max", 3: "zero"}
STRUCT_COLS = {0: "min", 1: "max", 2: "zero", 3: "alt"}
FIRST_SPECIAL = 4


def shapes(models: tuple[str, ...] = DEFAULT_MODELS, with_head: bool = True) -> list[tuple[str, int, int]]:
    """(name, N, K) of every distinct linear of `models`, the layers' first, then the heads."""
    out: dict[tuple[int, int], str] = {}
    for m in map(model, models):
        for name, n, k, _ in m.linears:
            out.setdefault((n, k), f"{m.name}:{name}")
    if with_head:
        for m in map(model, models):
            out.setdefault((HEAD_N, m.d), f"{m.name}:head")
    return [(name, n, k) for (n, k), name in out.items()]


def f32w(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


SPECIAL_SA = (0x7F800000, 0xFF800000, 0x7FC00000, 0x7F800001, 0xFFFFFFFF, 0x00000000, 0x80000000, 0x7F7FFFFF,
              f32w(3e38), 0x00800000, f32w(1e-41), 0x00000001, f32w(1e-44), f32w(1e-30), f32w(-1e-3), f32w(1.0))
SPECIAL_SW = (0x7F800000, 0xFF800000, 0x7FC00000, 0x7FBFFFFF, 0x00000000, 0x80000000, 0x7F7FFFFF, f32w(1e30),
              f32w(2.0**100), 0x00800000, f32w(1e-15), f32w(1e-36), f32w(1e-41), 0x00000001, f32w(-1e-2), f32w(1.0))


def structured(kind: str, k: int) -> np.ndarray:
    """One row of s8 codes: all -128, all 127, zero, or -128 and 127 alternating."""
    if kind == "min":
        return np.full(k, -128, dtype=np.int8)
    if kind == "max":
        return np.full(k, 127, dtype=np.int8)
    if kind == "zero":
        return np.zeros(k, dtype=np.int8)
    if kind == "alt":
        return np.where(np.arange(k) % 2 == 0, -128, 127).astype(np.int8)
    raise ValueError(kind)


def _spread(rng: np.random.Generator, count: int, lo: float, hi: float) -> np.ndarray:
    return np.exp2(rng.uniform(lo, hi, count)).astype(np.float32)


def _with_specials(s: np.ndarray, special: tuple[int, ...]) -> np.ndarray:
    for i, w in enumerate(special[:max(0, len(s) - FIRST_SPECIAL)]):
        s.view(np.uint32)[FIRST_SPECIAL + i] = w
    return s


def act_scales(m: int, seed: int, special: bool = True) -> np.ndarray:
    """s_a [m] float32: spread over 2^-24..2^2, and with `special` `SPECIAL_SA` from row 4 on."""
    s = _spread(np.random.default_rng(seed), m, -24, 2)
    return _with_specials(s, SPECIAL_SA) if special else s


def weight_scales(n: int, seed: int, special: bool = True) -> np.ndarray:
    """s_w [n] float32: spread over 2^-20..2^-2, and with `special` `SPECIAL_SW` from column 4 on."""
    s = _spread(np.random.default_rng(seed), n, -20, -2)
    return _with_specials(s, SPECIAL_SW) if special else s


def scales(m: int, n: int, seed: int, special: bool = True) -> tuple[np.ndarray, np.ndarray]:
    return act_scales(m, seed, special), weight_scales(n, seed + 1, special)


def fp_env_ok() -> bool:
    """Whether this process's float32 arithmetic keeps subnormals: no flush-to-zero on results, none on inputs."""
    tiny = np.float32(2.0**-126)
    with np.errstate(all="ignore"):
        ftz = np.multiply(tiny, np.float32(0.5)) == 0
        daz = np.multiply(np.uint32(1).view(np.float32), np.float32(2.0**23)) == 0
    return not ftz and not daz


def epilogue(z: np.ndarray, sa: np.ndarray, sw: np.ndarray) -> np.ndarray:
    """The Definition's words y [m, n] uint16 from the s32 sums z [m, n], s_a [m] and s_w [n] (float32)."""
    from verity.ml.kernels.tc import f2fp_bf16_batch

    if not fp_env_ok():
        raise RuntimeError("this process flushes float32 subnormals (an FTZ/DAZ library?); the twin would be wrong")
    z = np.asarray(z, dtype=np.int32)
    sa = np.asarray(sa, dtype=np.float32).reshape(-1, 1)
    sw = np.asarray(sw, dtype=np.float32).reshape(1, -1)
    with np.errstate(all="ignore"):
        t = np.multiply(np.multiply(z.astype(np.float32), sa), sw)
    return f2fp_bf16_batch(t.view(np.uint32))[0]


def sums(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """z = x w^T exactly, int32: x [m, K] and w [n, K] s8, by float64 matmul (every partial sum is an integer below 2^53)."""
    k = x.shape[1]
    if k >= 2**17 or w.shape[1] != k:
        raise ValueError(f"K {k}: an s8 x s8 sum is exact in s32 below 2^17")
    z = np.asarray(x, dtype=np.float64) @ np.asarray(w, dtype=np.float64).T
    if np.abs(z).max(initial=0) > S32_MAX:
        raise AssertionError("a sum left s32")
    return z.astype(np.int32)
