"""Subcircuit templates: each a registered IR Definition with typed parameters, bound to the ports an input set stores.

    template  = a named Definition family with typed parameters           (``gemm-coordinate``, ``attention-head``, ...)
    subcircuit = a template with every parameter bound                     (``gemm-coordinate/k2048/sm80-mma-bf16``)

A :class:`Subcircuit` carries what both kinds of input set need: the IR target its instances are evaluated with
(``verity.ml.circuits.evaluation.evaluate_batch``, the one evaluator: a registered kernel where one is self-checked against the gate
reference, else the reference), its input and output ports, and the ``relation`` / set name the vLLM VU exporter writes
(``verity_vllm.pipeline.vu_export``), so a synthetic set and a captured set of one subcircuit have the same name,
relation and ports.  :func:`identify` maps a set's ``relation`` back to its subcircuit, whichever wrote it.

The templates of row #101 (Llama-3.2-1B) and their Definitions:

  gemm-coordinate            GemmCoordinate_v2{K, DOT}          x[K], w[K] -> y            (``verity.ml.circuits.ops.gemm``)
  attention-head             AttentionHead_v3{T, D, BN}         q[D], k[T*D], v[T*D] -> out  (FA2; T varies per instance)
  rope-head                  RoPEHead_v1{D}                     x[D], cs[D] -> out
  rmsnorm-fused-cuda         RMSNormFusedCuda_v2{N, EPS}        in0 x, in1 res, in2 w -> out.0, out.1
  rmsnorm-triton             RMSNormTriton_v1{N, EPS}           in0 x, in1 w -> out.out
  silu-mul                   SiluMul_v1{I}                      in0 gu[2I] -> out.out
  gumbel-top-p-token-select  GumbelTopPTokenSelect_v1{V}        in0 logits, in1 top_p, in2 seed, in3 pos, in4 temp, in5 splits
                                                                -> out.sampled_token_ids (u64, as the row kernel returns it)

GLM-5.3's served SiLU·mul (vLLM ``_C.silu_and_mul``) is ``silu-mul-v2``, ``SiluMul_v2{I}`` (``registry.act``), on the same ports.
Its served RMSNorms (vLLM's width-8 CUDA kernels, ``registry.cuda_rmsnorm``) are ``rmsnorm-fused-cuda-vec8``,
``RMSNormFusedCudaVector8_v1{N, EPS, BLOCK}`` on ``rmsnorm-fused-cuda``'s ports, and ``rmsnorm-cuda-vec8``,
``RMSNormCudaVector8_v1{N, EPS, BLOCK}`` on ``rmsnorm-triton``'s; BLOCK is the launch's block, which the token count picks.

The registered compiler's dead-free Definitions (``registry.dead_free``) have templates of their own: ``rmsnorm-triton-v5``,
``RMSNormTriton_v5{N, EPS}`` on ``rmsnorm-triton``'s ports, and ``attention-head-v18``, ``AttentionHead_v18{T, D, BN, DOT, INV,
MASKED_FROM}`` on ``attention-head``'s, evaluated as ``Attention_v18{NH=1, KVH=1}``: sm120's FA2 steps at a decode-only launch,
whose MASKED_FROM is T's (``ceil(T/BN) - 1``).  ``TokenSelect_v4``'s two calls are ``token-select-chunk`` and
``token-select-combine``.

All but ``gemm-coordinate`` are Definitions of the vLLM integration's registry (``verity.ml.definitions.registry``), imported
lazily: this module and the GEMM template need only ``verity``.  Two templates are evaluated through an equal Definition
that has a kernel: ``AttentionHead_v3`` as ``Attention_v3{NH=1, KVH=1}`` (a batch of that head) and ``RoPEHead_v1`` as
``RoPE_v1{NHEADS=1}``; ``test_templates`` checks both against the head's gate reference.  The captured GEMM sets name the
vLLM step ``AmpereBF16TcDot16_v1``; it and core's ``_v2`` evaluate the same function (``tc_dot_total`` on
``AMPERE_BF16_M16N8K16``), so both evaluate through core's ``_v2`` (tested on sampled inputs).

Sampling (:meth:`Subcircuit.draw`): draw ``j`` of a synthetic set reads each port's words from the counter stream
``key.stream((port, j), ...)`` of ``verity.ml.randomness`` (SHA-512 counter mode, index-addressable), four bytes per word, so
draw ``j`` is computable from (key, j) alone in any language.  A BF16 word from 32 bits ``r``: sign ``r & 1``, mantissa
``(r >> 1) & 0x7F``, exponent field ``c - h + ((r >> 8) & 0xFFFF) mod (2h + 1)`` for the port's window (centre ``c``,
half-width ``h`` chosen per draw from the port's classes by ``key.uniform(len, ("class", port, j))``), replaced by an edge
word (``+-0``, ``+-`` smallest subnormal, ``+-1``) ``EDGES[(r >> 29) mod 6]`` when ``(r >> 24) & 0x1F == 0``.  GEMM draws
share their activation row in tiles of ``tile`` (the shape serving uses): ``x`` and its class are drawn at ``j // tile``.
"""
from __future__ import annotations

import hashlib
import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from verity.ml.randomness import Key

WORD_DTYPES = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "u64": np.uint64}
BF16_EDGES = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x3F80, 0xBF80], dtype=np.uint16)


@dataclass(frozen=True)
class Port:
    """One stored array per instance: ``words`` per instance (``None``: it varies, the set stores offsets)."""

    name: str
    dtype: str
    words: int | None


@dataclass(frozen=True)
class Draw:
    """One synthetic draw: its input words per port, the statics that vary per instance, and what ``index.json`` records."""

    inputs: dict[str, np.ndarray]
    instance_statics: dict[str, int]
    sub: dict[str, Any]


class NotEvaluable(ValueError):
    """The subcircuit has no IR Definition in this tree (its semantics are only a ``verity.ml.circuits.silicon`` model)."""


# --- word samplers ------------------------------------------------------------------------------------------------------------

def stream_u32(key: Key, index: tuple, n: int) -> np.ndarray:
    return np.frombuffer(key.stream(index, 4 * n), dtype="<u4").astype(np.uint32)


def stream_u64(key: Key, index: tuple, n: int) -> np.ndarray:
    return np.frombuffer(key.stream(index, 8 * n), dtype="<u8").astype(np.uint64)


def bf16_words(r: np.ndarray, centre: int, half: int, edges: np.ndarray = BF16_EDGES) -> np.ndarray:
    """BF16 words from 32 random bits each (the module docstring's rule): normal words in the exponent window, one in 32 an edge word."""
    r = np.asarray(r, dtype=np.uint32)
    lo = centre - half
    if lo < 1 or centre + half > 254:
        raise ValueError(f"exponent window {centre} +- {half} leaves the normal range")
    e = lo + ((r >> 8) & 0xFFFF) % np.uint32(2 * half + 1)
    w = ((r & 1) << 15) | (e << 7) | ((r >> 1) & 0x7F)
    edge = ((r >> 24) & 0x1F) == 0
    w = np.where(edge, edges[((r >> 29) % np.uint32(len(edges))).astype(np.int64)], w)
    return w.astype(np.uint16)


@dataclass(frozen=True)
class Bf16Window:
    """A BF16 port's sampler: exponent window centre and the half-widths one draw picks from."""

    centre: int = 127
    halves: tuple[int, ...] = (4,)

    def words(self, key: Key, index: tuple, n: int, cls_index: tuple) -> tuple[np.ndarray, int]:
        h = self.halves[key.uniform(len(self.halves), ("class", *cls_index))] if len(self.halves) > 1 else self.halves[0]
        return bf16_words(stream_u32(key, index, n), self.centre, h), h

    def describe(self) -> str:
        return f"bf16 window: exponent field {self.centre} +- h, h in {list(self.halves)} per draw; 1/32 edge words {BF16_EDGES.tolist()}"


def f32_uniform(key: Key, index: tuple, lo: float, hi: float) -> int:
    """The f32 word of lo + (hi - lo) * u / 2^32 for one 32-bit stream word u (rounded to f32)."""
    u = int(stream_u32(key, index, 1)[0])
    return int(np.asarray([lo + (hi - lo) * u / 2.0 ** 32], dtype=np.float32).view(np.uint32)[0])


def finite_bf16(words: np.ndarray) -> np.ndarray:
    """Per row: every BF16 word finite."""
    w = np.asarray(words, dtype=np.uint16)
    return ~(((w & 0x7F80) == 0x7F80).reshape(w.shape[0], -1).any(axis=1))


# --- statics as the Program records them ---------------------------------------------------------------------------------------

def encode_static(v: Any) -> Any:
    """A static as a vLLM Program's call records it (``verity.ml.circuits.codec``): a float is the JSON string ``{"f64":"<hex>"}``."""
    if isinstance(v, float):
        return json.dumps({"f64": v.hex()}, separators=(",", ":"))
    return v


def decode_static(v: Any) -> Any:
    if isinstance(v, str) and v.startswith("{"):
        d = json.loads(v)
        if isinstance(d, dict) and "f64" in d:
            return float.fromhex(d["f64"])
    return v


def row_set_name(family: str, statics: Mapping[str, Any]) -> str:
    """The exporter's name of a row-shaped set: family, its int statics, and a digest of the others (``vu_export._row_set_name``)."""
    ints = "-".join(f"{k.lower()}{v}" for k, v in sorted(statics.items()) if isinstance(v, int))
    other = {k: v for k, v in statics.items() if not isinstance(v, int)}
    tag = hashlib.sha256(json.dumps(other, sort_keys=True).encode()).hexdigest()[:8] if other else ""
    return "-".join(p for p in (family.lower().replace("_", "-"), ints, tag) if p)


def _slug_float(v: float) -> str:
    return repr(float(v)).replace("+", "").lower()


# --- subcircuits ----------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class Subcircuit:
    """A template with every parameter bound (``params``, census names).  ``target(**instance_statics)`` is the IR target the
    instances are evaluated with; ``definition(**instance_statics)`` the subcircuit's own Definition."""

    template: "Template"
    params: dict[str, Any]
    id: str
    display: str
    set_name: str
    relation: dict[str, Any]
    inputs: tuple[Port, ...]
    outputs: tuple[Port, ...]
    target: Callable[..., Any] = field(repr=False)
    definition: Callable[..., Any] = field(repr=False)
    sampler: Callable[["Subcircuit", Key, int], Draw] | None = field(default=None, repr=False)
    instance_statics_of: Callable[[Mapping[str, np.ndarray]], dict[str, int]] = field(default=lambda _p: {}, repr=False)
    output_words: Callable[[Port, Mapping[str, int]], int] | None = field(default=None, repr=False)
    admit: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None = field(default=None, repr=False)
    recipe: dict[str, Any] = field(default_factory=dict)
    #: (subcircuit, the input ports' stacked words in port order, instance statics) -> the outputs' flat words: for a Definition
    #: whose arguments are not the stored ports (a table or slab of which an input stores the one row it reads).
    evaluator: Callable[["Subcircuit", list[np.ndarray], dict[str, int]], np.ndarray] | None = field(default=None, repr=False)

    @property
    def ports(self) -> tuple[Port, ...]:
        return self.inputs + self.outputs

    def draw(self, key: Key, j: int) -> Draw:
        if self.sampler is None:
            raise NotEvaluable(f"{self.id}: no synthetic sampler")
        return self.sampler(self, key, j)

    def _out_words(self, p: Port, inst: Mapping[str, int]) -> int:
        if p.words is not None:
            return p.words
        if self.output_words is None:
            raise ValueError(f"{self.id}: output {p.name} has no fixed width")
        return self.output_words(p, inst)

    def evaluate(self, inputs: Mapping[str, Sequence[np.ndarray]], instance_statics: Sequence[Mapping[str, int]] | None = None
                 ) -> dict[str, list[np.ndarray]]:
        """Every instance's outputs through ``verity.ml.circuits.evaluation.evaluate_batch``: ``inputs[port][i]`` the words of instance i;
        instances are batched by their instance statics.  Returns ``{output port: [words of instance i]}``."""
        from verity.ml.circuits.evaluation import evaluate_batch

        n = len(next(iter(inputs.values()))) if inputs else 0
        if instance_statics is None:
            instance_statics = [self.instance_statics_of({p.name: inputs[p.name][i] for p in self.inputs}) for i in range(n)]
        groups: dict[tuple, list[int]] = {}
        for i, st in enumerate(instance_statics):
            groups.setdefault(tuple(sorted(st.items())), []).append(i)
        out: dict[str, list[np.ndarray | None]] = {p.name: [None] * n for p in self.outputs}
        for key, idx in groups.items():
            st = dict(key)
            arrays = [np.stack([np.asarray(inputs[p.name][i]).reshape(-1) for i in idx]) for p in self.inputs]
            res = self.evaluator(self, arrays, st) if self.evaluator else evaluate_batch(self.target(**st), *arrays)
            flat = np.asarray(res).reshape(len(idx), -1)
            off = 0
            for p in self.outputs:
                w = self._out_words(p, st)
                block = flat[:, off:off + w].astype(WORD_DTYPES[p.dtype])
                off += w
                for r, i in enumerate(idx):
                    out[p.name][i] = block[r]
            if off != flat.shape[1]:
                raise ValueError(f"{self.id}: the target returns {flat.shape[1]} leaves, the output ports hold {off}")
        return {k: list(v) for k, v in out.items()}  # type: ignore[arg-type]

    def admitted(self, outputs: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
        """Per instance: inside the domain a benchmark instance must be in (every BF16 output word finite, unless the template says)."""
        n = len(next(iter(outputs.values())))
        if self.admit is not None:
            return np.asarray(self.admit(outputs), dtype=bool)
        ok = np.ones(n, dtype=bool)
        for p in self.outputs:
            if p.dtype == "u16":
                ok &= np.array([bool(finite_bf16(np.asarray(w)[None, :])[0]) for w in outputs[p.name]], dtype=bool)
        return ok

    def program_digests(self, instance_statics: Sequence[Mapping[str, int]] = ({},)) -> dict[str, str]:
        """``{definition id: verity.ml.circuits.codec.program_digest}`` of the subcircuit's own Definition at each instance-statics binding.
        A ``ScalarReference`` (a silicon function no core Definition wraps yet) has no Program: its entry is
        :func:`reference_digest`."""
        from verity.ml.circuits.evaluation import ScalarReference
        from verity.ml.circuits.codec import program_digest
        from verity.ml.circuits.program import Program

        out = {}
        for st in {tuple(sorted(s.items())): s for s in instance_statics}.values():
            d = self.definition(**st)
            if isinstance(d, ScalarReference):
                out[f"{d.id}{json.dumps(d.bindings, sort_keys=True, separators=(',', ':'))}"] = reference_digest(d)
            else:
                out[d.id] = program_digest(Program(d))
        return out


#: The core sources a ``ScalarReference``'s function is defined by (its model, its total rules, the reference itself).
REFERENCE_SOURCES = ("verity/ml/kernels/tc.py", "verity/ml/circuits/silicon/models.py", "verity/ml/circuits/silicon/total_fp8.py",
                     "verity/ml/circuits/silicon/term.py")


def reference_digest(ref: Any) -> str:
    """sha256 of the reference's id, its bindings and the :data:`REFERENCE_SOURCES` (each a path under its import package), in
    that order."""
    import importlib
    from pathlib import Path

    h = hashlib.sha256(f"{ref.id}|{json.dumps(ref.bindings, sort_keys=True)}".encode())
    for rel in REFERENCE_SOURCES:
        package, _, path = rel.partition("/")
        root = Path(importlib.import_module(package).__file__).resolve().parent
        h.update(b"|" + rel.encode() + b"|" + (root / path).read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Template:
    """A subcircuit template: ``id`` (census-style kebab-case), typed ``parameters`` (name -> description), ``bind(**params)``,
    and ``identify(relation)`` for a relation a set records (``None`` when it is not this template's)."""

    id: str
    name: str
    description: str
    parameters: dict[str, str]
    bind: Callable[..., Subcircuit] = field(repr=False)
    identify: Callable[[Mapping[str, Any]], Subcircuit | None] = field(repr=False)
    example: dict[str, Any] = field(default_factory=dict)


# --- gemm-coordinate ------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _GemmSemantics:
    semantics: str
    slug: str
    display: str
    dots: tuple[str, ...]        # the DOT ids a recorded relation may name
    dot: str                     # the one a synthetic set's relation names (the exporter's)
    definition: str              # the row family the exporter names
    set_slug: str
    step: str                    # core step primitive attribute (verity.ml.circuits.ops.prims)


GEMM_SEMANTICS = {s.semantics: s for s in (
    _GemmSemantics("sm80.mma.m16n8k16.bf16", "sm80-mma-bf16", "sm80 mma · BF16", ("AmpereBF16TcDot16_v1", "AmpereBF16TcDot16_v2"),
                   "AmpereBF16TcDot16_v1", "Gemm_v1", "ampere-bf16", "AmpereBF16TcDot16"),
    _GemmSemantics("sm90.wgmma.m64n8k16.bf16", "sm90-wgmma-bf16", "sm90 wgmma · BF16", ("HopperBF16WgmmaDot16_v1",),
                   "HopperBF16WgmmaDot16_v1", "Gemm_v2", "hopper-bf16-wgmma", "HopperBF16WgmmaDot16"),
)}
@dataclass(frozen=True)
class _Fp8Semantics:
    semantics: str
    slug: str
    display: str
    reference: str               # the ScalarReference in verity.ml.kernels.tc: K/32 E4M3 steps from +0
    set_slug: str
    model: str                   # its verity.ml.circuits.silicon.models GroupSum (what backends chain with)


#: FP8 lines: the output is the FP32 accumulator word itself (no epilogue), as the backends' fp8 relations take it.
GEMM_FP8 = {s.semantics: s for s in (
    _Fp8Semantics("sm89.mma.m16n8k32.e4m3", "sm89-mma-e4m3", "sm89 mma · FP8 (E4M3)", "ADA_E4M3_DOT", "ada-e4m3", "ADA_E4M3_M16N8K32"),
    _Fp8Semantics("sm90.wgmma.m64n8k32.e4m3", "sm90-wgmma-e4m3", "sm90 wgmma · FP8 (E4M3)", "HOPPER_E4M3_DOT", "hopper-e4m3-wgmma",
                  "HOPPER_E4M3_K32"),
)}
FP8_DEFINITION = "GemmAccumulatorE4m3"
FP8_RULE = "y = the FP32 accumulator of K/32 ascending DOT steps over (x, w) from +0 (no epilogue)"
#: Census GEMM semantics with a ``verity.ml.circuits.silicon`` model but no IR target here: a subcircuit binds, it cannot be generated.
GEMM_MODEL_ONLY = {
    "sm90.mma.m16n8k16.bf16": ("sm90-mma-bf16", "sm90 mma · BF16", "bf16"),
    "sm120.mma.m16n8k32.e4m3": ("sm120-mma-e4m3", "sm120 mma · FP8 (E4M3)", "e4m3"),
}
#: NVFP4 on sm_120 (RTX 5090): E2M1 codes with UE4M3 scales per block of 16; y the FP32 accumulator of
#: ``verity.ml.kernels.tc.SM120_NVF4_DOT`` (``models.BLACKWELL_SM120_NVF4``, K/64 steps from +0), no epilogue.
NVFP4_SEMANTICS = "sm120.mma.m16n8k64.e2m1.nvf4"
NVFP4_DEFINITION = "GemmAccumulatorNvfp4"
NVFP4_BLOCK = 16
NVFP4_RECIPE = {"codes": "E2M1 codes uniform over all 16: r & 0xF for a 32-bit stream word r",
                "scales": "UE4M3 scale bytes uniform over the 127 valid codes 0x00..0x7E: r mod 127"}
GEMM_RULE = "y = f2fp_bf16(ascending k16 chain of DOT over (x, w) from +0)"
GEMM_X, GEMM_W = Bf16Window(127, (4, 16, 40)), Bf16Window(127, (4, 16, 40))


def _gemm_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    K, tile = sub.params["K"], int(sub.recipe.get("tile", 32))
    t = j // tile
    x, hx = GEMM_X.words(key, ("x", t), K, ("x", t))
    w, hw = GEMM_W.words(key, ("w", j), K, ("w", j))
    return Draw({"x": x, "w": w}, {}, {"tile": t, "hx": hx, "hw": hw})


def e4m3_finite_words(r: np.ndarray) -> np.ndarray:
    """E4M3 bytes uniform over the 254 finite codes from 32 random bits each: ``v = r mod 254``, the byte ``v + (v >= 127)``
    (skipping the NaN codes 0x7F and 0xFF)."""
    v = np.asarray(r, dtype=np.uint32) % np.uint32(254)
    return (v + (v >= 127)).astype(np.uint8)


E4M3_RECIPE = "E4M3 bytes uniform over the 254 finite codes: v = r mod 254 for a 32-bit stream word r, byte v + (v >= 127)"


def _fp8_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    K, tile = sub.params["K"], int(sub.recipe.get("tile", 32))
    t = j // tile
    return Draw({"x": e4m3_finite_words(stream_u32(key, ("x", t), K)), "w": e4m3_finite_words(stream_u32(key, ("w", j), K))},
                {}, {"tile": t})


def _finite_f32(outs: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
    return np.array([(int(np.asarray(y).reshape(-1)[0]) & 0x7F800000) != 0x7F800000 for y in outs["y"]], dtype=bool)


def _fp8_coordinate(K: int, s: _Fp8Semantics, tile: int) -> Subcircuit:
    if K % 32:
        raise ValueError(f"gemm-coordinate {s.semantics}: K = {K} is not a multiple of 32")

    def target(**_st: int) -> Any:
        from verity.ml.kernels import tc as kernels

        return getattr(kernels, s.reference).bind(K=K)

    ref_id = target().id
    rel = {"subcircuit": f"GemmCoordinate<{K}>", "definition": FP8_DEFINITION, "dot": ref_id, "model": s.model, "statics": {"K": K},
           "rule": FP8_RULE}
    return Subcircuit(TEMPLATES["gemm-coordinate"], {"K": K, "datatype": "e4m3", "semantics": s.semantics}, f"gemm-coordinate/k{K}/{s.slug}",
                      f"GemmCoordinate<{K}> · {s.display}", f"gemm-coordinate-{s.set_slug}-k{K}", rel,
                      (Port("x", "u8", K), Port("w", "u8", K)), (Port("y", "u32", 1),), target, target, sampler=_fp8_sample,
                      admit=_finite_f32, recipe={"tile": int(tile), "x": E4M3_RECIPE + "; one row per tile of `tile` draws", "w": E4M3_RECIPE})


def _nvfp4_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    K, tile = sub.params["K"], int(sub.recipe.get("tile", 32))
    t, nb = j // tile, sub.params["K"] // NVFP4_BLOCK
    codes = lambda idx: (stream_u32(key, idx, K) & 0xF).astype(np.uint8)  # noqa: E731
    scales = lambda idx: (stream_u32(key, idx, nb) % np.uint32(127)).astype(np.uint8)  # noqa: E731
    return Draw({"x": codes(("x", t)), "w": codes(("w", j)), "sx": scales(("sx", t)), "sw": scales(("sw", j))}, {}, {"tile": t})


def _nvfp4_coordinate(K: int, tile: int) -> Subcircuit:
    if K % 64:
        raise ValueError(f"gemm-coordinate {NVFP4_SEMANTICS}: K = {K} is not a multiple of 64")

    def target(**_st: int) -> Any:
        from verity.ml.kernels import tc as kernels

        return kernels.SM120_NVF4_DOT.bind(K=K)

    rel = {"subcircuit": f"GemmCoordinate<{K}>", "definition": NVFP4_DEFINITION, "dot": target().id, "model": "BLACKWELL_SM120_NVF4",
           "statics": {"K": K, "block": NVFP4_BLOCK},
           "rule": "y = the FP32 accumulator of K/64 ascending DOT steps over (x, w) with UE4M3 block scales (sx, sw) from +0 (no epilogue)"}
    return Subcircuit(TEMPLATES["gemm-coordinate"], {"K": K, "datatype": "e2m1", "semantics": NVFP4_SEMANTICS},
                      f"gemm-coordinate/k{K}/sm120-mma-e2m1-nvf4", f"GemmCoordinate<{K}> · sm120 mma · FP4 (NVFP4, E2M1)",
                      f"gemm-coordinate-sm120-nvf4-k{K}", rel,
                      (Port("x", "u8", K), Port("w", "u8", K), Port("sx", "u8", K // NVFP4_BLOCK), Port("sw", "u8", K // NVFP4_BLOCK)),
                      (Port("y", "u32", 1),), target, target, sampler=_nvfp4_sample, admit=_finite_f32,
                      recipe={"tile": int(tile), "x, sx": "one row and its scales per tile of `tile` draws", **NVFP4_RECIPE})


def gemm_coordinate(K: int, semantics: str = "sm80.mma.m16n8k16.bf16", datatype: str | None = None, tile: int = 32) -> Subcircuit:
    K = int(K)
    if K <= 0 or K % 16:
        raise ValueError(f"gemm-coordinate: K = {K} is not a positive multiple of 16")
    ports_in, ports_out = (Port("x", "u16", K), Port("w", "u16", K)), (Port("y", "u16", 1),)
    if semantics in GEMM_FP8:
        if datatype not in (None, "e4m3"):
            raise ValueError(f"gemm-coordinate {semantics}: datatype is e4m3, not {datatype!r}")
        return _fp8_coordinate(K, GEMM_FP8[semantics], tile)
    if semantics == NVFP4_SEMANTICS:
        if datatype not in (None, "e2m1"):
            raise ValueError(f"gemm-coordinate {semantics}: datatype is e2m1, not {datatype!r}")
        return _nvfp4_coordinate(K, tile)
    if semantics in GEMM_MODEL_ONLY:
        slug, disp, dt = GEMM_MODEL_ONLY[semantics]

        def none(**_st: int) -> Any:
            raise NotEvaluable(f"gemm-coordinate {semantics}: a verity.ml.circuits.silicon model, no IR Definition to evaluate or generate with")

        return Subcircuit(TEMPLATES["gemm-coordinate"], {"K": K, "datatype": dt, "semantics": semantics}, f"gemm-coordinate/k{K}/{slug}",
                          f"GemmCoordinate<{K}> · {disp}", f"gemm-coordinate-{slug}-k{K}", {}, ports_in, ports_out, none, none)
    if semantics not in GEMM_SEMANTICS:
        raise ValueError(f"gemm-coordinate: unknown semantics {semantics!r} "
                         f"(one of {sorted(GEMM_SEMANTICS) + sorted(GEMM_FP8) + [NVFP4_SEMANTICS] + sorted(GEMM_MODEL_ONLY)})")
    s = GEMM_SEMANTICS[semantics]
    if datatype not in (None, "bf16"):
        raise ValueError(f"gemm-coordinate {semantics}: datatype is bf16, not {datatype!r}")

    def target(**_st: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.circuits.ops import gemm, prims

        return bind(gemm.GemmCoordinate, K=K, DOT=getattr(prims, s.step))

    rel = {"subcircuit": f"GemmCoordinate<{K}>", "definition": s.definition, "dot": s.dot, "statics": {"K": K}, "rule": GEMM_RULE}
    return Subcircuit(TEMPLATES["gemm-coordinate"], {"K": K, "datatype": "bf16", "semantics": semantics}, f"gemm-coordinate/k{K}/{s.slug}",
                      f"GemmCoordinate<{K}> · {s.display}", f"gemm-coordinate-{s.set_slug}-k{K}", rel, ports_in, ports_out, target, target,
                      sampler=_gemm_sample,
                      recipe={"tile": int(tile), "x": GEMM_X.describe() + "; one row per tile of `tile` draws", "w": GEMM_W.describe()})


def _gemm_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    if rel.get("definition") == FP8_DEFINITION:
        K = int(rel["statics"]["K"])
        for s in GEMM_FP8.values():
            sub = _fp8_coordinate(K, s, 32)
            if rel.get("dot") == sub.relation["dot"]:
                return sub
        raise KeyError(f"gemm-coordinate: no FP8 semantics for DOT {rel.get('dot')!r}")
    if rel.get("definition") == NVFP4_DEFINITION:
        return _nvfp4_coordinate(int(rel["statics"]["K"]), 32)
    if rel.get("definition") not in ("Gemm_v1", "Gemm_v2"):
        return None
    for s in GEMM_SEMANTICS.values():
        if rel.get("dot") in s.dots:
            sub = gemm_coordinate(int(rel["statics"]["K"]), s.semantics)
            return sub if rel.get("dot") == s.dot else _with_relation(sub, rel)
    raise KeyError(f"gemm-coordinate: no semantics for DOT {rel.get('dot')!r}")


def _with_relation(sub: Subcircuit, rel: Mapping[str, Any]) -> Subcircuit:
    import dataclasses

    return dataclasses.replace(sub, relation=dict(rel))


# --- attention-head (FA2) -------------------------------------------------------------------------------------------------------

ATTN_RULE = "one query at position T-1 over keys 0..T-1 of its KV head, ceil(T/BN) key blocks last -> first (the Definition's head)"
ATTN_QKV = Bf16Window(127, (2, 4))


def _parse_range(v: Any) -> tuple[int, int]:
    if isinstance(v, int):
        return v, v
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return int(v[0]), int(v[1])
    lo, _, hi = str(v).partition(":")
    return int(lo), int(hi or lo)


def _attn_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    D = sub.params["D"]
    lo, hi = _parse_range(sub.recipe["T"])
    T = lo + key.uniform(hi - lo + 1, ("T", j))
    q, h = ATTN_QKV.words(key, ("q", j), D, ("qkv", j))
    k = bf16_words(stream_u32(key, ("k", j), T * D), ATTN_QKV.centre, h)
    v = bf16_words(stream_u32(key, ("v", j), T * D), ATTN_QKV.centre, h)
    return Draw({"q": q, "k": k, "v": v}, {"T": int(T)}, {"T": int(T), "h": h})


#: The attention heads by semantics: (the row family the exporter names, DOT id, INV id, set-name stem, id tail, display tail).
ATTN_SEMANTICS = {
    "sm80.fa2.bf16": ("Attention_v3", "AmpereBF16TcDot16_v1", "Fa2InvSum_v1", "attention-head-fa2", "sm80-fa2-bf16", "FA2", "sm80 mma"),
    "sm90.fa3.bf16": ("Attention_v2", "HopperBF16WgmmaDot16_v1", "Fa3InvSum_v1", "attention-head-hopper-bf16-wgmma-fa3invsum_v1",
                      "sm90-fa3-bf16", "FA3", "sm90 wgmma"),
}


def attention_head(D: int, BN: int, T: Any = "1:512", semantics: str = "sm80.fa2.bf16") -> Subcircuit:
    """One attention head, ``T`` per instance (the key count, from ``len(k) / D``); ``T`` here is the synthetic draw's range
    ``lo:hi`` (uniform per draw).  ``sm80.fa2.bf16``: FA2 (``AttentionHead_v3``, evaluated as ``Attention_v3{NH=1, KVH=1}``);
    ``sm90.fa3.bf16``: FA3 on the H100's wgmma step (``AttentionHead_v2{DOT=HopperBF16WgmmaDot16, INV=Fa3InvSum}``, evaluated as
    ``Attention_v2{NH=1, KVH=1}``)."""
    D, BN = int(D), int(BN)
    if semantics not in ATTN_SEMANTICS:
        raise ValueError(f"attention-head: semantics {semantics!r} (one of {sorted(ATTN_SEMANTICS)})")
    family, dot, inv, stem, tail, kind, step = ATTN_SEMANTICS[semantics]
    lo, hi = _parse_range(T)
    if not 1 <= lo <= hi:
        raise ValueError(f"attention-head: T range {T!r}")

    def _v2() -> dict[str, Any]:
        from verity.ml.definitions.registry import hopper

        return {"DOT": hopper.HopperBF16WgmmaDot16, "INV": hopper.Fa3InvSum}

    def target(T: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import b1

        if family == "Attention_v3":
            return bind(b1.AttentionV3, T=int(T), NH=1, KVH=1, D=D, BN=BN)
        return bind(b1.AttentionV2, T=int(T), NH=1, KVH=1, D=D, BN=BN, **_v2())

    def definition(T: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import b1

        if family == "Attention_v3":
            return bind(b1.AttentionHeadV3, T=int(T), D=D, BN=BN)
        return bind(b1.AttentionHeadV2, T=int(T), D=D, BN=BN, **_v2())

    rel = {"subcircuit": f"AttentionHead<T,{D}>", "definition": family, "dot": dot, "inv": inv, "statics": {"D": D, "BN": BN},
           "rule": ATTN_RULE}
    return Subcircuit(TEMPLATES["attention-head"], {"D": D, "BN": BN, "semantics": semantics}, f"attention-head/d{D}-bn{BN}/{tail}",
                      f"AttentionHead<T,{D}> · {kind} BN={BN} · {step} · BF16", f"{stem}-d{D}-bn{BN}", rel,
                      (Port("q", "u16", D), Port("k", "u16", None), Port("v", "u16", None)), (Port("out", "u16", D),),
                      target, definition, sampler=_attn_sample,
                      instance_statics_of=lambda p: {"T": int(np.asarray(p["k"]).size) // D},
                      recipe={"T": f"{lo}:{hi}", "T_rule": "T = lo + key.uniform(hi - lo + 1, ('T', j))", "qkv": ATTN_QKV.describe()
                              + "; one h per draw for q, k and v"})


def _attn_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    for sem, (family, dot, inv, *_rest) in ATTN_SEMANTICS.items():
        if rel.get("definition") == family and (family == "Attention_v3" or (rel.get("dot"), rel.get("inv")) == (dot, inv)):
            return attention_head(int(rel["statics"]["D"]), int(rel["statics"]["BN"]), semantics=sem)
    return None


# --- attention-head-v18 (the registered compiler's FA2 head) -------------------------------------------------------------------

ATTN_V18_SEMANTICS = "sm120.fa2.bf16.decode"
ATTN_V18_PRIMS = ("HopperBF16WgmmaDot16_v1", "Fa2InvSum_v1")
ATTN_V18_MASKED = "MASKED_FROM = ceil(T/BN) - 1, the guard of FA2's decode-only launch (one query row, non-causal)"


def attention_head_v18(D: int, BN: int, T: Any = "1:512", semantics: str = ATTN_V18_SEMANTICS) -> Subcircuit:
    """``dead_free.AttentionHead_v18{T, D, BN, DOT, INV, MASKED_FROM}`` on sm120's FA2 steps (``HopperBF16WgmmaDot16``,
    ``Fa2InvSum``), evaluated as ``Attention_v18{NH=1, KVH=1}``.  Its query row is a decode-only launch's, so MASKED_FROM is a
    function of T (:data:`ATTN_V18_MASKED`, ``attention_vocab.fa2_masked_from``); ``T`` is per instance as in ``attention-head``."""
    D, BN = int(D), int(BN)
    if semantics != ATTN_V18_SEMANTICS:
        raise ValueError(f"attention-head-v18: semantics {semantics!r} (only {ATTN_V18_SEMANTICS!r})")
    lo, hi = _parse_range(T)
    if not 1 <= lo <= hi:
        raise ValueError(f"attention-head-v18: T range {T!r}")

    def statics(T: int) -> dict[str, Any]:
        from verity.ml.definitions.registry import b1, hopper
        from verity.ml.definitions.registry.attention_vocab import fa2_masked_from

        return {"T": int(T), "D": D, "BN": BN, "DOT": hopper.HopperBF16WgmmaDot16, "INV": b1.P.Fa2InvSum,
                "MASKED_FROM": fa2_masked_from(int(T), 1, 0, BN, False)}

    def target(T: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import dead_free

        import verity.ml.definitions.numerics.rows  # noqa: F401  (registers Attention_v18's self-checked row kernel)
        return bind(dead_free.AttentionV18, NH=1, KVH=1, **statics(T))

    def definition(T: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import dead_free

        return bind(dead_free.AttentionHeadV18, **statics(T))

    rel = {"subcircuit": f"AttentionHead<T,{D}>", "definition": "Attention_v18", "dot": ATTN_V18_PRIMS[0], "inv": ATTN_V18_PRIMS[1],
           "masked_from": "decode", "statics": {"D": D, "BN": BN}, "rule": f"{ATTN_RULE}; {ATTN_V18_MASKED}"}
    return Subcircuit(TEMPLATES["attention-head-v18"], {"D": D, "BN": BN, "semantics": semantics},
                      f"attention-head-v18/d{D}-bn{BN}/sm120-fa2-bf16-decode", f"AttentionHead_v18<T,{D}> · FA2 BN={BN} · decode · sm120 mma · BF16",
                      f"attention-head-v18-decode-d{D}-bn{BN}", rel,
                      (Port("q", "u16", D), Port("k", "u16", None), Port("v", "u16", None)), (Port("out", "u16", D),),
                      target, definition, sampler=_attn_sample,
                      instance_statics_of=lambda p: {"T": int(np.asarray(p["k"]).size) // D},
                      recipe={"T": f"{lo}:{hi}", "T_rule": "T = lo + key.uniform(hi - lo + 1, ('T', j))", "MASKED_FROM": ATTN_V18_MASKED,
                              "qkv": ATTN_QKV.describe() + "; one h per draw for q, k and v"})


def _attn_v18_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    if rel.get("definition") != "Attention_v18" or rel.get("masked_from") != "decode" or (rel.get("dot"), rel.get("inv")) != ATTN_V18_PRIMS:
        return None
    return attention_head_v18(int(rel["statics"]["D"]), int(rel["statics"]["BN"]))


# --- rope-head ------------------------------------------------------------------------------------------------------------------

ROPE_RULE = ("out[i] = bf16_rn(RN32(x[i]*c[i] - x[i+R]*s[i])), out[i+R] = bf16_rn(RN32(x[i+R]*c[i] + x[i]*s[i])), cs = [c | s], "
             "R = D/2")
ROPE_X, ROPE_CS = Bf16Window(127, (4, 16)), Bf16Window(120, (6,))


def _rope_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    D = sub.params["D"]
    x, h = ROPE_X.words(key, ("x", j), D, ("x", j))
    cs, _ = ROPE_CS.words(key, ("cs", j), D, ("cs", j))
    return Draw({"x": x, "cs": cs}, {}, {"h": h})


def rope_head(D: int) -> Subcircuit:
    D = int(D)
    if D <= 0 or D % 2:
        raise ValueError(f"rope-head: D = {D} is not a positive even number")

    def target(**_st: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import b1

        return bind(b1.RoPE, NHEADS=1, D=D)

    def definition(**_st: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import b1

        return bind(b1.RoPEHead, D=D)

    rel = {"subcircuit": f"RoPEHead<{D}>", "definition": "RoPE_v1", "statics": {"D": D}, "rule": ROPE_RULE}
    return Subcircuit(TEMPLATES["rope-head"], {"D": D}, f"rope-head/d{D}/neox-bf16", f"RoPEHead<{D}> · NeoX · BF16", f"rope-head-d{D}", rel,
                      (Port("x", "u16", D), Port("cs", "u16", D)), (Port("out", "u16", D),), target, definition, sampler=_rope_sample,
                      recipe={"x": ROPE_X.describe(), "cs": ROPE_CS.describe() + " (|c|, |s| < 1)"})


def _rope_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    return rope_head(int(rel["statics"]["D"])) if rel.get("definition") == "RoPE_v1" else None


# --- row-shaped templates (the exporter's generic rows) ------------------------------------------------------------------------

@dataclass(frozen=True)
class _Row:
    template: str
    family: str
    registry: str                                  # module:attribute of the composite
    statics: tuple[str, ...]
    inputs: Callable[[Mapping[str, Any]], tuple[Port, ...]]
    outputs: Callable[[Mapping[str, Any]], tuple[Port, ...]]
    sample: Callable[[Mapping[str, Any], Key, int], tuple[dict[str, np.ndarray], dict[str, Any]]]
    slug: Callable[[Mapping[str, Any]], str]
    display: Callable[[Mapping[str, Any]], str]
    recipe: dict[str, str]
    admit: Callable[[Mapping[str, np.ndarray]], np.ndarray] | None = None
    pattern: str = ""                              # the id's parameter segment as a regex, one named group per static
    example: dict[str, Any] = field(default_factory=dict)


def _composite(path: str) -> Any:
    import importlib

    mod, _, attr = path.partition(":")
    return getattr(importlib.import_module(mod), attr)


#: A row's relation names its rule by the row kernels' module as it was when #101 captured them, in two literals so that a layout
#: move leaves the text: the captured manifests record it, and their content digests cover it.
ROW_RULE = "the registered row kernel of {} (verity_vllm.program" ".kernels.rows)"


def _row_subcircuit(row: _Row, **params: Any) -> Subcircuit:
    missing = [k for k in row.statics if k not in params]
    extra = sorted(set(params) - set(row.statics))
    if missing or extra:
        raise TypeError(f"{row.template}: parameters {list(row.statics)} (missing {missing}, unknown {extra})")
    st = {k: (float(params[k]) if isinstance(params[k], float) or k == "EPS" else int(params[k])) for k in row.statics}

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind

        import verity.ml.definitions.numerics.rows  # noqa: F401  (registers the self-checked row kernels with verity.ml.circuits.evaluation)
        return bind(_composite(row.registry), **st)

    enc = {k: encode_static(v) for k, v in st.items()}
    rel = {"subcircuit": row.family, "definition": row.family, "statics": enc,
           "rule": ROW_RULE.format(row.family)}

    def sampler(sub: Subcircuit, key: Key, j: int) -> Draw:
        ins, info = row.sample(st, key, j)
        return Draw(ins, {}, info)

    return Subcircuit(TEMPLATES[row.template], dict(st), f"{row.template}/{row.slug(st)}", row.display(st), row_set_name(row.family, enc), rel,
                      row.inputs(st), row.outputs(st), target, target, sampler=sampler, admit=row.admit, recipe=dict(row.recipe))


def _row_identify(row: _Row) -> Callable[[Mapping[str, Any]], Subcircuit | None]:
    def ident(rel: Mapping[str, Any]) -> Subcircuit | None:
        if rel.get("definition") != row.family:
            return None
        return _row_subcircuit(row, **{k: decode_static(v) for k, v in rel["statics"].items()})
    return ident


RMS_X, RMS_W = Bf16Window(127, (4, 12)), Bf16Window(127, (2,))
SILU = Bf16Window(127, (4, 8))
LOGITS = Bf16Window(127, (2, 4))


def _rms_fused_sample(st: Mapping[str, Any], key: Key, j: int):
    N = st["N"]
    x, h = RMS_X.words(key, ("in0", j), N, ("in", j))
    res = bf16_words(stream_u32(key, ("in1", j), N), RMS_X.centre, h)
    w, _ = RMS_W.words(key, ("in2", j), N, ("in2", j))
    return {"in0": x, "in1": res, "in2": w}, {"h": h}


def _rms_triton_sample(st: Mapping[str, Any], key: Key, j: int):
    N = st["N"]
    x, h = RMS_X.words(key, ("in0", j), N, ("in", j))
    w, _ = RMS_W.words(key, ("in1", j), N, ("in1", j))
    return {"in0": x, "in1": w}, {"h": h}


def _silu_sample(st: Mapping[str, Any], key: Key, j: int):
    gu, h = SILU.words(key, ("in0", j), 2 * st["I"], ("in0", j))
    return {"in0": gu}, {"h": h}


def _gumbel_sample(st: Mapping[str, Any], key: Key, j: int):
    logits, h = LOGITS.words(key, ("in0", j), st["V"], ("in0", j))
    ins = {"in0": logits,
           "in1": np.asarray([f32_uniform(key, ("in1", j), 0.05, 1.0)], dtype=np.uint32),
           "in2": stream_u64(key, ("in2", j), 1),
           "in3": np.asarray([key.uniform(1 << 20, ("in3", j))], dtype=np.uint64),
           "in4": np.asarray([f32_uniform(key, ("in4", j), 0.3, 2.0)], dtype=np.uint32),
           "in5": np.asarray([1 << key.uniform(4, ("in5", j))], dtype=np.uint32)}
    return ins, {"h": h}


def _u16(n: int, *names: str) -> tuple[Port, ...]:
    return tuple(Port(nm, "u16", n) for nm in names)


TOKEN_LOGITS = Bf16Window(127, (8, 16))
ROUTER_LOGITS = Bf16Window(127, (2, 4))
MOE_Y = Bf16Window(127, (4, 8))
BIAS_Y, BIAS_B = Bf16Window(127, (4, 12)), Bf16Window(127, (2, 4))
SCALAR_X, SCALAR_S = Bf16Window(127, (4, 12)), Bf16Window(127, (2,))


def _one(window: Bf16Window, port: str, n: Callable[[Mapping[str, Any]], int]):
    def sample(st: Mapping[str, Any], key: Key, j: int):
        w, h = window.words(key, (port, j), n(st), (port, j))
        return {port: w}, {"h": h}
    return sample


def _two(first: Bf16Window, second: Bf16Window, n0: Callable[[Mapping[str, Any]], int], n1: Callable[[Mapping[str, Any]], int]):
    def sample(st: Mapping[str, Any], key: Key, j: int):
        a, h = first.words(key, ("in0", j), n0(st), ("in0", j))
        b, _ = second.words(key, ("in1", j), n1(st), ("in1", j))
        return {"in0": a, "in1": b}, {"h": h}
    return sample


def _all(outs: Mapping[str, Sequence[np.ndarray]]) -> np.ndarray:
    return np.ones(len(next(iter(outs.values()))), dtype=bool)


def _router(template: str, family: str, attr: str, what: str) -> _Row:
    return _Row(template, family, f"verity.ml.definitions.registry.moe:{attr}", ("E", "TOPK", "VPT"),
                lambda s: (Port("in0", "u16", s["E"]),), lambda s: (Port("out.router_w", "u32", s["TOPK"]), Port("out.router_ids", "u32", s["TOPK"])),
                _one(ROUTER_LOGITS, "in0", lambda s: s["E"]), lambda s: f"e{s['E']}-topk{s['TOPK']}-vpt{s['VPT']}/bf16",
                lambda s: f"{family}<E={s['E']}, TOPK={s['TOPK']}> · {what} · BF16 logits", {"in0 (logits)": ROUTER_LOGITS.describe()},
                pattern=r"e(?P<E>\d+)-topk(?P<TOPK>\d+)-vpt(?P<VPT>\d+)", example={"E": 64, "TOPK": 8, "VPT": 8})


ROWS: dict[str, _Row] = {r.template: r for r in (
    _Row("rmsnorm-fused-cuda", "RMSNormFusedCuda_v2", "verity.ml.definitions.registry.b1:RMSNormFusedCuda", ("N", "EPS"),
         lambda s: _u16(s["N"], "in0", "in1", "in2"), lambda s: _u16(s["N"], "out.0", "out.1"), _rms_fused_sample,
         lambda s: f"n{s['N']}-eps{_slug_float(s['EPS'])}/bf16", lambda s: f"RMSNormFusedCuda_v2<{s['N']}> · eps {s['EPS']:g} · BF16",
         {"in0/in1 (x, res)": RMS_X.describe() + "; one h per draw for both", "in2 (w)": RMS_W.describe()},
         pattern=r"n(?P<N>\d+)-eps(?P<EPS>.+)", example={"N": 2048, "EPS": 1e-5}),
    _Row("rmsnorm-triton", "RMSNormTriton_v1", "verity.ml.definitions.registry.b1:RMSNormTriton", ("N", "EPS"),
         lambda s: _u16(s["N"], "in0", "in1"), lambda s: _u16(s["N"], "out.out"), _rms_triton_sample,
         lambda s: f"n{s['N']}-eps{_slug_float(s['EPS'])}/bf16", lambda s: f"RMSNormTriton_v1<{s['N']}> · eps {s['EPS']:g} · BF16",
         {"in0 (x)": RMS_X.describe(), "in1 (w)": RMS_W.describe()}, pattern=r"n(?P<N>\d+)-eps(?P<EPS>.+)", example={"N": 2048, "EPS": 1e-5}),
    _Row("rmsnorm-triton-v5", "RMSNormTriton_v5", "verity.ml.definitions.registry.dead_free:RMSNormTritonV5", ("N", "EPS"),
         lambda s: _u16(s["N"], "in0", "in1"), lambda s: _u16(s["N"], "out.out"), _rms_triton_sample,
         lambda s: f"n{s['N']}-eps{_slug_float(s['EPS'])}/bf16", lambda s: f"RMSNormTriton_v5<{s['N']}> · eps {s['EPS']:g} · BF16",
         {"in0 (x)": RMS_X.describe(), "in1 (w)": RMS_W.describe()}, pattern=r"n(?P<N>\d+)-eps(?P<EPS>.+)", example={"N": 2048, "EPS": 1e-5}),
    _Row("silu-mul", "SiluMul_v1", "verity.ml.definitions.registry.b1:SiluMul", ("I",),
         lambda s: _u16(2 * s["I"], "in0"), lambda s: _u16(s["I"], "out.out"), _silu_sample,
         lambda s: f"i{s['I']}/bf16", lambda s: f"SiluMul_v1<{s['I']}> · BF16", {"in0 (gate | up)": SILU.describe()},
         pattern=r"i(?P<I>\d+)", example={"I": 8192}),
    _Row("silu-mul-v2", "SiluMul_v2", "verity.ml.definitions.registry.act:SiluMul", ("I",),
         lambda s: _u16(2 * s["I"], "in0"), lambda s: _u16(s["I"], "out.out"), _silu_sample,
         lambda s: f"i{s['I']}/bf16", lambda s: f"SiluMul_v2<{s['I']}> · BF16", {"in0 (gate | up)": SILU.describe()},
         pattern=r"i(?P<I>\d+)", example={"I": 256}),
    _Row("rmsnorm-fused-cuda-vec8", "RMSNormFusedCudaVector8_v1", "verity.ml.definitions.registry.cuda_rmsnorm:RMSNormFusedCudaVector8",
         ("N", "EPS", "BLOCK"), lambda s: _u16(s["N"], "in0", "in1", "in2"), lambda s: _u16(s["N"], "out.0", "out.1"), _rms_fused_sample,
         lambda s: f"n{s['N']}-b{s['BLOCK']}-eps{_slug_float(s['EPS'])}/bf16",
         lambda s: f"RMSNormFusedCudaVector8_v1<{s['N']}> · BLOCK {s['BLOCK']} · eps {s['EPS']:g} · BF16",
         {"in0/in1 (x, res)": RMS_X.describe() + "; one h per draw for both", "in2 (w)": RMS_W.describe()},
         pattern=r"n(?P<N>\d+)-b(?P<BLOCK>\d+)-eps(?P<EPS>.+)", example={"N": 6144, "EPS": 1e-5, "BLOCK": 1024}),
    _Row("rmsnorm-cuda-vec8", "RMSNormCudaVector8_v1", "verity.ml.definitions.registry.cuda_rmsnorm:RMSNormCudaVector8",
         ("N", "EPS", "BLOCK"), lambda s: _u16(s["N"], "in0", "in1"), lambda s: _u16(s["N"], "out.out"), _rms_triton_sample,
         lambda s: f"n{s['N']}-b{s['BLOCK']}-eps{_slug_float(s['EPS'])}/bf16",
         lambda s: f"RMSNormCudaVector8_v1<{s['N']}> · BLOCK {s['BLOCK']} · eps {s['EPS']:g} · BF16",
         {"in0 (x)": RMS_X.describe(), "in1 (w)": RMS_W.describe()},
         pattern=r"n(?P<N>\d+)-b(?P<BLOCK>\d+)-eps(?P<EPS>.+)", example={"N": 6144, "EPS": 1e-5, "BLOCK": 768}),
    _Row("gumbel-top-p-token-select", "GumbelTopPTokenSelect_v1", "verity.ml.definitions.registry.sampling:GumbelTopPTokenSelect", ("V",),
         lambda s: (Port("in0", "u16", s["V"]), Port("in1", "u32", 1), Port("in2", "u64", 1), Port("in3", "u64", 1),
                    Port("in4", "u32", 1), Port("in5", "u32", 1)),
         lambda s: (Port("out.sampled_token_ids", "u64", 1),), _gumbel_sample,
         lambda s: f"v{s['V']}/fp32", lambda s: f"GumbelTopPTokenSelect_v1<{s['V']}> · temperature + top-p · FP32",
         {"in0 (logits)": LOGITS.describe(), "in1 (top_p)": "f32 of 0.05 + 0.95 u / 2^32", "in2 (seed)": "u64 stream word",
          "in3 (pos)": "key.uniform(2^20)", "in4 (temp)": "f32 of 0.3 + 1.7 u / 2^32", "in5 (splits)": "1 << key.uniform(4)"},
         admit=_all, pattern=r"v(?P<V>\d+)", example={"V": 128256}),
    _Row("token-select-greedy", "TokenSelect_v1", "verity.ml.definitions.registry.b1:TokenSelect", ("V",),
         lambda s: (Port("in0", "u16", s["V"]),), lambda s: (Port("out.sampled_token_ids", "u64", 1),),
         _one(TOKEN_LOGITS, "in0", lambda s: s["V"]), lambda s: f"v{s['V']}/bf16", lambda s: f"TokenSelect_v1<{s['V']}> · greedy argmax · BF16",
         {"in0 (logits)": TOKEN_LOGITS.describe()}, admit=_all, pattern=r"v(?P<V>\d+)", example={"V": 151936}),
    _router("moe-router-topk", "MoeRouterTopK_v1", "MoeRouterTopK", "softmax top-k"),
    _router("moe-router-topk-norm", "MoeRouterTopKNorm_v1", "MoeRouterTopKNorm", "softmax top-k, renormalised"),
    _Row("moe-sum", "MoeSum_v1", "verity.ml.definitions.registry.moe:MoeSum", ("TOPK", "H"),
         lambda s: (Port("in0", "u16", s["TOPK"] * s["H"]),), lambda s: (Port("out.sum", "u16", s["H"]),),
         _one(MOE_Y, "in0", lambda s: s["TOPK"] * s["H"]), lambda s: f"topk{s['TOPK']}-h{s['H']}/bf16",
         lambda s: f"MoeSum_v1<TOPK={s['TOPK']}, H={s['H']}> · BF16", {"in0 (the TOPK expert rows)": MOE_Y.describe()},
         pattern=r"topk(?P<TOPK>\d+)-h(?P<H>\d+)", example={"TOPK": 8, "H": 2048}),
    _Row("bias-add", "BiasAdd_v1", "verity.ml.definitions.registry.b1:BiasAdd", ("N",),
         lambda s: _u16(s["N"], "in0", "in1"), lambda s: _u16(s["N"], "out.0"), _two(BIAS_Y, BIAS_B, lambda s: s["N"], lambda s: s["N"]),
         lambda s: f"n{s['N']}/bf16", lambda s: f"BiasAdd_v1<{s['N']}> · BF16", {"in0 (y)": BIAS_Y.describe(), "in1 (b)": BIAS_B.describe()},
         pattern=r"n(?P<N>\d+)", example={"N": 2048}),
    _Row("bf16-mul-scalar", "Bf16MulScalarTensor_v1", "verity.ml.definitions.registry.b1:Bf16MulScalarTensor", ("N",),
         lambda s: (Port("in0", "u16", s["N"]), Port("in1", "u16", 1)), lambda s: _u16(s["N"], "out.out"),
         _two(SCALAR_X, SCALAR_S, lambda s: s["N"], lambda s: 1), lambda s: f"n{s['N']}/bf16", lambda s: f"Bf16MulScalarTensor_v1<{s['N']}> · BF16",
         {"in0 (x)": SCALAR_X.describe(), "in1 (the scalar)": SCALAR_S.describe()}, pattern=r"n(?P<N>\d+)", example={"N": 2304}),
)}


# --- a table or slab of which an input reads one row: embedding-row, moe-expert-coordinate(-weighted) ---------------------------

EMB_ROW = Bf16Window(127, (2, 4))


def _embedding_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    V, H = sub.params["V"], sub.params["H"]
    row, h = EMB_ROW.words(key, ("row", j), H, ("row", j))
    return Draw({"tok": np.asarray([key.uniform(V, ("tok", j))], dtype=np.uint32), "row": row}, {}, {"h": h})


def _embedding_eval(sub: Subcircuit, arrays: list[np.ndarray], _st: Mapping[str, int]) -> np.ndarray:
    """``Embedding_v1{V, H}`` on each input: its registered row kernel (the one ``evaluate_batch`` dispatches) over a V x H table of
    +0 holding the input's row at ``tok``.  One instance at a time, so the table is materialised once rather than per instance
    (at V x H = 5.9e8 words the batch path runs out of memory)."""
    from verity.ml.definitions.numerics import kernel_registry as KR

    V, H = sub.params["V"], sub.params["H"]
    fn = KR.instance(sub.target(), "rows")
    tok, rows = arrays
    table = np.zeros((V, H), dtype=np.uint16)
    out = np.empty((len(tok), H), dtype=np.uint64)
    for i in range(len(tok)):
        t = int(tok[i, 0])
        table[t] = rows[i]
        out[i] = fn(tok[i], table, V=V, H=H)
        table[t] = 0
    return out


def embedding_row(V: int, H: int) -> Subcircuit:
    V, H = int(V), int(H)

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import b1

        import verity.ml.definitions.numerics.rows  # noqa: F401
        return bind(b1.Embedding, V=V, H=H)

    rel = {"subcircuit": "Embedding_v1", "definition": "Embedding_v1", "statics": {"V": V, "H": H},
           "rule": "out = table[tok]: an input stores tok and the row it opens, table[tok]; every other table row is +0"}
    return Subcircuit(TEMPLATES["embedding-row"], {"V": V, "H": H}, f"embedding-row/v{V}-h{H}/bf16", f"Embedding_v1<{V}, {H}> · BF16",
                      row_set_name("Embedding_v1", {"V": V, "H": H}), rel, (Port("tok", "u32", 1), Port("row", "u16", H)),
                      (Port("out", "u16", H),), target, target, sampler=_embedding_sample, evaluator=_embedding_eval,
                      recipe={"tok": "key.uniform(V, ('tok', j))", "row": EMB_ROW.describe(), "table": "+0 but for row tok"})


def _embedding_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    return embedding_row(rel["statics"]["V"], rel["statics"]["H"]) if rel.get("definition") == "Embedding_v1" else None


def _ingress_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    from verity.ml.definitions.registry import io_shell as X

    MODEL, LP, V = sub.params["MODEL"], sub.params["LP"], sub.params["V"]
    ids = [key.uniform(V, ("id", j, k)) for k in range(LP)]
    m = 1 + key.uniform(10 ** X.MAX_TOKENS_WIDTH - 1, ("max_tokens", j))
    return Draw({"body": np.frombuffer(X.encode(MODEL, ids, m, V), dtype=np.uint8).copy()}, {}, {})


def ingress_prompt(MODEL: str, LP: int, V: int) -> Subcircuit:
    """The I/O shell's ingress parse unit (``io_shell.IngressPrompt_v1{MODEL,LP,V}``): the attested request body of ingress v0,
    whole SHA-512 blocks of bytes, to (ok, the prompt's LP token ids, max_tokens)."""
    MODEL, LP, V = str(MODEL), int(LP), int(V)

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import io_shell as X

        return bind(X.IngressPrompt, MODEL=MODEL, LP=LP, V=V)

    from verity.ml.definitions.registry import io_shell as X

    rel = {"subcircuit": "IngressPrompt_v1", "definition": "IngressPrompt_v1", "statics": {"MODEL": MODEL, "LP": LP, "V": V},
           "rule": "io_shell.reference: ok = the body is ingress v0's layout at (MODEL, LP, V); ids and max_tokens its fields' sums"}
    return Subcircuit(TEMPLATES["ingress-prompt"], {"MODEL": MODEL, "LP": LP, "V": V}, f"ingress-prompt/{MODEL}-lp{LP}-v{V}/v0",
                      f"IngressPrompt_v1<{MODEL}, {LP}, {V}> · ingress v0", row_set_name("IngressPrompt_v1", {"MODEL": MODEL, "LP": LP, "V": V}), rel,
                      (Port("body", "u8", X.length(MODEL, LP, V)),), (Port("ok", "u8", 1), Port("ids", "u32", LP), Port("max_tokens", "u32", 1)),
                      target, target, sampler=_ingress_sample, admit=lambda out: np.array([int(o[0]) == 1 for o in out["ok"]]),
                      recipe={"body": "io_shell.encode(MODEL, ids, max_tokens, V): ids key.uniform(V, ('id', j, k)), max_tokens "
                                      "1 + key.uniform(10^6 - 1, ('max_tokens', j))"})


def _ingress_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    if rel.get("definition") != "IngressPrompt_v1":
        return None
    return ingress_prompt(rel["statics"]["MODEL"], rel["statics"]["LP"], rel["statics"]["V"])


def _chunk_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    x, h = TOKEN_LOGITS.words(key, ("x", j), sub.params["C"], ("x", j))
    return Draw({"x": x}, {}, {"h": h})


def token_select_chunk(C: int) -> Subcircuit:
    """A V-chunk of the registered compiler's greedy select (``dead_free.TokenSelectChunk_v2{C}``): C logits, one SHA-512 block at
    C = 64, to their first maximum above -inf and its index in the chunk."""
    C = int(C)

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import dead_free as DF

        return bind(DF.TokenSelectChunkV2, C=C)

    rel = {"subcircuit": "TokenSelectChunk_v2", "definition": "TokenSelectChunk_v2", "statics": {"C": C},
           "rule": "the first maximum above -inf of the chunk's bf16 logits and its index (NaN never taken; (-inf, 0) when none)"}
    return Subcircuit(TEMPLATES["token-select-chunk"], {"C": C}, f"token-select-chunk/c{C}/bf16", f"TokenSelectChunk_v2<{C}> · BF16",
                      row_set_name("TokenSelectChunk_v2", {"C": C}), rel, (Port("x", "u16", C),),
                      (Port("best", "u16", 1), Port("idx", "u32", 1)), target, target, sampler=_chunk_sample, admit=_all,
                      recipe={"x": TOKEN_LOGITS.describe()})


def _chunk_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    return token_select_chunk(rel["statics"]["C"]) if rel.get("definition") == "TokenSelectChunk_v2" else None


def _combine_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    from verity.ml.definitions.registry import b1

    n, C = len(b1.chunks(sub.params["V"], sub.params["C"])), sub.params["C"]
    x0, h = TOKEN_LOGITS.words(key, ("x0", j), 1, ("x0", j))
    best, _ = TOKEN_LOGITS.words(key, ("best", j), n, ("x0", j))
    idx = np.array([key.uniform(C, ("idx", j, c)) for c in range(n)], dtype=np.uint32)
    return Draw({"x0": x0, "best": best, "idx": idx}, {}, {"h": h})


def token_select_combine(V: int, C: int) -> Subcircuit:
    """The registered compiler's greedy token (``dead_free.TokenSelectCombine_v2{V,C}``) from logits[0] and the row's V-chunk
    partials: the egress id, a 32-bit word."""
    V, C = int(V), int(C)

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import dead_free as DF

        return bind(DF.TokenSelectCombineV2, V=V, C=C)

    from verity.ml.definitions.registry import b1

    n = len(b1.chunks(V, C))
    rel = {"subcircuit": "TokenSelectCombine_v2", "definition": "TokenSelectCombine_v2", "statics": {"V": V, "C": C},
           "rule": "TokenSelect_v1's token from logits[0] and its V-chunks' (best, index) partials"}
    return Subcircuit(TEMPLATES["token-select-combine"], {"V": V, "C": C}, f"token-select-combine/v{V}-c{C}/bf16",
                      f"TokenSelectCombine_v2<{V}, {C}> · BF16", row_set_name("TokenSelectCombine_v2", {"V": V, "C": C}), rel,
                      (Port("x0", "u16", 1), Port("best", "u16", n), Port("idx", "u32", n)), (Port("id", "u32", 1),), target, target,
                      sampler=_combine_sample, admit=_all,
                      recipe={"x0, best": TOKEN_LOGITS.describe(), "idx": "key.uniform(C, ('idx', j, c))"})


def _combine_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    if rel.get("definition") != "TokenSelectCombine_v2":
        return None
    return token_select_combine(rel["statics"]["V"], rel["statics"]["C"])


def _expert_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    E, K, tile = sub.params["E"], sub.params["K"], int(sub.recipe.get("tile", 32))
    t = j // tile
    x, hx = GEMM_X.words(key, ("x", t), K, ("x", t))
    w, hw = GEMM_W.words(key, ("w", j), K, ("w", j))
    ins = {"x": x, "e": np.asarray([key.uniform(E, ("e", j))], dtype=np.uint32), "w": w}
    if sub.relation["definition"] == "MoeExpertGemmW_v1":
        ins["g"] = np.asarray([f32_uniform(key, ("g", j), 2.0 ** -10, 1.0)], dtype=np.uint32)
    return Draw(ins, {}, {"tile": t, "hx": hx, "hw": hw})


def _expert_eval(sub: Subcircuit, arrays: list[np.ndarray], _st: Mapping[str, int]) -> np.ndarray:
    """``MoeExpertGemm(W)_v1{E, K, N=1}`` (its registered row kernel) on the input's x, e (and g) and a slab of +0 holding the
    input's row at expert e: the coordinate Definition's value (``test_templates``)."""
    from verity.ml.circuits.evaluation import evaluate_batch

    E, K = sub.params["E"], sub.params["K"]
    x, e, *rest = arrays
    w = rest[-1]
    slab = np.zeros((len(x), E, 1, K), dtype=np.uint16)
    slab[np.arange(len(x)), e[:, 0].astype(np.int64), 0] = w
    return evaluate_batch(sub.target(), x, e, *rest[:-1], slab)


def moe_expert_coordinate(E: int, K: int, weighted: bool = False, tile: int = 32) -> Subcircuit:
    E, K = int(E), int(K)
    tid = "moe-expert-coordinate-weighted" if weighted else "moe-expert-coordinate"
    fam, coord, gemm = (("MoeExpertGemmW_v1", "MoeExpertCoordinateW", "MoeExpertGemmW") if weighted
                        else ("MoeExpertGemm_v1", "MoeExpertCoordinate", "MoeExpertGemm"))

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import moe

        import verity.ml.definitions.numerics.rows  # noqa: F401
        return bind(getattr(moe, gemm), E=E, K=K, N=1)

    def definition(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import moe

        return bind(getattr(moe, coord), E=E, K=K)

    ins = (Port("x", "u16", K), Port("e", "u32", 1), *((Port("g", "u32", 1),) if weighted else ()), Port("w", "u16", K))
    name = f"MoeExpertCoordinate{'W' if weighted else ''}_v1"
    rel = {"subcircuit": f"MoeExpertCoordinate{'W' if weighted else ''}<{K}>", "definition": fam, "dot": "AmpereBF16TcDot16_v1",
           "statics": {"E": E, "K": K},
           "rule": f"y = {name}{{E, K}}(x, e{', g' if weighted else ''}, slab): an input stores e and the routed expert's row w = slab[e]; "
                   "every other expert's row is +0"}
    return Subcircuit(TEMPLATES[tid], {"E": E, "K": K}, f"{tid}/e{E}-k{K}/bf16",
                      f"{name}<E={E}, K={K}> ·{' routed weight ·' if weighted else ''} BF16", f"{tid}-e{E}-k{K}", rel, ins,
                      (Port("y", "u16", 1),), target, definition, sampler=_expert_sample, evaluator=_expert_eval,
                      recipe={"tile": int(tile), "x": GEMM_X.describe() + "; one row per tile of `tile` draws", "w": GEMM_W.describe(),
                              "e": "key.uniform(E, ('e', j))", "slab": "+0 but for row e",
                              **({"g": "f32 of 2^-10 + (1 - 2^-10) u / 2^32"} if weighted else {})})


def _expert_identify(weighted: bool) -> Callable[[Mapping[str, Any]], Subcircuit | None]:
    fam = "MoeExpertGemmW_v1" if weighted else "MoeExpertGemm_v1"
    return lambda rel: moe_expert_coordinate(rel["statics"]["E"], rel["statics"]["K"], weighted) if rel.get("definition") == fam else None


# --- scaled-mm-fp8-block-coordinate ----------------------------------------------------------------------------------------------

SCALE_EXP = (107, 125)                             # block scales: positive normal f32 words, exponent field in this range


def scale_words(r: np.ndarray) -> np.ndarray:
    """Positive normal f32 words from 32 random bits each: mantissa ``r & 0x7FFFFF``, exponent field
    ``lo + (r >> 23) mod (hi - lo + 1)`` over :data:`SCALE_EXP`."""
    lo, hi = SCALE_EXP
    r = np.asarray(r, dtype=np.uint32)
    return (((lo + (r >> 23) % np.uint32(hi - lo + 1)) << 23) | (r & 0x7FFFFF)).astype(np.uint32)


def _fp8_block_sample(sub: Subcircuit, key: Key, j: int) -> Draw:
    K, G, tile = sub.params["K"], sub.params["G"], int(sub.recipe.get("tile", 32))
    t = j // tile
    return Draw({"xq": e4m3_finite_words(stream_u32(key, ("xq", t), K)), "sx": scale_words(stream_u32(key, ("sx", t), K // G)),
                 "wrow": e4m3_finite_words(stream_u32(key, ("wrow", j), K)), "sw": scale_words(stream_u32(key, ("sw", j), K // G))},
                {}, {"tile": t})


def scaled_mm_fp8_block_coordinate(K: int, G: int = 128, semantics: str = "sm90.wgmma.m64n8k32.e4m3", tile: int = 32) -> Subcircuit:
    K, G = int(K), int(G)
    if semantics != "sm90.wgmma.m64n8k32.e4m3":
        raise ValueError(f"scaled-mm-fp8-block-coordinate: semantics {semantics!r} (the H100's wgmma E4M3 step only)")
    if K % G or G % 32:
        raise ValueError(f"scaled-mm-fp8-block-coordinate: K = {K}, G = {G} (K a multiple of G, G of 32)")

    def target(**_s: int) -> Any:
        from verity.ml.circuits.defs import bind
        from verity.ml.definitions.registry import fp8

        return bind(fp8.ScaledMmFp8BlockCoordinate, K=K, G=G)

    rel = {"subcircuit": f"ScaledMmFp8BlockCoordinate<{K},{G}>", "definition": "ScaledMmFp8Block_v1", "dot": "HopperE4m3QgmmaDot32_v1",
           "statics": {"K": K, "G": G},
           "rule": "y = ScaledMmFp8BlockCoordinate_v1{K, G}(xq, sx, wrow, sw): per G-wide k block the E4M3 wgmma chain from +0, "
                   "acc = FFMA(block, sx[b] * sw[b], acc) from +0, then cvt.rn.bf16.f32"}
    return Subcircuit(TEMPLATES["scaled-mm-fp8-block-coordinate"], {"K": K, "G": G, "semantics": semantics},
                      f"scaled-mm-fp8-block-coordinate/k{K}-g{G}/sm90-wgmma-e4m3",
                      f"ScaledMmFp8BlockCoordinate_v1<K={K}, G={G}> · sm90 wgmma · FP8 (E4M3), block scales",
                      f"scaled-mm-fp8-block-coordinate-k{K}-g{G}", rel,
                      (Port("xq", "u8", K), Port("sx", "u32", K // G), Port("wrow", "u8", K), Port("sw", "u32", K // G)),
                      (Port("y", "u16", 1),), target, target, sampler=_fp8_block_sample,
                      recipe={"tile": int(tile), "xq": E4M3_RECIPE + "; xq and sx one per tile of `tile` draws", "wrow": E4M3_RECIPE,
                              "sx, sw": f"positive normal f32, exponent field in {list(SCALE_EXP)}, mantissa uniform"})


def _fp8_block_identify(rel: Mapping[str, Any]) -> Subcircuit | None:
    if rel.get("definition") != "ScaledMmFp8Block_v1":
        return None
    return scaled_mm_fp8_block_coordinate(rel["statics"]["K"], rel["statics"]["G"])


# --- the catalogue -------------------------------------------------------------------------------------------------------------

TEMPLATES: dict[str, Template] = {}


def _add(t: Template) -> None:
    TEMPLATES[t.id] = t


_add(Template("gemm-coordinate", "GemmCoordinate<K>",
              "one output word of x @ W^T: K multiply-accumulates on one tensor-core instruction's exact semantics (FP32 accumulation), "
              "then the bf16 epilogue", {"K": "int, a multiple of 16", "semantics": "the tensor-core instruction (GEMM_SEMANTICS)",
                                         "datatype": "operand datatype (bf16)"},
              gemm_coordinate, _gemm_identify, {"K": 2048, "semantics": "sm80.mma.m16n8k16.bf16"}))
_add(Template("attention-head", "AttentionHead<T,D>",
              "FlashAttention forward, one query head at position T-1 over its KV head: FA2 (AttentionHead_v3) or FA3 on the H100 "
              "(AttentionHead_v2); T varies per instance",
              {"D": "int, head dim", "BN": "int, the kernel's kBlockN", "T": "synthetic key-count range lo:hi (per draw)",
               "semantics": "sm80.fa2.bf16 or sm90.fa3.bf16"}, attention_head, _attn_identify, {"D": 64, "BN": 128, "T": "1:512"}))
_add(Template("attention-head-v18", "AttentionHead_v18<T,D>",
              "the registered compiler's FA2 head (dead_free.AttentionHead_v18) on sm120's steps, one query head of a decode-only launch "
              "(MASKED_FROM = ceil(T/BN) - 1); T varies per instance",
              {"D": "int, head dim", "BN": "int, the kernel's kBlockN", "T": "synthetic key-count range lo:hi (per draw)",
               "semantics": ATTN_V18_SEMANTICS}, attention_head_v18, _attn_v18_identify, {"D": 64, "BN": 128, "T": "1:512"}))
_add(Template("rope-head", "RoPEHead<D>", "NeoX rotary embedding of one head with its bf16 cos/sin row (RoPEHead_v1)", {"D": "int, even"},
              rope_head, _rope_identify, {"D": 64}))
for _r in ROWS.values():
    _add(Template(_r.template, _r.family, f"{_r.family} (verity.ml.definitions.registry), one row", {k: "static" for k in _r.statics},
                  (lambda r: lambda **p: _row_subcircuit(r, **p))(_r), _row_identify(_r), dict(_r.example)))
_add(Template("embedding-row", "Embedding_v1", "one row of the token embedding table, gathered by token id (Embedding_v1): an input "
              "stores the id and the row it opens", {"V": "int, vocabulary", "H": "int, hidden size"}, embedding_row, _embedding_identify,
              {"V": 32768, "H": 4096}))
_add(Template("ingress-prompt", "IngressPrompt_v1", "the I/O shell's ingress parse unit (io_shell.IngressPrompt_v1): the attested request "
              "body of ingress v0 to its ok bit, the prompt's token ids and max_tokens", {"MODEL": "str, the served model's name",
              "LP": "int, prompt tokens", "V": "int, vocabulary"}, ingress_prompt, _ingress_identify,
              {"MODEL": "served-model", "LP": 16, "V": 128256}))
_add(Template("token-select-chunk", "TokenSelectChunk_v2", "a V-chunk of the registered compiler's greedy select "
              "(dead_free.TokenSelectChunk_v2): C logits to their first maximum above -inf and its index", {"C": "int, logits per chunk"},
              token_select_chunk, _chunk_identify, {"C": 64}))
_add(Template("token-select-combine", "TokenSelectCombine_v2", "the registered compiler's greedy token from logits[0] and the row's "
              "V-chunk partials (dead_free.TokenSelectCombine_v2): the egress id", {"V": "int, vocabulary", "C": "int, logits per chunk"},
              token_select_combine, _combine_identify, {"V": 128256, "C": 64}))
_add(Template("moe-expert-coordinate", "MoeExpertCoordinate<K>", "one output word of the routed expert's GEMM (MoeExpertCoordinate_v1): "
              "GemmCoordinate_v1 on expert e's row", {"E": "int, experts", "K": "int, a multiple of 16"},
              lambda **p: moe_expert_coordinate(**p), _expert_identify(False), {"E": 64, "K": 2048}))
_add(Template("moe-expert-coordinate-weighted", "MoeExpertCoordinateW<K>", "the routed expert's GEMM coordinate scaled by its routing "
              "weight (MoeExpertCoordinateW_v1)", {"E": "int, experts", "K": "int, a multiple of 16"},
              lambda **p: moe_expert_coordinate(**p, weighted=True), _expert_identify(True), {"E": 64, "K": 1024}))
_add(Template("scaled-mm-fp8-block-coordinate", "ScaledMmFp8BlockCoordinate<K,G>", "one bf16 output word of the block-scaled FP8 GEMM "
              "(ScaledMmFp8BlockCoordinate_v1): E4M3 wgmma per G-wide block, fp32 block scales", {"K": "int", "G": "int, the block (128)",
              "semantics": "sm90.wgmma.m64n8k32.e4m3"}, scaled_mm_fp8_block_coordinate, _fp8_block_identify, {"K": 2560, "G": 128}))


def get(template_id: str) -> Template:
    if template_id not in TEMPLATES:
        raise KeyError(f"unknown subcircuit template {template_id!r} (one of {sorted(TEMPLATES)})")
    return TEMPLATES[template_id]


def subcircuit(template_id: str, **params: Any) -> Subcircuit:
    return get(template_id).bind(**params)


def identify(relation: Mapping[str, Any]) -> Subcircuit:
    """The subcircuit a set's ``relation`` names, captured or synthetic."""
    for t in TEMPLATES.values():
        sub = t.identify(relation)
        if sub is not None:
            return sub
    raise KeyError(f"no subcircuit template for relation {dict(relation)!r}")


def parse_subcircuit_id(sid: str) -> Subcircuit:
    """A census-style subcircuit id (``gemm-coordinate/k2048/sm80-mma-bf16``) back to its subcircuit (default synthetic recipe)."""
    tid, _, rest = sid.partition("/")
    params, _, sem = rest.partition("/")
    if tid == "gemm-coordinate":
        K = int(params.lstrip("k"))
        for s, g in [*GEMM_SEMANTICS.items(), *GEMM_FP8.items()]:
            if g.slug == sem:
                return gemm_coordinate(K, s)
        for s, (slug, _d, _t) in GEMM_MODEL_ONLY.items():
            if slug == sem:
                return gemm_coordinate(K, s)
        if sem == "sm120-mma-e2m1-nvf4":
            return gemm_coordinate(K, NVFP4_SEMANTICS)
    elif tid == "attention-head":
        d, _, bn = params.partition("-")
        for s, (*_head, tail, _kind, _step) in ATTN_SEMANTICS.items():
            if tail == sem:
                return attention_head(int(d.lstrip("d")), int(bn.lstrip("bn")), semantics=s)
    elif tid == "attention-head-v18" and (m := re.fullmatch(r"d(\d+)-bn(\d+)", params)) and sem == "sm120-fa2-bf16-decode":
        return attention_head_v18(int(m[1]), int(m[2]))
    elif tid == "rope-head":
        return rope_head(int(params.lstrip("d")))
    elif tid in ROWS:
        m = re.fullmatch(ROWS[tid].pattern, params)
        if m:
            sub = _row_subcircuit(ROWS[tid], **{k: (float(v) if k == "EPS" else int(v)) for k, v in m.groupdict().items()})
            if sub.id == sid:
                return sub
    elif tid == "embedding-row" and (m := re.fullmatch(r"v(\d+)-h(\d+)", params)):
        return embedding_row(int(m[1]), int(m[2]))
    elif tid in ("moe-expert-coordinate", "moe-expert-coordinate-weighted") and (m := re.fullmatch(r"e(\d+)-k(\d+)", params)):
        return moe_expert_coordinate(int(m[1]), int(m[2]), weighted=tid.endswith("weighted"))
    elif tid == "scaled-mm-fp8-block-coordinate" and (m := re.fullmatch(r"k(\d+)-g(\d+)", params)) and sem == "sm90-wgmma-e4m3":
        return scaled_mm_fp8_block_coordinate(int(m[1]), int(m[2]))
    elif tid == "ingress-prompt" and (m := re.fullmatch(r"(.+)-lp(\d+)-v(\d+)", params)) and sem == "v0":
        return ingress_prompt(m[1], int(m[2]), int(m[3]))
    elif tid == "token-select-chunk" and (m := re.fullmatch(r"c(\d+)", params)) and sem == "bf16":
        return token_select_chunk(int(m[1]))
    elif tid == "token-select-combine" and (m := re.fullmatch(r"v(\d+)-c(\d+)", params)) and sem == "bf16":
        return token_select_combine(int(m[1]), int(m[2]))
    raise KeyError(f"not a subcircuit id this catalogue binds: {sid!r}")


__all__ = ["BF16_EDGES", "Draw", "GEMM_SEMANTICS", "GEMM_MODEL_ONLY", "NotEvaluable", "Port", "ROWS", "Subcircuit", "TEMPLATES", "Template",
           "bf16_words", "decode_static", "encode_static", "get", "identify", "parse_subcircuit_id", "row_set_name", "subcircuit"]
