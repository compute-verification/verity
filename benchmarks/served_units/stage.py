"""stage.py UNIT N OUT [--public-weights [--packed]]: N inputs of one served proof-block type's Q_word proof block, from synthetic
values at the model's served statics, staged for Flock as a ``verity/flock-circuit`` statement with hidden outputs. For a
`CENSUS` type N may be ``full``, the most instances a statement of `MAX_STATEMENT_BITS` holds at the built block's k_log
(a power of two, as `class_statement` cuts a PARTS type's N), or ``full/D``, that over D.

    UNIT  TYPE[:S=V,...]   a type of `UNITS`, `PARTS` or `CENSUS` at its statics (each type's defaults are its model's: GLM-5.3's
                           at TP8, s8-coord's Llama-3.2-1B's), e.g. fp8-coord:K=2048, router:C=1, s8-coord:K=8192, gq:K=384
          census:NAME      any block type of the serving rows by its census name verbatim, staged as a `CENSUS` type, e.g.
                           census:MoeSum_v1{TOPK=8,H=3072}
    --public-weights       a `UNITS` type's weight row ``w`` as public input ports (`UnitStager`'s ``public_inputs``, each leaf in
                           its own u16 words) instead of a row hashed in the circuit, on the same values: the price of the block
                           with its weight row's in-circuit opening removed, the in-circuit share a registered weight encoding
                           would take out of the block. It prices no registered encoding's own opening, and the verifier refuses
                           a statement with public input ports as outside the proved scope (``ProvedScope``) after every check.
    --packed               with --public-weights, each port its parameter's leaves concatenated as the row lays them (the row's
                           own words; `UnitStager`'s ``packed``): a registered encoding read packed, so the unit's input words
                           are the opened block's.

A type of `UNITS` is a Definition proved as one proof block, staged by `class_statement.stage_unit`: its Boolean Definition
lowered once, its input rows hashed in the circuit (the per-call row on ``hm96-sha512/row-seg/v1``, the weight row on
``hm96-sha512/row/v2``, PoUW's convention), its output word one ``row/v2`` output row, every input's output checked by
`stage_unit` against the Program's reference of the Boolean Definition and here against the served kernel's numpy twin on the same values (``twin``).

A type of `PARTS` is a word Definition whose Q_word proof blocks are parts of it: the Definition as a one-Call Program cut by
`Q_word` v1 (X = 16, W = 32), its classes ranked by word gates, and the class of rank ``C`` (static ``C``, default 0: the
largest) staged by `class_statement.stage` with every input's own ``sha512/row/v1`` rows (``share_rows=False``): the class
lowered once (Flock's pieces, and for a primitive without one the registered Boolean Definition's builder), its lanes the
flat Definition evaluated on uniform synthetic parameters (`class_statement.class_lanes`), the laid-out proof block checked against
them on every input (``twin`` null: no served twin cuts the same proof block).  The circuit binds the partition's
``verity/partition/v1`` digest, so the verifier is given its own copy of the partition object (partition.json, the Lean
verifier's ``--partition``); not the program, from which it derives a statement's proof blocks only where each is one whole
activation of the circuit's template (``Extract.claimedUnits``), and a class here is part of one, on synthetic Calls.

A type of `CENSUS` is a block type of the serving rows by its census name (`serving_templates.template`, the name a proof
overhead card prices it under), one whole Definition a proof block, laid out as the served composer lays its template
(`compose_circuit.py --template`): its served rows (`chunk_rows.unit_rows`, each parameter's leaves seeded at their width
in the template's word view, `census_leaves`) and statement (`chunk_rows.statement`: the 64-key sparse MLA split part's MUFU
reads in Flock's lookup slots, its unit cut into stage nets under the lowering's automatic per-net AND limits, which keep it
within the GPU prover's slot types; RMSNorm's inline, and a Boolean ``_v2`` template's flat, its MUFU reads in gates) under
its unit name, so the statement is the served one but for its binding: its ``class_sha512`` is the served composer's. The
circuit is streamed to circuit.txt once, every input's outputs checked against the Program's reference (``twin`` null).
``census:NAME`` stages any other block type of the serving rows the same way, its values seeded by the same rule and its
seeds keyed by the name.

A type of `READS` is a table read, ``EmbeddingRead_v1{V, H}`` on its served rows (`interface.unit_rows`), staged as serving
stages a read (`chunk_rows.statement`): the templated statement at the read layout on the verifier's own read template
(``structural``, `verity_flock.read_tmpl`, so ``flock-read-tmpl`` is built first). Its inputs are honest reads of one table of
V ranks whose first L hold filled rows and every later one the zero row under the zero salt, as a history table pads to its
capacity; each read's row and verdict are held to the table's (``twin``). Beside the circuit it writes the template and its
layout mode (``unit.tmpl``, ``unit.layout``), which ``flock-tmpl-unit`` places as the prover's ``--tmpl-unit`` rows.

Writes circuit.txt, inst.bin, pub.bin, pin.txt and stage.json under OUT, a PARTS type's partition.json, and beside a statement
reading in slots the MUFU tables the prover reads (``tables/``, ``flock-circuit --tables``), as the served composer does.

Types (GLM-5.3's statics from its config: hidden 6144, q_lora 2048, kv_lora 512, moe_intermediate 2048, 256 experts, top 8,
FP8 scale groups of 128, DSA indexer 32 heads of 128, index_topk 2048; per TP8 rank):

    fp8-coord   ScaledMmFp8BlockCoordinate_v2{K,G}: one output coordinate of a block-scaled FP8 linear (CUTLASS blockwise,
                `ScaledMmFp8Block_v1{K,N,128}`'s Q_word proof block), x = Fp8GroupQuant of a seeded N(0, 1) bf16 row, w = a seeded
                N(0, 0.02) bf16 weight row quantized per 128 the same way.  K = 6144 is fused_qkv_a and the dense and
                shared experts' gate_up; 2048 is q_b, the indexer's wq_b and o_proj (one TP8 rank); 256 is the shared
                expert's down_proj on one rank.  The routed experts run Triton's fused MoE (`moe-w13`, `moe-w2`).
    s8-coord    ScaledMmS8Coordinate_v2{K,DOT=BlackwellS8ImmaDot32_v2}: one output coordinate of a W8A8 linear on sm_120 (K/32
                satfinite s8 IMMA steps, then steps.py's int8_linear scaling), x = a seeded N(0, 1) row quantized per token to s8
                (scale absmax/127), w = a seeded N(0, 0.02) row quantized per channel the same way; the twin is the epilogue in
                numpy (the exact int sum, two float32 multiplies, the bf16 round).  Llama-3.2-1B's statics: K = 2048 is q, k, v,
                o, gate and up, 8192 is down.
    s8-linear   `PARTS`: ScaledMmS8_v1{K,N,DOT=BlackwellS8ImmaDot32_v1}, the W8A8 linear on one token row, whose `Q_word`
                proof blocks are its N coordinates (`ScaledMmS8Coordinate_v1`): the s8-coord proof block as a class of the partition.
    moe-chunk   `PARTS`: the routed experts as the registered circuit partitions them, the expert's rows gathered inside the
    moe-gemmw   block from all 256 experts' (`fp8_moe_experts`): K = 512 of a w13 coordinate (a middle chunk) and one w2
                coordinate, on the H200's wgmma step.  `moe-w13` and `moe-w2` are the same products with the expert's rows as
                the block's inputs instead.
    gq          `CENSUS`: Fp8GroupQuantUe8m0_v1{K,G}, one token row's quantization to E4M3 in groups of G with power-of-two
                scales, before an FP8 linear of K inputs (K = 6144: the hidden state's); a K below 6144 is a column block.
    fi          `CENSUS`: FlashInferFp8Linear_v2{K,N}, the FlashInfer FP8 linear's output coordinate (N = 1) on one token row
                as served, the Boolean version of ``FlashInferFp8Linear_v1``: its bf16 x quantized in the block, its RCP
                reads in gates, an E4M3 weight row and its f32 block scales (K = 2048: q_b and o_proj).
    fi-quant    `CENSUS`: FlashInferFp8Quant1x128_v2{K}, that linear's quantisation part, one token row's x to E4M3 in groups of
                128 and their scales, as served (the row's quantisation once, which its coordinates read).
    router-row  `CENSUS`: MoeRouterSigmoidBias_v2{E=256,TOPK=8,..}, the MoE router's sigmoid, bias and top-8 of one token row as
    router-slots  served, its EX2 and RCP reads in gates; ``router-slots`` is its slots form (the per-slot outputs).
    dsa-q-row   `CENSUS`: DsaIndexerQ_v2{H=32,D=128,ROPE=64}, DSA's indexer query row as served, its EX2 reads in gates.
    mla-split-part  `CENSUS`: SparseMlaSplitPart_v1 at T = 64, one query's sparse MLA over one split's 64 keys (FA3's).
    rmsnorm-v8  `CENSUS`: RMSNormFusedCudaVector8_v1{N=6144,BLOCK=256}, the fused residual add and RMSNorm of one token row.
    table-read  `READS`: EmbeddingRead_v1{V,H}, one read of a request's history table for sparse MLA: V = 131072 ranks (the
                table's capacity), H = 576 words (a kv row's 512 latent and 64 rope bf16 words), L = 8192 rows filled.
    dsa-chunk   DsaPositionsChunk_v1{K,W}: chunk c of the long row's index, the merge's K = 2048 slots in chunks of W = 32,
                each port its own row; its slots below N = 131072 (the longest row's class) but a tenth of any u32.
    dsa-slot    DsaPositionsSlot_v1{K,N}: slot j of a chunk (K = 32), held below N = 131072 whatever its inputs.
    the rest    `PARTS`: the capture map's Programs (``glm-moe-dsa-eager-hopper-tp8-v0``) at their GLM-5.3 statics; the MoE
                tiles' KB and SWAP are Triton's 128 and swapped until the H200 capture pins them."""
from __future__ import annotations

import hashlib
import json
import re
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from circuit_check import targets as TG
from verity.circuits import partition_object as PO
from verity.circuits.codec import program_digest
from verity.circuits.types import Array
from verity.circuits.defs import bind
from verity.definitions.ops import gemm as OG
from verity.definitions.ops.boolean import gemm as BG
from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.ops.prims import HopperE4m3QgmmaDot32
from verity.definitions.registry import act, b1, dsa, dsa_topk_reads, fp8_moe, fp8_moe_experts, moe, ref_prims, sparse_mla
from verity.definitions.registry import boolean_fp8 as BF


#: the prover's witness bits a statement (70-class-sweep.sh's MAX_STATEMENT_BITS): `class_statement.stage` cuts a PARTS type's
#: N to the most instances under it
MAX_STATEMENT_BITS = 1 << 35


def bf16(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def words(*parts: np.ndarray) -> np.ndarray:
    """A row's u16 words: the parts' little-endian bytes concatenated (a Boolean element's bit 0 is its word's low bit)."""
    raw = b"".join(np.ascontiguousarray(p).astype(p.dtype.newbyteorder("<")).tobytes() for p in parts)
    return np.frombuffer(raw, dtype="<u2").copy()


@dataclass(frozen=True)
class Unit:
    """A proof-block type: its Boolean Definition at statics ``S``, its input rows ``(row, schema, params)``, and
    ``values(S, n, rng)``: each row's words per input ``(n, words)`` and the served twin's output words ``(n, out_words)``."""

    definition: Callable[[dict], object]
    rows: tuple[tuple[str, str, tuple[str, ...]], ...]
    values: Callable[[dict, int, np.random.Generator], tuple[dict[str, np.ndarray], np.ndarray]]
    defaults: dict


def _fp8_coord_values(S: dict, n: int, rng: np.random.Generator):
    K, G = S["K"], S["G"]
    xs, ws, twin = [], [], []
    for _ in range(n):
        xq, sx = DR.fp8_group_quant_row(bf16(rng.standard_normal(K)), G=G)
        wq, sw = DR.fp8_group_quant_row(bf16(rng.standard_normal(K) * 0.02), G=G)
        out, nonfin = DR.scaled_mm_fp8_block_row(xq, sx, wq[None], sw[None], G=G)
        assert not nonfin.any(), "a synthetic coordinate is non-finite"
        xs.append(words(xq, sx))
        ws.append(words(wq, sw))
        twin.append(out[:1])
    return {"x": np.stack(xs), "w": np.stack(ws)}, np.stack(twin)


def _s8_quant(v: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """steps.py's `quant_rows` at top 127, in float32."""
    v = v.astype(np.float32)
    s = np.abs(v).max() / np.float32(127)
    return np.clip(np.rint(v / s), -127, 127).astype(np.int8), s


def _s8_coord_values(S: dict, n: int, rng: np.random.Generator):
    K = S["K"]
    xs, ws, twin = [], [], []
    for _ in range(n):
        xq, sx = _s8_quant(rng.standard_normal(K))
        wq, sw = _s8_quant(rng.standard_normal(K) * 0.02)
        acc = np.int32((xq.astype(np.int64) * wq).sum())
        y = (np.float32(acc) * np.float32(sx)) * np.float32(sw)
        xs.append(words(xq.view(np.uint8), np.array([sx], np.float32).view(np.uint32)))
        ws.append(words(wq.view(np.uint8), np.array([sw], np.float32).view(np.uint32)))
        twin.append(bf16(np.array([y], np.float32)))
    return {"x": np.stack(xs), "w": np.stack(ws)}, np.stack(twin)


def _dsa_index_rows(S: dict, n: int, rng: np.random.Generator):
    """A merge's slots (positions below N, a tenth of them any u32, as a slot unit's bound sees them) and an index below the
    unit's count, one word pair a port: a chunk's (slots, c) when S has W, a slot's (slots, j) else; the twin `dsa_topk_reads`'
    `chunk` or `slot`."""
    K, N, W = S["K"], S["N"], S.get("W")
    slots = np.where(rng.random((n, K)) < 0.9, rng.integers(0, N, (n, K)), rng.integers(0, 1 << 32, (n, K))).astype(np.uint32)
    idx = rng.integers(0, -(-K // W) if W else K, n).astype(np.uint32)
    twin = [dsa_topk_reads.chunk(s.tolist(), int(i), W) if W else [dsa_topk_reads.slot(s.tolist(), int(i), N)]
            for s, i in zip(slots, idx)]
    return {"slots": np.stack([words(s) for s in slots]), "i": idx.view("<u2").reshape(n, 2)}, \
        np.stack([words(np.array(t, np.uint32)) for t in twin])


UNITS: dict[str, Unit] = {
    "fp8-coord": Unit(lambda S: bind(BF.ScaledMmFp8BlockCoordinateV2, K=S["K"], G=S["G"]),
                      (("x", SEG, ("xq", "sx")), ("w", V2, ("wrow", "sw"))), _fp8_coord_values, {"K": 6144, "G": 128}),
    "s8-coord": Unit(lambda S: bind(BG.ScaledMmS8Coordinate, K=S["K"], **BG.GEMM_SM120_S8),
                     (("x", SEG, ("x", "sx")), ("w", V2, ("wrow", "sw"))), _s8_coord_values, {"K": 2048}),
    "dsa-chunk": Unit(lambda S: bind(dsa_topk_reads.DsaPositionsChunk, K=S["K"], W=S["W"]),
                      (("slots", SEG, ("slots",)), ("i", SEG, ("c",))), _dsa_index_rows, {"K": 2048, "W": 32, "N": 1 << 17}),
    "dsa-slot": Unit(lambda S: bind(dsa_topk_reads.DsaPositionsSlot, K=S["K"], N=S["N"]),
                     (("slots", SEG, ("slots",)), ("i", SEG, ("j",))), _dsa_index_rows, {"K": 32, "N": 1 << 17}),
}


@dataclass(frozen=True)
class Part:
    """A proof-block type cut from a word Definition: the family, bound at statics ``S`` (``C``, the class's rank, is not one)."""

    family: object
    defaults: dict
    fixed: tuple = ()

    def definition(self, S: dict):
        return bind(self.family, **{k: v for k, v in S.items() if k != "C"}, **dict(self.fixed))


_MLA = tuple((k, v) for k, v in sparse_mla.SPARSE_MLA_GLM53.items() if k not in ("NH", "DR", "DV", "BN", "SCALE"))
_HOPPER = (("DOT", HopperE4m3QgmmaDot32),)
PARTS: dict[str, Part] = {
    "router": Part(moe.MoeRouterSigmoidBias, {"E": 256, "TOPK": 8, "SCALE": 2.5, "KERNEL": "block"}),
    "moe-w13": Part(fp8_moe.MoeExpertFp8BlockGemm, {"K": 6144, "N": 2, "G": 128, "KB": 128, "SWAP": True}),
    "moe-w2": Part(fp8_moe.MoeExpertFp8BlockGemmW, {"K": 256, "N": 2, "G": 128, "KB": 128, "SWAP": True}),
    "moe-chunk": Part(fp8_moe_experts.MoeExpertFp8CoordinateChunk,
                      {"E": 256, "K": 512, "G": 128, "KB": 128, "SWAP": True, "FIRST": False, "LAST": False}, _HOPPER),
    "moe-gemmw": Part(fp8_moe_experts.MoeExpertFp8GemmW, {"E": 256, "K": 256, "N": 1, "G": 128, "KB": 128, "SWAP": True}, _HOPPER),
    "silu-quant": Part(fp8_moe.SiluMulFp8BlockQuant, {"I": 256, "G": 128}),
    "moe-sum": Part(moe.MoeSum, {"TOPK": 8, "H": 2}),
    "silu-mul": Part(act.SiluMul, {"I": 2}),
    "rope": Part(ref_prims.RoPEHeadGptJ, {"D": 64}),
    "rmsnorm-triton": Part(b1.RMSNormTriton, {"N": 2048, "EPS": 1e-6}),
    "rmsnorm-cuda": Part(b1.RMSNormFusedCuda, {"N": 6144, "EPS": 1e-5}),
    "dsa-krow": Part(dsa.DsaIndexerKRow, {"D": 128, "ROPE": 64, "EPS": 1e-6}),
    "dsa-q": Part(dsa.DsaIndexerQ, {"H": 32, "D": 128, "ROPE": 64}),
    "dsa-logits": Part(dsa.DsaMqaLogits, {"N": 2, "H": 32, "D": 128}),
    "dsa-topk": Part(dsa.DsaTopKShortRow, {"TOPK": 2048}),
    "sparse-mla-head": Part(sparse_mla.SparseMlaHead, {"T": 2048, "DR": 64, "DV": 512, "BN": 64, "SCALE": 0.0625}, _MLA),
    "token-select": Part(b1.TokenSelect, {"V": 154880}),
    "s8-linear": Part(OG.ScaledMmS8, {"K": 2048, "N": 2}, tuple(OG.GEMM_SM120_S8.items())),
}
for _p in PARTS.values():
    _p.defaults.setdefault("C", 0)


@dataclass(frozen=True)
class Census:
    """A census type: its family, bound at statics ``S`` in the order of ``defaults`` (its census name's), then ``fixed``, the
    statics no spec overrides (a function's or a float's, as the census name writes them)."""

    family: str
    defaults: dict
    fixed: str = ""

    def name(self, S: dict) -> str:
        inner = ",".join([*(f"{k}={v}" for k, v in S.items()), *filter(None, [self.fixed])])
        return f"{self.family}{{{inner}}}" if inner else self.family


_EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
_ROUTER = 'E=256,TOPK=8,SCALE={"f64":"0x1.4000000000000p+1"},KERNEL="block"'
CENSUS: dict[str, Census] = {
    "gq": Census("Fp8GroupQuantUe8m0_v1", {"K": 6144, "G": 128}),
    "fi": Census("FlashInferFp8Linear_v2", {"K": 2048, "N": 1}),
    "fi-quant": Census("FlashInferFp8Quant1x128_v2", {"K": 6144}),
    "router-row": Census("MoeRouterSigmoidBias_v2", {}, _ROUTER),
    "router-slots": Census("MoeRouterSigmoidBiasSlots_v2", {}, _ROUTER),
    "dsa-q-row": Census("DsaIndexerQ_v2", {}, "H=32,D=128,ROPE=64"),
    "mla-split-part": Census("SparseMlaSplitPart_v1", {}, 'T=64,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
                             'INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}'),
    "rmsnorm-v8": Census("RMSNormFusedCudaVector8_v1", {}, f"N=6144,EPS={_EPS},BLOCK=256"),
}


@dataclass(frozen=True)
class Read:
    """A table read type: ``EmbeddingRead_v1{V, H}`` at statics ``V``, ``H`` and ``L``, the rows its table holds."""

    defaults: dict


#: a history table's read at the served capacity and GLM-5.3's kv row (512 latent and 64 rope bf16 words)
READS: dict[str, Read] = {"table-read": Read({"V": 131072, "H": 576, "L": 8192})}


def census_leaves(rng: np.random.Generator, ty, n: int) -> np.ndarray:
    """``n`` rows of a parameter of type ``ty`` as u16 words, its leaves packed at their width: bf16 of N(0, 1) for 16-bit
    leaves, E4M3 codes other than NaN for 8-bit ones, f32 powers of two from 2^-10 to 2^-4 (a block scale) for 32-bit ones,
    and uniform bits for 1-bit ones, leaf j the low bit of word j / 16 on (a table read's token, digests, salt and path, and
    every port of a fused read's member, which its word view gives at bit level)."""
    w = set(ty.leaf_widths())
    if w == {16}:
        return bf16(rng.standard_normal((n, ty.leaves)))
    if w == {8} and ty.leaves % 2 == 0:
        c = rng.integers(0, 256, (n, ty.leaves), dtype=np.uint8)
        c[(c & 0x7F) == 0x7F] ^= 1
        return c.view("<u2")
    if w == {1} and ty.leaves % 16 == 0:
        return np.packbits(rng.integers(0, 2, (n, ty.leaves), dtype=np.uint8), axis=1, bitorder="little").view("<u2")
    if w == {32}:
        return np.ldexp(np.ones((n, ty.leaves), np.float32), rng.integers(-10, -3, (n, ty.leaves))).astype("<f4").view("<u2")
    raise SystemExit(f"stage.py: a census parameter of {ty.leaves} leaves of {sorted(w)} bits, which `census_leaves` doesn't generate")


def _static(default, v: str):
    if isinstance(default, bool):
        return v.lower() in ("1", "true")
    return type(default)(v)


def parse(spec: str) -> tuple[str, Unit | Part | Census | Read, dict]:
    name, _, rest = spec.partition(":")
    if name == "census":
        family, brace, inner = rest.partition("{")
        if not family or (brace and not inner.endswith("}")):
            raise SystemExit(f"stage.py: census:NAME takes a census name, FAMILY or FAMILY{{STATICS}}, not {rest!r}")
        return name, Census(family, {}, inner[:-1]), {}
    if name not in UNITS and name not in PARTS and name not in CENSUS and name not in READS:
        raise SystemExit(f"stage.py: unit type {name!r} is not one of "
                         f"{sorted(UNITS) + sorted(PARTS) + sorted(CENSUS) + sorted(READS)}")
    u = UNITS.get(name) or PARTS.get(name) or CENSUS.get(name) or READS[name]
    S = dict(u.defaults)
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        if k not in S:
            raise SystemExit(f"stage.py: {name} has no static {k!r} (its statics: {sorted(S)})")
        S[k] = _static(S[k], v)
    return name, u, S


def gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def stage_part(name: str, part: Part, S: dict, tag: str, n: int, out: Path, t0: float) -> dict:
    """The class of rank ``S["C"]`` of the type's word Definition under `Q_word`, ``n`` inputs, as stage.json's record."""
    TG.load_registries()
    CS.cache_slot_circuits()
    CS._CTX["max_statement_bits"] = MAX_STATEMENT_BITS
    fn = part.definition(S)
    prog = PU.one_call(fn)
    units = PU.partition_units(prog, "q-word")
    ranked = sorted(units.classes().items(), key=lambda kv: (-len(kv[1].representative.gates), -kv[1].units, kv[0]))
    s, cl = ranked[S["C"]]
    rep = cl.representative
    key = PU.derive(PU.SOURCE, PU.DOMAIN, {"program": program_digest(prog), "seed": 0})
    lanes = CS.class_lanes(units, cl, key, n, CS.broadcast_params(units))
    t_values = time.time() - t0
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    st = CS.stage(units, "q-word", s, cl, lanes, out, salt_key, packed=False, share_rows=False)
    (out / f"inst-{st['n']}.bin").rename(out / "inst.bin")
    (out / f"pub-{st['n']}.bin").rename(out / "pub.bin")
    (out / "partition.json").write_text(json.dumps(PO.build(units.program, units.query), sort_keys=True))
    meta = json.loads((out / "circuit.txt").read_text().split("\n", 2)[1][5:])
    return {"unit": name, "statics": S, "tag": tag, "n": st["n"], "definition": rep.definition, "type_definition": fn.id,
            "class": {"shape": s[:16], "unit": rep.unit, "units": cl.units, "word_gates": len(rep.gates),
                      "in_bits": sum(rep.in_widths), "out_bits": sum(rep.out_widths), "rank": S["C"], "classes": len(ranked),
                      "q_word_units": sum(c.units for _s, c in ranked)},
            "ands": st["ands"], "unit_nnz": st["unit_nnz"], "k_log": st["k_log"], "unit_log": meta.get("unit_log"),
            "vus_per_block": st["vus_per_block"], "row_schemas": meta.get("row_schemas"), "circuit_bytes": st["circuit_bytes"],
            "circuit_sha512": st["circuit_sha512"], "class_sha512": st["class_sha512"],
            "inst_bytes": (out / "inst.bin").stat().st_size, "pub_bytes": (out / "pub.bin").stat().st_size, "twin": None,
            "seconds": {"values": round(t_values, 1), "total": round(time.time() - t0, 1)}, "maxrss_gb": gb()}


def nets(low) -> list[dict]:
    """Each net's ANDs, rows and matrix entries a side: the unit's, then under read slots each tail stage's (the circuit's
    ``unit``, ``stage0``, ...; a lookup slot's table counts its entries once per statement, in the prover)."""
    stages = getattr(getattr(low, "tail", None), "stages", ())
    lays = [("unit", low.layout), *((f"stage{i}", st.layout) for i, st in enumerate(stages))]
    return [{"net": nm, **{k: int(lay.counts[k]) for k in ("ands", "rows", "nnz_a", "nnz_b")}} for nm, lay in lays]


def stage_census(name: str, cen: Census, S: dict, tag: str, n: int | str, out: Path, t0: float) -> dict:
    """``n`` inputs of the census type's Definition, a proof block each, on its served rows and statement, as stage.json's
    record; ``n`` ``full`` or ``full/D`` is the most instances a statement holds at the built block's k_log (as
    `class_statement` cuts a PARTS type's), over D."""
    from verity_vllm.pipeline import chunk_rows as CR
    from verity_vllm.pipeline.serving_rows import template

    try:
        tp = template(cen.name(S))
    except KeyError as e:
        raise SystemExit(f"stage.py: {cen.name(S)} lacks the static {e}: census:NAME takes the census name with every static, "
                         "as the serving rows name it") from None
    d = tp.fn
    params, _ret = template(tp.word).fn.signature()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    vals: dict[str, np.ndarray] = {}
    types = dict(params)

    def part(p: str) -> np.ndarray:
        """Parameter ``p``'s words, or for a cut row's ``q[i]`` the words of q's element i, q generated whole."""
        if p in vals:
            return vals[p]
        m = re.fullmatch(r"(\w+)\[(\d+)\]", p)
        if not (m and m[1] in vals and isinstance(types[m[1]], Array) and vals[m[1]].shape[1] % types[m[1]].n == 0):
            raise SystemExit(f"stage.py: {tp.id}'s served row part {p!r} is neither a parameter nor an element of one")
        w = vals[m[1]].shape[1] // types[m[1]].n
        return vals[m[1]][:, int(m[2]) * w:(int(m[2]) + 1) * w]

    in_rows, out_rows = CR.unit_rows(tp)
    stmt = CR.statement(tp)
    st = CS.UnitStager(d, "unit/serving-rows", in_rows, out_rows, circuit=out / "circuit.txt", **stmt)
    t_built = time.time() - t0
    if isinstance(n, str):
        g = st.meta.get("vus_per_block") or 1
        full = 1 << ((MAX_STATEMENT_BITS // ((1 << st.meta["k_log"]) // g)).bit_length() - 1)
        n = max(g, full // int(n.partition("/")[2] or 1))
    vals.update({p: census_leaves(rng, ty, n) for p, ty in params})
    rows = {r: np.concatenate([part(p) for p in ps], axis=1) for r, _s, _b, ps in in_rows}
    if any(rows[r].shape != (n, b // 16) for r, _s, b, _ps in in_rows):
        raise SystemExit(f"stage.py: {tp.id}'s generated rows {[rows[r].shape for r, *_x in in_rows]} are not its rows' words")
    t_values = time.time() - t0 - t_built
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    got = st.stage(rows, out, salt_key)
    if stmt.get("reads") == "slots":
        FC.IL.write_tables(out / "tables")
    meta = got["meta"]
    return {"unit": name, "statics": S, "tag": tag, "n": n, "definition": d.id, "word": tp.word, "census": cen.name(S),
            "in_rows": [[r, s, b, ps] for r, s, b, ps in in_rows], "public_inputs": [], "packed": False, "ands": got["ands"],
            "statement": stmt, "net_and_limits": getattr(st.low, "caps", None), "nets": nets(st.low), "lookups": meta.get("lookups"),
            "slot_types": len({r["net"] for r in meta.get("ranges", ())}),
            "unit_nnz": got["unit_nnz"], "k_log": meta["k_log"], "unit_log": meta.get("unit_log"),
            "vus_per_block": meta.get("vus_per_block"), "row_schemas": meta.get("row_schemas"),
            "circuit_bytes": (out / "circuit.txt").stat().st_size, "circuit_sha512": st.pin, "class_sha512": st.class_sha512,
            "inst_bytes": (out / "inst.bin").stat().st_size, "pub_bytes": (out / "pub.bin").stat().st_size, "macs": 0,
            "twin": None, "seconds": {"values": round(t_values, 1), "built": round(t_built, 1),
                                      "round": round(time.time() - t0 - t_built - t_values, 1), "total": round(time.time() - t0, 1)},
            "maxrss_gb": gb()}


def stage_read(name: str, S: dict, tag: str, n: int, out: Path, t0: float) -> dict:
    """``n`` honest reads of one table, a proof block each, on the verifier's read template, as stage.json's record. The
    table has V ranks of H words: its first L rows filled at random under random salts, every later rank the zero row under the
    zero salt (the history table's placeholder), its tokens random below L; each read's row and verdict are held to the table's."""
    from verity.commitments import row_table
    from verity.commitments.gates import embedding as E
    from verity.commitments.indexed import RangeIndexedDomain
    from verity.commitments.merkle import CommitmentDomain
    from verity.commitments.windows import interface as IF
    from verity_vllm.pipeline.chunk_rows import READ_LAYOUT

    V, H, L = S["V"], S["H"], S["L"]
    if not 1 <= L <= V:
        raise SystemExit(f"stage.py: a table of {V} ranks holds 1 to {V} rows, not L = {L}")
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    table = rng.integers(0, 1 << 16, (L, H), dtype=np.uint16)
    salts = [rng.bytes(E.SALT_BYTES) for _ in range(L)]
    dom = CommitmentDomain(rng.bytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    zero = row_table.string(bytes(2 * H), 16 * H, bytes(E.SALT_BYTES))
    tab = row_table.RowTable(dom, [row_table.string(table[r].astype("<u2").tobytes(), 16 * H, salts[r]) for r in range(L)]
                             + [zero] * (V - L))
    toks = [int(t) for t in rng.integers(0, L, n)]
    fn = E.read(V, H)
    in_rows, out_rows = IF.unit_rows(fn, None)
    cases = [(t, tab.root, dom.domain_id, table[t].tolist(), salts[t], [p[k:k + 64] for k in range(0, len(p), 64)])
             for t, p in ((t, tab.path(t)) for t in toks)]
    rows = {r: np.stack([np.packbits(np.array(E.arguments(V, *c)[g], np.uint8), bitorder="little").view("<u2") for c in cases])
            for g, (r, *_x) in enumerate(in_rows)}
    t_rows = time.time() - t0
    st = CS.UnitStager(fn, "unit/table-read", in_rows, out_rows, circuit=out / "circuit.txt", layout=READ_LAYOUT,
                       structural=True)
    t_built = time.time() - t0
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    got = st.stage(rows, out, salt_key)
    (out / "unit.tmpl").write_text(FC.template_of(st.text))
    (out / "unit.layout").write_text(st.meta["layout"])
    h, ok = np.asarray(got["outs"]["o0"], np.uint16), [int(w[0]) for w in got["outs"]["o1"]]
    if not np.array_equal(h, table[toks]) or ok != [1] * n:
        raise AssertionError(f"{tag}: the circuit's reads are not the table's rows on "
                             f"{int((h != table[toks]).any(1).sum())} instances, verdicts {sorted(set(ok))}")
    meta = got["meta"]
    return {"unit": name, "statics": S, "tag": tag, "n": n, "definition": fn.id,
            "in_rows": [[r, s, b, ps] for r, s, b, ps in in_rows], "public_inputs": [], "packed": False, "ands": got["ands"],
            "statement": {"layout": READ_LAYOUT, "structural": True, "id": st.meta.get("statement")}, "nets": nets(st.low),
            "unit_nnz": got["unit_nnz"], "k_log": meta["k_log"], "unit_log": meta.get("unit_log"),
            "vus_per_block": meta.get("vus_per_block"), "row_schemas": meta.get("row_schemas"),
            "circuit_bytes": (out / "circuit.txt").stat().st_size, "circuit_sha512": st.pin, "class_sha512": st.class_sha512,
            "inst_bytes": (out / "inst.bin").stat().st_size, "pub_bytes": (out / "pub.bin").stat().st_size, "macs": 0,
            "twin": {"agrees": True, "instances": n, "first": int(h[0, 0])},
            "seconds": {"rows": round(t_rows, 1), "built": round(t_built - t_rows, 1),
                        "round": round(time.time() - t0 - t_built, 1), "total": round(time.time() - t0, 1)},
            "maxrss_gb": gb()}


def public_words(d, row: np.ndarray, params: tuple[str, ...], packed: bool = False) -> dict[str, np.ndarray]:
    """Each of ``params``, read in order from the row words ``row`` ``(n, words)``, as a public input port: its leaves each in
    their own ceil(bits / 16) u16 words, low first (`class_statement.unit_lowering`'s public layout), or with ``packed`` the
    row's own words that hold them."""
    types = dict(d.signature()[0])
    bits = np.unpackbits(np.ascontiguousarray(row, dtype="<u2").view(np.uint8), axis=1, bitorder="little")
    out, at = {}, 0
    for p in params:
        ws = list(types[p].leaf_widths())
        w = ws[0]
        if packed:
            if at % 16 or sum(ws) % 16:
                raise SystemExit(f"stage.py: --packed takes parameters of whole u16 words; {p} is {sum(ws)} bits at bit {at}")
            out[p] = np.ascontiguousarray(row[:, at // 16:(at + sum(ws)) // 16], dtype=np.uint16)
            at += sum(ws)
            continue
        if any(x != w for x in ws):
            raise SystemExit(f"stage.py: --public-weights takes parameters of one leaf width; {p} has {sorted(set(ws))}")
        pad = np.zeros((len(row), len(ws), -(-w // 16) * 16), np.uint8)
        pad[..., :w] = bits[:, at:at + w * len(ws)].reshape(len(row), len(ws), w)
        out[p] = np.packbits(pad.reshape(len(row), -1), axis=1, bitorder="little").view("<u2").copy()
        at += w * len(ws)
    if at != bits.shape[1]:
        raise SystemExit(f"stage.py: the row's {bits.shape[1]} bits are not its parameters' {at}")
    return out


def main() -> None:
    args = [a for a in sys.argv[1:] if a not in ("--public-weights", "--packed")]
    public, packed = "--public-weights" in sys.argv, "--packed" in sys.argv
    if any(a.startswith("--") for a in args) or len(args) != 3:
        raise SystemExit(f"stage.py: usage: stage.py UNIT N OUT [--public-weights [--packed]], not {sys.argv[1:]}")
    if packed and not public:
        raise SystemExit("stage.py: --packed lays out --public-weights' ports")
    spec, out = args[0], Path(args[2])
    n: int | str = args[1] if re.fullmatch(r"full(/[1-9]\d*)?", args[1]) else int(args[1])
    t0 = time.time()
    name, u, S = parse(spec)
    if isinstance(n, str) and not isinstance(u, Census):
        raise SystemExit(f"stage.py: N {n} is for a CENSUS type, whose block's k_log sets it; {spec} takes a count")
    tag = f"{name}-" + "-".join(f"{k.lower()}{v}" for k, v in sorted(S.items()))
    if name == "census":
        tag += hashlib.sha256(u.name(S).encode()).hexdigest()[:16]
    if isinstance(u, (Part, Census, Read)):
        if public:
            raise SystemExit(f"stage.py: --public-weights is for a type of UNITS, not {name}")
        rec = (stage_part(name, u, S, tag, n, out, t0) if isinstance(u, Part)
               else stage_census(name, u, S, tag, n, out, t0) if isinstance(u, Census)
               else stage_read(name, S, tag, n, out, t0))
        (out / "pin.txt").write_text(rec["circuit_sha512"])
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)
        return
    d = u.definition(S)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    rows, twin = u.values(S, n, rng)
    t_values = time.time() - t0
    unit_rows, ports = u.rows, None
    if public:
        (w,) = [r for r in u.rows if r[0] == "w"]
        unit_rows = tuple(r for r in u.rows if r[0] != "w")
        rows = {**{r: rows[r] for r, *_ in unit_rows}, **public_words(d, rows["w"], w[2], packed)}
        ports = {p: {"seed": hashlib.sha256(f"verity/served-units/weights\0{tag}".encode()).hexdigest(),
                     "derivation": "served-units/public-weights"} for p in w[2]}
        tag += "-packed-weights" if packed else "-public-weights"
    in_rows = [(r, schema, 16 * rows[r].shape[1], list(ps)) for r, schema, ps in unit_rows]
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    got = CS.stage_unit(d, f"unit/served-{tag}", in_rows, [(V2, (0,))], rows, out, salt_key, public_inputs=ports, packed=packed)
    o0 = np.asarray(got["outs"]["o0"], dtype=np.uint16)
    if not np.array_equal(o0[:, :twin.shape[1]], twin):
        raise AssertionError(f"{tag}: the circuit's outputs are not the served twin's on {int((o0[:, :twin.shape[1]] != twin).any(1).sum())} instances")
    text = (out / "circuit.txt").read_text()
    pin = FC.digest(text)
    (out / "pin.txt").write_text(pin)
    meta = got["meta"]
    rec = {"unit": name, "statics": S, "tag": tag, "n": n, "definition": d.id,
           "in_rows": [[r, s, b] for r, s, b, _ in in_rows], "public_inputs": sorted(ports or {}), "packed": packed, "ands": got["ands"],
           "unit_nnz": got["unit_nnz"],
           "k_log": meta["k_log"], "unit_log": meta.get("unit_log"), "vus_per_block": meta.get("vus_per_block"),
           "row_schemas": meta.get("row_schemas"), "circuit_bytes": len(text), "circuit_sha512": pin,
           "class_sha512": FC.class_digest(text), "inst_bytes": (out / "inst.bin").stat().st_size,
           "pub_bytes": (out / "pub.bin").stat().st_size, "twin": {"agrees": True, "instances": n, "first": int(twin[0, 0])},
           "seconds": {"values": round(t_values, 1), "total": round(time.time() - t0, 1)}, "maxrss_gb": gb()}
    (out / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
