"""stage.py UNIT N OUT [--public-weights [--packed]]: N inputs of one served proof-block type's Q_word proof block, from synthetic
values at the model's served statics, staged for Flock as a ``verity/flock-circuit`` statement with hidden outputs.

    UNIT  TYPE[:S=V,...]   a type of `UNITS` or `PARTS` at its statics (each type's defaults are its model's: GLM-5.3's at TP8,
                           s8-coord's Llama-3.2-1B's), e.g. fp8-coord:K=2048, router:C=1, s8-coord:K=8192
    N     N1,N2,...        with a type of `UNITS`, several statements on one built circuit, each count's in OUT's ``{n}``
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
``verity/partition/v1`` digest, so the verifier is given its own copy of the partition object (partition.json,
``flock-verify --partition``); not the program, from which it derives a statement's proof blocks only where each is one whole
activation of the circuit's template (``Extract.claimedUnits``), and a class here is part of one, on synthetic Calls.

Writes circuit.txt, inst.bin, pub.bin, pin.txt and stage.json under OUT, and a PARTS type's partition.json.

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
    router-slots  the MoE cut per expert's blocks, each a whole Definition on the window interface's rows (`interface.unit_rows`:
    read          a ``row/v2`` row a parameter and one a return field), as `glm53-moe-per-expert` holds them:
    pe-w13        `MoeRouterSigmoidBiasSlots_v2`; `EmbeddingRead_v1{V, H}` opening a table of V = 256 random rows, one an
    pe-w2         expert, at a random tok (H = 3072 and 96 are w13's row and scale reads, 4096 and 4 w2's; each staged read is
                  checked to open, ok = 1 and h the registered row); `MoeExpertFp8BlockGemm_v4{K=6144,N=1}` (w13, mma.sync) and
                  `MoeExpertFp8BlockGemmW_v4{K=256,N=32}` (w2, wgmma) on random e4m3 operands and scales.
    the rest    `PARTS`: the capture map's Programs (``glm-moe-dsa-eager-hopper-tp8-v0``) at their GLM-5.3 statics; the MoE
                tiles' KB and SWAP are Triton's 128 and swapped until the H200 capture pins them."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from circuit_check import targets as TG
from verity.ml.circuits import partition_object as PO
from verity.ml.circuits.codec import program_digest
from verity.ml.circuits.defs import bind
from verity.ml.circuits.ops import gemm as OG
from verity.ml.circuits.ops.boolean import gemm as BG
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU
from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.circuits.ops.prims import HopperE4m3QgmmaDot32
from verity.ml.commitments import row_table as RT
from verity.ml.commitments.gates import embedding as EMB
from verity.ml.commitments.indexed import RangeIndexedDomain
from verity.ml.commitments.merkle import CommitmentDomain
from verity.ml.commitments.rowleaf import bit_row
from verity.ml.definitions.registry import act, b1, dsa, fp8_moe, fp8_moe_experts, moe, ref_prims, sparse_mla
from verity.ml.definitions.registry import boolean_fp8 as BF
from verity.ml.definitions.registry import boolean_fp8_moe as BM
from verity.ml.definitions.registry import boolean_moe as BMOE
from verity.ml.windows import interface as WI


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
    """A proof-block type: its Boolean Definition at statics ``S``, its input rows ``(row, schema, params)``, and ``draw(S, n, rng)``:
    each row's words per input ``(n, words)`` and the served twin's output words ``(n, out_words)``. A type with no ``rows``
    takes the window interface's (`interface.unit_rows`), and its draw's second value maps output rows to the words each
    must hold (None: no twin); ``macs(S)`` is a block's multiply-adds where they aren't one coordinate's K."""

    definition: Callable[[dict], object]
    rows: tuple[tuple[str, str, tuple[str, ...]], ...]
    draw: Callable[[dict, int, np.random.Generator], tuple[dict[str, np.ndarray], object]]
    defaults: dict
    macs: Callable[[dict], int] | None = None
    io: Callable[[object], tuple[list, list]] | None = None


def _fp8_coord_draw(S: dict, n: int, rng: np.random.Generator):
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


def _s8_coord_draw(S: dict, n: int, rng: np.random.Generator):
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


def _bits_words(bits) -> np.ndarray:
    """A parameter's flat bits in leaf order as its row's u16 words (bit 0 the first word's low bit)."""
    return np.packbits(np.asarray(bits, np.uint8), bitorder="little").view("<u2")


def _read_draw(S: dict, n: int, rng: np.random.Generator):
    """A table of V random rows of H words, registered one a row (`row_table.commit` under a dense domain, owner -2), opened
    at a random tok per input."""
    V, H = S["V"], S["H"]
    table = rng.integers(0, 1 << 16, (V, H), dtype=np.uint16)
    dom = CommitmentDomain(rng.bytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    salts = [rng.bytes(EMB.SALT_BYTES) for _ in range(V)]
    tree = RT.commit(dom, [bit_row((r.tolist(), 16)) for r in table], salts)
    names = [p for p, _ty in EMB.read(V, H).signature()[0]]
    rows: dict[str, list] = {p: [] for p in names}
    toks = rng.integers(0, V, n)
    for t in toks:
        p = tree.path(int(t))
        args = EMB.arguments(V, int(t), tree.root, dom.domain_id, table[t].tolist(), salts[t], [p[k:k + 64] for k in range(0, len(p), 64)])
        for name, bits in zip(names, args):
            rows[name].append(_bits_words(bits))
    return {p: np.stack(v) for p, v in rows.items()}, {"o0": table[toks], "o1": np.ones((n, 1), np.uint16)}


def _log_f32(rng: np.random.Generator, lo: float, hi: float, n: int) -> np.ndarray:
    return np.exp(rng.uniform(np.log(lo), np.log(hi), n)).astype(np.float32)


def _template_draw(S: dict, n: int, rng: np.random.Generator, fn, skip: tuple = ()) -> dict[str, np.ndarray]:
    """Random words for each of a per-expert GEMM's or the router's parameters by name (but ``skip``): e4m3 operands (no NaN
    code), log-uniform f32 scales, a routed weight in (0, 1), N(0, 2) logits and N(0, 0.01) biases."""
    out = {}
    for p, w in WI.layout(fn).rows:
        if p in skip:
            continue
        if p in ("xq", "w"):
            b = rng.integers(0, 256, (n, 2 * w), dtype=np.uint8)
            b[(b & 0x7F) == 0x7F] ^= 1
            v = b
        elif p in ("sx", "ws"):
            v = _log_f32(rng, *((1e-3, 1e-1) if p == "sx" else (1e-4, 1e-2)), n * w // 2).reshape(n, w // 2)
        elif p == "g":
            v = rng.uniform(0.01, 1.0, (n, w // 2)).astype(np.float32)
        elif p in ("logits", "bias"):
            v = (rng.standard_normal((n, w // 2)) * (2.0 if p == "logits" else 1e-2)).astype(np.float32)
        else:
            raise SystemExit(f"stage.py: no draw for {fn.id}'s parameter {p!r}")
        out[p] = np.ascontiguousarray(v).view("<u2").reshape(n, w)
    return out


def _fused_draw(S: dict, n: int, rng: np.random.Generator, fn):
    """A fused read-GEMM's ports: its `READ` opening a table of 256 random e4m3 rows of N K / 2 words (one an expert, registered
    as `_read_draw` registers its table) at a random tok, then the GEMM's operands as `_template_draw` draws them. The twin
    holds only the read's verdict, ok = 1."""
    V, H = 256, S["N"] * S["K"] // 2
    b = rng.integers(0, 256, (V, 2 * H), dtype=np.uint8)
    b[(b & 0x7F) == 0x7F] ^= 1
    table = b.view("<u2")
    dom = CommitmentDomain(rng.bytes(64), -2, RangeIndexedDomain(0, V), hash="sha512")
    salts = [rng.bytes(EMB.SALT_BYTES) for _ in range(V)]
    tree = RT.commit(dom, [bit_row((r.tolist(), 16)) for r in table], salts)
    names = [p for p, _ty in EMB.read(V, H).signature()[0]]
    rows: dict[str, list] = {p: [] for p in names}
    for t in rng.integers(0, V, n):
        p = tree.path(int(t))
        args = EMB.arguments(V, int(t), tree.root, dom.domain_id, table[t].tolist(), salts[t], [p[k:k + 64] for k in range(0, len(p), 64)])
        for name, bits in zip(names, args):
            rows[name].append(_bits_words(bits))
    out = {p: np.stack(v) for p, v in rows.items()}
    return {**out, **_template_draw(S, n, rng, fn, skip=tuple(names))}, {"o1": np.ones((n, 1), np.uint16)}


def _fused(gemm, dot, defaults: dict) -> Unit:
    """A per-expert GEMM with its weight-row read fused in (`MoeExpertFp8ReadGemm(W)_v1`, READ = EmbeddingRead_v1{V=256, H=N K/2})."""
    def definition(S):
        return bind(gemm, READ=EMB.read(256, S["N"] * S["K"] // 2), **S, DOT=dot)
    return Unit(definition, (), lambda S, n, rng: _fused_draw(S, n, rng, definition(S)), defaults, lambda S: S["K"] * S["N"])


def _window(definition: Callable[[dict], object], defaults: dict, macs: Callable[[dict], int] | None = None) -> Unit:
    """A whole Definition on the window interface's rows, drawn by `_template_draw`, no twin."""
    return Unit(definition, (), lambda S, n, rng: (_template_draw(S, n, rng, definition(S)), None), defaults, macs)


_GEMM_ST = {"G": 128, "KB": 128}
UNITS: dict[str, Unit] = {
    "fp8-coord": Unit(lambda S: bind(BF.ScaledMmFp8BlockCoordinateV2, K=S["K"], G=S["G"]),
                      (("x", SEG, ("xq", "sx")), ("w", V2, ("wrow", "sw"))), _fp8_coord_draw, {"K": 6144, "G": 128}),
    "s8-coord": Unit(lambda S: bind(BG.ScaledMmS8Coordinate, K=S["K"], **BG.GEMM_SM120_S8),
                     (("x", SEG, ("x", "sx")), ("w", V2, ("wrow", "sw"))), _s8_coord_draw, {"K": 2048}),
    "router-slots": _window(lambda S: bind(BMOE.MoeRouterSigmoidBiasSlots, **S), {"E": 256, "TOPK": 8, "SCALE": 2.5, "KERNEL": "block"}),
    "read": Unit(lambda S: EMB.read(S["V"], S["H"]), (), _read_draw, {"V": 256, "H": 3072}),
    "pe-w13": _window(lambda S: bind(BM.MoeExpertFp8BlockGemmV4, **S, DOT=BG.HopperE4m3MmaV2Dot32),
                      {"K": 6144, "N": 1, **_GEMM_ST, "SWAP": False}, lambda S: S["K"] * S["N"]),
    "pe-w2": _window(lambda S: bind(BM.MoeExpertFp8BlockGemmWV4, **S, DOT=BG.HopperE4m3QgmmaDot32),
                     {"K": 256, "N": 32, **_GEMM_ST, "SWAP": True}, lambda S: S["K"] * S["N"]),
    "pe-fw13": _fused(BM.MoeExpertFp8ReadGemmV1, BG.HopperE4m3MmaV2Dot32, {"K": 6144, "N": 1, **_GEMM_ST, "SWAP": False}),
    "pe-fw2": _fused(BM.MoeExpertFp8ReadGemmWV1, BG.HopperE4m3QgmmaDot32, {"K": 256, "N": 32, **_GEMM_ST, "SWAP": True}),
}

_EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
_WGMMA = '{"fn":"HopperBF16WgmmaDot16_v1"}'
_TREE = "[[[[[[0,[1,2]],3],5],6],4],7]"
#: registered types the census names by descriptor id and no type above stages, each with its multiply-adds a block
CENSUS: dict[str, tuple[str, int]] = {
    "ar-multimem": (f"AllReduceMultimem_v1{{WORLD=8,SIZE=3072,TREES=[{','.join([_TREE] * 8)}],FTZ=false}}", 0),
    "mla-split-part": ('SparseMlaSplitPart_v1{T=64,DR=64,DV=512,BN=64,DOT=' + _WGMMA
                       + ',INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}}', 69632),
    "gq-k6144": ("Fp8GroupQuantUe8m0_v1{K=6144,G=128}", 0),
    "residual-add": ("ResidualAdd_v1{N=6144}", 0),
    "gq-k2048": ("Fp8GroupQuantUe8m0_v1{K=2048,G=128}", 0),
    "gq-k1536": ("Fp8GroupQuantUe8m0_v1{K=1536,G=128}", 0),
    "fi-linear-k2048": ("FlashInferFp8Linear_v1{K=2048,N=1}", 2048),
    "dsa-norm-n2048": (f"DsaRMSNorm_v1{{N=2048,EPS={_EPS}}}", 0),
    "mla-combine": ('SparseMlaCombine_v1{NSPLIT=5,DV=512,INV={"fn":"Fa3InvSum_v1"}}', 0),
    "ar-residual-mnnvl": ("AllReduceResidualMnnvl_v1{WORLD=8,N=3072,SHOT=1}", 0),
    "fi-linear-k6144": ("FlashInferFp8Linear_v1{K=6144,N=1}", 6144),
    "dsa-norm-n512": (f"DsaRMSNorm_v1{{N=512,EPS={_EPS}}}", 0),
    "rmsnorm-mnnvl": (f"RmsNormMnnvl_v1{{N=6144,SHOT=1,CLUSTER=8,EPS={_EPS},FMA=true}}", 0),
    "cublas-splitk-f32": (f"CublasSplitKGemmF32Out_v1{{K=6144,N=1,SPLIT_K=3,DOT={_WGMMA}}}", 6144),
    "bgemm-k512": (f"BatchedGemm_v1{{B=1,K=512,N=1,DOT={_WGMMA}}}", 512),
    "sm-fp8-k1536": ("ScaledMmFp8BlockSharedScale_v1{K=1536,N=1,G=128}", 1536),
    "fi-linear-k256": ("FlashInferFp8Linear_v1{K=256,N=1}", 256),
    "gq-k256": ("Fp8GroupQuantUe8m0_v1{K=256,G=128}", 0),
}


def _census_fn(spec: str):
    from verity_vllm.pipeline import serving_rows as SRW

    return SRW.template(spec).fn


def _census_io(fn) -> tuple[list, list]:
    """A V2 row per parameter and per output, as `one_stage/m1_unit.py` stages a census row."""
    params, ret = fn.signature()
    n_out = len(ret.fields) if hasattr(ret, "fields") else 1
    return [(p, V2, sum(t.leaf_widths()), [p]) for p, t in params], [(V2, (j,)) for j in range(n_out)]


def _census(spec: str, macs: int) -> Unit:
    """A census type whole, on N(0, 1) words rounded to bf16 by truncation in every row (`m1_unit.bf16_rows`), no twin."""
    def draw(S, n, rng):
        return {p: (rng.standard_normal((n, b // 16)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)
                for p, _s, b, _ps in _census_io(_census_fn(spec))[0]}, None
    return Unit(lambda S: _census_fn(spec), (), draw, {}, lambda S: macs, _census_io)


UNITS.update({name: _census(spec, macs) for name, (spec, macs) in CENSUS.items()})


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


def _static(default, v: str):
    if isinstance(default, bool):
        return v.lower() in ("1", "true")
    return type(default)(v)


def parse(spec: str) -> tuple[str, Unit | Part, dict]:
    name, _, rest = spec.partition(":")
    if name not in UNITS and name not in PARTS:
        raise SystemExit(f"stage.py: unit type {name!r} is not one of {sorted(UNITS) + sorted(PARTS)}")
    u = UNITS.get(name) or PARTS[name]
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
    t_draw = time.time() - t0
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
            "seconds": {"draw": round(t_draw, 1), "total": round(time.time() - t0, 1)}, "maxrss_gb": gb()}


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
    if packed and not public:
        raise SystemExit("stage.py: --packed lays out --public-weights' ports")
    spec, ns = args[0], [int(x) for x in args[1].split(",")]
    if len(ns) > 1 and "{n}" not in args[2]:
        raise SystemExit("stage.py: several counts take an OUT naming each count's directory with {n}")
    outs = [Path(args[2].format(n=n)) for n in ns]
    n, out = ns[0], outs[0]
    t0 = time.time()
    name, u, S = parse(spec)
    tag = f"{name}-" + "-".join(f"{k.lower()}{v}" for k, v in sorted(S.items()))
    if isinstance(u, Part):
        if public:
            raise SystemExit(f"stage.py: --public-weights is for a type of UNITS, not {name}")
        if len(ns) > 1:
            raise SystemExit(f"stage.py: {name} is a class of PARTS, staged one count a run")
        rec = stage_part(name, u, S, tag, n, out, t0)
        (out / "pin.txt").write_text(rec["circuit_sha512"])
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)
        return
    d = u.definition(S)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    rows, twin = u.draw(S, max(ns), rng)
    t_draw = time.time() - t0
    unit_rows, ports = u.rows, None
    if public and not u.rows:
        raise SystemExit(f"stage.py: --public-weights is for a type with a weight row, not {name}")
    if public:
        (w,) = [r for r in u.rows if r[0] == "w"]
        unit_rows = tuple(r for r in u.rows if r[0] != "w")
        rows = {**{r: rows[r] for r, *_ in unit_rows}, **public_words(d, rows["w"], w[2], packed)}
        ports = {p: {"seed": hashlib.sha256(f"verity/served-units/weights\0{tag}".encode()).hexdigest(),
                     "derivation": "served-units/public-weights"} for p in w[2]}
        tag += "-packed-weights" if packed else "-public-weights"
    if unit_rows:
        in_rows, out_rows = [(r, schema, 16 * rows[r].shape[1], list(ps)) for r, schema, ps in unit_rows], [(V2, (0,))]
        twin = {"o0": twin}
    else:
        in_rows, out_rows = (u.io or WI.unit_rows)(d)
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    if name in CENSUS:
        gc.disable()
    circuit = outs[0] / "circuit.txt"
    stager = CS.UnitStager(d, f"unit/served-{tag}", in_rows, out_rows, ports, circuit=circuit, packed=packed)
    t_built = time.time() - t0
    for n, out in zip(ns, outs):
        t1 = time.time()
        got = stager.stage({r: v[:n] for r, v in rows.items()}, out, salt_key)
        for port, want in (twin or {}).items():
            have = np.asarray(got["outs"][port], dtype=np.uint16)[:, :want.shape[1]]
            if not np.array_equal(have, want[:n]):
                raise AssertionError(f"{tag}: the circuit's {port} is not the twin's on {int((have != want[:n]).any(1).sum())} instances")
        if out != outs[0]:
            (out / "circuit.txt").unlink(missing_ok=True)
            os.link(circuit, out / "circuit.txt")
        pin = stager.pin
        (out / "pin.txt").write_text(pin)
        meta = got["meta"]
        rec = {"unit": name, "statics": S, "tag": tag, "n": n, "definition": d.id,
               "in_rows": [[r, s, b] for r, s, b, _ in in_rows], "public_inputs": sorted(ports or {}), "packed": packed,
               "ands": got["ands"], "unit_nnz": got["unit_nnz"],
               "k_log": meta["k_log"], "unit_log": meta.get("unit_log"), "vus_per_block": meta.get("vus_per_block"),
               "row_schemas": meta.get("row_schemas"), "circuit_bytes": circuit.stat().st_size, "circuit_sha512": pin,
               "class_sha512": stager.class_sha512, "inst_bytes": (out / "inst.bin").stat().st_size,
               "pub_bytes": (out / "pub.bin").stat().st_size, "macs": u.macs(S) if u.macs else None,
               "twin": twin and {"agrees": True, "instances": n, "ports": sorted(twin),
                                 "first": int(next(iter(twin.values()))[0, 0])},
               "seconds": {"draw": round(t_draw, 1), "built": round(t_built, 1), "round": round(time.time() - t1, 1),
                           "total": round(t_built + time.time() - t1, 1)}, "maxrss_gb": gb()}
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
