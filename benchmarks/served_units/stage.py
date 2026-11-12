"""stage.py UNIT N OUT [--public-weights [--packed]]: N inputs of one served proof-block type's Q_word proof block, from synthetic
values at the model's served statics, staged for Flock as a ``verity/flock-circuit`` statement with hidden outputs.

    UNIT  TYPE[:S=V,...]   a type of `UNITS`, `PARTS` or `CENSUS` at its statics (each type's defaults are its model's: GLM-5.3's
                           at TP8, s8-coord's Llama-3.2-1B's), e.g. fp8-coord:K=2048, router:C=1, s8-coord:K=8192, gq:K=384
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

A type of `CENSUS` is a block type of the serving rows by its census name (`serving_templates.template`, the name a proof
overhead card prices it under), one whole Definition a proof block, laid out as the served composer lays its template
(`compose_circuit.py --template`): its served rows (`chunk_rows.unit_rows`, each parameter's leaves seeded at their width,
`census_leaves`) and statement (`chunk_rows.statement`: the 64-key sparse MLA split part's MUFU reads in Flock's lookup
slots, its unit cut into stage nets under the automatic caps, which keep it within the GPU prover's slot types; RMSNorm's
and the FP8 linears' inline) under its unit name, so the statement is the served one but for its binding: its
``class_sha512`` is the served composer's. The circuit is streamed to circuit.txt once, every input's outputs checked against
the Program's reference (``twin`` null).

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
    fi          `CENSUS`: FlashInferFp8Linear_v1{K,N}, the FlashInfer FP8 linear's output coordinate (N = 1) on one token row:
                its bf16 x quantized in the block, an E4M3 weight row and its f32 block scales (K = 2048: q_b and o_proj).
    mla-split-part  `CENSUS`: SparseMlaSplitPart_v1 at T = 64, one query's sparse MLA over one split's 64 keys (FA3's).
    rmsnorm-v8  `CENSUS`: RMSNormFusedCudaVector8_v1{N=6144,BLOCK=256}, the fused residual add and RMSNorm of one token row.
    the rest    `PARTS`: the capture map's Programs (``glm-moe-dsa-eager-hopper-tp8-v0``) at their GLM-5.3 statics; the MoE
                tiles' KB and SWAP are Triton's 128 and swapped until the H200 capture pins them."""
from __future__ import annotations

import hashlib
import json
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
from verity.ml.definitions.registry import act, b1, dsa, fp8_moe, fp8_moe_experts, moe, ref_prims, sparse_mla
from verity.ml.definitions.registry import boolean_fp8 as BF


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
    each row's words per input ``(n, words)`` and the served twin's output words ``(n, out_words)``."""

    definition: Callable[[dict], object]
    rows: tuple[tuple[str, str, tuple[str, ...]], ...]
    draw: Callable[[dict, int, np.random.Generator], tuple[dict[str, np.ndarray], np.ndarray]]
    defaults: dict


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


UNITS: dict[str, Unit] = {
    "fp8-coord": Unit(lambda S: bind(BF.ScaledMmFp8BlockCoordinateV2, K=S["K"], G=S["G"]),
                      (("x", SEG, ("xq", "sx")), ("w", V2, ("wrow", "sw"))), _fp8_coord_draw, {"K": 6144, "G": 128}),
    "s8-coord": Unit(lambda S: bind(BG.ScaledMmS8Coordinate, K=S["K"], **BG.GEMM_SM120_S8),
                     (("x", SEG, ("x", "sx")), ("w", V2, ("wrow", "sw"))), _s8_coord_draw, {"K": 2048}),
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
        return f"{self.family}{{{','.join([*(f'{k}={v}' for k, v in S.items()), *filter(None, [self.fixed])])}}}"


_EPS = '{"f64":"0x1.4f8b588e368f1p-17"}'
CENSUS: dict[str, Census] = {
    "gq": Census("Fp8GroupQuantUe8m0_v1", {"K": 6144, "G": 128}),
    "fi": Census("FlashInferFp8Linear_v1", {"K": 2048, "N": 1}),
    "mla-split-part": Census("SparseMlaSplitPart_v1", {}, 'T=64,DR=64,DV=512,BN=64,DOT={"fn":"HopperBF16WgmmaDot16_v1"},'
                             'INV={"fn":"Fa3InvSum_v1"},SCALE={"f64":"0x1.0000000000000p-4"}'),
    "rmsnorm-v8": Census("RMSNormFusedCudaVector8_v1", {}, f"N=6144,EPS={_EPS},BLOCK=256"),
}


def census_leaves(rng: np.random.Generator, ty, n: int) -> np.ndarray:
    """``n`` rows of a parameter of type ``ty`` as u16 words, its leaves packed at their width: bf16 of N(0, 1) for 16-bit
    leaves, E4M3 codes other than NaN for 8-bit ones, f32 powers of two from 2^-10 to 2^-4 (a block scale) for 32-bit ones."""
    w = set(ty.leaf_widths())
    if w == {16}:
        return bf16(rng.standard_normal((n, ty.leaves)))
    if w == {8} and ty.leaves % 2 == 0:
        c = rng.integers(0, 256, (n, ty.leaves), dtype=np.uint8)
        c[(c & 0x7F) == 0x7F] ^= 1
        return c.view("<u2")
    if w == {32}:
        return np.ldexp(np.ones((n, ty.leaves), np.float32), rng.integers(-10, -3, (n, ty.leaves))).astype("<f4").view("<u2")
    raise SystemExit(f"stage.py: a census parameter of {ty.leaves} leaves of {sorted(w)} bits, which `census_leaves` doesn't draw")


def _static(default, v: str):
    if isinstance(default, bool):
        return v.lower() in ("1", "true")
    return type(default)(v)


def parse(spec: str) -> tuple[str, Unit | Part | Census, dict]:
    name, _, rest = spec.partition(":")
    if name not in UNITS and name not in PARTS and name not in CENSUS:
        raise SystemExit(f"stage.py: unit type {name!r} is not one of {sorted(UNITS) + sorted(PARTS) + sorted(CENSUS)}")
    u = UNITS.get(name) or PARTS.get(name) or CENSUS[name]
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


def nets(low) -> list[dict]:
    """Each net's ANDs, rows and matrix entries a side: the unit's, then under read slots each tail stage's (the circuit's
    ``unit``, ``stage0``, ...; a lookup slot's table counts its entries once per statement, in the prover)."""
    stages = getattr(getattr(low, "tail", None), "stages", ())
    lays = [("unit", low.layout), *((f"stage{i}", st.layout) for i, st in enumerate(stages))]
    return [{"net": nm, **{k: int(lay.counts[k]) for k in ("ands", "rows", "nnz_a", "nnz_b")}} for nm, lay in lays]


def stage_census(name: str, cen: Census, S: dict, tag: str, n: int, out: Path, t0: float) -> dict:
    """``n`` inputs of the census type's Definition, a proof block each, on its served rows and statement, as stage.json's
    record."""
    from verity_vllm.pipeline import chunk_rows as CR
    from verity_vllm.pipeline.serving_rows import template

    tp = template(cen.name(S))
    d = tp.fn
    params, _ret = d.signature()
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    vals = {p: census_leaves(rng, ty, n) for p, ty in params}
    in_rows, out_rows = CR.unit_rows(tp)
    if any(p not in vals for *_r, ps in in_rows for p in ps):
        raise SystemExit(f"stage.py: {tp.id}'s served rows {in_rows} cut a parameter, which a census type draws whole")
    rows = {r: np.concatenate([vals[p] for p in ps], axis=1) for r, _s, _b, ps in in_rows}
    if any(rows[r].shape != (n, b // 16) for r, _s, b, _ps in in_rows):
        raise SystemExit(f"stage.py: {tp.id}'s drawn rows {[rows[r].shape for r, *_x in in_rows]} are not its rows' words")
    stmt = CR.statement(tp)
    t_draw = time.time() - t0
    st = CS.UnitStager(d, "unit/serving-rows", in_rows, out_rows, circuit=out / "circuit.txt", **stmt)
    t_built = time.time() - t0
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    got = st.stage(rows, out, salt_key)
    if stmt.get("reads") == "slots":
        FC.IL.write_tables(out / "tables")
    meta = got["meta"]
    return {"unit": name, "statics": S, "tag": tag, "n": n, "definition": d.id, "census": cen.name(S),
            "in_rows": [[r, s, b, ps] for r, s, b, ps in in_rows], "public_inputs": [], "packed": False, "ands": got["ands"],
            "statement": stmt, "caps": getattr(st.low, "caps", None), "nets": nets(st.low), "lookups": meta.get("lookups"),
            "slot_types": len({r["net"] for r in meta.get("ranges", ())}),
            "unit_nnz": got["unit_nnz"], "k_log": meta["k_log"], "unit_log": meta.get("unit_log"),
            "vus_per_block": meta.get("vus_per_block"), "row_schemas": meta.get("row_schemas"),
            "circuit_bytes": (out / "circuit.txt").stat().st_size, "circuit_sha512": st.pin, "class_sha512": st.class_sha512,
            "inst_bytes": (out / "inst.bin").stat().st_size, "pub_bytes": (out / "pub.bin").stat().st_size, "macs": 0,
            "twin": None, "seconds": {"draw": round(t_draw, 1), "built": round(t_built - t_draw, 1),
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
    spec, n, out = args[0], int(args[1]), Path(args[2])
    t0 = time.time()
    name, u, S = parse(spec)
    tag = f"{name}-" + "-".join(f"{k.lower()}{v}" for k, v in sorted(S.items()))
    if isinstance(u, (Part, Census)):
        if public:
            raise SystemExit(f"stage.py: --public-weights is for a type of UNITS, not {name}")
        rec = (stage_part(name, u, S, tag, n, out, t0) if isinstance(u, Part)
               else stage_census(name, u, S, tag, n, out, t0))
        (out / "pin.txt").write_text(rec["circuit_sha512"])
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)
        return
    d = u.definition(S)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    rows, twin = u.draw(S, n, rng)
    t_draw = time.time() - t0
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
           "seconds": {"draw": round(t_draw, 1), "total": round(time.time() - t0, 1)}, "maxrss_gb": gb()}
    (out / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
