"""stage.py UNIT N OUT: N instances of one served unit type's Q_word unit, from synthetic values at the model's served statics,
staged for C-Flock as a ``verity/flock-circuit`` statement with hidden outputs.

    UNIT  TYPE[:S=V,...]   a type of `UNITS` or `PARTS` at its statics (each type's defaults are GLM-5.3's at TP8), e.g.
                           fp8-coord:K=2048, router:C=1

A type of `UNITS` is a unit Definition, staged by `class_statement.stage_unit`: its Boolean Definition lowered once, its input
rows hashed in the circuit (the per-call row on ``hm96-sha512/row-seg/v1``, the weight row on ``hm96-sha512/row/v2``, PoUW's
convention), its output word one ``row/v2`` output row, every instance's output checked by `stage_unit` against the IR
reference of the Boolean Definition and here against the served kernel's numpy twin on the same values (``twin``).

A type of `PARTS` is a word Definition whose Q_word units are parts of it: the Definition as a one-Call Program cut by
`Q_word` v1 (X = 16, W = 32), its classes ranked by word gates, and the class of rank ``C`` (static ``C``, default 0: the
largest) staged by `class_statement.stage` with every instance's own ``sha512/row/v1`` rows (``share_rows=False``): the class
lowered once (C-Flock's pieces, and for a primitive without one the registered Boolean Definition's builder), its lanes the
flat Definition evaluated on uniform synthetic parameters (`class_statement.class_lanes`), the laid-out unit checked against
them on every instance (``twin`` null: no served twin cuts the same unit).  The circuit binds the partition's
``verity/partition/v1`` digest, so the verifier is given its own copy of the partition object (partition.json,
``flock-verify --partition``); not the program, from which it derives a statement's units only where each is one whole
activation of the circuit's template (``Extract.claimedUnits``), and a class here is part of one, on synthetic Calls.

Writes circuit.txt, inst.bin, pub.bin, pin.txt and stage.json under OUT, and a PARTS type's partition.json.

Types (GLM-5.3's statics from its config: hidden 6144, q_lora 2048, kv_lora 512, moe_intermediate 2048, 256 experts, top 8,
FP8 blocks of 128, DSA indexer 32 heads of 128, index_topk 2048; per TP8 rank):

    fp8-coord   ScaledMmFp8BlockCoordinate_v2{K,G}: one output coordinate of a block-scaled FP8 linear (CUTLASS blockwise,
                `ScaledMmFp8Block_v1{K,N,128}`'s Q_word unit), x = Fp8GroupQuant of a seeded N(0, 1) bf16 row, w = a seeded
                N(0, 0.02) bf16 weight row quantized per 128 the same way.  K = 6144 is fused_qkv_a and the dense and
                shared experts' gate_up; 2048 is q_b, the indexer's wq_b and o_proj (one TP8 rank); 256 is the shared
                expert's down_proj on one rank.  The routed experts run Triton's fused MoE (`moe-w13`, `moe-w2`).
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
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_flock import partition_units as PU
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.registry import act, b1, dsa, fp8_moe, moe, ref_prims, sparse_mla
from verity_vllm.program.registry import boolean_fp8 as BF


def bf16(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def words(*parts: np.ndarray) -> np.ndarray:
    """A row's u16 words: the parts' little-endian bytes concatenated (a Boolean element's bit 0 is its word's low bit)."""
    raw = b"".join(np.ascontiguousarray(p).astype(p.dtype.newbyteorder("<")).tobytes() for p in parts)
    return np.frombuffer(raw, dtype="<u2").copy()


@dataclass(frozen=True)
class Unit:
    """A unit type: its Boolean Definition at statics ``S``, its input rows ``(row, schema, params)``, and ``draw(S, n, rng)``:
    each row's words per instance ``(n, words)`` and the served twin's output words ``(n, out_words)``."""

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


UNITS: dict[str, Unit] = {
    "fp8-coord": Unit(lambda S: bind(BF.ScaledMmFp8BlockCoordinateV2, K=S["K"], G=S["G"]),
                      (("x", SEG, ("xq", "sx")), ("w", V2, ("wrow", "sw"))), _fp8_coord_draw, {"K": 6144, "G": 128}),
}


@dataclass(frozen=True)
class Part:
    """A unit type cut from a word Definition: the family, bound at statics ``S`` (``C``, the class's rank, is not one)."""

    family: object
    defaults: dict
    fixed: tuple = ()

    def definition(self, S: dict):
        return bind(self.family, **{k: v for k, v in S.items() if k != "C"}, **dict(self.fixed))


_MLA = tuple((k, v) for k, v in sparse_mla.SPARSE_MLA_GLM53.items() if k not in ("NH", "DR", "DV", "BN", "SCALE"))
PARTS: dict[str, Part] = {
    "router": Part(moe.MoeRouterSigmoidBias, {"E": 256, "TOPK": 8, "SCALE": 2.5, "KERNEL": "block"}),
    "moe-w13": Part(fp8_moe.MoeExpertFp8BlockGemm, {"K": 6144, "N": 2, "G": 128, "KB": 128, "SWAP": True}),
    "moe-w2": Part(fp8_moe.MoeExpertFp8BlockGemmW, {"K": 256, "N": 2, "G": 128, "KB": 128, "SWAP": True}),
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
    """The class of rank ``S["C"]`` of the type's word Definition under `Q_word`, ``n`` instances, as stage.json's record."""
    TG.load_registries()
    CS.cache_slot_circuits()
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


def main() -> None:
    spec, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    t0 = time.time()
    name, u, S = parse(spec)
    tag = f"{name}-" + "-".join(f"{k.lower()}{v}" for k, v in sorted(S.items()))
    if isinstance(u, Part):
        rec = stage_part(name, u, S, tag, n, out, t0)
        (out / "pin.txt").write_text(rec["circuit_sha512"])
        (out / "stage.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec), flush=True)
        return
    d = u.definition(S)
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f"verity/served-units/v1\0{tag}".encode()).digest()[:8], "little"))
    rows, twin = u.draw(S, n, rng)
    t_draw = time.time() - t0
    in_rows = [(r, schema, 16 * rows[r].shape[1], list(ps)) for r, schema, ps in u.rows]
    salt_key = hashlib.sha256(f"verity/served-units/salts\0{tag}".encode()).digest()
    got = CS.stage_unit(d, f"unit/served-{tag}", in_rows, [(V2, (0,))], rows, out, salt_key)
    o0 = np.asarray(got["outs"]["o0"], dtype=np.uint16)
    if not np.array_equal(o0[:, :twin.shape[1]], twin):
        raise AssertionError(f"{tag}: the circuit's outputs are not the served twin's on {int((o0[:, :twin.shape[1]] != twin).any(1).sum())} instances")
    text = (out / "circuit.txt").read_text()
    pin = FC.digest(text)
    (out / "pin.txt").write_text(pin)
    meta = got["meta"]
    rec = {"unit": name, "statics": S, "tag": tag, "n": n, "definition": d.id,
           "in_rows": [[r, s, b] for r, s, b, _ in in_rows], "ands": got["ands"], "unit_nnz": got["unit_nnz"],
           "k_log": meta["k_log"], "unit_log": meta.get("unit_log"), "vus_per_block": meta.get("vus_per_block"),
           "row_schemas": meta.get("row_schemas"), "circuit_bytes": len(text), "circuit_sha512": pin,
           "class_sha512": FC.class_digest(text), "inst_bytes": (out / "inst.bin").stat().st_size,
           "pub_bytes": (out / "pub.bin").stat().st_size, "twin": {"agrees": True, "instances": n, "first": int(twin[0, 0])},
           "seconds": {"draw": round(t_draw, 1), "total": round(time.time() - t0, 1)}, "maxrss_gb": gb()}
    (out / "stage.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
