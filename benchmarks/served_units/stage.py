"""stage.py UNIT N OUT: N instances of one served unit type's Q_word unit, from synthetic values at the model's served statics,
staged for C-Flock by `class_statement.stage_unit` as a ``verity/flock-circuit`` statement with hidden outputs: the unit's
Boolean Definition lowered once, its input rows hashed in the circuit (the per-call row on ``hm96-sha512/row-seg/v1``, the
weight row on ``hm96-sha512/row/v2``, PoUW's convention), its output word one ``row/v2`` output row.

    UNIT  TYPE[:S=V,...]   a type of `UNITS` at its statics (each type's defaults are GLM-5.3's), e.g. fp8-coord:K=2048

Every instance's output is checked twice before anything is written: by `stage_unit` against the IR reference of the Boolean
Definition, and here against the served kernel's numpy twin on the same values (``twin`` in stage.json).  Writes
circuit.txt, inst.bin, pub.bin, pin.txt and stage.json under OUT.

Types (GLM-5.3's statics from its config: hidden 6144, q_lora 2048, kv_lora 512, moe_intermediate 2048, FP8 blocks of 128):

    fp8-coord   ScaledMmFp8BlockCoordinate_v2{K,G}: one output coordinate of a block-scaled FP8 linear (CUTLASS blockwise,
                `ScaledMmFp8Block_v1{K,N,128}`'s Q_word unit), x = Fp8GroupQuant of a seeded N(0, 1) bf16 row, w = a seeded
                N(0, 0.02) bf16 weight row quantized per 128 the same way.  K = 6144 is fused_qkv_a and every gate_up
                (dense, shared and routed experts); 2048 is q_b, o_proj (one TP8 rank) and the routed experts' w2."""
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

from verity.ml.circuits.defs import bind
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_SEG as SEG, SCHEMA_HM96_SHA512_ROW_V2 as V2
from verity_flock import circuit as FC
from verity_flock import class_statement as CS
from verity_vllm.program.kernels import derived_rows as DR
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


def parse(spec: str) -> tuple[str, Unit, dict]:
    name, _, rest = spec.partition(":")
    if name not in UNITS:
        raise SystemExit(f"stage.py: unit type {name!r} is not one of {sorted(UNITS)}")
    u = UNITS[name]
    S = dict(u.defaults)
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        if k not in S:
            raise SystemExit(f"stage.py: {name} has no static {k!r} (its statics: {sorted(S)})")
        S[k] = int(v)
    return name, u, S


def gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def main() -> None:
    spec, n, out = sys.argv[1], int(sys.argv[2]), Path(sys.argv[3])
    t0 = time.time()
    name, u, S = parse(spec)
    d = u.definition(S)
    tag = f"{name}-" + "-".join(f"{k.lower()}{v}" for k, v in sorted(S.items()))
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
