"""The orbit (orbit.py) against the plain int8 GEMM on a checkpoint's int8 linears: every layer's fused qkv, o, fused
gate_up and down, each at activations of three kinds, and the carry metadata the kernel reads for B.

Activations (M rows, seeded): `uniform` int8 on ±127; `gauss`, Gaussian rows quantized to a per-row max of 127, the
served activations' rule; `edge`, rows of ±127 only, which make every two-term form of A carry where the signs agree.

    python cpu_check.py --snapshot DIR --out OUT [--layers 0,1,...] [--m 256] [--seed 0]

Writes OUT/check.json (per linear: exact, B's carry count and density per form, the fraction of k with a carry per
column tile, the metadata's bytes, A's carry density per form and kind) and exits 4 if any output differs.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orbit as O  # noqa: E402

DTYPES = {"I8": np.int8, "F32": np.float32, "BF16": np.uint16, "I32": np.int32}
LINEARS = {
    "qkv": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "o": ("self_attn.o_proj",),
    "gate_up": ("mlp.gate_proj", "mlp.up_proj"),
    "down": ("mlp.down_proj",),
}
TWO_TERM = (0, 1, 4, 5, 6)


class Shards:
    """A safetensors snapshot's tensors by name, read with numpy alone."""

    def __init__(self, snapshot: Path):
        self.where: dict[str, tuple[Path, dict, int]] = {}
        for path in sorted(snapshot.glob("*.safetensors")):
            with open(path, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                header = json.loads(f.read(n))
            for name, meta in header.items():
                if name != "__metadata__":
                    self.where[name] = (path, meta, 8 + n)

    def get(self, name: str) -> np.ndarray:
        path, meta, base = self.where[name]
        lo, hi = meta["data_offsets"]
        dt = DTYPES[meta["dtype"]]
        a = np.fromfile(path, dtype=dt, count=(hi - lo) // np.dtype(dt).itemsize, offset=base + lo)
        return a.reshape(meta["shape"])


def activations(kind: str, rng: np.random.Generator, m: int, k: int) -> np.ndarray:
    if kind == "uniform":
        return rng.integers(-127, 128, (m, k)).astype(np.int8)
    if kind == "gauss":
        x = rng.standard_normal((m, k)).astype(np.float32)
        s = np.abs(x).max(axis=1, keepdims=True) / np.float32(127)
        return np.clip(np.rint(x / s), -127, 127).astype(np.int8)
    if kind == "edge":
        return np.where(rng.random((m, k)) < 0.5, -127, 127).astype(np.int8)
    raise ValueError(kind)


def b_stats(cb: np.ndarray, cl: O.Carries) -> dict:
    tiles = cl.offsets.shape[1] - 1
    per_form, k_frac = [], []
    for j in range(7):
        nz = int(np.count_nonzero(cb[j]))
        per_form.append({"carries": nz, "density": nz / cb[j].size})
        if j in TWO_TERM and nz:
            hit = 0
            for t in range(tiles):
                rows = cb[j][t * cl.tile:(t + 1) * cl.tile]
                hit += int(np.count_nonzero(rows.any(axis=0)))
            k_frac.append(hit / (tiles * cb.shape[2]))
        elif j in TWO_TERM:
            k_frac.append(0.0)
    return {"forms": per_form, "density_two_term": sum(f["carries"] for f in per_form) / (5 * cb[0].size),
            "k_with_carry_per_tile": k_frac, "metadata_bytes": int(cl.entries.nbytes + cl.offsets.nbytes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile-n", type=int, default=O.TILE_N)
    ap.add_argument("--kinds", default="uniform,gauss,edge")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sh = Shards(Path(a.snapshot))
    n_layers = 1 + max(int(n.split(".")[2]) for n in sh.where if n.startswith("model.layers."))
    layers = [int(v) for v in a.layers.split(",")] if a.layers else list(range(n_layers))
    kinds = a.kinds.split(",")
    rng = np.random.default_rng(a.seed)
    recs, ok = [], True
    for layer in layers:
        for lin, parts in LINEARS.items():
            t0 = time.perf_counter()
            w = np.concatenate([sh.get(f"model.layers.{layer}.{p}.weight") for p in parts])
            if w.dtype != np.int8:
                raise SystemExit(f"layer {layer} {lin}: {w.dtype}, not an int8 checkpoint")
            wb, cb = O.b_forms(w)
            cl = O.b_carries(cb, a.tile_n)
            rec = {"layer": layer, "linear": lin, "n": w.shape[0], "k": w.shape[1], "b": b_stats(cb, cl), "a": {}}
            for kind in kinds:
                x = activations(kind, rng, a.m, w.shape[1])
                want = O.plain(x, w)
                got = O.orbit(x, w, b=(wb, cl))
                bad = int(np.count_nonzero(got != want))
                ca = O.a_forms(x)[1]
                rec["a"][kind] = {"mismatches": bad, "density": [float(np.count_nonzero(ca[j])) / ca[j].size for j in range(7)],
                                  "uncorrected_mismatch_frac": float(np.mean(O.wrapped_only(x, w) != want))}
                ok &= bad == 0
            rec["exact"] = all(v["mismatches"] == 0 for v in rec["a"].values())
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            recs.append(rec)
            print(json.dumps({k: rec[k] for k in ("layer", "linear", "exact", "seconds")}
                             | {"b_density": rec["b"]["density_two_term"]}), flush=True)
    summary = {}
    for lin in LINEARS:
        rs = [r for r in recs if r["linear"] == lin]
        if rs:
            summary[lin] = {"layers": len(rs), "exact": all(r["exact"] for r in rs),
                            "b_density_two_term": float(np.mean([r["b"]["density_two_term"] for r in rs])),
                            "b_density_max_layer": float(max(r["b"]["density_two_term"] for r in rs)),
                            "k_with_carry_per_tile": float(np.mean([np.mean(r["b"]["k_with_carry_per_tile"]) for r in rs])),
                            "metadata_bytes_per_layer": int(np.mean([r["b"]["metadata_bytes"] for r in rs])),
                            "a_density_two_term": {kind: float(np.mean([np.mean([r["a"][kind]["density"][j] for j in TWO_TERM])
                                                                        for r in rs])) for kind in kinds}}
    doc = {"snapshot": a.snapshot, "m": a.m, "seed": a.seed, "tile_n": a.tile_n, "kinds": kinds, "exact": ok,
           "summary": summary, "linears": recs}
    (out / "check.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"exact": ok, "summary": summary}, indent=1), flush=True)
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main())
