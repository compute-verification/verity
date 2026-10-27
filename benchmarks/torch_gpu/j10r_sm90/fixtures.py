"""The expected values of J10-R's proof blocks at the frozen shape, from the Definition itself: for each (linear, M)
of `shape` and each sampled proof block (`shape.picks`), `h1tile.j10r_reference` (`PouwH1Tile_v5`) on its 32 rows of A
and of B gives its z (32 x 32, mod 2^32) and its receipt (2 (K / 32 + m) u32), under seed_c = `seeds.call_seed`(salt,
base, roots) at its `j10r_blocks` and sigma_R = `seeds.receipt_seed`(salt, base + u, roots). `--v6` proof blocks a call
also go through `PouwH1Tile_v6` (stream=True), whose z and receipt must be v5's.

    python fixtures.py expected --out OUT [--jobs 16] [--ms 4096,16384] [--linears q,kv,o,up,down,head] [--picks 16] [--v6 2]

OUT/expected.npz holds, per call "<linear>-<M>": units (S,) int64, receipts (S, W) uint32, z (S, 32, 32) uint32,
seed_c (64,) uint8 and sigma (S, 64) uint8; OUT/fixtures.json the calls' public inputs, each sampled proof block's operand
digests (the sha256 of its 32 rows of A and of B, which `check.py` holds the regenerated operands to), the arrays'
sha256 and the Definition's source digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shape as S

from verity.ml.protocols.pouw.circuit import h1tile as HT
from verity.ml.protocols.pouw.circuit import seeds as SD

SCHEMA = "j10r-sm90/fixtures/v1"
ARRAYS = ("units", "receipts", "z", "seed_c", "sigma")


def key(linear: str, m: int) -> str:
    return f"{linear}-{m}"


def unit_expected(job: tuple) -> dict:
    """One sampled proof block through j10r_reference (v5; with v6, PouwH1Tile_v6 too)."""
    linear, m, u, v6 = job
    c = S.call(linear, m)
    ra, rb = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
    ncb = c["n"] // S.T
    ri, cj = divmod(u, ncb)
    a, b = S.unit_operands(linear, m, u)
    blocks = HT.j10r_blocks(m, c["k"], (S.T, S.T), ri, cj)
    ac, bc = (a.astype(np.int64) & 0xFF).tolist(), (b.astype(np.int64) & 0xFF).tolist()
    t0 = time.time()
    z, rec = HT.j10r_reference(ac, bc, S.SALT, c["base"] + u, (ra, rb), c["base"], blocks)
    out = {"linear": linear, "m": m, "u": u, "blocks": list(blocks), "z": z, "receipt": rec, "v5_s": time.time() - t0,
           "a_sha256": S.sha256(a), "b_sha256": S.sha256(b)}
    if v6:
        t0 = time.time()
        z6, rec6 = HT.j10r_reference(ac, bc, S.SALT, c["base"] + u, (ra, rb), c["base"], blocks, stream=True)
        out.update(v6_s=time.time() - t0, v6_z_equal=z6 == z, v6_receipt_equal=rec6 == rec)
    return out


def _sources() -> dict:
    return {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (HT, SD, HT.SG)}


def expected(out: Path, linears: list[str], ms: list[int], count: int, v6: int, jobs: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    work = []
    for m in ms:
        for lin in linears:
            work += [(lin, m, u, i < v6) for i, u in enumerate(S.picks(lin, m, count))]
    work.sort(key=lambda j: -S.LINEARS[j[0]][1])
    t0 = time.time()
    if jobs <= 1:
        res = [unit_expected(j) for j in work]
    else:
        with ProcessPoolExecutor(jobs) as ex:
            res = list(ex.map(unit_expected, work))
    doc: dict = {"schema": SCHEMA, "definition": "h1tile.j10r_reference (PouwH1Tile_v5; v6 on the first --v6 units)",
                 "sources": _sources(), "salt": S.SALT.hex(), "weights_seed": S.WEIGHTS_SEED.decode(),
                 "weights_format": S.WEIGHTS_FORMAT, "activations_seed": S.X_SEED.decode(), "tile": [S.T, S.T],
                 "picks": count, "wall_s": None, "calls": {}}
    arrays: dict = {}
    for m in ms:
        for lin in linears:
            c = S.call(lin, m)
            mine = sorted((r for r in res if r["linear"] == lin and r["m"] == m), key=lambda r: r["u"])
            ra, rb = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
            k = key(lin, m)
            arrays[f"{k}/units"] = np.array([r["u"] for r in mine], np.int64)
            arrays[f"{k}/receipts"] = np.array([r["receipt"] for r in mine], np.uint32)
            arrays[f"{k}/z"] = np.array([r["z"] for r in mine], np.uint32).reshape(len(mine), S.T, S.T)
            arrays[f"{k}/seed_c"] = np.frombuffer(SD.call_seed(S.SALT, c["base"], ra, rb), np.uint8)
            arrays[f"{k}/sigma"] = np.stack([np.frombuffer(SD.receipt_seed(S.SALT, c["base"] + r["u"], ra, rb), np.uint8)
                                             for r in mine])
            ncb = c["n"] // S.T
            c.update({"words": int(arrays[f"{k}/receipts"].shape[1]), "digits": HT.Layout(S.T, S.T, c["k"]).m,
                      "row_blocks_sampled": len({r["u"] // ncb for r in mine}),
                      "rows_sampled": S.T * len({r["u"] // ncb for r in mine}),
                      "samples": [{"u": r["u"], "ri": r["u"] // ncb, "cj": r["u"] % ncb, "blocks": r["blocks"],
                                   "a_sha256": r["a_sha256"], "b_sha256": r["b_sha256"], "v5_s": round(r["v5_s"], 2),
                                   **({"v6_s": round(r["v6_s"], 2), "v6_z_equal": r["v6_z_equal"],
                                       "v6_receipt_equal": r["v6_receipt_equal"]} if "v6_s" in r else {})}
                                  for r in mine],
                      "sha256": {a: S.sha256(arrays[f"{k}/{a}"]) for a in ARRAYS}})
            doc["calls"][k] = c
    doc["wall_s"] = round(time.time() - t0, 1)
    doc["v6_all_equal"] = all(r.get("v6_z_equal", True) and r.get("v6_receipt_equal", True) for r in res)
    np.savez(out / "expected.npz", **arrays)
    doc["npz_sha256"] = hashlib.sha256((out / "expected.npz").read_bytes()).hexdigest()
    (out / "fixtures.json").write_text(json.dumps(doc, indent=1))
    return doc


def load(path: Path) -> tuple[dict, dict]:
    """fixtures.json and expected.npz from a directory, each array held to its digest."""
    doc = json.loads((path / "fixtures.json").read_text())
    with np.load(path / "expected.npz") as z:
        arrays = {k: z[k] for k in z.files}
    for k, c in doc["calls"].items():
        for a, want in c["sha256"].items():
            if S.sha256(arrays[f"{k}/{a}"]) != want:
                raise SystemExit(f"fixtures: {k}/{a} is not the array fixtures.json pins")
    return doc, arrays


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["expected"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--ms", default=",".join(map(str, S.MS)))
    ap.add_argument("--linears", default=",".join(S.ORDER))
    ap.add_argument("--picks", type=int, default=S.PICKS)
    ap.add_argument("--v6", type=int, default=2)
    a = ap.parse_args()
    doc = expected(Path(a.out), a.linears.split(","), [int(v) for v in a.ms.split(",")], a.picks, a.v6, a.jobs)
    print(json.dumps({k: doc[k] for k in ("schema", "wall_s", "v6_all_equal", "npz_sha256")}))
    for k, c in doc["calls"].items():
        print(json.dumps({"call": k, "units": c["units"], "samples": len(c["samples"]), "rows_sampled": c["rows_sampled"],
                          "words": c["words"], "receipts_sha256": c["sha256"]["receipts"][:16]}))
    return 0 if doc["v6_all_equal"] else 4


if __name__ == "__main__":
    sys.exit(main())
