"""The CPU half of a staging session's check, run on the node after the GPU is released:

- **decode**: every encoded block of the layer against P2's reference (`layout.check_blocks`: the spec's
  `reference.decode` under `reference.keys_after` of the store's prefix, refusing rows not below p and non-codewords),
  each block's 2,054 bytes against the store's W (`layout.payloads`: the stream's bytes, zero past each tensor's end), in
  a process pool;
- **gemm**: each `samples-<matrix>.npz` that `layer.py` wrote, every sampled word against
  `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` evaluated by the Definition's plain-integer reference.

    python check.py --layout LAYOUT --weights M --w W --store DIR --salt HEX --out OUT [--layer 0] [--workers N] [--blocks N]

Writes `OUT/check.json`; exits 1 if anything differs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L  # noqa: E402

_JOB: dict = {}


def _init(job: dict) -> None:
    _JOB.update(job)


def _read(path: str):
    def read(offset: int, size: int) -> bytes:
        with open(path, "rb") as f:
            data = os.pread(f.fileno(), size, offset)
        if len(data) != size:
            raise ValueError(f"{path} ends before byte {offset + size}")
        return data
    return read


def _blocks(r: tuple[int, int]) -> list[tuple[int, str]]:
    j0, j1 = r
    lay, n = _JOB["layout"], j1 - j0
    rows = _read(_JOB["c"])((j0 - _JOB["first"]) * L.ROW, n * L.ROW)
    return L.check_blocks(lay.spec, lay.prefix, j0, rows, L.payloads(lay, _JOB["src"], _read(_JOB["w"]), j0, n))


def check_decode(args, lay: L.StoreLayout, src: L.Weights) -> dict:
    if not (args.store / "c.json").exists():
        return {"checked": 0, "mismatches": 0, "store": "none yet"}
    meta = json.loads((args.store / "c.json").read_text())
    first, done = meta["first"], meta["done"]
    stop = first + (done if args.blocks is None else min(done, args.blocks))
    step = 2048
    ranges = [(j, min(stop, j + step)) for j in range(first, stop, step)]
    job = {"first": first, "c": str(args.store / "c.bin"), "w": str(args.w), "layout": lay, "src": src}
    bad: list[tuple[int, str]] = []
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(job,)) as pool:
        for b in pool.imap(_blocks, ranges):
            bad.extend(b)
    by_matrix = {}
    for t in lay.layer(args.layer):
        r = t.block_range
        by_matrix[t.matrix or t.name] = {"blocks": [r.start, r.stop], "checked": first <= r.start and r.stop <= stop}
    return {"blocks": [first, stop], "checked": stop - first, "of": meta["blocks"], "mismatches": len(bad),
            "first": bad[:8], "matrices": by_matrix}


def check_gemm(out: Path) -> dict:
    import gemm as G

    res = {}
    for path in sorted(out.glob("samples-*.npz")):
        d = np.load(path)
        want = G.definition_words(int(d["K"]), d["x"], d["sx"], d["w"], d["sw"])
        got = d["y"].astype(np.uint16)
        ne = np.nonzero(want != got)[0]
        res[path.stem.removeprefix("samples-")] = {
            "K": int(d["K"]), "samples": int(len(got)), "mismatches": int(len(ne)),
            "first": [{"m": int(d["m"][i]), "n": int(d["n"][i]), "got": int(got[i]), "definition": int(want[i])}
                      for i in ne[:8]]}
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--layout", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--w", type=Path, required=True)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--blocks", type=int, help="check only the first N encoded blocks")
    args = ap.parse_args(argv)
    lay = L.store_layout(json.loads(args.layout.read_text()), L.registered_prefix(bytes.fromhex(args.salt)))
    src = L.Weights.of(json.loads(args.weights.read_text()))
    doc = {"decode": check_decode(args, lay, src), "gemm": check_gemm(args.out)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    ok = doc["decode"]["mismatches"] == 0 and all(g["mismatches"] == 0 for g in doc["gemm"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
