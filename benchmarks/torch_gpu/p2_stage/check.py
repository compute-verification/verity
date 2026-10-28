"""The CPU half of a staging run's check, run on the node after the GPU is released:

- **decode**: every encoded block of the layer, from the development store or an encoded store's segment files (as
  `layer.py` takes them), against P2's reference (`layout.check_blocks`: the spec's `reference.decode` under
  `reference.keys_after` of the store's prefix, refusing rows not below p and non-codewords), each block's 2,054 bytes
  against the store's W (`layout.payloads`: the stream's bytes, from `W` or else the seeded generator, zero past each
  tensor's end), in a process pool;
- **gemm**: each `samples-<matrix>-m<M>.npz` that `layer.py` wrote, every sampled word against
  `ScaledMmS8Coordinate_v1{K, GEMM_SM90_S8}` evaluated by the Definition's plain-integer reference.

    python check.py --layout LAYOUT --weights M [--w W] (--store DIR --salt HEX | --encoded DIR --source HEX
                    --commitment HEX) --out OUT [--layer 0] [--workers N] [--blocks N] [--sample N --seed S]

With `--sample N` the decode half takes N labels of every tensor of the encoded store instead (`sample_blocks`: each
tensor's first and last label among them), for `full.py`'s whole-store run.

Writes `OUT/check.json`; exits 1 if anything differs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L

_JOB: dict = {}


def _init(job: dict) -> None:
    _JOB.update(job)
    _JOB["read"] = (L.reader(Path(job["w"])) if job["w"]
                    else L.generator_reader(job["generator"], job["shape"], job["weights_seed"]))


def _rows(j0: int, n: int) -> bytes:
    if _JOB["encoded"]:
        return L.segment_rows(Path(_JOB["encoded"]), _JOB["layout"].spec, j0, n)
    with open(_JOB["c"], "rb") as f:
        data = os.pread(f.fileno(), n * L.ROW, (j0 - _JOB["first"]) * L.ROW)
    if len(data) != n * L.ROW:
        raise ValueError(f"{_JOB['c']} ends before block {j0 + n}")
    return data


def _blocks(r: tuple[int, int]) -> list[tuple[int, str]]:
    j0, j1 = r
    lay, n = _JOB["layout"], j1 - j0
    return L.check_blocks(lay.spec, lay.prefix, j0, _rows(j0, n), L.payloads(lay, _JOB["src"], _JOB["read"], j0, n))


def check_decode(args, lay: L.StoreLayout, src: L.Weights) -> dict:
    if args.encoded is not None:
        r = lay.blocks(args.layer)
        first, done, of = r.start, len(r), len(r)
    elif (args.store / "c.json").exists():
        meta = json.loads((args.store / "c.json").read_text())
        first, done, of = meta["first"], meta["done"], meta["blocks"]
    else:
        return {"checked": 0, "mismatches": 0, "store": "none yet"}
    stop = first + (done if args.blocks is None else min(done, args.blocks))
    step = 2048
    ranges = [(j, min(stop, j + step)) for j in range(first, stop, step)]
    job = {"first": first, "c": str(args.store / "c.bin") if args.store else None,
           "encoded": str(args.encoded) if args.encoded else None, "w": str(args.w) if args.w else None,
           "generator": args.generator, "shape": args.shape, "weights_seed": args.weights_seed, "layout": lay, "src": src}
    bad: list[tuple[int, str]] = []
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(job,)) as pool:
        for b in pool.imap(_blocks, ranges):
            bad.extend(b)
    by_matrix = {}
    for t in lay.layer(args.layer):
        r = t.block_range
        by_matrix[t.matrix or t.name] = {"blocks": [r.start, r.stop], "checked": first <= r.start and r.stop <= stop}
    return {"store": str(args.encoded or args.store), "blocks": [first, stop], "checked": stop - first, "of": of,
            "mismatches": len(bad), "first": bad[:8], "matrices": by_matrix}


def sample_blocks(tensors: list[L.Tensor], per_tensor: int, seed: int) -> list[int]:
    """Each tensor's first and last block and `per_tensor - 2` others of its blocks sampled by `seed` (all of them when it
    has no more than `per_tensor`), in block order."""
    rng, out = random.Random(seed), set()
    for t in tensors:
        r = t.block_range
        inner = range(r.start + 1, r.stop - 1)
        out |= {r.start, r.stop - 1} | set(rng.sample(inner, min(len(inner), max(0, per_tensor - 2))))
    return sorted(out)


def check_sampled(args, lay: L.StoreLayout, src: L.Weights) -> dict:
    """`--sample` blocks of every tensor of the store (`sample_blocks`) against P2's reference, as `check_decode`."""
    tensors = list(lay.tensors)
    blocks = sample_blocks(tensors, args.sample, args.seed)
    job = {"first": 0, "c": None, "encoded": str(args.encoded), "w": str(args.w) if args.w else None,
           "generator": args.generator, "shape": args.shape, "weights_seed": args.weights_seed, "layout": lay, "src": src}
    bad: list[tuple[int, str]] = []
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(job,)) as pool:
        for b in pool.imap(_blocks, [(j, j + 1) for j in blocks], chunksize=16):
            bad.extend(b)
    return {"store": str(args.encoded), "per_tensor": args.sample, "seed": args.seed, "tensors": len(tensors),
            "layers": tensors[-1].layer + 1, "checked": len(blocks), "mismatches": len(bad), "first": bad[:8]}


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
    ap.add_argument("--w", type=Path, help="the stream's bytes at their offsets (else the generator's)")
    ap.add_argument("--generator", default="e2e_h200.served_weights")
    ap.add_argument("--shape", default="h200")
    ap.add_argument("--weights-seed", default="text:verity/cmp92/h200/weights/v1")
    store = ap.add_mutually_exclusive_group(required=True)
    store.add_argument("--store", type=Path, help="the development store")
    store.add_argument("--encoded", type=Path, help="an encoded store's directory, read only")
    ap.add_argument("--salt", help="the development store's public test salt")
    ap.add_argument("--source", help="the encoded store's coin (hex)")
    ap.add_argument("--commitment", help="the encoded store's commitment (hex)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--blocks", type=int, help="check only the layer's first N encoded blocks")
    ap.add_argument("--sample", type=int, help="instead of one layer's every block, N blocks of every tensor of the "
                    "encoded store (its first and last among them)")
    ap.add_argument("--seed", type=int, default=12, help="the sample of the sampled blocks")
    args = ap.parse_args(argv)
    if args.sample is not None and args.encoded is None:
        ap.error("--sample reads an encoded store")
    manifest = json.loads(args.weights.read_text())
    try:
        lay, _ = L.open_store(args.layout, manifest, args.encoded, args.source, args.commitment, args.salt)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    src = L.Weights.of(manifest)
    decode = check_sampled(args, lay, src) if args.sample is not None else check_decode(args, lay, src)
    doc = {"decode": decode, "gemm": check_gemm(args.out)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "check.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc))
    ok = doc["decode"]["mismatches"] == 0 and all(g["mismatches"] == 0 for g in doc["gemm"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
