"""The bandwidth audit's receipts on the GPU (`sketch.py`) held to serve.py's fixture and to the CPU reference, and their
cost at the store's size.

    python bw_check.py --out OUT [--lib LIB] [--random UNITS] [--bench] [--device cuda]

- **fixture**: `serve.fixture()` regenerated and held to `FIXTURE_SHA512`; then for each of its two chained passes, on
  the device: each block's π (`Pass.partials`), σ read in chunks of 1, 3, 5 and 33 blocks, y, and the root (each pass's
  seed from the one before's root); with `--lib`, the toy's store decoded layer by layer through the staging decode
  (`stage.decode_layer`) with the pass attached, the units no layer reads then read alone, and the root.
- **random**: `--random` units of random C* on the device against the CPU reference (`p2.Sketcher`): σ of a seeded
  sample of units from their blocks' π, every y from the device's σ, and the root (`receipts.tree`) under random salts.
- **bench** (`--bench`): CUDA-event times of σ over one decode chunk (65,536 blocks), as one unit GEMM and as block
  partials, and of y and the root at the H200 store's pass of 17,003,406 units (those its 209 tensors lie in, blocks
  0 .. 68,013,622), scaled to a step.

Writes `OUT/bw_check.json` and prints one verdict line; exits 1 if anything differs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from functools import partial
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "pous"))
import layout as L  # noqa: E402
import sketch as K  # noqa: E402
import stage as S  # noqa: E402
from p2_v1 import serve  # noqa: E402

from verity.core.protocols.remote_bandwidth import p2  # noqa: E402
from verity.core.protocols.remote_bandwidth import receipts as R  # noqa: E402

STORE_UNITS = -(-68_013_623 // p2.UNIT_BLOCKS)
CHUNK = 1 << 16


def le(t) -> list[str]:
    return [bytes(r.cpu().numpy().astype("<i4").tobytes()).hex() for r in t]


def hexes(t) -> list[str]:
    return [bytes(r.cpu().numpy()).hex() for r in t]


def same(a: list[str], b: list[str]) -> dict:
    bad = [i for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
    return {"match": len(a) - len(bad), "of": len(a), "first_bad": bad[:8]}


def fixture(args, dev: torch.device) -> dict:
    data = serve.fixture_bytes()
    fx = json.loads(data)
    _, doc, c_star = serve.toy()
    nb, units = fx["store"]["blocks"], fx["store"]["units"]
    rows = torch.frombuffer(bytearray(c_star), dtype=torch.uint8).view(nb, L.ROW).to(dev)
    tree = K.Tree(units, dev)
    lay = L.store_layout(doc, L.registered_prefix(bytes.fromhex(doc["salt"]))) if args.lib else None
    challenge, previous = bytes.fromhex(fx["challenge"]), bytes.fromhex(fx["registration"]["digest"])
    out = {"fixture_sha512": hashlib.sha512(data).hexdigest(), "pinned": hashlib.sha512(data).hexdigest() == serve.FIXTURE_SHA512,
           "passes": []}
    for p in fx["passes"]:
        seed = R.seed(challenge, p["pass"], previous)
        salts = tree.salts(b"".join(bytes.fromhex(s) for s in p["leaf_salts"]))
        P = K.Pass(seed, units, dev)
        rec = {"pass": p["pass"], "seed": seed.hex() == p["seed"],
               "pi": same(le(P.partials(0, rows[:p2.UNIT_BLOCKS * units])), p["partials"]), "sigma": {}}
        for step in (1, 3, 5, nb):
            Q = K.Pass(seed, units, dev)
            for j0 in range(0, nb, step):
                Q.read(rows, 0, j0, min(nb, j0 + step))
            rec["sigma"][str(step)] = same(le(Q.sigma), p["sigmas"])
        P.read(rows, 0, 0, nb)
        root = P.finish(tree, salts)
        rec["y"] = same(hexes(P.y), p["receipts"])
        rec["root"] = root.hex() == p["root"]
        if lay is not None:
            D = K.Pass(seed, units, dev)
            st = S.Store(lay, rows, 0, args.lib)
            st.sketch = D
            for layer, _, size, _ in doc["staging"]:
                S.decode_layer(st, layer, torch.empty(size, dtype=torch.uint8, device=dev))
            unread = [g for g in range(units) if not bool(D.seen[g])]
            for g in unread:
                D.read(rows, 0, p2.UNIT_BLOCKS * g, p2.UNIT_BLOCKS * (g + 1))
            rec["decode"] = {"units_the_decode_read": units - len(unread), "read_after": unread,
                             "sigma": same(le(D.sigma), p["sigmas"]), "root": D.finish(tree, salts).hex() == p["root"]}
        out["passes"].append(rec)
        previous = root
    ok = out["pinned"] and all(
        r["seed"] and r["pi"]["match"] == r["pi"]["of"] and all(s["match"] == s["of"] for s in r["sigma"].values())
        and r["y"]["match"] == r["y"]["of"] and r["root"]
        and ("decode" not in r or (r["decode"]["sigma"]["match"] == r["decode"]["sigma"]["of"] and r["decode"]["root"]))
        for r in out["passes"])
    out["ok"] = bool(ok)
    return out


def random_check(units: int, sample: int, dev: torch.device) -> dict:
    gen = torch.Generator(device=dev).manual_seed(units)
    rows = torch.randint(0, 256, (p2.UNIT_BLOCKS * units, L.ROW), dtype=torch.uint8, device=dev, generator=gen)
    seed = hashlib.sha512(b"verity/p2-stage/bw-check/random/%d" % units).digest()
    tree = K.Tree(units, dev)
    salts_raw = os.urandom(R.SALT_BYTES * tree.G)
    P = K.Pass(seed, units, dev)
    for j0 in range(0, rows.shape[0], 12345):
        P.read(rows, 0, j0, min(rows.shape[0], j0 + 12345))
    root = P.finish(tree, tree.salts(salts_raw))
    sk = p2.Sketcher(seed, R.SKETCH_ROWS)
    picks = sorted(random.Random(units).sample(range(units), min(sample, units)) + [0, units - 1])
    host = rows.cpu().numpy()
    sig = P.sigma.cpu().numpy().astype("<i4")
    bad_sigma = [g for g in picks if sk.total([sk.block(j, host[j].tobytes()) for j in range(4 * g, 4 * g + 4)])
                 != sig[g].tobytes()]
    ys = [sk.receipt(g, sig[g].tobytes()) for g in range(units)]
    y = P.y.cpu().numpy()
    bad_y = [g for g in range(units) if y[g].tobytes() != ys[g]]
    salts = [salts_raw[R.SALT_BYTES * c:R.SALT_BYTES * (c + 1)] for c in range(tree.G)]
    want = R.tree(seed, R.Plan.whole(units), ys, salts).commitment.root
    return {"units": units, "sigma_sampled": len(picks), "sigma_bad": bad_sigma[:8], "y_bad": len(bad_y),
            "root": root == want, "ok": not bad_sigma and not bad_y and root == want}


def fastest(fn, reps: int = 5) -> float:
    return min(S.timed(fn, reps, torch.device("cuda")))


def bench(dev: torch.device) -> dict:
    seed = hashlib.sha512(b"verity/p2-stage/bw-check/bench").digest()
    rows = torch.randint(0, 256, (CHUNK, L.ROW), dtype=torch.uint8, device=dev)
    P = K.Pass(seed, CHUNK // p2.UNIT_BLOCKS, dev)
    unit_s = fastest(partial(P.read, rows, 0, 0, CHUNK))
    block_s = fastest(partial(P.partials, 0, rows))
    del rows, P
    sigma = torch.randint(-(1 << 20), 1 << 20, (STORE_UNITS, R.SKETCH_ROWS), dtype=torch.int32, device=dev)
    tree = K.Tree(STORE_UNITS, dev)
    y_s = fastest(partial(K.receipts, seed, sigma, 0, tree.y))
    salts = tree.salts(torch.randint(0, 256, (tree.G * R.SALT_BYTES,), dtype=torch.uint8, device=dev))
    root_s = fastest(partial(tree.root, seed, salts), 3)
    E, did = tree.E, R.domain(seed, tree.plan).domain_id
    x, lt = E.row_digests(tree.table), tree.leaf_table(did)
    leaves = E.leaves(lt, x, salts).reshape(1, tree.G, 8)
    parts = {"domain_s": fastest(partial(R.domain, seed, tree.plan), 3), "leaf_table_s": fastest(partial(tree.leaf_table, did), 3),
             "row_digests_s": fastest(partial(E.row_digests, tree.table), 3), "leaves_s": fastest(partial(E.leaves, lt, x, salts), 3),
             "tree_s": fastest(partial(E.root, leaves, [did]), 3)}
    del x, lt, leaves
    chunks = STORE_UNITS * p2.UNIT_BLOCKS / CHUNK
    step = unit_s * chunks + y_s + root_s
    return {"chunk_blocks": CHUNK, "sigma_unit_gemm_s": round(unit_s, 6), "sigma_block_partials_s": round(block_s, 6),
            "chunk_gb_s": round(CHUNK * L.ROW / unit_s / 1e9, 1), "units": STORE_UNITS, "y_s": round(y_s, 6),
            "root_s": round(root_s, 6), "root_parts_s": {k: round(v, 6) for k, v in parts.items()}, "leaves": tree.G,
            "step_sigma_s": round(unit_s * chunks, 4),
            "step_receipts_s": round(step, 4), "vs_decode_4.883s": round(step / 4.883, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lib", type=Path, help="build.sh's libp2v1 directory: the toy through the staging decode too")
    ap.add_argument("--random", type=int, default=0, help="units of random C* against the CPU reference")
    ap.add_argument("--sample", type=int, default=2000, help="units whose σ the reference recomputes from C*")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    dev = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    res = {"device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu", "torch": torch.__version__}
    res["fixture"] = fixture(args, dev)
    ok = res["fixture"]["ok"]
    if args.random:
        res["random"] = random_check(args.random, args.sample, dev)
        ok &= res["random"]["ok"]
    if args.bench and dev.type == "cuda":
        res["bench"] = bench(dev)
    res["ok"] = bool(ok)
    (args.out / "bw_check.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    f = res["fixture"]
    line = [f"bw_check {'PASS' if ok else 'FAIL'}: fixture {'pinned' if f['pinned'] else 'MOVED'}"]
    for r in f["passes"]:
        s = min(r["sigma"].values(), key=lambda v: v["match"])
        line.append(f"pass {r['pass']}: pi {r['pi']['match']}/{r['pi']['of']} sigma {s['match']}/{s['of']} "
                    f"y {r['y']['match']}/{r['y']['of']} root {'match' if r['root'] else 'DIFFERS'}"
                    + (f" (decode path root {'match' if r['decode']['root'] else 'DIFFERS'})" if "decode" in r else ""))
    if "random" in res:
        r = res["random"]
        line.append(f"random {r['units']} units: sigma {r['sigma_sampled'] - len(r['sigma_bad'])}/{r['sigma_sampled']} "
                    f"y bad {r['y_bad']} root {'match' if r['root'] else 'DIFFERS'}")
    if "bench" in res:
        b = res["bench"]
        line.append(f"bench: sigma {b['sigma_unit_gemm_s'] * 1e3:.3f} ms/chunk, y {b['y_s'] * 1e3:.1f} ms, "
                    f"root {b['root_s'] * 1e3:.1f} ms; step {b['step_receipts_s']:.3f} s = {b['vs_decode_4.883s']:.4f}x of 4.883 s")
    print("; ".join(line))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
