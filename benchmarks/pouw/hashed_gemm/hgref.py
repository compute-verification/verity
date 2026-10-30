"""hgemm.cu's gate judged by the repository's own functions: every level of both trees of every call, by
``verity.ml.commitments`` (commit_gpu's reference.py: ``permute``, ``prf_salt``, ``params``), and sampled tiles' windowed
receipts by ``segment.PouwSegCheck_v2{16, FOLD}``'s own fold (``fold_keys``, ``fold_absorb``, ``fold_final`` of
`verity.ml.protocols.pouw.circuit.segment`, which `PouwSegCheck_v3` checks the same receipts with).

Call c's trees, each a ``CommitmentDomain(identity_digest_sha512(TAG, {"tree": T, "m", "n", "k", "leaves"}), -1,
RangeIndexedDomain(0, leaves), hash="sha512")`` of hm96-sha512/row/v2 rows of 16,384 bits:
    X  ``permute(X, m, k, 64)``'s rows (the tile leaves), salts ``prf_salt(Ks, 2c 2^40 + pos)``
    R  the call's receipts, a row a window of a tile (R[(tile nw + w) 512 + thread]), salts ``prf_salt(Ks, (2c + 1) 2^40 + pos)``
so no two leaves of any call or tree share a salt. Each sampled opening goes through ``verify_opening`` on the device's
path, and the controls must fail it: the leaf digest with one bit flipped, and the leaf of the row with one byte (X) or one
receipt word (R) changed.

    python3 hgref.py params DIR M N K [--ks-seed S]   DIR/px.bin and DIR/pr.bin, the trees' CGP1 files (one Ks)
    python3 hgref.py check DIR...                     a gate directory each (meta.json, X.bin, Wp_C.bin, R_KN_C.bin,
                                                      lv_KN_C_X.bin, lv_KN_C_R.bin); JSON lines, exit 1 on any difference

Run from the source tree's root with ``PYTHONPATH=.``.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "commit_gpu"))
import reference as cref  # noqa: E402

from verity.ml.protocols.pouw.circuit import segment as SG  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.limits import VerificationLimits  # noqa: E402
from verity.ml.commitments.merkle import Commitment, CommitmentDomain, Opening, tree_levels, verify_opening  # noqa: E402
from verity.ml.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2 as V2  # noqa: E402
from verity.ml.commitments.rowleaf import row_leaf, sha512_row_v2_digest  # noqa: E402

TAG = "verity/hashed-gemm/v1"
ROW, ROW_BITS, XR, S, NCT = 2048, 16384, 64, 16, 512
INST, KEY = cref.INST, cref.KEY


def domain(tree: str, m: int, n: int, k: int, leaves: int) -> CommitmentDomain:
    label = {"tree": tree, "m": m, "n": n, "k": k, "leaves": leaves}
    return CommitmentDomain(identity_digest_sha512(TAG, label), -1, RangeIndexedDomain(0, leaves), hash="sha512")


def counts(m: int, n: int, k: int) -> tuple[int, int, int]:
    """X's leaves, the receipts' leaves, windows a tile."""
    nw = k // (32 * S)
    return cref.tile_count(m, k, XR), (m // 128) * (n // 128) * nw, nw


def write_params(d: str, m: int, n: int, k: int, ks: bytes) -> None:
    nx, nr, _ = counts(m, n, k)
    os.makedirs(d, exist_ok=True)
    for name, tree, leaves in (("px.bin", "X", nx), ("pr.bin", "R", nr)):
        with open(os.path.join(d, name), "wb") as f:
            f.write(cref.params(leaves, ROW_BITS, ks, domain_id=domain(tree, m, n, k, leaves).domain_id))


# --- leaves, in forked workers --------------------------------------------------------------------------------------

_J: dict = {}


def leaf_hash(row: bytes, salt: bytes) -> bytes:
    return INST.tree_leaf(KEY, INST.commit_string(KEY, sha512_row_v2_digest(row, ROW_BITS), salt))


def _span(span: tuple[int, int]) -> tuple[list[bytes], list[bytes]]:
    j, hs, leaves = _J, [], []
    for pos in range(*span):
        h = leaf_hash(j["rows"][ROW * pos:ROW * pos + ROW], cref.prf_salt(j["ks"], j["off"] + pos))
        hs.append(h)
        leaves.append(row_leaf(j["dom"], pos, h, V2))
    return hs, leaves


def tree(rows: bytes, ks: bytes, off: int, dom: CommitmentDomain) -> tuple[list[bytes], bytes]:
    """Every leaf's hm96 digest and every level (leaves first, as the device lays them out)."""
    n = dom.count
    _J.clear()
    _J.update(rows=rows, ks=ks, off=off, dom=dom)
    step = max(1, min(2048, -(-n // (4 * (os.cpu_count() or 1)))))
    spans = [(a, min(n, a + step)) for a in range(0, n, step)]
    with mp.get_context("fork").Pool() as pool:   # forked after _J is set: the workers read it
        parts = pool.map(_span, spans)
    levels = tree_levels(dom, [x for p in parts for x in p[1]])
    return [x for p in parts for x in p[0]], b"".join(b"".join(level) for level in levels)


def device_levels(p: Path) -> bytes:
    """A level dump as digest bytes: the device stores each digest as 8 native (little-endian) u64 words, a word each 8
    big-endian digest bytes."""
    return np.fromfile(p, np.uint64).byteswap().tobytes()


def offsets(n: int) -> tuple[int, list[int]]:
    depth = (n - 1).bit_length() if n > 1 else 0
    offs = [0]
    for d in range(depth + 1):
        offs.append(offs[-1] + 64 * ((1 << depth) >> d))
    return depth, offs


def judge(name: str, dom: CommitmentDomain, rows: bytes, ks: bytes, off: int, g: bytes, ref: tuple[list[bytes], bytes],
          rng: random.Random, word: int, openings: int = 48) -> dict:
    """The device's levels g against the reference's; openings on the device's paths; the controls (word: the bytes a
    control changes in a row)."""
    hs, r = ref
    n = dom.count
    depth, offs = offsets(n)
    bad = [d for d in range(depth + 1) if g[offs[d]:offs[d + 1]] != r[offs[d]:offs[d + 1]]]
    bad0 = [i for i in range(n) if g[64 * i:64 * i + 64] != r[64 * i:64 * i + 64]]
    # the device's levels above its own leaves, by the reference's nodes: separates a node fault from a leaf fault
    own = b"".join(b"".join(level) for level in tree_levels(dom, [g[64 * i:64 * i + 64] for i in range(n)]))
    root = g[offs[depth]:offs[depth + 1]]
    commitment, limits = Commitment(root, n), VerificationLimits()
    picks = sorted({0, n - 1, *rng.sample(range(n), min(n, openings) - 2)})
    bad_open, controls = [], 0
    for pos in picks:
        path = tuple(g[offs[d] + 64 * ((pos >> d) ^ 1):offs[d] + 64 * ((pos >> d) ^ 1) + 64] for d in range(depth))
        ok = verify_opening(dom, commitment, Opening(pos, hs[pos], path), V2, limits)
        ok &= not verify_opening(dom, commitment, Opening(pos, bytes([hs[pos][0] ^ 1]) + hs[pos][1:], path), V2, limits)
        row = bytearray(rows[ROW * pos:ROW * pos + ROW])
        at = word * rng.randrange(ROW // word)
        row[at] ^= 1 << rng.randrange(8)
        h2 = leaf_hash(bytes(row), cref.prf_salt(ks, off + pos))
        ok &= not verify_opening(dom, commitment, Opening(pos, h2, path), V2, limits)
        controls += 2
        if not ok:
            bad_open.append(pos)
    return {"tree": name, "leaves": n, "depth": depth, "levels_ok": not bad and len(g) >= offs[-1], "bad_levels": bad[:8],
            "bad_leaves": len(bad0), "first_bad_leaves": bad0[:8], "nodes_ok": own == g[:offs[-1]],
            "openings": len(picks), "controls": controls, "bad_openings": bad_open[:8], "root": root.hex()[:32],
            "ok": not bad and not bad_open and len(g) >= offs[-1]}


# --- receipts --------------------------------------------------------------------------------------------------------

def thread_entries() -> tuple[np.ndarray, np.ndarray]:
    """Every thread's 32 entries' (row, col) in its tile: entry q at segment's row_slot(q), col_slot(q)."""
    ctid = np.arange(NCT)
    warp, lane = ctid // 32, ctid % 32
    wm, wn, g, qq = warp // 4, warp % 4, lane >> 2, lane & 3
    rs = np.array([SG.row_slot(q) for q in range(32)])
    cs = np.array([SG.col_slot(q) for q in range(32)])
    rows = wm[:, None] * 32 + g[:, None] + 8 * rs[None, :]
    cols = wn[:, None] * 32 + 2 * qq[:, None] + 8 * (cs[None, :] >> 1) + (cs[None, :] & 1)
    return rows, cols


def _tile_receipts(job: tuple) -> tuple:
    """One tile's receipts by segment's fold, (nw, NCT), from X's and W'_c's rows of the tile."""
    fold, seed, tile, tm, tn, xc, wc = job
    k = xc.shape[1]
    ks, nw = k // 32, k // (32 * S)
    inc = np.einsum("isk,jsk->sij", xc.reshape(128, ks, 32).astype(np.int64), wc.reshape(128, ks, 32).astype(np.int64))
    Ts = (np.cumsum(inc, axis=0) & 0xFFFFFFFF).astype(np.uint32)
    rows, cols = thread_entries()
    P = np.concatenate([np.zeros((1, NCT, 32), np.uint32), Ts[:, rows, cols]])   # P[j + 1] = T_j
    out = np.zeros((nw, NCT), np.uint32)
    for ctid in range(NCT):
        Pt = P[:, ctid, :].tolist()
        for w in range(nw):
            st = SG.fold_keys(fold, seed, tile, ctid, w, nw)
            for i in range(S + 1):
                st = SG.fold_absorb(fold, st, Pt[w * S + i])
            out[w, ctid] = SG.fold_final(fold, st)
    return fold, tile, out


def check(d: Path) -> dict:
    t0 = time.perf_counter()
    meta = json.loads((d / "meta.json").read_text())
    m, n, k, seed, calls = meta["m"], meta["n"], meta["k"], meta["seed"], meta["calls"]
    nx, nr, nw = counts(m, n, k)
    tiles, tiles_n = (m // 128) * (n // 128), n // 128
    px, pr = (d / "px.bin").read_bytes(), (d / "pr.bin").read_bytes()
    ks = px[4 + 3 * 128 + 64:4 + 3 * 128 + 128]
    dx, dr = domain("X", m, n, k, nx), domain("R", m, n, k, nr)
    out = {"dir": str(d), "m": m, "n": n, "k": k, "calls": calls, "grid": meta["grid"], "D": meta["D"], "nw": nw,
           "device_ok": meta["device_ok"], "device": meta["device"]}
    ok = bool(meta["device_ok"]) and meta["nw"] == nw and meta["xr"] == XR
    ok &= px == cref.params(nx, ROW_BITS, ks, domain_id=dx.domain_id) and pr == cref.params(nr, ROW_BITS, ks, domain_id=dr.domain_id)
    out["params_ok"] = bool(ok)
    X = (d / "X.bin").read_bytes()
    xrows = cref.permute(X, m, k, XR)
    assert len(xrows) == ROW * nx
    rng = random.Random(seed)
    trees, roots = [], set()
    for c in range(calls):
        offx, offr = (2 * c) << 40, (2 * c + 1) << 40
        refx = tree(xrows, ks, offx, dx)
        if c == 0 and (d / "cg_X.bin").exists():   # commit_gpu's Committer on the same X (PRF offset 0), and both salt phases
            cg = device_levels(d / "cg_X.bin")
            ds, cs = (d / "S_X.bin").read_bytes(), (d / "cgS_X.bin").read_bytes()
            out["committer_X"] = {"levels_ok": cg[:len(refx[1])] == refx[1], "salts_equal": ds == cs,
                                  "salts_bad": sum(ds[128 * i:128 * i + 128] != cs[128 * i:128 * i + 128] for i in range(nx)),
                                  "salts_first_bad": [i for i in range(nx) if ds[128 * i:128 * i + 128] != cs[128 * i:128 * i + 128]][:8]}
        for kn in meta["kernels"]:
            gx = device_levels(d / f"lv_{kn}_{c}_X.bin")
            trees.append({"kernel": kn, "call": c, **judge("X", dx, xrows, ks, offx, gx, refx, rng, 1)})
            rrows = (d / f"R_{kn}_{c}.bin").read_bytes()
            assert len(rrows) == ROW * nr
            gr = device_levels(d / f"lv_{kn}_{c}_R.bin")
            trees.append({"kernel": kn, "call": c, **judge("R", dr, rrows, ks, offr, gr, tree(rrows, ks, offr, dr), rng, 4)})
            roots |= {trees[-2]["root"], trees[-1]["root"]}
    out["trees"] = trees
    ok &= all(t["ok"] for t in trees)
    # every tree's root its own: X's per call (salts), R's per call and fold
    out["roots_distinct"] = len(roots) == calls * (1 + len(meta["kernels"]))
    ok &= out["roots_distinct"]
    picks = sorted({0, tiles - 1, rng.randrange(tiles), rng.randrange(tiles)})
    Xa = np.frombuffer(X, np.int8).reshape(m, k)
    jobs, wps = [], {}
    for c in range(calls):
        wps[c] = np.frombuffer((d / f"Wp_{c}.bin").read_bytes(), np.int8).reshape(n, k)
        for kn, fold in meta["kernels"].items():
            for tile in picks:
                tm, tn = divmod(tile, tiles_n)
                jobs.append((kn, c, (fold, seed, tile, tm, tn, Xa[128 * tm:128 * tm + 128], wps[c][128 * tn:128 * tn + 128])))
    with mp.get_context("fork").Pool() as pool:
        got = pool.map(_tile_receipts, [j[2] for j in jobs])
    rec = []
    for (kn, c, job), (fold, tile, want) in zip(jobs, got, strict=True):
        R = np.fromfile(d / f"R_{kn}_{c}.bin", np.uint32).reshape(tiles * nw, NCT)[tile * nw:(tile + 1) * nw]
        rec.append({"kernel": kn, "fold": fold, "call": c, "tile": tile, "words": int(want.size), "bad": int((R != want).sum())})
    out["receipts"] = rec
    ok &= all(r["bad"] == 0 for r in rec) and len(rec) == calls * len(meta["kernels"]) * len(picks)
    out["ok"] = bool(ok)
    out["ref_s"] = round(time.perf_counter() - t0, 1)
    return out


def main(argv: list[str]) -> int:
    if len(argv) >= 6 and argv[1] == "params":
        ks = random.Random(int(argv[argv.index("--ks-seed") + 1])).randbytes(64) if "--ks-seed" in argv else os.urandom(64)
        write_params(argv[2], int(argv[3]), int(argv[4]), int(argv[5]), ks)
        return 0
    if len(argv) >= 3 and argv[1] == "check":
        ok = True
        for a in argv[2:]:
            r = check(Path(a))
            print(json.dumps(r), flush=True)
            ok &= r["ok"]
        return 0 if ok else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
