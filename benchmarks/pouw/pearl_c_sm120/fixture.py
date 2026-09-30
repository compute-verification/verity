"""Pearl-C's sm_120 GPU check fixture, built off the card (needs verity and numpy): inputs at a small shape and every buffer
pearl_c_sm120.cu's kernels write, in its row-major layout, for forming v1 on `pearl_c_device.SM120` (G = 4) and
`SM120_UNPROMOTED`.  Forming, lines and hashes come from `verity_pouw.schemes.pearl_c` on the device record; the products
over every cell are batched in numpy with `verity.ml.kernels.group_sum_total_batch` on the record's atom and peel step
(the H100 fixture's helpers, `benchmarks/pouw/pearl_c/fixture.py`, do the rest), and one whole 64 x 64 tile per variant is
recomputed by the scheme (`encode`, `checked`, `digest`, `tickets`) and must agree.  run.py compares the device's buffers
with these bytes before it times anything.

  uv run python benchmarks/pouw/pearl_c_sm120/fixture.py OUT_DIR [m n k]     (default 256 384 1024; m, n multiples of 128)

Writes OUT_DIR/check.json (the shape, keys, and per buffer its file, dtype and SHA-256) and the buffers as raw little-endian
files: `common` (lines, inputs, forming, the plain and adopted baselines, row commitments) and per variant (`sm120`,
`sm120-unpromoted`) the chains and everything after them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pearl_c_h100_fixture", HERE.parent / "pearl_c" / "fixture.py")
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

from verity.commitments.turboshake import turboshake128  # noqa: E402
from verity.ml.kernels import group_sum_total_batch  # noqa: E402
from verity_pouw.protocol import Tile  # noqa: E402
from verity_pouw.schemes import pearl_c as C  # noqa: E402
from verity_pouw.schemes import pearl_kw as P  # noqa: E402
from verity_pouw.schemes.pearl_c_device import DEVICES, SM120, SM120_UNPROMOTED, Device  # noqa: E402

VARIANTS = (SM120, SM120_UNPROMOTED)
CASE = H.CASE


def rowmajor(x: np.ndarray) -> bytes:
    """pearl_c_sm120.cu's layout of a (rows, depth) byte matrix: row-major, depth contiguous (the layout function that
    replaces the H100's `tiled`)."""
    return np.ascontiguousarray(x, np.uint8).tobytes()


def chain_np(device: Device, a: np.ndarray, b: np.ndarray, ii: np.ndarray, jj: np.ndarray) -> np.ndarray:
    """C.chain on `device` for the cells (ii[c], jj[c]) of codes a (rows x k) and b."""
    k, total = a.shape[1], np.zeros(len(ii), np.uint32)
    window = device.window if device.g else k
    for ws in range(0, k, window):
        local = np.zeros(len(ii), np.uint32)
        for s in range(ws, min(ws + window, k), C.SLICE):
            local = group_sum_total_batch(device.atom, local, a[ii, s:s + C.SLICE], b[jj, s:s + C.SLICE])[0]
        if not device.g:
            return local
        total = (total.view(np.float32) + local.view(np.float32)).view(np.uint32)
        total[total == 0x80000000] = 0
    return total


def product(device: Device, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ii, jj = H.grid(a.shape[0], b.shape[0])
    return chain_np(device, a, b, ii, jj).reshape(a.shape[0], b.shape[0])


def peel_np(device: Device, c: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    ii, jj = H.grid(*c.shape)
    acc = c.ravel().copy()
    for s in range(0, C.PEEL_K, 16):
        acc = group_sum_total_batch(device.peel, acc, pa[ii, s:s + 16], pb[jj, s:s + 16])[0]
    return acc.reshape(c.shape)


def _side(job):
    device, rows, idx, side, seed_e, seed_b, k = job
    s = C.PearlC(k_min=32, tile=(16, 16), forming="v1", device=DEVICES[device])
    return s._side(rows, idx, side, seed_e, seed_b, k)


def form(pool: Pool, device: Device, rows: list[bytes], side: int, seed_e: bytes, seed_b: bytes, k: int, chunk: int = 16):
    jobs = [(device.name, rows[i:i + chunk], list(range(i, min(i + chunk, len(rows)))), side, seed_e, seed_b, k)
            for i in range(0, len(rows), chunk)]
    parts = pool.map(_side, jobs)
    built = parts[0][0]
    for p, _ in parts[1:]:
        for field in built.__slots__:
            getattr(built, field).extend(getattr(p, field))
    return built, [x for _, ep in parts for x in ep]


def build(out_dir: Path, m: int = 256, n: int = 384, k: int = 1024) -> dict:
    assert m % 128 == 0 and n % 128 == 0 and k % 1024 == 0, (m, n, k)
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = H.context(m, n, k)
    noise = C.PearlC(forming="v1", device=SM120).noise(ctx)
    manifest = {"schema": "verity-pouw/pearl-c-sm120-gpu-check/v1", "shape": [m, n, k], "case": CASE, "layout": "rowmajor",
                "keys": {"line_a": P._line_key(noise.seed_a).hex(), "line_b": P._line_key(noise.seed_b).hex(),
                         "jackpot": P.subkey("jackpot", noise.seed_a).hex()},
                "seeds": {"seed_a": noise.seed_a.hex(), "seed_b": noise.seed_b.hex()}, "variants": {}}
    o = H.Out(out_dir)
    lines = {name: np.array([P.sample_line(seed, side, 0, t) for t in range(count)], np.uint8)
             for name, seed, side, count in (("ea", noise.seed_a, 0, m), ("eb", noise.seed_b, 1, n))}
    for name, side in (("fa", 0), ("fb", 1)):
        lines[name] = np.array(C._basis(noise.seed_b, side, k, C.LINE_NORM_V1), np.uint8)
    for name, x in lines.items():
        o.put(f"{name}_codes", x, "u8")
    faT, fbT = lines["fa"].T.copy(), lines["fb"].T.copy()
    fbT128 = H.padded(fbT, 128)
    o.put("faT32", faT, "u8")
    o.put("fbT32", fbT, "u8")
    o.put("fbT128", fbT128, "u8")
    ra, rb = H.CV.rows_f32(CASE, "A", m, k), H.CV.rows_f32(CASE, "B", n, k)
    o.put("xa", b"".join(ra), "u8")
    o.put("xb", b"".join(rb), "u8")
    with Pool() as pool:
        a, e_a = form(pool, SM120, ra, 0, noise.seed_a, noise.seed_b, k)
        b, e_b = form(pool, SM120, rb, 1, noise.seed_b, noise.seed_b, k)
    inv = {}
    for tag, side in (("a", a), ("b", b)):
        o.put(f"alpha_{tag}", np.array(side.alpha, "<u4"), "u32")
        o.put(f"beta_{tag}", np.array(side.beta, "<u4"), "u32")
        o.put(f"aux_{tag}", np.array([[s, r] for s, r in zip(side.s, side.rho)], "<u4"), "u32")
        o.put(f"ep{tag}_codes", np.array(side.ep, np.uint8), "u8")
        inv[tag] = H.recip(np.array(side.alpha, np.uint32))
        o.put(f"ia_{tag}", inv[tag], "u32")
    at, bt = np.array(a.codes, np.uint8), np.array(b.codes, np.uint8)
    o.put("at", rowmajor(at), "u8")
    o.put("bt", rowmajor(bt), "u8")
    o.put("rows", H.row_commits(ra, H.F32_SEGMENT), "u8")
    fa_vals = np.frombuffer(b"".join(ra), "<f4").reshape(m, k)
    fb_vals = np.frombuffer(b"".join(rb), "<f4").reshape(n, k)
    qa, sa = H.plain(fa_vals)
    qb, sb = H.plain(fb_vals)
    o.put("qa", rowmajor(qa), "u8")
    o.put("qb", rowmajor(qb), "u8")
    o.put("sa", sa, "u32")
    o.put("sb", sb, "u32")
    o.put("y_plain", H.glue_y(product(SM120, qa, qb), sa, sb), "u16")
    o.put("y_fast", H.glue_y(product(SM120_UNPROMOTED, qa, qb), sa, sb), "u16")
    qa5, qb5 = H.adopted(fa_vals, a.alpha), H.adopted(fb_vals, b.alpha)
    o.put("qa_s5", rowmajor(qa5), "u8")
    o.put("qb_s5", rowmajor(qb5), "u8")
    o.put("y_adopt", H.glue_y(product(SM120, qa5, qb5), inv["a"], inv["b"]), "u16")
    manifest["common"] = o.files

    be_a, be_b = np.array(C.scaled_noise(a, e_a), np.uint16), np.array(C.scaled_noise(b, e_b), np.uint16)
    for device in VARIANTS:
        vdir = out_dir / device.name
        vdir.mkdir(exist_ok=True)
        v = H.Out(vdir)
        g = product(device, fbT128, faT)
        assert all(int(g[p, q]) == C.gram(noise.seed_b, k, C.LINE_NORM_V1, device)[p][q] for p, q in ((0, 0), (3, 17), (31, 31)))
        v.put("g", g, "u32")
        af, bf = product(device, at, fbT), product(device, bt, faT)
        v.put("af", af, "u32")
        v.put("bf", bf, "u32")
        pa, pb = H.pa_rows(be_a, af), H.pb_rows(be_b, bf, g)
        v.put("pa_t", rowmajor(H.bf16_bytes(pa)), "u8")
        v.put("pb_t", rowmajor(H.bf16_bytes(pb)), "u8")
        c = product(device, at, bt)
        u = peel_np(device, c, pa, pb)
        v.put("c", c, "u32")
        v.put("u", u, "u32")
        v.put("y", H.glue_y(u, inv["a"], inv["b"]), "u16")
        dg = H.digests(c, u)
        v.put("digests", dg, "u8")
        v.put("leaves", H.leaves(dg, m, n), "u8")
        v.put("tickets", H.tickets(c, noise.seed_a), "u8")
        agree = crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, dg, n)
        manifest["variants"][device.name] = {"scheme": C.PearlC(forming="v1", device=device).name, "g": device.g,
                                             "files": v.files, "crosscheck": agree}
    (out_dir / "check.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, dg, n) -> dict:
    """One 64 x 64 tile (rows 64-127, the last 64 columns) through the scheme on `device`: encode, checked, digest, tickets."""
    s = C.PearlC(forming="v1", device=device)
    rows, cols = range(64, 128), range(n - 64, n)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, [ra[i] for i in rows], [rb[j] for j in cols], Tile(tuple(rows), tuple(cols)))
    assert ops is not None
    checked = s.checked(ctx, ops)
    sub = lambda x: [[int(w) for w in row] for row in x[64:128, n - 64:n]]  # noqa: E731
    assert [list(r) for r in ops.a.codes] == at[64:128].tolist() and [list(r) for r in ops.b.codes] == bt[n - 64:].tolist()
    assert ops.pa == pa[64:128].tolist() and ops.pb == pb[n - 64:].tolist()
    assert checked.c == sub(c) and checked.u == sub(u)
    per = n // C.CHUNK
    assert s.admissible(ctx, ops)
    mine = turboshake128(b"".join(dg[r * per + per - 2 + h] for r in rows for h in range(2)), C.D_LEAF, 32)
    assert s.digest(ctx, noise, checked) == mine
    assert s.tickets(noise, checked) == H.tickets(np.array(checked.c, np.uint32), noise.seed_a)
    return {"tile": [64, n - 64], "cells": 64 * 64, "agree": True}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dims = [int(x) for x in sys.argv[2:5]] or [256, 384, 1024]
    man = build(Path(sys.argv[1]), *dims)
    total = sum(f["bytes"] for part in [man["common"], *(v["files"] for v in man["variants"].values())] for f in part.values())
    print(f"fixture {man['shape']} -> {sys.argv[1]}: {total / 1e6:.1f} MB, cross-check "
          + ", ".join(f"{k} {v['crosscheck']['cells']} cells" for k, v in man["variants"].items()))
