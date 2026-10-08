"""Pearl-C's sm_120 GPU check fixture, built off the card (needs verity and numpy): inputs at a small shape and every buffer
pearl_c_sm120.cu's kernels write, in its row-major layout (A' and B~'s codes and the transposed F lines in K32 order along k,
`k32`; every product over k is the unpermuted operands'), for forming v1 on `pearl_c_device.SM120` (G = 4) and
`SM120_UNPROMOTED`.  Forming, lines and hashes come from `verity.core.protocols.pouw.schemes.pearl_c` on the device record; the products
over every cell are batched in numpy (the chains by the twin, `benchmarks/pouw/pearl_c/twin.py`, the peel with
`verity.ml.kernels.tc.group_sum_total_batch` on the record's peel step; the H100 fixture's helpers,
`benchmarks/pouw/pearl_c/fixture.py`, do the rest), and one whole 64 x 64 tile per variant is
recomputed by the scheme (`encode`, `checked`, `digest`) and must agree.  Pearl's tickets are not part of
`pearl-c-sm120` (no PoUW check reads them), so nothing here computes them.  run.py compares the device's buffers
with these bytes before it times anything.

  uv run python benchmarks/pouw/pearl_c_sm120/fixture.py OUT_DIR [m n k]     (default 256 384 1024; m, n multiples of 128)

The instance is a one-unit epoch as `verity.core.protocols.pouw.audit` starts it, hashing format -h1: B's frame-b3 row tree, the salt from
the beacon and B's root, A's frame-b3 row tree, and the noise from A's committed root (so E_A's line key is an output the
device derives, `line_a`).  Writes OUT_DIR/check.json (the shape, keys, seeds, and per buffer its file, dtype and SHA-256)
and the buffers as raw little-endian files: `common` (both row trees, seed_A and E_A's line key, lines, inputs, forming, the
plain and adopted baselines) and per variant (`sm120`, `sm120-unpromoted`) the chains and everything after them, through
the message digests, the tile leaves and the tile tree.
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import sys
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _load("pearl_c_h100_fixture", HERE.parent / "pearl_c" / "fixture.py")
W = _load("pearl_c_twin", HERE.parent / "pearl_c" / "twin.py")

from verity.ml.crypto.blake3 import blake3  # noqa: E402
from verity.ml.kernels.tc import group_sum_total_batch  # noqa: E402
from verity.core.protocols.pouw.audit import (ROW_SCHEMA, TILE_SCHEMA, Epoch, Workload, activation_binding, commit_rows,  # noqa: E402
                               rows_domain, transcript_domain, weight_binding)
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape, Tile  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_c as C  # noqa: E402
from verity.core.protocols.pouw.schemes import pearl_kw as P  # noqa: E402
from verity.core.protocols.pouw.schemes.pearl_c_device import (  # noqa: E402
    DEVICES,
    SM120,
    SM120_UNPROMOTED,
    Device,
)

VARIANTS = (SM120, SM120_UNPROMOTED)
CASE, INDEX, WEIGHT = H.CASE, H.INDEX, H.WEIGHT
BEACON = b"verity-pouw/pearl-c-sm120-gpu-check/beacon"


def unit_block(salt: bytes, root_b: bytes, index: int, hashing: str = "h1") -> bytes:
    """seed_line_a's p1: salt ‖ root_B ‖ u64 index ‖ BLAKE3("pearl-c/v0/seed-A") (seed_A's key); under -h3s s512_seeds' p3,
    salt ‖ root_B (64 bytes) ‖ u64 index."""
    head = salt + root_b + index.to_bytes(8, "little")
    return head if hashing == "h3s" else head + blake3(C.LABELS["seed-A"])


def epoch_of(matmul: Matmul, b_rows: list[bytes], beacon: bytes, hash: str) -> tuple[Epoch, bytes, bytes]:
    """A one-unit epoch on B's rows as `verity.core.protocols.pouw.audit` starts it: (epoch, B's tree bytes under tree hash `hash`, its
    root)."""
    dom_b = rows_domain(weight_binding(matmul.weight), -2, len(b_rows), hash)
    tree_b = H.tree_bytes(dom_b, b_rows, ROW_SCHEMA)
    return Epoch.unverified(Workload((matmul,)), {matmul.weight: tree_b[-32:]}, beacon), tree_b, tree_b[-32:]


def keys_of(scheme: C.PearlC, epoch: Epoch, matmul: Matmul, seed_b: bytes) -> dict[str, bytes]:
    """The pipeline's key inputs for a unit of `epoch` (run.Pipeline's `keys`; -h2's templates are run.b3s_template's): the
    domains in the scheme's tree hash and the digests' keys the scheme's (none under -h3s, whose salt key and hm96 constants
    the caller adds)."""
    s, salt, h, dk = matmul.shape, epoch.salt, scheme.commitment_hash, scheme.digest_keys
    return {"line_b": C.line_key(seed_b),
            "dom_a": rows_domain(activation_binding(salt, matmul.index), matmul.index, s.m, h).domain_id,
            "dom_b": rows_domain(weight_binding(matmul.weight), -2, s.n, h).domain_id,
            "dom_t": transcript_domain(epoch, len(scheme.tiles(s)), h).domain_id,
            **({"k_msg": dk["message"], "k_leaf": dk["leaf"]} if dk else {}),
            "unit": unit_block(salt, epoch.weight_roots[matmul.weight], matmul.index, scheme.hashing)}


def rowmajor(x: np.ndarray) -> bytes:
    """pearl_c_sm120.cu's layout of a (rows, depth) byte matrix: row-major, depth contiguous (the layout function that
    replaces the H100's `tiled`)."""
    return np.ascontiguousarray(x, np.uint8).tobytes()


def k32(x: np.ndarray) -> np.ndarray:
    """pearl_c_sm120.cu's K32 order along the last axis (form_s5's codes, the lines' transposed F lines): within each
    aligned 32, position 8q + 2b + s holds column 8b + 2q + s, the transpose of the 4 x 4 grid of column pairs (an
    involution, so the same gather undoes it)."""
    t = np.arange(x.shape[-1])
    return x[..., (t & ~31) | (t >> 1 & 3) << 3 | (t >> 3 & 3) << 1 | (t & 1)]


def product(device: Device, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C.chain on `device` for every cell of codes a (rows x k) and b, through the twin's batched chain."""
    ii, jj = H.grid(a.shape[0], b.shape[0])
    return W.chain(a, b, ii, jj, device=device).reshape(a.shape[0], b.shape[0])


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
    ra, rb = H.CV.rows_f32(CASE, "A", m, k), H.CV.rows_f32(CASE, "B", n, k)
    scheme = C.PearlC(forming="v1", device=SM120, hashing="h1")
    h = scheme.commitment_hash
    mm = Matmul(INDEX, Shape(m, k, n), WEIGHT)
    epoch, tree_b, root_b = epoch_of(mm, rb, BEACON, h)
    assert root_b == commit_rows(weight_binding(WEIGHT), -2, rb, h).commitment.root
    dom_a = rows_domain(activation_binding(epoch.salt, INDEX), INDEX, m, h)
    tree_a = H.tree_bytes(dom_a, ra, ROW_SCHEMA)
    assert tree_a[-32:] == commit_rows(activation_binding(epoch.salt, INDEX), INDEX, ra, h).commitment.root
    ctx = Context(epoch.salt, mm, tree_a[-32:], root_b)
    noise = scheme.noise(ctx)
    keys = keys_of(scheme, epoch, mm, noise.seed_b)
    dom_t = transcript_domain(epoch, len(scheme.tiles(mm.shape)), h)
    manifest = {"schema": "verity-pouw/pearl-c-sm120-gpu-check/v3", "shape": [m, n, k], "case": CASE, "layout": "rowmajor, k32",
                "hashing": scheme.hashing, "beacon": BEACON.hex(), "keys": {name: x.hex() for name, x in keys.items()},
                "seeds": {"seed_a": noise.seed_a.hex(), "seed_b": noise.seed_b.hex(), "salt": epoch.salt.hex(),
                          "root_a": tree_a[-32:].hex(), "root_b": root_b.hex()}, "variants": {}}
    o = H.Out(out_dir)
    o.put("tree_a", tree_a, "u8")
    o.put("tree_b", tree_b, "u8")
    o.put("seed_a", noise.seed_a, "u8")
    o.put("line_a", P._line_key(noise.seed_a), "u8")
    lines = {name: np.array([P.sample_line(seed, side, 0, t) for t in range(count)], np.uint8)
             for name, seed, side, count in (("ea", noise.seed_a, 0, m), ("eb", noise.seed_b, 1, n))}
    for name, side in (("fa", 0), ("fb", 1)):
        lines[name] = np.array(C._basis(noise.seed_b, side, k, C.LINE_NORM_V1), np.uint8)
    for name, x in lines.items():
        o.put(f"{name}_codes", x, "u8")
    faT, fbT = lines["fa"].T.copy(), lines["fb"].T.copy()
    fbT128 = H.padded(fbT, 128)
    o.put("faT32", k32(faT), "u8")
    o.put("fbT32", k32(fbT), "u8")
    o.put("fbT128", k32(fbT128), "u8")
    o.put("xa", b"".join(ra), "u8")
    o.put("xb", b"".join(rb), "u8")
    # fork: the workers unpickle `_side` from this module, which callers load from its path, not by an importable name
    with multiprocessing.get_context("fork").Pool() as pool:
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
    o.put("at", rowmajor(k32(at)), "u8")
    o.put("bt", rowmajor(k32(bt)), "u8")
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
        lv = H.leaves(dg, m, n)
        v.put("digests", dg, "u8")
        v.put("leaves", lv, "u8")
        v.put("tree_t", H.tree_bytes(dom_t, lv, TILE_SCHEMA), "u8")
        agree = crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, lv, n)
        manifest["variants"][device.name] = {"scheme": C.PearlC(forming="v1", device=device, hashing="h1").name, "g": device.g,
                                             "files": v.files, "crosscheck": agree}
    (out_dir / "check.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def crosscheck(device, ctx, ra, rb, at, bt, pa, pb, c, u, lv, n) -> dict:
    """One 64 x 64 tile (rows 64-127, the last 64 columns) through the scheme on `device` at -h1: encode, checked, digest."""
    s = C.PearlC(forming="v1", device=device, hashing="h1")
    rows, cols = range(64, 128), range(n - 64, n)
    noise = s.noise(ctx)
    ops = s.encode(ctx, noise, [ra[i] for i in rows], [rb[j] for j in cols], Tile(tuple(rows), tuple(cols)))
    assert ops is not None
    checked = s.checked(ctx, ops)
    sub = lambda x: [[int(w) for w in row] for row in x[64:128, n - 64:n]]  # noqa: E731
    assert [list(r) for r in ops.a.codes] == at[64:128].tolist() and [list(r) for r in ops.b.codes] == bt[n - 64:].tolist()
    assert ops.pa == pa[64:128].tolist() and ops.pb == pb[n - 64:].tolist()
    assert checked.c == sub(c) and checked.u == sub(u)
    assert s.admissible(ctx, ops)
    assert s.digest(ctx, noise, checked) == lv[(n // 64) + n // 64 - 1]
    return {"tile": [64, n - 64], "cells": 64 * 64, "agree": True}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    dims = [int(x) for x in sys.argv[2:5]] or [256, 384, 1024]
    man = build(Path(sys.argv[1]), *dims)
    total = sum(f["bytes"] for part in [man["common"], *(v["files"] for v in man["variants"].values())] for f in part.values())
    print(f"fixture {man['shape']} -> {sys.argv[1]}: {total / 1e6:.1f} MB, cross-check "
          + ", ".join(f"{k} {v['crosscheck']['cells']} cells" for k, v in man["variants"].items()))
