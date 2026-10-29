"""The device check's fixture for pearl_c4.cu: inputs and every buffer the kernels write, at a small shape, from the numpy
twin (`twin.py`) and `verity.core.protocols.pouw.schemes.pearl_c4`'s hashing.  dev.py's check compares each device buffer with these
byte for byte.

  uv run python benchmarks/pouw/pearl_c4/fixture.py OUT [--fmt nvfp4|mxfp4|both] [--shape MxNxK]

OUT/<fmt>/ gets one raw little-endian file per buffer (the layouts are pearl_c4.cu's header) and manifest.json: the shape,
the kernels' parameters, and each file's size and sha256.  Inputs are derived: word t of row i of operand X is the BF16 value
with sign b0 >> 7, mantissa b0 & 0x7F and exponent −4 + (b1 mod 6), (b0, b1) bytes 2t, 2t + 1 of
SHAKE256("verity/pouw/pearl-c4-fixture/v1", X, i).  The salt is a SHA-256 label; A's and B's roots are `audit.commit_rows`
over the rows (matmul 0, weight 1), and the seeds are the scheme's `noise` of that context, so the device check holds
pearl_c4.cu's commitments and seed derivation to the verifier's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from verity.ml.crypto.turboshake import turboshake128
from verity.core.protocols.pouw import audit as AU
from verity.core.protocols.pouw.protocol import Context, Matmul, Shape
from verity.core.protocols.pouw.schemes import pearl_c as C
from verity.core.protocols.pouw.schemes import pearl_c4 as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import twin as T

SHAPE = (128, 128, 1024)
WEIGHT = 1


def rows(name: str, count: int, k: int) -> np.ndarray:
    out = np.empty((count, k), np.uint16)
    for i in range(count):
        raw = np.frombuffer(hashlib.shake_256(f"verity/pouw/pearl-c4-fixture/v1\0{name}\0{i}".encode()).digest(2 * k),
                            np.uint8).astype(np.uint16)
        b0, b1 = raw[0::2], raw[1::2]
        out[i] = (b0 >> 7) << 15 | (123 + b1 % 6) << 7 | (b0 & 0x7F)
    return out


def raw_lines(seed: bytes, role: int, count: int) -> np.ndarray:
    blocks = b"".join(hashlib.sha512(F.LABEL_BLOCKS["line"] + seed + bytes([role]) + c.to_bytes(4, "little")).digest()
                      for c in range(-(-count // 4)))
    return np.frombuffer(blocks[:16 * count], np.uint8).reshape(count, 16)


def nibbles(raw: np.ndarray) -> np.ndarray:
    """(count, 16) line bytes -> (count, 32) codes, low nibble first."""
    return np.stack([raw & 0xF, raw >> 4], axis=2).reshape(raw.shape[0], 32)


def pack(codes: np.ndarray) -> np.ndarray:
    return (codes[:, 0::2] | codes[:, 1::2] << 4).astype(np.uint8)


def build(fmt_name: str, shape: tuple[int, int, int] = SHAPE) -> tuple[dict, dict[str, bytes]]:
    m, n, k = shape
    fmt = F.FORMATS[fmt_name]
    s = F.PearlC4(fmt_name)
    salt = hashlib.sha256(b"salt/pearl-c4-fixture").digest()
    xa, xb = rows("A", m, k), rows("B", n, k)
    matmul = Matmul(0, Shape(m, k, n), WEIGHT)
    tree_a = AU.commit_rows(AU.activation_binding(salt, matmul.index), matmul.index, [r.astype("<u2").tobytes() for r in xa],
                            "sha256", public=True)
    tree_b = AU.commit_rows(AU.weight_binding(WEIGHT), -2, [r.astype("<u2").tobytes() for r in xb], "sha256", public=True)
    ctx = Context(salt, matmul, tree_a.commitment.root, tree_b.commitment.root)
    s.check(ctx.matmul)
    noise = s.noise(ctx)
    seed_a, seed_b = noise.seed_a, noise.seed_b

    lea, leb = raw_lines(seed_a, F.ROLE_EA, m), raw_lines(seed_b, F.ROLE_EB, n)
    lfa, lfb = raw_lines(seed_b, F.ROLE_FA, k), raw_lines(seed_b, F.ROLE_FB, k)
    fa, fb = nibbles(lfa), nibbles(lfb)
    assert fa.tolist() == [list(ln) for ln in F.basis(seed_b, F.ROLE_FA, k)]
    jb = T.Job(fmt, seed_a, seed_b, fa, fb, None, np.full((F.R, k // fmt.block), fmt.one, np.uint8))
    gi, gj = T.cells(F.R, F.R)
    g = T.chain(fmt, fb.T.copy(), jb.ones, fa.T.copy(), jb.ones, gi, gj).reshape(F.R, F.R)

    a = T.form(fmt, xa.astype(np.uint32) << 16, nibbles(lea), fa)
    b = T.form(fmt, xb.astype(np.uint32) << 16, nibbles(leb), fb)
    ri, rq = T.cells(m, F.R)
    af = T.chain(fmt, a.codes, a.scales, fb.T.copy(), jb.ones, ri, rq).reshape(m, F.R)
    ci, cq = T.cells(n, F.R)
    bf = T.chain(fmt, b.codes, b.scales, fa.T.copy(), jb.ones, ci, cq).reshape(n, F.R)
    ea, eb = T.e_tilde(a), T.e_tilde(b)
    pa = np.concatenate([ea, T.rne_bf16(af)], axis=1)
    gl = g.tolist()
    mid = []
    for et, bfr in zip(eb.tolist(), bf.tolist()):
        row = []
        for q in range(F.R):
            acc = 0
            for p in range(F.R):
                acc = F.fp32.fma(et[p] << 16, gl[p][q], acc)
            row.append(C._bf16_of_word(F.fp32.add(acc, F.fp32.neg(bfr[q]))))
        mid.append(row)
    pb = np.concatenate([np.array(mid, np.uint16), eb ^ np.uint16(0x8000)], axis=1)

    ti, tj = T.cells(m, n)
    c = T.chain(fmt, a.codes, a.scales, b.codes, b.scales, ti, tj)
    u = c.copy()
    for st in range(0, F.PEEL_K, 16):
        u = T.group_sum_total_batch(T.HOPPER_BF16_M16N8K16, u, pa[ti, st:st + 16], pb[tj, st:st + 16])[0]
    c, u = c.reshape(m, n), u.reshape(m, n)

    inv_a = T.u32(np.float32(1) / T.f32(a.alpha))
    inv_b = T.u32(np.float32(1) / T.f32(b.alpha))
    unscale = lambda w: T.rne_bf16(T.u32((T.f32(w) * T.f32(inv_a)[:, None]) * T.f32(inv_b)[None, :]))  # noqa: E731

    digests = np.empty((m, n // C.CHUNK, 32), np.uint8)
    for i in range(m):
        for hc in range(n // C.CHUNK):
            cols = slice(C.CHUNK * hc, C.CHUNK * (hc + 1))
            digests[i, hc] = np.frombuffer(turboshake128(c[i, cols].astype("<u4").tobytes() + u[i, cols].astype("<u4").tobytes(),
                                                         C.D_MESSAGE, 32), np.uint8)
    leaves = np.empty((m // F.TILE, n // F.TILE, 32), np.uint8)
    for ti_ in range(m // F.TILE):
        for tj_ in range(n // F.TILE):
            rs, cs = slice(F.TILE * ti_, F.TILE * (ti_ + 1)), slice(F.TILE * tj_, F.TILE * (tj_ + 1))
            leaves[ti_, tj_] = np.frombuffer(C.leaf(C.Checked(c[rs, cs].tolist(), u[rs, cs].tolist())), np.uint8)

    le = lambda arr, dt: np.ascontiguousarray(arr, dt).tobytes()  # noqa: E731
    # pearl_c4.cu's scales are k-tile major: k/128 x rows x 128/block
    tiled = lambda s: s.reshape(s.shape[0], k // 128, 128 // fmt.block).transpose(1, 0, 2)  # noqa: E731
    files = {
        "xa": le(xa, "<u2"), "xb": le(xb, "<u2"), "salt": salt, "dom_a": tree_a.domain.domain_id,
        "dom_b": tree_b.domain.domain_id, "root_a": ctx.root_a, "root_b": ctx.root_b, "seed_a": seed_a, "seed_b": seed_b,
        "lines_ea": le(lea, np.uint8), "lines_eb": le(leb, np.uint8),
        "lines_fa": le(lfa, np.uint8), "lines_fb": le(lfb, np.uint8),
        "fat": le(pack(fa.T.copy()), np.uint8), "fbt": le(pack(fb.T.copy()), np.uint8),
        "a_codes": le(pack(a.codes), np.uint8), "a_scales": le(tiled(a.scales), np.uint8), "a_alpha": le(a.alpha, "<u4"),
        "a_inv": le(inv_a, "<u4"), "a_ae": le(a.ae, np.uint8),
        "b_codes": le(pack(b.codes), np.uint8), "b_scales": le(tiled(b.scales), np.uint8), "b_alpha": le(b.alpha, "<u4"),
        "b_inv": le(inv_b, "<u4"), "b_ae": le(b.ae, np.uint8),
        "pa": le(pa, "<u2"), "pb": le(pb, "<u2"), "g": le(g, "<u4"), "bf": le(bf, "<u4"),
        "c": le(c, "<u4"), "u": le(u, "<u4"), "y": le(unscale(u), "<u2"), "y_plain": le(unscale(c), "<u2"),
        "digests": le(digests, np.uint8), "leaves": le(leaves, np.uint8),
    }
    manifest = {
        "scheme": s.name, "fmt": fmt_name, "shape": {"m": m, "n": n, "k": k}, "block": fmt.block,
        "roles": {"ea": F.ROLE_EA, "fa": F.ROLE_FA, "eb": F.ROLE_EB, "fb": F.ROLE_FB},
        "admissible": bool(all(F.row_passes(fmt, [int(w) for w in x]) for x in np.concatenate([a.x, b.x]))),
        "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()},
    }
    return manifest, files


def write(out: Path, fmt_name: str, shape: tuple[int, int, int] = SHAPE) -> dict:
    manifest, files = build(fmt_name, shape)
    d = out / fmt_name
    d.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (d / f"{name}.bin").write_bytes(data)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--fmt", default="both", choices=("nvfp4", "mxfp4", "both"))
    ap.add_argument("--shape", default="x".join(map(str, SHAPE)))
    args = ap.parse_args()
    shape = tuple(int(v) for v in args.shape.split("x"))
    for fmt in (("nvfp4", "mxfp4") if args.fmt == "both" else (args.fmt,)):
        mf = write(args.out, fmt, shape)
        print(f"{fmt}: {len(mf['files'])} files, admissible {mf['admissible']}, {mf['scheme']} at {shape}")


if __name__ == "__main__":
    main()
