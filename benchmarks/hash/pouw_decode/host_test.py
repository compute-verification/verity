#!/usr/bin/env python3
"""pd.cuh on the host (pd_host.cpp's build), held byte for byte to Pearl-C's own kernels built beside it (s512_row_segs and
s512_folds, hm_leaves, s512_frame_leaves, s512_pads, s512_tree_small, s512_seeds, s512_tile_segs) and to the Python
references (refs.py), at row counts with and without pads and segment counts odd and even, every pd_fold variant and
both pd_commit modes; every pd output starts poisoned.  pd_probe's forms are held to each other.  Exit 1 on any difference.

  python3 host_test.py [SO]   (default: builds pd_host.so with g++ into a temporary directory)
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import refs  # noqa: E402
from verity.core.protocols.pouw.audit import hm96_kernel_constants  # noqa: E402

RS_MAX_SEGS, POISON, INDEX = 128, 0xA5, 5
VARIANTS = (0, 1, 2, 3)   # pd_fold's: rs_fold, rs_rounds over precomputed schedules, then both with pd_round


class Args(C.Structure):
    _fields_ = [("p", C.c_uint64 * 16), ("i", C.c_int32 * 16)]


def build() -> str:
    out = os.path.join(tempfile.mkdtemp(), "pd_host.so")
    subprocess.run(["g++", "-O2", "-std=c++20", "-pthread", "-shared", "-fPIC", "-Wno-unknown-pragmas", "-o", out,
                    os.path.join(HERE, "pd_host.cpp")], check=True)
    return out


class Host:
    def __init__(self, so: str):
        self.lib = C.CDLL(so)
        self.lib.pd_run.argtypes = [C.c_char_p, C.c_int, C.c_int, C.POINTER(Args)]
        self.keep = []
        self.run("s512_heads", -(-(RS_MAX_SEGS * (RS_MAX_SEGS + 1) // 2 + RS_MAX_SEGS + 1) // 128), 128)
        self.run("pd_heads", 1, 32)

    def buf(self, data: bytes | int, fill: int = 0) -> np.ndarray:
        a = np.frombuffer(data, np.uint8).copy() if isinstance(data, bytes) else np.full(max(data, 16), fill, np.uint8)
        self.keep.append(a)
        return a

    def run(self, name: str, grid: int, block: int, p=(), i=()) -> None:
        a = Args()
        for j, v in enumerate(p):
            a.p[j] = v.ctypes.data if isinstance(v, np.ndarray) else v
        for j, v in enumerate(i):
            a.i[j] = v
        if self.lib.pd_run(name.encode(), grid, block, C.byref(a)):
            raise RuntimeError(f"pd_run {name}")


def pow2(n: int) -> int:
    return 1 << max(n - 1, 0).bit_length()


def keys(m16: int) -> dict[str, bytes]:
    shake = lambda tag, n: hashlib.shake_256(f"pouw-decode/host/{tag}".encode()).digest(n)  # noqa: E731
    salt, root_b = shake("salt", 32), shake("root_b", 64)
    return {"salt": salt, "root_b": root_b, "salt_key": shake("salt-key", 32), "unit": salt + root_b + INDEX.to_bytes(8, "little")}


def reference_commit(h: Host, x, m16: int, k: int, kb, dom, hm) -> dict[str, bytes]:
    """Pearl-C's launches for A's commitment (Pipeline.row_tree, seed_a, seeds_a), host-built."""
    s, width = k // 256, pow2(m16)
    seg, rowdig = h.buf(64 * s * m16), h.buf(64 * m16)
    tree = h.buf(64 * (2 * width - 1))
    unit, sk = h.buf(kb["unit"]), h.buf(kb["salt_key"])
    h.run("s512_row_segs", -(-m16 * s // 128), 128, p=(x, seg), i=(m16, k))
    h.run("s512_folds", -(-m16 // 32), 32, p=(seg, rowdig), i=(m16, s, m16, s))
    if width > m16:
        h.run("s512_pads", -(-(width - m16) // 64), 64, p=(tree, dom), i=(m16, width))
    h.run("hm_leaves", -(-m16 // 64), 64, p=(rowdig, tree, sk, hm, 0, unit.ctypes.data + 96), i=(m16, 0, 0))
    h.run("s512_frame_leaves", -(-m16 // 64), 64, p=(tree, dom), i=(0, m16))
    if width > 1:
        h.run("s512_tree_small", 1, 64, p=(tree, dom), i=(width, 0))
    seed_a, line_a, seeds, lkeys = h.buf(32), h.buf(32), h.buf(32 * m16), h.buf(32 * m16)
    root = tree.ctypes.data + tree.size - 64 if width > 1 else tree.ctypes.data
    h.run("s512_seeds", 1, 128, p=(root, seed_a, line_a, unit), i=(1, 0, 0))
    h.run("s512_seeds", -(-m16 // 128), 128, p=(rowdig, seeds, lkeys, unit), i=(m16, 0, 1))
    return {"rowdig_a": rowdig.tobytes(), "tree_a": tree.tobytes()[:64 * (2 * width - 1)], "seed_a": seed_a.tobytes()[:32],
            "line_a": line_a.tobytes()[:32], "seeds_a": seeds.tobytes()[:32 * m16], "lkeys_a": lkeys.tobytes()[:32 * m16]}


def pd_commit(h: Host, x, m16: int, k: int, kb, dom, hm, mode: int, variant: int) -> dict[str, bytes]:
    s, width = k // 256, pow2(m16)
    out = {"rowdig_a": h.buf(64 * m16, POISON), "tree_a": h.buf(64 * (2 * width - 1), POISON), "seeds_a": h.buf(32 * m16, POISON),
           "lkeys_a": h.buf(32 * m16, POISON), "seed_a": h.buf(32, POISON), "line_a": h.buf(32, POISON)}
    scratch = h.buf(16)
    unit, sk = h.buf(kb["unit"]), h.buf(kb["salt_key"])
    b = out
    p = (x, b["rowdig_a"], b["tree_a"], b["seeds_a"], b["lkeys_a"], unit, dom, sk, hm, scratch, b["seed_a"], b["line_a"])
    h.run("pd_commit", m16, 32 * (-(-s // 32) + 2), p=p, i=(m16, k, mode, variant))
    if mode == 1:
        h.run("pd_tree", 1, 128, p=(0, 0, b["tree_a"], 0, 0, unit, dom, 0, 0, 0, b["seed_a"], b["line_a"]), i=(m16,))
    if int.from_bytes(scratch[:4].tobytes(), "little"):
        raise RuntimeError("pd_commit left its counter set")
    return {name: v.tobytes()[:len(v) if name not in ("seed_a", "line_a") else 32] for name, v in out.items()}


def diff(name: str, got: bytes, want: bytes) -> str | None:
    if got == want:
        return None
    bad = [i for i in range(0, len(want), 64) if got[i:i + 64] != want[i:i + 64]]
    return f"{name}: {len(bad)} of {len(want) // 64 or 1} 64-byte words differ, first at {bad[:4]} ({len(got)} vs {len(want)} bytes)"


def main(argv: list[str]) -> int:
    h = Host(argv[1] if len(argv) > 1 else build())
    hm = h.buf(hm96_kernel_constants())
    rng = np.random.default_rng(7)
    fails, cases = [], 0
    for m16, k in ((32, 4096), (48, 1280), (128, 1024), (16, 2048), (1, 1024)):
        kb = keys(m16)
        rows = rng.standard_normal((m16, k), dtype=np.float32)
        x = h.buf(rows.tobytes())
        py = refs.commit([rows[r].tobytes() for r in range(m16)], kb["salt"], kb["root_b"], kb["salt_key"], INDEX)
        dom = h.buf(py["dom_a"])
        ref = reference_commit(h, x, m16, k, kb, dom, hm)
        for name, want in ref.items():
            if (e := diff(f"m16={m16} k={k} reference {name} vs Python", want, py[name])):
                fails.append(e)
        for mode, variant in ((m, v) for m in (0, 1) for v in VARIANTS):
            got = pd_commit(h, x, m16, k, kb, dom, hm, mode, variant)
            for name, want in ref.items():
                cases += 1
                if (e := diff(f"m16={m16} k={k} mode {mode} variant {variant} {name}", got[name], want)):
                    fails.append(e)
        print(f"commit m16={m16} k={k}: done, {len(fails)} failures so far", flush=True)
    for rows_, cols in ((32, 128), (48, 128), (64, 192), (128, 128), (16, 64)):
        c = rng.integers(0, 1 << 32, (rows_, cols), dtype=np.uint32)
        u = rng.integers(0, 1 << 32, (rows_, cols), dtype=np.uint32)
        cb, ub = h.buf(c.tobytes()), h.buf(u.tobytes())
        tn, tiles = cols // 64, -(-rows_ // 64) * (cols // 64)
        want = b"".join(refs.tile_leaf(c, u, t // tn, t % tn, rows_) for t in range(tiles))
        full, last = rows_ // 64 * tn, rows_ % 64 // 2
        seg, leaves = h.buf(64 * (32 * full + tn * last)), h.buf(64 * tiles, POISON)
        h.run("s512_tile_segs", -(-(32 * full + tn * last) // 128), 128, p=(cb, ub, seg), i=(rows_, cols))
        h.run("s512_folds", -(-tiles // 32), 32, p=(seg, leaves), i=(tiles, 32, full, last))
        if (e := diff(f"tiles {rows_}x{cols} reference vs Python", leaves.tobytes(), want)):
            fails.append(e)
        for variant in VARIANTS:
            out = h.buf(64 * tiles, POISON)
            h.run("pd_tiles", tiles, 32, p=(cb, ub, out), i=(rows_, cols, 0, variant))
            cases += 1
            if (e := diff(f"tiles {rows_}x{cols} variant {variant}", out.tobytes(), want)):
                fails.append(e)
        print(f"tiles {rows_}x{cols}: done, {len(fails)} failures so far", flush=True)
    for n in (1, 3):
        st = {}
        for v in range(5):
            out = h.buf(17 * 8, POISON)
            h.run("pd_probe", 1, 32, p=(out,), i=(n, v))
            st[v] = out.tobytes()
        cases += 1
        for a, b, what in ((0, 2, "pd_compress vs rs_compress"), (1, 3, "pd_rounds vs rs_rounds")):
            if st[a][8:72] != st[b][8:72]:
                fails.append(f"probe n={n}: {what}")
        if st[4][8:72] != st[0][8:72] or st[4][72:136] == st[4][8:72]:
            fails.append(f"probe n={n}: two chains")
    print(f"probe: done, {len(fails)} failures so far", flush=True)
    for f in fails:
        print("FAIL", f)
    print(f"host_test: {cases} pd comparisons, {len(fails)} failures")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
