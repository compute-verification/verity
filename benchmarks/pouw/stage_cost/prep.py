#!/usr/bin/env python3
"""stage.cu's commitment constants and gate references, written before the lease.

For each operand width k (bytes a row) and row count R, the hm96-sha512/row-seg/v1 rows of a frame-v3-sha512 tree exactly as
served_zk's hm96_rows.cu commits them (rows_bench.Commit): the segment and outer prefixes, every leaf's and node's message
head, the pad leaves, and a gate: R rows and their salts from the OS, with Python's hm96 leaves, b ‖ c and root
(verity.ml.commitments). stage.cu loads a file, commits the rows on the device into buffers poisoned to 0xA5, and times
nothing unless its leaves, b ‖ c and root equal these, and a no-write control leaves them unequal.

  python3 prep.py DIR [--rows 2048,8192] [--widths 256,4096,6144,14336]

DIR/hm.bin: M_key's byte tables (192 x 256 entries of 64 bytes) and hm96's salt and leaf prefixes (two 128-byte blocks).
DIR/c_<k>_<R>.bin: a header of 8 little-endian u64 (magic, k, R, s, last, width, blocks, 0), then the s + 1 prefix
blocks, R leaf-head slots, width - 1 node-head slots (one if width is 1), (width - R) pad leaves, the R x k rows, R x 192
salts, and the references: R x 64 hm96 leaves, R x 128 b ‖ c, the 64-byte root. Then stage.cu's fast path's templates:
each frame leaf's whole padded message (two 128-byte blocks) with its 64-byte value zeroed and the value's byte offset
(R x 256, R x u32), and each node's (three blocks) with left and right zeroed and left's offset (right's is 72 later;
(width - 1) x 384, (width - 1) x u32), in node-head order.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import struct
import sys
import time
from pathlib import Path

from verity.ml.commitments import merkle as MK
from verity.ml.commitments import rowleaf as RL
from verity.ml.commitments.indexed import RangeIndexedDomain

HERE = Path(__file__).resolve().parent
MAGIC = 0x5354414745434F53
PREFIX = "stage-cost"


def rows_bench():
    spec = importlib.util.spec_from_file_location("served_zk_rows_bench", HERE.parent / "served_zk" / "rows_bench.py")
    mod = sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def write_hm(rb, out: Path) -> None:
    tables = b"".join(v.to_bytes(64, "little") for row in rb.INST.byte_tables(rb.KEY) for v in row)
    out.write_bytes(tables + rb.INST.salt_prefix + rb.INST.leaf_prefix(rb.KEY))


def padded(msg: bytes, size: int) -> bytes:
    """SHA-512's padding of msg (a 64-bit length, as hm96_rows.cu's sz_frame writes it) to exactly `size` bytes."""
    total = (len(msg) + 17 + 127) // 128 * 128
    if total != size:
        raise ValueError(f"a {len(msg)}-byte message pads to {total} bytes, not {size}")
    out = bytearray(msg) + b"\x80" + bytes(total - len(msg) - 1)
    out[-8:] = (8 * len(msg)).to_bytes(8, "big")
    return bytes(out)


def templates(leaf_heads: list[bytes], heads: list[bytes]) -> bytes:
    lt = b"".join(padded(h + bytes(64), 256) for h in leaf_heads)
    lo = b"".join(struct.pack("<I", len(h)) for h in leaf_heads)
    nt = b"".join(padded(h + bytes(64) + (64).to_bytes(8, "big") + bytes(64), 384) for h in heads)
    no = b"".join(struct.pack("<I", len(h)) for h in heads)
    return lt + lo + nt + no


def write_one(rb, k: int, rows: int, out: Path) -> dict:
    n_bits = 8 * k
    segs = RL.sha512_row_seg_segments(n_bits)
    if any(m % 1024 for m in segs):
        raise ValueError(f"hm96_rows.cu takes segments of whole 128-byte chunks, not {segs}")
    s, last = len(segs), segs[-1] // 8
    blocks = [RL.sha512_row_seg_segment_prefix(j, s, m) for j, m in enumerate(segs)] + [RL.sha512_row_seg_prefix(n_bits)]
    dom = MK.CommitmentDomain(hashlib.sha256(f"{PREFIX}/k{k}/r{rows}".encode()).digest(), -1, RangeIndexedDomain(0, rows),
                              hash="sha512")
    width = 1 << max(rows - 1, 0).bit_length()
    lh = [dom.leaf_prefix(r, r, rb.SEG, 64) for r in range(rows)]
    leaf_heads = rb.slots(lh)
    heads, w, lvl = [], width, 0
    while w > 1:
        heads += [rb.node_head(dom, lvl, j) for j in range(w // 2)]
        w, lvl = w // 2, lvl + 1
    node_heads = rb.slots(heads) if heads else bytes(rb.SLOT)
    tpl = templates(lh, heads) if heads else templates(lh, []) + bytes(384 + 4)
    pads = b"".join(dom.pad(r) for r in range(rows, width))
    data, salts = os.urandom(rows * k), os.urandom(192 * rows)
    xs = [RL.sha512_row_seg_digest(data[r * k:(r + 1) * k], n_bits) for r in range(rows)]
    bcs = [rb.INST.commit_string(rb.KEY, x, salts[192 * i:192 * (i + 1)]) for i, x in enumerate(xs)]
    hls = [rb.INST.tree_leaf(rb.KEY, bc) for bc in bcs]
    levels = MK.tree_levels(dom, [RL.row_leaf(dom, i, hl, rb.SEG) for i, hl in enumerate(hls)])
    head = struct.pack("<8Q", MAGIC, k, rows, s, last, width, len(blocks), 0)
    out.write_bytes(head + b"".join(blocks) + leaf_heads + node_heads + pads + data + salts + b"".join(hls) + b"".join(bcs)
                    + levels[-1][0] + tpl)
    return {"k": k, "rows": rows, "s": s, "last": last, "width": width, "root": levels[-1][0].hex()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--rows", default="2048,8192")
    ap.add_argument("--widths", default="256,4096,6144,14336")
    a = ap.parse_args()
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    rb = rows_bench()
    write_hm(rb, d / "hm.bin")
    for rows in (int(x) for x in a.rows.split(",")):
        for k in (int(x) for x in a.widths.split(",")):
            t = time.perf_counter()
            info = write_one(rb, k, rows, d / f"c_{k}_{rows}.bin")
            print(f"prep {info} in {time.perf_counter() - t:.1f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
