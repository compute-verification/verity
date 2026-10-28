"""lowreg's gate: every digest of `lowreg check TILES` (CTA 0, each kernel at 1 and 2 warpgroups) against hashlib.

A thread's row is its lane (thread % 128), its second chain's (lane + 64) % 128; the row's bytes are TILES passes over the
tile row lowreg.cu draws (splitmix64 from SEED, little-endian), hashed as sha512/row/v2: SHA-512(P(8 nbytes) || row), the
first digest also against `verity.ml.commitments.rowleaf.sha512_row_v2_digest` where it imports.

    python3 lowreg_check.py CHECK.jsonl    # a VERDICT line a kernel, then one for all; exit 1 on any mismatch
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys

MASK = (1 << 64) - 1


def splitmix(seed: int, i: int) -> int:
    z = (seed + (i + 1) * 0x9E3779B97F4A7C15) & MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return z ^ (z >> 31)


def tile_rows(seed: int, rows: int, row_bytes: int) -> list[bytes]:
    raw = b"".join(struct.pack("<Q", splitmix(seed, i)) for i in range(rows * row_bytes // 8))
    return [raw[r * row_bytes:(r + 1) * row_bytes] for r in range(rows)]


def row_v2(row: bytes) -> bytes:
    """SHA-512(P(8 nbytes) || row), P = ("verity/sha512-row/v2" NUL || u64be(8 nbytes)) zero-padded to 128 bytes."""
    p = (b"verity/sha512-row/v2\x00" + (8 * len(row)).to_bytes(8, "big")).ljust(128, b"\x00")
    return hashlib.sha512(p + row).digest()


def main(path: str) -> int:
    lines = [json.loads(s) for s in open(path) if s.startswith("{")]
    head, kernels = lines[0], [x for x in lines[1:] if "digests" in x]
    rows = tile_rows(head["seed"], 128, head["row_bytes"])
    want = [row_v2(r * head["tiles"]) for r in rows]
    try:
        from verity.ml.commitments import rowleaf

        ref = rowleaf.sha512_row_v2_digest(rows[0] * head["tiles"], 8 * len(rows[0]) * head["tiles"])
        print(f"reference: rowleaf.sha512_row_v2_digest {'equals' if ref == want[0] else 'DIFFERS FROM'} hashlib's P || row")
        bad_ref = ref != want[0]
    except ImportError as e:
        print(f"reference: rowleaf not imported ({e}); hashlib's P || row alone")
        bad_ref = False
    failed = 0
    for k in kernels:
        got = bytes.fromhex(k["digests"])
        n, c = k["threads"], k["chains"]
        ok = 0
        for t in range(n):
            lane = t % 128
            for ch in range(c):
                at = (t * c + ch) * 64
                ok += got[at:at + 64] == want[(lane + 64 * ch) % 128]
        good = ok == n * c
        failed += not good
        print(f"VERDICT check {k['var']} cap {k['cap']} wgs {k['wgs']}: {'PASS' if good else 'FAIL'} {ok}/{n * c} digests "
              f"equal hashlib's (tiles {head['tiles']}, regs {k['regs']}, local {k['local_bytes']} B)")
    print(f"VERDICT lowreg check: {'PASS' if not failed and not bad_ref and kernels else 'FAIL'} "
          f"{len(kernels) - failed}/{len(kernels)} kernels bit-exact against hashlib")
    return 1 if failed or bad_ref or not kernels else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
