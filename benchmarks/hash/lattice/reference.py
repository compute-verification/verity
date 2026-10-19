"""The SIS hash's reference, independent of sis_bench.cu: derives A from its tag with hashlib and recomputes the digests
sis_bench dumped (NAME-L<level>-{rows.bin,digests.bin,index.txt}, NAME = ENC-m<M>-n<N>).

    python3 reference.py DUMPDIR        # one JSON line a dump; exit 1 on any mismatch
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ENCODINGS = {"q32": (4, 1), "q16": (2, 1), "q8n": (1, 2)}  # limbs L, digits a byte


def derive_a(enc: str, n: int, k: int) -> np.ndarray:
    limbs, _ = ENCODINGS[enc]
    tag = f"verity/hash-lattice/sis/v0|{enc}|{n}|{k}".encode()
    total = limbs * n * k
    stream = b"".join(hashlib.sha512(tag + c.to_bytes(8, "big")).digest() for c in range((total + 63) // 64))
    return np.frombuffer(stream[:total], dtype=np.int8).reshape(limbs * n, k)


def digits(enc: str, rows: np.ndarray) -> np.ndarray:
    """rows: (S, m) uint8 -> (S, K) int64 digits."""
    if ENCODINGS[enc][1] == 1:
        return rows.view(np.int8).astype(np.int64)
    out = np.empty((rows.shape[0], rows.shape[1] * 2), dtype=np.int64)
    out[:, 0::2] = rows & 15
    out[:, 1::2] = rows >> 4
    return out


def digest(enc: str, a: np.ndarray, n: int, rows: np.ndarray) -> np.ndarray:
    """(S, m) uint8 rows -> (S, n * L) uint8 digests, L bytes a word, little-endian."""
    limbs, _ = ENCODINGS[enc]
    prod = digits(enc, rows) @ a.astype(np.int64).T  # (S, L n), exact
    h = np.zeros((rows.shape[0], n), dtype=np.uint64)
    for limb in range(limbs):
        h += (prod[:, limb * n:(limb + 1) * n].astype(np.uint64) << np.uint64(8 * limb))
    h &= np.uint64((1 << (8 * limbs)) - 1)
    return h.astype("<u8").view(np.uint8).reshape(rows.shape[0], n, 8)[:, :, :limbs].reshape(rows.shape[0], -1)


def main() -> None:
    d = Path(sys.argv[1])
    bad = 0
    cache: dict = {}
    for idx in sorted(d.glob("*-index.txt")):
        stem = idx.name[: -len("-index.txt")]
        mt = re.fullmatch(r"(q32|q16|q8n)-m(\d+)-n(\d+)-L(\d+)", stem)
        enc, m, n, level = mt.group(1), int(mt.group(2)), int(mt.group(3)), int(mt.group(4))
        k = m * ENCODINGS[enc][1]
        key = (enc, n, k)
        if key not in cache:
            cache[key] = derive_a(enc, n, k)
        rows = np.fromfile(d / f"{stem}-rows.bin", dtype=np.uint8).reshape(-1, m)
        got = np.fromfile(d / f"{stem}-digests.bin", dtype=np.uint8).reshape(rows.shape[0], -1)
        want = digest(enc, cache[key], n, rows)
        mismatches = int((got != want).any(axis=1).sum())
        bad += mismatches
        print(json.dumps(dict(dump=stem, rows=int(rows.shape[0]), digest_bytes=int(got.shape[1]), mismatched_rows=mismatches,
                              a_sha256=hashlib.sha256(cache[key].tobytes()).hexdigest()[:16])))
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
