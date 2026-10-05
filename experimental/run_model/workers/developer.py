"""The developer's worker for the toy audit: it holds the claimed table, commits to it and opens rows on request.

It serves frames (8-byte big-endian length, then the bytes) on stdin and stdout, so the auditor's Lean program can drive
it. Untrusted in every mode: a wrong reply can only make the audit refuse.

    python developer.py --depth D [--corrupt F] [--seed S] [--echo]

* ``--corrupt F``: a fraction F of the rows are wrong, and the worker commits to the wrong table and opens it honestly.
* ``--echo``: reply to every frame with the frame itself (the latency probe).
"""
from __future__ import annotations

import argparse
import hashlib
import random
import struct
import sys


def want(i: int) -> bytes:
    return struct.pack(">Q", (i * i + 7) % 2**64)


def leaf(row: bytes) -> bytes:
    return hashlib.sha512(b"\x00" + row).digest()


def node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha512(b"\x01" + left + right).digest()


class Table:
    """The claimed rows, `corrupt` of them wrong, and the Merkle tree over them."""

    def __init__(self, depth: int, corrupt: float = 0.0, seed: int = 0) -> None:
        n = 2**depth
        rng = random.Random(seed)
        self.bad = set(rng.sample(range(n), round(corrupt * n)))
        self.rows = [struct.pack(">Q", 0) if i in self.bad else want(i) for i in range(n)]
        self.levels = [[leaf(r) for r in self.rows]]
        while len(self.levels[-1]) > 1:
            lv = self.levels[-1]
            self.levels.append([node(lv[2 * j], lv[2 * j + 1]) for j in range(len(lv) // 2)])

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    def opening(self, u: int) -> bytes:
        sibs = [self.levels[j][(u >> j) ^ 1] for j in range(len(self.levels) - 1)]
        return self.rows[u] + b"".join(sibs)


def read_frame(f) -> bytes | None:
    hdr = f.read(8)
    if len(hdr) < 8:
        return None
    (n,) = struct.unpack(">Q", hdr)
    return f.read(n)


def write_frame(f, b: bytes) -> None:
    f.write(struct.pack(">Q", len(b)) + b)
    f.flush()


def serve(table: Table | None) -> None:
    rd, wr = sys.stdin.buffer, sys.stdout.buffer
    while (req := read_frame(rd)) is not None:
        if table is None:
            write_frame(wr, req)
        elif req == b"register":
            write_frame(wr, table.root)
        elif req.startswith(b"open") and len(req) == 12:
            write_frame(wr, table.opening(struct.unpack(">Q", req[4:])[0]))
        else:
            write_frame(wr, b"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--corrupt", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--echo", action="store_true")
    a = ap.parse_args()
    serve(None if a.echo else Table(a.depth, a.corrupt, a.seed))


if __name__ == "__main__":
    main()
