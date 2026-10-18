"""The CPU reference for candidates.cu's outputs, bit for bit.

    python3 check.py operand PATH N     N random 4 KiB leaves (a fixed seed) for `candidates dump`
    python3 check.py check DIR OPERAND  one dump against the reference (one JSON line; exit 1 on any difference)

The reference: SHA-512 at R rounds and SHA-256 in plain Python (FIPS 180-4's round and schedule, the constants derived
from the primes), each first held to hashlib (R = 80, and SHA-256) in the same run; BLAKE3 is verity.ml.crypto.blake3.
A dump's raw chains are checked at 64 threads of the timed grid (first, last, warp and block edges, random), every
compression of each chain; its 4 KiB-leaf tree at every leaf and node. The baseline (full SHA-512, sha512.cuh) is
checked the same way in every dump. Run from the source tree's root with PYTHONPATH=.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

import numpy as np

from verity.ml.crypto import blake3 as b3

M64, M32 = (1 << 64) - 1, (1 << 32) - 1


def primes(n: int) -> list[int]:
    out, k = [], 2
    while len(out) < n:
        if all(k % p for p in out if p * p <= k):
            out.append(k)
        k += 1
    return out


def iroot(x: int, k: int) -> int:
    lo, hi = 0, 1 << (x.bit_length() // k + 2)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if mid ** k <= x:
            lo = mid
        else:
            hi = mid - 1
    return lo


K512 = [iroot(p << 192, 3) & M64 for p in primes(80)]
IV512 = [iroot(p << 128, 2) & M64 for p in primes(8)]
K256 = [iroot(p << 96, 3) & M32 for p in primes(64)]
IV256 = [iroot(p << 64, 2) & M32 for p in primes(8)]


def _rotr(x: int, n: int, w: int, m: int) -> int:
    return ((x >> n) | (x << (w - n))) & m


def sha512_comp(st: list[int], w: list[int], rounds: int) -> list[int]:
    """One compression of `rounds` rounds; w is the 16-word ring the device leaves (the schedule overwrites it)."""
    r = lambda x, n: _rotr(x, n, 64, M64)  # noqa: E731
    a, b, c, d, e, f, g, h = st
    for t in range(rounds):
        i = t & 15
        if t >= 16:
            x2, x15 = w[(i - 2) & 15], w[(i - 15) & 15]
            w[i] = ((r(x2, 19) ^ r(x2, 61) ^ (x2 >> 6)) + w[(i - 7) & 15] + (r(x15, 1) ^ r(x15, 8) ^ (x15 >> 7)) + w[i]) & M64
        t1 = (h + (r(e, 14) ^ r(e, 18) ^ r(e, 41)) + ((e & f) ^ (~e & g)) + K512[t] + w[i]) & M64
        t2 = ((r(a, 28) ^ r(a, 34) ^ r(a, 39)) + ((a & b) ^ (a & c) ^ (b & c))) & M64
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M64, c, b, a, (t1 + t2) & M64
    return [(x + y) & M64 for x, y in zip(st, (a, b, c, d, e, f, g, h))]


def sha256_comp(st: list[int], w: list[int]) -> list[int]:
    r = lambda x, n: _rotr(x, n, 32, M32)  # noqa: E731
    a, b, c, d, e, f, g, h = st
    for t in range(64):
        i = t & 15
        if t >= 16:
            x2, x15 = w[(i - 2) & 15], w[(i - 15) & 15]
            w[i] = ((r(x2, 17) ^ r(x2, 19) ^ (x2 >> 10)) + w[(i - 7) & 15] + (r(x15, 7) ^ r(x15, 18) ^ (x15 >> 3)) + w[i]) & M32
        t1 = (h + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + K256[t] + w[i]) & M32
        t2 = ((r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & M32
        h, g, f, e, d, c, b, a = g, f, e, (d + t1) & M32, c, b, a, (t1 + t2) & M32
    return [(x + y) & M32 for x, y in zip(st, (a, b, c, d, e, f, g, h))]


def md(msg: bytes, block: int, word: int, iv: list[int], comp) -> bytes:
    """Merkle-Damgard with FIPS 180-4's padding (a 2-word length field)."""
    lenf = 2 * word
    total = -(-(len(msg) + 1 + lenf) // block) * block
    padded = msg + b"\x80" + b"\0" * (total - len(msg) - 1 - lenf) + (8 * len(msg)).to_bytes(lenf, "big")
    st = list(iv)
    for off in range(0, total, block):
        w = [int.from_bytes(padded[off + word * i: off + word * (i + 1)], "big") for i in range(16)]
        st = comp(st, w)
    return b"".join(x.to_bytes(word, "big") for x in st)


def hasher(kind: int, rounds: int):
    """(H(msg) -> digest bytes, raw chain(t, chain) -> final state words, word bytes) of a candidate."""
    if kind in (0, 1):
        r = 80 if kind == 0 else rounds
        h = (lambda m: hashlib.sha512(m).digest()) if r == 80 else (lambda m: md(m, 128, 8, IV512, lambda s, w: sha512_comp(s, w, r)))

        def chain(t: int, n: int) -> list[int]:
            st, w = list(IV512), [(t * 16 + i) & M64 for i in range(16)]
            for _ in range(n):
                for i in range(8):
                    w[i] ^= st[i]
                st = sha512_comp(st, w, r)
            return st
        return h, chain, 8
    if kind == 2:
        def chain(t: int, n: int) -> list[int]:
            st, w = list(IV256), [(t * 16 + i) & M32 for i in range(16)]
            for _ in range(n):
                for i in range(8):
                    w[i] ^= st[i]
                st = sha256_comp(st, w)
            return st
        return (lambda m: hashlib.sha256(m).digest()), chain, 4
    if kind in (3, 4):
        def chain(t: int, n: int) -> list[int]:
            cv, m = list(b3.IV), [(t * 16 + i) & M32 for i in range(16)]
            for k in range(n):
                for i in range(8):
                    m[i] ^= cv[i]
                cv = b3.compress(tuple(cv), b"".join(x.to_bytes(4, "little") for x in m), k, 64, 0)[:8]
            return cv
        return b3.blake3, chain, 4
    raise ValueError(kind)


def self_test() -> dict:
    rng = random.Random(1)
    msgs = [bytes(rng.getrandbits(8) for _ in range(n)) for n in (0, 3, 55, 56, 64, 111, 112, 127, 128, 129, 4096)]
    ok512 = all(md(m, 128, 8, IV512, lambda s, w: sha512_comp(s, w, 80)) == hashlib.sha512(m).digest() for m in msgs)
    ok256 = all(md(m, 64, 4, IV256, sha256_comp) == hashlib.sha256(m).digest() for m in msgs)
    okb3 = b3.blake3(b"").hex() == "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    return {"sha512_vs_hashlib": ok512, "sha256_vs_hashlib": ok256, "blake3_empty_vector": okb3}


def digests(path: str, count: int, word: int, be: bool) -> list[bytes]:
    raw = open(path, "rb").read()
    if len(raw) != count * 8 * word:
        raise SystemExit(f"{path}: {len(raw)} bytes, want {count * 8 * word}")
    if not be:
        return [raw[8 * word * i: 8 * word * (i + 1)] for i in range(count)]
    a = np.frombuffer(raw, dtype="<u8" if word == 8 else "<u4").reshape(count, 8)
    return [b"".join(int(x).to_bytes(word, "big") for x in row) for row in a]


def check_one(dump: str, operand: str, side: str, kind: int, rounds: int, meta: dict) -> dict:
    h, chain, word = hasher(kind, rounds)
    threads = meta["rate_threads"] * meta["rate_blocks"]
    raw = np.frombuffer(open(f"{dump}/rate-{side}.bin", "rb").read(), dtype="<u8" if word == 8 else "<u4").reshape(threads, 8)
    rng = random.Random(7)
    sample = sorted({0, 1, 31, 32, 255, 256, threads - 1, *(rng.randrange(threads) for _ in range(57))})
    raw_bad = [t for t in sample if [int(x) for x in raw[t]] != chain(t, meta["chain"])]
    n = meta["leaves"]
    op = open(operand, "rb").read()
    lv = digests(f"{dump}/levels-{side}.bin", 2 * n - 1, word, be=kind < 3)
    want = [h(op[4096 * j: 4096 * (j + 1)]) for j in range(n)]
    tree_bad, off, width = 0, 0, n
    while True:
        tree_bad += sum(a != b for a, b in zip(lv[off: off + width], want))
        if width == 1:
            break
        want = [h(want[2 * i] + want[2 * i + 1]) for i in range(width // 2)]
        off, width = off + width, width // 2
    return {"side": side, "raw_threads_checked": len(sample), "raw_mismatches": len(raw_bad), "tree_digests_checked": 2 * n - 1,
            "tree_mismatches": tree_bad, "root": lv[-1].hex(), "ok": not raw_bad and tree_bad == 0}


def main(argv: list[str]) -> int:
    if argv[1] == "operand":
        n = int(argv[3])
        with open(argv[2], "wb") as f:
            f.write(np.random.default_rng(20261018).bytes(4096 * n))
        return 0
    dump, operand = argv[2], argv[3]
    meta = json.load(open(os.path.join(dump, "meta.json")))
    st = self_test()
    base = check_one(dump, operand, "base", 0, 80, meta)
    cand = check_one(dump, operand, "cand", meta["cand_kind"], meta["cand_r"], meta)
    ok = all(st.values()) and base["ok"] and cand["ok"]
    print(json.dumps({"check": meta["candidate"], "sha_roll": meta["sha_roll"], "gpu": meta["gpu"], "self_test": st,
                      "base": base, "cand": cand, "ok": ok}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
