"""Hold `pouw_hash_bench check`'s lines (host or device) to their references.

    python check.py LINES.jsonl [...]                  exit 0 only if every build's every row matches and none is missing
    python check.py --against REF.jsonl LINES.jsonl    the same against rows already checked (stdlib only: on the device's host)

The references: hashlib (SHA-256, SHAKE128/256), the `blake3` package and verity.ml.crypto.blake3, the repo's
TurboSHAKE128 (verity.ml.crypto.turboshake), Pearl-C's own `leaf` and Pearl's ticket (`hash_labelled(extract(...))`). The
three "stream" rows are the timed kernels' first iteration; SHA-256's and BLAKE3's are raw compressions (no padding), which
hashlib can't produce, so their SHA-256 compression is here, held to hashlib on a padded message first.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

M32, M64 = 0xFFFFFFFF, (1 << 64) - 1
PHI, PHI32 = 0x9E3779B97F4A7C15, 0x9E3779B9


def ptn(n: int) -> bytes:
    return bytes(i % 251 for i in range(n))


def mix32(x: int) -> int:
    x ^= x >> 16
    x = (x * 0x7FEB352D) & M32
    x ^= x >> 15
    x = (x * 0x846CA68B) & M32
    return x ^ (x >> 16)


def tile_word(which: int, r: int, c: int) -> int:
    return mix32((2 * (64 * r + c) + which) & M32) & 0xBFFFFFFF


def quad_word(a: int, row: int, col: int) -> int:
    return mix32(0xA5000000 ^ (a << 16 | row << 8 | col))


def le32(words) -> bytes:
    return struct.pack(f"<{len(words)}I", *words)


# ---- SHA-256's compression, for the raw chain -----------------------------------------------------------------------------

_K = [int(x, 16) for x in """
428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5 d807aa98 12835b01 243185be 550c7dc3 72be5d74 80deb1fe
9bdc06a7 c19bf174 e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da 983e5152 a831c66d b00327c8 bf597fc7
c6e00bf3 d5a79147 06ca6351 14292967 27b70a85 2e1b2138 4d2c6dfc 53380d13 650a7354 766a0abb 81c2c92e 92722c85 a2bfe8a1 a81a664b
c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070 19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3
748f82ee 78a5636f 84c87814 8cc70208 90befffa a4506ceb bef9a3f7 c67178f2""".split()]
SHA_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)


def _rotr(x: int, n: int) -> int:
    return (x >> n | x << (32 - n)) & M32


def sha256_compress(h: list[int], blk: list[int]) -> list[int]:
    w = list(blk)
    for t in range(16, 64):
        s0 = _rotr(w[t - 15], 7) ^ _rotr(w[t - 15], 18) ^ (w[t - 15] >> 3)
        s1 = _rotr(w[t - 2], 17) ^ _rotr(w[t - 2], 19) ^ (w[t - 2] >> 10)
        w.append((w[t - 16] + s0 + w[t - 7] + s1) & M32)
    a, b, c, d, e, f, g, hh = h
    for t in range(64):
        t1 = (hh + (_rotr(e, 6) ^ _rotr(e, 11) ^ _rotr(e, 25)) + ((e & f) ^ (~e & g)) + _K[t] + w[t]) & M32
        t2 = ((_rotr(a, 2) ^ _rotr(a, 13) ^ _rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) & M32
        a, b, c, d, e, f, g, hh = (t1 + t2) & M32, a, b, c, (d + t1) & M32, e, f, g
    return [(x + y) & M32 for x, y in zip(h, (a, b, c, d, e, f, g, hh))]


def _sha256_padded(m: bytes) -> bytes:
    p = m + b"\x80" + b"\0" * ((55 - len(m)) % 64) + struct.pack(">Q", 8 * len(m))
    h = list(SHA_IV)
    for i in range(0, len(p), 64):
        h = sha256_compress(h, list(struct.unpack(">16I", p[i:i + 64])))
    return struct.pack(">8I", *h)


def _chain_block(b: int) -> list[int]:
    z32 = PHI >> 32
    return [(z32 + (16 * b + i) * PHI32) & M32 for i in range(16)]


# ---- every row's expected output ------------------------------------------------------------------------------------------

def expected() -> dict[tuple, str]:
    """(what, len, d) -> hex, for every row `pouw_hash_bench check` prints."""
    import blake3 as blake3_pkg
    from verity.ml.crypto import blake3 as B3
    from verity.ml.crypto.turboshake import turboshake128
    from verity.ml.protocols.pouw.schemes import pearl_c as C
    from verity.ml.protocols.pouw.schemes import pearl_kw as P

    out: dict[tuple, str] = {}
    for n, d, olen in ((0, 0x1F, 32), (0, 0x1F, 200), (1, 0x1F, 32), (17, 0x01, 32), (167, 0x06, 32), (168, 0x1F, 32),
                       (169, 0x21, 64), (255, 0x21, 32), (256, 0x21, 32), (289, 0x0B, 32), (4096, 0x22, 32)):
        out["turboshake128 bytes", n, d, olen] = turboshake128(ptn(n), d, olen).hex()
    for i, nl in enumerate((0, 1, 20, 21, 22, 32, 41, 42, 63, 128, 160, 512)):
        d = 0x21 + i % 3
        out["turboshake128 lanes", nl, d] = turboshake128(ptn(8 * nl), d, 32).hex()
    for nl in (21, 42, 512):
        out["turboshake128 stream", nl, 0x22] = turboshake128(ptn(8 * nl), 0x22, 32).hex()
    out["turboshake128 other rotations", 21, 0x21] = turboshake128(ptn(168), 0x21, 32).hex()
    for n in (0, 135, 136, 137, 168, 1000):
        out["shake128", n, 0x1F] = hashlib.shake_128(ptn(n)).hexdigest(32)
        out["shake256", n, 0x1F] = hashlib.shake_256(ptn(n)).hexdigest(32)
    for n in (0, 1, 55, 56, 63, 64, 65, 119, 120, 1000):
        out["sha256 bytes", n, 0] = hashlib.sha256(ptn(n)).hexdigest()
    for nw in (0, 1, 13, 14, 16, 30, 64):
        out["sha256 words", nw, 0] = hashlib.sha256(ptn(4 * nw)).hexdigest()
    key = bytes(range(32))
    for n in (0, 1, 63, 64, 65, 104, 1023, 1024):
        plain, keyed = blake3_pkg.blake3(ptn(n)).digest(), blake3_pkg.blake3(ptn(n), key=key).digest()
        assert plain == B3.blake3(ptn(n)) and keyed == B3.keyed(key, ptn(n))
        out["blake3", n, 0], out["blake3 keyed", n, 0] = plain.hex(), keyed.hex()
    out["blake3 keyed block", 64, 0] = blake3_pkg.blake3(ptn(64), key=key).hexdigest()

    c = [[tile_word(0, r, col) for col in range(64)] for r in range(64)]
    u = [[tile_word(1, r, col) for col in range(64)] for r in range(64)]
    out["pearl-c digest 0", 256, 0x21] = turboshake128(le32(c[0][:32]) + le32(u[0][:32]), C.D_MESSAGE, 32).hex()
    out["pearl-c digest 127", 256, 0x21] = turboshake128(le32(c[63][32:]) + le32(u[63][32:]), C.D_MESSAGE, 32).hex()
    leaf = C.leaf(C.Checked(c, u)).hex()
    for what in ("pearl-c leaf", "pearl-c leaf stream", "pearl-c leaf quad", "pearl-c leaf narrow"):
        out[what, 4096, 0x22] = leaf
    out["pearl-c row segment", 1024, 0x23] = turboshake128(ptn(1024), 0x23, 32).hex()
    seed_a = bytes(0xC0 + i for i in range(32))
    for r0, c0 in ((0, 0), (48, 32)):
        sub = [[P.f32_of_bits(c[r0 + r][c0 + j]) for j in range(16)] for r in range(16)]
        out[f"pearl-c ticket {r0} {c0}", 64, 0] = P.hash_labelled(P.extract(sub), "jackpot", seed_a).hex()

    for lane in range(32):
        row, chunk = (lane >> 2) + 8 * (lane & 1), (lane >> 1) & 1
        msg = b"".join(le32([quad_word(a, row, 32 * chunk + j) for j in range(32)]) for a in range(2))
        out[f"quad lane {lane}", 256, 0x21] = turboshake128(msg, 0x21, 32).hex()
    msg = b"".join(((PHI + j * PHI) & M64).to_bytes(8, "little") for j in range(32))
    out["msg stream", 256, 0x21] = turboshake128(msg, 0x21, 32).hex()

    assert all(_sha256_padded(ptn(n)) == hashlib.sha256(ptn(n)).digest() for n in (0, 55, 56, 64, 119, 200))
    h = list(SHA_IV)
    for b in range(3):
        h = sha256_compress(h, _chain_block(b))
    out["sha stream 3 blocks", 192, 0] = struct.pack(">8I", *h).hex()
    cv = tuple(((PHI & M32) + i * PHI32) & M32 for i in range(8))
    for b in range(3):
        cv = tuple(B3.compress(cv, le32(_chain_block(b)), 0, 64, 0)[:8])
    out["blake3 stream 3 blocks", 192, 0] = le32(cv).hex()

    def fine(data: bytes) -> bytes:   # BLAKE3's tree over 64-byte leaves (a candidate form, not BLAKE3 of the bytes)
        cvs = [tuple(B3.compress(B3.IV, data[i:i + 64], i // 64, 64, B3.CHUNK_START | B3.CHUNK_END)[:8]) for i in range(0, len(data), 64)]
        while len(cvs) > 1:
            root = B3.ROOT if len(cvs) == 2 else 0
            cvs = [tuple(B3.compress(B3.IV, le32(cvs[i] + cvs[i + 1]), 0, 64, B3.PARENT | root)[:8]) for i in range(0, len(cvs), 2)]
        return le32(cvs[0])
    for n in (2048, 16384, 32768):
        out["blake3 tree", n, 0] = blake3_pkg.blake3(ptn(n)).hexdigest()
        out["blake3 fine tree", n, 0] = fine(ptn(n)).hex()
    for n in (2048, 32768):
        out["blake3 tree warp", n, 0] = blake3_pkg.blake3(ptn(n)).hexdigest()
    out["blake3 fine tree warp", 2048, 0] = fine(ptn(2048)).hex()
    out["blake3 fine tree block", 32768, 0] = fine(ptn(32768)).hex()
    out["leaf32 flat", 2048, 0x22] = turboshake128(ptn(2048), 0x22, 32).hex()
    subs = b"".join(turboshake128(ptn(2048)[512 * s:512 * (s + 1)], 0x24, 32) for s in range(4))
    out["leaf32 two-level", 2048, 0x25] = turboshake128(subs, 0x25, 32).hex()
    for lane in range(32):
        row, chunk = (lane >> 2) + 8 * (lane & 1), (lane >> 1) & 1
        msg = b"".join(le32([quad_word(a, row, 32 * chunk + j) for j in range(32)]) for a in range(2))
        out[f"quad b3 lane {lane}", 256, 0] = blake3_pkg.blake3(msg, key=key).hexdigest()
    return out


def row_key(row: dict) -> tuple:
    if row["what"] == "turboshake128 bytes":
        return row["what"], row["len"], row["d"], len(row["out"]) // 2
    return row["what"], row["len"], row["d"]


def check(rows: list[dict], want: dict[tuple, str] | None = None) -> list[dict]:
    """One summary per build (`where`): rows compared, the mismatches, the missing and the unexpected rows."""
    want = expected() if want is None else want
    builds: dict[str, dict] = {}
    for row in rows:
        if row.get("group") != "hash_check":
            continue
        b = builds.setdefault(row["where"], {"seen": set(), "bad": [], "extra": []})
        k = row_key(row)
        if k not in want:
            b["extra"].append(k)
        elif want[k] != row["out"] or k in b["seen"]:
            b["bad"].append(k)
        b["seen"].add(k)
    out = []
    for where, b in builds.items():
        missing = sorted(set(want) - b["seen"])
        out.append({"group": "hash_check_summary", "where": where, "rows": len(b["seen"]), "expected": len(want),
                    "bad": [list(k) for k in b["bad"]], "missing": [list(k) for k in missing],
                    "extra": [list(k) for k in b["extra"]],
                    "ok": not (b["bad"] or missing or b["extra"])})
    return out


def read(paths: list[str]) -> list[dict]:
    return [json.loads(line) for p in paths for line in Path(p).read_text().splitlines() if line.startswith("{")]


def main(argv: list[str]) -> int:
    want = None
    if argv[:1] == ["--against"]:
        want = {row_key(r): r["out"] for r in read(argv[1:2]) if r.get("group") == "hash_check"}
        argv = argv[2:]
    summaries = check(read(argv), want)
    for s in summaries:
        print(json.dumps(s))
    return 0 if summaries and all(s["ok"] for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
