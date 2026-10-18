"""hm96.cu's reference: its parameter file, and every leaf and level it computes, bit for bit.

    python3 hm96_ref.py params OUT CAND N   the parameter file of an N-leaf domain for base SHA-512 and candidate CAND
                                            (sha512, sha512r<R>, blake3k): each arm's templates, midstates and T table
    python3 hm96_ref.py check DIR           one `hm96 dump` (DIR/meta.json, levels-{base,cand}.bin, bc-{base,cand}.bin)
                                            against the reference (one JSON line; exit 1 on any difference)

The format is hm96-sha512/v1 leaves of sha512/row-seg/v1 rows (4 KiB: four 1 KiB segments) in a frame-v3-sha512 tree, as
verity.ml.commitments computes it; for a candidate primitive P the same structure with P in every hash position, sizes
scaled to P's digest (the hypothetical hm96-<P> format, README). The generic reference below computes both; at P =
SHA-512 it is held to verity.ml.commitments (rowleaf.sha512_row_seg_digest, hm96.SHA512, CommitmentDomain, tree_levels) in
every check, leaf by leaf and level by level. Up to 2048 leaves a check recomputes the whole tree; above, sampled leaves
(both ends, the nb = 3 positions, pads), and sampled nodes at every level from the dump's own children, the root included.
Run from the source tree's root with PYTHONPATH=.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check import IV512, md, sha512_comp  # noqa: E402

from verity.ml.commitments import hm96  # noqa: E402
from verity.ml.commitments.identity import identity_digest_sha512  # noqa: E402
from verity.ml.commitments.indexed import RangeIndexedDomain  # noqa: E402
from verity.ml.commitments.merkle import CommitmentDomain, domain_message, tree_levels  # noqa: E402
from verity.ml.commitments.rowleaf import (SCHEMA_HM96_SHA512_ROW_SEG, SHA512_ROW_SEG_SEGMENT_TAG, SHA512_ROW_SEG_TAG,  # noqa: E402
                                           row_leaf, sha512_row_seg_digest, sha512_row_seg_prefix)
from verity.ml.crypto import blake3 as b3  # noqa: E402

LEAF, SEG, NSEG = 4096, 1024, 4
FRAME = b"veritor/protocol/merkle/frame/v3\0"
SCHEMA = SCHEMA_HM96_SHA512_ROW_SEG.encode()
INST = hm96.SHA512
BINDING_TAG = "verity/hash-intpipe/hm96/v1"
#: the hypothetical blake3k arm's hash: BLAKE3 keyed by this public key (README)
B3_KEY = b3.blake3(b"verity/hash-intpipe/hm96-blake3k/hash-key/v1 (hypothetical)")
KS_SEED = 20261018
FILL_SEED = 7
MAXW, LEVELS = 48, 33


class Prim:
    """A primitive in every hash position: SHA-512 at `rounds` rounds (64-byte digests, 128-byte blocks, big-endian u64
    words) or keyed BLAKE3 (32-byte digests, 64-byte blocks, little-endian u32 words)."""

    def __init__(self, name: str):
        self.name = name
        self.b3 = name == "blake3k"
        self.rounds = 0 if self.b3 else (80 if name == "sha512" else int(name.removeprefix("sha512r")))
        self.nb, self.block, self.word = (32, 64, 4) if self.b3 else (64, 128, 8)
        self.salt = 2 * self.nb + 64
        self.keybytes = 3 * self.nb + 64

    def h(self, data: bytes) -> bytes:
        if self.b3:
            return b3.keyed(B3_KEY, data)
        if self.rounds == 80:
            return hashlib.sha512(data).digest()
        return md(data, 128, 8, IV512, lambda s, w: sha512_comp(s, w, self.rounds))

    def words(self, data: bytes) -> list[int]:
        order = "little" if self.b3 else "big"
        return [int.from_bytes(data[i:i + self.word], order) for i in range(0, len(data), self.word)]

    def iv(self) -> list[int]:
        return self.words(B3_KEY) if self.b3 else list(IV512)

    def first(self, message: bytes) -> list[int]:
        """The chaining value after the message's first block (the message has more than one block)."""
        blk = message[:self.block]
        if self.b3:
            return b3.compress(tuple(self.iv()), blk, 0, 64, b3.KEYED_HASH | b3.CHUNK_START)[:8]
        return sha512_comp(list(IV512), self.words(blk), self.rounds)

    def digest_of(self, raw: bytes) -> bytes:
        """A device digest (8 words, little-endian in memory) as the digest's bytes."""
        if self.b3:
            return raw
        return b"".join(struct.unpack_from("<Q", raw, 8 * i)[0].to_bytes(8, "big") for i in range(8))


def uint(v: int) -> bytes:
    return v.to_bytes(max(1, (v.bit_length() + 7) // 8), "big")


def message(tag: bytes, *parts: bytes) -> bytes:
    return FRAME + len(tag).to_bytes(4, "big") + tag + b"".join(len(p).to_bytes(8, "big") + p for p in parts)


def u64(v: int) -> bytes:
    return v.to_bytes(8, "big")


class Format:
    """hm96-<P> for one primitive: every prefix, the key, the domain id and each hash of the leaf and tree."""

    def __init__(self, p: Prim, n: int):
        self.p, self.n = p, n
        pad = lambda b: b.ljust(p.block, b"\0")  # noqa: E731
        self.q = pad(SHA512_ROW_SEG_TAG + u64(8 * LEAF) + u64(NSEG) + SEG.to_bytes(4, "big"))
        self.s = [pad(SHA512_ROW_SEG_SEGMENT_TAG + u64(j) + u64(NSEG) + u64(8 * SEG)) for j in range(NSEG)]
        blocks = -(-p.keybytes // p.nb)
        raw = bytearray(b"".join(p.h(INST.key_label + i.to_bytes(4, "big")) for i in range(blocks))[:p.keybytes])
        raw[-1] &= (1 << (8 * p.nb + 8 * p.salt - 1 - 8 * (p.keybytes - 1))) - 1
        self.key = bytes(raw)
        self.salt_prefix = pad(INST.salt_tag)
        self.leaf_prefix = pad(INST.leaf_tag + p.h(self.key))
        k = int.from_bytes(self.key, "little")
        rows = [0] * 256
        for v in range(1, 256):
            rows[v] = rows[v & (v - 1)] ^ (k >> ((v & -v).bit_length() - 1))
        self.trows = rows
        self.tbytes = b"".join(r.to_bytes(p.keybytes, "little") for r in rows)
        self.dom = domain(n)
        self.dm = domain_message(self.dom.binding, self.dom.owner, self.dom.positions.identity_sha512(), n, "sha512")
        self.did = p.h(self.dm)
        self.ks = random.Random(KS_SEED).randbytes(p.nb)

    def row_digest(self, row: bytes) -> bytes:
        return self.p.h(self.q + b"".join(self.p.h(self.s[j] + row[SEG * j:SEG * (j + 1)]) for j in range(NSEG)))

    def prf(self, pos: int) -> bytes:
        return b"".join(self.p.h(self.ks + u64(pos) + u64(r)) for r in range(self.p.salt // self.p.nb))

    def mask(self, y: bytes) -> bytes:
        acc = 0
        for i, v in enumerate(y):
            acc ^= self.trows[v] >> (8 * i)
        return (acc & ((1 << (8 * self.p.nb)) - 1)).to_bytes(self.p.nb, "little")

    def bc(self, x: bytes, y: bytes) -> bytes:
        return bytes(a ^ b for a, b in zip(x, self.mask(y))) + self.p.h(self.salt_prefix + y)

    def hleaf(self, bc: bytes) -> bytes:
        return self.p.h(self.leaf_prefix + bc)

    def frame_leaf(self, pos: int, h: bytes) -> bytes:
        return self.p.h(message(b"leaf", self.did, uint(pos), uint(pos), SCHEMA, h))

    def node(self, level: int, index: int, left: bytes, right: bytes) -> bytes:
        return self.p.h(message(b"node", self.did, uint(level), uint(index), left, right))

    def pad(self, rank: int) -> bytes:
        return self.p.h(message(b"pad", self.did, uint(rank)))

    def leaf(self, pos: int, row: bytes) -> tuple[bytes, bytes]:
        """(b || c, the frame leaf) at pos."""
        bc = self.bc(self.row_digest(row), self.prf(pos))
        return bc, self.frame_leaf(pos, self.hleaf(bc))

    def params(self) -> tuple[bytes, bytes]:
        """The arm's HP struct (hm96.cu: every field a u64, a BLAKE3 word in its low half) and its T table."""
        p, w = self.p, self.p.words
        nb_bytes = lambda v: len(uint(v))  # noqa: E731
        out = [*p.iv()]
        for j in range(NSEG):
            out += p.first(self.s[j])
        out += p.first(self.q) + p.first(self.salt_prefix) + p.first(self.leaf_prefix)
        for level in range(LEVELS):
            out += p.first(message(b"node", self.did, uint(level), b"\0", bytes(p.nb), bytes(p.nb)))
        out += w((self.ks + bytes(16)).ljust(p.block, b"\0"))

        def tpl(msg: bytes) -> list[int]:
            assert len(msg) <= MAXW * p.word
            return w(msg.ljust(MAXW * p.word, b"\0"))
        for name, make in (("leaf", lambda k: message(b"leaf", self.did, bytes(k), bytes(k), SCHEMA, bytes(p.nb))),
                           ("node", lambda k: message(b"node", self.did, b"\0", bytes(k), bytes(p.nb), bytes(p.nb))),
                           ("pad", lambda k: message(b"pad", self.did, bytes(k)))):
            for k in range(4):
                out += tpl(make(k)) if k else [0] * MAXW
        assert nb_bytes(0) == 1 and len(out) == 8 + 32 + 24 + 8 * LEVELS + 16 + 3 * 4 * MAXW
        return struct.pack(f"<{len(out)}Q", *out), self.tbytes


def domain(n: int) -> CommitmentDomain:
    return CommitmentDomain(identity_digest_sha512(BINDING_TAG, {"session": "hash-intpipe-hm96", "leaves": n}), -1,
                            RangeIndexedDomain(0, n), hash="sha512")


def fill_rows(pos: int, count: int = 1, seed: int = FILL_SEED) -> bytes:
    """hm96.cu's `fill` (candidates.cu's): byte i of the operand from a 32-bit mix of i."""
    i = np.arange(pos * LEAF, (pos + count) * LEAF, dtype=np.uint64).astype(np.uint32)
    with np.errstate(over="ignore"):
        x = i * np.uint32(0x9E3779B1) ^ np.uint32(seed)
        x ^= x >> np.uint32(15)
        x *= np.uint32(0x2C1B3C6D)
        x ^= x >> np.uint32(12)
        x *= np.uint32(0x297A2D39)
        x ^= x >> np.uint32(15)
    return (x & np.uint32(0xFF)).astype(np.uint8).tobytes()


def hold_to_verity(f: Format, pos: int, row: bytes, bc: bytes, leaf: bytes) -> None:
    """The SHA-512 arm's format is verity.ml.commitments' hm96-sha512/row-seg/v1 in frame-v3-sha512, bit for bit."""
    assert f.q == sha512_row_seg_prefix(8 * LEAF)
    assert f.key == INST.default_key and f.salt_prefix == INST.salt_prefix and f.leaf_prefix == INST.leaf_prefix(f.key)
    assert f.did == f.dom.domain_id
    x = sha512_row_seg_digest(row, 8 * LEAF)
    assert x == f.row_digest(row)
    want = INST.commit_string(f.key, x, f.prf(pos))
    assert want == bc, f"b || c at {pos}"
    assert row_leaf(f.dom, pos, INST.tree_leaf(f.key, want), SCHEMA_HM96_SHA512_ROW_SEG) == leaf, f"leaf at {pos}"


def width_of(n: int) -> tuple[int, int]:
    d = (n - 1).bit_length() if n > 1 else 0
    return 1 << d, d


def check_arm(f: Format, raw_levels: bytes, raw_bc: bytes, full: bool, rng: random.Random) -> dict:
    p, n = f.p, f.n
    width, depth = width_of(n)
    nbw = 8 * p.word
    off = [0]
    for d in range(depth + 1):
        off.append(off[-1] + (width >> d))
    if len(raw_levels) != nbw * off[-1] or len(raw_bc) != 2 * nbw * n:
        return {"ok": False, "error": f"sizes {len(raw_levels)} {len(raw_bc)}"}
    dig = lambda d, i: p.digest_of(raw_levels[nbw * (off[d] + i):nbw * (off[d] + i + 1)])  # noqa: E731
    bcat = lambda i: p.digest_of(raw_bc[2 * nbw * i:2 * nbw * i + nbw]) + p.digest_of(raw_bc[2 * nbw * i + nbw:2 * nbw * (i + 1)])  # noqa: E731
    bad: list[str] = []

    def check_bc(pos: int, bc: bytes) -> None:
        got = bcat(pos)
        bad.extend(f"{half} {pos}" for half, sl in (("b", slice(0, p.nb)), ("c", slice(p.nb, None))) if got[sl] != bc[sl])
    verity = p.name == "sha512"
    if full:
        leaves, held = [], 0
        for pos in range(n):
            row = fill_rows(pos)
            bc, leaf = f.leaf(pos, row)
            if verity:
                hold_to_verity(f, pos, row, bc, leaf)
                held += 1
            check_bc(pos, bc)
            leaves.append(leaf)
        levels = [leaves + [f.pad(r) for r in range(n, width)]]
        for d in range(depth):
            lv = levels[-1]
            levels.append([f.node(d, i // 2, lv[i], lv[i + 1]) for i in range(0, len(lv), 2)])
        if verity:
            assert [list(x) for x in tree_levels(f.dom, leaves)] == levels, "the generic tree is not tree_levels"
        for d in range(depth + 1):
            for i, want in enumerate(levels[d]):
                if dig(d, i) != want:
                    bad.append(f"level {d} node {i}")
        return {"ok": not bad, "full": True, "leaves_checked": n, "nodes_checked": off[-1], "held_to_verity": held,
                "bad_count": len(bad), "bad": bad[:8],
                "root": levels[depth][0].hex()}
    picks = sorted({0, 1, 255, 256, n // 2, n - 2, n - 1, *(q for q in (65535, 65536, 131072, n - 3) if q < n),
                    *rng.sample(range(n), 24)})
    held = 0
    for pos in picks:
        row = fill_rows(pos)
        bc, leaf = f.leaf(pos, row)
        if verity:
            hold_to_verity(f, pos, row, bc, leaf)
            held += 1
        check_bc(pos, bc)
        if dig(0, pos) != leaf:
            bad.append(f"leaf {pos}")
    pads = sorted({n, width - 1, *rng.sample(range(n, width), min(16, width - n))}) if width > n else []
    for r in pads:
        if dig(0, r) != f.pad(r):
            bad.append(f"pad {r}")
    nodes = 0
    for d in range(1, depth + 1):
        m = width >> d
        for i in sorted({0, m - 1, *(q for q in ((n - 1) >> d, -(-n >> d), 65536) if q < m), *rng.sample(range(m), min(m, 8))}):
            nodes += 1
            if dig(d, i) != f.node(d - 1, i, dig(d - 1, 2 * i), dig(d - 1, 2 * i + 1)):
                bad.append(f"level {d} node {i}")
    return {"ok": not bad, "full": False, "leaves_checked": len(picks), "pads_checked": len(pads), "nodes_checked": nodes,
            "held_to_verity": held, "bad_count": len(bad), "bad": bad[:8], "root": dig(depth, 0).hex()}


def main(argv: list[str]) -> int:
    if len(argv) == 5 and argv[1] == "params":
        n = int(argv[4])
        (hp0, t0), (hp1, t1) = Format(Prim("sha512"), n).params(), Format(Prim(argv[3]), n).params()
        with open(argv[2], "wb") as fh:
            fh.write(b"HMP1" + hp0 + hp1 + t0 + t1)
        return 0
    if len(argv) == 3 and argv[1] == "check":
        d = argv[2]
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        n = meta["leaves"]
        out = {"candidate": meta["candidate"], "leaves": n, "gpu": meta["gpu"]}
        ok = True
        for arm, name in (("base", "sha512"), ("cand", meta["candidate"])):
            f = Format(Prim(name), n)
            with open(os.path.join(d, f"levels-{arm}.bin"), "rb") as fh:
                lv = fh.read()
            with open(os.path.join(d, f"bc-{arm}.bin"), "rb") as fh:
                bc = fh.read()
            r = check_arm(f, lv, bc, n <= 2048, random.Random(n))
            out[arm] = r
            ok &= r["ok"]
        print(json.dumps({**out, "ok": ok}))
        return 0 if ok else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
