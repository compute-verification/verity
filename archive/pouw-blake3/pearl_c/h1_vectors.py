"""The device gates of h1_bench.py, from the verifier's reference: `verity.ml.commitments`' frame-b3 and Pearl-C's h1 keys.

  uv run python benchmarks/pouw/pearl_c/h1_vectors.py OUT.json

Inputs are SHAKE128 streams by tag (h1_bench.py regenerates them with the standard library), so only the expected outputs
are written. The row leaves at 1 … 258 chunks (frame-b3 at #449's 6a1a051f: a one-block header, then the row) go through the `blake3` package for speed; every leaf of the first 16 chunk
counts is also held to frame-b3's own `CommitmentDomain.leaf`, and the trees are `MerkleTree`'s roots.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import blake3

from verity.ml.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.ml.commitments.merkle import b3s_level_key

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "verity" / "core" / "protocols" / "pouw"))
from verity.core.protocols.pouw.schemes.pearl_c import H1_KEYS  # noqa: E402

ROW, TILE, M, N = "verity/pouw/row/v1", "verity/pouw/tile/v1", 32, 8192
BINDING = bytes(range(32))


def shake(tag: str, n: int) -> bytes:
    return hashlib.shake_128(tag.encode()).digest(n)


def domain(owner: int, count: int) -> CommitmentDomain:
    return CommitmentDomain(BINDING, owner, RangeIndexedDomain(count), hash="blake3")


def rows_cases() -> dict:
    d = domain(3, 2)
    cases = []
    for c in range(1, 258):
        for rb in (1024 * c - 64, 1024 * c - 27, 1024 * c):     # exactly c chunks with the header, 37 bytes past, and 1 KB rows
            rows = shake(f"gate rows {rb}", 2 * rb)
            leaves = []
            for r in range(2):
                v = rows[r * rb:(r + 1) * rb]
                leaf = blake3.blake3(d.leaf_prefix(r, r, ROW, rb) + v, key=d.domain_id).digest()
                if c <= 16:
                    assert leaf == d.leaf(r, r, ROW, v)
                leaves.append(leaf.hex())
            cases.append({"rb": rb, "chunks": (64 + rb + 1023) // 1024, "leaves": leaves})
    return {"domain": d.domain_id.hex(), "cases": cases}


def tree_case(tag: str, rb: int) -> dict:
    d = domain(5, M)
    rows = shake(tag, M * rb)
    root = MerkleTree(d, {r: rows[r * rb:(r + 1) * rb] for r in range(M)}, lambda _: ROW).commitment.root
    return {"tag": tag, "m": M, "rb": rb, "domain": d.domain_id.hex(), "root": root.hex()}


def tile_case(rows: int) -> dict:
    """C~ and U of 32 rows (the gate tags): the digests, the leaves of the tiles over those rows (decode's partial 32 x 64
    tiles), and the tile tree's root."""
    c = shake("gate c", 4 * M * N) + bytes(4 * (rows - M) * N)
    u = shake("gate u", 4 * M * N) + bytes(4 * (rows - M) * N)
    per, tn = N // 32, N // 64
    k_msg, k_leaf = H1_KEYS["message"], H1_KEYS["leaf"]
    dg = [blake3.blake3(c[4 * (r * N + 32 * h):4 * (r * N + 32 * h + 32)] + u[4 * (r * N + 32 * h):4 * (r * N + 32 * h + 32)],
                        key=k_msg).digest() for r in range(rows) for h in range(per)]
    tiles = -(-rows // 64) * tn
    leaves = []
    for t in range(tiles):
        ti, tj = divmod(t, tn)
        joined = b"".join(dg[r * per + 2 * tj + h] for r in range(64 * ti, min(rows, 64 * ti + 64)) for h in range(2))
        leaves.append(blake3.blake3(joined, key=k_leaf).digest())
    d = domain(9, tiles)
    root = MerkleTree(d, dict(enumerate(leaves)), lambda _: TILE).commitment.root
    tile_leaves.cache[rows] = leaves
    return {"rows": rows, "n": N, "tiles": tiles, "domain": d.domain_id.hex(),
            "digests_sha256": hashlib.sha256(b"".join(dg)).hexdigest(), "leaves_sha256": hashlib.sha256(b"".join(leaves)).hexdigest(),
            "root": root.hex()}


def tile_leaves(case: dict) -> list[bytes]:
    return tile_leaves.cache[case["rows"]]


tile_leaves.cache = {}


def blk_case() -> dict:
    """The per-block Merkle row layout (hash_rows_blk; a candidate, not -h1): two rows of 32 KB, each a tree over its 512
    blocks' one-block keyed hashes under the row-leaf key, then nodes under level keys; the keys are the gate's own."""
    rb, key = 4 * N, shake("gate blk key", 32)
    levels = [shake(f"gate blk level {l}", 32) for l in range(9)]
    rows = shake("gate blk rows", 2 * rb)
    roots = []
    for r in range(2):
        level = [blake3.blake3(rows[r * rb + 64 * j:r * rb + 64 * j + 64], key=key).digest() for j in range(rb // 64)]
        lv = 0
        while len(level) > 1:
            level = [blake3.blake3(level[2 * i] + level[2 * i + 1], key=levels[lv]).digest() for i in range(len(level) // 2)]
            lv += 1
        roots.append(level[0].hex())
    return {"rb": rb, "key": key.hex(), "levels": [k.hex() for k in levels], "roots": roots}


def seg_cases(blk: dict) -> dict:
    """P1's prototype (hash_rows_seg; not -h1): blk_case's rows, keys and levels, each s-byte segment's leaf the standard keyed
    BLAKE3 of the segment. At s = 64 it is the block layout, so its roots must be blk_case's."""
    rb, key, levels = blk["rb"], bytes.fromhex(blk["key"]), [bytes.fromhex(k) for k in blk["levels"]]
    rows = shake("gate blk rows", 2 * rb)
    out = {}
    for sz in (64, 256):
        roots = []
        for r in range(2):
            level = [blake3.blake3(rows[r * rb + j:r * rb + j + sz], key=key).digest() for j in range(0, rb, sz)]
            lv = 0
            while len(level) > 1:
                level = [blake3.blake3(level[2 * i] + level[2 * i + 1], key=levels[lv]).digest() for i in range(len(level) // 2)]
                lv += 1
            roots.append(level[0].hex())
        out[str(sz)] = roots
    assert out["64"] == blk["roots"]
    return out


def h2_cases(tile: dict) -> dict:
    """Format -h2 (frame-b3s, `verity.ml.commitments.merkle`'s own leaf): two rows at each length at ranks 3 and 4 (short last
    segments, odd levels, a length that isn't a multiple of 16), A's m = 32 tree over FP32 rows, and the m = 32 tile tree
    over tile_case's leaves."""
    d = CommitmentDomain(BINDING, 3, RangeIndexedDomain(8), hash="blake3-s256")
    rows = []
    for rb in (256, 1000, 1300, 4100, 16384, 32768):
        v = shake(f"gate h2 rows {rb}", 2 * rb)
        rows.append({"rb": rb, "leaves": [d.leaf(3 + r, 3 + r, ROW, v[r * rb:(r + 1) * rb]).hex() for r in range(2)]})
    da = CommitmentDomain(BINDING, 5, RangeIndexedDomain(M), hash="blake3-s256")
    a = shake("gate tree fp32", M * 4 * N)
    root_a = MerkleTree(da, {r: a[r * 4 * N:(r + 1) * 4 * N] for r in range(M)}, lambda _: ROW).commitment.root
    dt = CommitmentDomain(BINDING, 9, RangeIndexedDomain(tile["tiles"]), hash="blake3-s256")
    return {"rows": {"domain": d.domain_id.hex(), "cases": rows,
                     "level_keys": [b3s_level_key(d.domain_id, l).hex() for l in range(10)]},
            "tree": {"domain": da.domain_id.hex(), "root": root_a.hex()},
            "tiles": {"domain": dt.domain_id.hex(), "root": MerkleTree(dt, dict(enumerate(tile_leaves(tile))), lambda _: TILE)
                      .commitment.root.hex()}}


def lat_chain_case(n: int = 512) -> dict:
    """lat_chain's last chaining value after n dependent keyed compressions (h1_bench's latency count), from
    `verity.ml.crypto.blake3.compress`."""
    from verity.ml.crypto.blake3 import IV, compress
    cv, m = tuple(IV), [(0x9E3779B9 * (i + 1)) & 0xFFFFFFFF for i in range(16)]
    for k in range(n):
        cv = tuple(compress(cv, b"".join(w.to_bytes(4, "little") for w in m), k, 64, 16)[:8])
        m[0] ^= cv[0]
    return {"n": n, "cv": b"".join(w.to_bytes(4, "little") for w in cv).hex()}


def decode_domains() -> dict:
    """The decode chain's two frame-b3 domains (A's 32 rows and the 128 tiles' tree), so that h1_bench's kernels hash under
    domain ids the verifier (verify_h1.py) rebuilds from the same binding and owner."""
    out = {}
    for name, tag, count, h in (("a", b"h1_bench decode A rows", M, "blake3"), ("t", b"h1_bench decode tiles", 128, "blake3"),
                                ("a2", b"h1_bench decode A rows", M, "blake3-s256"), ("t2", b"h1_bench decode tiles", 128, "blake3-s256"),
                                ("prefill_a2", b"h1_bench prefill A rows", 8192, "blake3-s256")):
        binding = hashlib.sha256(tag).digest()
        d = CommitmentDomain(binding, 0, RangeIndexedDomain(count), hash=h)
        out[name] = {"binding": binding.hex(), "owner": 0, "count": count, "hash": h, "domain": d.domain_id.hex()}
    return out


def main() -> None:
    blk, tile = blk_case(), tile_case(M)
    out = {"source": "frame-b3 (verity.ml.commitments.merkle, hash='blake3') and pearl_c.H1_KEYS",
           "rows": rows_cases(),
           "trees": {"A m=32 fp32 rows": tree_case("gate tree fp32", 4 * N), "A m=32 bf16 rows": tree_case("gate tree bf16", 2 * N)},
           "tiles": {"m32": tile}, "blk": blk, "seg": seg_cases(blk), "h2": h2_cases(tile), "lat_chain": lat_chain_case(),
           "decode": decode_domains()}
    Path(sys.argv[1]).write_text(json.dumps(out))
    print(f"{len(out['rows']['cases'])} row cases, {len(out['trees'])} trees, {len(out['tiles'])} tile case(s) -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
