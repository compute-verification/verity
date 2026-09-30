"""The device gates of h1_bench.py, from the verifier's reference: `verity.commitments`' frame-b3 and Pearl-C's h1 keys.

  uv run python benchmarks/pouw/pearl_c/h1_vectors.py OUT.json

Inputs are SHAKE128 streams by tag (h1_bench.py regenerates them with the standard library), so only the expected outputs
are written. The row leaves at 1 … 258 chunks go through the `blake3` package for speed; every leaf of the first 16 chunk
counts is also held to frame-b3's own `CommitmentDomain.leaf`, and the trees are `MerkleTree`'s roots.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import blake3

from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "protocols" / "pouw"))
from verity_pouw.schemes.pearl_c import H1_KEYS  # noqa: E402

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
        for rb in (1024 * c - 37, 1024 * c):
            rows = shake(f"gate rows {rb}", 2 * rb)
            leaves = []
            for r in range(2):
                v = rows[r * rb:(r + 1) * rb]
                leaf = blake3.blake3(d.leaf_prefix(r, r, ROW, rb) + v, key=d.domain_id).digest()
                if c <= 16:
                    assert leaf == d.leaf(r, r, ROW, v)
                leaves.append(leaf.hex())
            cases.append({"rb": rb, "chunks": (37 + rb + 1023) // 1024, "leaves": leaves})
    return {"domain": d.domain_id.hex(), "cases": cases}


def tree_case(tag: str, rb: int) -> dict:
    d = domain(5, M)
    rows = shake(tag, M * rb)
    root = MerkleTree(d, {r: rows[r * rb:(r + 1) * rb] for r in range(M)}, lambda _: ROW).commitment.root
    return {"tag": tag, "m": M, "rb": rb, "domain": d.domain_id.hex(), "root": root.hex()}


def tile_case(rows: int) -> dict:
    """C~ and U of 32 real rows (the gate tags) and rows − 32 zero filler rows; the digests, the leaves of the tiles over
    those rows, and the tile tree's root."""
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
    return {"rows": rows, "n": N, "tiles": tiles, "domain": d.domain_id.hex(),
            "digests_sha256": hashlib.sha256(b"".join(dg)).hexdigest(), "leaves_sha256": hashlib.sha256(b"".join(leaves)).hexdigest(),
            "root": root.hex()}


def main() -> None:
    out = {"source": "frame-b3 (verity.commitments.merkle, hash='blake3') and pearl_c.H1_KEYS",
           "rows": rows_cases(),
           "trees": {"A m=32 fp32 rows": tree_case("gate tree fp32", 4 * N), "A m=32 bf16 rows": tree_case("gate tree bf16", 2 * N)},
           "tiles": {"padded64": tile_case(64), "m32": tile_case(M)}}
    Path(sys.argv[1]).write_text(json.dumps(out))
    print(f"{len(out['rows']['cases'])} row cases, 2 trees, 2 tile cases -> {sys.argv[1]}")


if __name__ == "__main__":
    main()
