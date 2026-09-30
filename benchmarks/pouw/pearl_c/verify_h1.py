"""The reference verifier on h1_bench.py's decode transcripts: -h1's hashing recomputed from the run's own A, C~ and U with the
verifier's code, each result held to what the kernels wrote.

  PYTHONPATH=packages/verity/src:protocols/pouw python3 benchmarks/pouw/pearl_c/verify_h1.py RUN_ID TRANSCRIPT...

Standard library only. For each transcript:
- A's commitment: `verity.commitments`' frame-b3 `MerkleTree` over the transcript's rows, under the domain its header names
  (binding, owner, the row count): every level of it (`tree_levels`) against the tree the kernels wrote;
- when the transcript has the post-GEMM hashing: every (row, 32-column chunk) message digest (`pearl_c.message_digest`), every
  tile leaf (`pearl_c.leaf` over the tile's `Checked` C~ and U words) and every level of the tile tree (frame-b3 over the
  leaves, schema `verity/pouw/tile/v1`), against the kernels'.

It prints one line per transcript, `h1 verifier accept: <run>:<path> sha256:<hex> …` or `h1 verifier reject: …`, and exits 1
if any transcript is rejected.

A transcript is one JSON header line (the run, the arm, m, the row bytes, n, the two domains, and its sections in order with
their byte lengths), then the sections' bytes.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.commitments.merkle import tree_levels
from verity_pouw.schemes import pearl_c

ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"


def read(path: Path) -> tuple[dict, dict[str, bytes]]:
    raw = path.read_bytes()
    nl = raw.index(b"\n")
    head = json.loads(raw[:nl])
    body, off, parts = raw[nl + 1:], 0, {}
    for name, size in head["sections"]:
        parts[name] = body[off:off + size]
        off += size
    if off != len(body):
        raise ValueError(f"{path}: {len(body) - off} bytes past the sections")
    return head, parts


def laid_out(domain: CommitmentDomain, leaves: list[bytes]) -> bytes:
    """Every level of the domain's tree over ``leaves`` (pads included), level 0 first, 32 bytes a node, as the kernels lay
    them out one after another."""
    return b"".join(b"".join(level) for level in tree_levels(domain, leaves))


def verify(run: str, path: Path) -> tuple[bool, str]:
    head, p = read(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    m, rb, n = head["m"], head["rb"], head["n"]
    notes = []
    da = CommitmentDomain(bytes.fromhex(head["a"]["binding"]), head["a"]["owner"], RangeIndexedDomain(m), hash="blake3")
    if da.domain_id.hex() != head["a"]["domain"]:
        return False, f"{run}:{path} sha256:{digest} A's domain id is not its binding's"
    rows = p["a_rows"]
    tree = MerkleTree(da, {r: rows[r * rb:(r + 1) * rb] for r in range(m)}, lambda _: ROW)
    want = laid_out(da, [da.leaf(r, r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)])
    if p["a_tree"] != want or want[-32:] != tree.commitment.root:
        return False, f"{run}:{path} sha256:{digest} A's commitment differs from the kernels' (every level checked)"
    notes.append(f"A root {tree.commitment.root.hex()} ({m} rows of {rb} B)")
    if "c" in p:
        per, tn = n // 32, n // 64
        cw = struct.unpack(f"<{m * n}I", p["c"])
        uw = struct.unpack(f"<{m * n}I", p["u"])
        dg = p["digests"]
        for r in range(m):
            for h in range(per):
                s = r * n + 32 * h
                msg = struct.pack("<32I", *cw[s:s + 32]) + struct.pack("<32I", *uw[s:s + 32])
                if pearl_c.message_digest(msg, "h1") != dg[32 * (r * per + h):32 * (r * per + h + 1)]:
                    return False, f"{run}:{path} sha256:{digest} message digest ({r}, {h}) differs"
        tiles = -(-m // 64) * tn
        leaves = {}
        for t in range(tiles):
            ti, tj = divmod(t, tn)
            rs = range(64 * ti, min(m, 64 * ti + 64))
            checked = pearl_c.Checked(c=[list(cw[r * n + 64 * tj:r * n + 64 * tj + 64]) for r in rs],
                                      u=[list(uw[r * n + 64 * tj:r * n + 64 * tj + 64]) for r in rs])
            leaves[t] = pearl_c.leaf(checked, "h1")
            if leaves[t] != p["leaves"][32 * t:32 * t + 32]:
                return False, f"{run}:{path} sha256:{digest} tile {t}'s leaf differs"
        dt = CommitmentDomain(bytes.fromhex(head["t"]["binding"]), head["t"]["owner"], RangeIndexedDomain(tiles), hash="blake3")
        if dt.domain_id.hex() != head["t"]["domain"]:
            return False, f"{run}:{path} sha256:{digest} the tile tree's domain id is not its binding's"
        troot = MerkleTree(dt, leaves, lambda _: TILE).commitment.root
        want_t = laid_out(dt, [dt.leaf(t, t, TILE, leaves[t]) for t in range(tiles)])
        if p["tile_tree"] != want_t or want_t[-32:] != troot:
            return False, f"{run}:{path} sha256:{digest} the tile tree differs from the kernels' (every level checked)"
        notes.append(f"{m * per} message digests, {tiles} tile leaves, tile root {troot.hex()}")
    return True, f"{run}:{path} sha256:{digest} " + "; ".join(notes)


def main() -> None:
    run, paths = sys.argv[1], [Path(x) for x in sys.argv[2:]]
    bad = 0
    for path in paths:
        ok, line = verify(run, path)
        print(f"h1 verifier {'accept' if ok else 'reject'}: {line}", flush=True)
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
