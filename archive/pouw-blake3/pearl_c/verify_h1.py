"""The reference verifier on h1_bench.py's transcripts: -h1's or -h2's hashing (the header's ``hashing``) recomputed from
the run's own A, C~ and U with the verifier's code, each result held to what the kernels wrote.

  PYTHONPATH=.:catalog:kernels python3 benchmarks/pouw/pearl_c/verify_h1.py [--jobs J] RUN_ID TRANSCRIPT...

Standard library only. For each transcript:
- A's commitment: `verity.ml.commitments`' `MerkleTree` over the transcript's rows (frame-b3 under -h1, frame-b3s under -h2),
  under the domain its header names (binding, owner, the row count): every level of it (`tree_levels`) against the tree the
  kernels wrote.  A header with ``a_rows_tag`` (prefill's A) carries no rows: row r is SHAKE128 of the tag with r filled in,
  regenerated here, its leaves computed by J processes;
- when the transcript has the post-GEMM hashing: every (row, 32-column chunk) message digest (`pearl_c.message_digest`), every
  tile leaf (`pearl_c.leaf` over the tile's `Checked` C~ and U words) and every level of the tile tree (the format's tree
  over the leaves, schema `verity/pouw/tile/v1`), against the kernels'.

It prints one line per transcript, `<format> verifier accept: <run>:<path> sha256:<hex> …` or `… reject: …` (format h1 or
h2), and exits 1 if any transcript is rejected.

A transcript is one JSON header line (the run, the arm, m, the row bytes, n, the two domains, and its sections in order with
their byte lengths), then the sections' bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from verity.ml.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.ml.commitments.merkle import tree_levels
from verity.core.protocols.pouw.schemes import pearl_c

ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"


def tagged_leaves(job: tuple) -> list[bytes]:
    """Leaves of rows r0 … r1 − 1, each row SHAKE128(tag with r) of rb bytes, under the domain (binding, owner, count, hash)."""
    binding, owner, count, hash, tag, rb, r0, r1 = job
    d = CommitmentDomain(binding, owner, RangeIndexedDomain(count), hash=hash)
    return [d.leaf(r, r, ROW, hashlib.shake_128(tag.format(r=r).encode()).digest(rb)) for r in range(r0, r1)]


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


def verify(run: str, path: Path, jobs: int = 1) -> tuple[bool, str, str]:
    head, p = read(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    m, rb, n = head["m"], head["rb"], head["n"]
    hashing = head.get("hashing", "h1")
    tree_hash, notes = pearl_c.TREE_HASH[hashing], []
    binding, owner = bytes.fromhex(head["a"]["binding"]), head["a"]["owner"]
    da = CommitmentDomain(binding, owner, RangeIndexedDomain(m), hash=tree_hash)
    if da.domain_id.hex() != head["a"]["domain"]:
        return False, hashing, f"{run}:{path} sha256:{digest} A's domain id is not its binding's"
    if "a_rows_tag" in head:
        step = -(-m // max(1, jobs))
        work = [(binding, owner, m, tree_hash, head["a_rows_tag"], rb, r0, min(m, r0 + step)) for r0 in range(0, m, step)]
        # fork: the workers unpickle their function from this module, which callers load from its path, not by an importable name
        with ProcessPoolExecutor(max(1, jobs), mp_context=multiprocessing.get_context("fork")) as pool:
            leaves = [leaf for part in pool.map(tagged_leaves, work) for leaf in part]
        want = laid_out(da, leaves)
        root = want[-32:]
    else:
        rows = p["a_rows"]
        root = MerkleTree(da, {r: rows[r * rb:(r + 1) * rb] for r in range(m)}, lambda _: ROW).commitment.root
        want = laid_out(da, [da.leaf(r, r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)])
    if p["a_tree"] != want or want[-32:] != root:
        return False, hashing, f"{run}:{path} sha256:{digest} A's commitment differs from the kernels' (every level checked)"
    notes.append(f"A root {root.hex()} ({m} rows of {rb} B" + (", regenerated from their tag)" if "a_rows_tag" in head else ")"))
    if "c" in p:
        per, tn = n // 32, n // 64
        cw = struct.unpack(f"<{m * n}I", p["c"])
        uw = struct.unpack(f"<{m * n}I", p["u"])
        dg = p["digests"]
        for r in range(m):
            for h in range(per):
                s = r * n + 32 * h
                msg = struct.pack("<32I", *cw[s:s + 32]) + struct.pack("<32I", *uw[s:s + 32])
                if pearl_c.message_digest(msg, hashing) != dg[32 * (r * per + h):32 * (r * per + h + 1)]:
                    return False, hashing, f"{run}:{path} sha256:{digest} message digest ({r}, {h}) differs"
        tiles = -(-m // 64) * tn
        leaves = {}
        for t in range(tiles):
            ti, tj = divmod(t, tn)
            rs = range(64 * ti, min(m, 64 * ti + 64))
            checked = pearl_c.Checked(c=[list(cw[r * n + 64 * tj:r * n + 64 * tj + 64]) for r in rs],
                                      u=[list(uw[r * n + 64 * tj:r * n + 64 * tj + 64]) for r in rs])
            leaves[t] = pearl_c.leaf(checked, hashing)
            if leaves[t] != p["leaves"][32 * t:32 * t + 32]:
                return False, hashing, f"{run}:{path} sha256:{digest} tile {t}'s leaf differs"
        dt = CommitmentDomain(bytes.fromhex(head["t"]["binding"]), head["t"]["owner"], RangeIndexedDomain(tiles), hash=tree_hash)
        if dt.domain_id.hex() != head["t"]["domain"]:
            return False, hashing, f"{run}:{path} sha256:{digest} the tile tree's domain id is not its binding's"
        troot = MerkleTree(dt, leaves, lambda _: TILE).commitment.root
        want_t = laid_out(dt, [dt.leaf(t, t, TILE, leaves[t]) for t in range(tiles)])
        if p["tile_tree"] != want_t or want_t[-32:] != troot:
            return False, hashing, f"{run}:{path} sha256:{digest} the tile tree differs from the kernels' (every level checked)"
        notes.append(f"{m * per} message digests, {tiles} tile leaves, tile root {troot.hex()}")
    return True, hashing, f"{run}:{path} sha256:{digest} " + "; ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processes for a tagged A's leaves")
    ap.add_argument("run")
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    bad = 0
    for path in args.paths:
        ok, hashing, line = verify(args.run, path, args.jobs)
        print(f"{hashing} verifier {'accept' if ok else 'reject'}: {line}", flush=True)
        bad += not ok
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
