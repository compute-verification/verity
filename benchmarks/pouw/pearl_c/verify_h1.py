"""The reference verifier on h1_bench.py's and prefill_fused.py's transcripts: -h1's, -h2's or -h3's hashing (the header's
``hashing``) recomputed from the run's own A, C~ and U with the verifier's code, each result held to what the kernels wrote.

  PYTHONPATH=packages/verity/src:protocols/pouw python3 benchmarks/pouw/pearl_c/verify_h1.py [--jobs J] RUN_ID TRANSCRIPT...

Standard library only. For each transcript:
- A's commitment: `verity.commitments`' `MerkleTree` over the transcript's rows (frame-b3 under -h1, frame-b3s under -h2),
  under the domain its header names (binding, owner, the row count): every level of it (`tree_levels`) against the tree the
  kernels wrote.  A header with ``a_rows_tag`` (prefill's A) carries no rows: row r is SHAKE128 of the tag with r filled in,
  regenerated here, its leaves computed by J processes; with ``a_rows_y`` (prefill_fused.py's) row r is
  `prefill_fused.a_half` of the tag and r, repeated to the row's length.  A header whose A names a seed (salt, root_B, unit
  index) must have the audit's activation domain for them; then a ``row_seeds`` section (-h3) is every row's
  `pearl_c.row_seed` and a ``unit_seed`` section (-h2 with its seed, and -h3's for the tickets) is `pearl_c.unit_seed_a`
  over A's root;
- when the transcript has the post-GEMM hashing: every (row, 32-column chunk) message digest (`pearl_c.message_digest`), every
  tile leaf (`pearl_c.leaf` over the tile's `Checked` C~ and U words), J processes over blocks of tiles, and every level of the
  tile tree (the format's tree over the leaves, schema `verity/pouw/tile/v1`), against the kernels'.  A transcript without a
  ``digests`` section (tile leaves hashed in the GEMM's epilogue, digests never stored) has its leaves checked the same way.

It prints one line per transcript, `<format> verifier accept: <run>:<path> sha256:<hex> …` or `… reject: …` (format h1, h2
or h3), and exits 1 if any transcript is rejected.

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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from verity.commitments import CommitmentDomain, MerkleTree, RangeIndexedDomain
from verity.commitments.merkle import tree_levels
from verity_pouw.audit import activation_binding
from verity_pouw.protocol import Context, Matmul, Shape
from verity_pouw.schemes import pearl_c

from prefill_fused import a_half

ROW, TILE = "verity/pouw/row/v1", "verity/pouw/tile/v1"
TREE_HASH = {"h1": "blake3", "h2": "blake3-s256", "h3": "blake3-s256"}
#: tiles per job of the tile checks (a tile row is split into blocks of this many 64 x 64 tiles)
TILE_BLOCK = 16


def index(path: Path) -> tuple[dict, dict[str, tuple[int, int]]]:
    """The header, and each section's (offset, length) in the file."""
    with open(path, "rb") as f:
        line = f.readline()
    head, off, where = json.loads(line), len(line), {}
    for name, size in head["sections"]:
        where[name] = (off, size)
        off += size
    if off != os.path.getsize(path):
        raise ValueError(f"{path}: {os.path.getsize(path) - off} bytes past the sections")
    return head, where


def section(path: Path, where: dict, name: str, lo: int = 0, hi: int | None = None) -> bytes:
    off, size = where[name]
    hi = size if hi is None else hi
    with open(path, "rb") as f:
        f.seek(off + lo)
        return f.read(hi - lo)


def generated_row(head: dict, r: int) -> bytes:
    rb = head["rb"]
    if "a_rows_tag" in head:
        return hashlib.shake_128(head["a_rows_tag"].format(r=r).encode()).digest(rb)
    spec = head["a_rows_y"]
    return a_half(spec["tag"], r, spec["half"]) * (rb // spec["half"])


def a_rows_job(job: tuple) -> list[bytes]:
    """Rows r0 … r1 − 1 of a generated A: their leaves under the domain (binding, owner, count, hash), or with a seed context
    their `pearl_c.row_seed`s."""
    head, binding, owner, hash, r0, r1, ctx = job
    if ctx is not None:
        return [pearl_c.row_seed(ctx, r, generated_row(head, r)) for r in range(r0, r1)]
    d = CommitmentDomain(binding, owner, RangeIndexedDomain(head["m"]), hash=hash)
    return [d.leaf(r, r, ROW, generated_row(head, r)) for r in range(r0, r1)]


def tile_job(job: tuple) -> tuple[str | None, list[bytes]]:
    """Tile row ti's tiles tj0 … tj1 − 1: every message digest of their chunks and every tile leaf, recomputed from C~ and U
    and held to the transcript's; the first difference, or None and the leaves."""
    path, where, m, n, hashing, ti, tj0, tj1 = job
    per, tn = n // 32, n // 64
    r0, r1 = 64 * ti, min(m, 64 * ti + 64)
    cw = struct.unpack(f"<{(r1 - r0) * n}I", section(path, where, "c", 4 * r0 * n, 4 * r1 * n))
    uw = struct.unpack(f"<{(r1 - r0) * n}I", section(path, where, "u", 4 * r0 * n, 4 * r1 * n))
    dg = section(path, where, "digests", 32 * r0 * per, 32 * r1 * per) if "digests" in where else None
    lv = section(path, where, "leaves", 32 * (ti * tn + tj0), 32 * (ti * tn + tj1))
    for r in range(r1 - r0 if dg is not None else 0):
        for h in range(2 * tj0, 2 * tj1):
            s = r * n + 32 * h
            msg = struct.pack("<32I", *cw[s:s + 32]) + struct.pack("<32I", *uw[s:s + 32])
            if pearl_c.message_digest(msg, hashing) != dg[32 * (r * per + h):32 * (r * per + h + 1)]:
                return f"message digest ({r0 + r}, {h}) differs", []
    leaves = []
    for tj in range(tj0, tj1):
        rows = range(r1 - r0)
        checked = pearl_c.Checked(c=[list(cw[r * n + 64 * tj:r * n + 64 * tj + 64]) for r in rows],
                                  u=[list(uw[r * n + 64 * tj:r * n + 64 * tj + 64]) for r in rows])
        leaf = pearl_c.leaf(checked, hashing)
        if leaf != lv[32 * (tj - tj0):32 * (tj - tj0 + 1)]:
            return f"tile {ti * tn + tj}'s leaf differs", []
        leaves.append(leaf)
    return None, leaves


def laid_out(domain: CommitmentDomain, leaves: list[bytes]) -> bytes:
    """Every level of the domain's tree over ``leaves`` (pads included), level 0 first, 32 bytes a node, as the kernels lay
    them out one after another."""
    return b"".join(b"".join(level) for level in tree_levels(domain, leaves))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify(run: str, path: Path, jobs: int = 1) -> tuple[bool, str, str]:
    head, where = index(path)
    digest = file_sha256(path)
    m, rb, n = head["m"], head["rb"], head["n"]
    hashing = head.get("hashing", "h1")
    tree_hash, notes = TREE_HASH[hashing], []
    binding, owner = bytes.fromhex(head["a"]["binding"]), head["a"]["owner"]
    da = CommitmentDomain(binding, owner, RangeIndexedDomain(m), hash=tree_hash)
    if da.domain_id.hex() != head["a"]["domain"]:
        return False, hashing, f"{run}:{path} sha256:{digest} A's domain id is not its binding's"
    generated = "a_rows_tag" in head or "a_rows_y" in head
    step = -(-m // max(1, jobs))
    spans = [(r0, min(m, r0 + step)) for r0 in range(0, m, step)]
    pool = ProcessPoolExecutor(max(1, jobs))
    try:
        if generated:
            work = [(head, binding, owner, tree_hash, r0, r1, None) for r0, r1 in spans]
            want = laid_out(da, [leaf for part in pool.map(a_rows_job, work) for leaf in part])
            root = want[-32:]
        else:
            rows = section(path, where, "a_rows")
            root = MerkleTree(da, {r: rows[r * rb:(r + 1) * rb] for r in range(m)}, lambda _: ROW).commitment.root
            want = laid_out(da, [da.leaf(r, r, ROW, rows[r * rb:(r + 1) * rb]) for r in range(m)])
        if section(path, where, "a_tree") != want or want[-32:] != root:
            return False, hashing, f"{run}:{path} sha256:{digest} A's commitment differs from the kernels' (every level checked)"
        notes.append(f"A root {root.hex()} ({m} rows of {rb} B" + (", regenerated from their tag)" if generated else ")"))
        seed = head["a"].get("seed")
        if seed:
            salt, root_b, index_ = bytes.fromhex(seed["salt"]), bytes.fromhex(seed["root_b"]), seed["index"]
            if binding != activation_binding(salt, index_) or owner != index_:
                return False, hashing, f"{run}:{path} sha256:{digest} A's domain is not the activation domain of its seed's unit"
            if "row_seeds" in where:
                ctx = Context(salt, Matmul(index_, Shape(m, seed["k"], seed["n"]), 0), root, root_b)
                got = section(path, where, "row_seeds")
                if generated:
                    seeds = b"".join(s for part in pool.map(a_rows_job, [(head, binding, owner, tree_hash, r0, r1, ctx)
                                                                         for r0, r1 in spans]) for s in part)
                else:
                    rows = section(path, where, "a_rows")
                    seeds = b"".join(pearl_c.row_seed(ctx, r, rows[r * rb:(r + 1) * rb]) for r in range(m))
                for r in range(m):
                    if seeds[32 * r:32 * r + 32] != got[32 * r:32 * r + 32]:
                        return False, hashing, f"{run}:{path} sha256:{digest} row {r}'s seed differs"
                notes.append(f"{m} row seeds (one-row units {index_}·2^32 + r)")
            if "unit_seed" in where:
                if pearl_c.unit_seed_a(salt, root, root_b, index_) != section(path, where, "unit_seed"):
                    return False, hashing, f"{run}:{path} sha256:{digest} the unit's seed_A differs"
                notes.append("the unit's seed_A")
        elif "row_seeds" in where or "unit_seed" in where:
            return False, hashing, f"{run}:{path} sha256:{digest} seeds without their unit's salt, root_B and index"
        if "c" in where:
            per, tn = n // 32, n // 64
            work = [(path, where, m, n, hashing, ti, tj, min(tn, tj + TILE_BLOCK))
                    for ti in range(-(-m // 64)) for tj in range(0, tn, TILE_BLOCK)]
            leaves = []
            for bad, part in pool.map(tile_job, work):
                if bad:
                    return False, hashing, f"{run}:{path} sha256:{digest} {bad}"
                leaves += part
            tiles = len(leaves)
            dt = CommitmentDomain(bytes.fromhex(head["t"]["binding"]), head["t"]["owner"], RangeIndexedDomain(tiles), hash=tree_hash)
            if dt.domain_id.hex() != head["t"]["domain"]:
                return False, hashing, f"{run}:{path} sha256:{digest} the tile tree's domain id is not its binding's"
            troot = MerkleTree(dt, dict(enumerate(leaves)), lambda _: TILE).commitment.root
            want_t = laid_out(dt, [dt.leaf(t, t, TILE, leaves[t]) for t in range(tiles)])
            if section(path, where, "tile_tree") != want_t or want_t[-32:] != troot:
                return False, hashing, f"{run}:{path} sha256:{digest} the tile tree differs from the kernels' (every level checked)"
            notes.append((f"{m * per} message digests, " if "digests" in where else "no digests section (the leaves from C~ and U), ")
                         + f"{tiles} tile leaves, tile root {troot.hex()}")
    finally:
        pool.shutdown(cancel_futures=True)
    return True, hashing, f"{run}:{path} sha256:{digest} " + "; ".join(notes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processes for a generated A's leaves and the tiles")
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
