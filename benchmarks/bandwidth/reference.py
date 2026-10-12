"""The GPU tagger's expected roots, from the core reference (`verity.core.protocols.bandwidth`), on CPU processes.

    python reference.py --blocks 524288 --seed HEX --out expect.json [--leaves store_leaves.bin] [--workers 32]

The store is the generated test store of `bandwidth_vectors.py` (`block`), which `tagger.cu`'s `fill_store` writes on the
device.  `expect.json` holds every tag's roots over `--blocks` blocks under `--seed`, and the same over 37 blocks under
`SEED`, recomputed here and checked against `vectors.json`'s roots before anything is written.  `--leaves`
also writes the store's PoUS leaf digests (64 bytes each, block order) under `geometry(blocks)` and `SALT`, from which
`gpu.py windows` builds `vk` and the store's audit paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from verity.core.protocols.bandwidth import audit, tags
from verity.core.protocols.pous.schemes.p2 import POINT, P2Params
from verity.core.protocols.pous.verifier import MerkleTree, leaf_hash

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "verity/core/protocols/bandwidth/tests/vectors.json"
BLOCK_DOMAIN = b"verity/bandwidth/test-block/v1\0"
BLOCK_BYTES = 2056
SALT = bytes(range(100, 124))
SEED = bytes(range(32))
SMALL = 37
CHUNK = 4096


def block(j: int, block_bytes: int = BLOCK_BYTES) -> bytes:
    """Block `j` of the generated test store."""
    out = b"".join(hashlib.sha512(BLOCK_DOMAIN + j.to_bytes(8, "big") + c.to_bytes(4, "big")).digest()
                   for c in range(-(-block_bytes // 64)))
    return out[:block_bytes]


def geometry(blocks: int) -> P2Params:
    """P2's geometry at its registered point over `blocks` blocks: one segment up to 2^16, else whole segments of 2^16."""
    w, c0, per = POINT
    if blocks <= per:
        return P2Params(w, c0, blocks, 1)
    if blocks % per:
        raise ValueError(f"more than {per} blocks come in whole segments of {per}")
    return P2Params(w, c0, per, blocks // per)


def _chunk(args: tuple[int, int, int, bytes, bool]) -> tuple[dict[str, list[bytes]], list[bytes]]:
    lo, hi, blocks, seed, want_leaves = args
    keyed = {name: tags.get(name).keyed(seed) for name in tags.TAGS}
    params = geometry(blocks) if want_leaves else None
    ys: dict[str, list[bytes]] = {name: [] for name in tags.TAGS}
    store_leaves = []
    for j in range(lo, hi):
        b = block(j)
        for name in tags.TAGS:
            ys[name].append(keyed[name](j, b))
        if want_leaves:
            store_leaves.append(leaf_hash(params, SALT, j, b))
    return ys, store_leaves


def roots(blocks: int, seed: bytes, workers: int, leaves_out: Path | None = None) -> dict[str, str]:
    """Every tag's root over the first `blocks` generated blocks under `seed`; `leaves_out` gets the store's leaves."""
    ys: dict[str, list[bytes]] = {name: [] for name in tags.TAGS}
    store_leaves: list[bytes] = []
    jobs = [(lo, min(lo + CHUNK, blocks), blocks, seed, leaves_out is not None) for lo in range(0, blocks, CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part, sl in pool.map(_chunk, jobs):
            for name, v in part.items():
                ys[name] += v
            store_leaves += sl
    if leaves_out is not None:
        leaves_out.write_bytes(b"".join(store_leaves))
    return {name: MerkleTree(tags.leaves(tags.get(name), seed, v)).root.hex() for name, v in ys.items()}


def small() -> dict:
    """The 37-block case under `SEED`, recomputed and held to `vectors.json`'s roots."""
    vectors = json.loads(VECTORS.read_text())
    pinned = {r["tag"]: r["tag_root"] for r in vectors["roots"] if r["blocks"] == SMALL and r["seed"] == SEED.hex()}
    blocks = [block(j) for j in range(SMALL)]
    got = {name: audit.tag_pass(audit.Store(geometry(SMALL), SALT, blocks), name, SEED).root.hex() for name in tags.TAGS}
    if got != pinned:
        raise SystemExit(f"the reference disagrees with vectors.json over {SMALL} blocks: {got} != {pinned}")
    return {"blocks": SMALL, "seed": SEED.hex(), "roots": got}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=1 << 19)
    ap.add_argument("--seed", default=SEED.hex())
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--leaves", type=Path)
    a = ap.parse_args(argv)
    seed = bytes.fromhex(a.seed)
    if len(seed) != tags.SEED_BYTES:
        raise SystemExit(f"a seed is {tags.SEED_BYTES} bytes")
    out = {"format": "verity-bandwidth/expect/v1", "block_bytes": BLOCK_BYTES, "small": small(),
           "full": {"blocks": a.blocks, "seed": seed.hex(), "roots": roots(a.blocks, seed, a.workers, a.leaves)}}
    if a.leaves is not None:
        out["full"]["store_leaves_sha256"] = hashlib.sha256(a.leaves.read_bytes()).hexdigest()
        out["full"]["salt"] = SALT.hex()
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
