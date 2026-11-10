"""The GPU tagger's expected roots, from the core reference (`verity.core.protocols.bandwidth`), on CPU processes.

    python reference.py --blocks 524288 --seed HEX --out expect.json [--workers 32]

The store is the generated test store of `bandwidth_vectors.py` (`block`), which `tagger.cu`'s `fill_store` writes on the
device.  `expect.json` holds every tag's roots over `--blocks` labels under `--seed`, and the same over 37 labels under
`SEED`, recomputed here and checked against `vectors.json`'s roots before anything is written.  `store_leaves` gives the
store's PoUS leaf digests under `geometry(blocks)` and a salt, from which `gpu.py windows` builds `vk` and the store's
audit paths once it knows its device: the salt is `audit.draw_salt` for that device under `COMMITMENT`, the test
store's stand-in for the commitment to `W`.
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
COMMITMENT = hashlib.sha512(BLOCK_DOMAIN).digest()
SEED = bytes(range(32))
SMALL = 37
CHUNK = 4096


def block(j: int, block_bytes: int = BLOCK_BYTES) -> bytes:
    """Label `j` of the generated test store."""
    out = b"".join(hashlib.sha512(BLOCK_DOMAIN + j.to_bytes(8, "big") + c.to_bytes(4, "big")).digest()
                   for c in range(-(-block_bytes // 64)))
    return out[:block_bytes]


def geometry(blocks: int) -> P2Params:
    """P2's geometry at its registered point over `blocks` labels: one segment up to 2^16, else whole segments of 2^16."""
    w, c0, per = POINT
    if blocks <= per:
        return P2Params(w, c0, blocks, 1)
    if blocks % per:
        raise ValueError(f"more than {per} blocks come in whole segments of {per}")
    return P2Params(w, c0, per, blocks // per)


def _chunk(args: tuple[int, int, bytes]) -> dict[str, list[bytes]]:
    lo, hi, seed = args
    keyed = {name: tags.get(name).keyed(seed) for name in tags.TAGS}
    ys: dict[str, list[bytes]] = {name: [] for name in tags.TAGS}
    for j in range(lo, hi):
        b = block(j)
        for name in tags.TAGS:
            ys[name].append(keyed[name](j, b))
    return ys


def roots(blocks: int, seed: bytes, workers: int) -> dict[str, str]:
    """Every tag's root over the first `blocks` generated labels under `seed`."""
    ys: dict[str, list[bytes]] = {name: [] for name in tags.TAGS}
    jobs = [(lo, min(lo + CHUNK, blocks), seed) for lo in range(0, blocks, CHUNK)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(_chunk, jobs):
            for name, v in part.items():
                ys[name] += v
    return {name: MerkleTree(tags.leaves(tags.get(name), seed, v)).root.hex() for name, v in ys.items()}


def _leaves_chunk(args: tuple[int, int, int, bytes]) -> list[bytes]:
    lo, hi, blocks, salt = args
    params = geometry(blocks)
    return [leaf_hash(params, salt, j, block(j)) for j in range(lo, hi)]


def store_leaves(blocks: int, salt: bytes, workers: int) -> list[bytes]:
    """The generated store's PoUS leaf digests under `geometry(blocks)` and `salt`, in label order."""
    jobs = [(lo, min(lo + CHUNK, blocks), blocks, salt) for lo in range(0, blocks, CHUNK)]
    if workers <= 1:
        return [leaf for job in jobs for leaf in _leaves_chunk(job)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [leaf for part in pool.map(_leaves_chunk, jobs) for leaf in part]


def small() -> dict:
    """The 37-label case under `SEED`, recomputed and held to `vectors.json`'s roots."""
    vectors = json.loads(VECTORS.read_text())
    pinned = {r["tag"]: r["tag_root"] for r in vectors["roots"] if r["blocks"] == SMALL and r["seed"] == SEED.hex()}
    store = audit.Store(geometry(SMALL), bytes.fromhex(vectors["salts"][0]["salt"]), [block(j) for j in range(SMALL)])
    got = {name: audit.tag_pass(store, name, SEED).root.hex() for name in tags.TAGS}
    if got != pinned:
        raise SystemExit(f"the reference disagrees with vectors.json over {SMALL} blocks: {got} != {pinned}")
    return {"blocks": SMALL, "seed": SEED.hex(), "roots": got}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--blocks", type=int, default=1 << 19)
    ap.add_argument("--seed", default=SEED.hex())
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    seed = bytes.fromhex(a.seed)
    if len(seed) != tags.SEED_BYTES:
        raise SystemExit(f"a seed is {tags.SEED_BYTES} bytes")
    out = {"format": "verity-bandwidth/expect/v1", "block_bytes": BLOCK_BYTES, "small": small(),
           "full": {"blocks": a.blocks, "seed": seed.hex(), "roots": roots(a.blocks, seed, a.workers)}}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
