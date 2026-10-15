"""Expected tags, leaves and roots of `bench.py`'s stores from #1805's reference, on the CPU before the lease.

For each of Llama-3.1-8B's linears: W from a seeded CPU generator (N(0, 0.25^2) rounded to e4m3, the word codec's
weights), its plain fragment layout, and for each block map (`MAPS`: lane spans 32 and 64, chunk blocks k1 .. k8) every
block's ro-blake3/v1 tag by `tags.Blake3Tag().keyed(seed)` over a process pool, the leaves by `tags.leaf_hash` and the
root by `layout.tree_root`.  The shapes are one layer's store in order (qkv, o, gate_up, down), so block j's index starts
at `JBASE` (bit 33 set, so the index's high word is exercised) plus the blocks before it.  Writes OUT/w-NAME.bin (W,
row-major), OUT/tags-NAME-MAP.bin (32 bytes a block, in index order), OUT/leaves-NAME-MAP.bin (64 bytes a leaf) and
OUT/expect.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import layout  # noqa: E402
from verity.core.protocols.bandwidth import tags  # noqa: E402

SEED = bytes(range(32))
JBASE = 1 << 33
CHUNK = 2048

_DATA: bytes = b""
_BB = 0
_JB = 0
_KEYED = None


def weights(torch, n: int, k: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, k, generator=g) * 0.25).to(torch.float8_e4m3fn).view(torch.uint8)


def weight_seed(name: str) -> int:
    return 1000 + list(layout.SHAPES).index(name)


def plain_layout(w8):
    n, k = w8.shape
    return w8.view(n // 16, 2, 8, k // 512, 16, 2, 4, 4).permute(0, 3, 4, 2, 6, 5, 1, 7).contiguous().view(-1)


def block_view(plain, n: int, k: int, span: int):
    ns = layout.spans(k, span)
    return plain.view(n // 16, ns, span, 32, 16).permute(0, 1, 3, 2, 4).contiguous().view(-1, 16 * span)


def chunk_view(plain, n: int, k: int, lanes: int):
    """`layout.chunk_blocks` as rows of a tensor: each lane's D rotated left by 8 bytes, `lanes` lanes a row."""
    import torch

    ns = layout.chunk_spans(k)
    d = plain.view(n // 16, ns, layout.CHUNK_TILES, 32, 16).permute(1, 0, 3, 2, 4).reshape(-1, 1024)
    return torch.cat([d[:, 8:], d[:, :8]], 1).contiguous().view(-1, 1024 * lanes)


MAPS = ("32", "64", "k1", "k2", "k4", "k8")  # lane spans of 512 and 1,024 bytes; chunk blocks of 1, 2, 4, 8 lanes


def map_view(plain, n: int, k: int, name: str):
    return chunk_view(plain, n, k, int(name[1:])) if name[0] == "k" else block_view(plain, n, k, int(name))


def map_blocks(n: int, k: int, name: str) -> int:
    return layout.chunk_block_count(n, k, int(name[1:])) if name[0] == "k" else layout.block_count(n, k, int(name))


def _tag_range(lo_hi: tuple[int, int]) -> bytes:
    lo, hi = lo_hi
    out = bytearray()
    for i in range(lo, hi):
        out += _KEYED(_JB + i, _DATA[i * _BB:(i + 1) * _BB])
    return bytes(out)


def reference_tags(data: bytes, block_bytes: int, jbase: int, seed: bytes, workers: int) -> bytes:
    global _DATA, _BB, _JB, _KEYED
    _DATA, _BB, _JB, _KEYED = data, block_bytes, jbase, tags.Blake3Tag().keyed(seed)
    count = len(data) // block_bytes
    ranges = [(lo, min(lo + CHUNK, count)) for lo in range(0, count, CHUNK)]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        return b"".join(pool.imap(_tag_range, ranges))


def check_layout(torch) -> None:
    """The torch layout and block view equal `layout`'s on a small W."""
    n, k = 32, 2048
    w8 = weights(torch, n, k, 7)
    plain = plain_layout(w8)
    want = layout.plain_fragments(bytes(w8.contiguous().view(-1).numpy()), n, k)
    if bytes(plain.numpy()) != want:
        raise SystemExit("torch's plain layout differs from layout.plain_fragments")
    for span in (32, 64):
        got = bytes(block_view(plain, n, k, span).numpy())
        if got != b"".join(layout.blocks(want, n, k, span)):
            raise SystemExit(f"torch's block view differs from layout.blocks at span {span}")
    for lanes in layout.GROUPS:
        got = bytes(chunk_view(plain, n, k, lanes).numpy())
        if got != b"".join(layout.chunk_blocks(want, n, k, lanes)):
            raise SystemExit(f"torch's chunk view differs from layout.chunk_blocks at {lanes} lanes")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shapes", default=",".join(layout.SHAPES))
    ap.add_argument("--maps", default=",".join(MAPS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    check_layout(torch)
    spans = a.maps.split(",")
    tag = tags.Blake3Tag()
    res = {"seed": SEED.hex(), "key": tag.key(SEED).hex(), "tag": tag.name, "jbase": JBASE, "shapes": {},
           "workers": a.workers}
    jbase = {s: JBASE for s in spans}
    for name in layout.SHAPES:
        n, k = layout.SHAPES[name]
        if name not in a.shapes.split(","):
            for s in spans:
                jbase[s] += map_blocks(n, k, s)
            continue
        t0 = time.time()
        w8 = weights(torch, n, k, weight_seed(name))
        wb = bytes(w8.contiguous().view(-1).numpy())
        (out / f"w-{name}.bin").write_bytes(wb)
        plain = plain_layout(w8)
        entry = {"N": n, "K": k, "weight_seed": weight_seed(name), "w_sha256": hashlib.sha256(wb).hexdigest(),
                 "spans": {}}
        for s in spans:
            view = map_view(plain, n, k, s)
            data, bb = bytes(view.numpy()), view.shape[1]
            ys_b = reference_tags(data, bb, jbase[s], SEED, a.workers)
            count = len(data) // bb
            ys = [ys_b[32 * i:32 * i + 32] for i in range(count)]
            # the store's leaf index: this shape's leaf c is leaf jbase / 16 + c
            head, c0 = tags.leaf_head(tag, SEED), jbase[s] // 16
            leaves = [tags.leaf_hash(head, c0 + c, ys[16 * c:16 * c + 16]) for c in range(count // 16)]
            (out / f"tags-{name}-{s}.bin").write_bytes(ys_b)
            (out / f"leaves-{name}-{s}.bin").write_bytes(b"".join(leaves))
            entry["spans"][s] = {"blocks": count, "block_bytes": bb, "jbase": jbase[s], "leaf0": jbase[s] // 16,
                                 "leaves": len(leaves), "root": layout.tree_root(leaves).hex(),
                                 "tags_sha256": hashlib.sha256(ys_b).hexdigest()}
            jbase[s] += count
        res["shapes"][name] = entry
        print(f"{name}: {time.time() - t0:.1f} s", flush=True)
    (out / "expect.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
