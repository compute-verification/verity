"""The fused tag's block map and the kernel's order of work, against #1805's reference, on the CPU."""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import layout  # noqa: E402
from verity.core.protocols.bandwidth import tags  # noqa: E402
from verity.core.protocols.pous.verifier import MerkleTree  # noqa: E402
from verity.ml.crypto import blake3  # noqa: E402

SEED = bytes(range(32))


@pytest.mark.parametrize("span", layout.SPANS)
def test_lane_tag_is_the_reference(span):
    rng = random.Random(span)
    tag = tags.Blake3Tag()
    keyed, key = tag.keyed(SEED), tag.key(SEED)
    for j in (0, 1, 31, (1 << 33) + 17, (1 << 64) - 1, rng.getrandbits(64)):
        block = rng.randbytes(16 * span)
        assert layout.lane_tag(key, j, block) == keyed(j, block) == blake3.keyed(key, j.to_bytes(8, "big") + block)


def test_lane_tag_refuses_other_sizes():
    with pytest.raises(ValueError, match="a block is 16 bytes times one of"):
        layout.lane_tag(bytes(32), 0, bytes(16 * 48))


def test_plain_fragments_are_the_mma_a_fragments():
    rng = random.Random(3)
    n, k = 32, 128
    w = rng.randbytes(n * k)
    store = layout.plain_fragments(w, n, k)
    tiles = k // 32
    for nb in range(n // 16):
        for t in range(tiles):
            for lane in range(32):
                g, tq = divmod(lane, 4)
                at = ((nb * tiles + t) * 32 + lane) * 16
                a = [store[at + 4 * i:at + 4 * i + 4] for i in range(4)]
                r0, r1, c0 = nb * 16 + g, nb * 16 + g + 8, t * 32 + 4 * tq
                assert a == [w[r0 * k + c0:r0 * k + c0 + 4], w[r1 * k + c0:r1 * k + c0 + 4],
                             w[r0 * k + c0 + 16:r0 * k + c0 + 20], w[r1 * k + c0 + 16:r1 * k + c0 + 20]]


@pytest.mark.parametrize("span", (32, 64))
def test_blocks_permute_the_store(span):
    """Every 16-byte piece of the store lands in exactly one block, at the place the kernel's index says."""
    n, k = 32, 64 * 32
    pieces = n * k // 16
    store = b"".join(i.to_bytes(16, "little") for i in range(pieces))
    bl = layout.blocks(store, n, k, span)
    assert len(bl) == layout.block_count(n, k, span)
    seen = sorted(int.from_bytes(b[16 * i:16 * i + 16], "little") for b in bl for i in range(span))
    assert seen == list(range(pieces))
    ns, tiles = layout.spans(k, span), k // 32
    for nb in range(n // 16):
        for s in range(ns):
            for lane in (0, 5, 31):
                j = (nb * ns + s) * 32 + lane
                first = int.from_bytes(bl[j][:16], "little")
                assert first == (nb * tiles + s * span) * 32 + lane


@pytest.mark.parametrize("lanes", layout.GROUPS)
def test_chunk_tag_is_the_reference(lanes):
    rng = random.Random(100 + lanes)
    tag = tags.Blake3Tag()
    keyed, key = tag.keyed(SEED), tag.key(SEED)
    for j in (0, 7, (1 << 33) + 17, (1 << 64) - 1, rng.getrandbits(64)):
        ds = [rng.randbytes(1024) for _ in range(lanes)]
        block = b"".join(d[8:] + d[:8] for d in ds)
        assert layout.chunk_tag(key, j, ds) == keyed(j, block) == blake3.keyed(key, j.to_bytes(8, "big") + block)


def test_chunk_tag_refuses_other_shapes():
    with pytest.raises(ValueError, match="lanes' 1,024 bytes"):
        layout.chunk_tag(bytes(32), 0, [bytes(1024)] * 3)
    with pytest.raises(ValueError, match="lanes' 1,024 bytes"):
        layout.chunk_tag(bytes(32), 0, [bytes(512)])


@pytest.mark.parametrize("lanes", layout.GROUPS)
def test_chunk_blocks_permute_the_store(lanes):
    """Every 16-byte piece lands in one chunk block; block j's first lane is the kernel's, its D rotated by 8 bytes."""
    n, k = 32, 2 * 64 * 32
    pieces = n * k // 16
    store = b"".join(i.to_bytes(16, "little") for i in range(pieces))
    bl = layout.chunk_blocks(store, n, k, lanes)
    assert len(bl) == layout.chunk_block_count(n, k, lanes)
    ds = [seg[-8:] + seg[:-8] for blk in bl for seg in (blk[i:i + 1024] for i in range(0, len(blk), 1024))]
    seen = sorted(int.from_bytes(d[16 * i:16 * i + 16], "little") for d in ds for i in range(64))
    assert seen == list(range(pieces))
    rts, tiles = n // 16, k // 32
    for s in range(layout.chunk_spans(k)):
        for nb in range(rts):
            u = s * rts + nb
            for lane in range(0, 32, lanes):
                j = u * (32 // lanes) + lane // lanes
                d0 = (nb * tiles + s * 64) * 32 + lane
                # E_j opens with D_0[8:]: the high half of the lane's first piece, then its second piece
                assert bl[j][:8] == d0.to_bytes(16, "little")[8:]
                assert bl[j][8:24] == (d0 + 32).to_bytes(16, "little")
                assert bl[j][1016:1024] == d0.to_bytes(16, "little")[:8]


def test_tree_root_is_rfc6962():
    rng = random.Random(9)
    for count in (1, 2, 3, 5, 16, 17, 33):
        leaves = [rng.randbytes(64) for _ in range(count)]
        assert layout.tree_root(leaves) == MerkleTree(leaves).root


def test_kernel_constants_match_the_reference():
    src = (HERE / "tf.cu").read_text()
    iv = re.search(r"IV0 = (0x[0-9A-Fa-f]+)u, IV1 = (0x[0-9A-Fa-f]+)u, IV2 = (0x[0-9A-Fa-f]+)u, IV3 = (0x[0-9A-Fa-f]+)u",
                   src)
    assert [int(v, 16) for v in iv.groups()] == list(blake3.IV[:4])
    fl = re.search(r"START = (\d+), END = (\d+), PARENT = (\d+), ROOT = (\d+), KEYED = (\d+);", src)
    assert [int(v) for v in fl.groups()] == [blake3.CHUNK_START, blake3.CHUNK_END, blake3.PARENT, blake3.ROOT,
                                             blake3.KEYED_HASH]
    perm = re.search(r"const uint32_t t\[16\] = \{([^}]*)\}", src).group(1)
    assert [int(v) for v in re.findall(r"m\[(\d+)\]", perm)] == list(blake3.MSG_PERMUTATION)
    assert '"verity/bandwidth/tag/v1"' in src and tags.TAG_DOMAIN == b"verity/bandwidth/tag/v1\0"
    assert '"verity/bandwidth/tag-leaf/v1"' in src and tags.LEAF_DOMAIN == b"verity/bandwidth/tag-leaf/v1\0"


def test_store_blocks_count_one_step():
    per_layer = (6144 + 4096 + 28672) * 4096 + 4096 * 14336
    assert layout.store_blocks(32) == 32 * per_layer // 512 == 2 * layout.store_blocks(64)
    assert [layout.store_chunk_blocks(g) for g in layout.GROUPS] == [32 * per_layer // (1024 * g) for g in layout.GROUPS]
