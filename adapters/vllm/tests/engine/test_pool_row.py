"""`engine.pool_row`: a prefix-cache pool between windows as its pool row (`verity.commitments.pool_row`), on stand-ins with vLLM
d9105ea80's ``BlockPool`` layout (its blocks, free queue and hash map); what the row can't hold is refused by name; the
genesis reset is the empty row; and BlockKey_v1 replaces vLLM's block hash. A recorded trace drives vLLM's own pool in
``benchmarks/network_e2e/encodings`` (``pool_row_trace``)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from verity.commitments import pool_row as R
from verity.definitions.commitments.gates import block_key as BK

from verity_vllm.engine import pool_row as PR
from verity_vllm.errors import Refused

#: the stand-ins' pool: NB 6 (blocks 1 … 5), blocks of B 4 tokens
NB, B = 6, 4
G0 = bytes(PR.GROUP_BYTES)
KA, KB = BK.block_key(BK.ROOT, [1, 2, 3, 4], 4), BK.block_key(BK.ROOT, [5, 6, 7, 8], 4)


class Block:
    def __init__(self, i: int) -> None:
        self.block_id, self.ref_cnt, self.is_null = i, 0, False
        self.block_hash: bytes | None = None
        self.block_hash_num_tokens: int | None = None
        self.prev_free_block = self.next_free_block = None

    def reset_hash(self) -> None:
        self.block_hash = self.block_hash_num_tokens = None


def pool(free=(3, 1, 5, 2, 4), cached=None, held=(), nb=6, B=4, by_id=False):
    """A pool of ``nb`` blocks, block 0 null: the free queue in ``free``'s order, ``cached`` = {key: blocks, first copy
    first} under group 0, ``held`` holding one reference each. ``by_id``: the hash map as vLLM 0.34 keeps it, the id of
    the block a hit takes under each key and, for a key cached by several blocks, their ids in ``_dup_ids``."""
    cached = {KA: (1,), KB: (5, 2)} if cached is None else cached
    blocks = [Block(i) for i in range(nb)]
    head, tail = Block(-1), Block(-1)
    chain = [head, *(blocks[b] for b in free), tail]
    for a, b in zip(chain, chain[1:]):
        a.next_free_block, b.prev_free_block = b, a
    cache, several = {}, {}
    for k, ids in cached.items():
        for b in ids:
            blocks[b].block_hash, blocks[b].block_hash_num_tokens = k + G0, B
        if by_id:
            cache[k + G0] = ids[0]
            if len(ids) > 1:
                several[k + G0] = dict.fromkeys(ids)
        else:
            cache[k + G0] = blocks[ids[0]] if len(ids) == 1 else {b: blocks[b] for b in ids}
    for b in held:
        blocks[b].ref_cnt = 1
    p = SimpleNamespace(blocks=blocks, null_block=blocks[0], hash_block_size=B, num_gpu_blocks=nb,
                        free_block_queue=SimpleNamespace(fake_free_list_head=head, fake_free_list_tail=tail, num_free_blocks=len(free)),
                        cached_block_hash_to_block=SimpleNamespace(_cache=cache, **({"_dup_ids": several} if by_id else {})),
                        cached_block_hashes_by_block={})

    def reset_prefix_cache() -> bool:                  # vLLM's: only with every block but the null one free
        if nb - p.free_block_queue.num_free_blocks != 1:
            return False
        p.cached_block_hash_to_block = SimpleNamespace(_cache={})
        p.cached_block_hashes_by_block.clear()
        for b in p.blocks:
            b.reset_hash()
        return True

    p.reset_prefix_cache = reset_prefix_cache
    return p


WANT = R.PoolRow(3, 4, prev=(3, 5, 0, 2, 1), next=(5, 4, 1, 0, 2), cprev=(0, 5, 0, 0, 0), cnext=(0, 0, 0, 0, 2),
                 mlast=(1, 0, 0, 0, 2), key=(KA, KB, R.NOKEY, R.NOKEY, KB))


@pytest.mark.parametrize("by_id", [False, True])
def test_the_row_is_the_free_queue_and_each_key_s_copies_in_vllm_s_order(tmp_path, by_id) -> None:
    """The free queue 3 1 5 2 4 links ``prev``/``next`` with 0 at its ends; KB's copies 5 then 2 (5 the one a hit takes)
    link ``cprev``/``cnext``, and ``mlast`` names each key's last copy at its first, KA's one copy naming itself. The row
    is the same whether the hash map holds blocks or, as vLLM 0.34's does, their ids."""
    p = pool(by_id=by_id)
    snap = PR.snapshot(p, 4)
    assert snap.free == (3, 1, 5, 2, 4) and snap.copies == {KA: (1,), KB: (5, 2)} and snap.ref == (0,) * 6
    assert PR.row(snap) == WANT
    rec = PR.export(p, B, tmp_path / "w0.json")
    assert rec == {"nb": 6, "root": R.root(WANT).hex()}
    on_disk = json.loads((tmp_path / "w0.json").read_text())
    assert on_disk["root"] == rec["root"] and on_disk["next"] == list(WANT.next) and on_disk["key"][1] == KB.hex()


def test_a_fresh_pool_is_the_empty_row_and_the_genesis_reset_relinks_the_queue_in_id_order() -> None:
    """vLLM's own reset keeps the free queue's order, which isn't a fresh pool's; `genesis_reset` also relinks it, so the
    pool's root is the chain's genesis. It refuses while a block is held."""
    assert PR.row(PR.snapshot(pool(free=(1, 2, 3, 4, 5), cached={}))) == R.empty(NB)
    p = pool()
    p.reset_prefix_cache()
    assert PR.row(PR.snapshot(p)) != R.empty(NB)
    p = pool()
    PR.genesis_reset(p)
    assert PR.export(p, B)["root"] == R.genesis(NB).hex()
    held = pool(free=(3, 1, 5, 2), held=(4,))
    with pytest.raises(Refused, match="1 blocks are held: reset at an edge"):
        PR.genesis_reset(held)


def _bad(change):
    p = pool()
    change(p)
    return p


@pytest.mark.parametrize("change, match", [
    (lambda p: setattr(p.null_block, "block_id", 3), "the null block is block 3"),
    (lambda p: setattr(p.blocks[1], "block_hash", KA + (1).to_bytes(4, "big")), "KV cache group 1"),
    (lambda p: setattr(p.blocks[1], "block_hash", KA[:32] + G0), "32 bytes, not BlockKey_v1's 64"),
    (lambda p: setattr(p.blocks[5], "block_hash_num_tokens", 6), "covers 6 tokens, inside a block of 4"),
    (lambda p: p.cached_block_hashes_by_block.setdefault(2, {KA + G0}), "block 2 is cached under a second key"),
    (lambda p: setattr(p.free_block_queue, "num_free_blocks", 4), "links 5 blocks but counts 4"),
    (lambda p: setattr(p.blocks[5], "prev_free_block", p.blocks[3]), "breaks after 2 blocks"),
    (lambda p: p.cached_block_hash_to_block._cache.update({KA + G0: p.blocks[3]}), "lists blocks \\[3\\]"),
    (lambda p: setattr(p.blocks[3], "block_hash", KA + G0), "a block has a key that the hash map doesn't list"),
])
def test_what_the_row_can_t_hold_is_refused_by_name(change, match) -> None:
    with pytest.raises(Refused, match=match):
        PR.snapshot(_bad(change))


def test_a_map_of_ids_whose_hit_is_not_its_key_s_first_copy_is_refused() -> None:
    """vLLM 0.34 keeps the block a hit takes apart from a key's list of copies: the row's order is the list's, so the two
    must begin alike."""
    p = pool(by_id=True)
    p.cached_block_hash_to_block._cache[KB + G0] = 2
    with pytest.raises(Refused, match="block 2 is the one a hit takes, and its key's copies begin with block 5"):
        PR.snapshot(p)
    p = pool(by_id=True)
    p.cached_block_hash_to_block._cache[KA + G0] = 3
    with pytest.raises(Refused, match="lists blocks \\[3\\]"):
        PR.snapshot(p)


def test_a_held_block_or_a_pool_of_another_block_size_is_refused() -> None:
    """A window ends with no block held: a running request's blocks are refused (the edge's cut frees them first). The
    pool must key blocks of the engine's size."""
    with pytest.raises(Refused, match=r"1 blocks are held \(block 4 by 1\)"):
        PR.row(PR.snapshot(pool(free=(3, 1, 5, 2), held=(4,))))
    with pytest.raises(Refused, match="keys blocks of 4 tokens, not the engine's 16"):
        PR.export(pool(), 16)


def test_blockkey_v1_replaces_vllm_s_block_hash() -> None:
    """`install_keys` patches ``hash_block_tokens`` (which the request block hasher looks up at each call) with BlockKey_v1
    at the token width (4 bits in blocks of 4, a whole two bytes; 3 bits widen to 4): ROOT under a request's first block,
    the parent's key under the rest; extra keys (LoRA, multimodal, a cache salt) are refused; uninstalling puts vLLM's
    back."""
    def vllm_hash(fn, parent, tokens, extra_keys=None):
        return b"vllm"

    mod = SimpleNamespace(hash_block_tokens=vllm_hash)
    hooks = PR.install_keys(3, B, module=mod)
    first = mod.hash_block_tokens(None, None, [1, 2, 3, 4])
    assert first == KA and mod.hash_block_tokens(None, first, [5, 6, 7, 8], None) == BK.block_key(KA, [5, 6, 7, 8], 4)
    with pytest.raises(Refused, match="BlockKey_v1 keys its tokens alone"):
        mod.hash_block_tokens(None, None, [1, 2, 3, 4], ("salt",))
    hooks.uninstall()
    assert mod.hash_block_tokens is vllm_hash
