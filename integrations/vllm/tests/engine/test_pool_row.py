"""`engine.pool_row`: a prefix-cache pool between windows as `compile_slice_v6`'s pool row, on stand-ins with vLLM
d9105ea80's ``BlockPool`` layout (its blocks, free queue and hash map); what the row can't hold is refused by name; the
genesis reset is the empty row; and BlockKey_v1 replaces vLLM's block hash. A recorded trace drives vLLM's own pool in
``benchmarks/network_e2e/encodings`` (``pool_row_trace``)."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from verity.ml.commitments.gates import block_key as BK
from verity.ml.protocols.certifier import compile_slice_v6 as V6
from verity_vllm.engine import pool_row as PR

#: `compile_slice_v6.demo`'s class: NB 6 (blocks 1 … 5), B 4, CTW 4
S = V6.demo()[0]
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


def pool(free=(3, 1, 5, 2, 4), cached=None, held=(), nb=6, B=4):
    """A pool of ``nb`` blocks, block 0 null: the free queue in ``free``'s order, ``cached`` = {key: blocks, first copy
    first} under group 0, ``held`` holding one reference each."""
    cached = {KA: (1,), KB: (5, 2)} if cached is None else cached
    blocks = [Block(i) for i in range(nb)]
    head, tail = Block(-1), Block(-1)
    chain = [head, *(blocks[b] for b in free), tail]
    for a, b in zip(chain, chain[1:]):
        a.next_free_block, b.prev_free_block = b, a
    cache = {}
    for k, ids in cached.items():
        for b in ids:
            blocks[b].block_hash, blocks[b].block_hash_num_tokens = k + G0, B
        cache[k + G0] = blocks[ids[0]] if len(ids) == 1 else {b: blocks[b] for b in ids}
    for b in held:
        blocks[b].ref_cnt = 1
    p = SimpleNamespace(blocks=blocks, null_block=blocks[0], hash_block_size=B, num_gpu_blocks=nb,
                        free_block_queue=SimpleNamespace(fake_free_list_head=head, fake_free_list_tail=tail, num_free_blocks=len(free)),
                        cached_block_hash_to_block=SimpleNamespace(_cache=cache), cached_block_hashes_by_block={})

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


WANT = V6.PoolRow(3, 4, prev=(3, 5, 0, 2, 1), next=(5, 4, 1, 0, 2), cprev=(0, 5, 0, 0, 0), cnext=(0, 0, 0, 0, 2),
                  mlast=(1, 0, 0, 0, 2), key=(KA, KB, V6.NOKEY, V6.NOKEY, KB))


def test_the_row_is_the_free_queue_and_each_key_s_copies_in_vllm_s_order(tmp_path) -> None:
    """The free queue 3 1 5 2 4 links ``prev``/``next`` with 0 at its ends; KB's copies 5 then 2 (5 the one a hit takes)
    link ``cprev``/``cnext``, and ``mlast`` names each key's last copy at its first, KA's one copy naming itself."""
    p = pool()
    snap = PR.snapshot(p, 4)
    assert snap.free == (3, 1, 5, 2, 4) and snap.copies == {KA: (1,), KB: (5, 2)} and snap.ref == (0,) * 6
    assert PR.row(snap) == WANT
    rec = PR.export(p, S, tmp_path / "w0.json")
    assert rec == {"nb": 6, "root": V6.root(SimpleNamespace(**S), WANT).hex()}
    on_disk = json.loads((tmp_path / "w0.json").read_text())
    assert on_disk["root"] == rec["root"] and on_disk["next"] == list(WANT.next) and on_disk["key"][1] == KB.hex()


def test_a_fresh_pool_is_the_empty_row_and_the_genesis_reset_relinks_the_queue_in_id_order() -> None:
    """vLLM's own reset keeps the free queue's order, which isn't a fresh pool's; `genesis_reset` also relinks it, so the
    pool's root is the chain's genesis. It refuses while a block is held."""
    st = SimpleNamespace(**S)
    assert PR.row(PR.snapshot(pool(free=(1, 2, 3, 4, 5), cached={}))) == V6.empty_row(st)
    p = pool()
    p.reset_prefix_cache()
    assert PR.row(PR.snapshot(p)) != V6.empty_row(st)
    p = pool()
    PR.genesis_reset(p)
    assert PR.export(p, S)["root"] == V6.genesis(st).hex()
    held = pool(free=(3, 1, 5, 2), held=(4,))
    with pytest.raises(PR.Refused, match="1 blocks are held: reset at an edge"):
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
    with pytest.raises(PR.Refused, match=match):
        PR.snapshot(_bad(change))


def test_a_held_block_or_a_pool_of_another_size_is_refused() -> None:
    """A window ends with no block held: a running request's blocks are refused (the edge's cut frees them first). The
    class's NB and B must be the pool's."""
    with pytest.raises(PR.Refused, match=r"1 blocks are held \(block 4 by 1\)"):
        PR.row(PR.snapshot(pool(free=(3, 1, 5, 2), held=(4,))))
    with pytest.raises(PR.Refused, match="keys blocks of 4 tokens, not the engine's 16"):
        PR.snapshot(pool(), 16)
    with pytest.raises(PR.Refused, match="a pool of 6 blocks under a class of NB = 7"):
        PR.root({**S, "NB": 7}, WANT)


def test_the_pool_s_class_is_the_served_rows_and_its_own_statics() -> None:
    extra = {k: S[k] for k in PR.POOL_STATICS}
    served = {k: v for k, v in S.items() if k not in PR.POOL_STATICS}
    assert PR.pool_class(served, extra) == S
    with pytest.raises(ValueError, match="past the served rows', not"):
        PR.pool_class(served, {**extra, "Q": 1})
    with pytest.raises(ValueError, match="DA decisions are at most D"):
        PR.pool_class(served, {**extra, "DA": S["D"] + 1})


def test_blockkey_v1_replaces_vllm_s_block_hash() -> None:
    """`install_keys` patches ``hash_block_tokens`` (which the request block hasher looks up at each call) with BlockKey_v1
    at the class's token width: ROOT under a request's first block, the parent's key under the rest; extra keys (LoRA,
    multimodal, a cache salt) are refused; uninstalling puts vLLM's back."""
    def vllm_hash(fn, parent, tokens, extra_keys=None):
        return b"vllm"

    mod = SimpleNamespace(hash_block_tokens=vllm_hash)
    hooks = PR.install_keys(S, module=mod)
    first = mod.hash_block_tokens(None, None, [1, 2, 3, 4])
    assert first == KA and mod.hash_block_tokens(None, first, [5, 6, 7, 8], None) == BK.block_key(KA, [5, 6, 7, 8], 4)
    with pytest.raises(PR.Refused, match="BlockKey_v1 keys its tokens alone"):
        mod.hash_block_tokens(None, None, [1, 2, 3, 4], ("salt",))
    hooks.uninstall()
    assert mod.hash_block_tokens is vllm_hash
