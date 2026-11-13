"""The carried pool's row: vLLM d9105ea80's prefix-cache ``BlockPool`` at a window's edge as the pool row the network
certifier's compile slice carries between windows (`verity.commitments.pool_row`), with vLLM keying its blocks by
BlockKey_v1 so that the row's keys are the ones vLLM hits on.

**The snapshot** (`snapshot`) reads a live pool as plain data (`Pool`). It takes each block's reference count and key,
the free queue from its head, and each key's copies in vLLM's order: one block, or a ``{block_id: block}`` dict in the
order they were cached, the first being the one ``get_one_block`` hits. A block's ``_block_hash`` is its key followed by
its KV cache group's id, 4 bytes big-endian (``make_block_hash_with_group_id``). The snapshot refuses (`Refused`)
anything the row can't hold:
- a null block other than block 0;
- a key of a group other than 0, or one that isn't 64 bytes;
- a key reaching inside a block (``_block_hash_num_tokens``, a partial hit's), or a block cached under two keys
  (``cached_block_hashes_by_block``);
- a free queue whose links or count disagree;
- a copy list that disagrees with its blocks' keys.

**The row** (`row`) has, per block b ≥ 1 at index b − 1, these fields:
- ``prev`` and ``next``: b's neighbours in the free queue, 0 at an end; the queue's ends are ``head`` and ``tail``;
- ``cprev`` and ``cnext``: b's neighbours among its key's copies;
- ``mlast``: at a key's first copy, its last copy (itself when it is the only one), else 0;
- ``key``: ``NOKEY`` for none.

A pool between windows holds no block, so `row` refuses one with a holder. A window's edge comes where no request runs,
so every block 1 … NB − 1 is in the free queue. `export` writes the row and its root (`pool_row.root`).

**Genesis.** A fresh pool is the empty row (`pool_row.empty`). vLLM's ``reset_prefix_cache`` clears every
key but leaves the free queue in the order it was in, so `genesis_reset` also relinks the queue in id order. It is the
reset a run starts its chain from, and the one after a failed window.

**The keys** (`install_keys`) replace ``vllm.v1.core.kv_cache_utils.hash_block_tokens``, which the engine core's request
block hasher looks up at each call, with BlockKey_v1. A block's key is SHA-512(parent ‖ its tokens), each token TOK bits
(`verity.definitions.commitments.gates.block_key.block_key`), with `ROOT` as the parent of a request's first block. vLLM then
caches and hits under the circuit's keys. A block with extra keys is refused, since BlockKey_v1 keys tokens alone:
LoRA, multimodal inputs, a cache salt and prompt embeddings all bring extra keys.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from verity.commitments import pool_row as R
from verity.definitions.commitments.gates import block_key as BK
from verity_vllm.engine.hooks import Hooks
from verity_vllm.errors import Refused

__all__ = ["GROUP_BYTES", "Pool", "export", "genesis_reset", "install_keys", "row", "snapshot"]

#: the group id's bytes at the end of a vLLM block hash (``make_block_hash_with_group_id``)
GROUP_BYTES = 4
KEY_BYTES = 64
KV_CACHE_UTILS = "vllm.v1.core.kv_cache_utils"


class Pool(NamedTuple):
    """A block pool as plain data, by block id (the null block 0 included)."""
    ref: tuple[int, ...]
    key: tuple[bytes | None, ...]
    #: the free queue from its head
    free: tuple[int, ...]
    #: each cached key's blocks, the first copy (the one a hit takes) first
    copies: dict[bytes, tuple[int, ...]]


def _key(h: bytes, b: int) -> bytes:
    h = bytes(h)
    group = int.from_bytes(h[-GROUP_BYTES:], "big")
    if group:
        raise Refused(f"block {b} is cached in KV cache group {group}: the pool row holds one group's keys")
    if len(h) != KEY_BYTES + GROUP_BYTES:
        raise Refused(f"block {b}'s key is {len(h) - GROUP_BYTES} bytes, not BlockKey_v1's {KEY_BYTES} (install_keys)")
    return h[:-GROUP_BYTES]


def _free(pool: Any, n: int) -> tuple[int, ...]:
    fq = pool.free_block_queue
    head, tail = fq.fake_free_list_head, fq.fake_free_list_tail
    out: list[int] = []
    prev, b = head, head.next_free_block
    while b is not tail:
        if b is None or b.prev_free_block is not prev or len(out) == n:
            raise Refused(f"the free queue breaks after {len(out)} blocks")
        out.append(b.block_id)
        prev, b = b, b.next_free_block
    if tail.prev_free_block is not prev or len(out) != fq.num_free_blocks:
        raise Refused(f"the free queue links {len(out)} blocks but counts {fq.num_free_blocks}")
    return tuple(out)


def snapshot(pool: Any, block_size: int | None = None) -> Pool:
    """``pool`` (vLLM d9105ea80's ``BlockPool``) as plain data; `Refused` where the row can't hold it, or where it keys
    blocks of other than ``block_size`` tokens when that is given."""
    blocks = pool.blocks
    if pool.null_block.block_id != 0:
        raise Refused(f"the null block is block {pool.null_block.block_id}, not 0")
    B = pool.hash_block_size
    if block_size is not None and B != block_size:
        raise Refused(f"the pool keys blocks of {B} tokens, not the engine's {block_size}")
    if pool.cached_block_hashes_by_block:
        b = next(iter(pool.cached_block_hashes_by_block))
        raise Refused(f"block {b} is cached under a second key (a partial hit's)")
    keys: list[bytes | None] = []
    for blk in blocks:
        h = blk.block_hash
        n = blk.block_hash_num_tokens
        if h is not None and n is not None and n % B:
            raise Refused(f"block {blk.block_id}'s key covers {n} tokens, inside a block of {B}")
        keys.append(None if h is None else _key(h, blk.block_id))
    copies: dict[bytes, tuple[int, ...]] = {}
    for h, v in pool.cached_block_hash_to_block._cache.items():
        ids = (v.block_id,) if hasattr(v, "block_id") else tuple(v)
        k = _key(h, ids[0] if ids else -1)
        if not ids or any(keys[b] != k for b in ids):
            raise Refused(f"key {k[:8].hex()}… lists blocks {list(ids)}, not the blocks cached under it")
        copies[k] = ids
    if sum(map(len, copies.values())) != sum(k is not None for k in keys):
        raise Refused("a block has a key that the hash map doesn't list")
    return Pool(tuple(blk.ref_cnt for blk in blocks), tuple(keys), _free(pool, len(blocks)), copies)


def row(p: Pool) -> R.PoolRow:
    """``p`` as its pool row; `Refused` where a block is held or not free."""
    nb = len(p.ref)
    held = [b for b in range(1, nb) if p.ref[b]]
    if held:
        raise Refused(f"{len(held)} blocks are held (block {held[0]} by {p.ref[held[0]]}): a window ends with none")
    if sorted(p.free) != list(range(1, nb)):
        raise Refused(f"the free queue holds {len(p.free)} blocks, not each of 1 … {nb - 1} once")
    links = {f: [0] * (nb - 1) for f in ("prev", "next", "cprev", "cnext", "mlast")}

    def chain(ids: Sequence[int], back: str, fwd: str) -> None:
        for j, b in enumerate(ids):
            links[back][b - 1] = ids[j - 1] if j else 0
            links[fwd][b - 1] = ids[j + 1] if j + 1 < len(ids) else 0

    chain(p.free, "prev", "next")
    for ids in p.copies.values():
        chain(ids, "cprev", "cnext")
        links["mlast"][ids[0] - 1] = ids[-1]
    return R.PoolRow(p.free[0] if p.free else 0, p.free[-1] if p.free else 0,
                     *(tuple(links[f]) for f in R.LINKS + ("mlast",)), tuple(p.key[b] or R.NOKEY for b in range(1, nb)))


def export(pool: Any, block_size: int, path: Path | None = None) -> dict[str, Any]:
    """``pool``'s row (`snapshot` at ``block_size``, `row`) and root, written to ``path`` (the row's fields, the root)
    when given; the public record (NB and the root)."""
    r = row(snapshot(pool, block_size))
    rt = R.root(r)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = {f: getattr(r, f) for f in r._fields if f != "key"}
        path.write_text(json.dumps({"nb": len(r.key) + 1, "root": rt.hex(), **fields, "key": [k.hex() for k in r.key]}))
    return {"nb": len(r.key) + 1, "root": rt.hex()}


def genesis_reset(pool: Any) -> None:
    """Reset ``pool`` to the empty row: every key cleared (``reset_prefix_cache``) and the free queue relinked in id
    order, as a fresh pool's is. `Refused` while a block is held."""
    if pool.free_block_queue.num_free_blocks != len(pool.blocks) - 1:
        raise Refused(f"{len(pool.blocks) - 1 - pool.free_block_queue.num_free_blocks} blocks are held: reset at an edge")
    if not pool.reset_prefix_cache():
        raise Refused("the pool refused to reset its prefix cache")
    fq = pool.free_block_queue
    prev = fq.fake_free_list_head
    for blk in pool.blocks[1:]:
        prev.next_free_block, blk.prev_free_block, prev = blk, prev, blk
    prev.next_free_block, fq.fake_free_list_tail.prev_free_block = fq.fake_free_list_tail, prev


def hasher(tok: int) -> Callable[..., bytes]:
    """``hash_block_tokens`` under BlockKey_v1 at TOK = ``tok`` bits a token."""

    def hash_block_tokens(hash_function: Any, parent_block_hash: bytes | None, curr_block_token_ids: Sequence[int],
                          extra_keys: tuple[Any, ...] | None = None) -> bytes:
        if extra_keys:
            raise Refused(f"a block with extra keys {extra_keys!r:.80}: BlockKey_v1 keys its tokens alone")
        return BK.block_key(parent_block_hash or BK.ROOT, curr_block_token_ids, tok)

    return hash_block_tokens


def install_keys(ctw: int, block_size: int, hooks: Hooks | None = None, module: Any = None) -> Hooks:
    """Key vLLM's blocks of ``block_size`` tokens by BlockKey_v1 (`hasher`) for tokens of ``ctw`` bits
    (`block_key.token_width`), in this process: before the engine core hashes a request."""
    if module is None:
        import importlib
        module = importlib.import_module(KV_CACHE_UTILS)
    hooks = hooks or Hooks(tag="pool-row-keys")
    hooks.patch(module, "hash_block_tokens", hasher(BK.token_width(ctw, block_size)))
    return hooks
