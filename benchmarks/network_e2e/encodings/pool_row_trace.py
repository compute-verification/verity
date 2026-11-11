"""The carried pool's row export (`verity_vllm.engine.pool_row`) over a recorded trace: vLLM d9105ea80's prefix-cache
``BlockPool`` (`BlockPool`, transcribed below: one KV cache group, full blocks only) driven by the replay's pool
(`replay._KV`), keyed by BlockKey_v1.

`drive` replays a trace with `_KV` tapped (not a fork's, not a dry run) and makes each of its pool operations on the
transcribed pool, in vLLM's order:
- a hit's ``touch``;
- ``get_new_blocks``;
- ``cache_full_blocks``;
- ``free_blocks``.
After each one it holds the two pools to each other: the same block ids, references, keys (BlockKey_v1 of `_KV`'s interned
hashes, `_Hashes`), free queue order and each key's copies in order. Each lookup's hit is the transcribed hash map's first
copy (``get_one_block``), and the block past a hit isn't cached. At every operation after which no block is held (an
idle engine, and the trace's end), `pool_row.export` exports the transcribed pool. Its row must equal the one built
here from `_KV` alone, and so must its root (`compile_slice_v6.root`). After the end, `pool_row.genesis_reset` leaves
the empty row (`compile_slice_v6.genesis`).

    python3 pool_row_trace.py tests/fixtures/async-prefix-cache.jsonl [--every 1]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_pool_vectors as CPV
import replay as RP

from verity.definitions.commitments.gates import block_key as BK
from verity.protocols.certifier.prover import compile_slice_v4 as V4, compile_slice_v6 as V6
from verity_vllm.engine import pool_row as PR

GROUP = bytes(PR.GROUP_BYTES)
#: the replay's pool and block hashes, under the names the advice decoder's module gives them where it has them
KV, HASHES = getattr(RP, "_KV", None) or RP.BlockPool, getattr(RP, "_Hashes", None) or RP.BlockHashes
NEW_BLOCKS = "_new_blocks" if hasattr(KV, "_new_blocks") else "_get_new_blocks"


# --- vLLM d9105ea80's BlockPool (vllm/v1/core/block_pool.py, kv_cache_utils.py), one group, full blocks ----------------


class Block:
    """``KVCacheBlock``."""

    def __init__(self, block_id: int) -> None:
        self.block_id, self.ref_cnt, self.is_null = block_id, 0, False
        self._block_hash: bytes | None = None
        self._block_hash_num_tokens: int | None = None
        self.prev_free_block: Block | None = None
        self.next_free_block: Block | None = None

    @property
    def block_hash(self) -> bytes | None:
        return self._block_hash

    @property
    def block_hash_num_tokens(self) -> int | None:
        return self._block_hash_num_tokens

    def reset_hash(self) -> None:
        self._block_hash = self._block_hash_num_tokens = None


class FreeQueue:
    """``FreeKVCacheBlockQueue``."""

    def __init__(self, blocks: list[Block]) -> None:
        self.num_free_blocks = len(blocks)
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]
        self.fake_free_list_head, self.fake_free_list_tail = Block(-1), Block(-1)
        self.fake_free_list_head.next_free_block, blocks[0].prev_free_block = blocks[0], self.fake_free_list_head
        self.fake_free_list_tail.prev_free_block, blocks[-1].next_free_block = blocks[-1], self.fake_free_list_tail

    def popleft(self) -> Block:
        return self.popleft_n(1)[0]

    def popleft_n(self, n: int) -> list[Block]:
        if n == 0:
            return []
        assert self.num_free_blocks >= n
        self.num_free_blocks -= n
        curr, ret = self.fake_free_list_head.next_free_block, []
        for _ in range(n):
            assert curr is not None
            ret.append(curr)
            last, curr = curr, curr.next_free_block
            last.prev_free_block = last.next_free_block = None
        if curr is not None:
            self.fake_free_list_head.next_free_block, curr.prev_free_block = curr, self.fake_free_list_head
        return ret

    def remove(self, b: Block) -> None:
        assert b.prev_free_block is not None and b.next_free_block is not None
        b.prev_free_block.next_free_block, b.next_free_block.prev_free_block = b.next_free_block, b.prev_free_block
        b.prev_free_block = b.next_free_block = None
        self.num_free_blocks -= 1

    def prepend_n(self, blocks: list[Block]) -> None:
        if not blocks:
            return
        first, prev = self.fake_free_list_head.next_free_block, self.fake_free_list_head
        for b in blocks:
            b.prev_free_block, prev.next_free_block, prev = prev, b, b
        prev.next_free_block, first.prev_free_block = first, prev
        self.num_free_blocks += len(blocks)

    def append_n(self, blocks: list[Block]) -> None:
        if not blocks:
            return
        last = self.fake_free_list_tail.prev_free_block
        for b in blocks:
            b.prev_free_block, last.next_free_block, last = last, b, b
        last.next_free_block, self.fake_free_list_tail.prev_free_block = self.fake_free_list_tail, last
        self.num_free_blocks += len(blocks)


class HashMap:
    """``BlockHashToBlockMap``."""

    def __init__(self) -> None:
        self._cache: dict[bytes, Block | dict[int, Block]] = {}

    def get_one_block(self, key: bytes) -> Block | None:
        v = self._cache.get(key)
        return v if isinstance(v, Block) or v is None else next(iter(v.values()))

    def insert(self, key: bytes, b: Block) -> None:
        v = self._cache.get(key)
        if v is None:
            self._cache[key] = b
        elif isinstance(v, Block):
            self._cache[key] = {v.block_id: v, b.block_id: b}
        else:
            v[b.block_id] = b

    def pop(self, key: bytes, block_id: int) -> Block | None:
        v = self._cache.pop(key, None)
        if v is None:
            return None
        if isinstance(v, Block):
            if v.block_id == block_id:
                return v
            self._cache[key] = v
            return None
        b = v.pop(block_id, None)
        if v:
            self._cache[key] = v
        return b


class BlockPool:
    """``BlockPool`` with caching on: its free queue, its hash map and the calls `_KV` makes."""

    def __init__(self, num_gpu_blocks: int, hash_block_size: int) -> None:
        self.num_gpu_blocks, self.hash_block_size = num_gpu_blocks, hash_block_size
        self.blocks = [Block(i) for i in range(num_gpu_blocks)]
        self.free_block_queue = FreeQueue(self.blocks)
        self.cached_block_hash_to_block = HashMap()
        self.cached_block_hashes_by_block: dict[int, set[bytes]] = {}
        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

    def get_num_free_blocks(self) -> int:
        return self.free_block_queue.num_free_blocks

    def _remove_cached_block_hashes(self, b: Block) -> None:
        for h in [b.block_hash] if b.block_hash is not None else []:
            self.cached_block_hash_to_block.pop(h, b.block_id)
        b.reset_hash()

    def get_new_blocks(self, n: int) -> list[Block]:
        if n > self.get_num_free_blocks():
            raise ValueError(f"Cannot get {n} free blocks from the pool")
        ret = self.free_block_queue.popleft_n(n)
        for b in ret:
            self._remove_cached_block_hashes(b)
            assert b.ref_cnt == 0
            b.ref_cnt += 1
        return ret

    def touch(self, blocks: Sequence[Block]) -> None:
        for b in blocks:
            if b.ref_cnt == 0 and not b.is_null:
                self.free_block_queue.remove(b)
            b.ref_cnt += 1

    def cache_full_blocks(self, keys: Sequence[bytes], blocks: Sequence[Block], num_cached: int, num_full: int,
                          block_size: int) -> None:
        for i, b in enumerate(blocks[num_cached:num_full]):
            assert b.block_hash is None, f"block {b.block_id} is cached already"
            key = keys[num_cached + i] + GROUP
            b._block_hash, b._block_hash_num_tokens = key, (num_cached + i + 1) * block_size
            self.cached_block_hash_to_block.insert(key, b)

    def free_blocks(self, ordered: Iterable[Block]) -> None:
        last, first = [], []
        for b in ordered:
            b.ref_cnt -= 1
            if b.ref_cnt == 0 and not b.is_null:
                (first if b.block_hash is None else last).append(b)
        self.free_block_queue.prepend_n(first)
        self.free_block_queue.append_n(last)

    def reset_prefix_cache(self) -> bool:
        if self.num_gpu_blocks - self.get_num_free_blocks() != 1:
            return False
        self.cached_block_hash_to_block = HashMap()
        self.cached_block_hashes_by_block.clear()
        for b in self.blocks:
            b.reset_hash()
        return True


# --- the row from `_KV` alone ----------------------------------------------------------------------------------------


def kv_row(kv: Any, key: dict[int, bytes]) -> V6.PoolRow:
    """The pool row of `_KV` with no block held: its free queue's order, each hash's blocks in order, BlockKey_v1 keys."""
    n = len(kv.ref) - 1
    free = list(kv.free.d)
    prev, nxt, cprev, cnext, mlast = ([0] * n for _ in range(5))
    for j, b in enumerate(free):
        prev[b - 1], nxt[b - 1] = free[j - 1] if j else 0, free[j + 1] if j + 1 < len(free) else 0
    for bs in kv.by_hash.values():
        ids = list(bs)
        for j, b in enumerate(ids):
            cprev[b - 1], cnext[b - 1] = ids[j - 1] if j else 0, ids[j + 1] if j + 1 < len(ids) else 0
        mlast[ids[0] - 1] = ids[-1]
    keys = tuple(V6.NOKEY if kv.block_hash[b] is None else key[kv.block_hash[b]] for b in range(1, n + 1))
    return V6.PoolRow(free[0] if free else 0, free[-1] if free else 0, tuple(prev), tuple(nxt), tuple(cprev), tuple(cnext),
                      tuple(mlast), keys)


def statics(inp: CPV.Input) -> dict[str, int]:
    """A carried pool's class for the trace's engine: `compile_slice_v6.demo`'s, at the trace's NB, B and token width."""
    e = inp.engine
    ctw = max(max(max(r.tokens) for r in inp.requests).bit_length(), 1)
    st = {**V6.demo()[0], "NB": e.num_gpu_blocks, "B": e.block_size, "CTW": ctw}
    V6.V5.shape(SimpleNamespace(**st))
    return st


# --- the drive -------------------------------------------------------------------------------------------------------


class Driver:
    """The transcribed pool beside `_KV`, each interned hash's BlockKey_v1 key (`keyed`) and the counts: `check` holds the
    two pools to each other after an operation (every ``every``-th while a block is held), and `edge` exports the row
    wherever none is."""

    def __init__(self, S: dict[str, int], every: int) -> None:
        self.S, self.tok, self.every = S, V4.key_tok(SimpleNamespace(**S)), every
        self.T = BlockPool(S["NB"], S["B"])
        self.key: dict[int, bytes] = {}
        self.counts: Counter = Counter()
        self.roots: list[str] = []

    def keys_of(self, r: Any) -> list[bytes]:
        return [self.key[h] for h in r.block_hashes]

    def check(self, kv: Any) -> None:
        self.counts["ops"] += 1
        T = self.T
        if any(kv.ref[b] for b in range(1, len(kv.ref))):
            if self.counts["ops"] % self.every:
                return
        else:
            self.edge(kv)
        if [b.ref_cnt for b in T.blocks[1:]] != kv.ref[1:]:
            raise AssertionError(f"op {self.counts['ops']}: references leave _KV's")
        p = PR.snapshot(T, self.S["B"])
        if list(p.free) != list(kv.free.d):
            raise AssertionError(f"op {self.counts['ops']}: the free queue leaves _KV's")
        want = {self.key[h]: tuple(bs) for h, bs in kv.by_hash.items()}
        if p.copies != want:
            raise AssertionError(f"op {self.counts['ops']}: the cached copies leave _KV's")
        if any(p.key[b] != (None if kv.block_hash[b] is None else self.key[kv.block_hash[b]]) for b in range(len(p.key))):
            raise AssertionError(f"op {self.counts['ops']}: a block's key leaves _KV's")
        self.counts["checks"] += 1

    def edge(self, kv: Any) -> None:
        got = PR.export(self.T, self.S["B"])
        want = kv_row(kv, self.key)
        if PR.row(PR.snapshot(self.T)) != want:
            raise AssertionError(f"op {self.counts['ops']}: the exported row leaves the one _KV gives")
        if got["root"] != V6.root(SimpleNamespace(**self.S), want).hex():
            raise AssertionError(f"op {self.counts['ops']}: the exported root leaves the one _KV gives")
        self.roots.append(got["root"])
        self.counts["edges"] += 1
        self.counts["cached at an edge"] = max(self.counts["cached at an edge"], sum(1 for k in want.key if k != V6.NOKEY))
        self.counts["copies at an edge"] = max(self.counts["copies at an edge"], max(map(len, kv.by_hash.values()), default=0))


@contextmanager
def tapped(drv: Driver):
    """`_KV`'s pool operations, each made on ``drv``'s transcribed pool too and checked (the module's docstring)."""
    K = KV
    names = ("hit", "allocate_slots", "cache_blocks", NEW_BLOCKS, "_free_blocks")
    orig = {n: getattr(K, n) for n in names}
    T = drv.T

    def hit(self, r):
        out = orig["hit"](self, r)
        if self.mine is None and self.caching:
            keys = drv.keys_of(r)
            for k, b in enumerate(out):
                got = T.cached_block_hash_to_block.get_one_block(keys[k] + GROUP)
                if got is None or got.block_id != b:
                    raise AssertionError(f"request {r.i}'s hit at block {k}: {got and got.block_id}, _KV's {b}")
            limit = (r.num_tokens - 1) // self.B
            if len(out) < limit and T.cached_block_hash_to_block.get_one_block(keys[len(out)] + GROUP) is not None:
                raise AssertionError(f"request {r.i}'s hit stops at block {len(out)}, which vLLM caches")
            drv.counts["lookups"] += 1
            drv.counts["hit blocks"] += len(out)
        return out

    def allocate_slots(self, r, n, hit=(), *, full_sequence_must_fit=False, has_scheduled_reqs=True, dry=False):
        kw = {"full_sequence_must_fit": full_sequence_must_fit, "has_scheduled_reqs": has_scheduled_reqs}
        if dry or self.mine is not None:
            return orig["allocate_slots"](self, r, n, hit, dry=dry, **kw)
        fits = orig["allocate_slots"](self, r, n, hit, dry=True, **kw)
        if fits and hit:
            T.touch([T.blocks[b] for b in hit])
        ok = orig["allocate_slots"](self, r, n, hit, **kw)
        assert ok == fits
        if ok:
            drv.check(self)
        return ok

    def cache_blocks(self, r, num_tokens):
        done = self.num_cached.get(r.i, 0)
        orig["cache_blocks"](self, r, num_tokens)
        if self.mine is None:
            full = self.num_cached.get(r.i, 0)
            if full > done:
                T.cache_full_blocks(drv.keys_of(r), [T.blocks[b] for b in self.req_blocks[r.i]], done, full, self.B)
                drv.counts["cached"] += full - done

    def new_blocks(self, k):
        out = orig[NEW_BLOCKS](self, k)
        if self.mine is None:
            got = [b.block_id for b in T.get_new_blocks(k)]
            if got != out:
                raise AssertionError(f"get_new_blocks({k}): {got}, _KV's {out}")
        return out

    def _free_blocks(self, ordered):
        ordered = list(ordered)
        orig["_free_blocks"](self, ordered)
        if self.mine is None:
            T.free_blocks([T.blocks[b] for b in ordered])
            drv.check(self)

    taps = {"hit": hit, "allocate_slots": allocate_slots, "cache_blocks": cache_blocks, NEW_BLOCKS: new_blocks,
            "_free_blocks": _free_blocks}
    for n in names:
        setattr(K, n, taps[n])
    try:
        yield
    finally:
        for n in names:
            setattr(K, n, orig[n])


@contextmanager
def keyed(drv: Driver):
    """The replay's interned block hashes, each keyed by BlockKey_v1 in ``drv.key`` as it is interned."""
    H = HASHES
    intern = H.__call__

    def call(self, p, block):
        h = intern(self, p, block)
        if h not in drv.key:
            drv.key[h] = BK.block_key(drv.key[p] if p else BK.ROOT, block, drv.tok)
        return h

    H.__call__ = call
    try:
        yield
    finally:
        H.__call__ = intern


def drive(path: Path, *, every: int = 1) -> dict[str, Any]:
    """The transcribed pool over the trace at ``path``, held to `_KV` (the module's docstring); its counts and the root
    at each edge. Raises AssertionError where they disagree."""
    inp = CPV.replay_input(path)
    if not inp.engine.enable_prefix_caching:
        raise ValueError(f"{path.name}: the pool row needs prefix caching on")
    drv = Driver(statics(inp), every)
    with keyed(drv), tapped(drv):
        rp = RP.replay(inp.engine, inp.requests, admit=inp.before)
    if [list(s.rows) for s in rp.steps] != inp.rows:
        raise AssertionError("the replay leaves the trace")
    PR.genesis_reset(drv.T)
    if PR.export(drv.T, drv.S["B"])["root"] != V6.genesis(SimpleNamespace(**drv.S)).hex():
        raise AssertionError("the reset pool isn't the empty row")
    return {**dict(drv.counts), "steps": len(rp.steps), "keys": len(drv.key), "roots": drv.roots}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--every", type=int, default=1, help="hold the two pools to each other every so many operations")
    a = ap.parse_args(argv)
    for t in a.traces:
        got = drive(t, every=a.every)
        print(json.dumps({"trace": t.name, **{k: v for k, v in got.items() if k != "roots"}, "edges_roots": len(got["roots"])}))


if __name__ == "__main__":
    main()
