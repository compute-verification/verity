"""The KV cache the scheduler replay runs (`replay.Scheduler`): one full-attention group, vLLM V1's `KVCacheManager`,
`FullAttentionManager` and `BlockPool` at the replay's pin.

`BlockPool` holds the free-block queue (`_Free`: blocks taken from the front, freed to either end, and a cached one taken from
wherever it is on a hit), the hash map of full blocks, each request's blocks and the watermark, and gives prefix-cache
hits (`BlockPool.hit`), allocation, caching and freeing. `BlockPool.fork` copies it for a what-if run, its free queue (`_FreeFork`)
and hash map overlays on the parent's, which stays as it is.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Sequence

from verity_vllm.pipeline.advice_code.schedule import PREEMPTED, WAITING, Engine, RequestState, cdiv

__all__ = ["BlockPool"]


class _Free:
    """`BlockPool`'s free-block queue: blocks are taken from the front, freed to either end, and a cached one is taken
    from wherever it is on a hit."""

    def __init__(self, blocks: Iterable[int]) -> None:
        self.d: OrderedDict[int, None] = OrderedDict((b, None) for b in blocks)

    def __len__(self) -> int:
        return len(self.d)

    def pop(self) -> int:
        return self.d.popitem(last=False)[0]

    def take(self, b: int) -> None:
        del self.d[b]

    def push(self, b: int, *, front: bool) -> None:
        self.d[b] = None
        if front:
            self.d.move_to_end(b, last=False)


class _FreeFork:
    """A fork's free-block queue over its parent's, which stays as it is while the fork runs: the blocks the fork freed to
    the front, then the parent's it hasn't taken, then those it freed to the back."""

    def __init__(self, base: _Free) -> None:
        self.base, self.it, self.gone = base.d, iter(base.d), set()
        self.front: OrderedDict[int, None] = OrderedDict()
        self.back: OrderedDict[int, None] = OrderedDict()
        self.n = len(base.d)

    def __len__(self) -> int:
        return self.n

    def pop(self) -> int:
        self.n -= 1
        if self.front:
            return self.front.popitem(last=False)[0]
        for b in self.it:
            if b not in self.gone:
                return b
        return self.back.popitem(last=False)[0]

    def take(self, b: int) -> None:
        self.n -= 1
        if b in self.front:
            del self.front[b]
        elif b in self.back:
            del self.back[b]
        else:
            self.gone.add(b)

    def push(self, b: int, *, front: bool) -> None:
        self.n += 1
        if front:
            self.front[b] = None
            self.front.move_to_end(b, last=False)
        else:
            self.back[b] = None


class BlockPool:
    def __init__(self, engine: Engine) -> None:
        self.B = engine.block_size
        self.L = engine.max_model_len
        self.caching = engine.enable_prefix_caching
        self.watermark_blocks = int(engine.watermark * engine.num_gpu_blocks)
        self.free: _Free | _FreeFork = _Free(range(1, engine.num_gpu_blocks))  # block 0 is the null block
        self.ref = [0] * engine.num_gpu_blocks
        self.block_hash: list[int | None] = [None] * engine.num_gpu_blocks
        self.by_hash: dict[int, OrderedDict[int, None]] = {}
        self.mine: set[int] | None = None       # a fork's: the hashes whose blocks it has copied from its parent's
        self.req_blocks: dict[int, list[int]] = {}
        self.num_cached: dict[int, int] = {}
        self.owner: list[int | None] = [None] * engine.num_gpu_blocks     # the request that computes a block's KV

    def fork(self) -> BlockPool:
        """A copy to run on while this one waits (its free queue and hash map are overlays on this one's)."""
        assert isinstance(self.free, _Free)
        k = object.__new__(BlockPool)
        k.B, k.L, k.caching, k.watermark_blocks = self.B, self.L, self.caching, self.watermark_blocks
        k.free, k.ref, k.block_hash, k.owner = _FreeFork(self.free), list(self.ref), list(self.block_hash), list(self.owner)
        k.by_hash, k.mine = dict(self.by_hash), set()
        k.req_blocks = {i: list(bs) for i, bs in self.req_blocks.items()}
        k.num_cached = dict(self.num_cached)
        return k

    def _blocks(self, h: int) -> OrderedDict[int, None]:
        """The blocks cached under h, to change (a fork's own copy)."""
        bs = self.by_hash.get(h)
        if bs is None or (self.mine is not None and h not in self.mine):
            bs = self.by_hash[h] = OrderedDict() if bs is None else OrderedDict(bs)
            if self.mine is not None:
                self.mine.add(h)
        return bs

    def _pop_hash(self, b: int) -> None:
        h = self.block_hash[b]
        if h is None:
            return
        blocks = self._blocks(h)
        del blocks[b]
        if not blocks:
            del self.by_hash[h]
        self.block_hash[b] = None

    def _get_new_blocks(self, k: int) -> list[int]:
        assert k <= len(self.free)
        out = []
        for _ in range(k):
            b = self.free.pop()
            if self.caching:
                self._pop_hash(b)
            self.ref[b] = 1
            out.append(b)
        return out

    def _free_blocks(self, ordered: Iterable[int]) -> None:
        first, last = [], []
        for b in ordered:
            self.ref[b] -= 1
            if self.ref[b] == 0:
                (first if self.block_hash[b] is None or not self.caching else last).append(b)
        for b in reversed(first):
            self.free.push(b, front=True)
        for b in last:
            self.free.push(b, front=False)

    def hit(self, r: RequestState) -> list[int]:
        """`find_longest_cache_hit` over the request's full-block hashes, at most `num_tokens - 1` tokens."""
        if not self.caching:
            return []
        out = []
        for h in r.block_hashes[: (r.num_tokens - 1) // self.B]:
            blocks = self.by_hash.get(h)
            if not blocks:
                break
            out.append(next(iter(blocks)))
        return out

    def blocks_to_allocate(self, r: RequestState, num_tokens: int, hit: Sequence[int]) -> int:
        held = len(self.req_blocks.get(r.i, ()))
        evictable = sum(1 for b in hit if self.ref[b] == 0)
        return max(cdiv(num_tokens, self.B) - len(hit) - held, 0) + evictable

    def allocate_slots(self, r: RequestState, n: int, hit: Sequence[int] = (), *, full_sequence_must_fit: bool = False,
                       has_scheduled_reqs: bool = True, dry: bool = False) -> bool:
        num_local = r.num_computed + len(hit) * self.B
        total = min(num_local, self.L)
        wm = self.watermark_blocks if has_scheduled_reqs and r.status in (WAITING, PREEMPTED) else 0
        if full_sequence_must_fit and self.blocks_to_allocate(r, min(r.num_tokens, self.L), hit) + wm > len(self.free):
            return False
        need_slot = min(total + n, self.L)
        if self.blocks_to_allocate(r, need_slot, hit) + wm > len(self.free):
            return False
        if dry:
            return True
        blocks = self.req_blocks.setdefault(r.i, [])
        if hit:
            for b in hit:
                if self.ref[b] == 0:
                    self.free.take(b)
                self.ref[b] += 1
            blocks.extend(hit)
            self.num_cached[r.i] = len(blocks)
        k = cdiv(need_slot, self.B) - len(blocks)
        if k > 0:
            new = self._get_new_blocks(k)
            for b in new:
                self.owner[b] = r.i
            blocks.extend(new)
        if self.caching:
            self.cache_blocks(r, min(total + n, r.num_tokens))
        return True

    def cache_blocks(self, r: RequestState, num_tokens: int) -> None:
        done = self.num_cached.get(r.i, 0)
        full = num_tokens // self.B
        if done >= full:
            return
        blocks = self.req_blocks[r.i]
        for k in range(done, full):
            b = blocks[k]
            if self.block_hash[b] is not None:
                self._pop_hash(b)
            h = r.block_hashes[k]
            self._blocks(h)[b] = None
            self.block_hash[b] = h
        self.num_cached[r.i] = full

    def free_request(self, r: RequestState) -> None:
        blocks = self.req_blocks.pop(r.i, [])
        self.num_cached.pop(r.i, None)
        self._free_blocks(reversed(blocks))
