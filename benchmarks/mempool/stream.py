"""`CacheStep_v3`'s specification (`verity.ml.protocols.certifier.cache_step_v3.Stream`) held to the RAM pool (`pool_ram`,
itself held to vLLM's `BlockPool`) after every pool call, over the encodings' traces and synthetic ones: the two pools' states
agree (but for the memo's stamps, which the RAM pool doesn't have), and so do each call's results, every memoized hit's
among them. The stream is cut into windows of about ``--window`` slots, the pool carried from one to the next, and each
window passes its memory check (R = W, every read's t below its now). ``--vectors`` writes each trace's first windows
that evict for the circuit's test (`test_network_certifier_cache_step_v3`). `run` with ``check`` off and ``every`` 0 is
`cost.py`'s measurement of a load's stream slots.

    python3 stream.py [--traces DIR] [--seeds N] [--window SLOTS] [--vectors OUT] [--json OUT]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pool_ram as PR
from difftest import FIXTURES, load, synthetic
from pool_ram import RP

from verity.ml.protocols.certifier import cache_step_v3 as V3

T = V3.TAGS
TRACES = ("async-prefix-cache.jsonl", "async-crush.json.gz", "sync-crush.json.gz")


def statics(eng: RP.Engine, reqs: list, TW: int = 40) -> SimpleNamespace:
    """The widths a run needs: request numbers below 2^RW − 1, a request's blocks at most 2^KW, holders below
    2^REFW − 1."""
    return SimpleNamespace(NB=eng.num_gpu_blocks, RW=max(len(reqs).bit_length(), 1),
                           KW=max(PR.cdiv(eng.max_model_len, eng.block_size).bit_length(), 1),
                           REFW=(eng.max_num_seqs + 1).bit_length(), TW=TW)


def key128(h) -> int:
    """A vLLM block hash's 128-bit stand-in (in the circuit, its SHA-512 key's print under the window's coin)."""
    return int.from_bytes(hashlib.sha512(repr(h).encode()).digest()[:16], "little") or 1


class _Keys:
    """RK's keys, read from the replay's requests when the stream first reads them."""

    def __init__(self, pool: StreamPool) -> None:
        self.pool = pool

    def get(self, addr: int, default: int = 0) -> int:
        S = self.pool.S
        hs = self.pool.reqs[addr & ((1 << S.RW) - 1)].block_hashes
        k = addr >> S.RW
        return key128(hs[k]) if k < len(hs) else default


class StreamPool(PR.RamPool):
    """The RAM pool, with a `Stream` running the same calls beside it, held to it after each top-level call."""

    def start(self, S: SimpleNamespace, window: int, keep: int, check: bool = True, every: int = 1) -> None:
        """Without ``check``, a window's memory check is skipped; the pools are compared after every ``every``-th call
        (0: at the end only), each call's results after every call."""
        self.S, self.window, self.keep_windows, self.check, self.every = S, window, keep, check, every
        self.reqs: dict[int, object] = {}
        self.carried: dict[tuple[int, int], int] = {}
        self.base = V3.empty_pool(S, _Keys(self))
        self.windows: list[dict] = []
        self.counts: Counter = Counter()
        self.depth = self.calls = self.total = 0
        self._open(1, S.NB - 1)

    def _open(self, head: int, tail: int) -> None:
        self.head0, self.tail0, self.ev0 = head, tail, self.ops["evict"]

        def initial(tag: int, addr: int) -> int:
            return self.carried[(tag, addr)] if (tag, addr) in self.carried else self.base(tag, addr)

        self.stream = V3.Stream(self.S, initial, head, tail)

    def close(self) -> None:
        """Ends the window: its memory check, its vectors if kept, and the pool it leaves carried to the next."""
        st = self.stream
        if not st.slots:
            return
        if self.check:
            rng = random.Random(len(self.windows))
            alpha, gamma = rng.getrandbits(128), rng.getrandbits(128)
            R, W = st.fingerprints(alpha, gamma)
            assert R == W and all(s.t < s.now for s in st.slots), f"window {len(self.windows)} fails its memory check"
        self.counts["windows"] += 1
        self.counts["slots"] += len(st.slots)
        self.counts["entries"] += len(st.entries())
        self.counts["max window slots"] = max(self.counts["max window slots"], len(st.slots))
        self.counts["max window entries"] = max(self.counts["max window entries"], len(st.entries()))
        evicts = self.ops["evict"] - self.ev0
        self.counts["evictions"] += evicts
        if len(self.windows) < self.keep_windows and evicts:
            self.windows.append({"head": self.head0, "tail": self.tail0,
                                 "slots": [list(s[:5]) + [s.v, s.t] for s in st.slots],
                                 "entries": [list(e) for e in st.entries()], "records": [list(r) for r in st.records],
                                 "end": [st.head, st.tail]})
        for k, (v, _) in st.cells.items():
            self.carried[k] = v
        self._open(st.head, st.tail)

    def _call(self, kind: str, r: int, a: int, b: int = 0) -> tuple[int, int]:
        self.counts[kind] += 1
        st = self.stream
        n0, m0, w0 = len(st.slots), st.memos, st.rewalks
        out = st.call(V3.Call(kind, r, a, b))
        self.total += len(st.slots) - n0
        self.counts["memo hits"] += st.memos - m0
        self.counts["re-walks"] += st.rewalks - w0
        return out

    def _agree(self) -> None:
        self.calls += 1
        where = self.every and self.calls % self.every == 0 and disagreement(self)
        if where:
            raise AssertionError(f"the stream leaves the RAM pool at its {where}")
        if len(self.stream.slots) >= self.window:
            self.close()

    def hit(self, r):
        self.reqs[r.i] = r
        out = super().hit(r)
        lim = min(len(r.block_hashes), (r.num_tokens - 1) // self.B)
        if self.caching and lim > 0:
            got = self._call("hit", r.i, lim)
            if got != (len(out), self.looked[r.i][1]):
                raise AssertionError(f"hit of request {r.i}: the stream's {got}, the pool's {len(out), self.looked[r.i][1]}")
        self._agree()
        return out

    def allocate_slots(self, r, n, hit=(), **kw):
        self.reqs[r.i] = r
        held0, nc0 = self.held.get(r.i, 0), self.num_cached.get(r.i, 0)
        self.depth += 1
        try:
            ok = super().allocate_slots(r, n, hit, **kw)
        finally:
            self.depth -= 1
        if ok:
            h, held1 = len(hit), self.held[r.i]
            if h:
                self._call("take", r.i, held0, h)
            if held1 - held0 - h:
                self._call("alloc", r.i, held0 + h, held1 - held0 - h)
            done = held0 + h if h else nc0
            full = self.num_cached.get(r.i, done)
            if self.caching and full > done:
                self._call("cache", r.i, done, full - done)
        self._agree()
        return ok

    def cache_blocks(self, r, num_tokens):
        if self.depth:
            return super().cache_blocks(r, num_tokens)
        self.reqs[r.i] = r
        done = self.num_cached.get(r.i, 0)
        super().cache_blocks(r, num_tokens)
        full = self.num_cached.get(r.i, done)
        if full > done:
            self._call("cache", r.i, done, full - done)
        self._agree()

    def free_request(self, r):
        held = self.held.get(r.i, 0)
        ret = super().free_request(r)
        if held:
            x, _ = self._call("free", r.i, held)
            if x != ret:
                raise AssertionError(f"free of request {r.i}: the stream returns {x}, the pool {ret}")
        self._agree()
        return ret


def disagreement(p: StreamPool) -> str | None:
    """Where the stream's pool differs from the RAM pool's, or None: the free queue and its ends, every block's fields
    and key, every cached key's copies, every held request's blocks."""
    S, st = p.S, p.stream
    mask = (1 << V3.widths(S).NW) - 1
    q, b = [], st.head
    while b:
        q.append(b)
        b = V3.bsplit(S, st.value(T["B"], b))["next"]
        if len(q) >= S.NB:
            return "free queue (a cycle)"
    if q != p.queue():
        return "free queue"
    if st.tail != p.ram.value(("B", 0)).prev:
        return "queue's tail"
    for b in range(1, S.NB):
        if {k: v for k, v in V3.bsplit(S, st.value(T["B"], b)).items() if k != "stamp"} != p.ram.value(("B", b))._asdict():
            return f"block {b}"
        k = p.ram.value(("K", b))
        if st.value(T["K"], b) != (key128(k) if k else 0):
            return f"key of block {b}"
    keys = {key128(h): h for h in p.keys()}
    seen = {a for (tag, a) in [*p.carried, *st.cells] if tag == T["M"] and st.value(T["M"], a) & mask}
    if seen != set(keys):
        return "cached keys"
    for a, h in keys.items():
        m = st.value(T["M"], a)
        if (m & mask, m >> V3.widths(S).NW & mask) != tuple(p.ram.value(("M", h))):
            return f"copies of key {h}"
    for i, n in p.held.items():
        if [st.value(T["RB"], V3.rk(S, i, k)) for k in range(n)] != [p.ram.value(("RB", i, k)) for k in range(n)]:
            return f"blocks of request {i}"
    return None


class StreamMirror(PR.Mirror):
    """`pool_ram`'s mirror with a `StreamPool` as its RAM pool; each step's count includes its stream slots."""

    def __init__(self, S: SimpleNamespace, window: int, keep: int, check: bool = True, every: int = 1) -> None:
        super().__init__(every=max(every, 1) if every else 10**12, keep=False)
        self.S, self.window_slots, self.keep_windows, self.check_windows, self.every_call = S, window, keep, check, every
        self.seen = 0

    def _pool(self, kv):
        if self.pool is None:
            self.pool, self.kv = StreamPool.like(kv, self.keep), kv
            self.pool.start(self.S, self.window_slots, self.keep_windows, self.check_windows, self.every_call)
        assert kv is self.kv, "one pool per replay"
        return self.pool

    def _tally(self, before: Counter) -> None:
        super()._tally(before)
        self.steps[-1]["slots"] += self.pool.total - self.seen
        self.seen = self.pool.total


def run(eng: RP.Engine, reqs: list, how: dict, window: int, keep: int = 0, check: bool = True,
        every: int = 1) -> StreamMirror:
    """The replay with its pool mirrored on a `StreamPool` (`StreamPool.start`'s ``check`` and ``every``)."""
    S = statics(eng, reqs)
    m = StreamMirror(S, window, keep, check, every)
    with PR.mirrored(m):
        RP.replay(eng, reqs, **how)
    m.pool.close()
    where = PR.disagreement(m.kv, m.pool) or disagreement(m.pool)
    if where:
        raise AssertionError(f"at the end: the pools disagree at the {where}")
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", type=Path, default=FIXTURES)
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--window", type=int, default=4096, help="slots a window holds, about (a call isn't split)")
    ap.add_argument("--vectors", type=Path, help="write each trace's first --keep windows of --vector-window slots that "
                    "evict")
    ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--vector-window", type=int, default=160)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    runs: list[tuple[str, tuple, bool]] = []
    for name in TRACES:
        p = a.traces / name
        if p.exists():
            runs.append((name, load(p), False))
    for s in range(a.seeds):
        rng = random.Random(1000 + s)
        kw = {"NB": rng.choice((24, 48, 96)), "B": rng.choice((4, 16)), "R": rng.choice((4, 8, 16)),
              "T": rng.choice((32, 128, 256)), "asy": rng.random() < 0.5, "wm": rng.choice((0.0, 0.0, 0.05))}
        runs.append((f"synthetic {s} {kw}", synthetic(s, **kw), True))
    report, vectors, total = [], [], Counter()
    for name, (eng, reqs, admit), arrivals in runs:
        if not eng.enable_prefix_caching:
            continue
        how = {"decide": lambda i, k, o, admit=admit: admit[i] <= k} if arrivals else {"admit": admit}
        p = run(eng, reqs, how, a.window).pool
        row = {"run": name, "NB": eng.num_gpu_blocks, "B": eng.block_size, "statics": vars(p.S), **p.counts}
        report.append(row)
        total.update({k: v for k, v in p.counts.items() if not k.startswith("max")})
        total["runs"] += 1
        print(f"{name}: {dict(p.counts)}; agrees after every call, every window's memory check passes")
        if a.vectors and not arrivals:
            v = run(eng, reqs, how, a.vector_window, a.keep).pool
            vectors.append({"run": name, "statics": vars(v.S), "windows": v.windows})
    print("total", dict(total))
    if a.json:
        a.json.write_text(json.dumps({"runs": report, "total": dict(total)}, indent=1))
    if a.vectors:
        a.vectors.write_text(json.dumps({"source": "benchmarks/mempool/stream.py", "window": a.vector_window,
                                         "runs": vectors}, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
