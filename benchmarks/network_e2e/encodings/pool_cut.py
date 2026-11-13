"""The carried pool over recorded traces cut at every window edge (form (b)): at each edge vLLM's block pool, the
replay's (`replay._KV`) and CompileSlice_v6's pool window leave one pool row.

`cut_replay` runs the replay's scheduler on a trace's requests at their admission steps, and before every
``every``-th scheduled step it cuts, as the cut plugin does:
- it executes the step in flight (async scheduling);
- it preempts every running request by vLLM's `_preempt_request` loop: `running.pop()` first, each request's blocks
  freed tail first and the request prepended to the waiting queue, its outputs so far kept;
- it puts the waiting queue in release order.
A cut run drifts from the recorded one, so an idle engine takes the next request at once. A window is the steps between
two edges. Its members are every request in the engine during it, numbered in release order.

`drive` runs a trace's cut run with three pools beside each other:
- `pool_row_trace`'s transcription of vLLM d9105ea80's `BlockPool`, held to `_KV` after every operation;
- `_KV` itself, whose row (`pool_row_trace.kv_row`) is built from its own state;
- CompileSlice_v6's pool window. For each window, CacheStep_v3's stream is opened from the previous edge's row
  (`compile_slice_v6.opened`; the first window opens the empty row) over the members' certified tokens (prompt, then
  outputs). It runs `_KV`'s calls in order (hit, take, alloc, cache, free), and each call's results must be `_KV`'s.
At each edge the window's stream must hold no block and pass its memory check. The row it leaves
(`compile_slice_v6.closed`) must be `_KV`'s, and the exported row (`pool_row.export`) and root must be the same.

    python3 pool_cut.py tests/fixtures/async-prefix-cache.jsonl [--every 16]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, deque
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_pool_vectors as CPV
import pool_row_trace as PRT
import replay as RP

from verity.protocols.certifier.verifier import compile_slice as CS
from verity.protocols.certifier.prover import cache_step_v3 as V3S, compile_slice_v4 as V4, compile_slice_v6 as V6
from verity_vllm.engine import pool_row as PR

#: the seed each window's memory check flips its coins from
COIN_SEED = 0
#: the replay's scheduler, under the name the advice decoder's module gives it where it has it
SCHED = getattr(RP, "_Sched", None) or RP.Scheduler


# --- the cut run -----------------------------------------------------------------------------------------------------


def cut_replay(engine: RP.Engine, requests: Sequence[RP.RequestIO], admit: Sequence[int], every: int, *,
               on_edge: Callable[[Any, int], None] | None = None) -> SimpleNamespace:
    """The replay's scheduler on `requests`, request i added before the first schedule after ``admit[i]`` scheduled
    steps (at once into an idle engine), cut before every ``every``-th scheduled step (the module's docstring). Each
    edge calls ``on_edge(scheduler, w)`` after window w's cut, and the run's end once more for its last window. Gives
    the steps, each window's members and each edge's cut set in preemption order."""
    if every < 1:
        raise ValueError(f"a window of {every} steps")
    s = SCHED(engine, requests)
    n, nxt, edge = len(requests), 0, every
    steps: list[RP.Step] = []
    queue: deque = deque()
    windows: list[list[int]] = [[]]
    cuts: list[list[int]] = []
    limit = 4 * (sum(r.prompt_len + r.output_len for r in requests) + n) + 64

    def add() -> None:
        nonlocal nxt
        s.add(nxt)
        windows[-1].append(nxt)
        nxt += 1

    def top() -> bool:
        while nxt < n and admit[nxt] <= len(steps):
            add()
        if nxt < n and s.quiet() and not queue:
            add()
        if len(steps) > limit:
            raise RuntimeError(f"no progress after {limit} steps")
        return s.has_requests() or bool(queue)

    def schedule() -> tuple[RP.Step, dict[int, bool]]:
        step, samples = s.schedule(None)
        if step.rows:
            steps.append(step)
        return step, samples

    def cut() -> None:
        while queue:
            s.update(*queue.pop())
        order = []
        while s.running:
            r = s.running.pop()
            s._preempt(r)
            order.append(r.i)
        s.waiting, s.skipped = deque(sorted((*s.skipped, *s.waiting), key=lambda r: r.i)), deque()
        cuts.append(order)
        if on_edge is not None:
            on_edge(s, len(windows) - 1)
        windows.append([r.i for r in s.waiting])

    while True:
        if len(steps) >= edge:
            cut()
            edge += every
        if not top():
            break
        if not engine.async_scheduling:
            s.update(*schedule())
            continue
        if s.has_requests():
            st = schedule()
            queue.appendleft(st)
            if len(queue) < 2 and (st[0].rows or s.has_requests()):
                continue
        elif not queue:
            continue
        s.update(*queue.pop())
    if on_edge is not None:
        on_edge(s, len(windows) - 1)
    return SimpleNamespace(steps=steps, windows=windows, cuts=cuts)


# --- CompileSlice_v6's pool window -----------------------------------------------------------------------------------


def statics(inp: CPV.Input, windows: Sequence[Sequence[int]]) -> dict[str, int]:
    """`pool_row_trace.statics`' class for the trace, its arrival slots K the most members a window has, CU the most
    blocks a request's tokens fill, holders to the most requests the engine runs and stream and M run widths no window
    passes."""
    e = inp.engine
    cu = max(-(-len(r.tokens) // e.block_size) for r in inp.requests)
    st = {**PRT.statics(inp), "K": max(map(len, windows)), "CU": cu, "R": e.max_num_seqs, "NS": 1 << 32, "NM": 1 << 32}
    V4.shape(SimpleNamespace(**st))
    return st


class Window:
    """CompileSlice_v6's pool window over the cut run, one window at a time: its stream, opened from the row the last
    one left over its members' tokens, and the calls `_KV` makes, each run on the stream under its member's number."""

    def __init__(self, S: dict[str, int], tokens: Sequence[Sequence[int]], windows: Sequence[Sequence[int]]) -> None:
        self.S, self.ns, self.tokens, self.windows = S, SimpleNamespace(**S), tokens, windows
        self.delta = V6.COINS[2]
        self.row = V6.empty_row(self.ns)
        self.counts: Counter = Counter()
        self.roots: list[str] = []
        self._open(0)

    def _open(self, w: int) -> None:
        members = self.windows[w]
        self.w, self.num = w, {i: k for k, i in enumerate(members)}
        self.sl = CS.Slice((), (), 0, tuple(tuple(self.tokens[i]) for i in members))
        self.stream = V6.opened(self.ns, self.row, self.sl, self.delta)

    def call(self, kind: str, i: int, a: int, b: int = 0, want: tuple[int, int] | None = None) -> None:
        if i not in self.num:
            raise AssertionError(f"window {self.w}: request {i} calls the pool but isn't one of its members")
        got = self.stream.call(V3S.Call(kind, self.num[i], a, b))
        if want is not None and got != want:
            raise AssertionError(f"window {self.w}: request {i}'s {kind} gives {got} on the stream, {want} on _KV")
        self.counts[kind] += 1

    def edge(self, kv: Any, drv: PRT.Driver) -> None:
        """Closes the window at its edge against `_KV` and the transcribed pool, and opens the next."""
        st, w = self.stream, self.w
        rng = random.Random(COIN_SEED + w)
        alpha, gamma = rng.getrandbits(128), rng.getrandbits(128)
        R, W = st.fingerprints(alpha, gamma)
        if R != W or any(x.t >= x.now for x in st.slots):
            raise AssertionError(f"window {w}'s stream fails its memory check")
        row, quiet = V6.closed(self.ns, st, self.row, self.sl, self.delta)
        if not quiet:
            raise AssertionError(f"window {w}'s stream ends holding a block")
        want = PRT.kv_row(kv, drv.key)
        if row != want:
            raise AssertionError(f"window {w}: CompileSlice_v6's pool window leaves another row than _KV's")
        got = PR.export(drv.T, self.S["B"])
        if got["root"] != V6.root(self.ns, want).hex():
            raise AssertionError(f"window {w}: the exported root isn't the row CompileSlice_v6 leaves")
        self.roots.append(got["root"])
        c = self.counts
        c["closed windows"] += 1
        c["max window slots"] = max(c["max window slots"], len(st.slots))
        c["max window calls"] = max(c["max window calls"], len(st.records))
        c["max M run"] = max(c["max M run"], len(V6.mrun(self.ns, st, V6.heads(self.ns, self.row, self.delta))))
        c["max carried keys"] = max(c["max carried keys"], sum(1 for k in row.key if k != V6.NOKEY))
        c["max later copies carried"] = max(c["max later copies carried"], sum(1 for p in row.cprev if p))
        c["edges carrying keys"] += any(k != V6.NOKEY for k in row.key)
        self.row = row
        if w + 1 < len(self.windows):
            self._open(w + 1)


@contextmanager
def streamed(win: Window):
    """`_KV`'s calls (not a fork's, not a dry run), each run on ``win``'s stream too, with the results `_KV`'s."""
    K = PRT.KV
    names = ("hit", "allocate_slots", "cache_blocks", "free_request")
    orig = {n: getattr(K, n) for n in names}
    depth = 0

    def hit(self, r):
        out = orig["hit"](self, r)
        if self.mine is None and self.caching:
            lim = min(len(r.block_hashes), (r.num_tokens - 1) // self.B)
            if lim > 0:
                win.call("hit", r.i, lim, want=(len(out), sum(1 for b in out if self.ref[b] == 0)))
        return out

    def allocate_slots(self, r, n, hit=(), *, dry=False, **kw):
        nonlocal depth
        if dry or self.mine is not None:
            return orig["allocate_slots"](self, r, n, hit, dry=dry, **kw)
        held0, nc0 = len(self.req_blocks.get(r.i, ())), self.num_cached.get(r.i, 0)
        depth += 1
        try:
            ok = orig["allocate_slots"](self, r, n, hit, **kw)
        finally:
            depth -= 1
        if ok:
            h, held1 = len(hit), len(self.req_blocks[r.i])
            if h:
                win.call("take", r.i, held0, h)
            if held1 - held0 - h:
                win.call("alloc", r.i, held0 + h, held1 - held0 - h)
            done = held0 + h if h else nc0
            full = self.num_cached.get(r.i, done)
            if self.caching and full > done:
                win.call("cache", r.i, done, full - done)
        return ok

    def cache_blocks(self, r, num_tokens):
        if depth or self.mine is not None:
            return orig["cache_blocks"](self, r, num_tokens)
        done = self.num_cached.get(r.i, 0)
        orig["cache_blocks"](self, r, num_tokens)
        full = self.num_cached.get(r.i, done)
        if full > done:
            win.call("cache", r.i, done, full - done)

    def free_request(self, r):
        blocks = list(self.req_blocks.get(r.i, ()))
        orig["free_request"](self, r)
        if self.mine is None and blocks:
            win.call("free", r.i, len(blocks), want=(sum(1 for b in blocks if self.ref[b] == 0), 0))

    taps = {"hit": hit, "allocate_slots": allocate_slots, "cache_blocks": cache_blocks, "free_request": free_request}
    for n in names:
        setattr(K, n, taps[n])
    try:
        yield
    finally:
        for n in names:
            setattr(K, n, orig[n])


# --- the drive -------------------------------------------------------------------------------------------------------


def drive(inp: CPV.Input, *, every: int) -> dict[str, Any]:
    """The trace ``inp`` (its engine, requests and admission steps) cut every ``every`` scheduled steps, with the three
    pools held to each other (the module's docstring); the counts and each edge's root. Raises AssertionError where
    they disagree."""
    if not inp.engine.enable_prefix_caching:
        raise ValueError("the carried pool needs prefix caching on")
    plan = cut_replay(inp.engine, inp.requests, inp.before, every)
    S = statics(inp, plan.windows)
    drv = PRT.Driver(S, 1)
    win = Window(S, [r.tokens for r in inp.requests], plan.windows)
    if PR.row(PR.snapshot(drv.T, S["B"])) != win.row:
        raise AssertionError("a fresh pool isn't CompileSlice_v6's empty row")
    with PRT.keyed(drv), PRT.tapped(drv), streamed(win):
        run = cut_replay(inp.engine, inp.requests, inp.before, every, on_edge=lambda s, w: win.edge(s.kv, drv))
    if run.windows != plan.windows or run.cuts != plan.cuts:
        raise AssertionError("the cut run isn't the one its class was sized on")
    return {**dict(drv.counts), **dict(win.counts), "steps": len(run.steps), "windows": len(run.windows),
            "cut": sum(map(len, run.cuts)), "K": S["K"], "keys": len(drv.key), "roots": win.roots}


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--every", type=int, nargs="+", default=[16], help="window lengths in scheduled steps")
    a = ap.parse_args(argv)
    for t in a.traces:
        for every in a.every:
            got = drive(CPV.replay_input(t), every=every)
            print(json.dumps({"trace": t.name, "every": every, **{k: v for k, v in got.items() if k != "roots"}}))


if __name__ == "__main__":
    main()
