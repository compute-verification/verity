"""One vLLM v1 scheduler step as a Boolean Definition: the compile unit's core, between advice validity and the Programs.

    ReplayStep_v1{R, W, A, M, T, L, B, H, CH, WM, FULL, PC, NB, IW, AS, EW} (state, arrivals, bits, prefix[, row_pfx])
        -> (ok, taken, used[, looked], rows, state')

The precommitted compiler (committed Programs = compile(advice) under the advice's hiding root, total, refusing every
ill-formed advice up to the cap ℓ) need not read the schedule from the advice: vLLM's scheduler is
replayed step for step from the certified I/O and a residual of a few bits per certifier window
(`benchmarks/network_e2e/encodings/replay.py`). This Definition is one step of that replay, `reference` its
specification in Python, and the replay's own steps are its difftest (`benchmarks/network_e2e/encodings/replay_vectors.py`).

**What it models.** vLLM v1's `Scheduler.schedule` and `update_from_output` (the pinned 0.28.1rc1.dev472) under
synchronous or async scheduling, FCFS, one full-attention KV group, as `replay._Sched` runs them with `decide`:
- *Arrivals.* Requests reach the engine in their certified order. When the engine is quiet (nothing running or
  waiting) the next arrival enters the waiting queue unasked. Otherwise an arrival enters only where the waiting loop
  reaches the queue's end with room for it (an *open* step: budget left, fewer than M running, and the request would
  be scheduled there), and one residual bit says whether it had arrived by then; a 0 ends the step's admissions.
- *The running loop.* Each running request in order gets n = min(P + out − computed, H if set, the budget left,
  L − computed − 1) tokens; one whose new blocks don't fit preempts the last running request (to the front of the
  waiting queue, computed reset) until they do, or preempts itself and ends the loop.
- *The waiting loop* (only in a step that preempted nothing): each waiting request, then each admitted arrival, starts
  at its prefix-cache hit (h blocks) with n = P + out − h·B tokens (clamped by H; without chunked prefill one that
  doesn't fit the budget ends the loop), if its full sequence (FULL) and its first n tokens fit the free blocks after
  the watermark (WM, when something runs); the first that doesn't ends the loop.
- *Close and update.* A row samples iff it reaches the request's known tokens; a request stops at its certified
  output length O and frees its blocks. A row's kind is decode iff n = 1 and start ≥ P (I1 v0's rule).
- *Async scheduling* (AS = 1; `AsyncScheduler`, whose update of a step runs after the next schedule). One step here is
  one `schedule()` and the update of the step before it, so a running request's out counts the output in flight. A
  request whose last output was sampled the step before is `fin`: at this schedule it still runs, holds its blocks and
  counts toward M and the watermark, and gets its lag row (n as above, sampling nothing) unless it is `cap` (O is its
  max_tokens: the scheduler skips a request whose tokens in flight reach it). At this step's update it leaves and
  frees its blocks. A request reaching O sets `fin` and keeps its slot. A `fin` victim frees its blocks at preemption
  and isn't requeued, but the step still preempted, so its waiting loop doesn't run. The certified O forces every lag
  row, so async costs no decision.
- *Walk entries* (EW ≥ 1, `CacheStep_v1`'s entry width): each running slot's
  `pfx` moves with its request through the compaction and placement. A slot with a row takes that row's entry after
  its `CacheStep_v1` calls (a running row's one, an admitted row's second), a kept slot without one keeps its own, and
  a preempted or stopped request drops its own, since it resumes as an admitting row from the root. The step also
  says which candidates' lookups it read (`looked`): each waiting slot its loop reaches, the one whose allocation ends
  the loop included, and each arrival it checks for an ask. Past the pool bound advice v1 sends one shortfall for each.

**Statics** (all public): the size class R running slots, W waiting slots and A arrivals looked ahead; the engine's
max_num_seqs M, max_num_batched_tokens T, max_model_len L, block_size B (a power of 2), long_prefill_token_threshold
H (0 for none), enable_chunked_prefill CH, watermark blocks WM, scheduler_reserve_full_isl FULL, enable_prefix_caching
PC and num_gpu_blocks NB; IW, a request id's width; async_scheduling AS; EW, the walk entries' width (0 for none).
Widths: LW = bitlen(L), TW = bitlen(T), BW = bitlen(NB),
HW = bitlen(⌈L / B⌉), AC = bitlen(A). Every field is little-endian.

**Inputs.**
- the committed state: R running slots (valid, id, P, O, out, computed), W waiting slots (valid, id, P, O, out; a
  waiting request has computed nothing) and the free block count. A request holds ⌈computed / B⌉ blocks, so the
  state carries no block counts per request. The scan starts from the empty engine with NB − 1 free blocks (block 0
  is vLLM's null block), so the state's own invariants (valid slots a prefix, computed < P + out ≤ L, out < O) hold
  by induction and the step doesn't recheck them. With EW each running slot's walk entry `run_pfx` follows `run_comp`.
  With AS the state ends with its async group: each running slot's `fin` and `cap`, then each waiting slot's `cap`;
- the next A certified arrivals (valid, id, P, O, and with AS cap), in order from the scan's cursor;
- A residual bits: the a-th open arrival's decision is bit a;
- the prefix layer's counts, read only with PC: per candidate (W waiting slots, then A arrivals) its hit h and how
  many of those blocks are evictable (cached, unreferenced: taking them shrinks the free count), and per running slot
  how many blocks it returns when it is preempted or stops (its blocks whose last reference it drops). Without PC
  there are no hits and a request returns all it holds, and these inputs are unread;
- with EW, `row_pfx`: each row's walk entry after its `CacheStep_v1` calls, by the row's slot.

**Outputs.** ok; taken, the arrivals consumed (the scan's cursor moves by it); used, the residual bits read; with EW,
`looked`, one bit per candidate (W waiting slots, then A arrivals); R rows (valid, request, kind, start, n), in the
scheduler's order (running first, then the admitted, in order) at the slots of the running list before the step's
leaving requests leave it; and the next state.

**ok = 0 refuses** what this size class can't express or the inputs make meaningless:
- an admission into a full running list (R slots, fewer than M), a waiting list past W after preemption, or a step
  still admitting after its A arrivals (the next might have arrived);
- an arrival taken with P = 0, O = 0 or P + O > L;
- with PC, a hit or return the scheduler couldn't see: more evictable blocks than hit blocks, a hit reaching the
  request's last known token (h·B ≥ P + out; vLLM keeps one to compute), or more blocks returned than held.

**Not modeled.** The prefix-hash state (which blocks hold which tokens, the free queue's order, eviction):
the derived encoding's canonical source and hit rules carry it and supply h, the evictable count and the returns, which
this circuit bounds but doesn't derive: `compile_slice_v3` derives them, in vLLM's order on CacheStep_v2's pool, through
`_build`'s ``derive`` hooks. Also priority scheduling, speculative decoding, LoRA, structured output, encoder inputs
and KV connectors.

**Composition** (the compile unit, all inside one ZK proof under the advice's hiding root):
1. the committed advice is the decision bits themselves, one per open-step ask in the scan's order, read directly;
2. a scan of this step over the session's steps, from the empty engine, the arrivals' cursor moving by `taken` and
   the decisions' by `used`; the certified request records (P, O) are the arrivals, in the warden's release order;
3. the charge: advice v1's arithmetic code run forward over the same decisions (`arith_encode`), each window's coded
   length at most its declared charge, so no decoder runs in the circuit;
4. `compile_steps` (`integrations/vllm/verity_vllm/program/predict/steps.py`): each row (request, start, n) becomes a `RowRef`
   into its request's Program, the rows being exactly I1 v0's `steps`, which v0 carried in the advice.

**The circuit**, traced whole (`_build`): the running loop over static slots, each slot's n read off the budget the
slots before it leave, and a slot kept iff the free blocks plus the returns of the slots after it cover the needs up to
it (the victims are a suffix and no need or return is negative, so that is the scheduler's pops); a scan of the
candidates (W waiting slots, then A arrivals); the admitted placed after the running list by two right shifts; close
and update per slot; the leaving (stopped, or with AS the `fin`) compacted out by a log-depth shifter (each slot moves
left by the leaving before it, low bit first, which never collides); the waiting list rebuilt by shifting left by the
admitted and right by the preempted (with AS, the victims not `fin`, compacted first). `size` counts its ANDs by
stage.

**Cost** (`size`, on the smallkv engine: T = 2048, NB = 512, chunked prefill and prefix caching): 12,235 ANDs at
(R, W, A) = (4, 8, 4), 92,146 at (16, 64, 16), 123,611 at (64, 4, 8), 250,544 at (128, 4, 8) and 521,358 at
(256, 4, 8), the class that covers max_num_seqs = 256. Every stage is linear in R up to a shifter's log factor: at
(256, 4, 8) the compaction is 259k ANDs (about 124·R·bitlen(R)), the placing 82k, the running loop 75k (292·R), the
requeue 55k (about 131·W·bitlen(R) + 215·R) and the update 43k (168·R); the candidates' scan is about 210·W + 245·A.
The replay's own waiting list holds only the preempted and the forced arrival (at most 5 on the corpus), since the
arrivals not yet scheduled stay in the certified stream, so W stays small. Async adds 132 ANDs at (4, 8, 4), 3,108
at (16, 64, 16), 15,640 at (64, 4, 8), 33,326 at (128, 4, 8) and 70,109 at (256, 4, 8) (1.1% to 13.4%), most of it the
requeue's compaction of the victims not `fin` (59k of the 70k at R = 256). Walk entries of 9 bits add 207, 1,431,
8,055 and 41,463 at (4, 8, 4), (16, 64, 16), (64, 4, 8) and (256, 4, 8); `looked` is the candidates' scan's own bits
and adds none.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import SimpleNamespace
from typing import Sequence

from verity.ml.circuits.boolean.forms import ONE, ZERO, Circuit, add, and_reduce, const_bits, mux, or_reduce, sub
from verity.ml.circuits.evaluation.bits import evaluate_bits
from verity.ml.circuits.defs import bind
from verity.ml.circuits.types import Array, Tuple
from verity.core.protocols.certifier.head_check import BIT, _part, ands

STATICS = ("R", "W", "A", "M", "T", "L", "B", "H", "CH", "WM", "FULL", "PC", "NB", "IW", "AS", "EW")


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _bitlen(v: int) -> int:
    return max(int(v).bit_length(), 1)


def widths(S) -> SimpleNamespace:
    if S.B < 1 or S.B & (S.B - 1):
        raise ValueError(f"block size {S.B}: a power of 2")
    if (min(S.R, S.W, S.A, S.M, S.T, S.L, S.NB, S.IW) < 1 or min(S.H, S.WM, S.EW) < 0
            or {S.CH, S.FULL, S.PC, S.AS} - {0, 1}):
        raise ValueError(f"statics {dict(S.__dict__)}: sizes ≥ 1, H, WM and EW ≥ 0, CH, FULL, PC and AS bits")
    return SimpleNamespace(LW=_bitlen(S.L), TW=_bitlen(S.T), BW=_bitlen(S.NB), HW=_bitlen(cdiv(S.L, S.B)),
                           AC=_bitlen(S.A), RC=_bitlen(S.R), WC=_bitlen(S.W), LB=S.B.bit_length() - 1)


def _state_fields(S) -> list[tuple[str, int, int]]:
    """The committed state, as the step reads and writes it: the sync step's part, with EW's `run_pfx` after
    `run_comp`, then async's group (AS)."""
    Z = widths(S)
    R, W = S.R, S.W
    return ([("run_valid", R, 1), ("run_id", R, S.IW), ("run_P", R, Z.LW), ("run_O", R, Z.LW), ("run_out", R, Z.LW),
             ("run_comp", R, Z.LW)] + ([("run_pfx", R, S.EW)] if S.EW else [])
            + [("wait_valid", W, 1), ("wait_id", W, S.IW), ("wait_P", W, Z.LW), ("wait_O", W, Z.LW), ("wait_out", W, Z.LW),
               ("free", 1, Z.BW)]
            + ([("run_fin", R, 1), ("run_cap", R, 1), ("wait_cap", W, 1)] if S.AS else []))


def _in_fields(S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    R, W, A = S.R, S.W, S.A
    return (_state_fields(S)
            + [("arr_valid", A, 1), ("arr_id", A, S.IW), ("arr_P", A, Z.LW), ("arr_O", A, Z.LW)]
            + ([("arr_cap", A, 1)] if S.AS else [])
            + [("bits", A, 1), ("hit", W + A, Z.HW), ("evict", W + A, Z.HW), ("ret", R, Z.HW)]
            + ([("row_pfx", R, S.EW)] if S.EW else []))


def _out_fields(S) -> list[tuple[str, int, int]]:
    Z = widths(S)
    R = S.R
    return ([("ok", 1, 1), ("taken", 1, Z.AC), ("used", 1, Z.AC)] + ([("looked", S.W + S.A, 1)] if S.EW else [])
            + [("row_valid", R, 1), ("row_id", R, S.IW), ("row_kind", R, 1), ("row_start", R, Z.LW), ("row_n", R, Z.TW)]
            + _state_fields(S))


def _type(fields) -> Tuple:
    return Tuple(*[Array(n, Array(w, BIT)) for _, n, w in fields])


# --- the specification ---------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Req:
    """A request in the engine: its id, certified prompt and output lengths, outputs so far and tokens computed. With
    AS, `cap` (O is its max_tokens) and, running, `fin` (its last output was sampled the step before); with EW, a
    running request's walk entry `pfx`."""

    id: int
    P: int
    O: int
    out: int = 0
    comp: int = 0
    cap: int = 0
    fin: int = 0
    pfx: int = 0


@dataclass(frozen=True)
class State:
    running: tuple[Req, ...]
    waiting: tuple[Req, ...]
    free: int


@dataclass(frozen=True)
class Inputs:
    arrivals: tuple[Req, ...]               # the next certified arrivals (at most A), in order
    bits: tuple[int, ...] = ()              # the residual's decisions, the a-th open arrival's at a
    hit: tuple[int, ...] = ()               # per candidate (W waiting slots, then A arrivals): hit blocks
    evict: tuple[int, ...] = ()             # per candidate: of those, the evictable
    ret: tuple[int, ...] = ()               # per running slot: blocks returned when it leaves
    row_pfx: tuple[int, ...] = ()           # per slot (the rows' slots): the walk entry after its row's CacheStep call


@dataclass(frozen=True)
class Out:
    ok: bool
    rows: tuple[tuple[int, int, int, int, int], ...] = ()   # (slot, request, kind, start, n)
    state: State | None = None
    taken: int = 0
    used: int = 0
    looked: tuple[int, ...] = ()            # with EW, per candidate: whether the step read its lookup


def empty(S) -> State:
    return State((), (), S.NB - 1)


def _at(xs: Sequence[int], k: int) -> int:
    return xs[k] if k < len(xs) else 0


def reference(S, st: State, inp: Inputs) -> Out:
    """One step as `replay._Sched` schedules and updates it (FCFS; with AS, one schedule and the update of the step
    before it), on the size class `S`; `Out(False)` refuses."""
    if len(st.running) > S.R or len(st.waiting) > S.W or len(inp.arrivals) > S.A:
        raise ValueError("the state or the arrivals exceed the size class")
    AS, EW = S.AS, S.EW

    def norm(r: Req, running: bool) -> dict:
        return {**r.__dict__, "cap": r.cap if AS else 0, "fin": r.fin if AS and running else 0,
                "pfx": r.pfx if EW and running else 0}

    run = [norm(r, True) for r in st.running]
    wait = [norm(r, False) for r in st.waiting]
    arr = list(inp.arrivals)
    free, taken, used, budget = st.free, 0, 0, S.T

    def held(r: dict) -> int:
        return cdiv(r["comp"], S.B)

    def returned(slot: int, r: dict) -> int | None:
        if not S.PC:
            return held(r)
        x = _at(inp.ret, slot)
        return x if x <= held(r) else None

    def arrival_ok(a: Req) -> bool:
        return a.P >= 1 and a.O >= 1 and a.P + a.O <= S.L

    def as_req(a: Req) -> dict:
        return {"id": a.id, "P": a.P, "O": a.O, "out": 0, "comp": 0, "cap": a.cap if AS else 0, "fin": 0, "pfx": 0}

    shift = 0
    if not run and not wait and arr:
        a = arr[0]
        if not arrival_ok(a):
            return Out(False)
        wait, shift, taken = [as_req(a)], 1, 1
    sched: dict[int, int] = {}
    victims: list[dict] = []
    preempted = False
    i = 0
    while i < len(run) and budget > 0:
        r = run[i]
        n = r["P"] + r["out"] - r["comp"]
        if 0 < S.H < n:
            n = S.H
        n = min(n, budget, S.L - r["comp"] - 1)
        if r["fin"] and r["cap"]:
            n = 0
        if n <= 0:
            i += 1
            continue
        need = cdiv(r["comp"] + n, S.B) - held(r)
        while need > free:
            j = len(run) - 1
            v = run.pop()
            back = returned(j, v)
            if back is None:
                return Out(False)
            free += back
            preempted = True
            if not v["fin"]:
                victims.insert(0, {**v, "comp": 0, "pfx": 0})
            if j == i:
                break
        if i >= len(run):
            break
        free -= need
        sched[i] = n
        budget -= n
        i += 1

    k = 0
    looked = [0] * (S.W + S.A)
    if not preempted:
        def admit(r: dict, c: int, ask: bool) -> int:
            """0: the loop ends here; 1: r is admitted; 2: refused (ok = 0). Reads bit `ask`'s index when asked."""
            nonlocal free, budget, used
            if budget <= 0 or len(run) >= S.M:
                return 0
            looked[c] = 1
            h, ev = (_at(inp.hit, c), _at(inp.evict, c)) if S.PC else (0, 0)
            ntok = r["P"] + r["out"]
            if S.PC and (ev > h or h * S.B >= ntok):
                return 2
            hb = h * S.B
            n = ntok - hb
            if 0 < S.H < n:
                n = S.H
            if not S.CH and n > budget:
                return 0
            n = min(n, budget)
            wm = S.WM if run else 0
            if S.FULL and max(cdiv(min(ntok, S.L), S.B) - h, 0) + ev + wm > free:
                return 0
            new = max(cdiv(min(hb + n, S.L), S.B) - h, 0)
            if new + ev + wm > free:
                return 0
            if ask is not None:
                used += 1
                if not _at(inp.bits, ask):
                    return 0
            if len(run) >= S.R:
                return 2
            free -= new + ev
            budget -= n
            run.append({**r, "comp": hb})
            sched[len(run) - 1] = n
            return 1

        going = True
        for w in range(len(wait)):
            got = admit(wait[w], w, None)
            if got == 2:
                return Out(False)
            if not got:
                going = False
                break
            k += 1
        if going:
            for a in range(S.A):
                if a + shift >= len(arr):
                    going = False
                    break
                got = admit(as_req(arr[a + shift]), S.W + a, a)
                if got == 2 or (got == 1 and not arrival_ok(arr[a + shift])):
                    return Out(False)
                if not got:
                    going = False
                    break
                taken += 1
            if going:
                return Out(False)

    rows = []
    keep = []
    for p, r in enumerate(run):
        lag = r["fin"]
        stop = False
        if p in sched:
            n = sched[p]
            rows.append((p, r["id"], int(n == 1 and r["comp"] >= r["P"]), r["comp"], n))
            r["comp"] += n
            if EW:
                r["pfx"] = _at(inp.row_pfx, p)
            if not lag and r["comp"] >= r["P"] + r["out"]:
                r["out"] += 1
                stop = r["out"] == r["O"]
        if lag or (stop and not AS):
            back = returned(p, r)
            if back is None:
                return Out(False)
            free += back
            continue
        r["fin"] = int(stop)
        keep.append(r)
    waiting = victims + wait[k:]
    if len(waiting) > S.W:
        return Out(False)
    return Out(True, tuple(rows), State(tuple(Req(**r) for r in keep), tuple(Req(**{**r, "comp": 0}) for r in waiting), free),
               taken, used, tuple(looked) if EW else ())


# --- the circuit -----------------------------------------------------------------------------------------------------------


def _pad(x, w):
    return (list(x) + [ZERO] * w)[:w]


def _eq(C, a, b):
    w = max(len(a), len(b))
    return and_reduce(C, [(x ^ y).inv() for x, y in zip(_pad(a, w), _pad(b, w))])


def _lt(C, a, b):
    w = max(len(a), len(b))
    return sub(C, _pad(a, w), _pad(b, w), w)[1]


def _min(C, a, b, w):
    a, b = _pad(a, w), _pad(b, w)
    return mux(C, _lt(C, a, b), a, b)


def _add(C, a, b, w):
    return add(C, _pad(a, w), _pad(b, w), width=w)


def _sub(C, a, b, w):
    return sub(C, _pad(a, w), _pad(b, w), w)[0]


def _monus(C, a, b, w):
    """max(a − b, 0)."""
    d, borrow = sub(C, _pad(a, w), _pad(b, w), w)
    return [C.AND(borrow.inv(), x) for x in d]


def _sel(C, s, x):
    return [C.AND(s, b) for b in x]


def _cdiv(C, x, lb, w):
    """⌈x / 2^lb⌉ in w bits."""
    if lb == 0:
        return _pad(x, w)
    y = add(C, _pad(x, len(x) + 1), const_bits((1 << lb) - 1, len(x) + 1), width=len(x) + 1)
    return _pad(y[lb:], w)


def _popcount(C, bits, w):
    out = [ZERO] * w
    for b in bits:
        out = add(C, out, [b], width=w)
    return out


def _clamp(C, n, H, w):
    """n, or H when 0 < H < n."""
    if not H:
        return n
    h = const_bits(H, w)
    return mux(C, _lt(C, h, n), h, n)


def _shift_right(C, slots, amt, n_out):
    """Slot p of the result is slot p − amt of `slots` (zero below amt), for n_out slots; every slot a list of bits."""
    w = len(slots[0])
    cur = [list(s) for s in slots[:n_out]] + [[ZERO] * w for _ in range(n_out - len(slots))]
    for k, s in enumerate(amt):
        sh = 1 << k
        if sh >= n_out:
            cur = [_sel(C, s.inv(), x) for x in cur]
            continue
        cur = [mux(C, s, cur[p - sh], cur[p]) if p >= sh else _sel(C, s.inv(), cur[p]) for p in range(n_out)]
    return cur


def _shift_left(C, slots, amt):
    """Slot p of the result is slot p + amt of `slots` (zero past the end)."""
    n, w = len(slots), len(slots[0])
    cur = [list(s) for s in slots]
    for k, s in enumerate(amt):
        sh = 1 << k
        if sh >= n:
            cur = [_sel(C, s.inv(), x) for x in cur]
            continue
        cur = [mux(C, s, cur[p + sh], cur[p]) if p + sh < n else _sel(C, s.inv(), cur[p]) for p in range(n)]
    return cur


def _compact(C, slots, keep, cw):
    """The kept slots, in order, at the front (the rest zero): each moves left by the dropped before it, by the bits of
    that count low first; two kept slots never meet (i < j kept: j − i exceeds the dropped between them)."""
    n = len(slots)
    z, cnt = [], [ZERO] * cw
    for p in range(n):
        z.append(cnt)
        cnt = add(C, cnt, [keep[p].inv()], width=cw)
    cur = [_sel(C, keep[p], list(slots[p]) + z[p]) for p in range(n)]
    w = len(slots[0])
    for k in range(cw):
        sh = 1 << k
        if sh >= n:
            break
        nxt = []
        for p in range(n):
            stay = _sel(C, cur[p][w + k].inv(), cur[p])
            if p + sh < n:
                come = _sel(C, cur[p + sh][w + k], cur[p + sh])
                stay = [x ^ y for x, y in zip(stay, come)]
            nxt.append(stay)
        cur = nxt
    return [c[:w] for c in cur]


def _split(fields, elements) -> dict:
    out, at = {}, 0
    for name, n, _w in fields:
        out[name] = [list(e) for e in elements[at:at + n]]
        at += n
    return out


def _build(C, S, *elements, tap: dict | None = None, derive=None):
    """The step traced on `C`. A caller tracing it inline may pass `tap`, a dict the trace fills with wires it has
    already made, so a composition binds them without new gates: ``run``, ``wait`` and ``arr`` (the slots' records,
    after a quiet step's arrival moves into waiting slot 0), ``forced``, ``free`` (the state's); per running slot
    ``needs``, ``backs``, ``avail`` (free plus the returns of the slots after it), ``upto`` (the needs up to it),
    ``kept``, ``popped`` and ``run_n``; ``cnt0``; per candidate ``cands`` (reach, admit, new, ntok, hb, n, and ``at``,
    the running count before it: its placed slot when admitted), filled as the waiting loop goes; the placed slots'
    records ``slots``; and per placed slot ``update`` (on, comp after the row, sampled before AS's fin mask, leave,
    held, keep). With PC, a caller that computes the prefix layer's counts rather than reading them passes `derive`,
    whose methods the trace calls where it reads them, so their wires can depend on the step's own:
    ``returns(tap)`` once before the running loop (each running slot's returns were it freed after the slots past
    it), ``running(tap)`` after it, ``hit(c, rec, ntok, tap)`` at candidate c's turn (its hit and evictable count),
    ``candidates(tap)`` after the waiting loop, and ``leave(p, tap)`` per placed slot once ``update[p]`` has its
    leave and held (its returns). The inputs ``hit``, ``evict`` and ``ret`` are then unread."""
    tap = {} if tap is None else tap
    derive = derive if S.PC else None
    Z = widths(S)
    LW, TW, BW, HW, RC, AC, LB = Z.LW, Z.TW, Z.BW, Z.HW, Z.RC, Z.AC, Z.LB
    NW = max(LW, TW) + 1
    R, W, A = S.R, S.W, S.A
    AS, EW = S.AS, S.EW
    I = _split(_in_fields(S), elements)
    bad = []
    phase = getattr(C, "phase", None) is not None

    def stage(name):
        if phase:
            C.phase = name

    def rec(kind, p):
        return {"valid": I[f"{kind}_valid"][p][0], "id": I[f"{kind}_id"][p], "P": I[f"{kind}_P"][p], "O": I[f"{kind}_O"][p],
                "out": I[f"{kind}_out"][p] if kind != "arr" else [ZERO] * LW,
                "comp": I[f"{kind}_comp"][p] if kind == "run" else [ZERO] * LW,
                "cap": I[f"{kind}_cap"][p][0] if AS else ZERO,
                "fin": I["run_fin"][p][0] if AS and kind == "run" else ZERO}

    def arrival_bad(r):
        po = _add(C, r["P"], r["O"], LW + 1)
        return or_reduce(C, [or_reduce(C, r["P"]).inv(), or_reduce(C, r["O"]).inv(), _lt(C, const_bits(S.L, LW + 1), po)])

    stage("arrivals")
    run = [rec("run", p) for p in range(R)]
    wait = [rec("wait", w) for w in range(W)]
    arr = [rec("arr", a) for a in range(A)]
    quiet = C.AND(or_reduce(C, [r["valid"] for r in run]).inv(), wait[0]["valid"].inv())
    forced = C.AND(quiet, arr[0]["valid"])
    bad.append(C.AND(forced, arrival_bad(arr[0])))
    keys = ("valid", "id", "P", "O", "out", "comp") + (("cap", "fin") if AS else ())
    one_bit = {"valid", "cap", "fin"}
    wait[0] = {k: (mux(C, forced, [arr[0][k]], [wait[0][k]])[0] if k in one_bit else mux(C, forced, arr[0][k], wait[0][k]))
               for k in keys}
    zero = {k: (ZERO if k in one_bit else [ZERO] * len(arr[0][k])) for k in keys}
    arr = [{k: (mux(C, forced, [nxt[k]], [cur[k]])[0] if k in one_bit else mux(C, forced, nxt[k], cur[k])) for k in keys}
           for cur, nxt in zip(arr, arr[1:] + [zero])]
    free = I["free"][0]
    tap.update(run=run, wait=wait, arr=arr, forced=forced, free=free)

    def ret_of(p, held):
        if not S.PC:
            return _pad(held, HW)
        return I["ret"][p]

    # the running loop. The victims are a suffix and no need or return is negative, so the slots kept are those i with
    # free + (the returns of the slots after i) ≥ (the needs of slots 0..i), each slot's n taken from the budget the
    # slots before it leave as if all were kept: one comparison a slot instead of a pass over the victims.
    stage("running")
    SW = max(BW, HW + 1 + RC) + 1
    budget = const_bits(S.T, TW)
    ns, needs = [], []
    for i in range(R):
        r = run[i]
        ntok = _add(C, r["P"], r["out"], LW)
        n = _clamp(C, _sub(C, ntok, r["comp"], NW), S.H, NW)
        n = _min(C, n, budget, NW)
        live = C.AND(r["valid"], C.AND(r["fin"], r["cap"]).inv()) if AS else r["valid"]
        n = _sel(C, live, _min(C, n, _sub(C, const_bits(S.L - 1, LW), r["comp"], LW), NW))
        held = _cdiv(C, r["comp"], LB, HW + 1)
        needs.append(_sub(C, _cdiv(C, _add(C, r["comp"], n, LW + 1), LB, HW + 1), held, HW + 1))
        budget = _sub(C, budget, n, TW)
        ns.append(_pad(n, TW))
    if derive is not None:
        rets = derive.returns(tap)
        backs = [_sel(C, run[j]["valid"], rets[j]) for j in range(R)]
    else:
        backs = [_sel(C, run[j]["valid"], ret_of(j, _cdiv(C, run[j]["comp"], LB, HW))) for j in range(R)]
    after = [[ZERO] * SW]
    for j in range(R - 1, 0, -1):
        after.insert(0, _add(C, after[0], backs[j], SW))
    room, upto, kept, uptos, avail = _pad(free, SW), [ZERO] * SW, [], [], []
    for i in range(R):
        upto = _add(C, upto, needs[i], SW)
        avail.append(_add(C, room, after[i], SW))
        uptos.append(upto)
        kept.append(C.AND(run[i]["valid"], _lt(C, avail[i], upto).inv()))
    popped = [C.AND(run[j]["valid"], kept[j].inv()) for j in range(R)]
    tap.update(needs=needs, backs=backs, avail=avail, upto=uptos, kept=kept, popped=popped)
    budget = const_bits(S.T, TW)
    run_n = []
    for j in range(R):
        if S.PC:
            bad.append(C.AND(popped[j], _lt(C, _cdiv(C, run[j]["comp"], LB, HW + 1), backs[j])))
        free = _add(C, free, _sel(C, popped[j], backs[j]), BW)
        free = _sub(C, free, _sel(C, kept[j], needs[j]), BW)
        run_n.append(_sel(C, kept[j], ns[j]))
        budget = _sub(C, budget, run_n[j], TW)
    preempted = or_reduce(C, popped)

    # the waiting loop: W waiting slots, then A arrivals
    stage("waiting")
    cnt0 = _popcount(C, [r["valid"] for r in run], RC + 1)
    cnt = cnt0
    going = preempted.inv()
    k_wait = [ZERO] * (RC + 1)
    used = [ZERO] * AC
    taken = [ZERO] * AC
    taken[0] = forced
    adm, looked, cands = [], [], []
    tap.update(run_n=run_n, cnt0=cnt0, cands=cands)
    if derive is not None:
        derive.running(tap)
    for c in range(W + A):
        is_arr = c >= W
        r = arr[c - W] if is_arr else wait[c]
        present = r["valid"]
        room = _lt(C, cnt, const_bits(S.M, max(RC + 1, _bitlen(S.M)))) if S.M <= R else ONE
        reach = and_reduce(C, [going, present, or_reduce(C, budget), room])
        looked.append([reach])
        ntok = _add(C, r["P"], r["out"], LW)
        if derive is not None:
            h, ev = derive.hit(c, r, ntok, tap)
        else:
            h, ev = (I["hit"][c], I["evict"][c]) if S.PC else ([ZERO] * HW, [ZERO] * HW)
        hb = _pad([ZERO] * LB + list(h), LW + 1)
        if S.PC:
            bad.append(C.AND(reach, OR2(C, _lt(C, h, ev), _lt(C, hb, ntok).inv())))
        n = _clamp(C, _sub(C, ntok, hb, NW), S.H, NW)
        fits = ONE if S.CH else _lt(C, budget, n).inv()
        n = _min(C, n, budget, NW)
        wm = _sel(C, or_reduce(C, cnt), const_bits(S.WM, BW)) if S.WM else [ZERO] * BW
        extra = _add(C, ev, wm, BW + 1)
        conds = [reach, fits]
        if S.FULL:
            full = _monus(C, _cdiv(C, _min(C, ntok, const_bits(S.L, LW + 1), LW + 1), LB, HW + 1), h, HW + 1)
            conds.append(_lt(C, free, _add(C, full, extra, BW + 1)).inv())
        new = _monus(C, _cdiv(C, _min(C, _add(C, hb, n, NW), const_bits(S.L, NW), NW), LB, HW + 1), h, HW + 1)
        conds.append(_lt(C, free, _add(C, new, extra, BW + 1)).inv())
        go = and_reduce(C, conds)
        if is_arr:
            a = c - W
            used = _add(C, used, [go], AC)
            admit = C.AND(go, I["bits"][a][0])
            bad.append(C.AND(admit, arrival_bad(r)))
            going = admit
            taken = _add(C, taken, [admit], AC)
        else:
            admit = go
            going = C.AND(going, present.inv() ^ go)
            k_wait = _add(C, k_wait, [admit], RC + 1)
        bad.append(C.AND(admit, _eq(C, cnt, const_bits(R, RC + 1))))
        at = cnt
        cnt = _add(C, cnt, [admit], RC + 1)
        free = _sub(C, free, _sel(C, admit, _add(C, new, ev, BW)), BW)
        budget = _sub(C, budget, _sel(C, admit, _pad(n, TW)), TW)
        adm.append((admit, hb, _pad(n, TW)))
        cands.append({"reach": reach, "admit": admit, "new": new, "ntok": ntok, "at": at, "hb": hb, "n": _pad(n, TW)})
    bad.append(going)
    if derive is not None:
        derive.candidates(tap)

    # the admitted after the running list
    stage("place")

    tail = (("cap", 1), ("fin", 1)) if AS else ()

    def full_rec(r, valid, comp, n):
        return [valid] + r["id"] + r["P"] + r["O"] + r["out"] + _pad(comp, LW) + n + [r[k] for k, _w in tail]

    def from_rec(x):
        at, out = 0, {}
        for k, w in (("valid", 1), ("id", S.IW), ("P", LW), ("O", LW), ("out", LW), ("comp", LW), ("n", TW)) + tail:
            out[k] = x[at:at + w]
            at += w
        return out

    width = 1 + S.IW + 4 * LW + TW + len(tail)
    wslots = [_sel(C, adm[w][0], full_rec(wait[w], ONE, adm[w][1], adm[w][2])) for w in range(min(W, R))]
    aslots = [_sel(C, adm[W + a][0], full_rec(arr[a], ONE, adm[W + a][1], adm[W + a][2])) for a in range(A)]
    zs = _shift_right(C, aslots, k_wait[:RC], R) if aslots else [[ZERO] * width for _ in range(R)]
    zs = [[x ^ y for x, y in zip(zz, ww)] for zz, ww in zip(zs, wslots + [[ZERO] * width] * (R - len(wslots)))]
    placed = _shift_right(C, zs, cnt0[:RC], R)
    kept = [C.AND(run[p]["valid"], popped[p].inv()) for p in range(R)]
    slots = []
    for p in range(R):
        mine = _sel(C, kept[p], full_rec(run[p], ONE, run[p]["comp"], run_n[p]))
        slots.append(from_rec([x ^ y for x, y in zip(mine, placed[p])]))

    # close and update
    stage("update")
    O = {k: [] for k in ("row_valid", "row_id", "row_kind", "row_start", "row_n")}
    keep, nxt, upd = [], [], []
    tap.update(slots=slots, update=upd)
    for p, s in enumerate(slots):
        n = s["n"]
        on = C.AND(s["valid"][0], or_reduce(C, n))
        O["row_valid"].append([on])
        O["row_id"].append(_sel(C, on, s["id"]))
        O["row_kind"].append([and_reduce(C, [on, _eq(C, n, const_bits(1, TW)), _lt(C, s["comp"], s["P"]).inv()])])
        O["row_start"].append(_sel(C, on, s["comp"]))
        O["row_n"].append(_sel(C, on, n))
        comp = _add(C, s["comp"], n, LW)
        sample = C.AND(on, _lt(C, comp, _add(C, s["P"], s["out"], LW)).inv())
        upd.append({"on": on, "comp": comp, "sampled": sample})
        if AS:
            sample = C.AND(sample, s["fin"][0].inv())
        out = _add(C, s["out"], [sample], LW)
        stop = C.AND(sample, _eq(C, out, s["O"]))
        # with AS a request stopping here runs on (vLLM updates this step after scheduling the next), and the one that
        # stopped the step before leaves
        leave = C.AND(s["valid"][0], s["fin"][0]) if AS else stop
        held = _cdiv(C, comp, LB, HW + 1)
        if derive is not None:
            upd[p].update(leave=leave, held=held)
            back = derive.leave(p, tap)
        else:
            back = ret_of(p, _pad(held, HW))
        if S.PC:
            bad.append(C.AND(leave, _lt(C, held, back)))
        free = _add(C, free, _sel(C, leave, back), BW)
        keep.append(C.AND(s["valid"][0], leave.inv()))
        upd[p].update(leave=leave, held=held, keep=keep[p])
        pfx = []
        if EW:
            pfx = mux(C, on, I["row_pfx"][p], _sel(C, kept[p], I["run_pfx"][p]))
        nxt.append(s["valid"] + s["id"] + s["P"] + s["O"] + out + comp + (s["cap"] + [stop] if AS else []) + pfx)

    stage("compact")
    rw = 1 + S.IW + 4 * LW + 2 * AS + EW
    run_next = _compact(C, nxt, keep, RC + 1)

    # the waiting list: the preempted, in order, then the waiting not admitted
    stage("requeue")
    ww = 1 + S.IW + 3 * LW + AS

    def wfields(r, valid):
        return [valid] + r["id"] + r["P"] + r["O"] + r["out"] + ([r["cap"]] if AS else [])

    wrec = [wfields(wait[w], wait[w]["valid"]) for w in range(W)]
    rest = _shift_left(C, wrec, k_wait)
    # a victim that stopped the step before leaves at this step's update instead of waiting
    vic = [C.AND(popped[p], run[p]["fin"].inv()) for p in range(R)] if AS else popped
    n_pre = _popcount(C, vic, RC + 1)
    n_wait = _popcount(C, [x[0] for x in rest], _bitlen(W) + 1)
    bad.append(_lt(C, const_bits(W, _bitlen(W) + RC + 2), _add(C, n_wait, n_pre, _bitlen(W) + RC + 2)))
    rest = _shift_right(C, rest, n_pre, W)
    vrec = [_sel(C, vic[p], wfields(run[p], ONE)) for p in range(R)]
    if AS:
        front = _compact(C, vrec, vic, RC + 1)
    else:
        front = _shift_left(C, vrec, _popcount(C, kept, RC + 1))
    wait_next = [[x ^ y for x, y in zip(rest[q], front[q])] if q < R else rest[q] for q in range(W)]

    stage("out")
    res = {"ok": [[or_reduce(C, bad).inv()]], "taken": [taken], "used": [used], "looked": looked, **O}
    at = 0
    for k, w in (("run_valid", 1), ("run_id", S.IW), ("run_P", LW), ("run_O", LW), ("run_out", LW), ("run_comp", LW),
                 ("run_cap", AS), ("run_fin", AS), ("run_pfx", EW)):
        res[k] = [x[at:at + w] for x in run_next]
        at += w
    at = 0
    for k, w in (("wait_valid", 1), ("wait_id", S.IW), ("wait_P", LW), ("wait_O", LW), ("wait_out", LW), ("wait_cap", AS)):
        res[k] = [x[at:at + w] for x in wait_next]
        at += w
    res["free"] = [free]
    assert rw == len(run_next[0]) and ww == len(wait_next[0])
    return [e for name, _n, _w in _out_fields(S) for e in res[name]]


def OR2(C, a, b):
    return a ^ b ^ C.AND(a, b)


def _sig(S):
    return tuple((name, Array(n, Array(w, BIT))) for name, n, w in _in_fields(S)), _type(_out_fields(S))


ReplayStep = _part(
    "ReplayStep", STATICS, _sig, _build,
    "one vLLM v1 scheduler step (sync or async, FCFS) on a committed queue state: the certified arrivals it admits, read "
    "with the residual's decisions, its rows and the next state, 0 for ok to refuse what the size class or the prefix layer's "
    "counts can't express (`replay_step.reference`)")


def statics(R: int, W: int, A: int, engine: dict, IW: int, EW: int = 0) -> dict:
    """The binding for size class (R, W, A) of an engine configuration (`replay.Engine`'s fields), with EW the walk
    entries' width when the scan carries `CacheStep_v1`'s table (0 for none)."""
    e = engine
    return {"R": R, "W": W, "A": A, "M": int(e["max_num_seqs"]), "T": int(e["max_num_batched_tokens"]),
            "L": int(e["max_model_len"]), "B": int(e.get("block_size", 16)),
            "H": int(e.get("long_prefill_token_threshold") or 0), "CH": int(bool(e.get("enable_chunked_prefill", True))),
            "WM": int(float(e.get("watermark", 0.0) or 0.0) * int(e["num_gpu_blocks"])),
            "FULL": int(bool(e.get("scheduler_reserve_full_isl", True))), "PC": int(bool(e.get("enable_prefix_caching"))),
            "NB": int(e["num_gpu_blocks"]), "IW": IW, "AS": int(bool(e.get("async_scheduling"))), "EW": EW}


def unit(**st):
    return bind(ReplayStep, **st)


# --- arguments and evaluation ----------------------------------------------------------------------------------------------


def _bits(v: int, w: int) -> list[int]:
    if v < 0 or v >> w:
        raise ValueError(f"{v} doesn't fit {w} bits")
    return [(v >> j) & 1 for j in range(w)]


def arguments(S, st: State, inp: Inputs) -> list[list[int]]:
    """The unit's arguments (each a flat list of bits, as `evaluate_bits` takes one instance)."""
    Z = widths(S)
    vals: dict[str, list[int]] = {}

    def slots(kind, reqs, n, ks):
        vals[f"{kind}_valid"] = [1] * len(reqs) + [0] * (n - len(reqs))
        for k in ks:
            vals[f"{kind}_{k}"] = [getattr(r, k) for r in reqs] + [0] * (n - len(reqs))

    slots("run", st.running, S.R, ("id", "P", "O", "out", "comp", "cap", "fin", "pfx"))
    slots("wait", st.waiting, S.W, ("id", "P", "O", "out", "cap"))
    slots("arr", inp.arrivals, S.A, ("id", "P", "O", "cap"))
    vals["free"] = [st.free]
    for k, n in (("bits", S.A), ("hit", S.W + S.A), ("evict", S.W + S.A), ("ret", S.R), ("row_pfx", S.R)):
        xs = list(getattr(inp, k))[:n]
        vals[k] = xs + [0] * (n - len(xs))
    del Z
    return [[b for v in vals[name] for b in _bits(v, w)] for name, _n, w in _in_fields(S)]


def decode(S, flat: Sequence[int]) -> Out:
    """The unit's output bits as an `Out` (rows by slot, the states' valid slots)."""
    at, f = 0, {}
    for name, n, w in _out_fields(S):
        f[name] = [sum(int(flat[at + e * w + j]) << j for j in range(w)) for e in range(n)]
        at += n * w

    def reqs(kind, comp):
        def get(k, p):
            return f[f"{kind}_{k}"][p] if f"{kind}_{k}" in f else 0
        return tuple(Req(get("id", p), get("P", p), get("O", p), get("out", p), get("comp", p) if comp else 0,
                         get("cap", p), get("fin", p), get("pfx", p))
                     for p in range(len(f[f"{kind}_valid"])) if f[f"{kind}_valid"][p])

    if not f["ok"][0]:
        return Out(False)
    rows = tuple((p, f["row_id"][p], f["row_kind"][p], f["row_start"][p], f["row_n"][p]) for p in range(S.R) if f["row_valid"][p])
    return Out(True, rows, State(reqs("run", True), reqs("wait", False), f["free"][0]), f["taken"][0], f["used"][0],
               tuple(f.get("looked", ())))


def evaluate(st_: dict, cases: Sequence[tuple[State, Inputs]]) -> list[Out]:
    """The unit at statics `st_` on each (state, inputs), in one evaluation."""
    S = SimpleNamespace(**st_)
    args = [list(rows) for rows in zip(*(arguments(S, s, i) for s, i in cases))]
    out = evaluate_bits(unit(**st_), *args)
    return [decode(S, row) for row in out.reshape(len(cases), -1).tolist()]


def _req(r: Req) -> list[int]:
    """(id, P, O, out, comp), then (cap, fin, pfx) up to the last nonzero."""
    x = [r.id, r.P, r.O, r.out, r.comp, r.cap, r.fin, r.pfx]
    while len(x) > 5 and not x[-1]:
        x.pop()
    return x


def to_json(state: State, inp: Inputs, want: Out) -> list:
    """A difftest vector (a step as the replay ran it) as JSON: sparse inputs, the outputs of an accepted step."""
    def nz(xs):
        return [[k, x] for k, x in enumerate(xs) if x]

    assert want.ok and want.state is not None
    return [[[_req(r) for r in state.running], [_req(r) for r in state.waiting], state.free],
            [[_req(r) if r.cap else _req(r)[:3] for r in inp.arrivals], [k for k, b in enumerate(inp.bits) if b], nz(inp.hit),
             nz(inp.evict), nz(inp.ret)] + ([nz(inp.row_pfx)] if any(inp.row_pfx) else []),
            [[list(r) for r in want.rows], [[_req(r) for r in want.state.running], [_req(r) for r in want.state.waiting],
                                            want.state.free], want.taken, want.used]
            + ([[k for k, b in enumerate(want.looked) if b]] if want.looked else [])]


def from_json(st_: dict, v: list) -> tuple[State, Inputs, Out]:
    (run, wait, free), (arr, ones, hit, evict, ret, *pfx), (rows, (nrun, nwait, nfree), taken, used, *lk) = v

    def dense(pairs, n):
        out = [0] * n
        for k, x in pairs:
            out[k] = x
        return tuple(out)

    def reqs(xs):
        return tuple(Req(*x) for x in xs)

    A, W, R = st_["A"], st_["W"], st_["R"]
    return (State(reqs(run), reqs(wait), free),
            Inputs(reqs(arr), dense([(k, 1) for k in ones], A), dense(hit, W + A), dense(evict, W + A), dense(ret, R),
                   dense(pfx[0], R) if pfx else ()),
            Out(True, tuple(tuple(r) for r in rows), State(reqs(nrun), reqs(nwait), nfree), taken, used,
                dense([(k, 1) for k in (lk[0] if lk else ())], W + A) if st_.get("EW") else ()))


# --- cost ------------------------------------------------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _staged(key: tuple) -> dict[str, int]:
    st_ = dict(zip(STATICS, key))
    S = SimpleNamespace(**st_)
    C = Circuit(cse=True)
    C.phase = "arrivals"
    params, _ = _sig(S)
    ins = [C.input(f"p{i}e{j}", w) for i, (_n, t) in enumerate(params) for j, w in enumerate([t.elem.n] * t.n)]
    C.output("out", [b for e in _build(C, S, *ins) for b in e])
    return C.phase_counts()


def size(**st_) -> dict[str, int]:
    """The unit's ANDs by stage at statics `st_` (traced on a counting circuit, no gates written)."""
    out = dict(_staged(tuple(st_[k] for k in STATICS)))
    out["ands"] = sum(out.values())
    return out


__all__ = ["Inputs", "Out", "ReplayStep", "Req", "STATICS", "State", "arguments", "decode", "empty", "evaluate", "from_json",
           "reference", "size", "statics", "to_json", "unit", "widths"]
