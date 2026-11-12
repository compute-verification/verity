"""The certified tick as one step of a scan: the rule `replay.replay(tick=...)` runs, as a function of a state, the next
certified arrivals and their release buckets, with nothing asked.

`reference` is that step's specification in Python. It is `ReplayStep_v1`'s step
(`verity.protocols.certifier.verifier.replay_step.reference`) with its timing input replaced: where that step reads one
decision bit per open arrival, this one keeps the tick's clock T in its state and gives the scheduler an arrival exactly
when T has reached the close of its release bucket. No Definition implements it; the replay's own steps are its
difftest (`adapters/vllm/tests/pipeline/test_tick_step.py`).

**State** (`State`): `ReplayStep_v1`'s state (the running list, the waiting list, the free blocks) and T, an integer
of microseconds. The waiting list holds the preempted requests and an arrival that entered a quiet engine and wasn't
scheduled; an arrival whose bucket has closed and that no step has taken stays in the certified stream, as an arrival
no decision has admitted does in `ReplayStep_v1`.

**Inputs** (`Inputs`): the next certified arrivals in release order, at most A, each with its release bucket, and
`ReplayStep_v1`'s prefix counts (`hit`, `evict`, `ret`, `row_pfx`), by the same candidates and slots. No decision bits.

**Outputs** (`Out`): ok, the rows, the next state with the next T, `taken` (how far the arrivals' cursor moves) and
`looked`. No count of decisions read.

**What it computes**, with the tick's registered constants τ_b, τ_sat and τ_free (`schedule.Tick`), in order:
1. If nothing is running or waiting and there is a next arrival, of bucket a: T ← max(T, (a + 1)·τ_b).
2. The arrivals *given* to the scheduler are the leading ones whose buckets have closed, (a + 1)·τ_b ≤ T. The buckets
   never decrease in release order, so these are all the closed ones in view.
3. `ReplayStep_v1`'s step runs on the state and the given arrivals, every decision bit 1: the waiting loop takes each
   given arrival it has room for, in order, and none it wasn't given.
4. The step is *saturated* if it preempted a request, or the next state's waiting list is nonempty, or a given arrival
   was not taken: the three ways vLLM's waiting queue is nonempty once the schedule returns, which is when the replay
   reads it.
5. If the step has a row, T advances by τ_sat when saturated and by τ_free when not. A step with no row leaves T.

**Three choices inside it.**
- *An iteration of the engine core's loop that only frees finished requests is no step of the scan.* When nothing is
  running or waiting and the update before finished a request, the replay's loop makes one `schedule()` that adds
  nothing, schedules nothing and leaves T, and the state above is the same before and after it. The step after a drain
  is therefore the one that moves T to the next bucket's close and admits. Keeping the iteration would cost a state
  bit (the update before finished a request) whose one effect is a step with no row.
- *A ≥ min(M, R) + 1, and `reference` raises on any other class.* Saturation asks whether a closed arrival was left
  waiting. A step takes at most min(M, R) arrivals, so with one more in view it always sees the first arrival it
  doesn't take: if that one has closed the step is saturated, and if it hasn't, no later one has. A smaller class
  could take all it sees and not know.
- *The replay's queue of skipped requests is taken as empty.* The replay reads it beside the waiting queue, and
  `ReplayStep_v1`'s state doesn't hold it. A request is skipped when the waiting loop reaches it with a row of its own
  still in flight. A preempted request's rows in flight are the step before's, the schedule that preempts runs no
  waiting loop, and the engine core's loop updates the step before ahead of the next schedule, so the queue is empty
  wherever T advances. The difftest asserts it after every schedule.

**The step preempted.** `ReplayStep_v1`'s reference doesn't say so, and its two states don't either: under async
scheduling a preempted request whose last output is in flight is not requeued, so it leaves the running list as a
request that finished does, yet it is in vLLM's waiting queue when the replay reads it. `_preempts` is therefore the
running loop's own test, walked to its first preemption: each running request in order takes the tokens the budget
leaves it and their new blocks, and the first whose new blocks exceed the free ones preempts.

**The scan.** From `empty` (the empty engine, T = 0) over the certified arrivals in release order, the cursor moving
by `taken`, this step gives the replay's rows and its T after each step with rows (`replay.Replay.clock`). T is state:
a window of the scan opens with the T the window before it left, on a drained engine too. One that restarts T there
moves it to the next bucket's close, which the rule's clock may already have passed, and then admits a later arrival
a step late.

**Not modeled:** what `ReplayStep_v1` doesn't model, and which blocks a prefix hit finds, evicts or returns: the
prefix counts are inputs here as they are there.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from verity.protocols.certifier.verifier import replay_step as RS

from verity_vllm.pipeline.advice_code.schedule import Tick

__all__ = ["Inputs", "Out", "State", "empty", "reference"]


@dataclass(frozen=True)
class State:
    step: RS.State                          # `ReplayStep_v1`'s state
    clock: int = 0                          # the tick's T, in microseconds


@dataclass(frozen=True)
class Inputs:
    arrivals: tuple[RS.Req, ...]            # the next certified arrivals (at most A), in release order
    released: tuple[int, ...]               # each one's release bucket
    hit: tuple[int, ...] = ()               # per candidate (W waiting slots, then A arrivals): hit blocks
    evict: tuple[int, ...] = ()             # per candidate: of those, the evictable
    ret: tuple[int, ...] = ()               # per running slot: blocks returned when it leaves
    row_pfx: tuple[int, ...] = ()           # per slot (the rows' slots): the walk entry after its row's lookup


@dataclass(frozen=True)
class Out:
    ok: bool
    rows: tuple[tuple[int, int, int, int, int], ...] = ()   # (slot, request, kind, start, n)
    state: State | None = None
    taken: int = 0
    looked: tuple[int, ...] = ()            # with EW, per candidate: whether the step read its lookup


def empty(S: SimpleNamespace) -> State:
    """The empty engine before any arrival, T = 0."""
    return State(RS.empty(S), 0)


def _preempts(S: SimpleNamespace, st: RS.State) -> bool:
    """Whether `replay_step.reference`'s running loop preempts on `st`: its test, up to the first preemption."""
    free, budget = st.free, S.T
    for r in st.running:
        n = r.P + r.out - r.comp
        if 0 < S.H < n:
            n = S.H
        n = min(n, budget, S.L - r.comp - 1)
        if n <= 0 or (S.AS and r.fin and r.cap):
            continue
        need = RS.cdiv(r.comp + n, S.B) - RS.cdiv(r.comp, S.B)
        if need > free:
            return True
        free -= need
        budget -= n
    return False


def reference(S: SimpleNamespace, tick: Tick, st: State, inp: Inputs) -> Out:
    """One step of the certified tick on the size class `S` (`replay_step.statics`' binding); `Out(False)` refuses
    what `replay_step.reference` refuses."""
    most = min(S.M, S.R)
    if most >= S.A:
        raise ValueError(f"A = {S.A} arrivals in view with up to {most} taken a step: the tick step needs "
                         f"A ≥ min(M, R) + 1 = {most + 1}, to see the first arrival it doesn't take")
    if len(inp.released) != len(inp.arrivals):
        raise ValueError(f"{len(inp.released)} release buckets for {len(inp.arrivals)} arrivals")
    clock = st.clock
    if not st.step.running and not st.step.waiting and inp.arrivals:
        clock = max(clock, tick.close(inp.released[0]))
    given = 0
    while given < len(inp.arrivals) and tick.close(inp.released[given]) <= clock:
        given += 1
    asked = RS.Inputs(inp.arrivals[:given], (1,) * S.A, inp.hit, inp.evict, inp.ret, inp.row_pfx)
    out = RS.reference(S, st.step, asked)
    if not out.ok or out.state is None:
        return Out(False)
    if out.rows:
        saturated = _preempts(S, st.step) or bool(out.state.waiting) or out.taken < given
        clock += tick.sat_us if saturated else tick.free_us
    return Out(True, out.rows, State(out.state, clock), out.taken, out.looked)
