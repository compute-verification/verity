"""The difftest of the replay step circuit (`verity.core.protocols.certifier.replay_step`) against the replay itself.

`vectors(trace, R=, W=, A=)` replays a trace (`replay.replay` with the trace's own decisions, sync or async scheduling)
with its scheduler instrumented, and returns each executed step as the step circuit sees it (under async scheduling a
step is one schedule and the update of the step before it): the state after the step before
(running, waiting, free blocks), the next A arrivals, the step's decisions, the prefix layer's counts (each candidate's
hit and evictable blocks, as `_KV.hit` found them; each leaving request's returned blocks, as `_KV.free_request`
freed them) and what the replay did (rows, next state, arrivals taken, decisions read). It also checks the one fact the
circuit's state rests on, that a running request holds ⌈computed / B⌉ blocks.

    python3 replay_vectors.py TRACE... [--R 16 --W 64 --A 16] [--EW 9] [--circuit] [--out OUT.json] [--fixture N]

checks `replay_step.reference` on every step that fits the size class (the rest are counted, out of class), and with
`--circuit` the Definition on them too; with `--EW` each step's `looked` against the lookups the scheduler made (the
candidates whose `_KV.hit` it called), which is what v1 sends a shortfall for past the pool bound; `--fixture N` writes each trace's N consecutive fitting steps with the most
preemptions, cache hits, admissions and stops (`window`) as the vectors the certifier's test reads
(`verity/core/protocols/certifier/tests/fixtures/replay_step_vectors.json`).

**A served window as the compile slice's input.** `served_window` cuts one window out of a served run's step trace (the
server's `window` open and close records, `verity_vllm.engine.step_trace`), renumbered from its first step, and
refuses a window that doesn't open on a drained engine, since the slice's replay starts from the empty one.
`slice_inputs` gives `compile_slice.CompileSlice_v1`'s inputs for such a window in `compile_slice_smallkv.json`'s
layout: the statics, the certified arrivals (id, P, O, and cap at AS = 1), the decisions in the scan's order, the
executed step count and each step's rows (slot, request, kind, start, n). The scheduler (AS) is the engine class's, the
trace's own unless the class is the sync one (AS = 0), and it refuses a window whose schedule at that AS isn't the
served one.

    python3 replay_vectors.py TRACE --slice N [--max-model-len L] [--served SEAL] [--AS 0|1] --out FIXTURE.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP  # noqa: E402

REPO = Path(__file__).resolve().parents[3]

from verity.core.protocols.certifier import replay_step as RS  # noqa: E402


class _Recorder:
    """What the scan's units saw. A unit is one `schedule()` and, under async scheduling, the update of the step before it
    (vLLM updates a step after scheduling the next); sync, the update of its own step."""

    def __init__(self) -> None:
        self.adds = 0
        self.asked_pending = False
        self.in_schedule = False
        self.open: dict[str, Any] | None = None             # the unit not yet closed
        self.hits: list[tuple[int, int, int]] = []          # (request, hit blocks, evictable) per `_KV.hit`
        self.frees: dict[int, int] = {}                      # request -> blocks returned
        self.asked: list[int] = []                           # the step's decisions, in order
        self.units: list[dict[str, Any]] = []

    def close(self, s: RP._Sched) -> None:
        if self.open is not None:
            self.units.append({**self.open, "frees": dict(self.frees), "hits": list(self.hits), "adds": self.adds,
                               "next": _snap(s)})
            self.open = None


def _snap(s: RP._Sched) -> RS.State:
    """The scheduler's state as the step circuit carries it. Under async scheduling a running request's outputs count
    its placeholder (the output its last row samples, which vLLM appends at that step's update), `fin` says that output
    is its last, and `cap` that its output length is its max_tokens."""
    B, asy = s.e.block_size, s.e.async_scheduling
    if s.skipped:
        raise ValueError("a skipped request (a stale output still in flight), which the step circuit doesn't model")
    for r in s.running:
        held = len(s.kv.req_blocks.get(r.i, ()))
        if r.placeholders > int(asy) or held != RS.cdiv(r.num_computed, B):
            raise AssertionError(f"request {r.i}: holds {held} blocks at {r.num_computed} computed, "
                                 f"{r.placeholders} placeholders")
    for r in s.waiting:
        if r.num_computed or s.kv.req_blocks.get(r.i) or r.placeholders or r.stale:
            raise AssertionError(f"waiting request {r.i}: {r.num_computed} computed")

    def cap(r: RP._R) -> int:
        return int(asy and r.io.output_len >= r.io.max_tokens)

    return RS.State(tuple(RS.Req(r.i, r.P, r.io.output_len, r.n_out + r.placeholders, r.num_computed, cap(r),
                                 int(asy and r.n_out + r.placeholders == r.io.output_len)) for r in s.running),
                    tuple(RS.Req(r.i, r.P, r.io.output_len, r.n_out, 0, cap(r)) for r in s.waiting), len(s.kv.free))


@contextmanager
def _instrumented(rec: _Recorder):
    S, K = RP._Sched, RP._KV
    orig = {"add": S.add, "schedule": S.schedule, "update": S.update, "hit": K.hit, "free_request": K.free_request}

    def add(self, i):
        if not rec.in_schedule:
            rec.close(self)
        rec.adds += 1
        return orig["add"](self, i)

    def schedule(self, on_open=None):
        rec.close(self)
        before = [r.i for r in self.running]
        rec.hits, rec.frees, rec.asked = [], {}, []

        def asked(p):
            rec.asked_pending = False
            got = on_open(p)
            if rec.asked_pending:
                rec.asked.append(int(got))
            return got

        rec.in_schedule = True
        try:
            step, samples = orig["schedule"](self, asked if on_open is not None else None)
        finally:
            rec.in_schedule = False
        rec.open = {"step": step, "before": before, "running": [r.i for r in self.running], "asked": list(rec.asked)}
        return step, samples

    def update(self, step, samples):
        out = orig["update"](self, step, samples)
        rec.close(self)
        return out

    def hit(self, r):
        out = orig["hit"](self, r)
        rec.hits.append((r.i, len(out), sum(1 for b in out if self.ref[b] == 0)))
        return out

    def free_request(self, r):
        rec.frees[r.i] = rec.frees.get(r.i, 0) + sum(1 for b in self.req_blocks.get(r.i, ()) if self.ref[b] == 1)
        return orig["free_request"](self, r)

    S.add, S.schedule, S.update, K.hit, K.free_request = add, schedule, update, hit, free_request
    try:
        yield
    finally:
        S.add, S.schedule, S.update = orig["add"], orig["schedule"], orig["update"]
        K.hit, K.free_request = orig["hit"], orig["free_request"]


def vectors(trace: RP.Trace, *, R: int, W: int, A: int, EW: int = 0) -> tuple[dict, list[tuple[RS.State, RS.Inputs, RS.Out]]]:
    """The binding for `trace`'s engine at (R, W, A), and every unit of the scan as (state, inputs, the replay's
    outputs): each `schedule()`, empty ones included, since under async scheduling one still updates the step before.
    With EW the outputs carry `looked`, the candidates whose `_KV.hit` the step called (the walk entries stay 0)."""
    e = trace.engine
    n = len(trace.requests)
    st = RS.statics(R, W, A, dict(e.__dict__), max(n - 1, 1).bit_length(), EW)
    first = trace.first_scheduled()
    rec = _Recorder()

    def decide(i: int, k: int, _o: Any) -> bool:
        rec.asked_pending = True
        return k >= first[i]

    with _instrumented(rec):
        rp = RP.replay(e, trace.requests, decide=decide)
        if rec.open is not None:
            raise AssertionError("the replay ended on a schedule with no update after it")
    m = RP.compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch}")
    reqs = trace.requests

    def cap(i: int) -> int:
        return int(e.async_scheduling and reqs[i].output_len >= reqs[i].max_tokens)

    out = []
    prev, adds_before = RS.State((), (), e.num_gpu_blocks - 1), 0
    for sr in rec.units:
        step: RP.Step = sr["step"]
        quiet = not prev.running and not prev.waiting
        forced = int(quiet and adds_before < n)
        first_open = adds_before + forced
        wait_ids = [r.id for r in prev.waiting] or ([adds_before] if forced else [])
        hit, evict, looked = [0] * (W + A), [0] * (W + A), [0] * (W + A)
        for i, h, ev in sr["hits"]:
            c = wait_ids.index(i) if i in wait_ids else W + (i - first_open)
            if 0 <= c < W + A:
                hit[c], evict[c], looked[c] = h, ev, 1
        ret = [0] * R
        if [r.id for r in prev.running] != sr["before"]:
            raise AssertionError("a unit's running list isn't the one the unit before it left")
        for i, back in sr["frees"].items():
            p = sr["running"].index(i) if i in sr["running"] else sr["before"].index(i)
            if p < R:
                ret[p] = back
        arrivals = tuple(RS.Req(i, reqs[i].prompt_len, reqs[i].output_len, cap=cap(i))
                         for i in range(adds_before, min(adds_before + A, n)))
        inp = RS.Inputs(arrivals, tuple(sr["asked"]) + (0,) * (A - len(sr["asked"])), tuple(hit), tuple(evict), tuple(ret))
        rows = tuple((sr["running"].index(i), i, int(nn == 1 and start >= reqs[i].prompt_len), start, nn)
                     for i, start, nn in step.rows)
        want = RS.Out(True, rows, sr["next"], sr["adds"] - adds_before, len(sr["asked"]), tuple(looked) if EW else ())
        out.append((prev, inp, want))
        prev, adds_before = sr["next"], sr["adds"]
    return st, out


def fits(st: dict, state: RS.State, inp: RS.Inputs, want: RS.Out) -> bool:
    R, W = st["R"], st["W"]
    return (len(state.running) <= R and len(state.waiting) <= W and want.state is not None and len(want.state.waiting) <= W
            and all(p < R for p, *_ in want.rows) and want.taken < st["A"])


def window(cases: Sequence[tuple[RS.State, RS.Inputs, RS.Out]], n: int) -> list:
    """The n consecutive cases with the most events: a preemption counts 4, a cache hit or an async lag step 2, an
    admission or a stop 1."""
    def score(c) -> int:
        s, i, w = c
        before = {r.id for r in s.running}
        return (4 * any(r.id in before for r in w.state.waiting) + 2 * any(i.hit) + 2 * any(r.fin for r in s.running)
                + (w.taken > 0) + (len(w.state.running) < len(s.running) + w.taken))
    sc = [score(c) for c in cases]
    if len(cases) <= n:
        return list(cases)
    best = max(range(len(cases) - n + 1), key=lambda k: (sum(sc[k:k + n]), -k))
    return list(cases[best:best + n])


def check(st: dict, cases: Sequence[tuple[RS.State, RS.Inputs, RS.Out]], *, circuit: bool, batch: int = 512) -> dict:
    from types import SimpleNamespace
    S = SimpleNamespace(**st)
    ref_bad = [k for k, (s, i, w) in enumerate(cases) if RS.reference(S, s, i) != w]
    out = {"checked": len(cases), "reference_mismatches": len(ref_bad), "reference_first": ref_bad[:5]}
    if circuit:
        bad = []
        for at in range(0, len(cases), batch):
            chunk = cases[at:at + batch]
            got = RS.evaluate(st, [(s, i) for s, i, _ in chunk])
            bad += [at + k for k, (g, (_, _, w)) in enumerate(zip(got, chunk)) if g != w]
        out.update(circuit_mismatches=len(bad), circuit_first=bad[:5])
    return out


def served_window(records: Sequence[Mapping[str, Any]], window: int) -> tuple[list[dict[str, Any]], int]:
    """Window `window` of a served step trace as a trace of its own: the header, then the arrivals the engine core added
    before the window's steps and the steps themselves, each renumbered from its first step; and the session index of
    its first arrival. Refuses a window the trace doesn't open and close, and one that opens on an engine still holding a
    request (one that arrived before it and no step before it finished)."""
    recs = list(records)
    marks = {(r["window"], r["event"]): int(r["before_step"]) for r in recs if r.get("kind") == "window"}
    if (window, "open") not in marks or (window, "close") not in marks:
        raise ValueError(f"window {window}: the trace doesn't open and close it")
    lo, hi = marks[window, "open"], marks[window, "close"]
    arrivals = [r for r in recs if r.get("kind") == "arrive"]
    finished = {a for r in recs if r.get("kind") == "step" and r["step"] < lo for a in r.get("finished") or {}}
    held = [r["request"] for r in arrivals if r["before_step"] < lo and r["request"] not in finished]
    if held:
        raise ValueError(f"window {window} opens at step {lo} with {len(held)} requests unfinished, {held[0]} first")
    out = [dict(recs[0])]
    for r in recs[1:]:
        if r.get("kind") == "arrive" and lo <= r["before_step"] < hi:
            out.append({**r, "before_step": r["before_step"] - lo})
        elif r.get("kind") == "step" and lo <= r["step"] < hi:
            out.append({**r, "step": r["step"] - lo})
    return out, sum(1 for r in arrivals if r["before_step"] < lo)


def slice_inputs(trace: RP.Trace, *, R: int, W: int, A: int, base: int = 0, session: int | None = None,
                 Q: int | None = None, D: int | None = None, P: int | None = None,
                 K: int | None = None, AS: int | None = None) -> dict[str, Any]:
    """CompileSlice_v1's inputs for `trace`, one window from the empty engine (`served_window`'s), at size class
    (R, W, A), in `compile_slice_smallkv.json`'s layout. A request's id is `base` plus its index in
    the trace (with `served_window`'s base, its index in the session), and IW holds the ids of a session of `session`
    requests (default the window's last). Q and K default to the window's arrivals, D to its decisions and P to its
    executed units; a public class larger than the window pads it (empty arrival slots, decisions past the count, idle
    steps), and one smaller is refused. AS is the class's scheduler, so one AS holds for every window of a run: by
    default the trace's own (its header's `scheduler`), and 0 only for the sync class. At AS = 0 the window is replayed
    under sync scheduling. At AS = 1 it is replayed under the trace's async scheduling: the engine keeps
    `async_scheduling`, each arrival is (id, P, O, cap), cap meaning O is its max_tokens, and a unit is one schedule and
    the update of the step before it. It is refused if that schedule isn't the trace's (`vectors`) or a unit is outside
    the class."""
    if AS is None:
        AS = int(trace.engine.async_scheduling)
    if AS and not trace.engine.async_scheduling:
        raise ValueError("AS = 1 needs a trace served under async scheduling")
    e = trace.engine if AS else dataclasses.replace(trace.engine, async_scheduling=False)
    st, vs = vectors(dataclasses.replace(trace, engine=e), R=R, W=W, A=A)
    while vs and not vs[-1][2].rows and vs[-1][2].state == vs[-1][0]:
        vs.pop()                                   # the engine loop's last schedule, on the drained engine
    out_of_class = [k for k, v in enumerate(vs) if not fits(st, *v)]
    if out_of_class:
        raise ValueError(f"steps {out_of_class[:5]} are outside (R, W, A) = ({R}, {W}, {A})")
    reqs = trace.requests
    decisions = [b for _s, inp, out in vs for b in inp.bits[:out.used]]
    own = {"Q": len(reqs), "D": len(decisions), "P": len(vs), "K": len(reqs)}
    cls = {k: own[k] if v is None else v for k, v in (("Q", Q), ("D", D), ("P", P), ("K", K))}
    small = [f"{k} = {cls[k]} < {own[k]}" for k in own if cls[k] < own[k]]
    if small:
        raise ValueError(f"the class is smaller than the window: {', '.join(small)}")
    engine = {k: v for k, v in dataclasses.asdict(e).items() if k != "policy" and (AS or k != "async_scheduling")}
    return {"engine": engine, "R": R, "W": W, "A": A,
            "IW": max((session if session is not None else base + len(reqs)) - 1, 1).bit_length(), **cls,
            "arrivals": [[base + i, r.prompt_len, r.output_len] + ([int(r.output_len >= r.max_tokens)] if AS else [])
                         for i, r in enumerate(reqs)], "decisions": decisions,
            "steps": len(vs),
            "rows": [[[slot, base + i, kind, start, n] for slot, i, kind, start, n in out.rows] for _s, _i, out in vs]}


def served_record(seal: Sequence[Mapping[str, Any]], window: int, requests: Sequence[str],
                  base: int = 0) -> tuple[list[list[list[Any]]], dict[str, int]]:
    """Window `window`'s served rows as the Commit's seal recorded them, each step's `rows` in
    `SamplerInputsSource.rows`' form (the runner's request id, its token rows [lo, hi) of the batch, its first position
    and its prefill length), and the map from a runner's request id to its arrival id: `base` plus the index in
    `requests` (the engine's ids, which the runner's extend by one suffix) of the request it serves."""
    steps = sorted((r for r in seal if r.get("window") == window), key=lambda r: r["step"])
    if [r["step"] for r in steps] != list(range(len(steps))):
        raise ValueError(f"window {window}'s sealed steps are not 0..S-1 each once")
    index = {rid: base + k for k, rid in enumerate(requests)}
    ids: dict[str, int] = {}
    for r in steps:
        for req, *_ in r["rows"]:
            engine_id = req.rsplit("-", 1)[0]
            if engine_id not in index:
                raise ValueError(f"window {window}: the runner's {req} serves no request of the window")
            ids[req] = index[engine_id]
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"window {window}: two runner ids serve one request")
    return [[list(row) for row in r["rows"]] for r in steps], ids


def by_unit(rows: Sequence[Sequence[Any]], served: Sequence[Any]) -> list[Any]:
    """The seal's steps (`served_record`'s) placed at the async units that ran them: under async scheduling a unit
    whose schedule runs nothing (the lag schedule after a finish, before its update) is no executed step, so the
    sealed steps are the units with rows, in order, and the others get none."""
    ran = [u for u, r in enumerate(rows) if r]
    if len(ran) != len(served):
        raise ValueError(f"{len(served)} sealed steps for {len(ran)} units with rows")
    out: list[Any] = [[] for _ in rows]
    for u, s in zip(ran, served):
        if len(s) != len(rows[u]):
            raise ValueError(f"unit {u}: {len(rows[u])} rows, sealed {len(s)}")
        out[u] = s
    return out


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="The replay step circuit's difftest against the replay, step for step.")
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--EW", type=int, default=0, help="walk entries of this width, and `looked` checked against `_KV.hit`")
    ap.add_argument("--circuit", action="store_true", help="evaluate the Definition on the fitting steps too")
    ap.add_argument("--fixture", type=int, default=0, help="keep the first N fitting steps of each trace")
    ap.add_argument("--slice", type=int, metavar="WINDOW", help="write CompileSlice_v1's inputs for this served window of "
                    "the one trace (`served_window`, `slice_inputs`)")
    ap.add_argument("--served", type=Path, help="with --slice: the Commit's seal, whose rows for the window go in `served`")
    ap.add_argument("--max-model-len", type=int, help="with --slice: the engine's, where the trace's header lacks it")
    for k, what in (("Q", "pending arrivals"), ("D", "decisions"), ("P", "steps"), ("K", "certified arrival slots")):
        ap.add_argument(f"--{k}", type=int, help=f"with --slice: the public class's {what} (default the window's own)")
    ap.add_argument("--AS", type=int, choices=(0, 1),
                    help="with --slice: the engine class's scheduler, the same for every window of a run (default the "
                    "trace's own; 0 is the sync class, 1 the trace's async scheduling; `slice_inputs`)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    if a.slice is not None:
        (p,) = a.traces
        recs = RP.records(p)
        if a.max_model_len:
            recs[0]["config"]["max_model_len"] = a.max_model_len
        sub, base = served_window(recs, a.slice)
        tr = RP.load_step_trace(sub)
        name = p.resolve().relative_to(REPO) if p.resolve().is_relative_to(REPO) else p
        cls = {k: getattr(a, k) for k in ("Q", "D", "P", "K") if getattr(a, k) is not None}
        AS = int(tr.engine.async_scheduling) if a.AS is None else a.AS
        given = {"R": a.R, "W": a.W, "A": a.A, **cls, **({"AS": 1} if AS else {})}
        at = ", ".join(f"{k} {v}" for k, v in given.items())
        fx = {"source": f"{name} window {a.slice} (replay_vectors.py --slice {a.slice} at {at})",
              **slice_inputs(tr, R=a.R, W=a.W, A=a.A, base=base,
                             session=sum(r.get("kind") == "arrive" for r in recs), AS=AS, **cls)}
        if a.served:
            fx["served"], fx["ids"] = served_record(RP.records(a.served), a.slice, tr.ids, base)
            if AS:
                fx["served"] = by_unit(fx["rows"], fx["served"])
        text = json.dumps(fx, indent=1) + "\n"
        a.out.write_text(text) if a.out else print(text, end="")
        return
    report: dict[str, Any] = {"R": a.R, "W": a.W, "A": a.A, "traces": {}}
    fixture = []
    for p in a.traces:
        name = p.stem if p.is_file() else f"{p.resolve().parent.name}/{p.name}"
        tr = RP.read_any(p)
        try:
            st, vs = vectors(tr, R=a.R, W=a.W, A=a.A, EW=a.EW)
        except ValueError as e:
            report["traces"][name] = {"skipped": str(e)}
            print(f"{name}: skipped ({e})", flush=True)
            continue
        ok = [v for v in vs if fits(st, *v)]
        shape = {"max_running": max(len(s.running) for s, _, _ in vs), "max_waiting": max(len(s.waiting) for s, _, _ in vs),
                 "max_taken": max(w.taken for _, _, w in vs), "max_used": max(w.used for _, _, w in vs)}
        r = report["traces"][name] = {"statics": st, "steps": len(vs), "in_class": len(ok), **shape,
                                      **check(st, ok, circuit=a.circuit)}
        print(f"{name}: {json.dumps(r)}", flush=True)
        if a.fixture:
            fixture.append({"trace": name, "statics": st, "vectors": [RS.to_json(*v) for v in window(ok, a.fixture)]})
    if a.out:
        a.out.write_text(json.dumps(fixture if a.fixture else report, separators=(",", ":") if a.fixture else None,
                                    indent=None if a.fixture else 1))


if __name__ == "__main__":
    main()
