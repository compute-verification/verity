"""The difftest of the replay step circuit (`verity.core.protocols.certifier.replay_step`) against the replay itself.

`vectors(trace, R=, W=, A=)` replays a trace (`replay.replay` with the trace's own decisions, sync scheduling) with its
scheduler instrumented, and returns each executed step as the step circuit sees it: the state after the step before
(running, waiting, free blocks), the next A arrivals, the step's decisions, the prefix layer's counts (each candidate's
hit and evictable blocks, as `_KV.hit` found them; each leaving request's returned blocks, as `_KV.free_request`
freed them) and what the replay did (rows, next state, arrivals taken, decisions read). It also checks the one fact the
circuit's state rests on, that a running request holds ⌈computed / B⌉ blocks.

    python3 replay_vectors.py TRACE... [--R 16 --W 64 --A 16] [--circuit] [--out OUT.json] [--fixture N]

checks `replay_step.reference` on every step that fits the size class (the rest are counted, out of class), and with
`--circuit` the Definition on them too; `--fixture N` writes each trace's N consecutive fitting steps with the most
preemptions, cache hits, admissions and stops (`window`) as the vectors the certifier's test reads
(`verity/core/protocols/certifier/tests/fixtures/replay_step_vectors.json`).
"""
from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay as RP  # noqa: E402

from verity.core.protocols.certifier import replay_step as RS  # noqa: E402


class _Recorder:
    def __init__(self) -> None:
        self.adds = 0
        self.asked_pending = False
        self.hits: list[tuple[int, int, int]] = []          # (request, hit blocks, evictable) per `_KV.hit`
        self.frees: dict[int, int] = {}                      # request -> blocks returned
        self.asked: list[int] = []                           # the step's decisions, in order
        self.update_running: list[int] = []                  # the running list as the step's update saw it
        self.snaps: list[RS.State] = []
        self.steps: list[dict[str, Any]] = []


def _snap(s: RP._Sched) -> RS.State:
    B = s.e.block_size
    if s.skipped:
        raise ValueError("a skipped request: async scheduling, which the step circuit doesn't model")
    for r in s.running:
        held = len(s.kv.req_blocks.get(r.i, ()))
        if r.placeholders or held != RS.cdiv(r.num_computed, B):
            raise AssertionError(f"request {r.i}: holds {held} blocks at {r.num_computed} computed")
    for r in s.waiting:
        if r.num_computed or s.kv.req_blocks.get(r.i):
            raise AssertionError(f"waiting request {r.i}: {r.num_computed} computed")
    return RS.State(tuple(RS.Req(r.i, r.P, r.io.output_len, r.n_out, r.num_computed) for r in s.running),
                    tuple(RS.Req(r.i, r.P, r.io.output_len, r.n_out, 0) for r in s.waiting), len(s.kv.free))


@contextmanager
def _instrumented(rec: _Recorder):
    S, K = RP._Sched, RP._KV
    orig = {"add": S.add, "schedule": S.schedule, "update": S.update, "hit": K.hit, "free_request": K.free_request}

    def add(self, i):
        rec.adds += 1
        return orig["add"](self, i)

    def schedule(self, on_open=None):
        rec.hits, rec.frees, rec.asked = [], {}, []

        def asked(p):
            rec.asked_pending = False
            got = on_open(p)
            if rec.asked_pending:
                rec.asked.append(int(got))
            return got

        step, samples = orig["schedule"](self, asked if on_open is not None else None)
        rec.step = step
        return step, samples

    def update(self, step, samples):
        rec.update_running = [r.i for r in self.running]
        out = orig["update"](self, step, samples)
        rec.steps.append({"step": step, "frees": dict(rec.frees), "hits": list(rec.hits), "asked": list(rec.asked),
                          "running": list(rec.update_running), "adds": rec.adds})
        rec.snaps.append(_snap(self))
        return out

    def hit(self, r):
        out = orig["hit"](self, r)
        rec.hits.append((r.i, len(out), sum(1 for b in out if self.ref[b] == 0)))
        return out

    def free_request(self, r):
        rec.frees[r.i] = sum(1 for b in self.req_blocks.get(r.i, ()) if self.ref[b] == 1)
        return orig["free_request"](self, r)

    S.add, S.schedule, S.update, K.hit, K.free_request = add, schedule, update, hit, free_request
    try:
        yield
    finally:
        S.add, S.schedule, S.update = orig["add"], orig["schedule"], orig["update"]
        K.hit, K.free_request = orig["hit"], orig["free_request"]


def vectors(trace: RP.Trace, *, R: int, W: int, A: int) -> tuple[dict, list[tuple[RS.State, RS.Inputs, RS.Out]]]:
    """The binding for `trace`'s engine at (R, W, A), and every executed step as (state, inputs, the replay's outputs)."""
    e = trace.engine
    if e.async_scheduling:
        raise ValueError("async scheduling: the step circuit models sync scheduling")
    n = len(trace.requests)
    st = RS.statics(R, W, A, dict(e.__dict__), max(n - 1, 1).bit_length())
    first = trace.first_scheduled()
    rec = _Recorder()

    def decide(i: int, k: int, _o: Any) -> bool:
        rec.asked_pending = True
        return k >= first[i]

    with _instrumented(rec):
        rp = RP.replay(e, trace.requests, decide=decide)
    m = RP.compare(rp.steps, trace.steps)
    if m.first_mismatch is not None:
        raise ValueError(f"the replay diverges from the trace at step {m.first_mismatch}")
    reqs = trace.requests
    out = []
    prev, adds_before = RS.State((), (), e.num_gpu_blocks - 1), 0
    for k, (sr, nxt) in enumerate(zip(rec.steps, rec.snaps)):
        step: RP.Step = sr["step"]
        quiet = not prev.running and not prev.waiting
        forced = int(quiet and adds_before < n)
        first_open = adds_before + forced
        wait_ids = [r.id for r in prev.waiting] or ([adds_before] if forced else [])
        hit, evict = [0] * (W + A), [0] * (W + A)
        for i, h, ev in sr["hits"]:
            c = wait_ids.index(i) if i in wait_ids else W + (i - first_open)
            if 0 <= c < W + A:
                hit[c], evict[c] = h, ev
        ret = [0] * R
        pos_before = [r.id for r in prev.running]
        for i, back in sr["frees"].items():
            p = sr["running"].index(i) if i in sr["running"] else pos_before.index(i)
            if p < R:
                ret[p] = back
        arrivals = tuple(RS.Req(i, reqs[i].prompt_len, reqs[i].output_len) for i in range(adds_before, min(adds_before + A, n)))
        inp = RS.Inputs(arrivals, tuple(sr["asked"]) + (0,) * (A - len(sr["asked"])), tuple(hit), tuple(evict), tuple(ret))
        rows = tuple((sr["running"].index(i), i, int(nn == 1 and start >= reqs[i].prompt_len), start, nn)
                     for i, start, nn in step.rows)
        want = RS.Out(True, rows, nxt, sr["adds"] - adds_before, len(sr["asked"]))
        out.append((prev, inp, want))
        prev, adds_before = nxt, sr["adds"]
    return st, out


def fits(st: dict, state: RS.State, inp: RS.Inputs, want: RS.Out) -> bool:
    R, W = st["R"], st["W"]
    return (len(state.running) <= R and len(state.waiting) <= W and want.state is not None and len(want.state.waiting) <= W
            and all(p < R for p, *_ in want.rows) and want.taken < st["A"])


def window(cases: Sequence[tuple[RS.State, RS.Inputs, RS.Out]], n: int) -> list:
    """The n consecutive cases with the most events: a preemption counts 4, a cache hit 2, an admission or a stop 1."""
    def score(c) -> int:
        s, i, w = c
        before = {r.id for r in s.running}
        return (4 * any(r.id in before for r in w.state.waiting) + 2 * any(i.hit) + (w.taken > 0)
                + (len(w.state.running) < len(s.running) + w.taken))
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


def main(argv: Sequence[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="The replay step circuit's difftest against the replay, step for step.")
    ap.add_argument("traces", nargs="+", type=Path)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--A", type=int, default=16)
    ap.add_argument("--circuit", action="store_true", help="evaluate the Definition on the fitting steps too")
    ap.add_argument("--fixture", type=int, default=0, help="keep the first N fitting steps of each trace")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    report: dict[str, Any] = {"R": a.R, "W": a.W, "A": a.A, "traces": {}}
    fixture = []
    for p in a.traces:
        name = p.stem if p.is_file() else f"{p.resolve().parent.name}/{p.name}"
        tr = RP.read_any(p)
        try:
            st, vs = vectors(tr, R=a.R, W=a.W, A=a.A)
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
