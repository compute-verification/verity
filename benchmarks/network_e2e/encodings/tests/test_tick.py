"""The certified tick (``--admit tick``) in the replay and in advice v1. The serving side's gate
(`verity_vllm.pipeline.serve_admit.TickGate`, through its `install`) in front of the replay's vLLM scheduler, served in
simulated physical time with wall-clock jitter, schedules exactly the batches `replay.replay(tick=...)` computes from the
certified release buckets alone, under sync and async scheduling, and the same jitter without the gate changes them. v1
registered with the tick sends no decision and no order entry, its timing counts code in 0 bits, and its string round
trips; the same trace registered without the tick needs decisions. A tick-served trace-corpus directory gets the same
from v1's table (`v1.main`), which the GLM-5.3 tick load point is measured with. And what the tick refuses."""

from __future__ import annotations

import dataclasses
import heapq
import importlib.util
import json
import random
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity_vllm.pipeline import serve_admit as sa

HERE = Path(__file__).resolve().parents[1]


def _load(name: str):
    key = f"enc_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / f"{name}.py")
        sys.modules[key] = mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[key]


R, V, CV = _load("replay"), _load("v1"), _load("code_v1")

#: τ_b, τ_sat and τ_free, in µs
TICK = (100_000, 30_000, 40_000)
EOS = 2
RUN = "c0ffee"
N = 36
#: the certifier run's origin in the trace-corpus directories, unix seconds
T0 = 1_700_000_000.0


def engine(asy: bool):
    """A small engine that chunks prefills, preempts and caches prefixes."""
    return R.Engine(max_num_seqs=4, max_num_batched_tokens=64, max_model_len=256, num_gpu_blocks=8, block_size=16,
                    enable_prefix_caching=True, async_scheduling=asy)


def plan(seed: int = 7) -> tuple[list, list[float]]:
    """A fixed release sequence, the certified data: N requests released in bursts with idle gaps between (seconds from
    the certifier run's origin), each prompt a 32-token prefix shared with every third request then its own tail, and
    each output ending on EOS or running to max_tokens."""
    rng = random.Random(seed)
    t, ios, times = 0.05, [], []
    for i in range(N):
        t += rng.choice((0.0, 0.0, 0.004, 0.02, 0.06, 0.4, 1.2))
        times.append(round(t, 4))
        prompt = (100 + i % 3,) * 32 + tuple(1000 + 64 * i + j for j in range(rng.randint(1, 30)))
        n = rng.randint(1, 10)
        eos = rng.random() < 0.7
        ios.append(R.RequestIO(prompt_len=len(prompt), max_tokens=n + 3 if eos else n, output_len=n, eos=eos,
                               tokens=prompt + (3,) * (n - 1) + ((EOS,) if eos else (3,))))
    return ios, times


def buckets(times: list[float]) -> list[int]:
    """Each release's bucket, as the network certifier counts it from its run's origin."""
    return [int(t * 1e6) // TICK[0] for t in times]


class Scheduler:
    """The replay's scheduler (`replay._Sched`) as `serve_admit.install` and `TickGate` read vLLM's V1 one: its FCFS
    queues and finished ids, ``add_request``, and ``schedule``, whose output counts the step's tokens. It keeps the steps
    that ran a row and the step count each request was added at."""

    scheduler_config = SimpleNamespace(policy="fcfs")
    vllm_config = SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1))

    def __init__(self, s) -> None:
        self.s, self.steps, self.admitted = s, [], {}

    @property
    def waiting(self):
        return self.s.waiting

    @property
    def skipped_waiting(self):
        return self.s.skipped

    @property
    def running(self):
        return self.s.running

    @property
    def finished_req_ids(self):
        return self.s.finished_ids

    def add_request(self, request) -> None:
        self.admitted[request.i] = len(self.steps)
        self.s.add(request.i)

    def schedule(self) -> SimpleNamespace:
        step, samples = self.s.schedule()
        if step.rows:
            self.steps.append(step)
        return SimpleNamespace(step=step, samples=samples, total_num_scheduled_tokens=sum(n for _, _, n in step.rows))

    def finish_requests(self, ids, status) -> None:
        raise AssertionError("the plan aborts nothing")


@dataclasses.dataclass
class Served:
    steps: list
    admitted: list[int]
    release_t: list[float]          # each step's output back, seconds from the origin
    output_t: list[list[float]]     # each request's outputs, as they left
    emits: list[bool]
    gate: dict


def serve(e, ios, times, seed: int, *, gated: bool = True) -> Served:
    """Serve the plan in simulated physical time through the replay's engine loop (`replay._drive`). Release i is
    certified at times[i], its record reaches the gate up to 20 ms later and the request the engine up to 250 ms later,
    watermark line t comes up to 60 ms after bucket t closes, and each step takes 15 to 45 ms, all sampled from `seed`.
    With `gated`, the gate stands in front of the scheduler and the loop waits while it isn't ready; without, each
    request is added when it reaches the engine."""
    rng = random.Random(seed)
    a = buckets(times)
    now = [0.0]
    s = R._Sched(e, ios)
    sched = Scheduler(s)
    gate = sa.TickGate(sa.Tick(*TICK), hold_s=3600.0, clock=lambda: now[0])
    if gated:
        sa.install(sched, gate, key=lambda rid: rid)
    events: list = []
    for i, t in enumerate(times):
        events.append((t + rng.uniform(0, 0.02), len(events), "release", i))
        events.append((t + rng.uniform(0.02, 0.25), len(events), "request", i))
    last = a[-1] + 2
    for t in range(last + 1):
        events.append(((t + 1) * TICK[0] / 1e6 + rng.uniform(0, 0.06), len(events), "line", t))
    events.append(((last + 2) * TICK[0] / 1e6, len(events), "end", None))
    heapq.heapify(events)
    out = Served([], [], [], [[] for _ in ios], [], {})
    queue: deque = deque()

    def deliver() -> None:
        while events and events[0][0] <= now[0]:
            _, _, kind, x = heapq.heappop(events)
            if kind == "release":
                gate.take({"key": f"r{x}", "run": RUN, "n": x, "bucket": a[x]})
            elif kind == "request":
                sched.add_request(SimpleNamespace(request_id=f"r{x}", i=x))
            elif kind == "line":
                gate.take({"run": RUN, "t": x, "n": sum(b <= x for b in a)})
            else:
                gate.take({"run": RUN, "end": True})

    def top() -> bool:
        while True:
            deliver()
            ok = not gated or gate.ready(sched)
            if ok and (s.has_requests() or queue):
                return True
            if not events:
                assert ok and not gate.held, "the engine waits on the gate with nothing left to come"
                return False
            now[0] = events[0][0]

    def schedule():
        o = sched.schedule()
        return o.step, o.samples

    def executed(step, samples) -> None:
        before = {i: s.reqs[i].n_out for i, _, _ in step.rows}
        s.update(step, samples)
        if step.rows:
            now[0] += rng.uniform(0.015, 0.045)
            out.release_t.append(now[0])
            out.emits.append(any(s.reqs[i].n_out > m for i, m in before.items()))
            for i, m in before.items():
                out.output_t[i] += [now[0]] * (s.reqs[i].n_out - m)

    R._drive(s, queue, top=top, schedule=schedule, executed=executed)
    out.steps, out.admitted, out.gate = sched.steps, [sched.admitted[i] for i in range(len(ios))], gate.summary()
    return out


def trace_of(e, ios, times, run: Served):
    """The served run as a tick-admitted run's trace: its requests in release order, each output's time, and the
    certified releases, counted from the origin."""
    return R.Trace(engine=e, ids=[f"r{i}" for i in range(N)], requests=list(ios), before_step=run.admitted,
                   steps=run.steps, emits=run.emits, ingress_t=list(times), release_t=run.release_t, output_t=run.output_t,
                   releases=R.Releases(R.Tick(*TICK), 0.0, tuple(range(N)), tuple(buckets(times))))


@pytest.mark.parametrize("asy", (False, True), ids=("sync", "async"))
def test_the_gate_schedules_the_tick_replay(asy):
    e, (ios, times) = engine(asy), plan()
    ref = R.replay(e, ios, tick=R.Tick(*TICK), released=buckets(times))
    runs = [serve(e, ios, times, seed) for seed in (1, 2, 3)]
    for run in runs:
        assert run.steps == ref.steps and run.admitted == ref.admitted
        g = run.gate
        assert (g["admitted"], g["opened"], g["ended"]) == (N, None, True)
        assert not any(g[k] for k in ("unordered", "stale", "late", "skipped", "aborted", "past_end", "bucket_mismatch",
                                      "watermark_errors"))
        assert g["waits"]["n"] > 0 and g["max_held"] > 1
    assert len({tuple(r.release_t) for r in runs}) == len(runs)
    # the plan exercises the rule: the engine idles (T jumps) and saturates (a request waits in the queue)
    first = [min(k for k, st in enumerate(ref.steps) if any(r[0] == i for r in st.rows)) for i in range(N)]
    assert sum(ref.forced) > 1 and any(f > a for f, a in zip(first, ref.admitted))
    assert any(st.preempted for st in ref.steps)
    plain = [serve(e, ios, times, seed, gated=False).steps for seed in (1, 2)]
    assert plain[0] != plain[1] and ref.steps not in plain


def tick_case(asy: bool):
    e, (ios, times) = engine(asy), plan()
    tr = trace_of(e, ios, times, serve(e, ios, times, seed=1))
    rec = V.record_of(tr, (EOS,))
    return e, rec, V.advice_of(tr, rec), V.Registered(e, (EOS,), row_order="scheduling", T=10, tick=R.Tick(*TICK))


@pytest.mark.parametrize("asy", (False, True), ids=("sync", "async"))
def test_v1_sends_no_timing_under_the_tick(asy):
    e, rec, adv, reg = tick_case(asy)
    wins = V.residual(adv, rec, reg)
    assert len(wins) == V.windows(rec, reg.T) > 1
    assert not any(w.arrived or w.order for w in wins) and any(w.hits for w in wins)
    data = V.encode(adv, rec, reg)
    assert V.decode(data, rec, reg) == adv
    assert sum(V.bits_by_window(data, rec, reg).values()) == sum(V._string([CV.Window(hits=w.hits) for w in wins],
                                                                           reg)[1])


def test_without_the_tick_the_same_run_needs_decisions():
    """Registered without the tick, the same sync run's advice sends decisions, and more hits (the lookups of the
    questions its open steps ask), so more bits; its string isn't one the tick's registration reads. Under async
    scheduling the arrival decisions, indexed by step, can't place an admission that follows a ``schedule()`` that ran no
    row at the same step index, which the tick made here, so without the tick that run has no v1 advice at all."""
    _, rec, adv, reg = tick_case(False)
    plain = dataclasses.replace(reg, tick=None)
    pw = V.residual(adv, rec, plain)
    assert sum(len(w.arrived) for w in pw) > 0
    assert sum(len(w.hits) for w in V.residual(adv, rec, reg)) < sum(len(w.hits) for w in pw)
    data, pdata = V.encode(adv, rec, reg), V.encode(adv, rec, plain)
    assert V.decode(pdata, rec, plain) == adv
    assert sum(V.bits_by_window(pdata, rec, plain).values()) > sum(V.bits_by_window(data, rec, reg).values())
    with pytest.raises(V.V1Error):
        V.decode(pdata, rec, reg)
    _, rec, adv, reg = tick_case(True)
    with pytest.raises(V.V1Error, match="schedule isn't the replay's"):
        V.encode(adv, rec, dataclasses.replace(reg, tick=None))


def test_a_count_capped_at_0_costs_nothing():
    """`TICK_CAPS` caps the decisions and the order entries at 0, so their counts code in 0 bits: a window whose every
    count is capped at 0 costs nothing, and under v1's own caps an empty window costs bits."""
    zero = dataclasses.replace(V.TICK_CAPS, hits=0, resumes=0, lag=0)
    assert (V.TICK_CAPS.arrived, V.TICK_CAPS.order) == (0, 0)
    assert list(CV.encode([CV.Window()] * 3, CV.Book.uniform(zero)).lens) == [0, 0, 0]
    assert all(n > 0 for n in CV.encode([CV.Window()] * 3, CV.Book.uniform(V.CAPS)).lens)
    assert sum(CV.encode([CV.Window()] * 3, CV.Book.uniform(V.TICK_CAPS)).lens) < \
        sum(CV.encode([CV.Window()] * 3, CV.Book.uniform(V.CAPS)).lens)


def write_corpus(d: Path, e, ios, times, run: Served, *, tick: bool) -> Path:
    """The served run as a trace-corpus directory (`replay.read_corpus_trace`'s), each step's rows in the runner's batch
    order (a stable sort by length), and with `tick` its releases and meta.json's ``tick``."""
    d.mkdir()
    a = buckets(times)
    argv = ["--max-num-seqs", str(e.max_num_seqs), "--max-num-batched-tokens", str(e.max_num_batched_tokens),
            "--max-model-len", str(e.max_model_len), "--enable-prefix-caching",
            *(() if e.async_scheduling else ("--no-async-scheduling",))]
    meta: dict = {"server_argv": argv}
    if tick:
        meta["tick"] = {"bucket_us": TICK[0], "sat_us": TICK[1], "free_us": TICK[2], "origin_unix": T0}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "server.log").write_text(f"GPU KV cache size: {e.num_gpu_blocks * e.block_size:,} tokens\n")

    def row(i: int, start: int, n: int, **kw) -> dict:
        return {"id": f"r{i}", "num_computed_tokens": start, "num_scheduled_tokens": n, **kw}

    lines = [{"total_num_scheduled_tokens": sum(n for *_, n in st.rows),
              "new": [row(*r) for r in st.rows if r[0] in st.new],
              "cached": [row(*r, resumed=r[0] in st.resumed) for r in st.rows if r[0] not in st.new],
              "batch": [f"r{i}" for i, _, _ in sorted(st.rows, key=lambda r: r[2])],
              "preempted": [f"r{i}" for i in st.preempted], "sampled": [emit], "t_output": T0 + t}
             for st, t, emit in zip(run.steps, run.release_t, run.emits)]
    (d / "steps.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    reqs = []
    for i, io in enumerate(ios):
        r = {"id": f"r{i}", "prompt_len": io.prompt_len, "max_tokens": io.max_tokens, "output_len": io.output_len,
             "eos": io.eos, "finish_reason": "stop" if io.eos else "length", "prompt_ids": list(io.tokens[:io.prompt_len]),
             "output_ids": list(io.tokens[io.prompt_len:]), "send_unix": T0 + times[i]}
        reqs.append({**r, "release": i, "bucket": a[i]} if tick else r)
    (d / "requests.jsonl").write_text("".join(json.dumps(x) + "\n" for x in reqs))
    return d


@pytest.mark.parametrize("asy", (False, True), ids=("sync", "async"))
def test_the_corpus_table_charges_no_timing_under_the_tick(tmp_path, capsys, asy):
    """v1's table over two tick-served runs of the plan with different jitter, and under sync scheduling the first one's
    view without the tick (`test_without_the_tick_the_same_run_needs_decisions`: under async it has no advice)."""
    e, (ios, times) = engine(asy), plan()
    runs = [serve(e, ios, times, seed) for seed in (1, 2)]
    dirs = [write_corpus(tmp_path / f"tick{k}", e, ios, times, run, tick=True) for k, run in enumerate(runs)]
    tr = R.read_corpus_trace(dirs[0])
    assert tr.releases == R.Releases(R.Tick(*TICK), T0, tuple(range(N)), tuple(buckets(times)))
    assert [st.rows for st in tr.steps] == [sorted(st.rows, key=lambda r: r[2]) for st in runs[0].steps]
    labels = [f"tick{k}={d}" for k, d in enumerate(dirs)]
    if not asy:
        plain = write_corpus(tmp_path / "plain", e, ios, times, runs[0], tick=False)
        assert R.read_corpus_trace(plain).releases is None
        labels.append(f"plain={plain}")
    out = tmp_path / "table.json"
    assert V.main([*labels, "--stops", str(EOS), "--T", "10", "--jobs", "1", "--out", str(out)]) == 0
    assert "| tick0 (certified tick) |" in capsys.readouterr().out
    rows = {r["trace"]: r for r in json.loads(out.read_text())["rows"]}
    for t in (rows["tick0"], rows["tick1"]):
        assert t["tick"] == {"bucket_us": TICK[0], "sat_us": TICK[1], "free_us": TICK[2]}
        assert (t["decisions"], t["order_entries"], t["steps"]) == (0, 0, rows["tick0"]["steps"])
        for book in ("held_out", "uniform"):
            assert t[book]["timing_bits"] == 0 and t[book]["round_trip"]
    if not asy:
        t, p = rows["tick0"], rows["plain"]
        assert p["tick"] is None and p["decisions"] > 0 and t["hits_sent"] < p["hits_sent"]
        assert all(p[book]["timing_bits"] > 0 and p[book]["round_trip"] for book in ("held_out", "uniform"))
        assert t["uniform"]["bits"] < p["uniform"]["bits"]


def test_the_table_reads_a_tick_trace_on_its_own_buckets(tmp_path):
    """A tick-served trace whose τ_b isn't 100 ms: v1's table (`_trace`) counts its releases on its own tick's buckets,
    where its record at 100 ms buckets is refused."""
    e, (ios, times) = engine(False), plan()
    d = write_corpus(tmp_path / "tick", e, ios, times, serve(e, ios, times, seed=1), tick=True)
    meta = json.loads((d / "meta.json").read_text())
    meta["tick"]["bucket_us"] = 250_000
    (d / "meta.json").write_text(json.dumps(meta))
    a = [int(t * 1e6) // 250_000 for t in times]
    reqs = [json.loads(x) for x in (d / "requests.jsonl").read_text().splitlines()]
    (d / "requests.jsonl").write_text("".join(json.dumps({**r, "bucket": b}) + "\n" for r, b in zip(reqs, a)))
    tr, rec, _ = V._trace(str(d), (EOS,))
    assert tr.releases.tick.bucket_us == 250_000
    assert [q.end for q in rec.requests] == [b - a[0] for b in a]
    with pytest.raises(V.V1Error, match="250000 µs"):
        V.record_of(tr, (EOS,))


def test_what_the_tick_refuses():
    e, (ios, times) = engine(False), plan()
    a, tk = buckets(times), R.Tick(*TICK)
    with pytest.raises(ValueError, match="exactly one"):
        R.replay(e, ios, tick=tk, released=a, decide=lambda i, k, o: True)
    with pytest.raises(ValueError, match="release buckets with the tick"):
        R.replay(e, ios, tick=tk)
    with pytest.raises(ValueError, match="nondecreasing"):
        R.replay(e, ios, tick=tk, released=a[::-1])
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            R.Tick(bad, 1, 1)
    assert dataclasses.asdict(tk) == sa.Tick(*TICK).record()
    assert [tk.close(b) for b in range(4)] == [sa.Tick(*TICK).close(b) for b in range(4)]
    with pytest.raises(V.V1Error, match="arrived and order are 0"):
        V.Registered(e, (EOS,), tick=tk, book=CV.Book.uniform(V.CAPS))
    with pytest.raises(V.V1Error, match="replay.Tick"):
        V.Registered(e, (EOS,), tick=sa.Tick(*TICK))
    V.Registered(e, (EOS,), tick=tk, book=CV.Book.uniform(V.TICK_CAPS))
    _, rec, adv, reg = tick_case(False)
    q0, last = rec.requests[0], max(q.end for q in rec.requests) + 1
    later = V.Record((dataclasses.replace(q0, end=last, release=tuple(max(b, last) for b in q0.release)),
                      *rec.requests[1:]))
    with pytest.raises(V.V1Error, match="release order"):
        V.encode(adv, later, reg)
