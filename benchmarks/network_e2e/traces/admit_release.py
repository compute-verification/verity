"""What admitting requests in the warden's release order would cost, estimated offline from trace-corpus traces.

The design: serve-openai adds each request to the engine at the first bucket boundary after its release (the warden's
release bucket a, so at origin + (a + 1) tau), in release order (a, then the warden's ordinal). The engine's arrival order
is then the certified release order, and the replay's order bits are 0.

A request can't be added before the frontend has it ready (its recorded engine add), and in release order none goes
before an earlier-released one. So the counterfactual add of the k-th request in release order is

    hold:     max(origin + (a_k + 1) tau, max over j <= k of ready_j)
    aligned:  that, rounded up to the next bucket boundary (every add on a boundary)

`simulate` runs the replay's scheduler model (`replay._Sched` under the engine core's loop, `replay._drive`) on a clock:
before each `schedule` the requests whose add time has passed are added; each executed step takes the recorded step's
time at the same index (output to output, the engine being busy throughout), corrected by a fitted per-token and per-row
cost for any change in the batch; step k is scheduled the recorded lag after step k-2's output (async scheduling, one
step in flight), and a step from an empty pipeline pays the trace's cold-start latency. Run once with the recorded adds
(`base`, checked against the trace's own steps and first-token times) and once with the counterfactual adds (`hold`,
`aligned`); the price is the difference between the two simulations, so the model's error mostly cancels.

A trace without a warden takes the client's send bucket as its release and the send order as its ordinal, and the
frontend's arrival stamp plus `--add-lag-ms` as its ready time (those corpora don't record the engine's add).

Usage: python3 -P admit_release.py [--replay PATH] [--add-lag-ms NAME=MS ...] --out DIR NAME=TRACE_DIR[:ORDER_DIR] ...
(ORDER_DIR, the trace view whose ingress is the certified order for the replay's order bits; default the trace itself).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPLAY = HERE.parent / "encodings" / "replay.py"
WINDOW_BUCKETS = 600


def load_replay(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("enc_replay", path)
    assert spec is not None and spec.loader is not None
    sys.modules["enc_replay"] = mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"n": 0, "min": None, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    s = sorted(xs)

    def at(p: float) -> float:
        return s[min(len(s) - 1, math.ceil(p * len(s)) - 1)]

    return {"n": len(s), "min": s[0], "mean": sum(s) / len(s), "p50": at(0.5), "p95": at(0.95), "p99": at(0.99),
            "max": s[-1]}


def lstsq(X: list[list[float]], y: list[float]) -> list[float]:
    """Least squares by the normal equations (a few columns)."""
    m = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(m)] for i in range(m)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(m)]
    for c in range(m):
        p = max(range(c, m), key=lambda k: abs(A[k][c]))
        A[c], A[p], b[c], b[p] = A[p], A[c], b[p], b[c]
        for k in range(m):
            if k != c and A[c][c]:
                f = A[k][c] / A[c][c]
                A[k] = [a - f * x for a, x in zip(A[k], A[c])]
                b[k] -= f * b[c]
    return [b[i] / A[i][i] if A[i][i] else 0.0 for i in range(m)]


def features(rows: list[tuple[int, int, int]]) -> list[float]:
    return [1.0, float(sum(n for _, _, n in rows)), float(len(rows))]


def fit(X: list[list[float]], y: list[float]) -> list[float]:
    """Least squares with every per-token and per-row cost kept nonnegative (a negative one is dropped and the rest
    refit)."""
    keep = list(range(len(X[0])))
    while True:
        c = lstsq([[r[j] for j in keep] for r in X], y)
        neg = [j for j, v in zip(keep, c) if j and v < 0]
        if not neg:
            out = [0.0] * len(X[0])
            for j, v in zip(keep, c):
                out[j] = v
            return out
        keep.remove(neg[0])


@dataclass
class Clock:
    """The recorded engine's timing by step index (steps that scheduled tokens): each step's time (output to output),
    the lag from step k-2's output to step k's schedule, the cold-start latency, and the fitted cost per step."""

    period: list[float]
    lag: list[float]
    cold: float
    wake: float                     # a quiet engine: from the add that wakes it to its schedule
    coef: list[float]
    busy: int
    idle: int

    def cost(self, rows: list[tuple[int, int, int]]) -> float:
        return sum(c * f for c, f in zip(self.coef, features(rows)))

    def step_time(self, k: int, rows: list[tuple[int, int, int]], recorded: list[tuple[int, int, int]] | None) -> float:
        if k < len(self.period) and recorded is not None:
            return max(1e-3, self.period[k] + self.cost(rows) - self.cost(recorded))
        return max(1e-3, self.cost(rows))

    def lag_at(self, k: int) -> float:
        return self.lag[k] if k < len(self.lag) else sorted(self.lag)[len(self.lag) // 2]


def clock(raw: list[dict[str, Any]], steps: list[Any], first_add: float) -> Clock:
    t = [float(s["t"]) for s in raw]
    out = [float(s["t_output"]) for s in raw]
    busy = [k for k in range(1, len(raw)) if t[k] < out[k - 1]]
    coef = fit([features(steps[k].rows) for k in busy], [out[k] - out[k - 1] for k in busy])
    tmp = Clock([], [], 0.0, 0.0, coef, len(busy), len(raw) - 1 - len(busy))
    period = [tmp.cost(steps[0].rows)] + [out[k] - max(out[k - 1], t[k]) for k in range(1, len(raw))]
    lag = [0.0, max(0.0, t[1] - t[0])] + [max(0.0, t[k] - out[k - 2]) for k in range(2, len(raw))]
    return Clock(period, lag, max(0.0, out[0] - t[0] - period[0]), max(0.0, t[0] - first_add), coef, tmp.busy, tmp.idle)


@dataclass
class Sim:
    steps: list[Any]
    end: list[float]                    # per step, its output's time
    emitted: list[int]                  # per step, outputs it emitted
    first: list[float | None]           # per request, its first output's time
    done: list[float | None]            # per request, its last output's time
    visible: list[int | None]           # per request, the step it was first visible at


def simulate(rp: Any, engine: Any, requests: list[Any], order: list[int], ready: list[float], ck: Clock,
             recorded: list[Any] | None) -> Sim:
    """The engine core's loop on a clock: `order[k]` is added before the first `schedule` at or after `ready[k]`."""
    if not engine.async_scheduling:
        raise ValueError("the clock models async scheduling (step k scheduled after step k-2's output) only")
    if any(ready[k] > ready[k + 1] for k in range(len(ready) - 1)):
        raise ValueError("add times must be nondecreasing in add order")
    s = rp._Sched(engine, requests)
    q: deque = deque()
    n = len(order)
    out = Sim([], [], [], [None] * len(requests), [None] * len(requests), [None] * len(requests))
    st = {"nxt": 0, "now": ready[0], "sched": ready[0], "gpu": -math.inf}
    limit = 4 * (sum(r.prompt_len + r.output_len for r in requests) + n) + 64

    def top() -> bool:
        k = len(out.steps)
        if st["nxt"] < n and not s.has_requests() and not q and ready[st["nxt"]] >= st["now"]:
            st["now"] = ready[st["nxt"]]
            st["sched"] = st["now"] + ck.wake
        else:
            st["sched"] = st["now"] + ck.lag_at(k)
        while st["nxt"] < n and ready[st["nxt"]] <= st["sched"]:
            i = order[st["nxt"]]
            s.add(i)
            out.visible[i] = k
            st["nxt"] += 1
        if not s.has_requests() and not q:
            return st["nxt"] < n
        if k > limit:
            raise RuntimeError(f"no progress after {limit} steps")
        return True

    def schedule() -> tuple[Any, dict[int, bool]]:
        step, samples = s.schedule(None)
        if step.rows:
            k = len(out.steps)
            rec = recorded[k].rows if recorded is not None and k < len(recorded) else None
            d = ck.step_time(k, step.rows, rec)
            start = max(st["gpu"], st["sched"]) if q else st["sched"] + ck.cold
            step._end = st["gpu"] = start + d
            out.steps.append(step)
            out.end.append(step._end)
        return step, samples

    def executed(step: Any, samples: dict[int, bool]) -> None:
        if step.rows:
            st["now"] = max(st["now"], step._end)
        before = {i: s.reqs[i].n_out for i, _, _ in step.rows}
        for i in s.update(step, samples):
            out.done[i] = step._end
        if step.rows:
            got = 0
            for i, m in before.items():
                if s.reqs[i].n_out > m:
                    got += s.reqs[i].n_out - m
                    if m == 0:
                        out.first[i] = step._end
            out.emitted.append(got)

    rp._drive(s, q, top=top, schedule=schedule, executed=executed)
    if st["nxt"] < n or any(f is None for f in out.done):
        raise RuntimeError("the simulation ended with requests unfinished")
    return out


def as_trace(rp: Any, trace: Any, otrace: Any, order: list[int], sim: Sim) -> Any:
    """The simulated run as a replay `Trace`: its requests in the order they reached the engine (`order`), the certified
    ingress of `otrace`, and each step's output time as its release (egress at the engine step)."""
    inv = {i: j for j, i in enumerate(order)}
    steps = [rp.Step(rows=[(inv[i], s, n) for i, s, n in st.rows], new=[inv[i] for i in st.new],
                     resumed=[inv[i] for i in st.resumed], preempted=[inv[i] for i in st.preempted],
                     cache_hit={inv[i]: k for i, k in st.cache_hit.items()}) for st in sim.steps]
    out = rp.Trace(engine=trace.engine, ids=[trace.ids[i] for i in order], requests=[trace.requests[i] for i in order],
                   before_step=[], steps=steps, emits=[e > 0 for e in sim.emitted],
                   ingress_t=[otrace.ingress_t[i] for i in order], release_t=list(sim.end),
                   header={"kind": "header", "schema": "admit-release/simulated", "config": dict(trace.engine.__dict__)})
    out.before_step = out.first_scheduled()
    return out


def residual(rp: Any, tr: Any) -> dict[str, Any]:
    """`replay.table` on a trace: order bits, and the residual advice per window (the adaptive code with egress
    contexts, egress at the engine step)."""
    tb = rp.table(tr, bucket_s=0.1, window_buckets=WINDOW_BUCKETS)
    return {"order_bits": tb["order_bits"], "matched": tb["match_before_step"]["matched"], "steps": tb["steps"],
            "decoded": all(v["first_mismatch"] is None for v in tb["decoded"].values()),
            "windows": [{"window": w["window"], "decisions": w["decisions"], "residual_bits": w["adaptive_egress_bits"],
                         "residual_framed": rp.framed_bits(w["adaptive_egress_bits"])} for w in tb["windows"]]}


def order_costs(rp: Any, trace: Any) -> list[float]:
    """`replay.order_bits`, per request (engine order): the sum is the trace's order bits."""
    n = len(trace.requests)
    left = sorted(range(n), key=lambda i: (trace.ingress_t[i], i))
    model, out = rp._KT(), []
    for i in range(n):
        rank = left.index(i)
        f0, f1 = model.freqs(())
        c = -math.log2((f1 if rank == 0 else f0) / (f0 + f1))
        model.update((), rank == 0)
        if rank:
            c += rp.gamma_bits(rank)
        left.pop(rank)
        out.append(c)
    return out


@dataclass
class Release:
    origin: float                       # the bucket grid's origin (unix s)
    tau: float
    bucket: list[int]                   # per request (engine order), its release bucket
    ordinal: list[int]
    ready: list[float]                  # per request, when the frontend had it ready to add
    source: str
    egress_first: list[float] | None = None     # behind a warden: when its first egress cell leaves (a + D)


def release_of(d: Path, ids: list[str], add_lag_s: float) -> Release:
    meta = json.loads((d / "meta.json").read_text())
    reqs = {r["id"]: r for r in jsonl(d / "requests.jsonl")}
    w = meta.get("warden")
    if w:
        tau, o, D = w["params"]["bucket_ms"] / 1000, float(w["origin_unix"]), int(w["params"]["delay"])
        if not str(w.get("egress_base", "")).startswith("release"):
            raise ValueError(f"{d}: the warden's egress counts from {w.get('egress_base')!r}, not the release")
        a = [int(reqs[i]["warden"]["released"]) for i in ids]
        return Release(o, tau, a, [int(reqs[i]["warden"]["ordinal"]) for i in ids],
                       [float(reqs[i]["engine_add_unix"]) for i in ids],
                       f"the warden's release bucket and ordinal ({w['egress_base']}); ready = the engine's add",
                       [o + (b + D) * tau for b in a])
    tau = 0.1
    sends = {i: float(reqs[i]["send_unix"]) for i in ids}
    g0 = math.floor(min(sends.values()) / tau + 1e-9) * tau
    rank = {i: k for k, i in enumerate(sorted(ids, key=lambda i: (sends[i], i)))}
    return Release(g0, tau, [math.floor((sends[i] - g0) / tau + 1e-9) for i in ids], [rank[i] for i in ids],
                   [float(reqs[i]["arrival"]) + add_lag_s for i in ids],
                   f"no warden: the client's send bucket and send order; ready = the frontend's arrival + {add_lag_s * 1000:g} ms, "
                   "kept after the last open step that passed it over and before its first scheduling")


def pin(rp: Any, trace: Any, raw: list[dict[str, Any]], ready: list[float]) -> tuple[list[float], int]:
    """A trace without recorded adds: each estimated add moved into the interval the trace fixes, after the schedule
    of the last open step that passed the request over and before the schedule of its first scheduled step (so the
    clocked baseline reproduces the trace). Returns the adds and how many moved."""
    ds, _ = rp.decisions(trace)
    lo = [-math.inf] * len(ready)
    for dc in ds:
        if not dc.arrived:
            lo[dc.request] = max(lo[dc.request], float(raw[dc.step]["t"]))
    out, moved = [], 0
    for i, t in enumerate(ready):
        hi = float(raw[trace.before_step[i]]["t"]) - 1e-6
        v = min(max(t, lo[i] + 1e-6), hi)
        moved += v != t
        out.append(v)
    return out, moved


def counterfactual(rel: Release, aligned: bool) -> tuple[list[int], list[float]]:
    order = sorted(range(len(rel.bucket)), key=lambda i: (rel.bucket[i], rel.ordinal[i]))
    ready, hi = [], -math.inf
    for i in order:
        hi = max(hi, rel.ready[i])
        t = max(rel.origin + (rel.bucket[i] + 1) * rel.tau, hi)
        if aligned:
            t = rel.origin + math.ceil((t - rel.origin) / rel.tau - 1e-9) * rel.tau
        ready.append(t)
    return order, ready


def throughput(sim: Sim, requests: list[Any], t0: float) -> dict[str, float]:
    t1 = max(d for d in sim.done if d is not None)
    toks = sum(r.output_len for r in requests)
    return {"span_s": t1 - t0, "req_per_s": len(requests) / (t1 - t0), "tok_per_s": toks / (t1 - t0), "steps": len(sim.steps)}


def window_rate(sim: Sim, t0: float, t1: float) -> float | None:
    toks = sum(e for e, t in zip(sim.emitted, sim.end) if t0 <= t < t1)
    last = min(t1, max(sim.end))
    return toks / (last - t0) if last > t0 else None


def ms(q: dict[str, float | None]) -> dict[str, float | None]:
    return {k: (round(v * 1000, 2) if isinstance(v, float) and k != "n" else v) for k, v in q.items()}


def analyse(rp: Any, name: str, d: Path, order_dir: Path, add_lag_s: float,
            advice: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trace = rp.read_corpus_trace(d)
    raw = [s for s in jsonl(d / "steps.jsonl") if s["total_num_scheduled_tokens"]]
    rel = release_of(d, trace.ids, add_lag_s)
    moved = None
    if not json.loads((d / "meta.json").read_text()).get("warden"):
        rel.ready, moved = pin(rp, trace, raw, rel.ready)
    ck = clock(raw, trace.steps, min(rel.ready))
    n = len(trace.requests)
    base_order = list(range(n))
    base_ready = list(rel.ready)
    for k in range(1, n):
        base_ready[k] = max(base_ready[k], base_ready[k - 1])
    base = simulate(rp, trace.engine, trace.requests, base_order, base_ready, ck, trace.steps)
    match = rp.compare(base.steps, trace.steps)
    kept = trace.kept_steps()
    rec_first = [float(raw[ks[0]]["t_output"]) for ks in kept]
    rec_done = [float(raw[ks[-1]]["t_output"]) for ks in kept]
    t0 = min(rel.ready)
    rec_span = max(rec_done) - t0
    toks = sum(r.output_len for r in trace.requests)
    otrace = rp.read_corpus_trace(order_dir)
    if otrace.ids != trace.ids:
        raise ValueError(f"{order_dir}: not the same engine order as {d}")
    ocost = order_costs(rp, otrace)
    rw, _ = rp.windows(otrace, bucket_s=0.1, window_buckets=WINDOW_BUCKETS)
    first_bucket = min(rel.bucket)
    res: dict[str, Any] = {
        "trace": name, "dir": str(d), "order_view": str(order_dir), "requests": n, "release": rel.source,
        "adds_moved_into_the_trace": moved,
        "engine": dict(trace.engine.__dict__),
        "clock": {"coef_ms": [round(c * 1000, 5) for c in ck.coef], "features": ["1", "tokens", "rows"],
                  "busy_steps": ck.busy, "idle_steps": ck.idle, "cold_start_ms": round(ck.cold * 1000, 2)},
        "order_bits": round(sum(ocost), 2),
        "recorded": {"steps": len(trace.steps), "span_s": rec_span, "req_per_s": n / rec_span, "tok_per_s": toks / rec_span},
        "base": {**throughput(base, trace.requests, t0), "steps_matched": match.matched, "steps_compared": match.steps,
                 "first_mismatch": match.first_mismatch,
                 "first_token_error_ms": ms(quantiles([abs(a - b) for a, b in zip(base.first, rec_first)])),
                 "ttft_from_add_ms": ms(quantiles([f - r for f, r in zip(base.first, base_ready)])),  # type: ignore[operator]
                 "egress_margin_ms": None if rel.egress_first is None else
                 ms(quantiles([e - f for e, f in zip(rel.egress_first, base.first)]))},  # type: ignore[operator]
        "variants": {},
    }
    per_req: list[dict[str, Any]] = [{"trace": name, "id": trace.ids[i], "engine_index": i, "release_bucket": rel.bucket[i],
                                      "ordinal": rel.ordinal[i], "window": rw[i], "order_bits": round(ocost[i], 3),
                                      "ready": rel.ready[i], "first_recorded": rec_first[i], "first_base": base.first[i],
                                      "done_base": base.done[i], "visible_base": base.visible[i]}
                                     for i in range(n)]
    windows = sorted(set(rw))
    if advice:
        res["recorded"]["advice"] = residual(rp, otrace)
    for variant in ("hold", "aligned"):
        order, ready = counterfactual(rel, variant == "aligned")
        cf = simulate(rp, trace.engine, trace.requests, order, ready, ck, trace.steps)
        add = {i: t for i, t in zip(order, ready)}
        held = [add[i] - base_ready[i] for i in range(n)]
        dttft = [cf.first[i] - base.first[i] for i in range(n)]  # type: ignore[operator]
        dlast = [cf.done[i] - base.done[i] for i in range(n)]  # type: ignore[operator]
        tp_b, tp_c = throughput(base, trace.requests, t0), throughput(cf, trace.requests, t0)
        boundary = [rel.origin + (rel.bucket[i] + 1) * rel.tau for i in range(n)]
        v: dict[str, Any] = {
            "add_delay_ms": ms(quantiles(held)),
            "frontend_late_for_boundary": sum(1 for i in range(n) if rel.ready[i] > boundary[i]),
            "held_behind_earlier_release": sum(1 for i in range(n) if add[i] > max(boundary[i], rel.ready[i]) + 1e-9),
            "added_ttft_ms": ms(quantiles(dttft)),
            "added_ttft_later": sum(1 for x in dttft if x > 1e-9),
            "added_ttft_earlier": sum(1 for x in dttft if x < -1e-9),
            "added_completion_ms": ms(quantiles(dlast)),
            "egress_margin_ms": None if rel.egress_first is None else
            ms(quantiles([e - f for e, f in zip(rel.egress_first, cf.first)])),  # type: ignore[operator]
            "throughput": {"base": tp_b, "cf": tp_c,
                           "req_per_s_change_pct": 100 * (tp_c["req_per_s"] / tp_b["req_per_s"] - 1),
                           "tok_per_s_change_pct": 100 * (tp_c["tok_per_s"] / tp_b["tok_per_s"] - 1)},
            "steps": len(cf.steps),
            "windows": [],
        }
        for w in windows:
            rs = [i for i in range(n) if rw[i] == w]
            ws = rel.origin + (first_bucket + w * WINDOW_BUCKETS) * rel.tau
            we = ws + WINDOW_BUCKETS * rel.tau
            rb, rc = window_rate(base, ws, we), window_rate(cf, ws, we)
            v["windows"].append({"window": w, "requests": len(rs), "order_bits": round(sum(ocost[i] for i in rs), 2),
                                 "added_ttft_ms": ms(quantiles([dttft[i] for i in rs])),
                                 "add_delay_ms": ms(quantiles([held[i] for i in rs])),
                                 "tok_per_s_base": rb, "tok_per_s_cf": rc,
                                 "tok_per_s_change_pct": None if not rb or rc is None else 100 * (rc / rb - 1)})
        if advice:
            v["advice"] = residual(rp, as_trace(rp, trace, otrace, order, cf))
        res["variants"][variant] = v
        for i in range(n):
            per_req[i][f"add_{variant}"] = add[i]
            per_req[i][f"first_{variant}"] = cf.first[i]
            per_req[i][f"done_{variant}"] = cf.done[i]
            per_req[i][f"visible_{variant}"] = cf.visible[i]
    return res, per_req


def fmt(x: float | None, nd: int = 1) -> str:
    return "-" if x is None else f"{x:,.{nd}f}"


def markdown(results: list[dict[str, Any]]) -> str:
    out = [("| trace | order bits removed | add delay p50 / p95 / max (ms) | added TTFT p50 / p95 / p99 / max (ms) | "
           "TTFT later / earlier | req/s base -> cf | tok/s base -> cf | change | steps base -> cf |"),
           "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for variant, v in r["variants"].items():
            a, t, tp = v["add_delay_ms"], v["added_ttft_ms"], v["throughput"]
            out.append(f"| {r['trace']} ({variant}) | {r['order_bits']:,.2f} | {fmt(a['p50'])} / {fmt(a['p95'])} / {fmt(a['max'])} | "
                       f"{fmt(t['p50'])} / {fmt(t['p95'])} / {fmt(t['p99'])} / {fmt(t['max'])} | "
                       f"{v['added_ttft_later']} / {v['added_ttft_earlier']} | "
                       f"{tp['base']['req_per_s']:.2f} -> {tp['cf']['req_per_s']:.2f} | {tp['base']['tok_per_s']:,.0f} -> "
                       f"{tp['cf']['tok_per_s']:,.0f} | {tp['tok_per_s_change_pct']:+.2f}% | {tp['base']['steps']:,} -> {v['steps']:,} |")
    out += ["", ("| trace | window | requests | order bits removed | add delay p50 / max (ms) | added TTFT p50 / p95 / p99 / max (ms) "
            "| tok/s base -> cf | change |"), "|---|---|---|---|---|---|---|---|"]
    for r in results:
        v = r["variants"]["hold"]
        for w in v["windows"]:
            a, t = w["add_delay_ms"], w["added_ttft_ms"]
            out.append(f"| {r['trace']} | {w['window']} | {w['requests']} | {w['order_bits']:,.2f} | {fmt(a['p50'])} / {fmt(a['max'])} | "
                       f"{fmt(t['p50'])} / {fmt(t['p95'])} / {fmt(t['p99'])} / {fmt(t['max'])} | {fmt(w['tok_per_s_base'], 0)} -> "
                       f"{fmt(w['tok_per_s_cf'], 0)} | {fmt(w['tok_per_s_change_pct'], 2)}% |")
    out += ["", ("| trace | TTFT from the engine's add, base p50 / p95 / max (ms) | frontend not ready at the boundary "
            "| held behind an earlier release (hold) | first token to the warden's first egress cell, min base -> hold "
            "-> aligned (ms) |"), "|---|---|---|---|---|"]
    for r in results:
        b, h, al = r["base"], r["variants"]["hold"], r["variants"]["aligned"]
        t = b["ttft_from_add_ms"]
        margin = "- (no warden)" if b["egress_margin_ms"] is None else (
            f"{fmt(b['egress_margin_ms']['min'])} -> {fmt(h['egress_margin_ms']['min'])} -> {fmt(al['egress_margin_ms']['min'])}")
        out.append(f"| {r['trace']} | {fmt(t['p50'])} / {fmt(t['p95'])} / {fmt(t['max'])} | {h['frontend_late_for_boundary']} "
                   f"| {h['held_behind_earlier_release']} | {margin} |")
    if all("advice" in r["recorded"] for r in results):
        out += ["", ("| trace | window | order bits recorded -> hold -> aligned | residual bits (framed) recorded -> hold -> "
                "aligned | decisions recorded -> hold -> aligned |"), "|---|---|---|---|---|"]
        for r in results:
            adv = [r["recorded"]["advice"], r["variants"]["hold"]["advice"], r["variants"]["aligned"]["advice"]]
            ob = " -> ".join(f"{a['order_bits']:,.2f}" for a in adv)
            for k, w in enumerate(adv[0]["windows"]):
                ws = [a["windows"][k] if k < len(a["windows"]) else None for a in adv]
                rb = " -> ".join("-" if x is None else f"{x['residual_bits']} ({x['residual_framed']})" for x in ws)
                dc = " -> ".join("-" if x is None else f"{x['decisions']:,}" for x in ws)
                out.append(f"| {r['trace']} | {w['window']} | {ob if k == 0 else '(trace)'} | {rb} | {dc} |")
    out += ["", ("| trace | baseline simulation vs the trace: steps matched | first-token error p50 / p95 / max (ms) "
            "| span recorded / simulated (s) | step cost (ms): 1, per token, per row | cold start (ms) |"), "|---|---|---|---|---|---|"]
    for r in results:
        b, e, c = r["base"], r["base"]["first_token_error_ms"], r["clock"]
        out.append(f"| {r['trace']} | {b['steps_matched']:,}/{b['steps_compared']:,} | {fmt(e['p50'], 2)} / {fmt(e['p95'], 2)} / "
                   f"{fmt(e['max'], 2)} | {r['recorded']['span_s']:.2f} / {b['span_s']:.2f} | "
                   f"{' / '.join(f'{x:g}' for x in c['coef_ms'])} | {c['cold_start_ms']} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("traces", nargs="+", help="NAME=TRACE_DIR[:ORDER_DIR]")
    ap.add_argument("--replay", type=Path, default=REPLAY, help="the scheduler replay, replay.py")
    ap.add_argument("--add-lag-ms", action="append", default=[], help="NAME=MS: a trace without a warden's frontend-to-engine lag")
    ap.add_argument("--residual", action="store_true",
                    help="also the replay's order bits and residual advice on each simulated run (`replay.table`)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    rp = load_replay(a.replay)
    lags = {k: float(v) / 1000 for k, v in (x.split("=", 1) for x in a.add_lag_ms)}
    a.out.mkdir(parents=True, exist_ok=True)
    results, rows = [], []
    for spec in a.traces:
        name, _, dirs = spec.partition("=")
        d, _, od = dirs.partition(":")
        res, per = analyse(rp, name, Path(d), Path(od or d), lags.get(name, 0.0), a.residual)
        results.append(res)
        rows += per
        print(json.dumps({k: res[k] for k in ("trace", "order_bits", "base")}, default=str), flush=True)
    (a.out / "admit_release.json").write_text(json.dumps(results, indent=1))
    (a.out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    md = markdown(results)
    (a.out / "table.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
