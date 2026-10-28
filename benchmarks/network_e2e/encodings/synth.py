"""Synthetic serving traces in I1 v0's form, for measuring the codes before the trace corpus's real ones (and beside
them).

A vLLM-V1-like engine on the bucket clock:
- **Arrivals**: Poisson at `rate` requests per second. Request j arrives in bucket `b_j` (its certified anchor), and the
  ingress certifier delivers it at the bucket's end, `(b_j + 1)·τ`.
- **Admission**: before each step the engine admits every delivered request (its admission step is that step's index);
  an idle engine starts a step at the next delivery.
- **The scheduler**, per step, under a token budget (`max_num_batched_tokens`) and a sequence cap (`max_num_seqs`): the
  running requests first, in order, a prefill chunk taking as much of its prompt as the budget leaves and a decode one
  token; then the waiting ones, first come, first served. A prompt that shares a registered system prefix hits the
  prefix cache for that prefix once a request has computed it, and that request is the hit's source.
- **Lengths**: prompts log-uniform over [16, 1024]; outputs end at EOS (geometric, mean `eos_mean`) under `max_tokens`.
- **Step time**: `a + b·tokens + c·context/1024` µs, times a lognormal jitter, plus a rare stall (a GC pause, say).
- **Release**: a step's kept outputs are released in the bucket its end falls in.

The trace is a function of its parameters and seed alone (`random.Random(seed)`).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    rate: float = 2.0  # requests per second
    windows: int = 6
    T: int = 600
    tau_us: int = 100_000
    prompt_min: int = 16
    prompt_max: int = 1024
    eos_mean: float = 200.0
    max_tokens: int = 512
    max_num_seqs: int = 256
    max_batched_tokens: int = 2048
    system_prefix: int = 256  # tokens of the shared system prompt
    system_share: float = 0.3  # the share of requests that carry it
    a_us: int = 7_000
    b_us: int = 35
    c_us: int = 36  # per 1024 tokens of context
    jitter: float = 0.03
    stall_p: float = 0.002
    stall_us: int = 20_000
    seed: int = 0


def generate(p: Params, code):
    """(Advice, Public, admission steps, step durations in µs), with `code` the codec module (its dataclasses)."""
    rng = random.Random(p.seed)
    horizon_us = p.windows * p.T * p.tau_us
    arrivals, t = [], 0.0
    while True:
        t += rng.expovariate(p.rate) * 1e6
        if t >= horizon_us - 2 * p.tau_us:
            break
        arrivals.append(int(t))
    reqs = []
    for j, a in enumerate(arrivals):
        P = int(round(math.exp(rng.uniform(math.log(p.prompt_min), math.log(p.prompt_max)))))
        sys_prefix = rng.random() < p.system_share and P > p.system_prefix
        O = min(1 + int(rng.expovariate(1 / p.eos_mean)), p.max_tokens)
        reqs.append({"anchor": a // p.tau_us, "deliver": (a // p.tau_us + 1) * p.tau_us, "P": P, "O": O,
                     "sys": sys_prefix, "pos": 0, "kept": 0, "admit": None, "cache": 0})
    steps, release, durations = [], [], []
    running: list[int] = []
    waiting: list[int] = []
    nxt = 0
    now = 0
    cache_src = None  # the request that computed the system prefix's KV, once one has
    while nxt < len(reqs) or running or waiting:
        if not running and not waiting:
            now = max(now, reqs[nxt]["deliver"])
        while nxt < len(reqs) and reqs[nxt]["deliver"] <= now:
            reqs[nxt]["admit"] = len(steps)
            waiting.append(nxt)
            nxt += 1
        budget, rows = p.max_batched_tokens, []
        for r in running:
            q = reqs[r]
            if budget < 1:
                break
            if q["pos"] < q["P"]:
                n = min(q["P"] - q["pos"], budget)
                rows.append(code.Row(r, "prefill", q["pos"], n))
            else:
                n = 1
                rows.append(code.Row(r, "decode", q["pos"], 1))
            budget -= n
        while waiting and budget >= 1 and len(running) < p.max_num_seqs:
            r = waiting.pop(0)
            q = reqs[r]
            src = ()
            if q["sys"] and cache_src is not None:
                q["cache"] = p.system_prefix
                q["pos"] = p.system_prefix
                src = ((cache_src, p.system_prefix),)
            n = min(q["P"] - q["pos"], budget)
            rows.append(code.Row(r, "prefill", q["pos"], n, q["cache"], False, src))
            running.append(r)
            budget -= n
        tokens = sum(x.n for x in rows)
        context = sum(x.start + x.n for x in rows)
        d = (p.a_us + p.b_us * tokens + p.c_us * context / 1024) * math.exp(rng.gauss(0, p.jitter))
        if rng.random() < p.stall_p:
            d += p.stall_us
        d = int(d)
        now += d
        steps.append(code.StepAdvice(tuple(rows)))
        durations.append(d)
        for x in rows:
            q = reqs[x.request]
            if x.start + x.n == q["P"] + q["kept"]:
                q["kept"] += 1
                release.append(now // p.tau_us)
            q["pos"] = x.start + x.n
            if q["sys"] and cache_src is None and q["pos"] >= p.system_prefix:
                cache_src = x.request
        running = [r for r in running if reqs[r]["kept"] < reqs[r]["O"]]
    adv = code.Advice(
        tuple(code.RequestAdvice(f"u{j}", q["P"], q["O"], "length" if q["O"] == p.max_tokens else "eos")
              for j, q in enumerate(reqs)),
        tuple(steps), tuple(release))
    # arrivals stop after `windows` windows; the run covers the drain that follows
    pub = code.Public(tuple(q["anchor"] for q in reqs), p.T, max(p.windows, (release[-1] // p.T + 1) if release else 0))
    return adv, pub, [q["admit"] for q in reqs], durations
