"""The trace corpus's vLLM scheduler traces (`benchmarks/network_e2e/traces`, store kind `vllm-trace-corpus/v1`) as this
directory's traces: I1 v0's advice, the public anchors, the admission steps, the step durations and the serving caps.

A trace directory holds the joined `steps.jsonl` and `requests.jsonl` and its own `meta.json` (the server's argv); its
parent holds the run's `meta.json` (the traffic's `max_tokens` caps). The mapping:

- **requests** in order of their first row (I1), each with its prompt and output lengths, and `eos` or `length`;
- **a step** per scheduler call that ran a row, its rows in the scheduler's order (running, then new): start =
  `num_computed_tokens`, n = `num_scheduled_tokens`, a decode iff n = 1 and start ≥ the prompt (I1's rule). A
  request's first row carries its prefix-cache hit (its start), and a resumed row its re-hit (its start) and
  `preempted`;
- **each hit's sources**, block by block: the first request (in first-row order) whose prompt holds the same tokens up
  to the block's end (vLLM's block hashes chain the prefix), and for a resumed row's blocks past its prompt, the request
  itself. Eviction and recompute can make vLLM's own source a later request; nothing checks sources yet (`derived.py`);
- **releases**: each kept output at the bucket of its step's `t_output`; **anchors**: each request's `arrival` bucket,
  both from the first arrival;
- **admission**: the step of each request's first row;
- **durations**: per step, the µs to its output from the previous step's output (back to back) or from its own
  scheduling (after an idle engine);
- **caps** from the server's argv (`--max-num-batched-tokens`, `--max-num-seqs`, `--max-model-len`) and the traffic's
  largest `max_tokens`; vLLM's block of 16 tokens.

Stdlib only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BLOCK = 16
#: an engine is back to back when a step is scheduled within this long of the previous step's output
BUSY_S = 0.005


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _argv(argv: list[str], flag: str, default: int) -> int:
    return int(argv[argv.index(flag) + 1]) if flag in argv else default


def caps_of(C: Any, server_argv: list[str], max_tokens: int, steps_per_bucket: int = 32) -> Any:
    return C.Caps(prompt_max=_argv(server_argv, "--max-model-len", 2048), output_max=max_tokens,
                  seqs_max=_argv(server_argv, "--max-num-seqs", 256),
                  tokens_max=_argv(server_argv, "--max-num-batched-tokens", 8192),
                  steps_per_bucket=steps_per_bucket, block_size=BLOCK)


def _sources(prompt: list[int], c: int, me: int, owner: dict, prompt_len: int) -> tuple[tuple[int, int], ...]:
    """The runs of requests whose KV covers positions 0 … c − 1 of request `me` (an index in first-row order)."""
    runs: list[list[int]] = []
    for b in range(c // BLOCK):
        end = (b + 1) * BLOCK
        r = owner.get(tuple(prompt[:end]), me) if end <= prompt_len else me
        if runs and runs[-1][0] == r:
            runs[-1][1] += BLOCK
        else:
            runs.append([r, BLOCK])
    return tuple((r, n) for r, n in runs)


def load_trace(C: Any, d: Path, tau_us: int = 100_000, T: int = 600, max_tokens: int | None = None) -> dict:
    """One trace directory as `measure.py`'s trace: {name, advice, public, admit, durations, tau_us, caps}."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    run_meta = json.loads((d.parent / "meta.json").read_text()) if (d.parent / "meta.json").exists() else {}
    if max_tokens is None:
        max_tokens = max((run_meta.get("traffic") or {}).get("caps") or [1024])
    steps = [s for s in _jsonl(d / "steps.jsonl") if s["cached"] or s["new"]]
    reqs = {r["id"]: r for r in _jsonl(d / "requests.jsonl")}
    order: dict[str, int] = {}
    for s in steps:
        for x in s["cached"] + s["new"]:
            order.setdefault(x["id"], len(order))
    if set(order) != set(reqs):
        raise ValueError(f"{d.name}: {len(set(reqs) - set(order))} requests never ran, {len(set(order) - set(reqs))} rows "
                         "name a request requests.jsonl lacks")
    ids = sorted(order, key=order.get)
    t0 = min(reqs[i]["arrival"] for i in ids)
    requests = []
    for i in ids:
        q = reqs[i]
        if q["finish_reason"] == "length":
            finish = "length"
        elif q["eos"]:
            finish = "eos"
        else:
            raise ValueError(f"{d.name}: request {i} stopped on {q['finish_reason']!r}, which I1's finish can't say")
        requests.append(C.RequestAdvice(i, q["prompt_len"], q["output_len"], finish))
    owner: dict[tuple, int] = {}
    registered: set[int] = set()
    kept = [0] * len(ids)
    rows_out, release, admit, durations = [], [], [None] * len(ids), []
    prev_out = None
    for k, s in enumerate(steps):
        rows = []
        for x in s["cached"] + s["new"]:
            r = order[x["id"]]
            q = requests[r]
            start, n = x["num_computed_tokens"], x["num_scheduled_tokens"]
            first, resumed = admit[r] is None, bool(x.get("resumed"))
            if first:
                admit[r] = k
            hit = start if first or resumed else 0
            src = _sources(reqs[x["id"]]["prompt_ids"], hit, r, owner, q.prompt_len) if hit else ()
            kind = C.DECODE if n == 1 and start >= q.prompt_len else C.PREFILL
            rows.append(C.Row(r, kind, start, n, hit, resumed and not first, src))
            if r not in registered:
                registered.add(r)
                p = reqs[x["id"]]["prompt_ids"]
                for b in range(len(p) // BLOCK):
                    owner.setdefault(tuple(p[:(b + 1) * BLOCK]), r)
        bucket = int((s["t_output"] - t0) * 1e6) // tau_us
        for y in rows:
            q = requests[y.request]
            if y.start + y.n == q.prompt_len + kept[y.request] and kept[y.request] < q.output_len:
                kept[y.request] += 1
                release.append(bucket)
        busy = prev_out is not None and s["t"] - prev_out < BUSY_S
        durations.append(int(round((s["t_output"] - (prev_out if busy else s["t"])) * 1e6)))
        prev_out = s["t_output"]
        rows_out.append(C.StepAdvice(tuple(rows)))
    if kept != [q.output_len for q in requests]:
        bad = sum(1 for a, q in zip(kept, requests) if a != q.output_len)
        raise ValueError(f"{d.name}: {bad} requests keep a different number of outputs than they returned")
    anchors = tuple(int((reqs[i]["arrival"] - t0) * 1e6) // tau_us for i in ids)
    late = sum(1 for a, b in zip(anchors, anchors[1:]) if b < a)
    if late:
        raise ValueError(f"{d.name}: {late} requests ran first ahead of an earlier bucket's arrival")
    adv = C.Advice(tuple(requests), tuple(rows_out), tuple(release))
    W = max(release) // T + 1
    server_argv = meta.get("server_argv") or []
    return {"name": d.name, "advice": adv, "public": C.Public(anchors, T, W), "admit": admit, "durations": durations,
            "tau_us": tau_us, "caps": caps_of(C, server_argv, max_tokens)}


def load_corpus(C: Any, root: Path, tau_us: int = 100_000, T: int = 600) -> list[dict]:
    """Every trace under a corpus run's model directory (`<root>/<trace>/steps.jsonl`) that has records."""
    return [load_trace(C, p.parent, tau_us, T) for p in sorted(Path(root).glob("*/steps.jsonl"))]
