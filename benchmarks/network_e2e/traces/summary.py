"""Join a corpus run's raw records into each trace's steps.jsonl and requests.jsonl, and write summary.json and README.md.

Stdlib only. `summarize(DIR)` reads DIR/plan.jsonl, DIR/meta.json and, per trace, DIR/<trace>/{meta.json, client.jsonl,
raw/*.jsonl} (`vllm_trace_plugin`, `corpus.drive`).
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

EOS = (128001,)


def read_jsonl(path: Path) -> list[dict]:
    """The file's records, less an unterminated last line (a writer killed mid-line)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").split("\n")
    return [json.loads(line) for line in lines[:-1] if line.strip()]


def raw(d: Path, kind: str) -> list[dict]:
    """The hook's `kind` records over every engine-core pid (one, unless the server restarted its engine core)."""
    out = []
    for p in sorted((d / "raw").glob(f"{kind}.*.jsonl")):
        out += read_jsonl(p)
    return out


def classify(row: dict) -> None:
    """A row's `kind` and `final` in place. A request's known tokens are its prompt and outputs (L + o); a row schedules
    positions start .. start + n - 1 of them and samples the next token when it reaches the end (`final`). A decode row
    feeds the last output token alone; every other row is a prefill: of the prompt, or a recompute of prompt and outputs
    after a preemption."""
    L, o, s, n = row["prompt_len"], row.get("num_output_tokens", 0), row["num_computed_tokens"], row["num_scheduled_tokens"]
    total = L + o
    row["kind"] = "decode" if n == 1 and o > 0 and s == total - 1 else "prefill"
    row["final"] = s + n >= total


def stats(xs: list[float]) -> dict:
    if not xs:
        return {"n": 0}
    v = sorted(xs)

    def q(p: float):
        return v[min(len(v) - 1, int(p * len(v)))]

    return {"n": len(v), "min": v[0], "p10": q(0.10), "p50": q(0.50), "p90": q(0.90), "p99": q(0.99), "max": v[-1],
            "mean": round(sum(v) / len(v), 3)}


def histogram(xs: list[int], edges: list[int]) -> dict:
    """Counts in [edges[i], edges[i+1]), the last bin closed."""
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        last = hi == edges[-1]
        out[f"{lo}-{hi}"] = sum(1 for x in xs if lo <= x < hi or (last and x == hi))
    return out


def join_trace(d: Path, plan: dict[int, dict]) -> dict:
    steps, outputs, engine = raw(d, "steps"), raw(d, "outputs"), raw(d, "requests")
    client = {r["id"]: r for r in read_jsonl(d / "client.jsonl")}
    by_step = {o["step"]: o for o in outputs}
    first, last = {}, {}
    with open(d / "steps.jsonl", "w", encoding="utf-8") as f:
        for s in steps:
            for row in s["new"] + s["cached"]:
                classify(row)
                first.setdefault(row["id"], s["step"])
                last[row["id"]] = s["step"]
            o = by_step.get(s["step"])
            if o is not None:
                s["batch"], s["sampled"], s["t_output"] = o["batch"], o["sampled"], o["t"]
            f.write(json.dumps(s, separators=(",", ":")) + "\n")
    reqs = []
    for e in engine:
        c = client.get(e["id"], {})
        p = plan.get(c.get("i"), {})
        out_ids = e["output_ids"]
        reqs.append({
            "id": e["id"], "i": c.get("i"), "arrival": e["arrival"], "send_unix": c.get("send_unix"),
            "planned_s": c.get("planned_s"), "prompt_ids": e["prompt_ids"], "prompt_len": len(e["prompt_ids"]),
            "prefix": p.get("prefix"), "prefix_len": p.get("prefix_len"), "max_tokens": e["max_tokens"],
            "output_ids": out_ids, "output_len": len(out_ids), "finish_reason": e["finish_reason"],
            "eos": e["finish_reason"] == "stop" and bool(out_ids) and out_ids[-1] in EOS, "stop_reason": e["stop_reason"],
            "cache_hit": e["cache_hit"], "num_preemptions": e["num_preemptions"], "first_step": first.get(e["id"]),
            "last_step": last.get(e["id"]), "t_free": e["t_free"], "first_token_s": c.get("first_token_s"),
            "done_s": c.get("done_s"), "chunks": c.get("chunks"), "client_status": c.get("status"),
            "client_error": c.get("error"), "client_finish_reason": c.get("finish_reason"),
            "client_ids_match": c.get("output_ids") == out_ids if c else None})
    reqs.sort(key=lambda r: (r["i"] is None, r["i"]))
    with open(d / "requests.jsonl", "w", encoding="utf-8") as f:
        for r in reqs:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return summarize_trace(d, steps, reqs, client)


def summarize_trace(d: Path, steps: list[dict], reqs: list[dict], client: dict) -> dict:
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    work = [s for s in steps if s["total_num_scheduled_tokens"]]
    rows = [r for s in work for r in s["new"] + s["cached"]]
    pre = [r for r in rows if r["kind"] == "prefill"]
    partial = [r for r in pre if not r["final"]]
    chunks = Counter(r["id"] for r in pre if not r.get("resumed") and r.get("num_output_tokens", 0) == 0)
    preempt = Counter(x for s in steps for x in s["preempted"])
    kinds = [{r["kind"] for r in s["new"] + s["cached"]} for s in work]
    caps = sorted({r["max_tokens"] for r in reqs})
    finish = Counter("eos" if r["eos"] else r["finish_reason"] for r in reqs)
    prompt_tokens = sum(r["prompt_len"] for r in reqs)
    hit_tokens = sum(r["cache_hit"] or 0 for r in reqs)
    errors = sum(1 for c in client.values() if c.get("error") or c.get("status") != 200)
    hook_errors = sum(p.read_text().count("--- ") for p in (d / "raw").glob("hook_errors.*.txt"))
    return {
        "trace": d.name, "rate": meta.get("rate"), "why": meta.get("why"), "server_argv": meta.get("server_argv"),
        "startup_s": meta.get("startup_s"), "wall_s": meta.get("wall_s"),
        "requests": {"sent": len(client), "freed_by_engine": len(reqs), "client_errors": errors,
                     "unfinished_at_drain": meta.get("unfinished"),
                     "client_ids_match": sum(1 for r in reqs if r["client_ids_match"]),
                     "finish": dict(finish), "eos_fraction": round(finish["eos"] / len(reqs), 4) if reqs else None},
        "steps": {"recorded": len(steps), "with_tokens": len(work),
                  "empty_schedule_calls": sum(s["empty_before"] for s in steps),
                  "rows_per_step": stats([len(s["new"]) + len(s["cached"]) for s in work]),
                  "tokens_per_step": stats([s["total_num_scheduled_tokens"] for s in work]),
                  "running": stats([s["running"] for s in work]), "waiting": stats([s["waiting"] for s in work]),
                  "with_partial_chunk": sum(1 for s in work if any(r["kind"] == "prefill" and not r["final"]
                                                                   for r in s["new"] + s["cached"])),
                  "mixed_prefill_decode": sum(1 for k in kinds if k == {"prefill", "decode"}),
                  "prefill_only": sum(1 for k in kinds if k == {"prefill"}),
                  "decode_only": sum(1 for k in kinds if k == {"decode"})},
        "rows": {"total": len(rows), "prefill": len(pre), "decode": len(rows) - len(pre),
                 "partial_prefill_chunks": len(partial),
                 "requests_prefilled_in_chunks": sum(1 for n in chunks.values() if n > 1),
                 "prefill_tokens_per_row": stats([r["num_scheduled_tokens"] for r in pre])},
        "lengths": {"prompt": stats([r["prompt_len"] for r in reqs]),
                    "prompt_histogram": histogram([r["prompt_len"] for r in reqs], [16, 32, 64, 128, 256, 512, 768, 1024]),
                    "output": stats([r["output_len"] for r in reqs]),
                    "output_by_cap": {str(c): {"output": stats([r["output_len"] for r in reqs if r["max_tokens"] == c]),
                                               "eos": sum(1 for r in reqs if r["max_tokens"] == c and r["eos"]),
                                               "length": sum(1 for r in reqs if r["max_tokens"] == c and r["finish_reason"] == "length")}
                                      for c in caps}},
        "prefix_cache": {"requests_with_hit": sum(1 for r in reqs if (r["cache_hit"] or 0) > 0),
                         "hit_tokens": hit_tokens, "prompt_tokens": prompt_tokens,
                         "hit_fraction": round(hit_tokens / prompt_tokens, 4) if prompt_tokens else None,
                         "hit": stats([r["cache_hit"] for r in reqs if (r["cache_hit"] or 0) > 0])},
        "preemption": {"events": sum(preempt.values()), "requests_preempted": len(preempt),
                       "max_per_request": max(preempt.values(), default=0),
                       "resumed_rows": sum(1 for r in rows if r.get("resumed"))},
        "hook_errors": hook_errors,
    }


def summarize(root: Path) -> dict:
    root = Path(root)
    plan = {r["i"]: r for r in read_jsonl(root / "plan.jsonl")}
    meta = json.loads((root / "meta.json").read_text())
    traces = {}
    for name, m in meta.get("traces", {}).items():
        d = root / name
        if (d / "raw").is_dir():
            traces[name] = join_trace(d, plan)
        if m.get("error") or name not in traces:
            traces.setdefault(name, {"trace": name})["error"] = m.get("error", "no records")
    out = {"model_id": meta.get("model_id"), "model": meta.get("model"), "model_what": meta.get("model_what"),
           "versions": meta.get("versions"), "host": meta.get("host"), "env": meta.get("env"),
           "requests_per_trace": meta.get("requests"), "seed": meta.get("seed"), "traffic": meta.get("traffic"),
           "traces": traces}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    (root / "README.md").write_text(readme(out))
    return out


FIELDS = """\
## Files

- `plan.jsonl`: the one request plan every trace serves, in order (`traffic.plan`): `i`, `prompt_ids` (BOS first),
  `prompt_len`, `prefix` (the shared-prefix group, or null) and `prefix_len`, `doc` (the C4 document the body is cut from)
  and `gap` (tokens left between the prompt's end and the document's), `max_tokens`, and `gap_draw`, the unit-rate
  exponential gap before the request (a trace at rate λ sends request i at Σ_{j≤i} gap_draw_j / λ seconds).
- `meta.json`, `summary.json`: the run (model, versions, host, leased GPU) and the statistics below, per trace.
- `<trace>/steps.jsonl`: one line per scheduler step (`Scheduler.schedule()` call) that scheduled a token or reported a
  preempted or finished request, in order:
  - `step` (0, 1, ...), `t` (unix seconds when `schedule()` returned), `empty_before` (calls since the last line that
    scheduled nothing);
  - `new`: `scheduled_new_reqs`, the requests scheduled for the first time and (Model Runner V2) those resumed after a
    preemption: `id`, `num_computed_tokens` (the first position scheduled; at a first scheduling, the prefix-cache hit),
    `num_scheduled_tokens`, `prompt_len`, `num_output_tokens`, `resumed`, and `cache_hit` (first scheduling only);
  - `cached`: `scheduled_cached_reqs`, the running requests: `id`, `num_computed_tokens`, `num_scheduled_tokens`,
    `num_output_tokens` (outputs so far), `prompt_len`, `resumed`;
  - every row also has `kind` (`decode`: one token, the last output fed back; `prefill`: prompt positions, or after a
    preemption a recompute of prompt and outputs) and `final` (the row reaches the request's last known token, so the
    step samples its next one; a `prefill` row that isn't final is a partial chunk);
  - `total_num_scheduled_tokens`, `preempted` (`preempted_req_ids`: preempted in this call), `finished`
    (`finished_req_ids`: finished since the previous call), `waiting` and `running` (queue lengths after the call);
  - `batch`: the model runner's row order for the step (`ModelRunnerOutput.req_ids`; V2 puts decodes first), `sampled`:
    the token ids it sampled per batch row (empty for a partial chunk), `t_output` (unix seconds when the output was back).
- `<trace>/requests.jsonl`: one line per request the scheduler freed: `id` (the client's X-Request-Id, `<trace>-<i>`),
  `i` (the plan's index), `arrival` (unix seconds, vLLM's `Request.arrival_time`), `send_unix` and `planned_s` (the
  client's send time and the plan's arrival), `prompt_ids`, `prompt_len`, `prefix`, `prefix_len`, `max_tokens`,
  `output_ids`, `output_len`, `finish_reason` (`stop` or `length`), `eos` (it stopped on the model's end-of-text, 128001;
  no stop strings or stop ids are set), `cache_hit` (prefix-cache tokens at its first scheduling), `num_preemptions`,
  `first_step` and `last_step` (the steps that scheduled it first and last), `t_free`, the client's `first_token_s`,
  `done_s` and `chunks` (`[seconds after sending, tokens]` per streamed chunk), and `client_ids_match` (the streamed ids
  equal the engine's).
- `<trace>/client.jsonl`, `<trace>/raw/`: the client's and the hook's records as written (`raw/steps.PID.jsonl`,
  `raw/outputs.PID.jsonl`, `raw/requests.PID.jsonl`); `<trace>/server.log`, `<trace>/meta.json`: the server's log, argv
  and timings.
"""


def readme(s: dict) -> str:
    v = s.get("versions") or {}
    lines = [
        f"# vLLM V1 scheduler traces under varied traffic: {s.get('model_what')}", "",
        f"Model `{s.get('model')}` (`{s.get('model_id')}`) on {s.get('host')}, GPU "
        f"`{(s.get('env') or {}).get('CUDA_VISIBLE_DEVICES')}` (`{(s.get('env') or {}).get('GPU_LEASE_UUID')}`); vLLM "
        f"{v.get('vllm')}, torch {v.get('torch')}. Made by `benchmarks/network_e2e/traces/corpus.py` (its docstring and "
        "`traffic.py`'s say how): plain vLLM's OpenAI server, batch-invariant eager (the integration's bi-eager engine), "
        "async scheduling off, Model Runner V2, prefix caching on, at most 256 sequences; a scheduler hook "
        "(`vllm_trace_plugin.py`) records each step; an async client streams each request at its Poisson arrival, "
        "temperature 0, `max_tokens` 64 or 256, no stop strings.", "",
        f"Every trace serves the same {s.get('requests_per_trace')} requests (`plan.jsonl`, seed {s.get('seed')}) on a "
        "fresh server, so each starts with a cold prefix cache.", "",
        "| trace | rate (req/s) | what | requests freed | steps | rows | rows/step p50 / p90 / max | tokens/step p50 / max | running p50 / max | waiting p90 / max | EOS (share) | length | prefix-hit requests | hit tokens (of prompt tokens) | preemptions (requests) | partial prefill chunks (steps with one) | mixed steps |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, t in s["traces"].items():
        if "steps" not in t:
            lines.append(f"| {name} | {t.get('rate')} | {t.get('why')} | failed: {t.get('error')} |" + " |" * 13)
            continue
        st, rq, rw, pc, pe = t["steps"], t["requests"], t["rows"], t["prefix_cache"], t["preemption"]
        r, k, run, wait = st["rows_per_step"], st["tokens_per_step"], st["running"], st["waiting"]
        lines.append(
            f"| {name} | {t['rate']} | {t['why']}{' (failed: ' + t['error'] + ')' if t.get('error') else ''} | "
            f"{rq['freed_by_engine']} of {rq['sent']} | {st['with_tokens']} | {rw['total']} | "
            f"{r.get('p50')} / {r.get('p90')} / {r.get('max')} | {k.get('p50')} / {k.get('max')} | "
            f"{run.get('p50')} / {run.get('max')} | {wait.get('p90')} / {wait.get('max')} | "
            f"{rq['finish'].get('eos', 0)} ({rq['eos_fraction'] or 0:.1%}) | {rq['finish'].get('length', 0)} | "
            f"{pc['requests_with_hit']} | {pc['hit_tokens']} ({pc['hit_fraction'] or 0:.1%}) | "
            f"{pe['events']} ({pe['requests_preempted']}) | {rw['partial_prefill_chunks']} ({st['with_partial_chunk']}) | "
            f"{st['mixed_prefill_decode']} |")
    lines += ["", "## Lengths", ""]
    for name, t in s["traces"].items():
        if "lengths" not in t:
            continue
        ln = t["lengths"]
        p, o = ln["prompt"], ln["output"]
        by_cap = "; ".join(f"cap {c}: output p50 {x['output'].get('p50')}, EOS {x['eos']}, length {x['length']}"
                           for c, x in ln["output_by_cap"].items())
        lines.append(f"- **{name}**: prompt min {p.get('min')} / p50 {p.get('p50')} / p90 {p.get('p90')} / max {p.get('max')}; "
                     f"output min {o.get('min')} / p50 {o.get('p50')} / max {o.get('max')}; {by_cap}; "
                     f"prompt histogram {ln['prompt_histogram']}.")
    lines += ["", "Every statistic, per trace, is in `summary.json`; client errors, unfinished requests at the drain "
              "deadline and hook errors are there too (`requests.client_errors`, `requests.unfinished_at_drain`, "
              "`hook_errors`).", "", FIELDS]
    return "\n".join(lines)
