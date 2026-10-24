"""Join a corpus run's raw records into each trace's steps.jsonl and requests.jsonl, and write summary.json and README.md.

Stdlib only. `summarize(DIR)` reads DIR/plan.jsonl, DIR/meta.json and, per trace, DIR/<trace>/{meta.json, client.jsonl,
raw/*.jsonl} (`vllm_trace_plugin`, `corpus.drive`). `compare` checks that traces of one plan gave every request the same
output ids, and where they first differ.
"""
from __future__ import annotations

import hashlib
import json
import math
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


def sampled_index(row: dict) -> int:
    """The output index a final row samples: it runs positions up to start + n - 1 and samples position start + n. Under
    async scheduling a cached row's `num_output_tokens` counts the outputs still in flight too, so this holds there as
    well."""
    return row["num_computed_tokens"] + row["num_scheduled_tokens"] - row["prompt_len"]


def overrun(steps: list[dict], reqs: list[dict]) -> dict:
    """Rows that sample past a request's last output (async scheduling schedules the next step before the last one's
    tokens are back), by how the request finished."""
    last = {r["id"]: (r["output_len"], "eos" if r["eos"] else r["finish_reason"]) for r in reqs}
    by_finish, per = Counter(), Counter()
    for s in steps:
        for row in s["new"] + s["cached"]:
            n = last.get(row["id"])
            if n is not None and row["final"] and sampled_index(row) >= n[0]:
                by_finish[n[1]] += 1
                per[row["id"]] += 1
    return {"rows": sum(by_finish.values()), "by_finish": dict(by_finish), "requests": len(per),
            "max_per_request": max(per.values(), default=0)}


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
    calls = raw(d, "adversary")
    by_call = {c["step"]: c for c in calls}
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
            c = by_call.get(s.get("sched_step"))
            if c is not None:
                s["adversary"] = {k: c[k] for k in ("preempted", "held", "admissible", "splits", "hits")}
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
    return summarize_trace(d, steps, reqs, client) | {"adversary": adversary(d, calls)}


def adversary(d: Path, calls: list[dict]) -> dict | None:
    """What a hostile scheduler (`vllm_adversarial_plugin`) chose over the trace, from its records; None for the honest one."""
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if not meta.get("adversary") and not calls:
        return None
    forced = Counter(x for c in calls for x in c["preempted"])
    splits = [x for c in calls for x in c["splits"]]
    hits = [x for c in calls for x in c["hits"]]
    queued = [c for c in calls if c["held"] + c["admissible"]]
    return {**(meta.get("adversary") or {}), "calls": len(calls), "forced_preemptions": sum(forced.values()),
            "requests_force_preempted": len(forced), "max_forced_per_request": max(forced.values(), default=0),
            "calls_holding_back": sum(1 for c in queued if c["held"]),
            "held_share": stats([round(c["held"] / (c["held"] + c["admissible"]), 3) for c in queued]),
            "splits": len(splits), "requests_split": len({x[0] for x in splits}),
            "split_to": stats([x[2] for x in splits]), "hits_seen": len(hits),
            "hits_refused": sum(1 for x in hits if x[2] == 0), "hits_cut": sum(1 for x in hits if 0 < x[2] < x[1]),
            "errors": sum(p.read_text().count("--- ") for p in (d / "raw").glob("adversary_errors.*.txt"))}


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
        "overrun": overrun(steps, reqs),
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
           "variant": meta.get("variant", "greedy"), "sampling": meta.get("sampling", {"name": "greedy", "temperature": 0.0}),
           "async_scheduling": meta.get("async_scheduling", False),
           "versions": meta.get("versions"), "host": meta.get("host"), "env": meta.get("env"),
           "requests_per_trace": meta.get("requests"), "seed": meta.get("seed"), "traffic": meta.get("traffic"),
           "traces": traces,
           "outputs": compare({n: root / n for n, t in traces.items() if (root / n / "requests.jsonl").exists()})}
    (root / "summary.json").write_text(json.dumps(out, indent=1))
    (root / "README.md").write_text(readme(out))
    return out


def path(row: dict, kinds: set[str]) -> str:
    """The attention path of a row that samples: a decode (one query row over the request's KV) in a decode-only or a
    mixed step, or a prefill's last chunk, from position 0 or over KV already there (an earlier chunk's, a prefix-cache
    hit's, or that of a resumed request's earlier chunks). The linear layers run one batch-invariant matmul over every row
    of the step whatever the path."""
    if row["kind"] == "decode":
        return "decode, decode-only step" if kinds == {"decode"} else "decode, mixed step"
    return "prefill from 0" if row["num_computed_tokens"] == 0 else "prefill over KV"


def located(d: Path, wanted: dict[str, set[int]]) -> dict[tuple[str, int], dict]:
    """Where trace `d` sampled output token j of request id, for each (id, j) in `wanted` (id -> {j}): the step, its rows and
    tokens, the row (its kind, attention `path`, first position `start` and token count `n`, and whether a hostile
    scheduler cut that count), and the request's chunks and preemptions up to then (those a hostile scheduler forced
    too). A token sampled twice (an in-flight token dropped at a preemption and sampled again) is placed at its last
    sampling."""
    out, chunks, preempted, forced = {}, Counter(), Counter(), Counter()
    for s in read_jsonl(d / "steps.jsonl"):
        adv = s.get("adversary") or {}
        for x in s["preempted"]:
            preempted[x] += 1
        for x in adv.get("preempted", ()):
            forced[x] += 1
        rows = s["new"] + s["cached"]
        kinds = {r["kind"] for r in rows}
        batch = s.get("batch") or []
        for row in rows:
            rid = row["id"]
            if rid not in wanted:
                continue
            if row["kind"] == "prefill" and not row.get("resumed") and row.get("num_output_tokens", 0) == 0:
                chunks[rid] += 1
            if not row["final"] or (j := sampled_index(row)) not in wanted[rid]:
                continue
            k = batch.index(rid) if rid in batch else None
            out[(rid, j)] = {"step": s["step"], "rows": len(rows), "tokens": s["total_num_scheduled_tokens"],
                             "row": row["kind"], "path": path(row, kinds), "start": row["num_computed_tokens"],
                             "n": row["num_scheduled_tokens"], "cut": any(x[0] == rid for x in adv.get("splits", ())),
                             "resumed": bool(row.get("resumed")), "prefill_chunks": chunks[rid],
                             "preempted_before": preempted[rid], "forced_before": forced[rid], "batch_index": k,
                             "token": s["sampled"][k][0] if k is not None and s.get("sampled") and s["sampled"][k] else None}
    return out


def first_difference(a: list[int], b: list[int]) -> int:
    return next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def compare(traces: dict[str, Path]) -> dict:
    """Whether every trace (label -> summarized trace dir, all of one plan) gave each plan request the same output ids. A
    request that differs is placed at its first difference: the earliest output index where some trace's ids differ from
    the first trace's, located in every trace (`located`); `first` has each differing trace's earliest such step.

    `bits_per_request` measures what the traces' schedules chose among: per request, log2 of the number of distinct
    output-id sequences across the traces (0 where all agree), its mean over the requests and its maximum. With the first
    trace honest and the rest hostile, that is how many bits a scheduler steered into a request's outputs here, a lower
    bound on the covert channel's capacity (the traces tried only some schedules)."""
    labels = list(traces)
    reqs = {lb: {r["i"]: r for r in read_jsonl(d / "requests.jsonl") if r.get("i") is not None} for lb, d in traces.items()}
    every = sorted(set().union(*(set(x) for x in reqs.values()))) if reqs else []
    pairs = {f"{a} = {b}": 0 for k, a in enumerate(labels) for b in labels[k + 1:]}
    same = dict.fromkeys(labels, 0)
    missing, differ = [], []
    for i in every:
        rs = {lb: reqs[lb].get(i) for lb in labels}
        if any(r is None for r in rs.values()):
            missing.append(i)
            continue
        ids = {lb: r["output_ids"] for lb, r in rs.items()}
        for lb in labels:
            same[lb] += ids[lb] == ids[labels[0]]
        for k, a in enumerate(labels):
            for b in labels[k + 1:]:
                pairs[f"{a} = {b}"] += ids[a] == ids[b]
        if len({tuple(x) for x in ids.values()}) > 1:
            ref = ids[labels[0]]
            j = min(first_difference(ref, x) for x in ids.values() if x != ref)
            differ.append({"i": i, "token": j, "outputs": len({tuple(x) for x in ids.values()}),
                           "differs_from_first": [lb for lb in labels if ids[lb] != ref],
                           "at": {lb: {"id": rs[lb]["id"], "output_len": len(ids[lb]), "finish": rs[lb]["finish_reason"],
                                       "num_preemptions": rs[lb]["num_preemptions"], "cache_hit": rs[lb]["cache_hit"]}
                                  for lb in labels}})
    for lb in labels if differ else ():
        want: dict[str, set[int]] = {}
        for x in differ:
            want.setdefault(x["at"][lb]["id"], set()).add(x["token"])
        where = located(traces[lb], want)
        for x in differ:
            x["at"][lb] |= where.get((x["at"][lb]["id"], x["token"]), {"step": None})
    first = {}
    for lb in labels[1:]:
        placed = [x for x in differ if lb in x["differs_from_first"] and x["at"][lb].get("step") is not None]
        if placed:
            x = min(placed, key=lambda x: x["at"][lb]["step"])
            first[lb] = {**x["at"][lb], "i": x["i"], "index": x["token"]}
    n = len(every) - len(missing)
    bits = [math.log2(x["outputs"]) for x in differ]
    if not labels:
        verdict = "no traces"
    elif differ:
        verdict = f"differ: {len(differ)} of {n} requests have different output ids across {len(labels)} traces"
    else:
        verdict = f"identical: all {n} requests have the same output ids in all {len(labels)} traces"
    if missing:
        verdict += f" ({len(missing)} requests missing from some trace)"
    return {"traces": labels, "requests": n, "identical": n - len(differ), "verdict": verdict, "missing": missing,
            "same_as_first": same, "pairs_identical": pairs,
            "bits_per_request": {"mean": round(sum(bits) / n, 6) if n else 0.0, "max": round(max(bits, default=0.0), 6)},
            "first": first, "differ": differ}


def corpus_spec(arg: str) -> tuple[Path, list[str] | None]:
    """``CORPUS[:TRACE,...]``."""
    path, _, names = arg.partition(":")
    return Path(path), [n for n in names.split(",") if n] or None


def compare_corpora(specs: list[tuple[Path, list[str] | None]], out: Path) -> dict:
    """`compare` across summarized corpora that serve one plan (their plan.jsonl files are byte for byte the same), each
    trace labelled ``<variant>/<trace>``; writes OUT/compare.json and OUT/COMPARE.md."""
    traces, corpora, plans = {}, [], set()
    for root, names in specs:
        meta = json.loads((root / "meta.json").read_text())
        variant = meta.get("variant", "greedy")
        plans.add(hashlib.sha256((root / "plan.jsonl").read_bytes()).hexdigest())
        corpora.append({"dir": str(root), "variant": variant, "async_scheduling": meta.get("async_scheduling", False),
                        "sampling": meta.get("sampling", {"name": "greedy", "temperature": 0.0})})
        for name in names or [n for n in meta.get("traces", {}) if (root / n / "requests.jsonl").exists()]:
            label = f"{variant}/{name}"
            if label in traces:
                label = f"{variant}#{len(corpora)}/{name}"
            traces[label] = root / name
    if len(plans) != 1:
        raise SystemExit(f"the corpora serve {len(plans)} different plans; compare only corpora of one plan")
    result = {"plan_sha256": plans.pop(), "corpora": corpora, **compare(traces)}
    out.mkdir(parents=True, exist_ok=True)
    (out / "compare.json").write_text(json.dumps(result, indent=1))
    (out / "COMPARE.md").write_text("\n".join(["# Output ids across corpora", "", *outputs_section(result)]) + "\n")
    return result


def outputs_section(c: dict, limit: int = 20) -> list[str]:
    lines = [f"**{c['verdict']}.** Traces: {', '.join(f'`{x}`' for x in c['traces'])}.", ""]
    if "same_as_first" in c and len(c["traces"]) > 1:
        ref = c["traces"][0]
        lines += [f"Requests with the output ids of `{ref}`, per trace: " +
                  "; ".join(f"{k}: {v} of {c['requests']}" for k, v in c["same_as_first"].items() if k != ref) + ".", ""]
    if len(c["traces"]) > 2:
        lines += ["Requests with the same output ids, pair by pair: " +
                  "; ".join(f"{k}: {v}" for k, v in c["pairs_identical"].items()) + ".", ""]
    if "bits_per_request" in c:
        b = c["bits_per_request"]
        lines += [f"Bits per request (log2 of the distinct output-id sequences across the traces, 0 where they agree): mean "
                  f"{b['mean']}, max {b['max']}.", ""]
    if not c["differ"]:
        return lines
    for lb, a in (c.get("first") or {}).items():
        lines += [f"First divergence in `{lb}`: request {a['i']}, output index {a['index']} (token {a['token']}), step "
                  f"{a['step']} ({a['rows']} rows, {a['tokens']} tokens), {a['path']} from position {a['start']} ({a['n']} tokens"
                  f"{', cut by the adversary' if a.get('cut') else ''}), after {a['prefill_chunks']} prefill chunks and "
                  f"{a['preempted_before']} preemptions ({a.get('forced_before', 0)} forced).", ""]
    lines += [f"The first {min(limit, len(c['differ']))} of {len(c['differ'])} requests that differ (all in `differ`), at "
              "the first output index where some trace's ids differ from the first trace's; per trace: the step that "
              "sampled that index, the step's rows and tokens, the request's row there (its attention path, first "
              "position and tokens), its prefill chunks and the preemptions before it (forced ones in brackets).", "",
              "| request | index | trace | token | step | rows | tokens | row | path | start / n | prefill chunks | "
              "preempted before | outputs |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for x in c["differ"][:limit]:
        for lb, a in x["at"].items():
            lines.append(f"| {x['i']} | {x['token']} | {lb}{' (differs)' if lb in x['differs_from_first'] else ''} | "
                         f"{a.get('token')} | {a.get('step')} | {a.get('rows')} | {a.get('tokens')} | "
                         f"{a.get('row')}{' (resumed)' if a.get('resumed') else ''} | {a.get('path')} | "
                         f"{a.get('start')} / {a.get('n')}{' (cut)' if a.get('cut') else ''} | {a.get('prefill_chunks')} | "
                         f"{a.get('preempted_before')} [{a.get('forced_before', 0)}] | {a['output_len']} ({a['finish']}) |")
    return lines


FIELDS = """\
## Files

- `plan.jsonl`: the one request plan every trace serves, in order (`traffic.plan`): `i`, `prompt_ids` (BOS first),
  `prompt_len`, `prefix` (the shared-prefix group, or null) and `prefix_len`, `doc` (the C4 document the body is cut from)
  and `gap` (tokens left between the prompt's end and the document's), `max_tokens`, and `gap_draw`, the unit-rate
  exponential gap before the request (a trace at rate λ sends request i at Σ_{j≤i} gap_draw_j / λ seconds).
- `meta.json`, `summary.json`: the run (model, variant, sampling, versions, host, leased GPU) and the statistics below, per
  trace; `summary.json`'s `overrun` per trace counts the rows that sample past a request's last output (async scheduling
  schedules a step before the previous one's tokens are back), and its `outputs` compares the traces' output ids
  request by request (`summary.compare`).
- `<trace>/steps.jsonl`: one line per scheduler step (`Scheduler.schedule()` call) that scheduled a token or reported a
  preempted or finished request, in order:
  - `step` (0, 1, ...), `sched_step` (the scheduler's `current_step`, which counts every call), `t` (unix seconds when
    `schedule()` returned), `empty_before` (calls since the last line that scheduled nothing);
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
    the token ids it sampled per batch row (empty for a partial chunk), `t_output` (unix seconds when the output was back);
  - in a hostile trace, `adversary`: what `vllm_adversarial_plugin` chose for the call: `preempted` (the requests it
    preempted before it, also in `preempted`), `held` and `admissible` (the waiting requests it held back and left),
    `splits` (`[id, n, cut to]` per row whose token count it cut) and `hits` (`[id, hit, hit used]` per prefix-cache
    lookup that found one).
- `<trace>/requests.jsonl`: one line per request the scheduler freed: `id` (the client's X-Request-Id, `req-<i>` in every
  trace; corpora made before seeded sampling used `<trace>-<i>`),
  `i` (the plan's index), `arrival` (unix seconds, vLLM's `Request.arrival_time`), `send_unix` and `planned_s` (the
  client's send time and the plan's arrival), `prompt_ids`, `prompt_len`, `prefix`, `prefix_len`, `max_tokens`,
  `output_ids`, `output_len`, `finish_reason` (`stop` or `length`), `eos` (it stopped on the model's end-of-text, 128001;
  no stop strings or stop ids are set), `cache_hit` (prefix-cache tokens at its first scheduling), `num_preemptions`,
  `first_step` and `last_step` (the steps that scheduled it first and last), `t_free`, the client's `first_token_s`,
  `done_s` and `chunks` (`[seconds after sending, tokens]` per streamed chunk), and `client_ids_match` (the streamed ids
  equal the engine's).
- `<trace>/client.jsonl`, `<trace>/raw/`: the client's records as written (with the `seed` each request sent, if any) and
  the hook's (`raw/steps.PID.jsonl`,
  `raw/outputs.PID.jsonl`, `raw/requests.PID.jsonl`, and in a hostile trace the adversary's `raw/adversary.PID.jsonl`);
  `<trace>/server.log`, `<trace>/meta.json`: the server's log, argv, timings and adversary (seed and prefix-cache mode).
"""


def sampling_text(s: dict) -> str:
    p = s.get("sampling") or {}
    if p.get("name") != "seeded":
        return f"temperature {p.get('temperature', 0.0)}"
    return (f"temperature {p.get('temperature')}, top_p {p.get('top_p')}, top_k {p.get('top_k')}, each request seeded by "
            f"`corpus.request_seed` of its id ({p.get('seed_rule')})")


def readme(s: dict) -> str:
    v = s.get("versions") or {}
    lines = [
        f"# vLLM V1 scheduler traces under varied traffic: {s.get('model_what')}, {s.get('variant', 'greedy')}", "",
        f"Model `{s.get('model')}` (`{s.get('model_id')}`) on {s.get('host')}, GPU "
        f"`{(s.get('env') or {}).get('CUDA_VISIBLE_DEVICES')}` (`{(s.get('env') or {}).get('GPU_LEASE_UUID')}`); vLLM "
        f"{v.get('vllm')}, torch {v.get('torch')}. Made by `benchmarks/network_e2e/traces/corpus.py` (its docstring and "
        "`traffic.py`'s say how): plain vLLM's OpenAI server, batch-invariant eager (the integration's bi-eager engine), "
        f"async scheduling {'on' if s.get('async_scheduling') else 'off'}, Model Runner V2, prefix caching on, at most 256 "
        "sequences; a scheduler hook (`vllm_trace_plugin.py`) records each step; an async client streams each request at "
        f"its Poisson arrival, {sampling_text(s)}, `max_tokens` 64 or 256, no stop strings.", "",
        f"Every trace serves the same {s.get('requests_per_trace')} requests (`plan.jsonl`, seed {s.get('seed')}) on a "
        "fresh server, so each starts with a cold prefix cache.", "",
        "| trace | rate (req/s) | what | requests freed | steps | rows | rows/step p50 / p90 / max | tokens/step p50 / max | running p50 / max | waiting p90 / max | EOS (share) | length | prefix-hit requests | hit tokens (of prompt tokens) | preemptions (requests) | partial prefill chunks (steps with one) | mixed steps | rows past the last output (EOS / length) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, t in s["traces"].items():
        if "steps" not in t:
            lines.append(f"| {name} | {t.get('rate')} | {t.get('why')} | failed: {t.get('error')} |" + " |" * 14)
            continue
        st, rq, rw, pc, pe, ov = t["steps"], t["requests"], t["rows"], t["prefix_cache"], t["preemption"], t["overrun"]
        r, k, run, wait = st["rows_per_step"], st["tokens_per_step"], st["running"], st["waiting"]
        lines.append(
            f"| {name} | {t['rate']} | {t['why']}{' (failed: ' + t['error'] + ')' if t.get('error') else ''} | "
            f"{rq['freed_by_engine']} of {rq['sent']} | {st['with_tokens']} | {rw['total']} | "
            f"{r.get('p50')} / {r.get('p90')} / {r.get('max')} | {k.get('p50')} / {k.get('max')} | "
            f"{run.get('p50')} / {run.get('max')} | {wait.get('p90')} / {wait.get('max')} | "
            f"{rq['finish'].get('eos', 0)} ({rq['eos_fraction'] or 0:.1%}) | {rq['finish'].get('length', 0)} | "
            f"{pc['requests_with_hit']} | {pc['hit_tokens']} ({pc['hit_fraction'] or 0:.1%}) | "
            f"{pe['events']} ({pe['requests_preempted']}) | {rw['partial_prefill_chunks']} ({st['with_partial_chunk']}) | "
            f"{st['mixed_prefill_decode']} | {ov['rows']} ({ov['by_finish'].get('eos', 0)} / "
            f"{ov['by_finish'].get('length', 0)}) |")
    hostile = {name: t["adversary"] for name, t in s["traces"].items() if t.get("adversary")}
    if hostile:
        lines += ["", "## The hostile schedulers", "",
                  "What `vllm_adversarial_plugin` chose in each hostile trace, from its records (`raw/adversary.PID.jsonl`; "
                  "each step's own choices are its `adversary` in `steps.jsonl`).", "",
                  "| trace | seed | prefix cache | schedule calls | forced preemptions (requests; max per request) | calls "
                  "holding back (held share p50 / p90) | prefill rows cut (requests; cut to p50 tokens) | hits seen | "
                  "refused | cut to one block | errors |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for name, a in hostile.items():
            h = a["held_share"]
            lines.append(f"| {name} | {a.get('seed')} | {a.get('cache')} | {a['calls']} | {a['forced_preemptions']} "
                         f"({a['requests_force_preempted']}; {a['max_forced_per_request']}) | {a['calls_holding_back']} "
                         f"({h.get('p50')} / {h.get('p90')}) | {a['splits']} ({a['requests_split']}; "
                         f"{a['split_to'].get('p50')}) | {a['hits_seen']} | {a['hits_refused']} | {a['hits_cut']} | "
                         f"{a['errors']} |")
    if s.get("outputs"):
        lines += ["", "## Output ids across traces", "", *outputs_section(s["outputs"])]
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
