"""The static serving schedule: a run whose step counts, batch membership and admissions are a function of the public declaration only,
so the served circuit is static.  Torch-free.

A workload declares the static form with `sampling.ignore_eos: true`: every request runs exactly its `max_tokens` steps from its
arrival, so no generated token decides when a slot frees, who shares a launch, where a waiting prefill is cut by the token budget, or
when the next arrival is admitted.  The EOS a request may sample is a value: the circuit's done flag masks the steps after it, and the
host never acts on it.  The global Program's stop rule already reads the same declaration (`global_program.stop_rule`: ignore_eos -> every request
must reach its cap); `vllm_adapter.sampling_params` hands it to vLLM.

`check` holds a realized run to that form, failing by name on what the host decided from values:

    early stop         a request finished other than by length at its cap (an EOS, a stop id, max_model_len)
    preemption         a request's computed-token count went down (KV pressure: the host recomputed it)
    host prefix cache  a request read KV pages vLLM's hash match chose (`num_cached_tokens` > 0)

and records the realized schedule (`schedule_of`: per request in declared order, arrival, prefill forwards, token forwards, finish),
whose digest leaves out request ids and tokens.  Two runs of one declaration with different prompt contents must have the same
digest (`compare`): `m1_capture` holds its warm-up run (same lengths, random tokens) against the run of record.

vLLM's host prefix cache is refused before a static run (`host_prefix_cache_refusal`): its hash match decides which pages a request
reads, outside the circuit.  Prefix sharing comes back in the circuit-native form: a static page pool, each slot's attention wired to
its publicly capped candidate pages, and an in-circuit prefix match emitting the page mask.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping

SCHEMA = "verity-vllm/static-schedule/v1"
RULE = ("sampling.ignore_eos: every request finishes by length at exactly max_tokens, is never preempted and reads no KV block chosen by "
        "the host prefix cache; the realized schedule is then a function of the declaration (arrivals, prompt lengths, caps, engine args)")
SCHEDULE_FIELDS = ("arrive_step_realized", "first_step", "finish_step", "n_generated", "finish_reason", "prefill_steps", "token_steps")


def declared_static(sampling: Mapping[str, Any] | None) -> bool:
    return bool((sampling or {}).get("ignore_eos"))


def host_prefix_cache_refusal(enable_prefix_caching: Any) -> str | None:
    if not enable_prefix_caching:
        return None
    return ("the workload declares the static schedule (sampling.ignore_eos) but the engine has enable_prefix_caching on: vLLM's hash match "
            "would decide which KV pages a request reads, outside the circuit.  Serve it with --engine-arg enable_prefix_caching=false; "
            "prefix sharing needs the circuit-native form (static page pool + in-circuit prefix match)")


def preemptions(snapshots: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """{scheduler request id: times its computed-token count went down} over per-iteration `{rid: num_computed | (num_computed, ...)}`."""
    prev: dict[str, int] = {}
    out: dict[str, int] = {}
    for snap in snapshots:
        for rid, v in snap.items():
            nc = int(v[0] if isinstance(v, (tuple, list)) else v)
            if rid in prev and nc < prev[rid]:
                out[rid] = out.get(rid, 0) + 1
            prev[rid] = nc
    return out


def schedule_of(order: list[str], timing: Mapping[str, Any]) -> dict[str, Any]:
    """The realized schedule of `timing` (`run_requests`' record) over the requests in `order`, ids and tokens left out, and its digest."""
    recs = timing.get("requests") or {}
    rows = [{k: (recs.get(rid) or {}).get(k) for k in SCHEDULE_FIELDS} for rid in order]
    doc = {"engine_steps": timing.get("engine_steps"), "requests": rows}
    return {**doc, "digest": hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def check(requests: list[Mapping[str, Any]], timing: Mapping[str, Any], *, cached_tokens: Mapping[str, int | None] | None = None,
          preempted: Mapping[str, int] | None = None) -> dict[str, Any]:
    """`requests`: the declared ones (`request_id`, `max_tokens`), in order; `timing`: `run_requests`' record; `cached_tokens`: {request id:
    the engine's `num_cached_tokens`}; `preempted`: `preemptions(...)` keyed by any id that contains the request's."""
    recs = timing.get("requests") or {}
    problems: list[str] = []
    for r in requests:
        rid, cap = str(r["request_id"]), int(r["max_tokens"])
        rec = recs.get(rid) or {}
        reason, n = rec.get("finish_reason"), rec.get("n_generated")
        if any(str(x) != "length" for x in (reason if isinstance(reason, list) else [reason])) or n != cap:
            problems.append(f"{rid}: finished by {reason!r} after {n} of its {cap} tokens: the host stopped it early, so its step count "
                            f"depended on a generated value")
        hit = (cached_tokens or {}).get(rid)
        if hit:
            problems.append(f"{rid}: read {hit} prompt tokens' KV from blocks vLLM's prefix-cache hash match chose (host-decided pages)")
    for sid, k in sorted((preempted or {}).items()):
        problems.append(f"{sid}: preempted {k} time(s) and recomputed: KV pressure, not the declaration, decided its schedule "
                        f"(raise the KV pool or lower max_num_seqs)")
    return {"schema": SCHEMA, "rule": RULE, "ok": not problems, "problems": problems,
            "schedule": schedule_of([str(r["request_id"]) for r in requests], timing)}


def compare(reference: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    """Two `schedule_of` records of one declaration (different prompt contents): equal, or the first differing request and field named."""
    if reference.get("digest") == run.get("digest"):
        return {"ok": True, "digest": run.get("digest"), "problems": []}
    problems: list[str] = []
    if reference.get("engine_steps") != run.get("engine_steps"):
        problems.append(f"engine steps {reference.get('engine_steps')} vs {run.get('engine_steps')}")
    a, b = reference.get("requests") or [], run.get("requests") or []
    if len(a) != len(b):
        problems.append(f"{len(a)} vs {len(b)} requests")
    for i, (x, y) in enumerate(zip(a, b)):
        diff = [k for k in SCHEDULE_FIELDS if x.get(k) != y.get(k)]
        if diff:
            problems.append(f"request {i}: {', '.join(f'{k} {x.get(k)!r} vs {y.get(k)!r}' for k in diff[:3])}")
            break
    return {"ok": False, "digests": [reference.get("digest"), run.get("digest")],
            "problems": ["the schedule depends on the prompts' contents, not only on the declaration: " + "; ".join(problems or ["digests differ"])]}


def against_warmup(record: dict[str, Any] | None, warmup: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """`record` (the run of record's `check`) with `content_invariance` against the warm-up's, which fails it on a different schedule;
    unchanged when either run declared no static schedule."""
    if record is None or warmup is None or warmup.get("schedule") is None:
        return record
    record["content_invariance"] = compare(warmup["schedule"], record["schedule"])
    if not record["content_invariance"]["ok"]:
        record["ok"] = False
        record["problems"] += record["content_invariance"]["problems"]
    return record


def report_lines(record: Mapping[str, Any]) -> list[str]:
    ci = record.get("content_invariance")
    lines = [f"static schedule: {'ok' if record['ok'] else 'FAILED'} (digest {record['schedule']['digest'][:16]}, "
             + ("no warm-up to compare" if ci is None else f"warm-up {'equal' if ci['ok'] else 'DIFFERS'}") + ")"]
    if not record["ok"]:
        lines.append("REFUSE: the workload declares the static schedule (sampling.ignore_eos) and the run is not it:\n  "
                     + "\n  ".join(record["problems"][:8]))
    return lines
