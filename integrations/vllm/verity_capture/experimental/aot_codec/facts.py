"""[R8 C-codec] ScheduleHint extraction from an archived record + base-context variants.

A record directory (an extracted `<run>.small.tgz`) holds
  spec.json                 known_before_execution: request list (prompt_token_ids, max_tokens, arrive_step), engine args
  capture/log.jsonl.gz      RunHeader (profile: cache config) + one StepStart per engine step (rows, num_computed, seq_lens, logits_rows, ...)
  experiment.json           execution.realized.requests: n_generated, finish_reason, first/finish step (post-hoc)
  fold_summary.json         Program digest, n_instances, persisted program bytes (for the size ratios)

The **ScheduleHint** `h` is the structural residual of one run: everything in the StepStart events that shapes the logical Program and
is not a computed value (token ids) or a physical placement (block tables, slot mapping, storage views):

  h = {"requests": [{prompt_len, max_tokens, arrive_step, n_generated, finish_reason}...]   (submission order = spec.json order)
       "steps":    [{"rows": [[req_idx, num_computed, n_sched], ...]} ...]                    (row-group order as executed)
       "logits":   [[[req_idx, is_final], ...] per step]                                        (logits_rows order)}

`reconstruct_stepstarts(phi0, h)` regenerates the structural StepStart fields (rows, num_computed, seq_lens, logits_rows) and
`check_record(record_dir)` verifies that they equal the log bit-for-bit -- the codec's own exactness check, independent of C-builder's.

Base contexts (what the pre-execution builder knows):
  phi0  model + engine config + the request *contents* (prompt token ids -> prompt_len, max_tokens, sampling) in submission order
  phi1a phi0 + arrival schedule (arrive_step per request)
  phi1  phi1a + realized request lengths (n_generated per request)
  phi2  phi1 + other operational facts declared before execution (expected_engine_steps bound, KV pool size num_gpu_blocks)
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "aot-codec/schedule-hint/v1"
FINISH_REASONS = ["length", "stop", "abort", "other"]  # enum for A0; anything else -> "other" + flagged


def canonical_json(o: Any) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))


def digest(o: Any) -> str:
    return hashlib.sha256(canonical_json(o).encode()).hexdigest()


def _read_log(rec: Path):
    header = None
    steps = []
    samples = {}
    other = []
    with gzip.open(rec / "capture" / "log.jsonl.gz", "rt") as f:
        for line in f:
            e = json.loads(line)
            k = e.get("$e")
            if k == "RunHeader":
                header = e
            elif k == "StepStart":
                steps.append(e)
            elif k == "SampleEnd":
                samples[e["step"]] = e
            elif k in ("StepLink", "Note", "RunEnd"):
                other.append(e)
    steps.sort(key=lambda e: e["step"])
    return header, steps, samples, other


def load_record(rec: Path) -> dict:
    """Everything the extractor needs from one record dir (spec, header, StepStart events, realized)."""
    rec = Path(rec)
    spec = json.loads((rec / "spec.json").read_text())
    exp = json.loads((rec / "experiment.json").read_text()) if (rec / "experiment.json").exists() else {}
    fold = json.loads((rec / "fold_summary.json").read_text()) if (rec / "fold_summary.json").exists() else {}
    card = json.loads((rec / "card.json").read_text()) if (rec / "card.json").exists() else {}
    header, steps, samples, other = _read_log(rec)
    return {"dir": str(rec), "run_id": rec.name, "spec": spec, "experiment": exp, "fold": fold, "card": card,
            "header": header, "steps": steps, "samples": samples, "other": other}


def load_log_only(log_path: Path, run_id: str | None = None) -> dict:
    """A pseudo-record from a bare capture log (RunHeader + StepStart ...), for logs that have no spec.json / experiment.json
    (e.g. the sampling-variant runs under /private/tmp/ovs/runs).  The RunHeader's `workload` plays the role of spec.workload; realized
    facts are schedule-implied only (finish_reason defaults to `length` iff n_generated == max_tokens, else `other`)."""
    log_path = Path(log_path)
    rec = log_path.parent.parent if log_path.parent.name == "capture" else log_path.parent
    header, steps, samples, other = _read_log_file(log_path)
    if header is None:
        raise ValueError(f"{log_path}: no RunHeader")
    w = header.get("workload") or {}
    spec = {"workload": w, "configuration": {"checkpoint": {"repo": (header.get("model") or {}).get("repo", "?")}}}
    # the harness's tokens.json (next to the log) carries the generated ids per request: a request that stopped short of max_tokens
    # stopped on EOS -> realized finish_reason "stop" (the record has no experiment.json); n_generated is cross-checked by extract_hint
    experiment: dict = {}
    tokens_path = log_path.parent / "tokens.json"
    if tokens_path.exists():
        try:
            tk = json.loads(tokens_path.read_text())
            realized = []
            for q in tk.get("requests", []):
                g = q.get("generated_token_ids") or []
                realized.append({"request_id": q["request_id"], "n_generated": len(g),
                                 "finish_reason": "length" if len(g) >= int(q["max_tokens"]) else "stop"})
            experiment = {"execution": {"realized": {"requests": realized}}}
        except Exception:  # noqa: BLE001 -- tokens.json is optional evidence
            experiment = {}
    return {"dir": str(rec), "run_id": run_id or rec.name, "spec": spec, "experiment": experiment, "fold": {}, "card": {},
            "header": header, "steps": steps, "samples": samples, "other": other, "log_only": True}


def _read_log_file(path: Path):
    header, steps, samples, other = None, [], {}, []
    with gzip.open(path, "rt") as f:
        for line in f:
            e = json.loads(line)
            k = e.get("$e")
            if k == "RunHeader":
                header = e
            elif k == "StepStart":
                steps.append(e)
            elif k == "SampleEnd":
                samples[e["step"]] = e
            elif k in ("StepLink", "Note", "RunEnd"):
                other.append(e)
    steps.sort(key=lambda e: e["step"])
    return header, steps, samples, other


def logical_requests(spec: dict) -> list[dict]:
    """The engine-level request list in submission order.  A workload request with sampling `n > 1` is expanded by vLLM into n child
    requests with ids f"{k}_{request_id}" (SOURCE-INSPECTED vllm/v1/engine/parallel_sampling.py:92 at the pinned install), each with
    the parent's prompt and max_tokens -- derivable from phi, so the expansion costs no advice; every row group in the log refers to a
    child id.  `n` comes from the request (`n`) else workload.sampling.n else 1."""
    w = spec["workload"]
    default_n = int(((w.get("sampling") or {}).get("n")) or 1)
    out = []
    for q in w["requests"]:
        n = int(q.get("n") or default_n)
        if n <= 1:
            out.append(q)
        else:
            for k in range(n):
                out.append({**q, "request_id": f"{k}_{q['request_id']}", "parent_request_id": q["request_id"], "child_index": k})
    return out


def phi_from_record(r: dict) -> dict:
    """phi0 (the deterministic builder's base context) from the pre-execution spec + engine profile."""
    spec = r["spec"]
    reqs = logical_requests(spec)
    cache = (r["header"] or {}).get("profile", {}).get("cache", {})
    ea = spec["workload"].get("engine_args_required", {}) or {}
    mnbt = cache.get("max_num_batched_tokens") or ea.get("max_num_batched_tokens") or 8192
    B = int(cache.get("block_size", 16))
    return {
        "schema": "aot-codec/phi/v1",
        "run_id": r["run_id"],
        "model": spec["configuration"]["checkpoint"]["repo"],
        "engine": {
            "max_num_seqs": int(cache.get("max_num_seqs") or spec["configuration"].get("max_num_seqs") or 8),
            "max_num_batched_tokens": int(mnbt),
            "chunked_prefill": bool(cache.get("enable_chunked_prefill", True)),
            "prefix_caching": bool(cache.get("enable_prefix_caching", False)),
            "block_size": int(cache.get("block_size", 16)),
            # vLLM d9105ea80 enables the AsyncScheduler unless pooling / spec-decode / unsupported executor (SOURCE-INSPECTED
            # vllm/config/vllm.py:1273-1313): a request stopped by EOS at its k-th generated token still gets one placeholder decode
            # row in the next step (k+1 rows sampled, the last discarded); a max_tokens stop does not.  MEASURED on the eos4 record.
            "async_scheduling": bool(cache.get("async_scheduling", True)),
            # KV pool size: an operational fact fixed at engine start (profiled from GPU memory, or the engine arg
            # num_gpu_blocks_override) -> phi2 only (phi_variant strips it below phi2).  The RunHeader profile.cache does not record
            # it (harness gap, flagged in advice-audit.md); the workload's engine_args_required or an out-of-band override supplies it.
            "num_gpu_blocks": cache.get("num_gpu_blocks") or ea.get("num_gpu_blocks_override") or (r.get("phi_overrides") or {}).get("num_gpu_blocks"),
            # chunk cap (engine arg, phi0; default 0 = off): a prefill chunk is min(threshold if > 0, remaining) then min(., budget) --
            # SOURCE-INSPECTED vllm/v1/core/sched/scheduler.py:624-625 (running) and :1019-1033 (waiting); MEASURED gen/thr4.  Not in the
            # RunHeader either (harness gap): workload engine_args_required or an out-of-band override.
            "long_prefill_token_threshold": int(cache.get("long_prefill_token_threshold") or ea.get("long_prefill_token_threshold")
                                               or (r.get("phi_overrides") or {}).get("long_prefill_token_threshold") or 0),
        },
        # request contents in submission order (spec.json order): prompt token ids are phi0, so their lengths -- and the chain hashes of
        # their full KV blocks (what vLLM's prefix cache keys on) -- are derivable
        "requests": [{"request_id": q["request_id"], "prompt_len": len(q["prompt_token_ids"]), "max_tokens": int(q["max_tokens"]),
                      "block_hashes": block_hashes(q["prompt_token_ids"], B)} for q in reqs],
    }


def block_hashes(ids: list[int], B: int) -> list[str]:
    """Chain hash per *full* block of `ids` (h_k = sha256(h_{k-1} || block_k tokens), 16 hex): equal at k iff the token prefixes agree
    through block k -- the same equivalence vLLM's block hashes induce (the concrete hash function is irrelevant to the schedule)."""
    out, prev = [], b""
    for k in range(len(ids) // B):
        prev = hashlib.sha256(prev + json.dumps(ids[k * B:(k + 1) * B]).encode()).digest()
        out.append(prev.hex()[:16])
    return out


def phi_variant(phi0: dict, h: dict, level: str) -> dict:
    """phi0 | phi1a | phi1 | phi2 -- the later ones copy the named facts *from the realized hint* (they are 'known before execution'
    in the sense of the accounting: arrivals and request lengths are prescribed by the workload / declared by the operator)."""
    p = json.loads(json.dumps(phi0))
    p["level"] = level
    if level != "phi2":
        p["engine"]["num_gpu_blocks"] = None  # operational: known only at phi2
    if level == "phi0":
        return p
    for q, hq in zip(p["requests"], h["requests"]):
        q["arrive_step"] = hq["arrive_step"]
    if level == "phi1a":
        return p
    for q, hq in zip(p["requests"], h["requests"]):
        q["n_generated"] = hq["n_generated"]
    if level == "phi1":
        return p
    p["expected_engine_steps"] = h.get("_declared_engine_steps") or len(h["steps"])
    return p


def extract_hint(r: dict) -> tuple[dict, dict]:
    """ScheduleHint + audit facts from a loaded record.  Raises if a StepStart is not expressible (non-contiguous rows, unknown request)."""
    spec = r["spec"]
    reqs = logical_requests(spec)
    idx = {q["request_id"]: i for i, q in enumerate(reqs)}
    realized = {q["request_id"]: q for q in (r["experiment"].get("execution", {}).get("realized", {}).get("requests", []) or [])}
    audit = {"unexpressible": [], "events": [], "n_steps": len(r["steps"])}
    if any("parent_request_id" in q for q in reqs):
        audit["events"].append({"kind": "sampling n>1: request expanded into child requests (phi-derivable)", "n_children": len(reqs) - len(spec["workload"]["requests"])})
    steps_h, logits_h = [], []
    first_seen = {}
    prev_end = {}
    # tokens sampled so far per request: a row samples one kept token iff it reaches the end of the request's *known* sequence
    # (prompt + tokens generated so far) -- a full or final prompt chunk, a decode row, an async placeholder row, the *last* chunk of a
    # post-preemption re-prefill (MEASURED gen/preempt4 one chunk, gen/pre6 three chunks under the chunk cap); every other chunk samples nothing
    samp = [0] * len(reqs)
    for e in r["steps"]:
        t = e["step"]
        groups = []  # (req_idx, num_computed, n_sched)
        for rid, pos in e["rows"]:
            if rid not in idx:
                raise ValueError(f"step {t}: row for unknown request {rid}")
            i = idx[rid]
            if groups and groups[-1][0] == i:
                if pos != groups[-1][1] + groups[-1][2]:
                    audit["unexpressible"].append({"step": t, "kind": "non-consecutive positions", "req": i, "pos": pos})
                groups[-1][2] += 1
            else:
                if any(g[0] == i for g in groups):
                    audit["unexpressible"].append({"step": t, "kind": "non-contiguous rows for one request", "req": i})
                nc = e["num_computed"][rid]
                if pos != nc:
                    audit["unexpressible"].append({"step": t, "kind": "first row position != num_computed", "req": i, "pos": pos, "num_computed": nc})
                groups.append([i, nc, 1])
        for g in groups:
            i, nc, n = g
            rid = reqs[i]["request_id"]
            if e["seq_lens"][rid] != nc + n:
                audit["unexpressible"].append({"step": t, "kind": "seq_len != num_computed+n_sched", "req": i})
            if i not in first_seen:
                first_seen[i] = t
                if nc != 0:
                    audit["events"].append({"step": t, "kind": "first step with num_computed>0 (prefix-cache hit or resumed)", "req": i, "num_computed": nc})
            elif i in prev_end and nc != prev_end[i]:
                audit["events"].append({"step": t, "kind": "num_computed discontinuity (preemption/recompute?)", "req": i, "num_computed": nc, "expected": prev_end[i]})
            prev_end[i] = nc + n
            L = len(reqs[i]["prompt_token_ids"])
            if nc < L and nc + n < L:
                audit["events"].append({"step": t, "kind": "partial prefill chunk", "req": i, "num_computed": nc, "n_sched": n, "prompt_len": L})
        if e.get("num_tokens_padded", e["num_tokens"]) != e["num_tokens"]:
            audit["events"].append({"step": t, "kind": "num_tokens_padded != num_tokens", "num_tokens": e["num_tokens"], "padded": e["num_tokens_padded"]})
        if e["num_reqs"] != len(groups):
            audit["unexpressible"].append({"step": t, "kind": "num_reqs != row groups", "num_reqs": e["num_reqs"], "groups": len(groups)})
        # logits rows: (req_idx, is_final) where is_final = the row reaches the end of the known sequence (sample kept)
        lg = []
        gpos = {g[0]: g for g in groups}
        for rid, pos in e["logits_rows"]:
            i = idx[rid]
            g = gpos.get(i)
            if g is None or pos != g[1] + g[2] - 1:
                audit["unexpressible"].append({"step": t, "kind": "logits row not at the request's last scheduled token", "req": i, "pos": pos})
            L = len(reqs[i]["prompt_token_ids"])
            lg.append([i, 1 if (g is not None and g[1] + g[2] >= L + samp[i]) else 0])
        for i, nc, n in groups:
            if nc + n >= len(reqs[i]["prompt_token_ids"]) + samp[i]:
                samp[i] += 1
        n_sampled = len((r["samples"].get(t) or {}).get("sampled_token_ids", []) or [])
        if r["samples"].get(t) is not None and n_sampled != len(lg):
            audit["events"].append({"step": t, "kind": "sampled_token_ids count != logits rows", "sampled": n_sampled, "logits_rows": len(lg)})
        steps_h.append({"rows": [list(g) for g in groups]})
        logits_h.append(lg)
    # request-level facts
    hreqs = []
    for i, q in enumerate(reqs):
        rz = realized.get(q["request_id"], {})
        L = len(q["prompt_token_ids"])
        # realized n_generated from the schedule itself (rule above; incl. an async placeholder row)
        sampled = samp[i]
        M = int(q["max_tokens"])
        async_sched = bool(((r["header"] or {}).get("profile", {}).get("cache", {})).get("async_scheduling", True))
        if "n_generated" in rz:
            n_gen = int(rz["n_generated"])
            fr = rz.get("finish_reason", "length" if n_gen == M else "stop")
            extra = sampled - n_gen
            if not (extra == 0 or (extra == 1 and async_sched and fr != "length")):
                audit["unexpressible"].append({"kind": "realized n_generated != schedule-implied", "req": i, "realized": n_gen, "sampled_rows": sampled})
            elif extra == 1:
                audit["events"].append({"kind": "EOS stop: one async placeholder decode row after the stop token", "req": i, "n_generated": n_gen})
        elif sampled < M and async_sched:  # stopped short: under async scheduling the last sampled row is the placeholder
            n_gen, fr = sampled - 1, "stop"
            audit["events"].append({"kind": "EOS stop (schedule-implied, no realized record)", "req": i, "n_generated": n_gen})
        else:
            n_gen = sampled
            fr = "length" if n_gen == M else "other"
        if fr not in FINISH_REASONS:
            audit["events"].append({"kind": "finish_reason not in enum", "req": i, "finish_reason": fr})
            fr = "other"
        arrive = first_seen.get(i)
        if arrive is None:
            audit["unexpressible"].append({"kind": "request never scheduled", "req": i})
            arrive = -1
        if (rz.get("arrive_step_realized", rz.get("arrive_step", q.get("arrive_step"))) or 0) != arrive:  # absent declared arrival = step 0
            audit["events"].append({"kind": "first step != declared arrive_step", "req": i, "declared": q.get("arrive_step"), "realized_first_step": arrive})
        hreqs.append({"prompt_len": L, "max_tokens": int(q["max_tokens"]), "arrive_step": arrive, "n_generated": n_gen, "finish_reason": fr})
    h = {"schema": SCHEMA, "requests": hreqs, "steps": steps_h, "logits": logits_h,
         "_declared_engine_steps": spec["workload"].get("expected_engine_steps")}
    return h, audit


def strip_private(h: dict) -> dict:
    return {k: v for k, v in h.items() if not k.startswith("_")}


def reconstruct_stepstarts(phi0: dict, h: dict) -> list[dict]:
    """Structural StepStart fields implied by (phi0, h): rows [(request_id,pos)], num_computed, seq_lens, logits_rows, num_reqs, num_tokens."""
    rids = [q["request_id"] for q in phi0["requests"]]
    out = []
    for t, (s, lg) in enumerate(zip(h["steps"], h["logits"])):
        rows, nc, sl = [], {}, {}
        for i, c, n in s["rows"]:
            rows.extend([rids[i], c + k] for k in range(n))
            nc[rids[i]] = c
            sl[rids[i]] = c + n
        last = {i: c + n - 1 for i, c, n in s["rows"]}
        out.append({"step": t, "rows": rows, "num_computed": nc, "seq_lens": sl, "logits_rows": [[rids[i], last[i]] for i, _ in lg],
                    "num_reqs": len(s["rows"]), "num_tokens": len(rows)})
    return out


def check_record(r: dict, h: dict) -> dict:
    """Exactness: reconstructed structural StepStart fields == the log's.  Returns {ok, mismatches}."""
    phi0 = phi_from_record(r)
    rec = reconstruct_stepstarts(phi0, h)
    mism = []
    if len(rec) != len(r["steps"]):
        mism.append({"kind": "n_steps", "have": len(rec), "want": len(r["steps"])})
    for got, e in zip(rec, r["steps"]):
        for k in ("rows", "num_computed", "seq_lens", "logits_rows", "num_reqs", "num_tokens"):
            if got[k] != e[k]:
                mism.append({"step": e["step"], "field": k})
    return {"ok": not mism, "mismatches": mism[:20], "n_mismatches": len(mism)}


def program_stats(r: dict) -> dict:
    fs = r["fold"] or {}
    pers = (fs.get("persisted") or {}).get("program") or {}
    return {"program_digest": fs.get("digest"), "program_bytes": pers.get("bytes"), "n_instances": fs.get("n_instances"),
            "n_steps_fold": fs.get("n_steps"), "verdict": (r["card"] or {}).get("verdict")}


def io_stats(r: dict, h: dict) -> dict:
    """Ordinary request I/O: prompt token ids in (as int32/varint-ish; we count 2 bytes/token of ids as a conservative wire size) and generated ids out."""
    n_in = sum(q["prompt_len"] for q in h["requests"])
    n_out = sum(q["n_generated"] for q in h["requests"])
    prompt_bytes = len(canonical_json([q["prompt_token_ids"] for q in r["spec"]["workload"]["requests"]]).encode())
    return {"prompt_tokens": n_in, "generated_tokens": n_out, "io_tokens": n_in + n_out, "prompt_json_bytes": prompt_bytes,
            "io_bytes_2B_per_token": 2 * (n_in + n_out)}


if __name__ == "__main__":  # quick look
    import sys

    r = load_record(Path(sys.argv[1]))
    h, audit = extract_hint(r)
    print(canonical_json({"requests": h["requests"], "n_steps": len(h["steps"]), "audit": audit})[:3000])
    print(check_record(r, h))
