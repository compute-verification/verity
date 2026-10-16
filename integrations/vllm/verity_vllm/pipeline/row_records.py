"""The records `verity-vllm row` writes and reads beside a row: the row's shape, the Build and Match summaries, the
Program digest of record and the weights-of-record choices.  Torch-free; each function reads the row dir it is given.

A stage's summary (``build_summary.json``, ``match_summary.json``) is what the research outputs, the regression harness
and the verdict record read; the Match summary's decision is the Match stage's verdict.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import os
import time
from typing import Any

from verity_vllm.commit.partition import word_check_refusal
from verity_vllm.config import checkpoint_dir


def load_json(path: str) -> Any:
    """The JSON document at ``path``, None when it is absent or unreadable."""
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def row_shape(wl: str, checkpoints: str, hf_home: str) -> dict[str, Any]:
    """The row's batch and envelope, the request wrapper's default shape (LP, T), the Match snapshot steps and the pinned
    checkpoint dir, from the workload manifest.

    The request wrapper's T is the number of DECODE steps (max_output - 1: the prefill step samples token 1).  A
    stochastic row's request Program carries the split count of the request its shape identifies, so its default shape
    is the first declared request's (prompt_len, max_tokens - 1), and it snapshots every engine step (the sampler value
    check is complete only over every sampled row); greedy rows keep the envelope and snapshot the two X-09 steps."""
    w = json.load(open(wl)); s = w["sweep"]
    lp = max(s["input_lengths"]); t = max(s["output_caps"]) - 1
    sp = s.get("sampling_params") or {}
    snap = "0,1"
    if float(sp.get("temperature", 0) or 0) > 0 and w.get("requests"):
        r0 = w["requests"][0]
        lp = int(r0.get("prompt_len") or len(r0.get("prompt_token_ids") or [])); t = int(r0["max_tokens"]) - 1
        snap = "all"
    ck = checkpoint_dir(w["model"], w["tokenizer_revision"], hf_home, checkpoints)
    return {"B": int(s["batch"]), "I": int(s["max_input"]), "O": int(s["max_output"]), "LP": lp, "T": t, "snap": snap, "ck": ck}


def declared_target(wl: str, protocols: tuple[str, ...] | None = None, row: str | None = None) -> str:
    """The row's declared TargetProfile block as sorted JSON (Build's ``--target``), "" when it declares none: the workload's, over
    the device-class profile of ``row``'s id (`target_family.declared_profile`); ``protocols`` (the row's set,
    `protocol_options.at_row`) is laid over it as the target's ``protocols``."""
    from verity_vllm.target_family import declared_profile
    t = declared_profile(row, json.load(open(wl))) or None
    if protocols:
        t = {**(t or {}), "protocols": list(protocols)}
    return json.dumps(t, sort_keys=True) if t else ""


def request_shapes(wl: str) -> set[tuple[int, int]]:
    """Every distinct (prompt length, cap - 1) of the manifest's requests: the shapes the workload Program composes."""
    s = json.load(open(wl))["sweep"]
    return {(int(p), int(c) - 1) for p, c in zip(s["input_lengths"], s["output_caps"])}


def extra_shapes(wl: str, lp: int, t: int) -> list[tuple[int, int]]:
    """The batch's other request shapes: every distinct (prompt length, cap - 1) of the manifest but the (LP, T) the
    request wrapper was derived for, largest first (a derive's cost grows with LP + T, so parallel derives start with the
    longest)."""
    main = (int(lp), int(t))
    return [(p, c) for p, c in longest_first(request_shapes(wl)) if (p, c) != main]


def longest_first(shapes: Any) -> list[tuple[int, int]]:
    """(prompt length, decode steps) shapes, the longest LP + T first (then the longest prompt): a derive's cost grows with LP + T."""
    return sorted(shapes, key=lambda pc: (-(pc[0] + pc[1]), -pc[0]))


def derived_ok(d: str) -> bool:
    """A derived Program dir whose derive SUCCEEDED (derive_step writes result.json {"ok": false} on a crash too)."""
    try:
        return json.load(open(f"{d}/result.json")).get("ok") is True
    except (OSError, ValueError, AttributeError):
        return False


#: a request derive's peak host memory: a base, a term per Call and attention's term per layer x tokens^2 (every attention Call
#: lists each earlier key and value row), fit over the 89 request derives of the 2026-09-29 epoch's stored Builds (#11, #57, #67,
#: #68, #74, #75: within 5% on the large ones), with Calls bounded by the most Calls per (token, layer) measured (54: Qwen3-4B-FP8,
#: Qwen3-30B-A3B) and a 1.2 margin
DERIVE_BASE_BYTES = 2000 << 20
DERIVE_BYTES_PER_CALL = 6950
DERIVE_BYTES_PER_LAYER_TOKEN2 = 362
DERIVE_CALLS_PER_TOKEN_LAYER = 60
DERIVE_MARGIN = 1.2
#: a TP rank's GP-01 compose (`global-program`) peaks above its rank's largest request derive while it decodes that descriptor:
#: fitted on one Build only, Qwen3-235B-A22B TP8 with one shape (LP256 T31, run r20261003-070729-359b): 20.8 GiB per compose
#: against 14.1 GiB per derive (1.48x); a compose's planned peak is this multiple of its largest shape's `derive_bytes`
COMPOSE_PEAK_RATIO = 1.5


def derive_bytes(lp: int, t: int, layers: int) -> int:
    """The planned peak host memory of the request derive of shape (LP, T) for a model of `layers` decoder layers."""
    tokens = int(lp) + int(t) + 1
    calls = DERIVE_CALLS_PER_TOKEN_LAYER * layers * tokens
    return int(DERIVE_MARGIN * (DERIVE_BASE_BYTES + DERIVE_BYTES_PER_CALL * calls + DERIVE_BYTES_PER_LAYER_TOKEN2 * layers * tokens * tokens))


def model_layers(ck: str) -> int | None:
    """The decoder layers of the checkpoint at `ck` (`num_hidden_layers` of its config.json, or of its `text_config`), None when
    it does not say."""
    cfg = load_json(os.path.join(ck, "config.json")) if ck else None
    for c in (cfg, (cfg or {}).get("text_config")):
        n = c.get("num_hidden_layers") if isinstance(c, dict) else None
        if isinstance(n, int) and n > 0:
            return n
    return None


def _memory_headroom() -> int:
    """cgroup memory.max - memory.current, else MemAvailable."""
    def rd(p: str) -> int | None:
        try:
            v = open(p).read().strip(); return int(v) if v.isdigit() else None
        except OSError:
            return None
    mx = rd("/sys/fs/cgroup/memory.max") or rd("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    cur = rd("/sys/fs/cgroup/memory.current") or rd("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if mx is None or mx >= (1 << 60) or cur is None:
        kv = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in open("/proc/meminfo") if ln.split()[1].isdigit()}
        return kv.get("MemAvailable", 0)
    return mx - cur


def _cpus() -> int:
    return len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)


def build_jobs_auto(shapes: list[tuple[int, int]] | None = None, layers: int | None = None, *, derives: int | None = None,
                    peak_ratio: float = 1.0, headroom: int | None = None, cpus: int | None = None, budget_gb: float | None = None) -> int:
    """``--build-jobs auto`` (the default): as many concurrent derives as 80% of the memory headroom (or of `budget_gb`, when smaller)
    holds at `peak_ratio` x the largest shape's `derive_bytes`, at most one per CPU this process may use and one per derive (`derives`:
    by default the step's and each shape's; a TP Build's are every rank's).  Without the shapes or the model's layers: that memory /
    (`peak_ratio` x 20 GiB), 1..8 (and at most `derives`).  A derive that dies of OOM is rc 137, a Build FAIL by rc, never a wrong
    Program."""
    head = _memory_headroom() if headroom is None else headroom
    if budget_gb is not None:
        head = min(head, int(budget_gb * (1 << 30)))
    if not shapes or not layers:
        return max(1, min(8, derives or 8, head // int(peak_ratio * (20 << 30))))
    per = int(peak_ratio * max(derive_bytes(lp, t, layers) for lp, t in shapes))
    n = len(shapes) + 1 if derives is None else derives
    return max(1, min(_cpus() if cpus is None else cpus, n, int(0.8 * head) // per))


def build_summary(d: str) -> str:
    """Write ``build_summary.json`` (step, request and every other request shape's derive) -> PASS | FAIL."""
    out: dict[str, Any] = {}
    for w in ("step", "request"):
        p = f"{d}/build_{w}/result.json"
        r = json.load(open(p)) if os.path.exists(p) else {"ok": False, "message": "no result.json"}
        art = f"{d}/build_{w}/artifact.json"
        a = json.load(open(art)) if os.path.exists(art) else {}
        out[w] = {"ok": bool(r.get("ok")), "unsupported": r.get("unsupported"), "exception": r.get("exception"), "message": (r.get("message") or "")[:300],
                  "program_digest": (r.get("program") or {}).get("digest") or a.get("program_digest"), "identity": a.get("identity"),
                  "correspondence_digest": (r.get("program") or {}).get("correspondence_digest") or a.get("correspondence_digest"),
                  "target": (a.get("target_profile") or {}), "applicability": a.get("applicability")}
    for p in sorted(glob.glob(f"{d}/build_request_LP*_T*/result.json")):
        r = json.load(open(p)); w = os.path.basename(os.path.dirname(p))
        out[w] = {"ok": bool(r.get("ok")), "unsupported": r.get("unsupported"), "message": (r.get("message") or "")[:300], "program_digest": (r.get("program") or {}).get("digest"),
                  "correspondence_digest": (r.get("program") or {}).get("correspondence_digest")}
    json.dump(out, open(f"{d}/build_summary.json", "w"), indent=1, default=str)
    return "PASS" if all(v["ok"] for v in out.values()) else "FAIL"


def workload_digest(d: str) -> str:
    """GP-01's workload digest from ``build_workload/workload_program.json`` ("" when absent)."""
    try:
        w = json.load(open(f"{d}/build_workload/workload_program.json"))
    except (OSError, ValueError):
        return ""
    return w.get("workload_digest") or w.get("program_digest") or w.get("digest") or ""


def compose_reason(d: str) -> str:
    """The reason and message a failed workload Program compose recorded ("" when unreadable)."""
    try:
        w = json.load(open(f"{d}/build_workload/workload_program.json"))
    except (OSError, ValueError):
        return ""
    return f"{w.get('reason')} {w.get('message') or ''}"


def add_workload(d: str, wd: str, rc: int) -> None:
    """Record the workload Program (GP-01) in ``build_summary.json``: its digest is the row's Program digest; a refusal
    of its declared applicability is a named Build FAIL, not a compose error."""
    s = json.load(open(f"{d}/build_summary.json"))
    W = json.load(open(f"{d}/build_workload/workload_program.json")) if os.path.exists(f"{d}/build_workload/workload_program.json") else {}
    s["workload"] = {"ok": rc == 0 and bool(wd), "workload_digest": wd or None, "mechanism": "GP-01 cb_a.global_program", "artifact": "build_workload/workload_program.json",
                     "reason": W.get("reason"), "message": W.get("message"), "applicability": W.get("applicability"),
                     "stop_rule": W.get("stop_rule")}
    json.dump(s, open(f"{d}/build_summary.json", "w"), indent=1, default=str)


def build_line(d: str) -> tuple[str, str]:
    """The step Program digest and ``<manifest digest> <identities>`` (``none``) for the Build stage's stages.txt line."""
    digest = (json.load(open(f"{d}/build_summary.json"))["step"].get("program_digest") or "")[:16]
    try:
        m = json.load(open(f"{d}/manifest.json"))
        man = f"{str(m.get('manifest_digest'))[:16]} {m.get('populations', {}).get('identities')}"
    except (OSError, ValueError):
        man = "none"
    return digest, man


def missing_derived(d: str) -> list[tuple[int, int]]:
    """The request shapes X-09 named as not derived (``match_decomp.json`` missing_derived)."""
    return [(int(lp), int(t)) for lp, t in json.load(open(f"{d}/match_decomp.json"))["missing_derived"]]


def decomp_rc(d: str) -> int:
    """The compare stage's rc as batch_decomp's CLI returns it, read from ``match_decomp.json``: 0 PASS, 4 missing
    derived shapes, 3 otherwise."""
    p = f"{d}/match_decomp.json"
    try:
        v = json.load(open(p)).get("verdict") if os.path.exists(p) else None
    except (OSError, ValueError):
        return 3
    return 0 if v == "PASS" else 4 if v == "MISSING_DERIVED" else 3


def fold_steps(d: str, default: int) -> int:
    """The served decode steps of a B=1 fold (``match/fold_summary.json`` n_steps - 1), ``default`` when unreadable."""
    try:
        return int(json.load(open(f"{d}/match/fold_summary.json")).get("n_steps") or 0) - 1
    except (OSError, ValueError, TypeError):
        return default


def is_stochastic(wl: str) -> bool:
    """The workload's sampling variant has temperature > 0."""
    w = json.load(open(wl)); v = str((w.get("sampling") or {}).get("variant", "greedy"))
    p = (w.get("variants") or {}).get(v) or {"temperature": 0.0}
    return float(p.get("temperature", 0.0)) > 0.0


def match_summary(d: str, rcm: int, rcc: int, rcg: int, t0: int, rcs: int, resumed: bool) -> str:
    """Write ``match_summary.json`` -> PASS | FAIL: the Match stage's verdict.

    PASS = the per-request comparison (B=1: canonical instance sequences equal, same step count, no differing operand;
    B>1: the X-09 batch decomposition) and the global Match (GM-01) pass, the control and instrumented tokens are equal,
    the record was produced by this attempt (or is an explicit complete ``resumed`` CPU-phase record), and a stochastic
    row's sampler value check passes (``rcs`` -1 = a greedy row)."""
    cmp = json.load(open(f"{d}/match_compare.json")) if os.path.exists(f"{d}/match_compare.json") else {}
    c = cmp.get("compare") or {}
    tot = c.get("totals") or {}
    steps_a, steps_b = (c.get("a") or {}).get("steps"), (c.get("b") or {}).get("steps")
    canon = (c.get("canonical_sha256") or {}).get("equal")
    # PASS = canonical instance sequences equal (spec sequence + operand wiring per step), same step count, no differing operands
    verdict = None
    decomp = json.load(open(f"{d}/match_decomp.json")) if os.path.exists(f"{d}/match_decomp.json") else None
    if decomp is not None:
        # B>1: the batch decomposition is the verdict -- every request canonical-equal to its own (LP_r, T_r) Program, every instance
        # attributed, no cross-request reads, no missing derivation
        reqs = decomp.get("requests") or []
        canon = all(r.get("canonical_equal") is True for r in reqs) and bool(reqs)
        tot = {"requests": len(reqs), "operands_equal": sum(int((r.get("totals") or {}).get("operands_equal", 0) or 0) for r in reqs),
               "operands_different": sum(int((r.get("totals") or {}).get("operands_different", 0) or 0) for r in reqs),
               "unattributed": (decomp.get("attribution") or {}).get("unattributed"), "xreq": decomp.get("xreq_total"),
               "shapes": sorted({f"LP{r.get('LP')}_T{r.get('T')}" for r in reqs}),
               # the operand counts are X-09's POSITIONAL (sequence) comparison; under the dag criterion a request step that is a
               # dependency-safe permutation of its component's counts its displaced operands here while its DAG multiset is equal
               "within_step_criterion": decomp.get("within_step_criterion"),
               "operands_note": "positional comparison counts (sequence); verdict per within_step_criterion, see match_decomp.json per request"}
        steps_a = steps_b = sum(int(r.get("steps") or 0) for r in reqs)
        verdict = "PASS" if decomp.get("verdict") == "PASS" else f"FAIL batch_decomp {decomp.get('verdict')}: {'; '.join(decomp.get('reasons') or [])[:300]}"
    elif cmp:
        verdict = "PASS" if (canon is True and steps_a == steps_b and int(tot.get("operands_different", 0) or 0) == 0) else \
                  f"FAIL canonical_equal={canon} steps={steps_a}/{steps_b} operands_different={tot.get('operands_different')} hist_diff={len(cmp.get('histogram_diff_T_normalised') or {})}"
    stages = {}
    sp = f"{d}/match/stages.json"
    if os.path.exists(sp):
        for s in json.load(open(sp)).get("stages", []):
            stages[s.get("name") or s.get("stage")] = {"rc": s.get("returncode", s.get("rc")), "status": s.get("status"), "reason": (s.get("reason") or "")[:200]}
    # the record this verdict is about must have been PRODUCED BY THIS ATTEMPT: when the row dir is reused and the GPU capture fails, the
    # Match keeps the previous attempt's capture/control record, the fold + GM-01 run on that stale record and this stage would report
    # PASS with rcm=1.  The record's log must be newer than the stage start and the capture/control stages must have rc 0 in stages.json.
    rec_log = f"{d}/match/capture/log.jsonl.gz"
    rec_mtime = os.path.getmtime(rec_log) if os.path.exists(rec_log) else None
    # a CPU-phase resume is an EXPLICIT resume over the preserved capture: the record is older than this stage by construction, so the
    # mtime rule cannot apply -- the resumed record is accepted when it is complete (capture + control tokens.json beside the log) and
    # labelled `resumed` with its mtime.  Without it the rule of record stands: the log must be newer than the stage start.
    rec_complete = rec_mtime is not None and all(os.path.exists(f"{d}/match/{p}/tokens.json") for p in ("capture", "control"))
    rec_fresh = rec_mtime is not None and (rec_mtime >= t0 - 5 or (resumed and rec_complete))
    # the capture/control stage records must say rc 0; when the stage record was set aside after an interrupted save a resume has none
    # for the GPU stages -- accepted on the completion markers above, labelled stage_record_absent
    rec_stage_absent = resumed and not any(k in stages for k in ("capture", "control"))
    rec_stage_ok = all((stages.get(k) or {}).get("rc") == 0 for k in ("capture", "control")) or (rec_stage_absent and rec_complete)
    record = {"fresh": bool(rec_fresh and rec_stage_ok), "log_mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(rec_mtime)) if rec_mtime else None,
              "stage_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
              "resumed": resumed, "resumed_record_complete": rec_complete if resumed else None, "stage_record_absent": rec_stage_absent,
              "capture_rc": (stages.get("capture") or {}).get("rc"), "control_rc": (stages.get("control") or {}).get("rc"),
              "reason": None if (rec_fresh and rec_stage_ok) else
                        ("record predates this match stage (stale record from a previous attempt)" if not rec_fresh else
                         f"capture/control stage failed: capture rc={(stages.get('capture') or {}).get('rc')} control rc={(stages.get('control') or {}).get('rc')}")}
    tok_eq = None
    try:
        ct = json.load(open(f"{d}/match/control/tokens.json")); it = json.load(open(f"{d}/match/capture/tokens.json"))
        # tokens.json = {requests: [{request_id, prompt_token_ids, generated_token_ids, ...}], per_step_sampled, timing, observer, ...}:
        # compare the TOKENS per request (prompt + generated ids), not the whole document (timing / observer / profile_id always differ).

        def toks(doc):
            if isinstance(doc, dict) and "requests" in doc:
                return [(r.get("request_id"), r.get("prompt_token_ids"), r.get("generated_token_ids")) for r in doc["requests"]]
            return doc.get("tokens", doc) if isinstance(doc, dict) else doc
        a = toks(ct); b = toks(it)
        tok_eq = (a == b) and len(a) > 0
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        tok_eq = None
    fold_ok = os.path.exists(f"{d}/match/program.json") and os.path.exists(f"{d}/match/instances.jsonl")
    # the global check is the Match criterion; the per-request comparison above is evidence inside it
    G = json.load(open(f"{d}/global_match.json")) if os.path.exists(f"{d}/global_match.json") else {}
    gm_verdict = G.get("verdict")
    gm_ok = gm_verdict == "PASS" and rcg == 0
    if not G:
        gm_verdict = {97: "NOT RUN", 98: "NO WORKLOAD PROGRAM", 99: "NO FOLD", 124: "TIMEOUT"}.get(rcg, f"rc={rcg}")
    gm = {"comparator": G.get("comparator"), "rules_version": G.get("rules_version"), "rc": rcg, "verdict": gm_verdict, "failed_checks": G.get("failed_checks"),
          "checks": {c.get("id"): c.get("status") for c in (G.get("checks") or [])}, "within_step_criterion": G.get("within_step_criterion"),
          "workload_digest": (G.get("global_program") or {}).get("workload_digest"), "achieved_concurrency": G.get("achieved_concurrency"),
          "generated_lengths": {e.get("request_id"): e.get("generated_length") for e in (G.get("requests") or [])},
          "eos_truncated_requests": G.get("eos_truncated_requests"),
          "eos_verified": {e["request_id"]: e.get("eos_verified") for e in (G.get("requests") or []) if "eos_verified" in e} or None,
          # the two lengths kept distinct per request + the Program's declared LAG vs the record's attestation (GM-01 res.lag)
          "served_output_lens": {e.get("request_id"): e.get("served_output_len") for e in (G.get("requests") or []) if "served_output_len" in e} or None,
          "executed_prefix_lens": {e.get("request_id"): e.get("executed_prefix_len") for e in (G.get("requests") or []) if "executed_prefix_len" in e} or None,
          "lag": G.get("lag"), "alerts": G.get("alerts"),
          # the declared stop rule (set + source) beside every verdict with an early stop: outside the digest; the chronology
          # check is the one global-match check that records it
          "stop_rule": (next((c.get("stop_rule") for c in (G.get("checks") or []) if "stop_rule" in c), None)
                        if G.get("eos_truncated_requests") else None),
          "seconds": G.get("seconds"), "artifact": "global_match.json"}
    # the sampler value check (stochastic rows only; rcs == -1 -> a greedy row, not applicable)
    stoch = None
    if rcs != -1:
        sr = json.load(open(f"{d}/match_stoch.json")) if os.path.exists(f"{d}/match_stoch.json") else {}
        ss = sr.get("summary") or {}
        stoch = {"rc": rcs, "verdict": ss.get("verdict", "NOT-RUN"), "rows": ss.get("rows"), "device_facts_all_match": ss.get("device_facts_all_match"),
                 "masked_rows_all_equal": ss.get("masked_rows_all_equal"), "reasons": (sr.get("reasons") or [])[:6], "variant": sr.get("variant")}
        if verdict == "PASS" and (rcs != 0 or stoch["verdict"] != "PASS"):
            verdict = f"FAIL stoch value check {stoch['verdict']}: {'; '.join(stoch['reasons'])[:400]}"
    ok = (verdict == "PASS") and fold_ok and (tok_eq is not False) and gm_ok and record["fresh"] and (rcs in (-1, 0))   # the sampler value check must pass on a stochastic row
    if not record["fresh"] and verdict == "PASS":
        verdict = f"FAIL record: {record['reason']}"
    shape = json.load(open(f"{d}/match_shape.json")) if os.path.exists(f"{d}/match_shape.json") else None  # served (LP, T_r) at B=1
    json.dump({"run_config_rc": rcm, "compare_rc": rcc, "compare_verdict": verdict, "canonical_equal": canon, "steps": [steps_a, steps_b], "totals": tot,
               "request_shape": shape, "record": record, "stoch_value_check": stoch,
               "histogram_diff_T_normalised": cmp.get("histogram_diff_T_normalised"), "fold_ok": fold_ok,
               "comparator": (G.get("comparator") or "cb_a.global_match/v1") + (" (X-09 batch_decomp inside)" if decomp is not None else " (cb_a.compare inside)"),
               "per_request_comparator": "cb_a.batch_decomp (X-09)" if decomp is not None else "cb_a.compare",
               "global_match": gm,
               "control_vs_instrumented_tokens_equal": tok_eq, "stages": stages, "pass": ok}, open(f"{d}/match_summary.json", "w"), indent=1, default=str)
    return "PASS" if ok else "FAIL"


def match_line(d: str) -> str:
    """The Match stage's stages.txt detail, from ``match_summary.json``."""
    s = json.load(open(f"{d}/match_summary.json")); g = s.get("global_match") or {}
    return " ".join(map(str, ("verdict", s["compare_verdict"], "global", g.get("verdict"), g.get("failed_checks") or "", g.get("within_step_criterion") or "",
                              "concurrency", (g.get("achieved_concurrency") or {}).get("observed_max"), "tokens_equal",
                              s["control_vs_instrumented_tokens_equal"], "fold", s["fold_ok"])))


def served_shape(d: str) -> str:
    """``LP=.. T_r=.. cap_T=..`` when a B=1 request stopped before its cap (``match_shape.json``), else ""."""
    try:
        s = json.load(open(f"{d}/match_shape.json"))
        lp, t, cap = int(s["LP"]), int(s["T"]), int(s["cap_T"])
    except (OSError, ValueError, KeyError, TypeError):
        return ""
    return "LP=%d T_r=%d cap_T=%d" % (lp, t, cap) if t > 0 and t != cap else ""


#: the query the row builds its manifest of record under: `query.required.QUERY_OF_RECORD`, spelt here because the row driver
#: imports no core module (tests/pipeline/test_row.py pins the two equal)
QUERY_OF_RECORD = "Q_word_v1{X=16,W=32,R=no-recompute}"


#: the taps a row runs by default, NORM_TAP=0 etc. opting out for A/B only: (CommitConfig field, manifest build / verify flag, the query
#: header key its policy writes, which is also its identities' family)
TAP_POLICIES = (("norm_tap", "--norm-scales", "norm_scales"), ("router_tap", "--router-softmax", "router_softmax"),
                ("vocab_tap", "--vocab-range", "vocab_range"))
#: the tap policies and the guarded max, which lengthens the attention stream's identities instead of adding a family
POLICIES = (*TAP_POLICIES, ("guarded_max_tap", "--guarded-max", "guarded_max"))


def tap_policy_flags(ck: Any) -> list[str]:
    """The manifest build / verify flags of the row's CommitConfig, every policy explicit (`--no-norm-scales` etc. under NORM_TAP=0)."""
    return [flag if getattr(ck, field, "0") == "1" else f"--no-{flag[2:]}" for field, flag, _key in POLICIES]


def policy_mismatch(q: Any, ck: Any) -> list[str]:
    """The policies a manifest's query header `q` disagrees with the row's CommitConfig on, either way (every policy the row runs when
    there is no header)."""
    return [key for field, _flag, key in POLICIES if (getattr(ck, field, "0") == "1") != (isinstance(q, dict) and isinstance(q.get(key), dict))]


def manifest_policy_mismatch(path: str, ck: Any) -> list[str]:
    """`policy_mismatch` of the manifest at `path` (every policy the row runs when it is unreadable)."""
    doc = load_json(path)
    return policy_mismatch(doc.get("query") if isinstance(doc, dict) else None, ck)


def manifest_not_of_record(doc: Any, program_digest: str, ck: Any) -> tuple[str, str]:
    """Why the manifest `doc` beside a Build is not the Commit's manifest of record, as (the slug its moved-aside file is named by, the
    reason); ("", "") when it is: built under the query of record with a partition, of the Build's Program digest (`program_digest`,
    the one the Commit is keyed by; unknown is not a match), under the row's tap policies, and recording the word check it passed (the
    Commit refuses one without: `commit.partition.word_check_refusal`)."""
    q = doc.get("query") if isinstance(doc, dict) and isinstance(doc.get("query"), dict) else {}
    qid = str(q.get("query_id") or "")
    if qid != QUERY_OF_RECORD:
        return "other-query", f"is not built under the query of record {QUERY_OF_RECORD} (query.query_id={qid or 'none'})"
    if not isinstance(q.get("partition"), dict):
        return "no-partition", "names no partition (query.partition)"
    if not program_digest or doc.get("program_digest") != program_digest:
        return "other-program", (f"is of Program {str(doc.get('program_digest'))[:16]}, not the Build's "
                                 f"{program_digest[:16] if program_digest else '(unknown: no build_summary.json)'}")
    mismatch = policy_mismatch(q, ck)
    if mismatch:
        return "policy-" + "-".join(mismatch).replace("_", "-"), f"disagrees with this Commit's taps on the {', '.join(mismatch)} policy"
    unchecked = word_check_refusal(q)
    if unchecked:
        return "unchecked", f"does not record a passing word check ({unchecked})"
    return "", ""


def applicable_taps(path: str, ck: Any) -> list[str]:
    """The taps the row runs that apply to it: on, and the manifest names at least one identity of the tap's family (a dense model has
    no router, a TP1 row no vocabulary range).  The row passes a tap's Commit source for these only."""
    try:
        fams = {r.get("family") for r in json.load(open(path)).get("identities") or []}
    except (OSError, ValueError, AttributeError):
        fams = set()
    return [field for field, _flag, key in TAP_POLICIES if getattr(ck, field, "0") == "1" and key in fams]


def tap_source_args(taps: list[str], env: Any) -> list[str]:
    """The Commit's tap source flags (`acquire.sources.taps.TapSources`) of the applicable taps."""
    args = {"norm_tap": ["--norm-tap-so", env.norm_tap_so], "router_tap": ["--router-tap-so", env.router_tap_so], "vocab_tap": ["--vocab-range"]}
    return [a for t in taps for a in args[t]]


def missing_tap_build(taps: list[str], env: Any) -> str:
    """Why the row cannot run its applicable taps: a tap build that is not on disk ("" when every one is)."""
    for t, so, script in (("norm_tap", env.norm_tap_so, "pod_norm_tap.sh"), ("router_tap", env.router_tap_so, "pod_router_tap.sh")):
        if t in taps and not os.path.isfile(so):
            return f"{t.upper()}=1 and the manifest names its family, but the tap .so is missing: {so} (ops/{script}, run by ops/pod_bootstrap.sh)"
    return ""


def program_digest_of_record(d: str, batch: int) -> str:
    """The Build's Program digest of record, read from ``build_summary.json``, never from the manifest: B=1 the declared
    (cap-shape) request Program's digest; B>=2 GP-01's workload digest, else the fold of the per-shape request digests."""
    b = json.load(open(f"{d}/build_summary.json")); wd = ""
    if batch <= 1:
        return (b.get("request") or {}).get("program_digest") or ""
    wp = f"{d}/build_workload/workload_program.json"
    if os.path.exists(wp):
        w = json.load(open(wp)); wd = w.get("workload_digest") or w.get("program_digest") or w.get("digest") or ""
    comps = sorted(v["program_digest"] for k, v in b.items() if k.startswith("build_request_LP") and isinstance(v, dict) and v.get("program_digest"))
    return wd or (hashlib.sha256(("workload-components/" + ",".join(comps)).encode()).hexdigest() if comps else "")


def snapshot_steps_of_record(d: str) -> str:
    """The Match record's snapshot steps (``match/run.json``), comma-joined; "" when unreadable."""
    try:
        s = json.load(open(f"{d}/match/run.json")).get("snapshot_steps") or []
    except (OSError, ValueError, AttributeError):
        return ""
    return ",".join(map(str, s)) if s else ""


def wor_component(d: str) -> str:
    """The Program dir the weights of record are derived from at B>1: the manifest's FIRST component (r0) ->
    ``build_request_LP<LP>_T<T>``, or ``build_request`` when its own result.json declares that shape.  The record is bound
    to the of-record set (workload digest + manifest component digests), so it must come from a component; the weights
    struct is model-level, so any component gives the same fields and root."""
    m = json.load(open(os.path.join(d, "manifest.json")))
    comps = m.get("components") or {}
    for rid in sorted(comps, key=lambda r: (len(r), r)):
        c = comps[rid] or {}
        lp, t = c.get("LP"), c.get("T")
        if lp is None or t is None:
            continue
        for cand in (os.path.join(d, "build_request_LP%d_T%d" % (int(lp), int(t))), os.path.join(d, "build_request")):
            rj = os.path.join(cand, "result.json")
            if os.path.exists(rj):
                try:
                    r = json.load(open(rj))
                except (OSError, ValueError):
                    continue
                if r.get("ok") and int(r.get("LP", -1)) == int(lp) and int(r.get("T", -1)) == int(t):
                    return cand
    return os.path.join(d, "build_request")


def wor_stale(record: str, result: str, schema: str) -> str:
    """Why an existing weights-of-record file beside a component may not be reused ("" = reuse it): built for another
    Program, predating the engine-assumptions stamp, of another schema than this tree derives, or incomplete."""
    try:
        rec = json.load(open(record))
    except Exception as e:  # noqa: BLE001
        return "unreadable record (%s)" % str(e)[:80]
    res = json.load(open(result)) if os.path.exists(result) else {}
    # the component Program digest as a real result.json carries it (artifact.program_digest / program.digest; a top-level key on the
    # synthetic ones), else the instances.json.gz digest the derivation itself falls back to -- an empty value here would skip the check
    rd = str(res.get("program_digest") or (res.get("artifact") or {}).get("program_digest") or (res.get("program") or {}).get("digest") or "")
    if not rd:
        ig = os.path.join(os.path.dirname(result), "instances.json.gz")
        if os.path.exists(ig):
            try:
                rd = str(json.load(gzip.open(ig, "rt")).get("digest") or "")
            except (OSError, EOFError, ValueError, AttributeError):
                rd = ""
    pd = str(rec.get("program_digest") or "")
    if rd and pd != rd:
        return "built for Program %s, this component is %s (stale)" % (pd[:16] or "?", rd[:16])
    if not isinstance(rec.get("engine_assumptions"), dict):
        return "predates MAN-09c (no engine_assumptions; %d engine constant(s) unpinned by name)" % len(rec.get("unpinned") or [])
    if schema and str(rec.get("schema") or "") != schema:
        return "schema %s, this tree derives %s (root formula differs: the analytic entry is keyed by its inputs, not the host)" % (rec.get("schema") or "?", schema)
    if rec.get("complete") is not True:
        return "incomplete (%d unresolved field(s))" % len(rec.get("unresolved") or [])
    return ""


def commit_line(d: str) -> str:
    """``runs .. failed .. checks name=outcome ...`` from ``verdict.json`` for the Commit stage's stages.txt line."""
    try:
        v = json.load(open(f"{d}/verdict.json"))
    except (OSError, ValueError):
        return ""
    r = v.get("runs") or {}
    return "runs %s failed %s checks %s" % (r.get("instrumented"), r.get("failed"),
                                            " ".join("%s=%s" % (c["name"], c["outcome"]) for c in v.get("checks") or [] if c.get("rule") is None))


def manifest_verify_ok(d: str) -> str:
    """``commit/manifest_verify.json`` ok as text, ``not-run`` when absent."""
    try:
        return str(json.load(open(f"{d}/commit/manifest_verify.json")).get("ok"))
    except (OSError, ValueError):
        return "not-run"


# ---------------------------------------------------------------------------------------------------------------- tensor parallel

_READ = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError)


def _field(path: str, fn: Any) -> str:
    """``fn(document)`` as text, "" when the document or a key it reads is absent."""
    try:
        return str(fn(json.load(open(path))))
    except _READ:
        return ""


def tp_shape(wl: str, checkpoints: str, hf_home: str) -> dict[str, Any]:
    """A tensor-parallel row's batch, envelope, declared request shapes (LP, cap - 1), engine lengths and checkpoint dir.

    ``MNBT`` is at most ``B * MML``: no step can schedule more tokens than ``max_num_seqs`` requests of ``max_model_len``, so the cap
    changes no schedule, but vLLM's warm-up forward (`kernel_warmup.flashinfer_autotune`'s dummy run) packs the full budget into
    ``max_num_seqs`` requests, longer than ``max_model_len``, and FlashAttention's varlen kernel then faults (an illegal memory access
    on sm_120 at TP2 for head_dim > 64: 2048 tokens in one request of max_model_len 288 for Qwen3-4B, Phi-3-mini, Mistral-7B; plain
    vLLM does the same, and passes at 288)."""
    w = json.load(open(wl)); s = w["sweep"]
    req = w.get("engine_args_required") or {}
    mml = req.get("max_model_len") or int(s["max_input"]) + int(s["max_output"])
    batch = int(s["batch"])
    return {"B": batch, "I": int(s["max_input"]), "O": int(s["max_output"]),
            "shapes": sorted({(int(a), int(b) - 1) for a, b in zip(s["input_lengths"], s["output_caps"])}),
            "MML": mml, "MNBT": min(int(req.get("max_num_batched_tokens") or mml), batch * int(mml)),
            "ck": checkpoint_dir(w["model"], w["tokenizer_revision"], hf_home, checkpoints) or ""}


def tp_world(wl: str) -> int:
    """The manifest's ``sweep.tp`` (2 when it names none or does not read)."""
    try:
        return int((json.load(open(wl)).get("sweep") or {}).get("tp") or 2)
    except _READ:
        return 2


def tp_build_summary(b: str, world: int) -> str:
    """PASS | FAIL; writes ``build_summary.json`` beside ``b`` (per rank: step + request Programs + GP-01)."""
    out: dict[str, Any] = {"tp": {"world": world}, "ranks": {}}; ok = True
    for r in range(world):
        rd = f"{b}/rank{r}"; rk: dict[str, Any] = {}
        for p in [f"{rd}/build_step"] + sorted(p for p in glob.glob(f"{rd}/build_request_LP*_T*") if os.path.isdir(p)):   # not the .log/.rc beside each dir
            res = json.load(open(f"{p}/result.json")) if os.path.exists(f"{p}/result.json") else {"ok": False, "message": "no result.json"}
            rk[os.path.basename(p)] = {"ok": bool(res.get("ok")), "unsupported": res.get("unsupported"), "message": (res.get("message") or "")[:300],
                                       "program_digest": (res.get("program") or {}).get("digest")}
            ok &= bool(res.get("ok"))
        wp = f"{rd}/build_workload/workload_program.json"; W = json.load(open(wp)) if os.path.exists(wp) else {}
        rk["workload"] = {"ok": bool(W.get("workload_digest")), "workload_digest": W.get("workload_digest"), "reason": W.get("reason"), "message": W.get("message"), "mechanism": "GP-01 cb_a.global_program"}
        ok &= bool(W.get("workload_digest"))
        out["ranks"][f"rank{r}"] = rk
    out["step"] = {"program_digest": (out["ranks"]["rank0"].get("build_step") or {}).get("program_digest")}    # research_result reads step.program_digest
    json.dump(out, open(f"{os.path.dirname(b)}/build_summary.json", "w"), indent=1, default=str)
    return "PASS" if ok else "FAIL"


def tp_rank_workload_digest(b: str, r: int) -> str:
    return _field(f"{b}/rank{r}/build_workload/workload_program.json", lambda d: d.get("workload_digest") or "")


def tp_build_digests(d: str) -> tuple[str, str]:
    """(rank 0 step digest[:16], ranks 1.. step digests[:16] joined by /) from ``build_summary.json``."""
    p = f"{d}/build_summary.json"
    return (_field(p, lambda s: (s["step"]["program_digest"] or "")[:16]),
            _field(p, lambda s: "/".join(((s["ranks"][f"rank{r}"].get("build_step") or {}).get("program_digest") or "")[:16] for r in range(1, s["tp"]["world"]))))


def tp_manifest_line(d: str) -> str:
    return _field(f"{d}/manifest.json", lambda m: " ".join(str(x) for x in [
        "complete", m.get("complete"), "identities", (m.get("populations") or {}).get("identities"), "tp_peer_binding_n_unbound",
        (m.get("tp_peer_binding") or {}).get("n_unbound"), "unmodelled", json.dumps(m.get("unmodelled"))[:300], "digest", (m.get("manifest_digest") or "")[:16]]))


def tp_manifest_complete(d: str) -> bool:
    return _field(f"{d}/manifest.json", lambda m: m.get("complete")) == "True"


def tp_manifest_not_of_record(d: str, ck: Any) -> str:
    """Why a TP row's `<d>/manifest.json` is not the Commit's manifest of record ("" when it is): complete, built under the query of
    record with a partition, under the row's tap policies, recording the strict word check it passed.  The per-rank Program digests are
    tp-commit's own check (each rank's rows against its Build), so a manifest the Build or a CPU host wrote is reused, not rebuilt on the
    GPU host ([GLM-5.3] the rebuild is hours of CPU and ~470 GB beside the engine)."""
    doc = load_json(f"{d}/manifest.json")
    if not isinstance(doc, dict) or doc.get("complete") is not True:
        return "absent or not complete"
    q = doc.get("query") if isinstance(doc.get("query"), dict) else {}
    if q.get("query_id") != QUERY_OF_RECORD or not isinstance(q.get("partition"), dict):
        return f"not built under the query of record {QUERY_OF_RECORD} with a partition"
    mismatch = policy_mismatch(q, ck)
    if mismatch:
        return f"disagrees with this Commit's taps on the {', '.join(mismatch)} policy"
    return word_check_refusal(q) or ""


def tp_check_line(m: str) -> str:
    return _field(f"{m}/tp2_match.json", lambda d: " ".join(str(x) for x in [
        "pass", d["pass"], "structure", d["structure"]["pass"], "n_coll_r0", d["structure"]["per_rank"]["0"]["n_collectives"], "sig_len",
        d["structure"]["per_rank"]["0"]["build_signature_len"], "steps", d["structure"]["per_rank"]["0"]["n_steps"], "provenance",
        d["provenance"]["pass"], d["provenance"]["collectives_checked"], "mismatches", d["provenance"]["n_mismatches"], "tokens", d["tokens"].get("pass")]))


def tp_passed(path: str) -> bool:
    return _field(path, lambda d: d["pass"]) == "True"


def tp_served_shapes(m: str) -> list[str]:
    """``LP,T`` of every captured request (prompt length, generated - 1)."""
    s = _field(f"{m}/capture/tokens.json", lambda d: " ".join(f"{len(q['prompt_token_ids'])},{len(q['generated_token_ids']) - 1}" for q in d["requests"]))
    return s.split()


def tp_fold_line(m: str) -> str:
    return _field(f"{m}/fold_match/tp2_fold_match.json",
                  lambda d: " ".join(["pass", str(d["pass"]), "legs", json.dumps(d["legs"]), "problems", json.dumps(d["problems"][:3])[:400]]))


def tp_match_summary(m: str, rc_cap: int, rc_ctl: int, rc_chk: int, rc_fold: int, ok: str, prof: str) -> None:
    """``match_summary.json`` beside ``m``: the collective-level check and the fold Match's legs."""
    tm = json.load(open(f"{m}/tp2_match.json")) if os.path.exists(f"{m}/tp2_match.json") else {}
    fm = json.load(open(f"{m}/fold_match/tp2_fold_match.json")) if os.path.exists(f"{m}/fold_match/tp2_fold_match.json") else None
    json.dump({"pass": ok == "PASS", "level": "fold (D90 per-rank fold Match + collective-level)" if prof else "collective-level (tp.match check; no TP rank profile)",
               "capture_rc": rc_cap, "control_rc": rc_ctl, "check_rc": rc_chk, "fold_rc": None if rc_fold < 0 else rc_fold, "fold_profile": prof or None,
               "tp2_match": {k: tm.get(k) for k in ("pass", "structure", "provenance", "tokens") if k in tm}, "tp2_fold_match": None if fm is None else {k: fm.get(k) for k in ("pass", "legs", "problems", "seconds")}},
              open(f"{os.path.dirname(m)}/match_summary.json", "w"), indent=1, default=str)


def tp_commit_line(d: str) -> str:
    return _field(f"{d}/commit/summary.json", lambda s: " ".join(str(x) for x in [
        "pass", s.get("pass"), "pairs", s.get("pairs"), "delta", s.get("delta_wall_mean"), "roots", json.dumps(s.get("tp_run_roots"))[:80], "tokens",
        s.get("tokens_equal"), "replay", (s.get("sampled_replay") or {}).get("result"), "linkage", (s.get("boundary_linkage") or {}).get("result"), "xrank",
        (s.get("cross_rank_collectives") or {}).get("result"), "fold_binding", (s.get("fold_match_binding") or {}).get("result"), "weights_pin",
        (s.get("weights_pin") or {}).get("result")]))


def tp_commit_passed(d: str) -> bool:
    return _field(f"{d}/commit/summary.json", lambda s: s.get("pass")) == "True"
