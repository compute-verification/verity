"""Empirical admission planner (increment C of the runner-telemetry brief).

The reservation model for one row stage is

    reserved(stage) = base_serving + sum(simultaneously-live Verity allocation objects at the stage's peak point) + bounded_margin

where every term is a NAMED allocation object of the row procedure (see `INVENTORY` below and
`out/gen/sweep/evidence/admit/INVENTORY.md`) with a lifetime (first/last substage where it is resident) and a size
law in plan-time scale variables (tokens, requests, layers, instances, workers, ...).  Coefficients are measured
constants (each carries the attempt dir + numbers that pin it, in `CALIBRATION`); there is no global multiplier.
Simultaneity is computed by a sweep over the stage's substage timeline (`peak_overlap`): sequential reuse is not
simultaneous residency (the FP8 3-pair result), and a term that outlives its substage (the pymalloc heap residual,
the pinned pool under `--retain host`) is modelled as a term with a longer lifetime, not as slack.

API: `row_config_from_workload(workload) -> RowScale`, the plan-time inputs of a row;
`predict(row_config) -> {stage: {"reserved_bytes", "terms": {...}, "margin", "peak_point", "admit"}}`, whose commit stage the
Commit's refusal reads (`pipeline.admission_bound.committer_resident`); `peak_overlap(intervals)`.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import row_tokens, workload_file
from verity_vllm.engine.profiles.admission_calibration import (
    CALIBRATION,
    DENSE_DEFAULT,
    MODELS,
    uncalibrated_model_note,
)

MiB = 1 << 20
SCHEMA = "verity/admission-plan/v1"


#: the value families `--retain-exclude hidden_m1` names (native_collect.retain_exclude_families): the attention launch's
#: M1 export stream, FA2 (Ada/Ampere) or FA3 (Hopper) -- the families no C2 consumer reads back from host.
HIDDEN_M1_FAMILIES = ("fa2_hidden_m1", "fa3_hidden_m1")


# Named allocation objects (the inventory), with lifetime as (first substage, last substage) on the stage timeline.
INVENTORY: dict[str, dict[str, str]] = {
    "serving.engine_host": {"stage": "commit", "where": "commit_delta committer process at engine construction", "freed": "process exit", "scales": "constant per engine"},
    "serving.weights_mmap": {"stage": "commit", "where": "vLLM model load (safetensors mmap)", "freed": "engine teardown", "scales": "model weights bytes"},
    "build.program_graph": {"stage": "build", "where": "derive_request (Program object graph of one request)", "freed": "request derivation end (heap residual across requests)", "scales": "decode rows x context x layers"},
    "match.capture.instance_records": {"stage": "match", "where": "m1_capture observer, per matched instance", "freed": "capture end (written to match/)", "scales": "instances"},
    "match.capture.token_values": {"stage": "match", "where": "m1_capture observer value snapshots", "freed": "capture end", "scales": "tokens x declaration widths"},
    "match.capture.hidden_values": {"stage": "match", "where": "m1_capture observer M1 hidden-stream snapshot", "freed": "capture end", "scales": "prompt tokens x K x heads"},
    "match.fold.instance_state": {"stage": "match", "where": "fold (verity_vllm.observe.fold.fold) instance dicts", "freed": "fold end", "scales": "instances"},
    "match.fold.attention_operands": {"stage": "match", "where": "fold per-step attention operand rows", "freed": "fold end", "scales": "steps x context x layers"},
    "commit.pinned_pool": {"stage": "commit", "where": "bounded warm-up / first instrumented run (torch pinned host allocator)", "freed": "commit_delta exit (--retain host); per-pair release only under --retain none", "scales": "raw committed bytes x 1.25"},
    "commit.run_record.leaf_digests": {"stage": "commit", "where": "instrumented run per-step leaf table", "freed": "commit_delta exit", "scales": "raw bytes / 268 x 32 B"},
    "commit.binding.population_index": {"stage": "commit", "where": "binding_coverage (binding map + population)", "freed": "pair end", "scales": "instances"},
    "commit.c2.evaluator_base": {"stage": "commit", "where": "C2 evaluator construction", "freed": "commit_delta exit", "scales": "constant"},
    "commit.c2.match_oracle": {"stage": "commit", "where": "oracle_compare: match/instances parsed", "freed": "pair end (heap residual retained)", "scales": "instances x per-model bytes"},
    "commit.c2.programs": {"stage": "commit", "where": "sampled_replay pair 0: Program graph load", "freed": "pair-0 replay end (heap residual retained; replay cache reuses the record)", "scales": "instances x per-model bytes"},
    "commit.c2.heap_residual": {"stage": "commit", "where": "pymalloc/glibc arenas after pair-0 C2 frees", "freed": "process exit", "scales": "0.91 x pair-0 C2 heap"},
    "commit.c2.replay_workers": {"stage": "commit", "where": "forked replay pool children (sampled_replay._parallel_replay)", "freed": "pool close", "scales": "workers x per-worker COW private pages"},
    "fa2.tap_window": {"stage": "commit", "where": "device: FA2 hidden-M1 window ring", "freed": "commit_delta exit", "scales": "heads x prompt x NB (largest step)"},
}

# Stage timelines: ordered substage points.  Commit repeats the pair block `pairs` times.
BUILD_POINTS = ["derive"]
MATCH_POINTS = ["capture", "control", "resolve", "fold", "compare", "global_match"]
COMMIT_PREFIX = ["setup", "warmup"]
COMMIT_PAIR_BLOCK = ["control", "instrumented", "binding", "oracle", "replay", "release"]


def commit_points(pairs: int) -> list[str]:
    pts = list(COMMIT_PREFIX)
    for p in range(max(1, pairs)):
        pts += [f"p{p}.{s}" for s in COMMIT_PAIR_BLOCK]
    pts.append("exit")
    return pts


# ---------------------------------------------------------------------------------------------------------------------
# Row scale (plan-time inputs).  Artifact-derived overrides win over the structural estimates.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class RowScale:
    model: str
    requests: int = 1
    prompt_tokens: int = 1024            # sum over requests
    decode_rows: int = 128               # sum over requests of decode steps (cap + lag)
    steps: int = 129
    max_context: int = 1152              # K of the longest reachable context
    pairs: int = 3
    replay_workers: int | None = None    # None -> sampled_replay default min(32, cpu_count//2)
    host_cpus: int = 128
    retain: str = "host"
    replay_cache: bool = True
    largest_vu_in_parent: bool = False   # the largest strata are evaluated serially in the parent, never in a forked worker
    # sampled_replay._parallel_replay gc.freeze()s the parent's heap before the fork (default-on in the code, env
    # VERITY_REPLAY_FORK_GC_FREEZE=0 opts out): a child's collections never traverse the inherited C2 heap, so a child that would copy
    # the parent's programs + oracle heap on write (not 2.9 x the largest tensor -- a VU reads its own rows only) is a 250 MiB worker.
    # False models a forked pool without gc.freeze.
    fork_gc_freeze: bool = True
    # the RETAIN-windowed pool: `--bounded-staging --retain host --retain-exclude hidden_m1` keeps the
    # C2 byte source (module outputs) and drops the M1 export stream's chunk-aligned interior (no C2 consumer reads it; opens by replay)
    bounded_staging: bool = False
    retain_exclude: list[str] = field(default_factory=list)
    largest_request_rows: int | None = None   # (cap + lag) of the request with the largest rows x context product
    largest_request_ctx: int | None = None    # its prompt_len + cap + lag
    largest_tensor_bytes: int | None = None   # 'largest tensor bound' ledger line when the row has run before
    host_limit_bytes: int | None = None
    device_limit_bytes: int | None = None
    # measured-at-plan-time overrides (admission JSON, fold_summary.json, commit.log)
    scheduled_raw_bytes: int | None = None
    token_bytes: int | None = None
    hidden_bytes: int | None = None
    n_instances: int | None = None
    leaves: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def spec(self) -> dict[str, Any]:
        return MODELS.get(self.model, DENSE_DEFAULT)

    @property
    def tokens(self) -> int:
        return int(self.prompt_tokens) + int(self.decode_rows)

    def instances(self) -> int:
        if self.n_instances:
            return int(self.n_instances)
        return round(self.spec["instances_per_layer_token"] * self.spec["layers"] * self.tokens)

    def token_bytes_est(self) -> int:
        if self.token_bytes:
            return int(self.token_bytes)
        return int(self.tokens * self.spec["per_token_bytes"])

    def hidden_bytes_est(self) -> int:
        if self.hidden_bytes:
            return int(self.hidden_bytes)
        return int(self.token_bytes_est() * self.max_context * self.spec["hidden_per_token_K"])

    def raw_bytes(self) -> int:
        if self.scheduled_raw_bytes:
            return int(self.scheduled_raw_bytes)
        return self.token_bytes_est() + self.hidden_bytes_est()

    def workers(self) -> int:
        if self.replay_workers is not None:
            return max(1, int(self.replay_workers))
        return max(1, min(32, int(self.host_cpus) // 2))

    def largest_request(self) -> tuple[int, int]:
        """(decode rows, reachable context) of the request that dominates derive_request / the M1 stream."""
        if self.largest_request_rows and self.largest_request_ctx:
            return int(self.largest_request_rows), int(self.largest_request_ctx)
        if self.requests <= 1:
            return int(self.decode_rows), int(self.max_context)
        # a mixed row: the longest prompt with the longest cap is the row's (I, O) pair
        rows = int(self.max_context) - int(self.prompt_tokens // max(1, self.requests)) if self.max_context else self.decode_rows // self.requests
        return max(1, min(int(self.max_context), max(rows, self.decode_rows // self.requests))), int(self.max_context)

    def largest_tensor(self) -> int:
        if self.largest_tensor_bytes:
            return int(self.largest_tensor_bytes)
        per_prompt = self.spec.get("largest_tensor_per_prompt", 554_000_000)
        rows, ctx = self.largest_request()
        lp = max(1, ctx - rows)
        return int(per_prompt * self.requests * (lp * ctx) / (1024 * 1152))


# ---------------------------------------------------------------------------------------------------------------------
# Terms and the sweep.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Term:
    name: str
    bytes: int
    live_from: str
    live_until: str
    basis: str
    kind: str = "host"     # host | device | margin | base


def peak_overlap(intervals: Iterable[tuple[str, float, float, int]]) -> dict[str, Any]:
    """Sweep-line maximum simultaneous overlap of (name, t0, t1, bytes) intervals (half-open [t0, t1)).

    Returns {"peak_bytes", "t", "live": {name: bytes}} at the first time the maximum is reached.  Sizes are summed,
    so two intervals of the same name are two allocations.
    """
    ev: list[tuple[float, int, str, int]] = []
    for name, t0, t1, b in intervals:
        if t1 < t0:
            t0, t1 = t1, t0
        ev.append((float(t0), 1, name, int(b)))
        ev.append((float(t1), 0, name, int(b)))
    ev.sort(key=lambda e: (e[0], e[1]))   # frees (0) before allocs (1) at the same instant: touching intervals do not overlap
    cur = 0
    live: dict[str, int] = {}
    best: dict[str, Any] = {"peak_bytes": 0, "t": None, "live": {}}
    for t, is_alloc, name, b in ev:
        if is_alloc:
            cur += b
            live[name] = live.get(name, 0) + b
            if cur > best["peak_bytes"]:
                best = {"peak_bytes": cur, "t": t, "live": dict(live)}
        else:
            cur -= b
            live[name] = live.get(name, 0) - b
            if live[name] <= 0:
                live.pop(name, None)
    return best


def _terms_to_intervals(terms: list[Term], points: list[str]) -> list[tuple[str, float, float, int]]:
    idx = {p: i for i, p in enumerate(points)}
    out: list[tuple[str, float, float, int]] = []
    for t in terms:
        if t.kind == "device":
            continue
        a = idx[t.live_from]
        b = idx[t.live_until] + 1   # inclusive last point
        out.append((t.name, a, b, t.bytes))
    return out


def _lower_bound_note(basis: str) -> str:
    m = re.search(r"\[lower_bound: (.*?)\]", basis or "")
    return m.group(1) if m else "under-fitted term"


def _stage_plan(terms: list[Term], points: list[str], limit: int | None) -> dict[str, Any]:
    host = [t for t in terms if t.kind in ("host", "base", "row")]
    margin = [t for t in terms if t.kind == "margin"]
    device = [t for t in terms if t.kind == "device"]
    pk = peak_overlap(_terms_to_intervals(host, points))
    margin_bytes = sum(t.bytes for t in margin)
    reserved = int(pk["peak_bytes"] + margin_bytes)
    peak_point = points[int(pk["t"])] if pk["t"] is not None else None
    lower_bound = {t.name: _lower_bound_note(t.basis) for t in host if "lower_bound" in (t.basis or "")}
    lb_terms = sorted(lower_bound)
    decision = _decision(reserved, limit, pk["live"], margin_bytes, lower_bound)
    return {
        "reserved_bytes": reserved,
        "reserved_mib": reserved // MiB,
        "peak_point": peak_point,
        "terms": [{"name": t.name, "bytes": t.bytes, "mib": t.bytes // MiB, "live": [t.live_from, t.live_until], "basis": t.basis, "kind": t.kind} for t in terms],
        "live_at_peak": {k: v // MiB for k, v in sorted(pk["live"].items(), key=lambda kv: -kv[1])},
        "margin": {t.name: t.bytes // MiB for t in margin},
        "device_bytes": sum(t.bytes for t in device),
        "host_limit_bytes": limit,
        "lower_bound_terms": lb_terms,
        "decision": decision,
        "admit": decision["admit"],
        "verdict": decision["verdict"],
        "spare_mib": (None if limit is None else (limit - reserved) // MiB),
    }


def _decision(reserved: int, limit: int | None, live: dict[str, int], margin_bytes: int, lower_bound: dict[str, str]) -> dict[str, Any]:
    """The refusal names what does not fit: the live terms at the peak, largest first, and the shortfall in bytes.

    A stage whose reservation carries a lower-bound term never reads as a plain admit -- the model knows it
    under-fits that object (rows #57 and #60 were killed at 1.4-1.6x the prediction); admit is None with verdict LOWER_BOUND naming the
    terms and the observation, so the runner decides by name.  Over the limit it is still REFUSE (a lower bound above the limit is a refusal).
    """
    if limit is None:
        return {"admit": None, "verdict": "NO_LIMIT", "reason": "no host limit given"}
    top = sorted(live.items(), key=lambda kv: -kv[1])
    if reserved > limit:
        return {"admit": False, "verdict": "REFUSE", "short_by_bytes": reserved - limit, "short_by_mib": (reserved - limit) // MiB,
                "live_at_peak_mib": {k: v // MiB for k, v in top}, "margin_mib": margin_bytes // MiB, "lower_bound_terms": sorted(lower_bound),
                "reason": f"reserved {reserved // MiB} MiB > limit {limit // MiB} MiB; largest live terms: " + ", ".join(f"{k} {v // MiB}" for k, v in top[:4])
                + ("; and the reservation is a LOWER BOUND (" + ", ".join(sorted(lower_bound)) + ")" if lower_bound else "")}
    if lower_bound:
        why = "; ".join(sorted({v for v in lower_bound.values()}))
        return {"admit": None, "verdict": "LOWER_BOUND", "spare_bytes_if_exact": limit - reserved, "lower_bound_terms": sorted(lower_bound),
                "live_at_peak_mib": {k: v // MiB for k, v in top}, "margin_mib": margin_bytes // MiB,
                "reason": f"reserved {reserved // MiB} MiB <= limit {limit // MiB} MiB, but the reservation is a LOWER BOUND: the model under-fits "
                          + ", ".join(sorted(lower_bound)) + f" ({why}); not an admit -- decide by name"}
    return {"admit": True, "verdict": "ADMIT", "spare_bytes": limit - reserved, "live_at_peak_mib": {k: v // MiB for k, v in top}, "margin_mib": margin_bytes // MiB}


def predict(cfg: RowScale) -> dict[str, Any]:
    """Reservation per stage: base serving + simultaneously-live named terms + bounded (named) margin."""
    sp = cfg.spec
    C = CALIBRATION
    inst = cfg.instances()
    raw = cfg.raw_bytes()
    tok_b = cfg.token_bytes_est()
    hid_b = cfg.hidden_bytes_est()
    layers = sp["layers"]
    # models whose capture observer the recorded data under-fits carry the flag on the term itself (never silently admitted)
    lb = f" [lower_bound: {sp['capture_lower_bound']}]" if sp.get("capture_lower_bound") else ""
    margin_terms = [
        Term("margin.pod_cotenants", C["margin.pod_cotenants"]["value"], "", "", C["margin.pod_cotenants"]["what"], "margin"),
        Term("margin.kernel_unreclaimable", C["margin.kernel_unreclaimable"]["value"], "", "", C["margin.kernel_unreclaimable"]["what"], "margin"),
    ]
    out: dict[str, Any] = {"schema": SCHEMA, "row": asdict(cfg), "derived": {"instances": inst, "raw_bytes": raw, "token_bytes": tok_b, "hidden_bytes": hid_b,
                                                                          "workers": cfg.workers(), "model_spec": sp}}

    # ---- build: derive the largest request (rows x reachable context x layers) -------------------------------------
    # The B requests are derived sequentially; the largest (prompt LP_max, T_max) request dominates; use the row's
    # longest prompt (max_context - decode) x the longest decode (decode_rows / requests at least) as the plan-time proxy.
    t_max, ctx_max = cfg.largest_request()
    ctx = cfg.max_context
    graph = int(sp["build_graph_per_row_ctx_layer"] * t_max * ctx_max * layers)
    b_terms = [
        Term("process.python_base", C["process.python_base"]["value"], "derive", "derive", C["process.python_base"]["what"], "base"),
        Term("build.program_graph", graph, "derive", "derive", f"{sp['build_graph_per_row_ctx_layer']} B x rows {t_max} x ctx {ctx_max} x layers {layers} (largest request)"),
    ] + margin_terms
    out["build"] = _stage_plan(b_terms, BUILD_POINTS, cfg.host_limit_bytes)

    # ---- match ----------------------------------------------------------------------------------------------------
    fold_inst_coef = 26_000 if sp["family"] == "dense" else 12_000
    m_terms = [
        Term("process.python_base", C["process.python_base"]["value"], "capture", "global_match", C["process.python_base"]["what"], "base"),
        Term("match.capture.instance_records", int(C["match.capture.instance_records"]["value"] * inst), "capture", "capture", f"12.3 KB x {inst} instances{lb}"),
        Term("match.capture.token_values", int(C["match.capture.token_values"]["value"] * tok_b), "capture", "capture", f"1.03 x token_bytes {tok_b // MiB} MiB{lb}"),
        Term("match.capture.hidden_values", int(C["match.capture.hidden_values"]["value"] * hid_b), "capture", "capture", f"0.67 x hidden_bytes {hid_b // MiB} MiB{lb}"),
        Term("match.fold.instance_state", int(fold_inst_coef * inst), "fold", "fold", f"{fold_inst_coef} B x {inst} instances"),
        Term("match.fold.attention_operands", int(C["match.fold.attention_operands"]["value"] * cfg.steps * ctx * layers), "fold", "fold", f"3.4 KB x steps {cfg.steps} x ctx {ctx} x layers {layers}"),
    ] + margin_terms
    out["match"] = _stage_plan(m_terms, MATCH_POINTS, cfg.host_limit_bytes)

    # ---- commit ---------------------------------------------------------------------------------------------------
    pts = commit_points(cfg.pairs)
    last = pts[-1]
    lf = float(C["commit.pinned_pool.levels_factor"]["value"])
    excl_hidden = bool(set(cfg.retain_exclude) & set(HIDDEN_M1_FAMILIES))
    pool_kept = raw
    if cfg.bounded_staging and cfg.retain == "host" and excl_hidden:
        # the M1 export stream's interior is not copied off the staging window; the pool holds the C2-read families + levels
        pool_kept = max(0, raw - hid_b)
        pool = pool_kept + int(raw * (lf - 1.0))
        pool_basis = f"bounded staging, --retain host --retain-exclude {','.join(sorted(cfg.retain_exclude))}: (raw {raw // MiB} - hidden M1 {hid_b // MiB}) MiB + levels over every leaf raw {raw // MiB} MiB x {lf - 1.0:.2f}"
    elif cfg.bounded_staging and cfg.retain != "host":
        pool = int(raw * (lf - 1.0))
        pool_basis = f"bounded staging, --retain {cfg.retain}: Merkle levels only, raw {raw // MiB} MiB x {lf - 1.0:.2f} (raw released per window; openings by replay)"
    else:
        # the unbounded windowed collector copies every window to the pinned pool as the hit-plan hedge / learning re-hash source
        # WHATEVER --retain says (`--retain none` still allocates the whole pool) -- the pool is a whole-run term under every retention on
        # that path, never 0 MiB by name
        pool = int(raw * lf)
        pool_basis = f"raw {raw // MiB} MiB x {lf} (--retain {cfg.retain}{'' if cfg.retain == 'host' else ': allocated regardless of retention on the unbounded windowed path'})"
    leaves = int(cfg.leaves or raw // C["commit.run_record.bytes_per_leaf"]["value"])
    leaf_digests = leaves * 32
    pool_until = last
    oracle_b = int(sp["c2_oracle_per_instance"] * inst)
    programs_b = int(sp["c2_programs_per_instance"] * inst)
    base_b = C["commit.c2.evaluator_base"]["value"]
    c2_heap_p0 = oracle_b + programs_b
    residual_b = int(C["commit.c2.heap_residual.retention"]["value"] * c2_heap_p0)
    workers = cfg.workers()
    small_w = int(C["commit.c2.replay_workers.small_vu"]["value"])
    big_w = int(C["commit.c2.replay_workers.largest_vu"]["value"] * cfg.largest_tensor())
    # with gc.freeze before the fork the children's private pages are scratch + refcount-dirtied pages of the rows they
    # touch: every worker is a 250 MiB worker; without it the "2.9 x largest tensor" child is the parent's C2 heap copied on write by the child's GC
    n_big = 0 if (cfg.largest_vu_in_parent or cfg.fork_gc_freeze) else min(workers, int(C["commit.c2.replay_workers.largest_vu_picks"]["value"]))
    if workers > 1:
        workers_b = n_big * big_w + (workers - n_big) * small_w
        w_basis = (f"{n_big} workers x 2.9 x largest tensor {cfg.largest_tensor() // MiB} MiB + {workers - n_big} workers x 250 MiB"
                   if n_big else f"{workers} workers x 250 MiB (" + ("gc.freeze before the fork: the children never copy the C2 heap"
                                                                      if cfg.fork_gc_freeze else "largest strata evaluated in the parent") + ")")
    else:
        workers_b, w_basis = 0, "serial replay"
    # the parent evaluating the largest VU serially holds the same working set itself; a
    # VU's working set is its own rows, so no separate parent term is charged when gc.freeze governs
    parent_big = big_w if (workers == 1 or cfg.largest_vu_in_parent) and not cfg.fork_gc_freeze else 0
    c_terms = [
        Term("serving.engine_host", C["serving.engine_host"]["value"], "setup", last, C["serving.engine_host"]["what"], "base"),
        Term("serving.weights_mmap", int(sp["weights_bytes"]), "setup", last, f"safetensors {sp['weights_bytes'] // MiB} MiB file-mapped", "base"),
        Term("commit.pinned_pool", pool, "warmup", pool_until, pool_basis),
        Term("commit.run_record.leaf_digests", leaf_digests, "warmup", last, f"{leaves} leaves x 32 B"),
        Term("commit.c2.evaluator_base", base_b, "p0.oracle", last, C["commit.c2.evaluator_base"]["what"]),
    ]
    for p in range(cfg.pairs):
        c_terms.append(Term("commit.binding.population_index", int(C["commit.binding.population_index"]["value"] * inst), f"p{p}.binding", f"p{p}.release", f"2 KB x {inst} instances"))
        c_terms.append(Term("commit.c2.match_oracle", oracle_b, f"p{p}.oracle", f"p{p}.replay", f"{sp['c2_oracle_per_instance']} B x {inst} instances ({cfg.model}); freed at the pair's release, its arenas become heap_residual"))
        if p == 0 or not cfg.replay_cache:
            c_terms.append(Term("commit.c2.programs", programs_b, f"p{p}.replay", f"p{p}.replay", f"{sp['c2_programs_per_instance']} B x {inst} instances ({cfg.model})"))
            if workers_b:
                c_terms.append(Term("commit.c2.replay_workers", int(workers_b), f"p{p}.replay", f"p{p}.replay", w_basis))
            if parent_big:
                c_terms.append(Term("commit.c2.largest_vu_eval", int(parent_big), f"p{p}.replay", f"p{p}.replay", f"2.9 x largest tensor {cfg.largest_tensor() // MiB} MiB evaluated in the parent"))
    if cfg.pairs > 1:
        c_terms.append(Term("commit.c2.heap_residual", residual_b, "p0.release", last, f"0.91 x pair-0 C2 heap {c2_heap_p0 // MiB} MiB (retained arenas)"))
    c_terms.append(Term("fa2.tap_window", C["fa2.tap_window.device"]["value"], "warmup", last, C["fa2.tap_window.device"]["what"], "device"))
    c_terms += margin_terms
    plan = _stage_plan(c_terms, pts, cfg.host_limit_bytes)
    # the actionable knob for the replay-workers term: how many workers fit beside the parent at the replay point
    if cfg.host_limit_bytes:
        parent_at_replay = sum(t.bytes for t in c_terms if t.kind in ("host", "base") and t.name != "commit.c2.replay_workers"
                               and pts.index(t.live_from) <= pts.index("p0.replay") <= pts.index(t.live_until)) + sum(t.bytes for t in margin_terms)
        spare = cfg.host_limit_bytes - parent_at_replay
        # the two largest-stratum workers first, then 250 MiB workers
        adm = 0
        if spare > 0:
            if n_big and spare >= n_big * big_w:
                adm = n_big + int((spare - n_big * big_w) // small_w)
            elif not n_big:
                adm = int(spare // small_w)
            else:
                adm = 0 if spare < big_w else 1
        plan["replay_workers_admissible"] = min(adm, 32)
        plan["replay_workers_requested"] = workers
        plan["replay_workers_note"] = ("gc.freeze before the fork (default-on): every forked worker is a 250 MiB worker; the children never copy the C2 heap"
                                       if cfg.fork_gc_freeze else
                                       "largest strata in the parent: every forked worker is a 250 MiB worker" if cfg.largest_vu_in_parent else
                                       f"two forked workers may hold the largest VU ({big_w // MiB} MiB each); pass --replay-workers <= admissible or evaluate the largest strata in the parent")
    plan["parent_c2_heap_mib"] = (c2_heap_p0 + base_b) // MiB
    plan["largest_tensor_mib"] = cfg.largest_tensor() // MiB
    plan["pinned_pool"] = {"bytes": pool, "kept_raw_bytes": pool_kept, "basis": pool_basis, "bounded_staging": cfg.bounded_staging, "retain": cfg.retain,
                           "retain_exclude": sorted(cfg.retain_exclude), "hidden_bytes_est": hid_b}
    plan["fork_gc_freeze"] = cfg.fork_gc_freeze
    out["commit"] = plan
    return out


# ---------------------------------------------------------------------------------------------------------------------
# A row's plan-time config from its workload.
# ---------------------------------------------------------------------------------------------------------------------
def model_key_for_row(row_id: str) -> str:
    """`<model>__<dtype>__<device>__...` -> MODELS key (`-padded` variants share the base model's coefficients)."""
    key = row_tokens(row_id)[0]
    if key not in MODELS and key.endswith("-padded"):
        key = key[: -len("-padded")]
    return key


def request_prompt_len(request: dict[str, Any]) -> int:
    """A workload request's prompt length: its `prompt_len`, else how many `prompt_token_ids` it lists; 0 where it gives neither."""
    ids = request.get("prompt_token_ids")
    return int(request.get("prompt_len") or (len(ids) if ids else 0))


def row_config_from_workload(workload_path: str | dict[str, Any], row_id: str | None = None, pairs: int = 3, replay_workers: int | None = None,
                             host_limit_bytes: int | None = None) -> RowScale:
    """Plan-time config from the sweep workload (workloads/<row>.json, or its parsed dict): the same facts row_pod.sh reads for B/I/O/LP/T.
    A workload may leave out its `sweep`, its `requests` and its `engine` arguments: the row's shape is then its requests', else the
    sweep's, else one request of 1,024 input and 128 output tokens, and its longest context the longest input and output."""
    w = workload_path if isinstance(workload_path, dict) else json.loads(workload_file(workload_path).read_text())
    sweep, requests, engine = w.get("sweep"), w.get("requests"), w.get("engine")
    sw: dict[str, Any] = sweep if sweep else {}
    reqs: list[dict[str, Any]] = requests if requests else []
    row = row_id or sw.get("row_id") or ("" if isinstance(workload_path, dict) else Path(workload_path).stem)
    prompt = sum(request_prompt_len(r) for r in reqs) or int(sw.get("batch", 1)) * int(sw.get("max_input", 1024))
    decode = sum(int(r.get("max_tokens") or 0) for r in reqs) or int(sw.get("batch", 1)) * int(sw.get("max_output", 128))
    b = int(sw.get("batch") or len(reqs) or 1)
    i_max, o_max = int(sw.get("max_input") or 1024), int(sw.get("max_output") or 128)
    steps = int(w.get("expected_engine_steps") or o_max)
    lp_max = max((int(r.get("prompt_len") or 0) for r in reqs), default=i_max)
    t_max = max((int(r.get("max_tokens") or 0) for r in reqs), default=o_max)
    engine_args = engine.get("args") if engine else None
    key = model_key_for_row(row)
    notes = []
    if key not in MODELS:
        notes.append(uncalibrated_model_note(key))
    return RowScale(model=key, requests=b, prompt_tokens=prompt, decode_rows=decode, steps=steps, max_context=int((engine_args.get("max_model_len") if engine_args else None) or (i_max + o_max)),
                    pairs=pairs, replay_workers=replay_workers, host_limit_bytes=host_limit_bytes,
                    largest_request_rows=t_max, largest_request_ctx=lp_max + t_max, notes=notes)
