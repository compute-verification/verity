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

Two data paths feed the planner:

  (a) increment A telemetry (field names as zk/campaign/telemetry/{events,sample,cgroup,procs}.py writes them):
      `events*.jsonl` spans (`span_start`/`span_end`, `stage`) and
      named-allocation marks (`kind="mark", name="alloc"|"free", obj=<term>, bytes=<n>, alloc_id=<id>`), plus
      `resources*.jsonl` `kind="cgroup"` (memory_current_bytes, memory_stat.anon/shmem/file) and `kind="process"` rows
      (per-pid rss_bytes / rss_anon_bytes) — `observe_increment_a`.
  (b) the pre-telemetry fallback: `resources.jsonl` ({utc|t, max_python_rss_mib, cgroup_current_mib, cgroup_anon_shmem_mib,
      gpu_used_mib}) + `timeline.jsonl` (`ev` start/end per `stage`/`proc`) + `stages.txt`, the admission JSON
      under `commit/`, `match/fold_summary.json`, `commit.log` ledger lines and `cgroup_memory.events.txt` /
      `commit_oom_cgroup.txt` — `observe_r17` and `row_config_from_attempt`.

API: `predict(row_config) -> {stage: {"reserved_bytes", "terms": {...}, "margin", "peak_point", "admit"}}`,
`compare(attempt_dir) -> per-stage predicted vs observed peak + residual`, `peak_overlap(intervals)`.
CLI: `verity-vllm telemetry-admission compare <attempt_dir>... [--md]` / `predict --model ... --requests ...`.
"""
from __future__ import annotations

import datetime as _dt
import glob
import gzip
import json
import os
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.config import ROW_ID_PATTERN, row_tokens, workload_file
from verity_vllm.engine.profiles.admission_calibration import (CALIBRATION, DENSE_DEFAULT, DEFAULT_MODEL, MODELS, uncalibrated_model_note,
                                                               unknown_model_note)

MiB = 1 << 20
GiB = 1 << 30
SCHEMA = "verity/admission-plan/v1"


#: the value families `--retain-exclude hidden_m1` names (native_collect.retain_exclude_families): the attention launch's
#: M1 export stream, FA2 (Ada/Ampere) or FA3 (Hopper) -- the families no C2 consumer reads back from host.
HIDDEN_M1_FAMILIES = ("fa2_hidden_m1", "fa3_hidden_m1")
RETAIN_EXCLUDE_ALIASES = {"hidden_m1": HIDDEN_M1_FAMILIES, "hidden": HIDDEN_M1_FAMILIES}


def retain_exclude_families(spec: str | None) -> list[str]:
    """`--retain-exclude` / VERITY_RETAIN_EXCLUDE -> sorted family names (the `_stream` suffix tolerated, aliases expanded)."""
    out: set[str] = set()
    for f in (spec or "").split(","):
        f = f.strip().removesuffix("_stream")
        if f:
            out |= set(RETAIN_EXCLUDE_ALIASES.get(f, (f,)))
    return sorted(out)


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
# Row configuration (plan-time inputs).  Artifact-derived overrides win over the structural estimates.
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class RowConfig:
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
    # False models a forked pool without gc.freeze, for `compare` against records made without it.
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
    best = {"peak_bytes": 0, "t": None, "live": {}}
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
    out = []
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


def predict(cfg: RowConfig) -> dict[str, Any]:
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
# Data path (b): attempt directories.
# ---------------------------------------------------------------------------------------------------------------------
def _ts(s: str) -> float:
    s = s.replace("Z", "")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return _dt.datetime.strptime(s, fmt).replace(tzinfo=_dt.timezone.utc).timestamp()  # noqa: UP017, RUF100 (3.9 laptops read fixtures too)
        except ValueError:
            pass
    raise ValueError(s)


def _jsonl(path: str) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def r17_samples(attempt_dir: str) -> list[dict[str, Any]]:
    """A pre-telemetry attempt's `resources*.jsonl` -> [{t, rss, cgroup, anon_shmem, gpu, stage}] (MiB), time-sorted."""
    out = []
    for f in sorted(glob.glob(os.path.join(attempt_dir, "resources*.jsonl"))):
        for r in _jsonl(f):
            t = r.get("utc") or r.get("t")
            if not t:
                continue
            out.append({"t": _ts(t) if isinstance(t, str) else float(t), "rss": r.get("max_python_rss_mib"), "cgroup": r.get("cgroup_current_mib"),
                        "anon_shmem": r.get("cgroup_anon_shmem_mib"), "gpu": r.get("gpu_used_mib", r.get("gpu0_used_mib")), "stage": r.get("last_stage")})
    out.sort(key=lambda s: s["t"])
    return out


def r17_spans(attempt_dir: str) -> list[dict[str, Any]]:
    """A pre-telemetry attempt's `timeline.jsonl` -> [{stage, proc, t0, t1, closed}]; unclosed spans end at the last event (the kill)."""
    p = os.path.join(attempt_dir, "timeline.jsonl")
    if not os.path.exists(p):
        return []
    ev = sorted(_jsonl(p), key=lambda r: r.get("t", 0))
    open_: dict[tuple, list[float]] = {}
    sp = []
    for r in ev:
        k = (r.get("proc"), r.get("stage"))
        if r.get("ev") == "start":
            open_.setdefault(k, []).append(r["t"])
        elif r.get("ev") == "end" and open_.get(k):
            sp.append({"stage": k[1], "proc": k[0], "t0": open_[k].pop(), "t1": r["t"], "closed": True})
    last = ev[-1]["t"] if ev else 0
    for k, ts in open_.items():
        for t0 in ts:
            sp.append({"stage": k[1], "proc": k[0], "t0": t0, "t1": last, "closed": False})
    return sorted(sp, key=lambda s: s["t0"])


# substage -> plan point
_R17_POINT = {
    "build.derive_request": ("build", "derive"), "build.derive_step": ("build", "derive"),
    "work.match_capture": ("match", "capture"), "work.capture_run": ("match", "capture"), "work.match_control": ("match", "control"),
    "match.resolve": ("match", "resolve"), "match.fold": ("match", "fold"), "match.compare": ("match", "compare"), "match.global_match": ("match", "global_match"),
    "prep.committer_setup": ("commit", "setup"), "prep.register_weights": ("commit", "setup"), "prep.warmup_instrumented": ("commit", "warmup"),
    "validate.binding_coverage": ("commit", "binding"), "validate.oracle_compare": ("commit", "oracle"), "validate.sampled_replay": ("commit", "replay"),
    "validate.openings_after_release": ("commit", "release"),
}


def observe_r17(attempt_dir: str) -> dict[str, Any]:
    """Observed host peaks per stage / plan point from the pre-telemetry sampler + timeline, with the cgroup kill record if any."""
    S = r17_samples(attempt_dir)
    spans = r17_spans(attempt_dir)
    obs: dict[str, Any] = {"source": "r17", "attempt_dir": attempt_dir, "samples": len(S), "stages": {}, "points": {}, "kill": None, "notes": []}
    pair_ix: dict[str, int] = {}
    for s in spans:
        st = s["stage"]
        if st.startswith("stage."):
            name = st.split(".", 1)[1]
            win = [x for x in S if s["t0"] <= x["t"] <= s["t1"]]
            if win:
                obs["stages"][name] = {"rss_mib": max((x["rss"] or 0) for x in win), "cgroup_mib": max((x["cgroup"] or 0) for x in win),
                                       "anon_shmem_mib": max((x["anon_shmem"] or 0) for x in win), "gpu_mib": max((x["gpu"] or 0) for x in win),
                                       "n": len(win), "wall_s": int(s["t1"] - s["t0"]), "closed": s["closed"]}
            continue
        m = _R17_POINT.get(st)
        if not m:
            continue
        stage, point = m
        if stage == "commit" and point in COMMIT_PAIR_BLOCK:
            p = pair_ix.get(point, 0)
            pair_ix[point] = p + 1
            point = f"p{p}.{point}"
        win = [x for x in S if s["t0"] <= x["t"] <= s["t1"]]
        if not win:
            continue
        key = f"{stage}/{point}"
        rec = {"rss_mib": max((x["rss"] or 0) for x in win), "cgroup_mib": max((x["cgroup"] or 0) for x in win),
               "anon_shmem_mib": max((x["anon_shmem"] or 0) for x in win), "n": len(win), "wall_s": int(s["t1"] - s["t0"]), "closed": s["closed"], "proc": s["proc"]}
        prev = obs["points"].get(key)
        if prev is None or rec["rss_mib"] > prev["rss_mib"]:
            obs["points"][key] = rec
    # the committer's own ledger
    cl = os.path.join(attempt_dir, "commit.log")
    if os.path.exists(cl):
        txt = Path(cl).read_text(errors="replace")
        m = re.search(r"host RSS after run \(MB\) \| (\d+) \| (\d+)", txt)
        if m:
            obs["commit_ledger_host_rss_after_run_mb"] = [int(m.group(1)), int(m.group(2))]
        m = re.search(r"staging \(pinned, in use\) ([\d,]+) B", txt)
        if m:
            obs["commit_ledger_pinned_in_use_mib"] = int(m.group(1).replace(",", "")) // MiB
        m = re.search(r"predicted host (\d+) MiB vs limit (\d+) MiB: (ADMITTED|REFUSED)", txt)
        if m:
            obs["f_da_15"] = {"predicted_mib": int(m.group(1)), "limit_mib": int(m.group(2)), "verdict": m.group(3)}
    # kill record
    for name in ("commit_oom_cgroup.txt", "cgroup_memory.events.txt"):
        p = os.path.join(attempt_dir, name)
        if os.path.exists(p):
            txt = Path(p).read_text(errors="replace")
            km = re.search(r"oom_kill (\d+)", txt)
            mx = re.search(r"(?:^|\n)(?:max |# memory.max\n)(\d{9,})", txt)
            obs["kill"] = {"file": name, "oom_kill": int(km.group(1)) if km else None, "memory_max_bytes": int(mx.group(1)) if mx else None}
    st = os.path.join(attempt_dir, "stages.txt")
    if os.path.exists(st):
        for line in Path(st).read_text(errors="replace").splitlines():
            m = re.match(r"(build|match|commit) (PASS|FAIL) \S+ rc=([\d/]+)", line)
            if m:
                obs["stages"].setdefault(m.group(1), {})["verdict"] = m.group(2)
                obs["stages"][m.group(1)]["rc"] = m.group(3)
    return obs


def row_config_from_attempt(attempt_dir: str) -> RowConfig:
    """Plan-time inputs from an attempt directory's artifacts (preflight row, workload program, admission JSON, fold summary)."""
    notes: list[str] = []
    row = None
    pf = os.path.join(attempt_dir, "preflight.json")
    if os.path.exists(pf):
        row = json.loads(Path(pf).read_text()).get("row")
    if not row:
        for name in ("commit.log", "launcher.log", "row.log"):
            p = os.path.join(attempt_dir, name)
            if os.path.exists(p):
                m = re.search(rf"workloads/({ROW_ID_PATTERN})\.json", Path(p).read_text(errors="replace"))
                if m:
                    row = m.group(1)
                    break
    model, requests, K = DEFAULT_MODEL, 1, 1152
    if row:
        parts = row_tokens(row)
        model = parts[0]
        for p in parts:
            if re.fullmatch(r"b\d+", p):
                requests = int(p[1:])
        i_ = next((int(p[1:]) for p in parts if re.fullmatch(r"i\d+", p)), 1024)
        o_ = next((int(p[1:]) for p in parts if re.fullmatch(r"o\d+", p)), 128)
        K = i_ + o_
    cfg = RowConfig(model=model, requests=requests, max_context=K, notes=notes)
    # an attempt is modelled as it ran: a record whose commit.log lacks the frozen-fork line ran without gc.freeze (fork_gc_freeze False
    # -> the forked-children term); one that carries it is modelled with 250 MiB workers
    cl = os.path.join(attempt_dir, "commit.log")
    cfg.fork_gc_freeze = bool(os.path.exists(cl) and re.search(r"\[replay\] fork gc\.freeze=on", Path(cl).read_text(errors="replace")))
    if not cfg.fork_gc_freeze:
        notes.append("modelled without gc.freeze before the fork (no frozen-fork line in commit.log; --no-fork-gc-freeze equivalent)")
    if model not in MODELS:
        notes.append(unknown_model_note(model))
    # timeline row start carries B/LP/T
    tl = os.path.join(attempt_dir, "timeline.jsonl")
    if os.path.exists(tl):
        for r in _jsonl(tl):
            if r.get("stage") == "row" and r.get("ev") == "start":
                if r.get("B"):
                    cfg.requests = int(r["B"])
                if r.get("LP") and r.get("T"):
                    cfg.prompt_tokens, cfg.decode_rows = int(r["LP"]) * 1, int(r["T"]) + 1
                    cfg.max_context = int(r["LP"]) + int(r["T"]) + 1
                break
    # per-request shapes: build_request_LP<lp>_T<t> sub-dirs / build_summary keys (B > 1) — exact prompt/decode sums
    shapes: list[tuple[int, int]] = []
    bs = os.path.join(attempt_dir, "build_summary.json")
    keys = list(json.loads(Path(bs).read_text()).keys()) if os.path.exists(bs) else []
    keys += [os.path.basename(p) for p in glob.glob(os.path.join(attempt_dir, "build_request_LP*_T*")) if not p.endswith((".log", ".rc"))]
    for k in keys:
        m = re.fullmatch(r"build_request_LP(\d+)_T(\d+)", k)
        if m:
            shapes.append((int(m.group(1)), int(m.group(2))))
    shapes = sorted(set(shapes))
    if shapes:
        cfg.requests = len(shapes)
        cfg.prompt_tokens = sum(lp for lp, _ in shapes)
        cfg.decode_rows = sum(t + 1 for _, t in shapes)
        cfg.max_context = max(lp + t + 1 for lp, t in shapes)
        cfg.steps = max(t for _, t in shapes) + 2
        lp, t = max(shapes, key=lambda s: (s[1] + 1) * (s[0] + s[1] + 1))
        cfg.largest_request_rows, cfg.largest_request_ctx = t + 1, lp + t + 1
    # workload program (when kept): authoritative
    wp = glob.glob(os.path.join(attempt_dir, "build_workload", "workload_program.json*"))
    if wp:
        try:
            with (gzip.open(wp[0], "rt") if wp[0].endswith(".gz") else open(wp[0])) as fh:
                j = json.load(fh)
            reqs = j.get("requests") or []
            if reqs:
                lag = int(j.get("lag", 1) or 1)
                cfg.prompt_tokens = sum(int(r.get("prompt_len", 0)) for r in reqs)
                cfg.decode_rows = sum(int(r.get("cap", 0)) + lag for r in reqs)
                cfg.requests = len(reqs)
                cfg.max_context = max(int(r.get("prompt_len", 0)) + int(r.get("cap", 0)) + lag for r in reqs)
                cfg.steps = max(int(r.get("cap", 0)) for r in reqs) + 1 + lag
                big = max(reqs, key=lambda r: (int(r.get("cap", 0)) + lag) * (int(r.get("prompt_len", 0)) + int(r.get("cap", 0)) + lag))
                cfg.largest_request_rows, cfg.largest_request_ctx = int(big.get("cap", 0)) + lag, int(big.get("prompt_len", 0)) + int(big.get("cap", 0)) + lag
        except Exception as e:  # noqa: BLE001
            notes.append(f"workload_program unreadable: {e!r}")
    # the admission JSON (the runner's own schedule)
    adm = glob.glob(os.path.join(attempt_dir, "commit", "admission_*.json"))
    if adm:
        j = json.loads(Path(adm[0]).read_text())
        s = j.get("scheduled", {})
        cfg.scheduled_raw_bytes = int(j.get("scheduled_raw_bytes") or 0) or None
        cfg.host_limit_bytes = int(j.get("host_cap_bytes") or 0) or None
        cfg.retain = j.get("retain", cfg.retain)
        if s:
            cfg.prompt_tokens = int(s.get("prompt_tokens", cfg.prompt_tokens))
            cfg.decode_rows = int(s.get("decode_rows", cfg.decode_rows))
            cfg.steps = int(s.get("steps", cfg.steps))
            cfg.requests = int(s.get("requests", cfg.requests))
            comp = s.get("composition", {})
            cfg.token_bytes = int(comp.get("token_bytes") or 0) or None
            cfg.hidden_bytes = int(comp.get("hidden_bytes") or 0) or None
            ks = [int(p.get("K", 0)) for p in j.get("peak_steps", [])]
            if ks:
                cfg.max_context = max(ks)
    fs = os.path.join(attempt_dir, "match", "fold_summary.json")
    if os.path.exists(fs):
        j = json.loads(Path(fs).read_text())
        cfg.n_instances = int(j.get("n_instances") or 0) or None
        if j.get("n_steps"):
            cfg.steps = int(j["n_steps"]) + 1
    cl = os.path.join(attempt_dir, "commit.log")
    if os.path.exists(cl):
        txt = Path(cl).read_text(errors="replace")
        m = re.search(r"--replay-workers[ =](\d+)", txt)
        if m:
            cfg.replay_workers = int(m.group(1))
        m = re.search(r"--pairs[ =](\d+)", txt)
        if m:
            cfg.pairs = int(m.group(1))
        m = re.search(r"in ([\d,]+) leaves", txt)
        if m:
            cfg.leaves = int(m.group(1).replace(",", ""))
        m = re.search(r"--retain (\w+)", txt)
        if m:
            cfg.retain = m.group(1)
        m = re.search(r"largest tensor bound (\d+) MiB", txt)
        if m:
            cfg.largest_tensor_bytes = int(m.group(1)) * MiB
        if "--replay-cache" not in txt and "--no-replay-cache" in txt:
            cfg.replay_cache = False
    if cfg.host_limit_bytes is None:
        for name in ("commit_oom_cgroup.txt", "cgroup_memory.events.txt"):
            p = os.path.join(attempt_dir, name)
            if os.path.exists(p):
                mx = re.search(r"(\d{11,})", Path(p).read_text(errors="replace"))
                if mx:
                    cfg.host_limit_bytes = int(mx.group(1))
                    break
    if cfg.host_limit_bytes is None:
        # the sampler's cgroup peak is a lower bound on the limit; the pods measured were 175 GiB, 351 GiB and 125 GiB (L40S 2x)
        S = r17_samples(attempt_dir)
        pk = max((s["cgroup"] or 0) for s in S) if S else 0
        for lim in (119_209, 179_290, 359_535):
            if pk <= lim:
                cfg.host_limit_bytes = lim * MiB
                notes.append(f"host limit inferred from the sampler's cgroup peak {pk} MiB -> {lim} MiB class")
                break
    return cfg


# ---------------------------------------------------------------------------------------------------------------------
# Data path (a): increment A telemetry (events*.jsonl spans + alloc/free marks; resources*.jsonl cgroup/process rows).
# ---------------------------------------------------------------------------------------------------------------------


def observe_increment_a(attempt_dir: str) -> dict[str, Any]:
    """Named-allocation intervals from increment A marks + cgroup/process samples -> per-stage peak overlap.

    Marks: `{"kind":"mark","name":"alloc","obj":"commit.pinned_pool","bytes":N,"alloc_id":"...","stage":"..."}` and the
    matching `name:"free"`.  Unfreed allocations end at the last event of their stream.  Spans give stage windows.
    """
    recs: list[dict[str, Any]] = []
    for f in sorted(glob.glob(os.path.join(attempt_dir, "events*.jsonl"))):
        recs.extend(_jsonl(f))
    recs.sort(key=lambda r: (str(r.get("t_utc", "")), str(r.get("stream", "")), int(r.get("seq", 0) or 0)))
    obs: dict[str, Any] = {"source": "increment_a", "attempt_dir": attempt_dir, "events": len(recs), "stages": {}, "allocs": [], "notes": []}
    if not recs:
        obs["notes"].append("no events*.jsonl")
        return obs
    mono = lambda r: float(r["t_mono"]) if r.get("t_mono") is not None else None
    last_mono = max((mono(r) for r in recs if mono(r) is not None), default=0.0)
    spans: dict[str, tuple[float, float]] = {}
    open_spans: dict[str, tuple[str, float]] = {}
    for r in recs:
        if r.get("kind") == "span_start" and mono(r) is not None:
            open_spans[str(r.get("id"))] = (str(r.get("stage")), mono(r))
        elif r.get("kind") == "span_end" and str(r.get("id")) in open_spans and mono(r) is not None:
            st, t0 = open_spans.pop(str(r.get("id")))
            spans[st] = (t0, mono(r))
    for (st, t0) in open_spans.values():
        spans[st] = (t0, last_mono)
    live: dict[str, tuple[str, float, int, str]] = {}
    intervals: list[tuple[str, float, float, int, str]] = []
    for r in recs:
        if r.get("kind") != "mark" or mono(r) is None:
            continue
        if r.get("name") == "alloc" and r.get("obj") is not None:
            aid = str(r.get("alloc_id") or f"{r.get('obj')}#{len(intervals) + len(live)}")
            live[aid] = (str(r["obj"]), mono(r), int(r.get("bytes") or 0), str(r.get("stage") or ""))
        elif r.get("name") == "free":
            aid = str(r.get("alloc_id") or "")
            if aid in live:
                obj, t0, b, st = live.pop(aid)
                intervals.append((obj, t0, mono(r), b, st))
    for aid, (obj, t0, b, st) in live.items():
        intervals.append((obj, t0, last_mono, b, st))
        obs["notes"].append(f"alloc {aid} ({obj}) never freed: closed at the last event")
    obs["allocs"] = [{"obj": o, "t0": t0, "t1": t1, "bytes": b, "stage": st} for o, t0, t1, b, st in intervals]
    # per top-level stage window: peak overlap of named allocations + the sampler's view
    samples = []
    for f in sorted(glob.glob(os.path.join(attempt_dir, "resources*.jsonl"))):
        for r in _jsonl(f):
            if r.get("kind") == "cgroup" and r.get("t_mono") is not None:
                cg = r.get("cgroup") or {}
                ms = cg.get("memory_stat") or {}
                samples.append({"t": float(r["t_mono"]), "cgroup": cg.get("memory_current_bytes"), "anon_shmem": ((ms.get("anon") or 0) + (ms.get("shmem") or 0)) or None, "rss_sum": None})
            elif r.get("kind") == "process" and r.get("t_mono") is not None:
                rows = r.get("processes") or []
                samples.append({"t": float(r["t_mono"]), "cgroup": None, "anon_shmem": None, "rss_sum": sum(int(p.get("rss_anon_bytes") or p.get("rss_bytes") or 0) for p in rows),
                                "rss_max": max((int(p.get("rss_bytes") or 0) for p in rows), default=0)})
    for st, (t0, t1) in spans.items():
        top = st.split(".", 1)[-1] if st.startswith("stage.") else st
        inside = [(o, max(a, t0), min(b, t1), n) for o, a, b, n, _ in intervals if b > t0 and a < t1]
        pk = peak_overlap(inside) if inside else {"peak_bytes": 0, "t": None, "live": {}}
        win = [s for s in samples if t0 <= s["t"] <= t1]
        obs["stages"][top] = {"named_peak_bytes": pk["peak_bytes"], "named_peak_mib": pk["peak_bytes"] // MiB, "live_at_peak": {k: v // MiB for k, v in pk["live"].items()},
                              "cgroup_mib": max((s["cgroup"] or 0) for s in win) // MiB if win else None,
                              "anon_shmem_mib": max((s["anon_shmem"] or 0) for s in win) // MiB if win else None,
                              "rss_sum_mib": max((s["rss_sum"] or 0) for s in win) // MiB if win else None,
                              "rss_max_mib": max((s.get("rss_max") or 0) for s in win) // MiB if win else None, "wall_s": t1 - t0}
    return obs


# ---------------------------------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------------------------------
def _observed_stage_peak(obs: dict[str, Any], stage: str) -> tuple[int | None, str]:
    """Observed host peak (MiB) for a stage: the row's largest process RSS (pre-telemetry records: pod-wide max python RSS) — see notes."""
    st = obs.get("stages", {}).get(stage)
    if not st:
        return None, "no samples"
    return st.get("rss_mib"), f"max_python_rss over {st.get('n')} samples"


def compare(attempt_dir: str, cfg: RowConfig | None = None) -> dict[str, Any]:
    """Predicted vs observed peak per stage (+ per commit point) with the residual and the named attribution."""
    attempt_dir = str(attempt_dir)
    cfg = cfg or row_config_from_attempt(attempt_dir)
    pred = predict(cfg)
    obs = observe_increment_a(attempt_dir) if glob.glob(os.path.join(attempt_dir, "events*.jsonl")) else observe_r17(attempt_dir)
    out: dict[str, Any] = {"schema": "verity/admission-compare/v1", "attempt_dir": attempt_dir, "row": asdict(cfg), "derived": pred["derived"], "observed_source": obs["source"],
                           "stages": {}, "points": {}, "kill": obs.get("kill"), "notes": list(cfg.notes) + list(obs.get("notes", []))}
    limit = cfg.host_limit_bytes
    # hand-recorded observations for attempts whose sampler did not see the peak (kill-time cgroup/ps numbers)
    override: dict[str, Any] = {}
    ov = os.path.join(attempt_dir, "admit_observed.json")
    if os.path.exists(ov):
        override = json.loads(Path(ov).read_text())
        out["notes"] += [f"observed override: {override.get('note', '')}"] + list(override.get("notes", []))
    for stage in ("build", "match", "commit"):
        p = pred[stage]
        o_mib, how = _observed_stage_peak(obs, stage)
        if obs["source"] == "increment_a":
            s = obs["stages"].get(stage) or {}
            o_mib = s.get("rss_max_mib") or s.get("anon_shmem_mib")
            how = "increment A process rss_max / cgroup anon+shmem"
        o_all = None   # the whole row's charge (parent + children + co-resident), when a cgroup anon+shmem view exists
        st_obs = obs.get("stages", {}).get(stage) or {}
        if st_obs.get("anon_shmem_mib"):
            o_all = st_obs["anon_shmem_mib"]
        if stage in override:
            o = override[stage]
            o_mib = o.get("rss_mib", o_mib)
            o_all = o.get("anon_shmem_mib", o_all)
            how = f"override: {o.get('how', '')}"
        verdict = st_obs.get("verdict")
        killed = bool(obs.get("kill")) and stage == _killed_stage(obs)
        # parent-only prediction: the observation is one process's RSS; forked replay workers are charged to the cgroup, not the parent
        children = sum(t["bytes"] for t in p["terms"] if t["name"] == "commit.c2.replay_workers" and t["live"][0] == p["peak_point"]) if stage == "commit" else 0
        parent_pred = (p["reserved_bytes"] - children - sum(p["margin"].values()) * MiB) // MiB
        rec = {"predicted_mib": p["reserved_bytes"] // MiB, "predicted_parent_mib": parent_pred, "predicted_admit": p["admit"], "predicted_verdict": p["verdict"], "observed_mib": o_mib, "observed_all_mib": o_all,
               "observed_how": how, "verdict": verdict, "observed_is_lower_bound": killed or (verdict == "FAIL"), "peak_point": p["peak_point"],
               "live_at_peak_mib": p["live_at_peak"], "margin_mib": p["margin"], "limit_mib": (limit // MiB if limit else None), "cotenant": override.get(stage, {}).get("cotenant")}
        if o_mib is not None:
            rec["residual_parent_mib"] = parent_pred - int(o_mib)
            rec["residual_parent_pct"] = round(100.0 * rec["residual_parent_mib"] / max(1, int(o_mib)), 1)
        if o_all is not None:
            rec["residual_all_mib"] = rec["predicted_mib"] - int(o_all)
        out["stages"][stage] = rec
    # commit points
    for key, o in obs.get("points", {}).items():
        stage, point = key.split("/", 1)
        pts = commit_points(cfg.pairs) if stage == "commit" else (MATCH_POINTS if stage == "match" else BUILD_POINTS)
        if point not in pts:
            continue
        live: dict[str, int] = {}
        for t in pred[stage]["terms"]:
            if t["kind"] in ("host", "base") and pts.index(t["live"][0]) <= pts.index(point) <= pts.index(t["live"][1]):
                live[t["name"]] = live.get(t["name"], 0) + t["mib"]
        children = live.get("commit.c2.replay_workers", 0)
        parent = sum(v for n, v in live.items() if n != "commit.c2.replay_workers")
        pm = parent + children + sum(pred[stage]["margin"].values())
        out["points"][key] = {"predicted_mib": pm, "predicted_parent_mib": parent, "observed_rss_mib": o["rss_mib"], "observed_cgroup_mib": o["cgroup_mib"],
                              "observed_anon_shmem_mib": o["anon_shmem_mib"] or None, "residual_parent_mib": parent - int(o["rss_mib"]),
                              "residual_all_mib": (pm - int(o["anon_shmem_mib"])) if o["anon_shmem_mib"] else None, "live": live, "closed": o["closed"], "n": o["n"]}
    for k in ("commit_ledger_host_rss_after_run_mb", "commit_ledger_pinned_in_use_mib", "f_da_15"):
        if k in obs:
            out[k] = obs[k]
    out["replay_workers"] = {"requested": pred["commit"].get("replay_workers_requested"), "admissible": pred["commit"].get("replay_workers_admissible"),
                             "parent_c2_heap_mib": pred["commit"].get("parent_c2_heap_mib"), "largest_tensor_mib": pred["commit"].get("largest_tensor_mib"),
                             "regime": "largest strata in the parent" if cfg.largest_vu_in_parent else "two forked workers may hold the largest VU",
                             "note": pred["commit"].get("replay_workers_note")}
    return out


def _killed_stage(obs: dict[str, Any]) -> str | None:
    for st, rec in obs.get("stages", {}).items():
        if rec.get("verdict") == "FAIL" and rec.get("rc") in ("137", "137/0", "-9"):
            return st
    for st, rec in obs.get("stages", {}).items():
        if rec.get("closed") is False:
            return st
    return None


# ---------------------------------------------------------------------------------------------------------------------
# rendering / CLI
# ---------------------------------------------------------------------------------------------------------------------
def render_compare_md(results: list[dict[str, Any]]) -> str:
    lines = ["| attempt | model | B | stage | reserved MiB (parent + workers + margin) | parent MiB | observed process MiB | residual parent MiB (%) | observed row MiB (anon+shmem) | residual row | limit MiB | plan | outcome | peak point | live at the predicted peak |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        name = os.path.basename(r["attempt_dir"].rstrip("/"))
        for stage, s in r["stages"].items():
            if s["observed_mib"] is None and s["predicted_mib"] == 0:
                continue
            lb = "≥" if s["observed_is_lower_bound"] else ""
            obs = "—" if s["observed_mib"] is None else f"{lb}{s['observed_mib']}" + (" †" if s.get("cotenant") else "")
            res = "—" if "residual_parent_mib" not in s else f"{s['residual_parent_mib']:+d} ({s['residual_parent_pct']:+.0f}%)"
            o_all = "—" if s.get("observed_all_mib") is None else f"{lb}{s['observed_all_mib']}"
            r_all = "—" if s.get("residual_all_mib") is None else f"{s['residual_all_mib']:+d}"
            plan = {"ADMIT": "admit", "REFUSE": "REFUSE", "LOWER_BOUND": "LOWER_BOUND", "NO_LIMIT": "?"}.get(s.get("predicted_verdict"), "?")
            live = ", ".join(f"{k.split('.', 1)[-1]} {v}" for k, v in list(s["live_at_peak_mib"].items())[:5])
            lines.append(f"| {name} | {r['row']['model']} | {r['row']['requests']} | {stage} | {s['predicted_mib']} | {s['predicted_parent_mib']} | {obs} | {res} | {o_all} | {r_all} | {s['limit_mib'] or '—'} | {plan} | {s['verdict'] or '—'} | {s['peak_point']} | {live} |")
    return "\n".join(lines)


def render_points_md(r: dict[str, Any]) -> str:
    lines = [f"**{os.path.basename(r['attempt_dir'].rstrip('/'))}** — commit points (predicted = Σ live terms + margin; observed = max python RSS / cgroup current / cgroup anon+shmem, MiB)", "",
             "| point | predicted parent | predicted row (+workers +margin) | observed rss | observed cgroup | observed anon+shmem | residual parent | residual row | live terms |", "|---|---|---|---|---|---|---|---|---|"]
    for key, p in r["points"].items():
        if not key.startswith("commit/"):
            continue
        live = ", ".join(f"{k.split('.', 1)[-1]} {v}" for k, v in sorted(p["live"].items(), key=lambda kv: -kv[1]))
        ra = "—" if p["residual_all_mib"] is None else f"{p['residual_all_mib']:+d}"
        lines.append(f"| {key.split('/', 1)[1]}{'' if p['closed'] else ' (killed)'} | {p['predicted_parent_mib']} | {p['predicted_mib']} | {p['observed_rss_mib']} | {p['observed_cgroup_mib']} | {p['observed_anon_shmem_mib'] or '—'} | {p['residual_parent_mib']:+d} | {ra} | {live} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# The runner's entry point (row_pod.sh, right after precheck-target): ADVISORY plan for a sweep row from its workload file.
# ---------------------------------------------------------------------------------------------------------------------
ADMISSION_SCHEMA = "verity-sweep/admission/v1"


def model_key_for_row(row_id: str) -> str:
    """`<model>__<dtype>__<device>__...` -> MODELS key (`-padded` variants share the base model's coefficients)."""
    key = row_tokens(row_id)[0]
    if key not in MODELS and key.endswith("-padded"):
        key = key[: -len("-padded")]
    return key


def row_config_from_workload(workload_path: str | dict[str, Any], row_id: str | None = None, pairs: int = 3, replay_workers: int | None = None,
                             host_limit_bytes: int | None = None, snapshot_steps: str | None = None, snapshot_max_bytes: int | None = None,
                             stages: str = "build,match,commit") -> RowConfig:
    """Plan-time config from the sweep workload (workloads/<row>.json, or its parsed dict): the same facts row_pod.sh reads for B/I/O/LP/T."""
    w = workload_path if isinstance(workload_path, dict) else json.loads(workload_file(workload_path).read_text())
    sw = w.get("sweep") or {}
    row = row_id or sw.get("row_id") or ("" if isinstance(workload_path, dict) else Path(workload_path).stem)
    reqs = w.get("requests") or []
    prompt = sum(int(r.get("prompt_len") or len(r.get("prompt_token_ids") or [])) for r in reqs) or int(sw.get("batch", 1)) * int(sw.get("max_input", 1024))
    decode = sum(int(r.get("max_tokens") or 0) for r in reqs) or int(sw.get("batch", 1)) * int(sw.get("max_output", 128))
    b = int(sw.get("batch") or len(reqs) or 1)
    i_max, o_max = int(sw.get("max_input") or 1024), int(sw.get("max_output") or 128)
    steps = int(w.get("expected_engine_steps") or o_max)
    lp_max = max((int(r.get("prompt_len") or 0) for r in reqs), default=i_max)
    t_max = max((int(r.get("max_tokens") or 0) for r in reqs), default=o_max)
    ea = (w.get("engine") or {}).get("args") or {}
    key = model_key_for_row(row)
    notes = []
    if key not in MODELS:
        notes.append(uncalibrated_model_note(key))
    cfg = RowConfig(model=key, requests=b, prompt_tokens=prompt, decode_rows=decode, steps=steps, max_context=int(ea.get("max_model_len") or (i_max + o_max)),
                    pairs=pairs, replay_workers=replay_workers, host_limit_bytes=host_limit_bytes,
                    largest_request_rows=t_max, largest_request_ctx=lp_max + t_max, notes=notes)
    cfg.prefill_rows = int(ea.get("max_num_batched_tokens") or prompt)
    cfg.snapshot_steps = snapshot_steps or "0,1"
    cfg.snapshot_max_bytes = snapshot_max_bytes
    cfg.stages = stages
    cfg.hf_model = w.get("model")
    return cfg


def cgroup_memory_max() -> int | None:
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
        except OSError:
            continue
        if v and v != "max":
            n = int(v)
            if n < (1 << 60):
                return n
    return None


def plan_row(cfg: RowConfig, out_path: str | None, advisory: bool = True) -> dict[str, Any]:
    """predict() for one sweep row -> admission.json beside the attempt + the one-line summary row_pod.sh logs.

    ADVISORY: the document records per stage reserved_bytes, the named terms, verdict (ADMIT / REFUSE / LOWER_BOUND /
    NO_LIMIT) and short_by; `would_refuse` lists the stages a fail-closed runner would have stopped.  Nothing here stops a run.
    """
    pred = predict(cfg)
    stages = [s for s in ("build", "match", "commit") if s in pred and s in (getattr(cfg, "stages", "build,match,commit") or "").split(",")]
    doc: dict[str, Any] = {
        "schema": ADMISSION_SCHEMA, "advisory": advisory, "t_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),  # noqa: UP017, RUF100
        "model": cfg.model, "hf_model": getattr(cfg, "hf_model", None), "host_limit_bytes": cfg.host_limit_bytes,
        "config": {"requests": cfg.requests, "prompt_tokens": cfg.prompt_tokens, "decode_rows": cfg.decode_rows, "steps": cfg.steps, "max_context": cfg.max_context,
                   "pairs": cfg.pairs, "replay_workers": cfg.workers(), "prefill_rows": getattr(cfg, "prefill_rows", None),
                   "snapshot_steps": getattr(cfg, "snapshot_steps", None), "snapshot_max_bytes": getattr(cfg, "snapshot_max_bytes", None),
                   "largest_vu_in_parent": cfg.largest_vu_in_parent, "n_instances_est": cfg.instances(), "largest_tensor_bytes_est": cfg.largest_tensor(),
                   # the host-working-set mechanisms the plan models (gc.freeze default-on in the code; bounded staging by flags)
                   "fork_gc_freeze": cfg.fork_gc_freeze, "retain": cfg.retain, "bounded_staging": cfg.bounded_staging, "retain_exclude": sorted(cfg.retain_exclude)},
        "notes": list(cfg.notes), "stages": {}, "would_refuse": [], "lower_bound": [],
    }
    for s in stages:
        p = pred[s]
        doc["stages"][s] = {"reserved_bytes": p["reserved_bytes"], "reserved_mib": p["reserved_mib"], "peak_point": p["peak_point"], "verdict": p["verdict"],
                            "decision": p["decision"], "lower_bound_terms": p["lower_bound_terms"], "live_at_peak_mib": p["live_at_peak"], "margin_mib": p["margin"],
                            "device_bytes": p["device_bytes"], "terms": p["terms"]}
        if s == "commit":
            doc["stages"][s].update({k: p[k] for k in ("pinned_pool", "fork_gc_freeze", "parent_c2_heap_mib", "largest_tensor_mib", "replay_workers_requested",
                                                       "replay_workers_admissible", "replay_workers_note") if k in p})
        if p["verdict"] == "REFUSE":
            doc["would_refuse"].append(s)
        elif p["verdict"] == "LOWER_BOUND":
            doc["lower_bound"].append(s)
    parts = []
    for s in stages:
        d = doc["stages"][s]
        tail = f"(short {d['decision'].get('short_by_mib')} MiB)" if d["verdict"] == "REFUSE" else (f"({','.join(d['lower_bound_terms'])})" if d["verdict"] == "LOWER_BOUND" else "")
        parts.append(f"{s}={d['verdict']} {d['reserved_mib']}MiB{tail}")
    lim = f"limit {cfg.host_limit_bytes // MiB} MiB" if cfg.host_limit_bytes else "no cgroup limit read"
    flag = ("WARN would-refuse " + ",".join(doc["would_refuse"]) + " ") if doc["would_refuse"] else ("NOTE lower-bound " + ",".join(doc["lower_bound"]) + " " if doc["lower_bound"] else "")
    doc["line"] = f"ADMISSION advisory {flag}{cfg.model} {lim}: " + "; ".join(parts) + (f" [{'; '.join(cfg.notes)}]" if cfg.notes else "")
    if out_path:
        Path(out_path).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    return doc
