"""verity_vllm/check/gates.py — the frozen invariant set G1..G7 over the JSON artifacts of one run directory (torch-free).

THE GATES AND THE CONSTANTS BELOW ARE FROZEN: they change only through an approved decision recorded in
`out/capture/decisions.md`, never by editing this file alone.  Anything that is a heuristic (per-step histogram
equality, linear instance growth, the family-reference match, the overhead ladder) is a DIAGNOSTIC reported in the
card (`diagnostics()`), never a gate.

A *run directory* is what `bench/run_config.py` produces for one configuration (model × engine arguments × κ):

    RUN_DIR/run.json                 the driver's manifest (config_id, profile, case, workload, C, laptop, snapshot_steps,
                                     artifact path overrides)             — read by `RunArtifacts`
    RUN_DIR/stages.json              per-stage {status, seconds, returncode, log}   — a failed stage becomes evidence
    RUN_DIR/capture/{log.jsonl.gz, tokens.json, versions.json, stats.json, snapshots.json, cos_sin_cache.npy}
    RUN_DIR/control/tokens.json      the `--no-observer` control run: THE token reference of every gate
    RUN_DIR/resolution.json          `bench.resolve_log --out RUN_DIR/resolution`
    RUN_DIR/fold_summary.json        this module's `FOLD_SUMMARY_SCHEMA` (written twice: `fold_summary_2.json`)
    RUN_DIR/census/census.json       GPU census;  RUN_DIR/census_reconcile.json  its torch-free reconciliation
    RUN_DIR/nonint/noninterference.json
    RUN_DIR/golden.json              `verity_vllm.properties.golden --check` (integrator)
    RUN_DIR/quarantine.json          differential tests of newly registered definitions (when the run uses any)
    RUN_DIR/holdouts.json            written by the ORCHESTRATOR: the same gates G1..G6 over holdout workloads
    RUN_DIR/compare.json             `verity_vllm.check.fold_compare` (tokens, boundary, subcircuits, structure)
    RUN_DIR/wiring.json, replay_a.json, replay_chain.json, replay_b_step<N>.json, root_policy.json, provenance.json
                                     no producer in this tree: a gate that reads one is skipped with the file missing as
                                     its reason

Gates:
    G1 observation fidelity    non-interference PASS, tokens equal across a/b/c AND to the control run, every boundary
                               hash equal, steps == max_tokens, layers == expected_layers; census residue 0, ambiguous 0,
                               every declared kernel's launch count equal to its declaring events
    G2 semantic closure        0 Unsupported, 0 Unresolved, 0 ambiguous pattern matches
    G3 root closure            0 unknown roots, 0 opaque/prescribed fallbacks, root_policy pass with no new root class
    G4 primitive exactness     tier-a replay pass, k >= TIER_A_K per (specialization, regime) stratum, every captured
                               specialization compared; quarantine differential tests all pass when new definitions are used
    G5 composition exactness   (D82) wiring.json: EVERY operand read of EVERY instance reproduced by the independent
                               last-writer map (storage identity + generation + byte range + stream order), 0 uncovered,
                               one stream, 0 writes outside the schema-declared write set;  replay_a.json: the verifier-
                               seeded stratified sample passes and every served token at every snapshot step (>=
                               MIN_DECODE_TOKEN_STEPS decode steps) is reproduced from its observed logits;
                               replay_chain.json: >= CHAIN_MIN_SLICES connected slices replayed with computed intermediates,
                               all pass.  Full replay (replay_b_*) is reported when present, never gating.
    G6 canonicality+regression two independent folds -> one digest; golden.json pass
    G7 holdout generalization  holdouts.json present, >= 1 holdout, G1..G6 pass in every holdout

    G8 VU partition            STRUCTURAL, outside the frozen set (query track): `query_use.json`
                               (`verity-ir/query-use/v1`, the partition track's use configuration + verdict document);
                               `skipped` ("no query-use document") when absent, pass iff verdict.partition.status == "exact"
                               and verdict.width.status == "ok" (no violators, no unsupported) and verdict.status == "ok";
                               `fail` (with the codes) on overlap / gap / violation / `not-ready` / `fail` / a binding
                               mismatch with `query.json`; `unsupported` (a fifth status, lead 22:22Z) when the evaluator
                               could not decide -- neither pass nor fail, the run verdict INCOMPLETE, never REJECTED.
                               A run without the document keeps today's verdict: G8 `skipped` never makes it INCOMPLETE.

    R1 reference integrity     FAIL-CLOSED (`REF_INTEGRITY_RULE`; explicitly authorised checker
                               addition, outside the frozen set): `fold/construction errors > 0 ⇒ the Program is ineligible
                               as an accepted reference`.  Evidence = `fold_summary.json` (`errors`, `unresolved`,
                               `unsupported`; `error_classes` / `per_step` must agree with `errors`), `fold_summary_2.json`
                               when present, `resolution.json` `totals.errors` / `totals.unsupported` when present.  Any
                               count > 0 -> `fail` (verdict REJECTED).  Missing summary, missing / non-integer count or
                               inconsistent evidence -> `unverified` (verdict INCOMPLETE): missing error evidence is never
                               read as zero.  Never `skipped`.  `fold_integrity(...)` is the pure rule over the dicts so
                               any consumer of a fold Program (e.g. the AOT roundtrip's `--accepted`) can apply it.

Every gate is a pure function `gate_Gx(run: RunArtifacts) -> GateResult`; nothing here imports torch, replay, analytic,
root_policy or golden.  A missing artifact makes the gate `skipped` (with the producing stage's log as evidence), a
malformed one `error`; a gate never raises.  `evaluate_gates(run_dir)` runs the seven gates, G8 and R1; `verdict(results)` folds
them into ACCEPTED / REJECTED / INCOMPLETE (G1..G7 as before; G8 / R1 contribute only `fail` -> REJECTED, `error` /
`unsupported` / `unverified` -> INCOMPLETE; a record that predates them keeps the verdict of G1..G7).

The fold summary the driver writes (`run_config --worker fold`) is defined here: its fields are `FOLD_SUMMARY_FIELDS`, and
its id is `FOLD_SUMMARY_SCHEMA`, which `records.schema_ids` states (also reproduced in `out/gen/cards/SCHEMA.md`).

The code is in three modules.  `gate_base` holds the frozen constants, the artifacts and their reader, a gate's
result and the registry; `gate_rules` the gates and R1; this one evaluates them, folds the verdict, writes the gates
document and the diagnostics, and states the fold summary's fields.  Every name of the other two is handed on here:
other code reads it from `gates`.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from verity_vllm.check.gate_base import (
    _GATE_FNS,
    ALL_GATES,
    ARTIFACT_STAGE,
    ARTIFACTS,
    CHAIN_MIN_SLICES,
    COS_SIN_MAX_ULP,
    DIAG_MAX_LINEAR_RESIDUAL,
    MIN_DECODE_TOKEN_STEPS,
    QUERY_USE_KIND,
    REF_INTEGRITY_RULE,
    REGIMES,
    TIER_A_K,
    ArtifactError,
    GateResult,
    RunArtifacts,
    load_profile_config,
    replay_b_key,
    replay_b_rel,
)
from verity_vllm.check.gate_base import _gate as _gate
from verity_vllm.check.gate_base import _get as _get
from verity_vllm.check.gate_base import _int as _int
from verity_vllm.check.gate_base import _read_json as _read_json
from verity_vllm.check.gate_base import _result as _result
from verity_vllm.check.gate_base import _Skip as _Skip
from verity_vllm.check.gate_rules import _INTEGRITY_COUNTS as _INTEGRITY_COUNTS
from verity_vllm.check.gate_rules import _ambiguity as _ambiguity
from verity_vllm.check.gate_rules import _count_field as _count_field
from verity_vllm.check.gate_rules import (
    fold_integrity,
    gate_canonicality,
    gate_composition,
    gate_holdout,
    gate_observation,
    gate_primitive,
    gate_R1,
    gate_root_closure,
    gate_semantic_closure,
    gate_vu_partition,
    quarantine_specs,
)
from verity_vllm.check.verdict import match_verdict
from verity_vllm.check.verdict_names import (
    CANONICALITY,
    COMPOSITION,
    GATE_NAMES,
    GATES,
    HOLDOUT,
    HOLDOUT_GATES,
    INTEGRITY_GATES,
    OBSERVATION,
    PRIMITIVE,
    ROOT_CLOSURE,
    SEMANTIC_CLOSURE,
    STRUCTURAL_GATES,
    VU_PARTITION,
)
from verity_vllm.records.schema_ids import (
    FOLD_SUMMARY_SCHEMA,
    GATES_SCHEMA,
    QUERY_USE_SCHEMA,
)

# ---------------------------------------------------------------------------------------------------------
# evaluation and verdict
# ---------------------------------------------------------------------------------------------------------


def evaluate_gates(run_dir: str | Path, *, manifest: dict | None = None) -> list[GateResult]:
    """The seven frozen gates, G8 and R1 over one run directory, in `ALL_GATES` order.  Never raises for artifact problems."""
    run = RunArtifacts(run_dir, manifest)
    return [_GATE_FNS[g](run) for g in ALL_GATES]


def verdict(results: list[GateResult]) -> str:
    """ACCEPTED iff all seven frozen gates pass; REJECTED if any gate fails; INCOMPLETE otherwise (skipped / error / missing).  Gates outside the frozen set (G8, R1)
    contribute only `fail` -> REJECTED and `error` / `unsupported` / `unverified` -> INCOMPLETE: a run without a query-use document (G8 `skipped`) or a record that
    predates G8 / R1 keeps the verdict of G1..G7.  [R9 ref] R1 `unverified` (no fold error evidence) is INCOMPLETE, never ACCEPTED — the fail-closed rule."""
    return match_verdict([r.result() for r in results], GATES)


def gates_json(results: list[GateResult], run_dir: str | Path | None = None, diagnostics_: dict | None = None) -> dict:
    return {"schema": GATES_SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run_dir": str(run_dir) if run_dir is not None else None, "verdict": verdict(results),
            "order": [r.id for r in results], "frozen": list(GATES), "gates": {r.id: r.to_json() for r in results},
            "constants": {"TIER_A_K": TIER_A_K, "MIN_DECODE_TOKEN_STEPS": MIN_DECODE_TOKEN_STEPS, "CHAIN_MIN_SLICES": CHAIN_MIN_SLICES,
                          "REGIMES": list(REGIMES), "COS_SIN_MAX_ULP": COS_SIN_MAX_ULP,
                          "DIAG_MAX_LINEAR_RESIDUAL": DIAG_MAX_LINEAR_RESIDUAL,
                          "REF_INTEGRITY_RULE": f"{REF_INTEGRITY_RULE['id']}/{REF_INTEGRITY_RULE['version']}"},
            "diagnostics": diagnostics_}


def write_gates(run_dir: str | Path, results: list[GateResult], diagnostics_: dict | None = None) -> Path:
    p = Path(run_dir) / ARTIFACTS["gates"]
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(gates_json(results, run_dir, diagnostics_), indent=1) + "\n")
    return p


def gates_markdown(results: list[GateResult]) -> str:
    L = ["| gate | name | status | reason |", "| --- | --- | --- | --- |"]
    for r in results:
        L.append(f"| {r.id} | {r.name} | **{r.status}** | {(r.reason or '').replace('|', '/')[:200]} |")
    L.append("")
    L.append(f"verdict: **{verdict(results)}**")
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------------------
# DIAGNOSTICS — reported in the card, never gating
# ---------------------------------------------------------------------------------------------------------

_T_RE = re.compile(r"T=\d+,?")


def strip_T(label: str) -> str:
    """`Attention_v3{T=17,NH=12,...}` -> `Attention_v3{NH=12,...}` (the per-position attention length is workload shape)."""
    return _T_RE.sub("", label).replace(",}", "}")


def _linear_residual(xs: list[float], ys: list[float]) -> float:
    """Max |y - (a + b x)| of the least-squares line (0 when fewer than 3 points)."""
    n = len(xs)
    if n < 3:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    return max(abs(y - (a + b * x)) for x, y in zip(xs, ys))


def diag_structure(fs: dict | None) -> dict:
    """Per-decode-step specialization multiset equality modulo attention `T=`, constant instance count per decode step,
    cumulative instance count linear in the step."""
    if not isinstance(fs, dict) or not isinstance(fs.get("per_step"), list) or not fs["per_step"]:
        return {"status": "unavailable"}
    per_step = fs["per_step"]
    decode = [s for s in per_step if s.get("phase") == "decode"]
    prefill = [s for s in per_step if s.get("phase") == "prefill"]
    stripped: list[Counter] = []
    for s in decode:
        c: Counter = Counter()
        histogram = s.get("spec_histogram")   # a step that states none counts no specialization
        for label, n in histogram.items() if histogram else ():
            c[strip_T(label)] += int(n)
        stripped.append(c)
    counts = [int(s.get("n_instances", 0)) for s in decode]
    diffs = []
    for s, c in zip(decode, stripped):
        if stripped and c != stripped[0]:
            d = {k: [stripped[0].get(k, 0), c.get(k, 0)] for k in set(stripped[0]) | set(c) if stripped[0].get(k, 0) != c.get(k, 0)}
            diffs.append({"step": s.get("step"), "differs": dict(list(d.items())[:6])})
    xs = [float(s.get("step")) for s in decode]
    cum, acc = [], 0
    for s in per_step:
        acc += int(s.get("n_instances", 0))
        if s.get("phase") == "decode":
            cum.append(float(acc))
    resid = _linear_residual(xs, cum) if len(xs) == len(cum) and xs else float("inf")
    return {"status": "ok", "steps": len(per_step), "prefill_steps": len(prefill), "decode_steps": len(decode),
            "prefill_instances": [int(s.get("n_instances", 0)) for s in prefill], "decode_instances": sorted(set(counts)),
            "histograms_equal_mod_T": not diffs, "histogram_diffs": diffs[:3], "count_constant": len(set(counts)) <= 1,
            "linear_residual": None if math.isinf(resid) else round(resid, 4), "linear": resid <= DIAG_MAX_LINEAR_RESIDUAL,
            "decode_multiset_mod_T": dict(sorted(stripped[0].items())) if stripped else None}


def diag_family_match(cmp_: dict | None) -> dict:
    """The compare report's `structure` section (hand-authored family reference), with its unclassified extra classes."""
    st = (cmp_ or {}).get("structure") if isinstance(cmp_, dict) else None
    if not isinstance(st, dict):
        return {"status": "unavailable"}
    classes = st.get("classes")
    if not classes:
        classes = {}   # a structure report that states no class has none, classified or not
    uncl = sorted(k for k in classes if str(k).startswith("unclassified:"))
    return {"status": st.get("status"), "family": st.get("family"), "matched": st.get("matched"), "superset": st.get("superset"),
            "n_extra": st.get("n_extra"), "classes": {k: (v.get("count") if isinstance(v, dict) else v) for k, v in classes.items()},
            "unclassified": len(uncl), "unclassified_classes": uncl, "reason": st.get("reason"),
            "n_derived_nodes": st.get("n_derived_nodes"), "n_family_leaves": st.get("n_family_leaves")}


def diag_overhead(m0: dict | None, versions: dict | None) -> dict:
    """The M0 overhead ladder (step wall per rung, × baseline) with the host's clock-read cost, and the per-recorded-call
    cost with the hook's four clock reads subtracted (M6 addendum: 4 reads per recorded call)."""
    if not isinstance(m0, dict) or not isinstance(m0.get("results"), dict):
        return {"status": "unavailable"}
    recorded = m0.get("versions")   # the capture's versions.json states the host, else the ladder's own record does; else no field of it is known
    host = (versions.get("host") if versions else None) or (recorded.get("host") if recorded else None)
    if not host:
        host = {}
    clock_us = host.get("clock_read_us")
    base = m0.get("baseline_step_wall_s")
    rungs = {}
    for rung, r in m0["results"].items():
        timed = r.get("reps")   # a rung that states no repetition was not timed
        reps = [x for x in timed if isinstance(x, dict) and x.get("step_wall_mean_s") is not None] if timed else []
        if not reps:
            continue
        step = sum(x["step_wall_mean_s"] for x in reps) / len(reps)
        calls = sum(int(x.get("recorded_calls") or 0) + int(x.get("triton_launches") or 0) for x in reps) / len(reps)
        steps_n = sum(int(x.get("observed_steps") or 0) for x in reps) / len(reps) or 1
        row: dict[str, Any] = {"label": r.get("label"), "step_wall_ms": round(step * 1e3, 2),
                               "x_baseline": round(step / base, 2) if base else None, "events_per_step": round(calls / steps_n, 1) if calls else 0}
        if base and calls:
            per_call_us = (step - base) / (calls / steps_n) * 1e6
            row["overhead_per_event_us"] = round(per_call_us, 1)
            if clock_us is not None:
                row["overhead_per_event_us_minus_clock"] = round(per_call_us - 4 * float(clock_us), 1)
        rungs[rung] = row
    return {"status": "ok", "baseline_step_wall_ms": round(base * 1e3, 2) if base else None, "reps": m0.get("reps"),
            "host": {k: host.get(k) for k in ("cpu", "nproc", "cpu_quota", "clocksource", "clock_read_us")},
            "note": "compare ladders across pods only when host.clocksource matches; `minus_clock` subtracts the hook's 4 perf_counter reads per event",
            "rungs": rungs}


def diagnostics(run: RunArtifacts | str | Path) -> dict:
    """All diagnostics over a run directory (never gating): structure heuristics, family-reference match, overhead ladder."""
    if not isinstance(run, RunArtifacts):
        run = RunArtifacts(run)

    def safe(fn, *keys):
        try:
            return fn(*[run.json(k) for k in keys])
        except ArtifactError as e:
            return {"status": "error", "reason": str(e)}

    return {"structure": safe(diag_structure, "fold_summary"), "family_match": safe(diag_family_match, "compare"),
            "overhead": safe(diag_overhead, "m0", "versions")}


# ---------------------------------------------------------------------------------------------------------
# the fold summary the driver writes (schema owned here, produced by `bench/run_config.py --worker fold`)
# ---------------------------------------------------------------------------------------------------------

FOLD_SUMMARY_FIELDS: dict[str, str] = {
    "schema": f"`{FOLD_SUMMARY_SCHEMA}`",
    "utc": "UTC time the summary was written",
    "log": "path of the raw observation log folded",
    "profile": "profile module name (`verity_vllm.observe.fold.profiles.<name>`)",
    "profile_id": "`Profile.id` of that module",
    "with_program": "true when `Program()` and the digest were built (G6 needs it)",
    "digest": "`program_digest(Program(Captured_v1{PROFILE, WORKLOAD}))` — the structure digest G6 compares (empty without Program())",
    "spec": "the derived specialization label `Captured_v1{PROFILE=..., WORKLOAD=...}`",
    "n_instances": "instances emitted by the fold (= flat body nodes)",
    "n_steps": "engine steps folded",
    "unsupported": "Unsupported resolutions seen by the fold (`ResolutionReport.unsupported`)",
    "unresolved": "fold errors of class `Unresolved` (memory reads without provenance)",
    "opaque_fallbacks": "reads served by an opaque / prescribed value instead of provenance (0 by construction; fold errors of class Opaque/Prescribed)",
    "errors": "all fold errors recorded in development mode (0 under strict)",
    "error_classes": "histogram of fold error classes",
    "error_examples": "first fold errors (<= 16)",
    "unknown_roots": "weight/buffer roots the profile classifies as `unknown` (via its `root_class`, as `resolve_log` does)",
    "unbound_roots": "roots without a weights field (the fold's `unbound root` notes; ignored fp8-scale placeholders are expected here)",
    "per_step": "list of `{step, phase, rows, n_instances, computes, spec_histogram: {label: instances}, unsupported, errors}`",
    "tokens": "`{request_id: [sampled token ids in step order]}` (the fold's TokenSelect outputs, from SampleEnd)",
    "sampled_steps": "`{request_id: [steps at which the tokens above were sampled]}`",
    "requests": "`[{request_id, prompt_len, max_tokens, n_tokens}]` from the log header and the fold",
    "params": "the Program parameters `[[name, type], ...]`",
    "ret_type": "the Program return type",
    "memory": "memory-model statistics (`MemoryModel.stats()` + storage alloc/free counts)",
    "timings": "`{fold_s, program_s, digest_s}`",
    "seconds": "wall time of the worker",
    "max_rss_gb": "maximum resident set of the worker process (D69 footprint)",
    "fold": "`compare.summarize(result)` verbatim (the layout compare.py's report has under `fold`)",
}


__all__ = [
    "TIER_A_K", "MIN_DECODE_TOKEN_STEPS", "CHAIN_MIN_SLICES", "REGIMES", "COS_SIN_MAX_ULP", "DIAG_MAX_LINEAR_RESIDUAL", "GATES", "GATE_NAMES", "HOLDOUT_GATES",
    "STRUCTURAL_GATES", "INTEGRITY_GATES", "REF_INTEGRITY_RULE", "ALL_GATES", "QUERY_USE_SCHEMA", "QUERY_USE_KIND",
    "OBSERVATION", "SEMANTIC_CLOSURE", "ROOT_CLOSURE", "PRIMITIVE", "COMPOSITION", "CANONICALITY", "HOLDOUT", "VU_PARTITION",
    "GATES_SCHEMA", "FOLD_SUMMARY_SCHEMA", "FOLD_SUMMARY_FIELDS", "ARTIFACTS", "ARTIFACT_STAGE", "replay_b_key", "replay_b_rel",
    "GateResult", "ArtifactError", "RunArtifacts", "load_profile_config",
    "gate_observation", "gate_semantic_closure", "gate_root_closure", "gate_primitive", "gate_composition", "gate_canonicality", "gate_holdout",
    "gate_vu_partition", "gate_R1", "fold_integrity", "quarantine_specs",
    "evaluate_gates", "verdict", "gates_json", "write_gates", "gates_markdown",
    "strip_T", "diag_structure", "diag_family_match", "diag_overhead", "diagnostics",
]
