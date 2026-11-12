"""Merge the control-only and treatment-only commit_delta outputs of an Execution=compiled row into ONE paired summary
(summary.json / summary.md / runs.jsonl / verdict.json in --out), in the shape row_pod.sh's Commit phase reads.

The two arms are separate processes because the treatment's record ops live inside the compiled graphs for the whole process; pairing is
by run index (control run i <-> treatment run i).  Refuses when the two processes' compilation settings differ (CompilationMode,
CUDAGraphMode, capture sizes, enforce_eager) or when the generated tokens differ (tokens_digest), so an eager control cannot be paired with
a compiled treatment (and vice versa)."""
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from verity_vllm.config import option


def _rows(p: Path) -> list[dict]:
    f = p / "runs.jsonl"
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()] if f.exists() else []


def _json(p: Path, name: str) -> dict:
    f = p / name
    return json.loads(f.read_text()) if f.exists() else {}


def _bootstrap_ci(xs: list[float], n: int = 4000, *, seed: int) -> tuple[float, float]:
    from verity_vllm.commit import challenge as CH

    if len(xs) < 2:
        return (float("nan"), float("nan"))
    rng = CH.stream(seed)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


COMPILE_KEYS = ("compilation_mode", "cudagraph_mode", "cudagraph_capture_sizes", "enforce_eager", "use_inductor_graph_partition", "splitting_ops",
                "inductor_compile_config")


@dataclass(frozen=True, kw_only=True)
class Options:
    control: list[str] = option("--control", action="append",
                                help="control-process output dir (repeatable: one per round, in run order)")
    treatment: list[str] = option("--treatment", action="append", help="treatment-process output dir (repeatable)")
    out: str = option("--out")


def main(a: Options) -> None:
    Cs, Ts, out = [Path(x) for x in a.control], [Path(x) for x in a.treatment], Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctl = [r for C in Cs for r in _rows(C)]
    trt = [r for T in Ts for r in _rows(T)]
    for i, r in enumerate(ctl):
        r["pair"] = i
    for i, r in enumerate(trt):
        r["pair"] = i
    C, T = Cs[0], Ts[0]
    vc, vt = _json(C, "versions.json"), _json(T, "versions.json")
    cc, ct = vc.get("compilation", {}), vt.get("compilation", {})
    problems = []
    for X in Cs[1:] + Ts[1:]:                                   # every process of both arms must report the same compilation settings
        cx = _json(X, "versions.json").get("compilation", {})
        for k in COMPILE_KEYS:
            if cx.get(k) != ct.get(k):
                problems.append(f"compilation setting differs in {X.name}: {k}: {cx.get(k)!r} vs {ct.get(k)!r}")
    rounds = len(Cs)
    if not ctl:
        problems.append("control: no runs")
    if not trt:
        problems.append("treatment: no runs")
    for k in COMPILE_KEYS:
        if cc.get(k) != ct.get(k):
            problems.append(f"compilation setting differs: {k}: control {cc.get(k)!r} vs treatment {ct.get(k)!r}")
    if cc.get("enforce_eager") is not False:
        problems.append(f"control is not compiled (enforce_eager={cc.get('enforce_eager')!r})")
    if ct.get("compiled_taps") != "served":
        problems.append("treatment process did not run with --compiled-taps served")
    if cc.get("compiled_taps") not in (None, "off"):
        problems.append("control process ran with compiled taps")
    # the generated-kernel identity of both processes: the Triton kernel SOURCES Inductor emitted (kernel_set_sha256 over
    # {(name, source sha256)}) and the per-piece launch programs.  The treatment's wrappers carry its hooks' lines, so its launch
    # programs (call_set) legitimately differ; its KERNEL SET must equal the control's, and a difference is listed kernel by kernel.
    gkc, gkt = cc.get("generated_kernels") or {}, ct.get("generated_kernels") or {}
    kernels_identified = bool(gkc.get("kernel_set_sha256")) and bool(gkt.get("kernel_set_sha256"))
    kernel_set_equal = kernels_identified and gkc.get("kernel_set_sha256") == gkt.get("kernel_set_sha256")
    kc, kt = gkc.get("kernels") or {}, gkt.get("kernels") or {}
    kernel_diff = {"only_control": sorted(set(kc) - set(kt))[:40], "only_treatment": sorted(set(kt) - set(kc))[:40],
                   "same_name_different_source": sorted(k for k in set(kc) & set(kt) if kc[k] != kt[k])[:40]} if kernels_identified else None
    # served: the Build of record is derived from the treatment's own Inductor artifact and rebuilt after its runs (manifest_verify); the
    # control must have run the same generated kernels, else the record is not the served implementation
    if ct.get("compiled_taps") == "served":
        if not kernels_identified:
            problems.append("served path: generated kernels not identified in both arms (inductor_kernels.json)")
        elif not kernel_set_equal:
            problems.append(f"served path: the treatment's generated kernel set differs from the control's: {kernel_diff}")
    tok_c = {r.get("tokens_digest") for r in ctl}
    tok_t = {r.get("tokens_digest") for r in trt}
    tokens_equal = bool(tok_c) and tok_c == tok_t and len(tok_c) == 1
    if not tokens_equal:
        problems.append(f"generated tokens differ across arms/runs: control {sorted(x[:12] for x in tok_c if x)} treatment {sorted(x[:12] for x in tok_t if x)}")
    n = min(len(ctl), len(trt))
    pairs = []
    for i in range(n):
        c, t = ctl[i], trt[i]
        pairs.append({"pair": i, "control_wall_s": c["wall_s"], "treatment_wall_s": t["wall_s"], "delta_wall_s": t["wall_s"] - c["wall_s"],
                      "control_decode_mean_s": c.get("step_wall_decode_mean_s"), "treatment_decode_mean_s": t.get("step_wall_decode_mean_s"),
                      "control_prefill_s": c.get("step_wall_prefill_s"), "treatment_prefill_s": t.get("step_wall_prefill_s"),
                      "commit_pass": (t.get("commit") or {}).get("commit_pass"), "run_root": (t.get("commit") or {}).get("run_root")})
    deltas = [p["delta_wall_s"] for p in pairs]
    ratio = [p["treatment_wall_s"] / p["control_wall_s"] for p in pairs if p["control_wall_s"]]
    lo, hi = _bootstrap_ci(deltas, seed=0)
    commit_pass = bool(trt) and all(bool((r.get("commit") or {}).get("commit_pass")) for r in trt) and not problems
    fail_reasons = [(r.get("commit") or {}).get("commit_fail_reason") for r in trt if not (r.get("commit") or {}).get("commit_pass")]
    tv = _json(T, "verdict.json")
    for X in Ts[1:]:
        v2 = _json(X, "verdict.json")
        tv["run_roots"] = list(tv.get("run_roots") or []) + list(v2.get("run_roots") or [])
        tv["runs_instrumented"] = int(tv.get("runs_instrumented") or 0) + int(v2.get("runs_instrumented") or 0)
        tv["runs_failed"] = int(tv.get("runs_failed") or 0) + int(v2.get("runs_failed") or 0)
    summary = {
        "schema": "verity-commit-delta/v1/compiled-paired", "label": (trt[0].get("label") if trt else None), "pairs": n, "rounds": rounds,
        "pairing": f"separate processes (control: no record ops; treatment: record ops compiled into the graphs), {rounds} alternating round(s) "
                   f"(c,t / t,c), {len(ctl)} control + {len(trt)} treatment runs paired by global run index",
        "compilation": ct, "compilation_control": cc, "compilation_settings_equal": all(cc.get(k) == ct.get(k) for k in COMPILE_KEYS),
        "tokens_equal": tokens_equal,
        "generated_kernels": {"identified": kernels_identified, "kernel_set_equal": kernel_set_equal,
                              "control": {k: gkc.get(k) for k in ("wrappers", "distinct_kernel_sources", "kernel_set_sha256", "call_set_sha256", "wrapper_norm_set_sha256", "per_process_cache_dir")},
                              "treatment": {k: gkt.get(k) for k in ("wrappers", "distinct_kernel_sources", "kernel_set_sha256", "call_set_sha256", "wrapper_norm_set_sha256", "per_process_cache_dir")},
                              "kernel_diff": kernel_diff},
        "control_wall_s": [r["wall_s"] for r in ctl], "treatment_wall_s": [r["wall_s"] for r in trt],
        "delta_wall_mean": statistics.fmean(deltas) if deltas else None, "delta_wall_ci95": [lo, hi], "delta_wall": deltas,
        "wall_ratio_mean": statistics.fmean(ratio) if ratio else None,
        "decode_mean_control_s": statistics.fmean([p["control_decode_mean_s"] for p in pairs if p["control_decode_mean_s"]]) if pairs else None,
        "decode_mean_treatment_s": statistics.fmean([p["treatment_decode_mean_s"] for p in pairs if p["treatment_decode_mean_s"]]) if pairs else None,
        "prefill_control_s": statistics.fmean([p["control_prefill_s"] for p in pairs if p["control_prefill_s"]]) if pairs else None,
        "prefill_treatment_s": statistics.fmean([p["treatment_prefill_s"] for p in pairs if p["treatment_prefill_s"]]) if pairs else None,
        "per_pair": pairs, "commit_pass": commit_pass, "commit_fail_reasons": fail_reasons, "problems": problems,
        "openings_verified": [((r.get("commit") or {}).get("openings") or {}).get("verified") for r in trt],
        "coverage": [((r.get("commit") or {}).get("coverage") or {}).get("ok") for r in trt],
        "value_correspondence": [((r.get("commit") or {}).get("value_correspondence") or {}).get("result") for r in trt],
        "compiled_source": [((r.get("commit") or {}).get("compiled_source") or {}).get("stats") for r in trt],
        "run_roots": [(r.get("commit") or {}).get("run_root") for r in trt],
        "treatment_verdict": tv, "versions": vt, "versions_control": vc,
    }
    md = [f"### Execution=compiled paired Commit ({n} pairs over {rounds} alternating round(s); control/treatment separate processes, same compilation settings: {summary['compilation_settings_equal']})",
          f"- compiled: mode {ct.get('compilation_mode')} cudagraph {ct.get('cudagraph_mode')} sizes {ct.get('cudagraph_capture_sizes')} enforce_eager {ct.get('enforce_eager')} inductor-partition {ct.get('use_inductor_graph_partition')}",
          f"- wall control {summary['control_wall_s']} s; treatment {summary['treatment_wall_s']} s; delta mean {summary['delta_wall_mean']} s (95% CI {lo:.4f}..{hi:.4f}); ratio {summary['wall_ratio_mean']}",
          f"- decode step mean control {summary['decode_mean_control_s']} s vs treatment {summary['decode_mean_treatment_s']} s; prefill {summary['prefill_control_s']} vs {summary['prefill_treatment_s']} s",
          f"- tokens equal across arms: {tokens_equal}; openings verified {summary['openings_verified']}; manifest coverage {summary['coverage']}; value correspondence {summary['value_correspondence']}",
          f"- generated kernels (Inductor output code): identified {kernels_identified}; control {gkc.get('wrappers')} pieces / {gkc.get('distinct_kernel_sources')} kernel sources "
          f"kernel_set {str(gkc.get('kernel_set_sha256'))[:16]}; treatment {gkt.get('wrappers')} pieces / {gkt.get('distinct_kernel_sources')} sources kernel_set "
          f"{str(gkt.get('kernel_set_sha256'))[:16]}; kernel set equal across arms: {kernel_set_equal}"
          + (f"; diff {kernel_diff}" if kernels_identified and not kernel_set_equal else ""),
          f"- Commit {'PASS' if commit_pass else 'FAIL'} {fail_reasons if fail_reasons else ''} {problems if problems else ''}"]
    summary["table_md"] = "\n".join(md)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (out / "summary.md").write_text(summary["table_md"] + "\n")
    with (out / "runs.jsonl").open("w") as f:
        for r in ctl + trt:
            f.write(json.dumps(r, default=str) + "\n")
    verdict = dict(tv)
    verdict.update({"commit_pass": commit_pass, "paired": True, "problems": problems, "compilation": ct, "compilation_control": cc, "tokens_equal_across_arms": tokens_equal,
                    "first_fail_reason": (fail_reasons[0] if fail_reasons else (problems[0] if problems else None))})
    (out / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str) + "\n")
    print(summary["table_md"])
    sys.exit(0 if commit_pass else 3)
