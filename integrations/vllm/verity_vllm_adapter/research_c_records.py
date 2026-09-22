"""Machine-readable records and tables of research workstream C (RC, 2026-09-07).

Reads the pod's small result JSONs (fetched to a staging directory that mirrors
``/workspace/rc/results/B1``) and writes one record per table row under ``docs/data/vllm-poc/research-c/``
plus ``tables.md`` (the markdown tables pasted into ``docs/vllm-poc/research-c-replay.md``).

    python -m verity_vllm_adapter.research_c_records --results /tmp/rc-results --out ../docs/data/vllm-poc/research-c

Rows: experiment 1 (selective preparation) ``selective-q<k>.json`` and ``full-preparation-control.json``;
experiment 2 (boundary acquisition) ``acquisition-<family>.json``; experiment 3 (plan comparison)
``plan-<plan>.json``.  Counts, bytes and seconds are separate blocks in every record (memo section 6).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# --- proof-work estimate (labelled; never measured here) ------------------------------------------------------
# docs/cost-model.md SP1 constants (RTX 4090 campaign 2026-09-05/06, docs/data/campaign-2026-09-05/analysis/machine-fit.json):
SP1 = {
    "cycles_per_gate": (125, 238),           # gate evaluation in the guest
    "cycles_per_opened_position": 3296,      # per opened leaf
    "cycles_per_merkle_level": 2668,         # per path level
    "cycles_per_obligation": 5800,
    "cycles_per_batch": 50000,
    "seconds_per_million_cycles_core": 0.15513308438759368,  # machine-fit core mode, 4090
    "floor_seconds": 3.0577152424542504,
}
# gates per output word of the W3c synthesised slices (bundles/B1-b3eba618b1b5-20260907T1731Z-synth/README.md):
# o_proj+residual K=1536: 101 gates/word (1,536 units); down_proj+residual K=8960: 565; silu.mul incl. the two K=1536 dots: 199.
# generalised: a GEMM word of depth K costs K/16 tc_dot16 gates + ~5 casts/adds.
GATES_PER_WORD = {"gemm": lambda K: K // 16 + 5, "lm_head": lambda K: K // 16 + 5, "residual_add": lambda K: 1, "silu_mul": lambda K: 3}
# families without a gate-level lowering in the pinned gate set (REPORT section 6d: attention is not lowered; RMSNorm/RoPE/embedding/sampler have
# independent numerical models but no veritor gate DAG):
NO_GATE_MODEL = ("attention", "rmsnorm", "rope", "embedding", "sampler")


def load(p: Path) -> Any:
    return json.loads(p.read_text())


def du(p: Path) -> dict[str, int]:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            m = re.match(r"(\d+)\s+(.*)", line.strip())
            if m:
                out[Path(m.group(2)).name] = int(m.group(1))
    return out


def proof_estimate(check_families: dict[str, Any], results: list[dict[str, Any]] | None, witness: dict[str, Any]) -> dict[str, Any]:
    """SP1 proof work for the *checked* obligations under the campaign cost model: gates from the W3c per-word
    counts (GEMM-like families only), authentication from the openings; labelled an estimate, partial-family."""
    gates_low = gates_high = 0.0
    covered = 0
    uncovered: dict[str, int] = {}
    if results:
        for r in results:
            fam = r.get("op_type") or r.get("family")
            n_words = int(r.get("output_bytes", 0)) // 2
            K = int((r.get("K") or 0)) or (8960 if "down_proj" in str(r.get("op", "")) else 1536 if fam in ("gemm", "lm_head") else 0)
            if fam in GATES_PER_WORD and K > 0:
                g = GATES_PER_WORD[fam](K) * n_words
                gates_low += g * SP1["cycles_per_gate"][0]
                gates_high += g * SP1["cycles_per_gate"][1]
                covered += 1
            elif fam in GATES_PER_WORD and fam in ("residual_add", "silu_mul"):
                g = GATES_PER_WORD[fam](0) * n_words
                gates_low += g * SP1["cycles_per_gate"][0]
                gates_high += g * SP1["cycles_per_gate"][1]
                covered += 1
            else:
                uncovered[fam or "?"] = uncovered.get(fam or "?", 0) + 1
    n_obl = sum(v.get("attempted", 0) for fam in check_families.values() for v in fam.values()) if check_families else 0
    auth = witness.get("unique_openings", witness.get("openings", 0)) * SP1["cycles_per_opened_position"] + (witness.get("unique_path_bytes", witness.get("path_bytes", 0)) // 32) * SP1["cycles_per_merkle_level"]
    fixed = n_obl * SP1["cycles_per_obligation"] + SP1["cycles_per_batch"]
    cyc_low, cyc_high = gates_low + auth + fixed, gates_high + auth + fixed
    return {
        "label": "ESTIMATE from the campaign cost model (docs/cost-model.md SP1 constants; 4090 core-mode machine fit), NOT measured; gate counts only for "
                 "GEMM/lm_head/residual/silu words (W3c's synthesised-slice per-word counts); attention, RMSNorm, RoPE, embedding and sampler obligations have no "
                 "gate-level lowering and contribute only their authentication + per-obligation constants",
        "obligations": n_obl, "obligations_with_gate_model": covered, "obligations_without_gate_model": uncovered,
        "cycles_gates": [gates_low, gates_high], "cycles_authentication": auth, "cycles_fixed": fixed,
        "cycles_total": [cyc_low, cyc_high],
        "seconds_core_4090_fit": [SP1["floor_seconds"] + cyc_low / 1e6 * SP1["seconds_per_million_cycles_core"],
                                  SP1["floor_seconds"] + cyc_high / 1e6 * SP1["seconds_per_million_cycles_core"]],
        "constants": SP1,
    }


# --- experiment 1 ----------------------------------------------------------------------------------------------------


def workload_block(wl: dict[str, Any], prod_manifest: dict[str, Any] | None) -> dict[str, Any]:
    reqs = wl["requests"]
    prompt = [len(r.get("prompt_token_ids", [])) or int(r.get("prompt_len", 0)) for r in reqs]
    gen = [int(r.get("max_tokens", 0)) for r in reqs]
    return {
        "case": "B1", "model": wl.get("model"), "requests": len(reqs), "prompt_lengths": sorted(set(prompt)), "generated_lengths": sorted(set(gen)),
        "prompt_tokens_total": sum(prompt), "generated_tokens_total": sum(gen), "positions_total": sum(prompt) + sum(gen) - len(reqs),
        "production_batch": "all 16 requests in one engine pass (eager, custom ops, batch-invariant kernels; the recorded pass is the first pass of a fresh engine)",
        "profile_id": (prod_manifest or {}).get("profile_id"),
        "note": "positions_total = prompt tokens + generated tokens - 1 per request (the last generated token is never processed)",
    }


def selective_row(res: Path, tag: str, k: int, wl: dict[str, Any]) -> dict[str, Any]:
    run = res / f"selective-{tag}"
    ver = res / f"selective-{tag}.verify"
    p = load(run / "selective_prover.json")
    v = load(ver / "selective_verify.json")
    cmp_ = load(ver / "compare_selected.json") if (ver / "compare_selected.json").exists() else None
    d = du(ver / "du.txt")
    ph0, ph2, pb, pt = p["phase0"], p["phase2"], p["bytes"], p["timing"]
    vt, w, chk = v["timing"], v["witness"], v["check"]
    results = [json.loads(line) for line in (ver / "results.jsonl").read_text().splitlines()] if (ver / "results.jsonl").exists() else None
    prompt = p["phase2"]["replayed_tokens"]["prompt_tokens"]
    gen = p["phase2"]["replayed_tokens"]["generated_tokens"]
    row = {
        "schema": "verity/research-c/selective-preparation/v1", "workstream": "C", "experiment": 1,
        "row": f"selective q={k}/16 (fixed-size selector, k={k})", "kind": "selective", "k": k, "q": f"{k}/16",
        "workload": wl, "denominator": "per q: the production batch of 16 requests; costs are for the selected requests' replay + commitment and the verifier's sample",
        "identities": "capture-v2 canonical ids (TA1's typed B1 program had not landed)",
        "protocol": {
            "boundary_before_r_q": "production.jsonl (prompts, output tokens, timings), the 16 boundary trees (prescribed ingress + protected outputs), the weights tree "
                                   "(format v2, one leaf per parameter row), the description digest; nothing per-request materialised or committed",
            "T_0": v["challenge"]["phase_digests"]["statement"], "r_q": v["challenge"]["r_q"], "r_s": v["challenge"]["r_s"],
            "selector": v["challenge"]["selector"], "s": v["challenge"]["policy"]["s"], "coverage_size": 200, "challenge_mode": v["challenge"]["mode"],
            "rederived": v["challenge"]["rederived"]["ok"], "first_stage_ok": v["first_stage"]["ok"], "leaf_predicate_ok": v["leaf_predicate"]["ok"],
            "bindings_ok": v["bindings"]["ok"], "output_linkage_problems": sum(len(x["problems"]) for x in v["output_linkage"]),
            "architecture_observed_equals_registered": v.get("architecture_observed_equals_registered"),
        },
        "selected_requests": v["selected"],
        "counts": {
            "replayed_requests": k, "replayed_positions": p["phase2"]["replayed_tokens"]["positions_processed"], "replayed_prompt_tokens": prompt,
            "replayed_generated_tokens": gen, "engine_steps": ph2["engine_steps"],
            "logical_values_rows_committed": ph2["capture"]["leaves"], "logical_values_words_approx": pb["hashed_phase2_capture"] // 2,
            "replay_units": ph2["capture"]["replay_units"], "vus_population": v["population"]["vus"],
            "vus_certified": v["certified_sample"], "vus_coverage": v["coverage_sample"], "vus_adjacent": v["adjacent"], "vus_checked": v["sample_union"],
            "openings_per_use": w["openings"], "openings_unique": w.get("unique_openings"), "authentications": chk["authentication"]["valid"],
            "binding_checks": v["obligations"]["binding_checks"], "boundary_roots_bound": v["obligations"]["boundary_roots_bound"],
            "output_records_linked": v["obligations"]["output_records_linked"], "leaf_predicate_leaves": v["obligations"]["leaf_predicate_leaves"],
            "boundary_leaves_phase0": ph0["boundary_leaves"], "weights_leaves": ph0["weights"]["tree_leaves"],
            "statuses": chk["statuses"],
        },
        "bytes": {
            "hashed_phase0_weights_and_boundaries": pb["hashed_phase0"], "hashed_capture": pb["hashed_phase2_capture"], "rehashed_at_commit": pb["rehashed_at_commit_phase2"],
            "hashed_total": pb["hashed_phase0"] + pb["hashed_phase2_capture"] + pb["rehashed_at_commit_phase2"],
            "copied_device_to_host": pb["copied_device_to_host"], "retained_ru": pb["retained_ru_dir"], "retained_weights": pb["retained_weights_dir"], "retained_total": pb["retained_total"],
            "opened_value_bytes_per_use": w["value_bytes"], "path_bytes_per_use": w["path_bytes"], "witness_bytes_per_use": w["witness_bytes"],
            "witness_weights_value_bytes_per_use": w["weights_value_bytes"],
            "witness_bytes_unique": w.get("unique_witness_bytes"), "witness_weights_value_bytes_unique": w.get("unique_weights_value_bytes"),
            "witness_activation_value_bytes_unique": w.get("unique_activation_value_bytes"), "path_bytes_unique": w.get("unique_path_bytes"),
        },
        "seconds": {
            "prover": {k_: pt[k_] for k_ in pt},
            "verifier": {k_: vt[k_] for k_ in vt},
            "critical_path_excluding_engine_build": pt["phase0_total_s"] + vt["work_before_r_q_s"] + pt["replay_run_s"] + pt["phase2_boundaries_s"] + pt["phase2_commit_s"] + vt["interiors_and_r_s_s"] + vt["rederive_s"] + vt["check_s"] + vt["linkage_s"],
            "checker_seconds_sum": sum(c.get("checker_work", {}).get("seconds", 0.0) for fam in chk["families"].values() for c in fam.values()),
            "note": "prover phases: phase0 = boundaries + weights write + commit (before r_q); replay_run = engine steps of the selected requests with capture; "
                    "phase2 = boundaries rewrite + interior commit; verifier: bindings (model/architecture/output) before r_q, check = authenticated openings + relation "
                    "checks of the certified + coverage sample; witness_accounting is bookkeeping, not protocol work; wait_* is the other party's time",
        },
        "bit_exact_vs_full_replay_control": cmp_ and {k_: cmp_[k_] for k_ in ("bit_identical", "leaves_both", "leaves_only_selective", "compared_scalars", "mismatched_leaves", "mismatched_scalars", "per_family") if k_ in cmp_},
        "check_families": chk["families"],
        "proof_work_estimate": proof_estimate(chk["families"], results, w),
        "replay_batch": p.get("note_batch"), "engine_kwargs": p["engine_kwargs"], "effective_execution": p["effective_execution"],
        "adapter_revision": p["adapter_revision"], "gpu": "NVIDIA A100-SXM4-80GB (RunPod vllm-rc-a100)", "retained_du": d,
        "utc_phase0": ph0.get("utc"), "utc_phase2": ph2.get("utc"),
    }
    return row


def control_row(res: Path, wl: dict[str, Any]) -> dict[str, Any]:
    c = res / "control"
    ct = load(c / "control_timing.json")
    rp = load(c / "commit" / "replay.json") if (c / "commit" / "replay.json").exists() else load(c / "commit-v1" / "replay.json")
    cs = load(c / "commit" / "capture_summary.json") if (c / "commit" / "capture_summary.json").exists() else load(c / "commit-v1" / "capture_summary.json")
    v = load(c / "verify-security" / "verify.json")
    w = load(c / "witness.json") if (c / "witness.json").exists() else {}
    d = du(c / "du_commit.txt")
    conv = load(c / "commit" / "weights" / "conversion.json") if (c / "commit" / "weights" / "conversion.json").exists() else {}
    rt = rp["timing"]
    cap = rt.get("capture", {})
    drv = rt.get("driver", {})
    rep = v["report"]
    bytes_captured = int(cs.get("bytes_captured", 0))
    weights_bytes = int(rt.get("weights", {}).get("bytes", 0) or conv.get("bytes", 0) or 3120983488)
    results = rep.get("results")
    # RC2: verify.json as verify.py actually writes it (the control verify first completed on 2026-09-08 after the
    # 94734a9 fix): `bindings` is {"ok", "semantic_profile_sha256", model/architecture/output: {...}}; the certified /
    # coverage sizes are `sampling_plans[...]["vus"]` (`sample_size` is their union incl. the adjacent chains).
    vb = v.get("bindings") or {}
    bindings_ok = vb.get("ok") if isinstance(vb, dict) and "ok" in vb else (
        all((b.get("ok", True) if isinstance(b, dict) else bool(b)) for b in vb.values()) if isinstance(vb, dict) else None)
    plans = v.get("sampling_plans") or {}
    certified_n = (plans.get("certified") or {}).get("vus", v.get("sample_size"))
    coverage_n = (plans.get("coverage") or {}).get("vus")
    row = {
        "schema": "verity/research-c/selective-preparation/v1", "workstream": "C", "experiment": 1,
        "row": "FULL PREPARATION (control): every request replayed, captured and committed; verifier at q=1", "kind": "full_preparation_control", "k": 16, "q": "1",
        "workload": wl, "denominator": "the production batch of 16 requests, all replayed in one replay batch (the reference pipeline run_case.sh ref1/ref2/commit/compare)",
        "identities": "capture-v2 canonical ids",
        "protocol": {
            "note": "WB's staged challenge over the complete bundle (T_0..T_3; q=1 selects every RU); this is the reference verify.py, not the selective protocol",
            "challenge_digest": v.get("challenge_digest"), "s": "1/4096", "coverage_size": 200, "negatives": "disabled (--no-negatives) for comparability",
            "bindings_ok": bindings_ok, "verdict": v.get("verdict_label"), "first_blocker": (v.get("table1") or {}).get("first_blocker"),
            "phase_digests": (v.get("challenge") or {}).get("phase_digests"), "rederived": (v.get("challenge") or {}).get("transcript_rederived"),
            "first_stage_ok": (v.get("first_stage") or {}).get("ok"), "required_coverage": v.get("required_coverage"),
            "verifier_adapter_revision": (v.get("table1") or {}).get("adapter_revision"),
        },
        "selected_requests": "all 16",
        "counts": {
            "replayed_requests": 16, "replayed_positions": int(drv.get("prompt_tokens", 0)) + int(drv.get("generated_tokens", 0)) - 16 if drv else None,
            "replayed_prompt_tokens": drv.get("prompt_tokens"), "replayed_generated_tokens": drv.get("generated_tokens"), "engine_steps": drv.get("engine_steps"),
            "logical_values_rows_committed": cs.get("leaves"), "logical_values_words_approx": bytes_captured // 2, "replay_units": cs.get("replay_units"),
            "vus_population": cs.get("leaves"), "vus_certified": certified_n, "vus_coverage": coverage_n,
            "vus_adjacent": len(v.get("adjacent_vus") or []), "vus_checked": len(results or []),
            "openings_per_use": w.get("openings"), "openings_unique": w.get("unique_openings"), "authentications": (rep.get("authentication") or {}).get("valid"),
            "binding_checks": len([k for k in ("model", "architecture", "output") if isinstance(vb.get(k), dict)]) or 3,
            "boundary_roots_bound": ((vb.get("output") or {}).get("boundary_roots_matched") if isinstance(vb.get("output"), dict) else None),
            "output_records_linked": sum(int(x.get("checked", 0)) for x in (v.get("output_linkage") or [])) or None,
            "leaf_predicate_leaves": (v.get("capture_predicates") or {}).get("present_leaves"), "statuses": rep.get("statuses"),
        },
        "bytes": {
            "hashed_phase0_weights_and_boundaries": None, "hashed_capture": bytes_captured, "rehashed_at_commit": bytes_captured,
            "hashed_total": bytes_captured * 2 + weights_bytes, "hashed_note": "capture hashes every leaf once at capture and once at commit (as the selective prover does); weights hashed once (v1) and re-hashed per row at the v2 rewrite (counted once here)",
            "copied_device_to_host": bytes_captured + weights_bytes, "retained_ru": d.get("ru"), "retained_weights": d.get("weights"), "retained_total": d.get("commit"),
            "opened_value_bytes_per_use": w.get("value_bytes"), "path_bytes_per_use": w.get("path_bytes"), "witness_bytes_per_use": w.get("witness_bytes"),
            "witness_weights_value_bytes_per_use": w.get("weights_value_bytes"),
            "witness_bytes_unique": w.get("unique_witness_bytes"), "witness_weights_value_bytes_unique": w.get("unique_weights_value_bytes"),
            "witness_activation_value_bytes_unique": w.get("unique_activation_value_bytes"), "path_bytes_unique": w.get("unique_path_bytes"),
        },
        "seconds": {
            "prover": {
                "engine_build_s": rt.get("engine_build_s"), "replay_run_s": rt.get("run_s"), "capture": cap, "weights_write_s": rt.get("weights", {}).get("seconds"),
                "commit_s": (rp.get("commitments") or {}).get("seconds"), "weights_v2_rewrite_wall_s": ct.get("weights_v2_wall_s"),
                "stage_walls": {k_: ct[k_] for k_ in ct if k_.endswith("_wall_s")},
            },
            "verifier": v.get("timing"),
            "critical_path_excluding_engine_build": (rt.get("run_s") or 0) + ((rp.get("commitments") or {}).get("seconds") or 0) + (rt.get("weights", {}).get("seconds") or 0)
                                                    + (ct.get("weights_v2_wall_s") or 0) + ((v.get("timing") or {}).get("total_s") or 0),
            "checker_seconds_sum": sum(c.get("checker_work", {}).get("seconds", 0.0) for fam in (v.get("families") or {}).values() for c in fam.values()),
            "note": "the reference pipeline's commit-mode replay (one process: engine + capture + weights write + trees), then the v2 weights rewrite, then verify.py "
                    "(bindings + challenge + check); ref1/ref2 (two reference replays for noninterference) are the pipeline's own controls and not preparation cost",
        },
        "bit_exact_vs_full_replay_control": "is the control",
        "noninterference_compare": load(c / "compare.json") if (c / "compare.json").exists() else None,
        "check_families": v.get("families"),
        "proof_work_estimate": proof_estimate(v.get("families") or {}, results, w),
        "adapter_revision": rp.get("adapter_revision"), "gpu": "NVIDIA A100-SXM4-80GB (RunPod vllm-rc-a100)", "retained_du": d,
    }
    return row


# --- tables --------------------------------------------------------------------------------------------------------------


def fmt_b(x: Any) -> str:
    if x is None:
        return "—"
    x = float(x)
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if x >= div:
            return f"{x / div:.2f} {unit}"
    return f"{x:.0f} B"


def fmt_s(x: Any) -> str:
    return "—" if x is None else f"{float(x):.1f}"


def fmt_n(x: Any) -> str:
    return "—" if x is None else f"{int(x):,}"


def table_selective(rows: list[dict[str, Any]]) -> str:
    hdr = ["row", "k", "replayed positions (prompt+gen)", "rows committed", "≈BF16 words", "RUs", "VUs checked (cert+cov)", "openings (unique)", "bindings / linked outputs / leaf-predicate leaves"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c = r["counts"]
        lines.append("| " + " | ".join([
            r["row"].split(":")[0], str(c["replayed_requests"]),
            f"{fmt_n(c['replayed_positions'])} ({fmt_n(c['replayed_prompt_tokens'])}+{fmt_n(c['replayed_generated_tokens'])})",
            fmt_n(c["logical_values_rows_committed"]), fmt_n(c["logical_values_words_approx"]), fmt_n(c["replay_units"]),
            f"{fmt_n(c['vus_checked'])} ({fmt_n(c['vus_certified'])}+{fmt_n(c['vus_coverage'])})",
            f"{fmt_n(c['openings_per_use'])} ({fmt_n(c.get('openings_unique'))})",
            f"{c.get('binding_checks')} / {fmt_n(c.get('output_records_linked'))} / {fmt_n(c.get('leaf_predicate_leaves'))}",
        ]) + " |")
    out = ["**Counts**", "", *lines, ""]
    hdr = ["row", "hashed (capture + commit + weights)", "copied D→H", "retained (RU dirs)", "retained (weights)", "witness per use", "witness unique", "of which weights (unique)", "activations (unique)", "paths (unique)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        b = r["bytes"]
        lines.append("| " + " | ".join([
            r["row"].split(":")[0], fmt_b(b["hashed_total"]), fmt_b(b["copied_device_to_host"]), fmt_b(b["retained_ru"]), fmt_b(b["retained_weights"]),
            fmt_b(b["witness_bytes_per_use"]), fmt_b(b["witness_bytes_unique"]), fmt_b(b["witness_weights_value_bytes_unique"]), fmt_b(b["witness_activation_value_bytes_unique"]), fmt_b(b["path_bytes_unique"]),
        ]) + " |")
    out += ["**Bytes**", "", *lines, ""]
    hdr = ["row", "phase 0 (pre-r_q commit)", "bindings (verifier, pre-r_q)", "replay run", "phase 2 commit", "r_s + rederive", "check (openings + relations)", "of which checker seconds", "critical path (excl. engine build)", "proof work (ESTIMATE, s, 4090 core fit)"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        s = r["seconds"]
        pe = r["proof_work_estimate"]["seconds_core_4090_fit"]
        if r["kind"] == "selective":
            p, v = s["prover"], s["verifier"]
            lines.append("| " + " | ".join([
                r["row"].split(":")[0], fmt_s(p["phase0_total_s"]), fmt_s(v["bindings_s"]), fmt_s(p["replay_run_s"]), fmt_s(p["phase2_commit_s"] + p["phase2_boundaries_s"]),
                fmt_s(v["interiors_and_r_s_s"] + v["rederive_s"]), fmt_s(v["check_s"]), fmt_s(s["checker_seconds_sum"]), fmt_s(s["critical_path_excluding_engine_build"]),
                f"{pe[0]:.0f}–{pe[1]:.0f}",
            ]) + " |")
        else:
            p, v = s["prover"], s["verifier"] or {}
            lines.append("| " + " | ".join([
                r["row"].split(":")[0], f"— (weights write {fmt_s(p.get('weights_write_s'))} + v2 rewrite {fmt_s(p.get('weights_v2_rewrite_wall_s'))})", fmt_s(v.get("bindings_s")),
                fmt_s(p.get("replay_run_s")), fmt_s(p.get("commit_s")), fmt_s(v.get("challenge_s") or v.get("sample_s")), fmt_s(v.get("check_s")), fmt_s(s["checker_seconds_sum"]),
                fmt_s(s["critical_path_excluding_engine_build"]), f"{pe[0]:.0f}–{pe[1]:.0f}",
            ]) + " |")
    out += ["**Seconds** (wall, one A100 pod; the prover and verifier are two processes, waits excluded)", "", *lines, ""]
    hdr = ["row", "bit-identical vs control", "leaves compared", "scalars compared", "mismatched"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        c = r["bit_exact_vs_full_replay_control"]
        if isinstance(c, dict):
            lines.append(f"| {r['row'].split(':')[0]} | {c['bit_identical']} | {fmt_n(c['leaves_both'])} (only in selective: {fmt_n(c['leaves_only_selective'])}) | {fmt_n(c['compared_scalars'])} | {fmt_n(c['mismatched_leaves'])} leaves / {fmt_n(c['mismatched_scalars'])} scalars |")
        else:
            lines.append(f"| {r['row'].split(':')[0]} | {c} | | | |")
    out += ["**Batch invariance** (selective replay batch vs the full-replay control, every committed leaf of the selected requests)", "", *lines, ""]
    return "\n".join(out)


# --- experiment 2 -------------------------------------------------------------------------------------------------------------

COMPILED_NOTE = {
    # what compiled execution (W1cF's compiled row, W1d's compiled capture) can / cannot supply per family (REPORT section 2, GUIDANCE section 11-13)
    "embedding": "read: yes (module boundary; the gather is not fused)", "rmsnorm_triton": "read: NO in compiled mode (Inductor fuses the norm into neighbours; W1cF's compiled row exposes only the fused kernel's output = the GEMM input); recompute (b): from the residual row",
    "rmsnorm_fused": "read: NO (fused_add_rms_norm is one kernel already in eager; compiled keeps it or refuses); (b)/(c) from x + residual", "residual": "read: kernel output port 1 (eager) / interior of the fused kernel (compiled: NO); (c) IEEE add from bound neighbours",
    "gemm_qkv": "read: yes (GEMM outputs are kernel boundaries in compiled mode too; W1cF compiled row)", "gemm_o": "read: yes", "gemm_gate_up": "read: yes", "gemm_down": "read: yes", "lm_head": "read: yes",
    "bias_add": "read: eager aten.add; compiled: fused into the GEMM epilogue or the RoPE prologue -> NO direct read, (c) IEEE add", "rope_q": "read: eager _C.rotary_embedding; compiled: custom op kept (custom_ops all) -> yes", "rope_k": "as rope_q",
    "attention": "read: yes (unified_attention_with_output is a custom op in both modes)", "silu_mul": "read: eager _C.silu_and_mul; compiled: fused with the GEMM epilogue in Inductor's default (NO) unless custom_ops all keeps it (yes)",
    "sampler": "read: yes (boundary leaf; the sampler is outside the compiled graph)",
}


def acquisition_rows(res: Path) -> tuple[list[dict[str, Any]], str]:
    acq = res / "acq"
    sr = load(acq / "acquisition_same_runtime.json") if (acq / "acquisition_same_runtime.json").exists() else None
    ind = load(acq / "acquisition_independent.json") if (acq / "acquisition_independent.json").exists() else None
    cmp_ = None
    for cand in sorted(res.glob("selective-k*.verify/compare_selected.json")):
        cmp_ = load(cand)
        if "k16" in str(cand):
            break
    fams = (sr or ind or {}).get("families", {})
    rows = []
    hdr = ["family (canonical interface)", "population in bundle (prefill / decode rows, bytes)", "(a) direct read: leaf bytes; cross-batch bit-exact (compare_selected)", "(b) recompute, same-runtime relation: status n, wall median s / checker median s, input bytes", "(c) alternative construction: relation, status n, wall / checker s", "compiled execution can supply"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    op_type_of = {"embedding": "embedding/p=0", "rmsnorm_triton": "rmsnorm/p=0", "rmsnorm_fused": "rmsnorm/p=0", "residual": "rmsnorm/p=1", "gemm_qkv": "gemm/p=0", "gemm_o": "gemm/p=0", "gemm_gate_up": "gemm/p=0",
                  "gemm_down": "gemm/p=0", "lm_head": "lm_head/p=0", "bias_add": "residual_add/p=0", "rope_q": "rope/p=0", "rope_k": "rope/p=1", "attention": "attention/p=0", "silu_mul": "silu_mul/p=0", "sampler": "sampler/p=0"}
    for key, f in fams.items():
        a = (sr or {}).get("families", {}).get(key) or {}
        c = (ind or {}).get("families", {}).get(key) or {}
        pop = f.get("population") or {}
        fam_cmp = ((cmp_ or {}).get("per_family") or {}).get(op_type_of.get(key, ""), None)
        row = {
            "schema": "verity/research-c/acquisition/v1", "workstream": "C", "experiment": 2, "family": key, "label": f.get("label"), "identities": "capture-v2 canonical ids",
            "population": pop,
            "a_direct_read": {"leaf_bytes": f.get("output_bytes"), "cost": "one device->host copy + one hash at capture and one at commit per row (eager capture)",
                              "cross_batch_bit_exact": fam_cmp, "note": "compare_selected on the largest selective run available; the op_type/port key groups rmsnorm_triton with rmsnorm_fused"},
            "b_same_runtime": a and {k_: a.get(k_) for k_ in ("n", "statuses", "relation_run", "relation_class", "detail_example", "warmup_first_vu_seconds", "checker_seconds_median", "wall_seconds_median", "input_bytes_median", "inputs_opened_median", "output_bytes")},
            "c_independent": c and {k_: c.get(k_) for k_ in ("n", "statuses", "relation_run", "relation_class", "detail_example", "warmup_first_vu_seconds", "checker_seconds_median", "wall_seconds_median", "input_bytes_median", "inputs_opened_median", "diagnostic_ungated")},
            "compiled_execution": COMPILED_NOTE.get(key),
            "device_b": (sr or {}).get("device"), "device_c": (ind or {}).get("device"), "sample": (sr or ind or {}).get("sample"),
        }
        rows.append(row)

        def st(x: dict[str, Any] | None) -> str:
            if not x or not x.get("n"):
                return "—"
            s_ = ",".join(f"{k_} {v_}" for k_, v_ in (x.get("statuses") or {}).items())
            cs = x.get("checker_seconds_median")
            return f"{x.get('relation_run')}: {s_}; wall {x['wall_seconds_median']:.3f} / checker {cs:.4f} s; in {fmt_b(x.get('input_bytes_median'))}" if cs is not None else f"{x.get('relation_run')}: {s_}; wall {x['wall_seconds_median']:.3f} s"

        diag = (c or {}).get("diagnostic_ungated")
        cstr = st(c) + (f"; ungated diagnostic {diag['equal']}/{diag['n']} equal" if diag else "")
        cmpstr = f"{fam_cmp['equal']}/{fam_cmp['leaves']} leaves" if fam_cmp else "—"
        lines.append("| " + " | ".join([
            f"{key}: {f.get('label')}", f"{fmt_n(pop.get('prefill'))} / {fmt_n(pop.get('decode'))}, {fmt_b(pop.get('bytes'))}",
            f"{fmt_b(f.get('output_bytes'))}; {cmpstr}", st(a), cstr, COMPILED_NOTE.get(key, ""),
        ]) + " |")
    return rows, "\n".join(lines)


# --- experiment 3 --------------------------------------------------------------------------------------------------------


def plan_rows(res: Path) -> tuple[list[dict[str, Any]], str]:
    pl = res / "plan"
    g = load(pl / "plan_compare_gpu.json") if (pl / "plan_compare_gpu.json").exists() else None
    c = load(pl / "plan_compare_cpu.json") if (pl / "plan_compare_cpu.json").exists() else None
    if not (g or c):
        return [], "(not run)"
    sl = (g or c)["slice"]
    rows = []
    hdr = ["plan", "logical values (circuit rows / scalars; committed rows)", "committed bytes (values + tree nodes + roots)", "witness bytes to check the slice (values + paths; unique)", "checked / proved obligations", "Bound", "reconstruction work", "authentication work (path hashes / leaf hashes)", "wall s (checker s)", "exact"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    bound = "vacuous (out_bits = 0: every protected output is a check output)"
    if g:
        for name in ("fine", "composite"):
            p = g[name]
            cm = p["committed"]
            w = p["witness"]
            rows.append({"schema": "verity/research-c/plan-compare/v1", "workstream": "C", "experiment": 3, "plan": name, "slice": sl, "identities": "capture-v2 canonical ids",
                         "counts": {"circuit_rows": sl["circuit_rows"], "circuit_scalars": sl["circuit_scalars"], "committed_rows": cm["rows"], "vus": p["vus"], "checked": p["checked"], "proved": 0,
                                    "openings": w["openings"], "path_hashes": w["path_hashes"], "leaf_hashes": cm["leaf_hashes"], "rows_recomputed_uncommitted": p["reconstruction"]["rows_recomputed_uncommitted"]},
                         "bytes": {"committed_values": cm["value_bytes"], "tree_nodes": cm["tree_node_bytes"], "roots": cm["root_bytes"], "committed_total": cm["committed_bytes"],
                                   "witness_values": w["value_bytes"], "witness_paths": w["path_bytes"], "witness_total": w["witness_bytes"], "witness_weights_values": w["weights_value_bytes"], "witness_activation_values": w["activation_value_bytes"]},
                         "seconds": {"wall": p["wall_seconds"], "checker_sum": p["checker_seconds_total"]}, "bound": sl["bound"], "statuses": p["statuses"], "all_exact": p["all_exact"],
                         "reconstruction": p["reconstruction"], "device": g.get("device"), "results": p.get("results")})
            lines.append("| " + " | ".join([
                name, f"{fmt_n(sl['circuit_rows'])} / {fmt_n(sl['circuit_scalars'])}; committed {fmt_n(cm['rows'])}", f"{fmt_b(cm['committed_bytes'])} ({fmt_b(cm['value_bytes'])} + {fmt_b(cm['tree_node_bytes'])} + {cm['root_bytes']} B)",
                f"{fmt_b(w['witness_bytes'])} ({fmt_b(w['value_bytes'])} + {fmt_b(w['path_bytes'])}; weights {fmt_b(w['weights_value_bytes'])})", f"{p['checked']} checked / 0 proved", bound,
                f"{p['reconstruction']['rows_recomputed_uncommitted']} rows recomputed, not committed", f"{fmt_n(w['path_hashes'])} / {fmt_n(cm['leaf_hashes'])}", f"{p['wall_seconds']:.1f} ({p['checker_seconds_total']:.2f})", str(p["all_exact"]),
            ]) + " |")
    if c:
        rg = c["regrouped"]
        for gname, gp in rg["groupings"].items():
            rows.append({"schema": "verity/research-c/plan-compare/v1", "workstream": "C", "experiment": 3, "plan": f"regrouped: {gname}", "slice": sl, "identities": "capture-v2 canonical ids", "analytical": True,
                         "counts": {"circuit_rows": sl["circuit_rows"], "circuit_scalars": sl["circuit_scalars"], "committed_rows": rg["vus"], "vus": rg["vus"], "checked": "as plan fine (analytical)", "proved": 0,
                                    "path_hashes": gp["path_hashes_slice_all_vus"], "leaf_hashes": gp["leaf_hashes"], "trees": gp["trees"], "roots": gp["roots"], "path_len": gp["path_len"]},
                         "bytes": {"committed_values": rg["leaf_value_bytes"], "tree_nodes": gp["tree_node_bytes"], "roots": gp["root_bytes"], "committed_total": gp["committed_bytes"], "witness_paths": gp["witness_path_bytes_slice_all_vus"]},
                         "bound": sl["bound"], "note": rg["note"]})
            lines.append("| " + " | ".join([
                f"regrouped: {gname}", f"{fmt_n(sl['circuit_rows'])} / {fmt_n(sl['circuit_scalars'])}; committed {fmt_n(rg['vus'])}", f"{fmt_b(gp['committed_bytes'])} ({fmt_b(rg['leaf_value_bytes'])} + {fmt_b(gp['tree_node_bytes'])} + {gp['root_bytes']} B; {gp['trees']} tree(s))",
                f"values as fine + paths {fmt_b(gp['witness_path_bytes_slice_all_vus'])} ({gp['path_len']} levels/VU)", "as fine (analytical)", bound, "as fine", f"{fmt_n(gp['path_hashes_slice_all_vus'])} / {fmt_n(gp['leaf_hashes'])}", "—", "—",
            ]) + " |")
        co = c["coordinate"]
        ex = co.get("extrapolated_slice_all_coordinates") or {}
        rows.append({"schema": "verity/research-c/plan-compare/v1", "workstream": "C", "experiment": 3, "plan": "coordinate (W3c)", "slice": sl, "identities": "capture-v2 canonical ids",
                     "counts": {"circuit_rows": sl["circuit_rows"], "circuit_scalars": sl["circuit_scalars"], "committed_rows": sl["committed_rows"]["coordinate"], "vus_gemm_coordinates": co["vus"], "gemm_rows": co.get("gemm_rows"), "checked": co["checked"], "proved": 0,
                                "path_hashes_extrapolated": ex.get("path_hashes")},
                     "bytes": {"committed_values": sl["committed_value_bytes"]["coordinate"], "witness_bytes_extrapolated_all_coordinates": ex.get("witness_bytes")},
                     "seconds": {"checker_seconds_extrapolated_all_coordinates": ex.get("checker_seconds")}, "families": co["families"], "statuses": co["statuses"], "all_exact": co["all_exact"], "bound": sl["bound"], "note": co["note"], "label": ex.get("label"), "results": co.get("results")})
        lines.append("| " + " | ".join([
            "coordinate (W3c; GEMM/lm_head rows only, other rows as fine)", f"{fmt_n(sl['circuit_rows'])} / {fmt_n(sl['circuit_scalars'])}; committed {fmt_n(sl['committed_rows']['coordinate'])} rows; {fmt_n(co['vus'])} coordinate VUs",
            "as fine (row commitment unchanged; weights per row block)", f"EXTRAPOLATED {fmt_b(ex.get('witness_bytes'))} for all {fmt_n(co['vus'])} coordinates (sample medians x counts)", f"{co['checked']} checked (sample) / 0 proved", bound,
            "none (one word predicted per VU)", f"EXTRAPOLATED {fmt_n(ex.get('path_hashes'))} / as fine", f"EXTRAPOLATED checker {fmt_s(ex.get('checker_seconds'))}", str(co["all_exact"]),
        ]) + " |")
    return rows, "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wl_doc = load(res / "workload.json")
    prod_manifest = load(res / "control" / "production" / "manifest.json") if (res / "control" / "production" / "manifest.json").exists() else None
    wl = workload_block(wl_doc, prod_manifest)
    rows1 = []
    for k in (1, 2, 4, 8, 16):
        for tag in (f"k{k}-rerun", f"k{k}"):
            if (res / f"selective-{tag}.verify" / "selective_verify.json").exists():
                r = selective_row(res, tag, k, wl)
                r["tag"] = tag
                rows1.append(r)
                (out / f"selective-q{k}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
                break
    if (res / "control" / "verify-security" / "verify.json").exists():
        r = control_row(res, wl)
        rows1.append(r)
        (out / "full-preparation-control.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    t1 = table_selective(rows1) if rows1 else "(not run)"
    rows2, t2 = acquisition_rows(res)
    for r in rows2:
        (out / f"acquisition-{r['family']}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    rows3, t3 = plan_rows(res)
    for r in rows3:
        name = re.sub(r"[^a-z0-9]+", "-", r["plan"].lower()).strip("-")
        (out / f"plan-{name}.json").write_text(json.dumps(r, indent=1, default=str) + "\n")
    (out / "tables.md").write_text("## Experiment 1 — selective preparation\n\n" + t1 + "\n\n## Experiment 2 — boundary acquisition\n\n" + t2 + "\n\n## Experiment 3 — plan comparison\n\n" + t3 + "\n")
    print(f"wrote {len(rows1)} + {len(rows2)} + {len(rows3)} records to {out}")


if __name__ == "__main__":
    main()
