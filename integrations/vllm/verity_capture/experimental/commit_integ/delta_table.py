"""Build `delta_table.md` from commit_delta.py result directories (VALUE-FIRST §5 form).

  §1 per execution path: integrated ΔT = T(same workload + commitments) − T(same workload) with 95 % CI over interleaved pairs, the
     instrumented arm's component spans (acquisition / movement / hashing / root finalisation / openings / serving-thread wait), root-ready
     delay, steady state, coverage verdict across manifest rows (residual), value- and binding-correctness status, hardware/commit/command.
  §2 per value-family row (manifest v2 rows) for the headline path: bytes/request, tensors/step, acquisition + extra reads/writes,
     commitment work, temp/persistent memory, opening cost — attributed from the measured totals (labels: measured / calculated).
  §3 per-row coverage table of the headline path and the residual per path.

Every number is labelled: **measured** (from runs.jsonl / summary.json), **calculated** (attribution of a measured total by bytes or
tensor counts, or manifest logical bytes), or **not measured**.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from verity_capture.experimental.commit_integ import coverage_check as cc


def load_dir(d: Path) -> dict:
    s = json.loads((d / "summary.json").read_text())
    v = json.loads((d / "versions.json").read_text()) if (d / "versions.json").exists() else s.get("versions", {})
    decls = sorted(d.glob("declaration_*.json"))
    decl = json.loads(decls[0].read_text()) if decls else s.get("declaration")
    runs = [json.loads(l) for l in (d / "runs.jsonl").read_text().splitlines() if l.strip()] if (d / "runs.jsonl").exists() else []
    vc = json.loads((d / "value_check.json").read_text()) if (d / "value_check.json").exists() else None
    return {"dir": d, "summary": s, "versions": v, "declaration": decl, "runs": runs, "value_check": vc}


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:+.1f} %"


def ms(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x * 1e3:.{nd}f}"


# tensors per engine step per manifest row for the Llama architecture (vLLM d9105ea80 module outputs): calculated from n_layers
def tensors_per_step(row_id: str, n_layers: int) -> int | None:
    return {"R2a": 4 * n_layers, "R2b": 1, "R2c": 1, "R2d": n_layers - 1 + 1, "R2e": n_layers - 1 + 1, "R2f": n_layers, "R2g": 2 * n_layers,
            "R2h": n_layers, "R2i": 1, "R1b": 1, "R5": n_layers, "R6": n_layers, "R7": n_layers, "R8": n_layers}.get(row_id)

HIDDEN_ROWS = ("R5", "R6", "R7", "R8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="result DIR or DIR=label")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--tests", default=None, help="text file with the pytest summary line(s) run on the pod")
    ap.add_argument("--headline", default=None, help="label of the path used for the per-row table (default: first)")
    ap.add_argument("--hooks-only", default=None, help="label of the hooks_only ablation (per-tensor hook cost)")
    ap.add_argument("--n-layers", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Δ_commit per execution path and per value-family row")
    a = ap.parse_args()

    v2 = cc.is_v2(a.manifest)
    doc = json.loads(Path(a.manifest).read_text()) if v2 else None
    man, _meta = (cc._records_from_doc(doc) if v2 else cc.load_manifest(Path(a.manifest)))
    paths = []
    for spec in a.dirs:
        d, _, label = spec.partition("=")
        r = load_dir(Path(d))
        s, v = r["summary"], r["versions"]
        label = label or s.get("committer") or Path(d).name
        decls = [r["declaration"]] if r["declaration"] else []
        cov = cc.check_v2(doc, decls) if v2 else None
        covc = cc.check(man, decls)
        ctrl = [x for x in r["runs"] if x["arm"] == "control"]
        inst = [x for x in r["runs"] if x["arm"] == "instrumented"]
        c0 = ctrl[0] if ctrl else {}
        gpu = (v.get("gpu") or {}).get("name") if isinstance(v.get("gpu"), dict) else v.get("gpu", "")
        sp = s.get("spans_mean_s") or {}
        rr = s.get("root_ready_delay_s") or []
        wall_c = statistics.fmean(x["wall_s"] for x in ctrl) if ctrl else None
        wall_i = statistics.fmean(x["wall_s"] for x in inst) if inst else None
        commit0 = inst[0].get("commit", {}) if inst else {}
        paths.append({"label": label, "s": s, "v": v, "cov": cov, "covc": covc, "runs": r["runs"], "ctrl": ctrl, "inst": inst, "c0": c0, "gpu": gpu, "sp": sp,
                      "rr": statistics.fmean(rr) if rr else None, "wall_c": wall_c, "wall_i": wall_i, "commit0": commit0, "vc": r["value_check"],
                      "dir": str(r["dir"]), "n_steps": c0.get("engine_steps")})

    tests_txt = Path(a.tests).read_text().strip() if a.tests and Path(a.tests).exists() else "not run"
    L = [f"# {a.title}", "",
         f"manifest: `{a.manifest}`" + (f" ({doc['schema']} {doc['version']}; {doc['totals']['required']['rows']} required rows, "
                                         f"{doc['totals']['required']['logical_bytes']:,.0f} B required per request incl. weights)" if v2 else ""),
         f"binding-correctness tests (pytest, `verity_capture/experimental/commit_integ/tests`): {tests_txt}", "",
         "ΔT = T(same workload + commitments) − T(same workload): interleaved pairs on the SAME batch-invariant serving workload (vLLM d9105ea80, eager, bf16, "
         "`VLLM_BATCH_INVARIANT=1`), control = Verity disabled (no hooks, no observer). **GPU: NVIDIA L40S (sm89 Ada; the admitted RTX 4090 profile's "
         "architecture — no working 4090 was obtainable) — absolute times are L40S numbers; the Δ ratio on matched hardware is the quantity.** "
         "All Δ, spans, delays: **measured**; coverage bytes: **calculated** from the manifest.", "",
         "## 1. Per execution path", "",
         "| execution path | model / workload | pairs | T control ms | T + commitments ms | **ΔT wall (95 % CI)** | Δ decode step (95 % CI) | Δ prefill | acquisition ms | movement ms | hashing ms | root fin. ms | openings ms | serving wait ms | root-ready delay ms | steady state (2nd/1st half hashing) | coverage (rows secure / required; residual B) | value-correctness | openings verified | tokens identical |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---|"]
    for p in paths:
        s, cov, sp = p["s"], p["cov"], p["sp"]
        ci = s.get("delta_wall_ci95") or (None, None)
        cid = s.get("delta_decode_ci95") or (None, None)
        name = p["label"] + ("" if (cov or {}).get("verdict") == "complete" else " **(coverage PARTIAL)**")
        covtxt = (f"{cov['rows_secure']}/{cov['required_rows']} rows; residual {cov['residual_bytes']:,.0f} B ({(cov['residual_share_of_required_bytes'] or 0) * 100:.1f} %); "
                  f"not secure: {', '.join(cov['rows_not_secure'])}") if cov else f"{p['covc']['verdict']}"
        vc = p["vc"]
        vctxt = (f"**{vc['verdict']}** {vc['tap_equal']}/{vc['tensors']} tap-equal, re-exec {vc['reexec_equal']}/{vc['reexec_tensors']}" if vc else "not run")
        op = p["commit0"].get("openings", {}) or {}
        open_s = op.get("generate_s") if op.get("generate_s") is not None else sp.get("opening_generation")
        L.append(f"| {name} | {p['c0'].get('model', '')} / `{Path(s.get('workload', '')).name}` | {s.get('pairs')} | {ms(p['wall_c'])} | {ms(p['wall_i'])} | "
                 f"**{pct(s.get('delta_wall_mean'))}** [{pct(ci[0])}, {pct(ci[1])}] | {pct(statistics.fmean(s['delta_decode']) if s.get('delta_decode') else None)} "
                 f"[{pct(cid[0])}, {pct(cid[1])}] | {pct(statistics.fmean(s['delta_prefill']) if s.get('delta_prefill') else None)} | {ms(sp.get('acquisition'))} | "
                 f"{ms(sp.get('movement'))} | {ms(sp.get('hashing'))} | {ms(sp.get('root_finalisation'))} | {ms(open_s, 2)} ({op.get('n', 0)} openings; verify {ms(op.get('verify_s'))}) | {ms(sp.get('wait'))} | "
                 f"{ms(p['rr'])} | {pct(s.get('steady_growth'))} (backlog max {s.get('backlog_max')}) | {covtxt} | {vctxt} | {op.get('verified', 'n/a')}/{op.get('n', 'n/a')} | {s.get('tokens_equal')} |")
    L += ["", "Spans are the instrumented arm's separately timed components per run (sum over steps; hashing/movement run on a worker thread overlapped with "
          "serving, so they need not add to ΔT; the serving-thread cost that DOES add is acquisition + the D2D copy issue inside the hooks + GIL contention). "
          "`hooks_only` = the 365 forward hooks fire and return (nothing copied or hashed): the floor of any hook-based path.", ""]

    # ---- §2 per value-family row for the headline path
    head = next((p for p in paths if p["label"] == a.headline), paths[0])
    hooks = next((p for p in paths if p["label"] == a.hooks_only), None)
    if v2 and head["cov"]:
        s, sp, cov = head["s"], head["sp"], head["cov"]
        n_steps = head["n_steps"] or 16
        bound = head["commit0"].get("bytes_bound") or 1
        tens_total = head["commit0"].get("tensors") or 1
        d_wall_ms = (head["wall_i"] - head["wall_c"]) * 1e3 if head["wall_i"] and head["wall_c"] else None
        hook_ms_per_tensor = None
        if hooks and hooks["wall_i"] and hooks["wall_c"]:
            hook_ms_per_tensor = (hooks["wall_i"] - hooks["wall_c"]) * 1e3 / tens_total
        # serving-thread per-tensor cost of the headline path beyond the hook floor: (ΔT − hashing-induced) / tensors — attributed by tensor count
        L += [f"## 2. Per value-family row — headline path `{head['label']}` ({head['c0'].get('model', '')}, {n_steps} engine steps, {bound:,} B bound / request in "
              f"{tens_total:,} tensors; ΔT = {d_wall_ms:+.1f} ms measured)", "",
              "Attribution rule (calculated): serving-thread acquisition cost ∝ tensors (measured per-tensor hook floor from `hooks_only`: "
              f"{hook_ms_per_tensor * 1e3 if hook_ms_per_tensor else float('nan'):.1f} µs/tensor; total ΔT/tensor {d_wall_ms / tens_total * 1e3 if d_wall_ms else float('nan'):.1f} µs); "
              "movement and hashing ∝ bytes of the run's measured worker totals; temp memory = bytes staged (device arena + pinned host) = 2 × bytes; "
              "persistent = per-step roots + layout digests; opening = one 64 KB leaf + path (measured mean). Per-row costs need not add to ΔT (overlap/contention).", "",
              "| row | value family | bytes / request (manifest, calc.) | tensors / step (calc.) | acquisition + extra reads/writes | commitment work (hash + fold) | temp / persistent memory | sustained latency impact (serving thread) | opening generation | secure-commit coverage | status |",
              "|---|---|---:|---:|---|---|---|---|---|---:|---|"]
        op = head["commit0"].get("openings", {}) or {}
        for row in cov["rows"]:
            if not row["required"]:
                continue
            b = row["logical_bytes"]
            share = b / bound if bound else 0
            tps = tensors_per_step(row["row"], a.n_layers)
            secure = row["status"] in ("secure", "secure-derived")
            if row["row"] == "R1a":
                wr = head["commit0"].get("weight_registration") or {}
                acq = f"registered once per process: {wr.get('wall_s', 0):.2f} s for {wr.get('bytes', 0):,} B (measured; excluded from steady state)"
                work, mem, lat, opn = "in the registration", f"{wr.get('bytes', 0):,} B read once; persistent 32 B root", "none per request (root referenced)", "production weights-tree opening (not measured here)"
            elif row["status"] == "secure-derived":
                acq = "none: value never stored (in-register conversion); bound through its source row"
                work, mem = "none (no leaf)", "none"
                lat = "none"
                opn = f"source leaf opening ({op.get('bytes_mean', 0):,.0f} B mean) + φ = one shift (calc.)"
            elif secure and row["row"] in HIDDEN_ROWS:
                src = ((head["commit0"].get("extra") or {}).get("sources") or {})
                st = next(iter(src.values()), {}) if src else {}
                geo = st.get("geometries") or {}
                acq = (f"CMT-2 M1 export tap in place of `_vllm_fa2_C.varlen_fwd`: {tps} launches/step × {n_steps} steps, ONE stream per launch shared by R5–R8 "
                       f"(stream {', '.join(f'{k}: {v:,} B' for k, v in geo.items())}); kernel-side +34 % prefill / +26 % decode of a ~6 µs kernel and the stream's extra HBM "
                       f"writes (CMT-2 cost_table, measured on L40S); snapshot = one cudaMemcpyAsync per launch (fast path); tap_skipped {st.get('skipped')}")
                work = f"{(sp.get('hashing') or 0) * share * 1e3:.2f} ms/request SHA-256 pos-leaf 64 KB + fold (calc. share of measured {ms(sp.get('hashing'))} ms; worker thread; the stream bytes exceed the manifest bytes: unvisited/padded words are bound too)"
                mem = f"temp {2 * b:,.0f} B (arena + pinned, calc.) + the per-geometry stream buffers ({sum(geo.values()):,} B persistent, reused)"
                lat = f"≈ {tps * n_steps * (d_wall_ms / tens_total if d_wall_ms else 0):.2f} ms/request of ΔT by launch count (calc.) + the tapped kernel's own slowdown (inside the measured ΔT); movement share {(sp.get('movement') or 0) * share * 1e3:.2f} ms (worker)"
                opn = f"{op.get('bytes_mean', 0):,.0f} B/opening (measured mean; identity = (step, byte range) → launch + word offset via the stream layout)"
            elif secure and tps is not None:
                extra_rw = 2 * b  # D2D snapshot write + D2H read of the same bytes
                acq = (f"{tps} forward hooks/step × {n_steps} steps; extra device writes+reads {extra_rw:,} B/request (D2D snapshot + D2H) (calc.); "
                       f"hook floor {tps * n_steps * (hook_ms_per_tensor or 0):.2f} ms/request (calc. from hooks_only)")
                work = f"{(sp.get('hashing') or 0) * share * 1e3:.2f} ms/request SHA-256 pos-leaf 64 KB + fold (calc. share of measured {ms(sp.get('hashing'))} ms; worker thread)"
                mem = f"temp {2 * b:,.0f} B (arena + pinned, calc.); persistent 32 B/step root share"
                lat = f"≈ {tps * n_steps * (d_wall_ms / tens_total if d_wall_ms else 0):.2f} ms/request of ΔT by tensor count (calc.); movement share {(sp.get('movement') or 0) * share * 1e3:.2f} ms (worker)"
                opn = f"{op.get('bytes_mean', 0):,.0f} B/opening (measured mean), {ms(op.get('generate_s'), 2)} ms to generate {op.get('n', 0)} openings/run ({(op.get('generate_s') or 0) / max(op.get('n', 1), 1) * 1e6:.0f} µs each)"
            elif row["status"] == "treatment-declared":
                acq, work, mem, lat, opn = "treatment declared (see note)", "—", "—", "—", "—"
            elif row["status"] == "derived-declared":
                acq, work, mem, lat, opn = ("**derived-row route DECLARED, verifier NOT implemented** (φ = exact model over the bound source row + registered weights)",
                                            "none if the route is accepted (no leaf); else M1-style export of the row's bytes (not built)", "none", "none",
                                            "source-row leaf + weight opening + replay (not implemented)")
            else:
                acq, work, mem, lat, opn = "**not measured: no secure path in this implementation**", "—", "—", "—", "—"
            cvg = "—" if row["secure_commit_coverage"] is None else f"{row['secure_commit_coverage'] * 100:.1f} %"
            L.append(f"| {row['row']} | {row['name'][:60]} | {b:,.0f} | {tps if tps is not None else '—'} | {acq} | {work} | {mem} | {lat} | {opn} | {cvg} | "
                     f"{row['status']}{(' — ' + row['note'][:80]) if row['note'] else ''} |")
        L += ["", f"Integrated (measured): ΔT = {d_wall_ms:+.1f} ms per request ({pct(s.get('delta_wall_mean'))}); decode step {pct(statistics.fmean(s['delta_decode']))}; "
              f"root-ready delay {ms(head['rr'])} ms after the last step; steady-state hashing growth {pct(s.get('steady_growth'))}; backlog max {s.get('backlog_max')} steps "
              f"(roots on time at this request rate; sustained multi-request traffic NOT measured).", ""]
        L += ["", f"### Coverage rows — `{head['label']}`", "", cc.render_v2(cov, a.manifest, [str(Path(head["dir"]) / "declaration_*.json")]).split("\n", 6)[-1]]

    L += ["", "## 3. Hardware / commit / command", ""]
    for p in paths:
        v = p["v"]
        cmd = v.get("argv") or v.get("command", "")
        L.append(f"- **{p['label']}**: GPU `{p['gpu']}`; vLLM `{v.get('vllm', '')}`; torch `{v.get('torch', '')}`; commit `{(v.get('export') or {}).get('commit') or v.get('git_head', '')}`; "
                 f"bytes bound per run {p['commit0'].get('bytes_bound', 0):,}; command `{' '.join(cmd) if isinstance(cmd, list) else cmd}`; raw `{p['dir']}`")
    L += ["", "## 4. Residual per path (required manifest rows NOT securely committed)", ""]
    for p in paths:
        cov = p["cov"]
        if not cov:
            continue
        L.append(f"- **{p['label']}** — verdict **{cov['verdict'].upper()}**: " + "; ".join(
            f"{r['row']} {r['status']} ({r['residual_bytes']:,.0f} B)" for r in cov["rows"] if r["required"] and r["status"] not in ("secure", "secure-derived")))
    Path(a.out).write_text("\n".join(L) + "\n")
    print("\n".join(L[:14]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
