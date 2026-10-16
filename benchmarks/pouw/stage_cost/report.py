#!/usr/bin/env python3
"""The tables of note:20261016T0846Z-draft-stage-cost from one run of run.sh (STAGE-COST, cmp-13).

  python3 report.py RUN_OUT_DIR [--json OUT.json]

Prefill, per chunk of M tokens (32 layers, 6 stages a layer, 192 commitments and draws a chunk):
- serialized: (serial - gemm) / gemm, each stage's commitment then its GEMM on one stream;
- overlapped: conc / gemm - 1, the commitments and GEMMs on two streams with no dependency (conc_lo: hashes at low priority),
  and green N: on a partition of N SMs, against the GEMMs alone on the whole GPU; pipe_green: C chunks in flight with each
  stage's commitment and draw on the N-SM partition and its GEMM on the rest, against C x gemm;
- in flight: pipe C / (C x gemm) - 1, C chunks each committing, waiting DRAW_US, then multiplying, stage by stage; pipe 1 is a
  lone request's time to first token over the clean forward;
- the critical path a stage: (pipe1 - gemm) / 192, and the commitment's own latency (hash_latency / 32) plus the draw;
  for the fast path (hash=fr) the salt-only part, which runs ahead off the critical path, apart (salt, salt_latency).
Deferred, per T and composition: one layer's noised recompute, ms, share of the int8 peak at the read clock, and the stored
operands a replica (32 layers). Beside: the decode step's slowdown, and whether the pass keeps up with the tokens decode
produces (margin = the pass's token-layers a ms in the window over the step's).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS, STAGES = 32, 6


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


PROVED = ["before", "mid", "mid_host", "cpu16", "cpu64", "s4", "s8", "s4_cpu16", "s8_cpu16", "s8_cpu64", "before_nd", "mid_nd",
          "s8_nd", "cpu16_nd", "s8_cpu16_nd"]  # the arms of runs that don't list theirs (proved_arms)


def proved(r: dict) -> dict:
    """ARMS=r2: the proved forward by lever, its overhead over the GEMMs and its critical path a stage."""
    a = {k: v["ms"] for k, v in r["arms"].items()}
    g, per = a["gemm"], LAYERS * STAGES
    listed = r.get("proved_arms") or [{"name": nm, "k": int(nm[1]) if nm[0] == "s" else 1} for nm in PROVED]
    arms = {}
    for c in listed:
        nm, k = c["name"], c["k"]
        if nm not in a:
            continue
        gk = a.get(f"gemm_{'c' if c.get('col') else 's'}{k}", g)
        h = r.get("host", {}).get(nm, {})
        arms[nm] = {"ms": a[nm], "over_gemm": a[nm] / g - 1, "crit_ms_a_stage": (a[nm] - g) / per,
                    "over_slab_gemm_ms_a_stage": (a[nm] - gk) / per, "host_top_us": h.get("top_us_mean"),
                    "host_ok": not h or (h["host_timeouts"] == 0 and h["gpu_timeouts"] == 0)}
    return {"M": r["M"], "draw_us": r["draw_us"], "gemm_ms": g,
            "slab_gemm_over": {nm[5:]: v / g - 1 for nm, v in a.items() if nm.startswith(("gemm_s", "gemm_c"))},
            "host_sha": r.get("host_sha"), "arms": arms,
            "hash_latency_ms": {k: v["ms"] / LAYERS for k, v in r["hash_latency"].items()},
            "segments_latency_ms": {k: v["ms"] / LAYERS for k, v in r.get("segments_latency", {}).items()},
            "tree_latency_ms": {k: v["ms"] / LAYERS for k, v in r.get("tree_latency", {}).items()}}


def prefill(r: dict) -> dict | None:
    a = {k: v["ms"] for k, v in r["arms"].items()}
    if "serial" not in a:
        return None
    g = a["gemm"]
    per = LAYERS * STAGES
    lat = {k: v["ms"] / LAYERS for k, v in r["hash_latency"].items()}
    m = r["M"]
    sk = {"qkv": 4096, "qk": 6144, "pv": 256, "o": 4096, "gate_up": 4096, "down": 14336}
    draw_only = a.get("pipe1_draw", a.get("pipe1_nodraw_nohash"))  # the arm's name before the fast path
    out = {
        "hash": r.get("hash", "sz"), "M": m, "gemm_ms": g, "gemm_share": r["arms"]["gemm"]["share_of_peak"], "hash_ms": a["hash"],
        "hash_over_gemm": a["hash"] / g, "hash_gb_s": m * r["op_bytes_token_layer"] * LAYERS / a["hash"] / 1e6,
        "salt_ms": a.get("salt"), "tree_ms": a.get("tree"),
        "salt_latency_ms": {k: v["ms"] / LAYERS for k, v in r.get("salt_latency", {}).items()},
        "segments_latency_ms": {k: v["ms"] / LAYERS for k, v in r.get("segments_latency", {}).items()},
        "tree_latency_ms": {k: v["ms"] / LAYERS for k, v in r.get("tree_latency", {}).items()},
        "hash_only_ms_a_stage": (a["pipe1_hash"] - g) / per if "pipe1_hash" in a else None,
        "serialized": a["serial"] / g - 1, "overlapped": a["conc"] / g - 1, "overlapped_lo": a["conc_lo"] / g - 1,
        "green": [{"N": x["N"], "hash_sms": x["hash_sms"], "both_over_gemm_graph": x["both_ms"] / g - 1,
                   "both_over_gemm_split": x["both_ms"] / x["gemm_ms"] - 1, "gemm_split_over_full": x["gemm_ms"] / g - 1}
                  for x in r["green"]],
        "pipe_green": [{"N": x["N"], "C": x["C"], "hash_sms": x["hash_sms"], "over_gemm_graph": x["ms"] / (x["C"] * g) - 1,
                        "over_gemm_split": x["ms"] / x["gemm_ms"] - 1, "gemm_split_over_full": x["gemm_ms"] / (x["C"] * g) - 1}
                       for x in r.get("pipe_green", [])],
        "pipe": {c: a[f"pipe{c}"] / (c * g) - 1 for c in (1, 2, 3) if f"pipe{c}" in a},
        "lone_ttft": a["pipe1"] / g - 1,
        "crit_ms_a_stage": (a["pipe1"] - g) / per,
        "draw_only_ms_a_stage": (draw_only - g) / per,
        "hash_latency_ms": lat,
        "hash_latency_gb_s": {k: m * sk[k] / (v * 1e6) for k, v in lat.items()},
        "gemm_ms_a_stage": {x["stage"]: x["ms_alone"] for x in r["gemms"]},
        "draw_us": r["draw_us"], "draws_a_s_gpu": {c: c * per / (a[f"pipe{c}"] * 1e-3) for c in (1, 2, 3) if f"pipe{c}" in a},
        "node_priorities": r.get("node_priorities"),
    }
    return out


def deferred(r: dict) -> dict:
    a = r["arms"]
    n = r["layers_timed"]
    return {
        "T": r["T"], "comp": r["comp"], "att": r.get("att", "lt"), "seqs": r["seqs"], "tokens_a_seq": r["tokens_a_seq"],
        "ctx_mean": r["ctx_mean"],
        "att_lt_ms_layer": a["att_lt"]["ms"] / n if "att_lt" in a else None,
        "att_fused_ms_layer": a["att_fused"]["ms"] / n if "att_fused" in a else None,
        "share_att_fused": a["att_fused"]["share_of_peak"] if "att_fused" in a else None,
        "gemm_ms_layer": a["gemm"]["ms"] / n, "form_ms_layer": a["form"]["ms"] / n, "lin_ms_layer": a["lin"]["ms"] / n,
        "att_ms_layer": a["att"]["ms"] / n, "share_gemm": a["gemm"]["share_of_peak"], "share_form": a["form"]["share_of_peak"],
        "share_lin": a["lin"]["share_of_peak"], "share_att": a["att"]["share_of_peak"],
        "att_mac_frac": r["att_macs"] / (r["att_macs"] + r["lin_macs"]),
        "us_a_token_layer": a["gemm"]["ms"] / n / r["T"] * 1e3, "us_a_token_layer_form": a["form"]["ms"] / n / r["T"] * 1e3,
        "stored_gb_replica": r["stored_bytes_replica"] / 1e9, "mhz": a["gemm"]["mhz"],
    }


def beside(r: dict) -> dict:
    a = r["arms"]
    base = (a["full"]["ms"] + a["full2"]["ms"]) / 2
    tl = r["B"] * LAYERS  # token-layers a decode step produces

    def margin(arm: str, key: str) -> float | None:
        w = r[key]
        if not w["window_ms"] or arm not in a:
            return None
        return (w["pass_layers"] * r["T"] / w["window_ms"]) / (tl / a[arm]["ms"])

    return {
        "T": r["T"], "comp": r["comp"], "att": r.get("att", "lt"), "forming": r["forming"], "B": r["B"], "L": r["L"], "step_ms": base,
        "pass_alone_ms": a["pass_alone"]["ms"], "pass_alone_green_ms": a.get("pass_alone_green", {}).get("ms"),
        "prio_slowdown": r["prio"]["slowdown"] - 1, "prio_layers": r["prio"]["pass_layers"], "prio_margin": margin("prio", "prio"),
        "green_slowdown": r["green"]["slowdown"] - 1, "green_slowdown_vs_split": r["green"]["slowdown_vs_split"] - 1,
        "split_slowdown": a["split"]["ms"] / base - 1 if "split" in a else None, "green_layers": r["green"]["pass_layers"],
        "green_margin": margin("green", "green"), "fill_sms": r["fill_sms"], "stored_gb_replica": r["stored_bytes_replica"] / 1e9,
        "paced_prio_slowdown": r["paced"]["prio"]["slowdown_mean"] - 1, "paced_prio_tail_ms": r["paced"]["prio"]["tail_ms"],
        "paced_green_slowdown": r["paced"]["green"]["slowdown_mean"] - 1, "paced_green_tail_ms": r["paced"]["green"]["tail_ms"],
        "paced_reps": r["paced"]["reps"], "paced_layers": r["paced"]["prio"]["pass_layers"],
        "steps_a_subround": r["T"] / r["B"], "subround_s": r["T"] / r["B"] * base / 1e3,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = Path(a.out)
    allp = rows(d / "prefill.jsonl")
    pre = [r for r in allp if r.get("mode") == "prefill"]
    res = {"gate": rows(d / "gate.jsonl"), "sha_latency": rows(d / "shalat.jsonl"),
           "prefill": [p for p in (prefill(r) for r in pre) if p],
           "proved": [proved(r) for r in pre if r.get("proved_arms") or "before" in r["arms"]],
           "phases": [r for r in allp if r.get("mode") == "phases"],
           "deferred": [deferred(r) for r in rows(d / "deferred.jsonl")], "beside": [beside(r) for r in rows(d / "beside.jsonl")]}
    pct = lambda x: "-" if x is None else f"{100 * x:+.1f}%"  # noqa: E731
    print("gate:", [(g.get("what"), g.get("k"), g.get("R"), g.get("m"), g.get("ok"), g.get("fast_ok"), g.get("mid_ok"),
                     g.get("slab4_ok"), g.get("fast_tree_equal"), g.get("control_rejected"), g.get("host_top")) for g in res["gate"]])
    for s in res["sha_latency"]:
        if s.get("mode") == "cpu_latency":
            print(f"host SHA-512 compression (us): C++ {s['cpp_us_a_compression']:.3f}, libcrypto {s['ossl_us_a_compression']:.3f} "
                  f"(equal {s['ossl_equal']}, uses {s['host_uses']}); GPU-host round trip through mapped memory "
                  f"{s['mapped_round_trip_us']:.2f} us")
        else:
            print(f"SHA-512 compression latency in a chain at {s['mhz']} MHz (us): {s['us_a_compression']}"
                  + (f", split equal {s['split_equal']}" if "split_equal" in s else ""))
    for p in res["proved"]:
        print(f"\nproved prefill M={p['M']} (draw {p['draw_us']} us): gemm {p['gemm_ms']:.2f} ms; slabbed gemm "
              + ", ".join(f"{k} {pct(v)}" for k, v in p["slab_gemm_over"].items()) + f"; host SHA {p['host_sha']}")
        print("  arm          ms      over gemm   critical path a stage (ms)   over slabbed gemm   host top (us)")
        for nm, x in p["arms"].items():
            ht = "-" if x["host_top_us"] is None else f"{x['host_top_us']:.1f}"
            print(f"  {nm:12s} {x['ms']:8.2f} {pct(x['over_gemm']):>10s}   {x['crit_ms_a_stage']:8.4f}"
                  f"                   {x['over_slab_gemm_ms_a_stage']:8.4f}           {ht}" + ("" if x["host_ok"] else "  HOST FAILED"))
        print("  commitment latency a stage (ms): " + ", ".join(f"{k} {v:.3f}" for k, v in p["hash_latency_ms"].items())
              + "; segments " + ", ".join(f"{k} {v:.3f}" for k, v in p["segments_latency_ms"].items())
              + "; leaves and tree " + ", ".join(f"{k} {v:.3f}" for k, v in p["tree_latency_ms"].items()))
        for dr, lab in ((True, "with the draw"), (False, "without")):
            ok = {nm: x for nm, x in p["arms"].items() if x["host_ok"] and ("nd" in nm.split("_")) != dr}
            if ok:
                nm = min(ok, key=lambda n: ok[n]["ms"])
                print(f"  best {lab}: {nm}, {ok[nm]['ms']:.2f} ms (lone-request TTFT), {pct(ok[nm]['over_gemm'])} over the GEMMs, "
                      f"critical path {ok[nm]['crit_ms_a_stage']:.4f} ms a stage")
    for ph in res["phases"]:
        top = [x for x in ph["last_us"][13:28] if x >= 0]
        print(f"phases M={ph['M']} {ph['stage']} ({ph['ctas']} CTAs, us from the segments' start): segments end {ph['seg_end_us']:.1f}, "
              f"tree start {ph['tree_start_us']:.1f}; CTA 0 x {ph['cta0_us'][1]:.1f}, hm96 {ph['cta0_us'][2]:.1f}, frame leaf "
              f"{ph['cta0_us'][3]:.1f}, levels " + " ".join(f"{x:.1f}" for x in ph["cta0_us"][5:13] if x >= 0)
              + f"; subtrees done min/median/max {'/'.join(f'{x:.1f}' for x in ph['subtree_done_us'])}; last CTA {ph['last_cta']}: "
              + " ".join(f"{x:.1f}" for x in top))
    for p in res["prefill"]:
        print(f"\nprefill M={p['M']} hash={p['hash']}: gemm {p['gemm_ms']:.2f} ms ({100 * p['gemm_share']:.1f}% of peak), hash "
              f"{p['hash_ms']:.2f} ms ({p['hash_gb_s']:.0f} GB/s, {100 * p['hash_over_gemm']:.1f}% of gemm)"
              + (f", salt part {p['salt_ms']:.2f} ms, rest {p['tree_ms']:.2f} ms" if p["salt_ms"] is not None else ""))
        print(f"  serialized {pct(p['serialized'])}  overlapped {pct(p['overlapped'])}  overlapped(lo) {pct(p['overlapped_lo'])}")
        for g in p["green"]:
            print(f"  green N={g['N']} ({g['hash_sms']} SMs): vs gemm(graph) {pct(g['both_over_gemm_graph'])}, vs gemm on split "
                  f"{pct(g['both_over_gemm_split'])}, split gemm vs full {pct(g['gemm_split_over_full'])}")
        print("  in flight: " + "  ".join(f"C={c}: {pct(v)}" for c, v in p["pipe"].items()) + f"   lone TTFT {pct(p['lone_ttft'])}")
        for g in p["pipe_green"]:
            print(f"  in flight on a split, C={g['C']} N={g['N']} ({g['hash_sms']} SMs): vs C x gemm(graph) {pct(g['over_gemm_graph'])}"
                  f", vs C gemms on split {pct(g['over_gemm_split'])}, split gemms vs C x gemm {pct(g['gemm_split_over_full'])}")
        ho = p["hash_only_ms_a_stage"]
        print(f"  critical path a stage {p['crit_ms_a_stage']:.3f} ms (draw alone {p['draw_only_ms_a_stage']:.3f}"
              + (f", hash alone {ho:.3f}" if ho is not None else "") + "); hash latency: "
              + ", ".join(f"{k} {v:.3f} ms" for k, v in p["hash_latency_ms"].items()))
        if p["salt_latency_ms"]:
            print("  salt-only part (ahead): " + ", ".join(f"{k} {v:.3f} ms" for k, v in p["salt_latency_ms"].items()))
        if p["segments_latency_ms"]:
            print("  of the rest, segment digests: " + ", ".join(f"{k} {v:.3f}" for k, v in p["segments_latency_ms"].items())
                  + "; leaves and tree: " + ", ".join(f"{k} {v:.3f}" for k, v in p["tree_latency_ms"].items()))
        print("  gemm a stage: " + ", ".join(f"{k} {v:.3f} ms" for k, v in p["gemm_ms_a_stage"].items()))
        print(f"  draws/s a GPU: {p['draws_a_s_gpu']}  node priorities: {p['node_priorities']}")
    if res["deferred"]:
        print("\ndeferred, one layer: T comp seqs tok/seq | gemm ms (share) | form ms (share) | lin share | att share (att MAC frac) | "
              "us/token-layer | stored GB")
    for r in res["deferred"]:
        fu = (f" | att lt {r['att_lt_ms_layer']:.3f} ms, fused {r['att_fused_ms_layer']:.3f} ms ({100 * r['share_att_fused']:.1f}%)"
              if r["att_fused_ms_layer"] is not None else "")
        print(f"  {r['T']:6d} {r['comp']:6s} {r['att']:5s} {r['seqs']:3d} {r['tokens_a_seq']:5d} | {r['gemm_ms_layer']:8.3f} "
              f"({100 * r['share_gemm']:.1f}%) | {r['form_ms_layer']:8.3f} ({100 * r['share_form']:.1f}%) | {100 * r['share_lin']:.1f}% | "
              f"{100 * r['share_att']:.1f}% ({100 * r['att_mac_frac']:.0f}%) | {r['us_a_token_layer']:.4f} / "
              f"{r['us_a_token_layer_form']:.4f} | {r['stored_gb_replica']:.1f}" + fu)
    if res["beside"]:
        print("\nbeside decode: T comp forming | step ms | saturated prio: slowdown, margin | saturated green: slowdown (vs split), "
              "margin | paced prio: slowdown, tail ms | paced green: slowdown, tail ms | sub-round s")
    for r in res["beside"]:
        mg = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
        print(f"  {r['T']:6d} {r['comp']:6s} {r['att']:5s} {str(r['forming']):5s} | {r['step_ms']:.3f} | {pct(r['prio_slowdown'])}, {mg(r['prio_margin'])} "
              f"({r['prio_layers']} layers) | {pct(r['green_slowdown'])} ({pct(r['green_slowdown_vs_split'])}), {mg(r['green_margin'])} "
              f"({r['green_layers']} layers) | {pct(r['paced_prio_slowdown'])}, {r['paced_prio_tail_ms']:.1f} | "
              f"{pct(r['paced_green_slowdown'])}, {r['paced_green_tail_ms']:.1f} | {r['subround_s']:.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
