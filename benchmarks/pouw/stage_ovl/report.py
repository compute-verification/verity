#!/usr/bin/env python3
"""The tables of note:20261016T1037Z-draft-stage-ovl from one or more runs of run.sh (STAGE-OVL, cmp-13).

  python3 report.py RUN_OUT_DIR... [--json OUT.json]

For each chunk size M and depth k (k chunks of M tokens in flight, each its own commit -> draw -> GEMM chain):
- cost: an arm's makespan over plain's (the same k sequences plain int8, linears batched over all k * M tokens) minus 1;
  cost_sat: over the plain rate at saturation, the best plain tokens/s at that M over every k measured;
- the protocol arms are the ones that commit and draw (eq, prio, hhi, hprio, rrW); best is the fastest of them;
- lone TTFT: k = 1's eq over plain at k = 1; under load: each chunk's end (from the draws' stamps) over plain's makespan;
- memory a chunk in flight: the bench's activations (operands, A, the int32 outputs, a cuBLASLt workspace) and commitment
  state (stage.cu's Commit: digests, leaves, tree, the salt-only parts of all 32 layers), measured by cudaMemGetInfo;
- beside.jsonl: plain's GEMMs against the same tokens' commitment compressions in small CTAs at the highest priority: its
  cost launched before (hfirst) or after (gfirst) the GEMMs as one kernel, or spread (a slice after each GEMM, waiting on
  it), its rate, and the rate the GEMMs need (need_rate);
- cells are keyed by (variant, M): the variant is the build's FR_ROT (rotations on the FMA pipe, stage.cu's bits) and
  CARVE (the commitments' and draws' kernels' preferred shared-memory carveout, when set), and
  shathru.jsonl's SHA-512 compressions a second on the whole GPU are printed by rotation choice and CTAs a SM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROTOCOL = ("eq", "prio", "hhi", "hprio", "hlo")


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


def is_protocol(name: str) -> bool:
    return name in PROTOCOL or name.startswith("rr")


def lite_kind(name: str) -> str | None:
    """the protocol's schedules with each commitment a stand-in of the same compressions in small CTAs (not real roots):
    'lite' flat (understated latency), 'litec' chained as the tree is"""
    if name in ("liteonly", "liteconly") or not name.startswith("lite"):
        return None
    return "litec" if name.startswith("litec") else "lite"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    gates, setups, ovl, thru, beside = [], {}, [], [], []
    for d in map(Path, a.out):
        gates += rows(d / "gate.jsonl")
        thru += rows(d / "shathru.jsonl")
        beside += rows(d / "beside.jsonl")
        rot = "rot0"
        for r in rows(d / "ovl.jsonl"):
            if r["mode"] == "ovl_setup":
                rot = f"rot{r.get('fr_rot', 0)}" + (f" carve{r['carve']}" if r.get("carve", -1) >= 0 else "")
                setups[(rot, r["M"])] = r
            else:
                ovl.append({**r, "fr_rot": rot})
    for t in thru:
        c = t["compressions_a_s"]
        best = max(c, key=c.get)
        print(f"SHA-512 on the whole GPU ({t['sms']} SMs, {t['mhz']:.0f}-{t.get('mhz_after', 0):.0f} MHz): best {best} "
              f"{c[best]:.3e}/s; " + ", ".join(f"{n} {v:.2e}" for n, v in c.items()))
    bad = [g for g in gates if g.get("ok") is False or g.get("fast_ok") is False or g.get("control_rejected") is False]
    print(f"gate: {len(gates)} lines, {len(bad)} failed" + (f": {bad}" if bad else ""))
    for b in beside:
        print(f"beside M = {b['M']}, k = {b['k']}: GEMMs alone {b['gemm_ms']:.1f} ms ({b['gemm_mhz']:.0f} MHz), "
              f"{b['H']:.3e} compressions (C = {b['C']}) need {b['need_rate']:.2e}/s to keep pace")
        for c in b["configs"]:
            al, hf, gf = c["alone"], c["hfirst"], c["gfirst"]
            print(f"  {c['thr']:3d} threads x {c['ctas_a_sm']} a SM, carveout {c.get('carve', -1):3d}: alone {al['ms']:.1f} ms ({100 * al['of_gemm']:.1f}% of the GEMMs, "
                  f"{al['hash_rate']:.2e}/s); hash first {100 * hf['cost']:+.1f}% (hash ends {hf['hash_end_ms']:.1f}, GEMMs "
                  f"{hf['gemm_end_ms']:.1f} ms, {hf['hash_rate']:.2e}/s); GEMMs first {100 * gf['cost']:+.1f}% (hash "
                  f"{gf['hash_start_ms']:.1f}-{gf['hash_end_ms']:.1f}, GEMMs {gf['gemm_end_ms']:.1f} ms)"
                  + (f"; spread over {c['slices']} slices {100 * c['spread']['cost']:+.1f}% (GEMMs end "
                     f"{c['spread']['gemm_end_ms']:.1f}, hash {c['spread']['hash_end_ms']:.1f} ms)" if "spread" in c else ""))
    res: dict = {"gate_lines": len(gates), "beside": beside, "gate_failed": len(bad), "setup": {f"{r}/{m}": v for (r, m), v in setups.items()},
                 "cells": []}
    res["shathru"] = thru
    plain_best = {}
    for r in ovl:
        p = r["arms"]["plain"]["ms"]
        plain_best[r["M"]] = max(plain_best.get(r["M"], 0), r["k"] * r["M"] / p)
    root_bad = []
    for rot, M in sorted({(r["fr_rot"], r["M"]) for r in ovl}):
        s = setups.get((rot, M), {})
        mem = s.get("mem_bytes", {})
        kmax = s.get("k_max", 1)
        print(f"\n{rot}, M = {M}: one chunk's GEMMs alone {s.get('gemm_alone_ms_chunk', 0):.2f} ms; plain at saturation "
              f"{plain_best[M]:.1f} k tokens/s; memory a chunk in flight: activations "
              f"{mem.get('activations', 0) / kmax / 2**20:.0f} MiB, commitment state {mem.get('commit_state', 0) / kmax / 2**20:.0f} MiB")
        for r in sorted((x for x in ovl if x["M"] == M and x["fr_rot"] == rot), key=lambda x: x["k"]):
            k, arms = r["k"], r["arms"]
            p = arms["plain"]["ms"]
            sat = k * M / plain_best[M]
            cell = {"fr_rot": rot, "M": M, "k": k, "plain_ms": p, "plain_mhz": arms["plain"]["mhz"], "plain_ktok_s": k * M / p,
                    "roots": r.get("roots", {}), "arms": {}}
            root_bad += [(rot, M, k, n, v) for n, v in cell["roots"].items() if v != ("differ" if n == "nohash" else "equal")]
            for n, v in arms.items():
                cell["arms"][n] = {"ms": v["ms"], "cost": v["ms"] / p - 1, "cost_sat": v["ms"] / sat - 1, "mhz": v["mhz"],
                                   "share_of_peak": v["share_of_peak"]}
            proto = {n: v for n, v in cell["arms"].items() if is_protocol(n)}
            if proto:
                b = min(proto, key=lambda n: proto[n]["ms"])
                cell["best"] = b
                cell["best_cost"] = proto[b]["cost"]
                cell["best_cost_sat"] = proto[b]["cost_sat"]
            for kind in ("lite", "litec"):
                lite = {n: v for n, v in cell["arms"].items() if lite_kind(n) == kind}
                if lite:
                    b = min(lite, key=lambda n: lite[n]["ms"])
                    cell[f"best_{kind}"] = b
                    cell[f"best_{kind}_cost"] = lite[b]["cost"]
            for n, c in r.get("chunks", {}).items():
                e = sorted(c["end_ms"])
                cell["arms"][n]["ends_over_plain"] = {"first": e[0] / p, "median": e[len(e) // 2] / p, "last": e[-1] / p}
                cell["arms"][n]["stage_spans_ms"] = {"draw": c["draw_ms_mean"], "gemm_commit": c["gemm_commit_ms_mean"]}
            res["cells"].append(cell)
            line = "  ".join(f"{n} {100 * v['cost']:+.1f}%" for n, v in cell["arms"].items() if n != "plain")
            print(f"  k={k:2d} plain {p:8.2f} ms ({cell['plain_ktok_s']:.1f} k tok/s, {arms['plain']['mhz']:.0f} MHz) | {line}")
            if "hashonly" in cell["arms"]:
                print(f"        the commitments alone: {100 * cell['arms']['hashonly']['ms'] / p:.1f}% of plain's makespan")
            for kind, only, what in (("lite", "liteonly", "flat"), ("litec", "liteconly", "chained")):
                if f"best_{kind}" in cell:
                    lo = cell["arms"].get(only)
                    print(f"        stand-in committer, {what} (small CTAs, same compressions): best {cell[f'best_{kind}']} "
                          f"{100 * cell[f'best_{kind}_cost']:+.1f}%" + (f"; alone {100 * lo['ms'] / p:.1f}% of plain's makespan" if lo else ""))
            if "best" in cell:
                print(f"        best {cell['best']} {100 * cell['best_cost']:+.1f}% (vs saturated plain {100 * cell['best_cost_sat']:+.1f}%); "
                      f"draws {k * 192 / (proto[cell['best']]['ms'] * 1e-3):.0f}/s; roots {cell['roots']}")
                for n in dict.fromkeys([cell["best"], "prio", "hprio"]):
                    tl = cell["arms"].get(n, {}).get("ends_over_plain")
                    if tl:
                        print(f"        {n}: chunk ends / plain makespan first {tl['first']:.2f}, median {tl['median']:.2f}, last {tl['last']:.2f}"
                              f"; a stage: GEMM + commitment {cell['arms'][n]['stage_spans_ms']['gemm_commit']:.3f} ms, draw "
                              f"{cell['arms'][n]['stage_spans_ms']['draw']:.3f} ms")
    print(f"\npoisoned root check: {len(root_bad)} cells off" + (f": {root_bad}" if root_bad else " (hashing arms equal, nohash differs)"))
    for rot in sorted({c["fr_rot"] for c in res["cells"]}):
        cs = [c for c in res["cells"] if c["fr_rot"] == rot]
        print(f"\nk x chunk, {rot}: the best protocol arm's cost against plain (against saturated plain)")
        ks = sorted({c["k"] for c in cs})
        Ms = sorted({c["M"] for c in cs})
        print("  k \\ M " + "".join(f"{M:>24d}" for M in Ms))
        for k in ks:
            cells = {c["M"]: c for c in cs if c["k"] == k}
            out = []
            for M in Ms:
                c = cells.get(M)
                out.append(f"{100 * c['best_cost']:+.1f}% ({100 * c['best_cost_sat']:+.1f}%) {c['best']}" if c and "best" in c else "-")
            print(f"  {k:5d} " + "".join(f"{x:>24s}" for x in out))
        for M in Ms:
            ok = [c for c in cs if c["M"] == M and "best" in c and c["best_cost"] <= 0.05]
            print(f"  smallest k at <= 5% for M = {M}: " + (f"{min(ok, key=lambda c: c['k'])['k']}" if ok else "none measured"))
        for kind, what in (("lite", "flat: latency understated"), ("litec", "chained as the tree: ~40 dependent compressions a stage")):
            if not any(f"best_{kind}" in c for c in cs):
                continue
            print(f"  with the stand-in committer, {what} (each commitment's compressions in 128-thread CTAs, no shared memory):")
            for k in ks:
                cells = {c["M"]: c for c in cs if c["k"] == k}
                out = [f"{100 * cells[M][f'best_{kind}_cost']:+.1f}% {cells[M][f'best_{kind}']}"
                       if M in cells and f"best_{kind}" in cells[M] else "-" for M in Ms]
                print(f"  {k:5d} " + "".join(f"{x:>24s}" for x in out))
            for M in Ms:
                ok = [c for c in cs if c["M"] == M and f"best_{kind}" in c and c[f"best_{kind}_cost"] <= 0.05]
                print(f"  smallest k at <= 5% with the {kind} stand-in, M = {M}: "
                      + (f"{min(ok, key=lambda c: c['k'])['k']}" if ok else "none measured"))
        lone = {}
        for c in cs:
            if c["k"] == 1 and "eq" in c["arms"]:
                lone[c["M"]] = c["arms"]["eq"]["ms"]
                print(f"  lone TTFT M = {c['M']}: {c['arms']['eq']['ms']:.1f} ms against plain {c['plain_ms']:.1f} ms "
                      f"({100 * c['arms']['eq']['cost']:+.0f}%)")
        print("  TTFT under load (ms): plain's makespan (every request's, batched); the best arm's median chunk end; "
              "hprio's first (top-priority) chunk end")
        for c in sorted(cs, key=lambda c: (c["M"], c["k"])):
            if c["k"] == 1 or "best" not in c:
                continue
            p = c["plain_ms"]
            med = c["arms"][c["best"]].get("ends_over_plain", {}).get("median")
            top = c["arms"].get("hprio", {}).get("ends_over_plain", {}).get("first")
            ref = f" ({top * p / lone[c['M']]:.2f}x lone)" if top and c["M"] in lone else ""
            print(f"    M = {c['M']:5d}, k = {c['k']:2d}: plain {p:7.1f}; {c['best']} {med * p if med else float('nan'):7.1f}; "
                  f"hprio first {top * p if top else float('nan'):7.1f}{ref}")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
