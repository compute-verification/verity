#!/usr/bin/env python3
"""The tables of note:20261016T1037Z-draft-stage-ovl from one or more runs of run.sh (STAGE-OVL, cmp-13).

  python3 report.py RUN_OUT_DIR... [--json OUT.json]

For each chunk size M and depth k (k chunks of M tokens in flight, each its own commit -> draw -> GEMM chain):
- cost: an arm's makespan over plain's (the same k sequences plain int8, linears batched over all k * M tokens) minus 1;
  cost_best: over the fastest plain int8 schedule of the cell (plain, chunked or chunkedrrW, which run the k chunks'
  GEMMs on k streams);
  cost_sat: over the plain rate at saturation, the best plain tokens/s at that M over every k measured;
- the protocol arms are the ones that commit and draw (eq, prio, hhi, hprio, rrW); best is the fastest of them;
- lone TTFT: k = 1's eq over plain at k = 1; under load: each chunk's end (from the draws' stamps) over plain's makespan;
- memory a chunk in flight: the bench's activations (operands, A, the int32 outputs, a cuBLASLt workspace) and commitment
  state (stage.cu's Commit: digests, leaves, tree, the salt-only parts of all 32 layers), measured by cudaMemGetInfo;
- beside.jsonl: plain's GEMMs against the same tokens' commitment compressions in small CTAs at the highest priority: its
  cost launched before (hfirst) or after (gfirst) the GEMMs as one kernel, or spread (a slice after each GEMM, waiting on
  it), its rate, and the rate the GEMMs need (need_rate);
- cells are keyed by (variant, M): the variant is the build's FR_ROT (rotations on the FMA pipe, stage.cu's bits),
  CARVE (the commitments' and draws' kernels' preferred shared-memory carveout, when set) and, for the cg arms
  (commit_gpu's real committer, small path), its tile height R (ovl.jsonl and ovl-r*.jsonl), and
  shathru.jsonl's SHA-512 compressions a second on the whole GPU are printed by rotation choice and CTAs a SM;
- check-small.jsonl and verify.jsonl: commit_gpu's case set on the small path, and the timed operands' dumped
  commitments, against verity.ml.commitments.
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


def is_cg(name: str) -> bool:
    """the protocol's schedules with each commitment commit_gpu's small path (real roots)"""
    return name in ("cg", "cgeq") or name.startswith("cgrr")


def is_cgh(name: str) -> bool:
    """the same with layers 1..31's salt records made by host threads and copied in"""
    return name in ("cgh", "cgheq") or name.startswith("cghrr")


KIND = {"cg": is_cg, "cgh": is_cgh}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    gates, setups, ovl, thru, beside, checks, hosts = [], {}, [], [], [], {}, []
    for d in map(Path, a.out):
        gates += rows(d / "gate.jsonl")
        thru += rows(d / "shathru.jsonl")
        beside += rows(d / "beside.jsonl")
        for name in ("check-small.jsonl", "verify.jsonl"):
            s = [r["summary"] for r in rows(d / name) if "summary" in r]
            if s:
                checks[f"{d.name}/{name}"] = s[-1]
        for f in sorted(d.glob("ovl*.jsonl")):
            rot = "rot0"
            for r in rows(f):
                if r["mode"] == "ovl_setup":
                    rot = (f"rot{r.get('fr_rot', 0)}" + (f" carve{r['carve']}" if r.get("carve", -1) >= 0 else "")
                           + (f" R{r['cg_r']}" if r.get("cg_r") else ""))
                    setups[(rot, r["M"])] = r
                elif r["mode"] == "host_salt":
                    hosts.append(r)
                else:
                    ovl.append({**r, "fr_rot": rot})
    for name, s in checks.items():
        print(f"{name}: {s}")
    for h in hosts:
        per_chunk = 31 * h["leaves_a_layer"] * h["ns_leaf_core"] * 1e-9
        print(f"host salts, M = {h['M']}: {h['ns_leaf_core']:.0f} ns a leaf on one core, equal to the GPU's: {h['equal']}; "
              f"{h['leaves_a_layer']} leaves a layer, so layers 1..31 of a chunk take {per_chunk:.2f} core-s "
              f"({h['threads']} threads filled one layer in {h['fill_ms']:.1f} ms)")
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
                    "roots": r.get("roots", {}), "cg_roots": r.get("cg_roots", {}), "arms": {}}
            root_bad += [(rot, M, k, n, v) for n, v in cell["roots"].items() if v != ("differ" if n == "nohash" else "equal")]
            root_bad += [(rot, M, k, n, v) for n, v in cell["cg_roots"].items() if v != ("differ" if n.startswith("nohash") else "equal")]
            pb = min(v["ms"] for n, v in arms.items() if n == "plain" or n.startswith("chunked"))
            cell["plain_best_ms"] = pb
            for n, v in arms.items():
                cell["arms"][n] = {"ms": v["ms"], "cost": v["ms"] / p - 1, "cost_sat": v["ms"] / sat - 1,
                                   "cost_best": v["ms"] / pb - 1, "mhz": v["mhz"], "share_of_peak": v["share_of_peak"]}
            proto = {n: v for n, v in cell["arms"].items() if is_protocol(n)}
            if proto:
                b = min(proto, key=lambda n: proto[n]["ms"])
                cell["best"] = b
                cell["best_cost"] = proto[b]["cost"]
                cell["best_cost_sat"] = proto[b]["cost_sat"]
            if proto:
                cell["best_cost_best"] = proto[cell["best"]]["cost_best"]
            for kind in ("lite", "litec", "cg", "cgh"):
                lite = {n: v for n, v in cell["arms"].items() if (KIND[kind](n) if kind in KIND else lite_kind(n) == kind)}
                if lite:
                    b = min(lite, key=lambda n: lite[n]["ms"])
                    cell[f"best_{kind}"] = b
                    cell[f"best_{kind}_cost"] = lite[b]["cost"]
                    cell[f"best_{kind}_cost_best"] = lite[b]["cost_best"]
            for n, c in r.get("chunks", {}).items():
                e = sorted(c["end_ms"])
                cell["arms"][n]["ends_over_plain"] = {"first": e[0] / p, "median": e[len(e) // 2] / p, "last": e[-1] / p}
                cell["arms"][n]["stage_spans_ms"] = {"draw": c["draw_ms_mean"], "gemm_commit": c["gemm_commit_ms_mean"]}
            res["cells"].append(cell)
            line = "  ".join(f"{n} {100 * v['cost']:+.1f}%" for n, v in cell["arms"].items() if n != "plain")
            if pb < p:
                line += f"  (fastest plain {pb:.2f} ms)"
            print(f"  k={k:2d} plain {p:8.2f} ms ({cell['plain_ktok_s']:.1f} k tok/s, {arms['plain']['mhz']:.0f} MHz) | {line}")
            if "hashonly" in cell["arms"]:
                print(f"        the commitments alone: {100 * cell['arms']['hashonly']['ms'] / p:.1f}% of plain's makespan")
            for kind, only, what in (("lite", "liteonly", "stand-in committer, flat (small CTAs, same compressions)"),
                                     ("litec", "liteconly", "stand-in committer, chained (small CTAs, same compressions)"),
                                     ("cg", "cgonly", "commit_gpu's committer (small path, real roots)"),
                                     ("cgh", "cghonly", "commit_gpu's committer, salts from host threads (real roots)")):
                if f"best_{kind}" in cell:
                    lo = cell["arms"].get(only)
                    print(f"        {what}: best {cell[f'best_{kind}']} {100 * cell[f'best_{kind}_cost']:+.1f}% "
                          f"({100 * cell[f'best_{kind}_cost_best']:+.1f}% against the fastest plain)"
                          + (f"; alone {100 * lo['ms'] / p:.1f}% of plain's makespan" if lo else "")
                          + (f"; roots {cell['cg_roots']}" if kind in KIND else ""))
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
        print(f"\nk x chunk, {rot}: the best protocol arm's cost against plain (against saturated plain; against the fastest plain)")
        print("  (the stand-in tables: against plain (against the fastest plain of the cell: plain, chunked, chunkedrrW))")
        ks = sorted({c["k"] for c in cs})
        Ms = sorted({c["M"] for c in cs})
        print("  k \\ M " + "".join(f"{M:>32d}" for M in Ms))
        for k in ks:
            cells = {c["M"]: c for c in cs if c["k"] == k}
            out = []
            for M in Ms:
                c = cells.get(M)
                out.append(f"{100 * c['best_cost']:+.1f}% ({100 * c['best_cost_sat']:+.1f}%; {100 * c['best_cost_best']:+.1f}%) {c['best']}"
                           if c and "best" in c else "-")
            print(f"  {k:5d} " + "".join(f"{x:>32s}" for x in out))
        for M in Ms:
            ok = [c for c in cs if c["M"] == M and "best" in c and c["best_cost"] <= 0.05]
            okb = [c for c in cs if c["M"] == M and "best" in c and c["best_cost_best"] <= 0.05]
            print(f"  smallest k at <= 5% for M = {M}: " + (f"{min(ok, key=lambda c: c['k'])['k']}" if ok else "none measured")
                  + "; against the fastest plain: " + (f"{min(okb, key=lambda c: c['k'])['k']}" if okb else "none measured"))
        for kind, what in (("lite", "the stand-in committer, flat: latency understated"),
                           ("litec", "the stand-in committer, chained as the tree: ~40 dependent compressions a stage"),
                           ("cg", "commit_gpu's committer, the real roots"),
                           ("cgh", "commit_gpu's committer, salts from host threads, the real roots")):
            if not any(f"best_{kind}" in c for c in cs):
                continue
            print(f"  with {what} (128-thread CTAs, no shared memory):")
            for k in ks:
                cells = {c["M"]: c for c in cs if c["k"] == k}
                out = [f"{100 * cells[M][f'best_{kind}_cost']:+.1f}% ({100 * cells[M][f'best_{kind}_cost_best']:+.1f}%) {cells[M][f'best_{kind}']}"
                       if M in cells and f"best_{kind}" in cells[M] else "-" for M in Ms]
                print(f"  {k:5d} " + "".join(f"{x:>24s}" for x in out))
            for M in Ms:
                ok = [c for c in cs if c["M"] == M and f"best_{kind}" in c and c[f"best_{kind}_cost"] <= 0.05]
                okb = [c for c in cs if c["M"] == M and f"best_{kind}" in c and c[f"best_{kind}_cost_best"] <= 0.05]
                print(f"  smallest k at <= 5% with the {kind} arms, M = {M}: "
                      + (f"{min(ok, key=lambda c: c['k'])['k']}" if ok else "none measured")
                      + "; against the fastest plain: " + (f"{min(okb, key=lambda c: c['k'])['k']}" if okb else "none measured"))
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
    res["breakdown"] = breakdown([Path(d) for d in a.out])
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


DIAG_WHAT = {"base": "the committer", "noop": "no operand loads", "nos": "no salt-record loads (S, F)",
             "nomask": "no M_key y lookups", "notree": "a CTA's levels in registers", "all": "all four",
             "nosalts": "no salt kernels", "notop": "no levels above the CTAs"}


def breakdown(dirs: list[Path]) -> list[dict]:
    """run.sh's cgdiag: cgonly (the committer alone) and cgrr2 (beside the GEMMs) in each timing-only build or CG_SKIP,
    against the committer as built; each piece's share is base minus the build without it"""
    out = []
    for d in dirs:
        for f in sorted(d.glob("diag-*.jsonl")):
            _, M, name = f.stem.split("-", 2)
            for r in rows(f):
                if r.get("mode") == "ovl":
                    out.append({"M": int(M), "name": name, "k": r["k"], "arms": {n: v["ms"] for n, v in r["arms"].items()}})
    for M, k in sorted({(c["M"], c["k"]) for c in out}):
        cells = {c["name"]: c["arms"] for c in out if c["M"] == M and c["k"] == k}
        base = cells.get("base")
        if not base or "cgonly" not in base:
            continue
        lite = base.get("liteconly")
        print(f"\nbreakdown, M = {M}, k = {k}: cgonly (alone) {base['cgonly']:.1f} ms"
              + (f", the stand-in (liteconly) {lite:.1f} ms, {base['cgonly'] / lite:.2f}x" if lite else ""))
        for name, arms in cells.items():
            if "cgonly" not in arms:
                continue
            fast = arms.get("chunked")
            rr = f"; cgrr2 {100 * (arms['cgrr2'] / fast - 1):+.1f}% against chunked" if fast and "cgrr2" in arms else ""
            lrr = f" (litecrr2 {100 * (arms['litecrr2'] / fast - 1):+.1f}%)" if fast and "litecrr2" in arms and name == "base" else ""
            print(f"  {name:8s} {DIAG_WHAT.get(name, name):32s} cgonly {arms['cgonly']:6.1f} ms, saves "
                  f"{base['cgonly'] - arms['cgonly']:5.1f} ms ({100 * (1 - arms['cgonly'] / base['cgonly']):4.1f}%){rr}{lrr}")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
