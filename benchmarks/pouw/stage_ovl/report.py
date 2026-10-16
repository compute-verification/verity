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
- cells are keyed by (FR_ROT, M): FR_ROT is the build's choice of rotations on the FMA pipe (stage.cu's bits), and
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", nargs="+")
    ap.add_argument("--json")
    a = ap.parse_args()
    gates, setups, ovl, thru = [], {}, [], []
    for d in map(Path, a.out):
        gates += rows(d / "gate.jsonl")
        thru += rows(d / "shathru.jsonl")
        rot = 0
        for r in rows(d / "ovl.jsonl"):
            if r["mode"] == "ovl_setup":
                rot = r.get("fr_rot", 0)
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
    res: dict = {"gate_lines": len(gates), "gate_failed": len(bad), "setup": setups, "cells": []}
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
        print(f"\nFR_ROT = {rot}, M = {M}: one chunk's GEMMs alone {s.get('gemm_alone_ms_chunk', 0):.2f} ms; plain at saturation "
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
            for n, c in r.get("chunks", {}).items():
                e = sorted(c["end_ms"])
                cell["arms"][n]["ends_over_plain"] = {"first": e[0] / p, "median": e[len(e) // 2] / p, "last": e[-1] / p}
                cell["arms"][n]["stage_spans_ms"] = {"draw": c["draw_ms_mean"], "gemm_commit": c["gemm_commit_ms_mean"]}
            res["cells"].append(cell)
            line = "  ".join(f"{n} {100 * v['cost']:+.1f}%" for n, v in cell["arms"].items() if n != "plain")
            print(f"  k={k:2d} plain {p:8.2f} ms ({cell['plain_ktok_s']:.1f} k tok/s, {arms['plain']['mhz']:.0f} MHz) | {line}")
            if "hashonly" in cell["arms"]:
                print(f"        the commitments alone: {100 * cell['arms']['hashonly']['ms'] / p:.1f}% of plain's makespan")
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
        print(f"\nk x chunk, FR_ROT = {rot}: the best protocol arm's cost against plain (against saturated plain)")
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
        for c in cs:
            if c["k"] == 1 and "eq" in c["arms"]:
                print(f"  lone TTFT M = {c['M']}: {c['arms']['eq']['ms']:.1f} ms against plain {c['plain_ms']:.1f} ms "
                      f"({100 * c['arms']['eq']['cost']:+.0f}%)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
