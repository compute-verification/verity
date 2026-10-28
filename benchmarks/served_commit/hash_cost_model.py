"""h(row, σ) for the cost model, from hash_cost.sh runs' out/ directories: stdlib only, no GPU.

    python3 benchmarks/served_commit/hash_cost_model.py OUT_JSON RUN_OUT...    (an ARMS run: each arm's directory)

h is in MAC-times a committed byte (hash_cost_h.py). For each T and row size rb:
  h_x(rb)      exclusive: the served kernel configuration's leaves alone (median over runs; the others are listed), plus the
               cheapest tree form at that row size
  h_b(rb, b)   beside the GEMM, b bytes a forward: the least h_beside of a measured point that fits b bytes beside one forward,
               floored at 0 (a point inside the noise) and capped at h_x (past that the bytes go exclusive); h_b_hi adds twice the
               A/A noise floor where an A-B-A run measured it
  h(rb, σ; B) = σ h_b(rb, σ B) + (1 - σ) h_x(rb), for B committed bytes a forward, σ of them hashed beside the GEMM
  h_b median   the robust beside figure: the median h_beside of the A-B-A points at the phases whose median is least
  h_cycles     the same with the clock's fall beside the hash taken out (NVML in each leg), the per-clock figure
A point's phases and cap are its kernel configuration: VERITY_ROWS_PHASES (data launches a leaf batch) and CTAs a launch. Each loop
is its own key: T for the bare GEMM loop, T:step for the loop shaped like the served step (T:step:nogap without the host gap), and
:wgmma after either for the persistent wgmma GEMM.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hash_cost_h import analyze

GRID_MB = (10, 25, 50, 100, 200, 400, 800, 1600, 3200, 6400, 12800)
SERVED = ("t256", "4", "split")                                # native_rows.cu as served: 256 threads, 4 data phases, split leaves


def collect(runs: list[Path]) -> dict:
    by: dict = {}
    for run in runs:
        a = analyze(run)
        if a["stride0"]:                                       # hashes one row over and over: a diagnostic, not a commitment
            continue
        rid = next((p for p in run.parts if p.startswith("r2026")), str(run))
        loop = (f"{a['T']}" + (":step" if a["step"] else "") + (":nogap" if a["step"] and not a["gap_cycles"] else "")
                + (":wgmma" if a["gemm_kind"] == "wgmma" else ""))
        for key, row in a["rows"].items():
            e = by.setdefault(loop, {}).setdefault(str(row["row_bytes"]), {"alone": [], "tree": [], "points": []})
            src = {"run": rid, "arm": run.name, "form": row["form"], "phases": a["phases"], "kind": a["form_kind"]}
            e["alone"].append({**src, "h_leaf": row["h_leaf"], "gbps": row["leaf_gbps"]})
            if "h_tree" in row:
                e["tree"].append({**src, "h_tree": row["h_tree"]})
            for p in row["beside"]:
                # the whole grid beside the bare GEMM loop displaces it rather than sharing; beside the step it's the served launch
                if (p["cap"] == 0 and not a["step"]) or not p["overlapped"]:
                    continue
                e["points"].append({**src, **{k: p.get(k) for k in ("cap", "beside_gbps", "kappa", "h_beside", "h_noise",
                                                                    "h_cycles", "w_alone", "w_beside", "bytes_per_fwd")}})
    return by


def model(by: dict) -> dict:
    out: dict = {}
    trees = {rb: min(t["h_tree"] for e in by.values() for t in e.get(rb, {}).get("tree", [])) for rb in
             {rb for e in by.values() for rb in e if any(e[rb]["tree"] for e in by.values() if rb in e)}}
    # alone, the hash doesn't see T: each kernel configuration's median over every run, then the cheapest configuration
    alone: dict = {}
    for rows in by.values():
        for rb, e in rows.items():
            for a in e["alone"]:
                alone.setdefault(rb, {}).setdefault((a["form"], a["phases"], a["kind"]), []).append(a)
    def summary(cfg: tuple, xs: list) -> dict:
        return {"form": cfg[0], "phases": cfg[1], "kind": cfg[2], "n": len(xs), "h_leaf": round(statistics.median(x["h_leaf"] for x in xs), 1),
                "h_leaf_range": [min(x["h_leaf"] for x in xs), max(x["h_leaf"] for x in xs)],
                "gbps": statistics.median(x["gbps"] for x in xs), "runs": sorted({x["run"] for x in xs})}

    best_alone = {}
    for rb, groups in alone.items():
        cfg = SERVED if SERVED in groups else min(groups, key=lambda c: statistics.median(x["h_leaf"] for x in groups[c]))
        best_alone[rb] = {**summary(cfg, groups[cfg]), "others": [summary(c, xs) for c, xs in groups.items() if c != cfg]}
    for T, rows in by.items():
        for rb, e in rows.items():
            best = best_alone[rb]
            h_x = best["h_leaf"] + trees.get(rb, 0.0)
            pts = sorted(e["points"], key=lambda p: p["bytes_per_fwd"])
            # a single-baseline point's noise is unmeasured: where A-B-A points exist they set the curve, up to their largest share
            paired = [p for p in pts if p.get("h_noise") is not None]
            reach = max((p["bytes_per_fwd"] for p in paired), default=0)
            use = [p for p in pts if p.get("h_noise") is not None or p["bytes_per_fwd"] > reach]

            def val(p: dict, hi: bool) -> float:
                return max(p["h_beside"], 0.0) + (2 * p["h_noise"] if hi and p.get("h_noise") is not None else 0.0)

            def h_b(b: float, hi: bool = False) -> tuple[float, dict | None]:
                fit = [p for p in use if p["bytes_per_fwd"] >= b]
                if not fit:
                    return h_x, None
                p = min(fit, key=lambda p: val(p, hi))
                return min(val(p, hi), h_x), p

            env = []
            for mb in GRID_MB:
                h, p = h_b(mb * 1e6)
                h_hi, _ = h_b(mb * 1e6, True)
                env.append({"mb_per_fwd": mb, "h_b": round(h, 1), "h_b_hi": round(h_hi, 1),
                            "at": p and {k: p[k] for k in ("run", "arm", "phases", "cap", "beside_gbps", "kappa")}})
            # the robust figure: every A-B-A point of the phases whose median is least (fallback: every point), median and quartiles
            base = paired or pts
            by_ph: dict = {}
            for p in base:
                by_ph.setdefault(p["phases"], []).append(p)
            ph, sel = min(by_ph.items(), key=lambda kv: statistics.median(p["h_beside"] for p in kv[1])) if by_ph else (None, [])
            hs = sorted(p["h_beside"] for p in sel)
            hc = sorted(p["h_cycles"] for p in sel if p.get("h_cycles") is not None)
            robust = {"phases": ph, "n": len(hs), "paired": bool(paired), "h_b_median": round(statistics.median(hs), 1) if hs else None,
                      "h_cycles_median": round(statistics.median(hc), 1) if hc else None,
                      "h_b_quartiles": [round(x, 1) for x in statistics.quantiles(hs, n=4)] if len(hs) >= 2 else None,
                      "share_of_h_x": round(statistics.median(hs) / h_x, 3) if hs else None,
                      "max_gbps": max((p["beside_gbps"] for p in sel), default=None)}
            out.setdefault(T, {})[rb] = {"h_x": round(h_x, 1), "h_b": robust, "h_leaf": best["h_leaf"], "h_tree": trees.get(rb),
                                         "alone_at": best,
                                         "h_b_by_mb_per_fwd": env, "points": pts}
    return out


def main() -> None:
    res = model(collect([Path(p) for p in sys.argv[2:]]))
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    for T, rows in sorted(res.items(), key=lambda kv: (-int(kv[0].split(":")[0]), kv[0])):
        for rb, m in sorted(rows.items(), key=lambda kv: int(kv[0])):
            env = "  ".join(f"{x['mb_per_fwd']}MB:{x['h_b']:.0f}" + (f"/{x['h_b_hi']:.0f}" if x["h_b_hi"] != x["h_b"] else "")
                            for x in m["h_b_by_mb_per_fwd"])
            r = m["h_b"]
            print(f"T {T:>5} rows {int(rb) // 1024:>2} KiB  h_x {m['h_x']:7.1f} (leaf {m['h_leaf']}, tree {m['h_tree']})  h_b median "
                  f"{r['h_b_median']} quartiles {r['h_b_quartiles']} ({r['share_of_h_x']} of h_x; per clock {r['h_cycles_median']}; "
                  f"phases {r['phases']}, {r['n']} points, paired {r['paired']})\n      envelope: {env}")


if __name__ == "__main__":
    main()
