"""Where 70B's loss at the recommended point comes from: perplexity ratios to bf16, paired window by window, from qat.py
result.json files (`cpu70b.sh` arms, `research fetch`ed).

Each measurement is keyed by its map (the --targets file's stem, or bf16), its serving spec and its set, and holds one
mean NLL per window (`window_nll`), indexed from the set's start (--wt-skip, --val-skip); chunks of one map and spec
merge. A ratio is exp of the mean per-window NLL difference over the windows both sides have, with that mean's standard
error: every window has the same token count, so the mean of window means is the set's NLL.
    python benchmarks/hardness_defence/loss70b.py RESULT.json... [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

F = "layers.0.self_attn.q_proj+layers.0.self_attn.k_proj+layers.0.self_attn.v_proj+layers.1.self_attn.v_proj"
#: the GPU arms' mean NLL behind the recommended point's +4.22% and +2.49% (FineWeb-Edu in halves of 256 windows)
GPU = {"bf16": {"wikitext2": 1.03289, "fineweb_val": (1.825869 + 1.794778) / 2},
       "e2:F": {"wikitext2": 1.074182, "fineweb_val": (1.851271 + 1.818574) / 2}}


def short(spec: str) -> str:
    return spec.replace(F, "F")


def load(paths: list[Path]) -> tuple[dict, list[str]]:
    data, notes = {}, []
    for p in paths:
        r = json.loads(p.read_text())
        cfg = r["config"]
        smap = "bf16" if cfg["target"] == "bf16" else Path(cfg["targets"]).stem
        start = {"wikitext2": cfg.get("wt_skip", 0), "fineweb_val": cfg.get("val_skip", 0)}
        ev = r["eval_before"]
        specs = [(cfg["serve"] if smap != "bf16" else "bf16", ev)] + list(ev.get("modes", {}).items())
        for spec, e in specs:
            for s in ("wikitext2", "fineweb_val"):
                if s not in e:
                    continue
                w = data.setdefault((smap, short(spec), s), {})
                for i, v in enumerate(e[s]["window_nll"], start[s]):
                    if i in w and abs(w[i] - v) > 1e-6:
                        notes.append(f"{smap} {short(spec)} {s} window {i}: {w[i]:.6f} and {v:.6f} ({p})")
                    w.setdefault(i, v)
    return data, notes


def ratio(a: dict, b: dict) -> dict | None:
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = [a[i] - b[i] for i in common]
    n, m = len(d), sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else float("nan")
    return {"windows": n, "first": common[0], "last": common[-1], "dnll": m, "se": se,
            "pct": 100 * math.expm1(m), "pct_se": 100 * math.exp(m) * se}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    data, notes = load(args.results)
    out = {"vs_bf16": [], "vs_a16": [], "full": [], "notes": notes}
    for (smap, spec, s), w in sorted(data.items()):
        base = data.get(("bf16", "bf16", s))
        if smap != "bf16" and base and (r := ratio(w, base)):
            out["vs_bf16"].append({"map": smap, "spec": spec, "set": s} | r)
        a16 = data.get((smap, "a16", s))
        if smap != "bf16" and spec != "a16" and a16 and (r := ratio(w, a16)):
            out["vs_a16"].append({"map": smap, "spec": spec, "set": s} | r)
    for s in ("wikitext2", "fineweb_val"):
        rec, base = data.get(("70b-defl", "e2:F", s), {}), data.get(("bf16", "bf16", s), {})
        if rec and base:
            common = sorted(set(rec) & set(base))
            cpu = {k: sum(v[i] for i in common) / len(common) for k, v in (("bf16", base), ("e2:F", rec))}
            out["full"].append({"set": s, "windows": len(common), "cpu_bf16": cpu["bf16"], "cpu_e2F": cpu["e2:F"],
                                "cpu_pct": 100 * math.expm1(cpu["e2:F"] - cpu["bf16"]),
                                "gpu_bf16": GPU["bf16"][s], "gpu_e2F": GPU["e2:F"][s],
                                "gpu_pct": 100 * math.expm1(GPU["e2:F"][s] - GPU["bf16"][s])})
    for key in ("vs_bf16", "vs_a16"):
        print(f"== {key}: map, spec, set, windows [first-last], ppl ratio - 1 (± paired SE)")
        for r in out[key]:
            print(f"{r['map']:18} {r['spec']:22} {r['set']:12} {r['windows']:4} [{r['first']}-{r['last']}] "
                  f"{r['pct']:+7.3f}% ± {r['pct_se']:.3f}")
    print("== full sets: windows, CPU bf16 and e2:F mean NLL, CPU ratio - 1, GPU's")
    for r in out["full"]:
        print(f"{r['set']:12} {r['windows']:4} {r['cpu_bf16']:.6f} {r['cpu_e2F']:.6f} {r['cpu_pct']:+.3f}%  "
              f"GPU {r['gpu_bf16']:.6f} {r['gpu_e2F']:.6f} {r['gpu_pct']:+.3f}%")
    for n in notes:
        print("note:", n)
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
