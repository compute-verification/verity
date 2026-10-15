"""The NoLookupGain table from lookup_coissue.cu and fused_coissue.cu summaries (JSON lines on stdin or as files).

For each (n, alphabet): T_tensor = the fastest tensor-only exact program (cuBLASLt int8, cuBLASLt FP8, our own int8
kernel); T_best = the fastest exact program over every route and α; η̂ = 1 − T_best/T_tensor. Also the note's
η̂_note = 1 − T_lookup/T_fixed, with T_fixed the fastest fixed-address program (tensor-only, DP4A and DP4A+HFMA2
hybrids) and T_lookup the fastest program that uses table lookups. Every row also gives the in-kernel gain of the
best fused lookup hybrid over the same kernel's tensor-only form (own_mma)."""
import json
import sys
from collections import defaultdict

TENSOR = {"int8", "fp8_e4m3", "own_mma"}


def kind(route):
    if route in TENSOR or route == "fused_fr_a0":
        return "tensor"
    if route.endswith("_tensor_part") or route.endswith("_lookup_part"):
        return "part"
    if route.startswith("fused_dp4a") or route.startswith("fused_mix"):
        return "fixed"
    return "lookup"  # fr_b*, hyb_a*, fused_fr_b*


def main():
    rows = []
    for src in sys.argv[1:] or ["-"]:
        for line in open(src) if src != "-" else sys.stdin:
            if '"event":"summary"' in line:
                rows.append(json.loads(line))
    by = defaultdict(dict)
    for r in rows:
        key = (r["n"], r["alphabet"])
        prev = by[key].get(r["route"])
        if prev is None or r["median_ms"] < prev["median_ms"]:
            by[key][r["route"]] = r
    print("| n | S | T_int8 ms | T_tensor (route) | best route | α | b | T_best/T_tensor | η̂ | best lookup / own_mma | "
          "η̂_note | SM clock MHz |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {"k2": 0, "k4": 1, "k6": 2, "k8": 3, "u6": 4}
    for (n, a) in sorted(by, key=lambda k: (k[0], order.get(k[1], 9))):
        d = by[(n, a)]
        full = {k: v for k, v in d.items() if kind(k) != "part"}
        tens = min((v for k, v in full.items() if k in TENSOR), key=lambda v: v["median_ms"])
        best = min(full.values(), key=lambda v: v["median_ms"])
        fixed = [v for k, v in full.items() if kind(k) in ("tensor", "fixed")]
        look = [v for k, v in full.items() if kind(k) == "lookup"]
        tfix = min(v["median_ms"] for v in fixed)
        tlook = min(v["median_ms"] for v in look) if look else float("nan")
        fused_look = [v for k, v in full.items() if k.startswith("fused_fr_b")]
        own = d.get("own_mma")
        inkernel = (min(v["median_ms"] for v in fused_look) / own["median_ms"]) if fused_look and own else float("nan")
        clk = (min(v["clk_min"] for v in full.values()), max(v["clk_max"] for v in full.values()))
        t8 = d["int8"]["median_ms"]
        print(f"| {n} | {a} | {t8:.4f} | {tens['median_ms']:.4f} ({tens['route']}) | {best['route']} | "
              f"{best['alpha']:.4f} | {best['b']} | {best['median_ms'] / tens['median_ms']:.4f} | "
              f"{1 - best['median_ms'] / tens['median_ms']:+.4f} | {inkernel:.4f} | {1 - tlook / tfix:+.4f} | "
              f"{clk[0]}–{clk[1]} |")


if __name__ == "__main__":
    main()
