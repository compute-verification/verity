"""serve_fill.py's arms side by side: the gates (every arm ran to its end; its greedy tokens and log-probs bit-identical to
the first stock arm's; a green arm's `execute_model` on its green stream only; a filled arm's fold equal to `gemm_pearlc_d`'s) and, per
point, the step time against the stock arms' (their mean, so the two stock runs bracket drift), the step's HBM ratio
(1 / slowdown), the filler's TFLOPS and credited W1/s, and the tensor-core utilization ((the step's FLOPs + the filler's)
over the E4M3 and BF16 peaks at 2,100 MHz).

  python3 compare_fill.py OUT_DIR ARM_JSON...      writes OUT_DIR/compare.json; exit 5 if a gate fails
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PEAK_E4M3, PEAK_BF16 = 807.2e12, 403.7e12


def step_flops(shape: dict, ctx: int) -> tuple[float, float]:
    """A decode token's FP8 linear FLOPs and BF16 attention FLOPs (Llama: QKV, O, gate-up, down, the LM head)."""
    h, i, q, kv, L, v = (shape[k] for k in ("hidden", "intermediate", "heads", "kv_heads", "layers", "vocab"))
    hd = h // q
    lin = h * (q + 2 * kv) * hd + q * hd * h + 2 * h * i + i * h
    return 2.0 * (lin * L + h * v), 2.0 * 2 * q * hd * ctx * L


def main() -> int:
    out, paths = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    arms = [json.loads(p.read_text()) for p in paths if p.exists()]
    stock = [a for a in arms if a["arm"] == "stock"]
    if not stock:
        print("no stock arm", file=sys.stderr)
        return 5
    ref = stock[0]["gate"]
    ok, rows = True, []
    for a in arms:
        g = {"identical": a["gate"] == ref and len(a["gate"]) == len(ref), "complete": a.get("complete") is True}
        if a["arm"] != "stock":
            g["green_stream_only"] = a.get("streams_seen") == [a.get("step_stream")]
        if "filler" in a:
            chk = a["filler"].get("check", {})
            g["filler_bit_exact"] = chk.get("mismatched") == 0 and chk.get("tiles_checked", 0) > 0
        ok &= all(g.values())
        a["gates"] = g
    for a in arms:
        for p in a["points"]:
            if "step_ms" not in p:
                continue
            base = [q["step_ms"] for s in stock for q in s["points"]
                    if q["batch"] == p["batch"] and q["context"] == p["context"] and "step_ms" in q]
            if not base:
                continue
            b = sum(base) / len(base)
            f8, f16 = step_flops(a["model_shape"], p["context"])
            tok = p["batch"] / (p["step_ms"] * 1e-3)
            util_step = tok * (f8 / PEAK_E4M3 + f16 / PEAK_BF16)
            fil = p.get("filler", {})
            rows.append({"arm": a["arm"], "batch": p["batch"], "context": p["context"], "step_ms": p["step_ms"],
                         "p90_ms": p.get("p90_ms"), "stock_ms": b, "stock_spread": (max(base) - min(base)) / b,
                         "slowdown": p["step_ms"] / b, "hbm_ratio": b / p["step_ms"], "util_step": util_step,
                         "util": util_step + fil.get("util", 0.0), "filler_tflops": fil.get("tflops", 0.0),
                         "credit_w1_per_s": fil.get("credit_w1_per_s", 0.0), "gates": a["gates"]})
    res = {"gates_pass": ok, "arms": {a["arm"]: a["gates"] for a in arms}, "points": rows}
    (out / "compare.json").write_text(json.dumps(res, indent=1) + "\n")
    for r in rows:
        print(f"{r['arm']:>14} b{r['batch']:<3} ctx{r['context']:<5} {r['step_ms']:7.3f} ms  x{r['slowdown']:.3f}  "
              f"util {100 * r['util']:5.2f}%  filler {r['filler_tflops']:6.1f} TFLOPS  {r['credit_w1_per_s']:.2e} W1/s")
    print("gates", "pass" if ok else "FAILED", json.dumps(res["arms"]))
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
