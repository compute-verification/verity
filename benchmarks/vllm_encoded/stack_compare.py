"""The stack's gates and points from serve.py's arm-<arm>[-<i>].json in OUT (stack.sh MODE=serve): exits non-zero unless
every stack and pstack arm installed with its gates and every stack arm generated the gate prompts of the first pstack arm
bit for bit (token ids and log-probs), as every later pstack arm did.  Against stock vLLM the gate is reported, not required
(the stack replaces CUTLASS's GEMM and FlashAttention's decode with wt.cu's and paged.cu's, and the lm_head's GEMM is
chunked): the tokens that agree before the first that differs, and the largest log-prob difference over them.

Writes OUT/stack-points.json: for each batch and context, each arm's decode step (serve.py's median, ms) by repeat, and the
ratios stack / stock, stack / pstack and pstack / stock: the mean of the repeats' ratios (repeat i of one arm over repeat i
of the other, run interleaved), with a 95% t-interval over the repeats when there are at least two, and TTFT likewise
where serve.py measured it.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
PAIRS = (("stack", "stock"), ("stack", "pstack"), ("pstack", "stock"))


def arms(out: Path) -> dict[str, list[dict]]:
    by: dict[str, list[tuple[int, dict]]] = {}
    for p in sorted(out.glob("arm-*.json")):
        name = p.stem[len("arm-"):]
        arm, _, i = name.partition("-")
        by.setdefault(arm, []).append((int(i or 0), json.loads(p.read_text())))
    return {a: [r for _, r in sorted(v, key=lambda t: t[0])] for a, v in by.items()}


def against(a: list[dict], b: list[dict]) -> dict:
    """Prompts whose tokens agree throughout, the tokens agreeing before the first difference, and the largest
    |Δ log-prob| over those."""
    same = agree = total = 0
    dlp = 0.0
    for x, y in zip(a, b):
        total += len(x["token_ids"])
        n = 0
        for t, u, lx, ly in zip(x["token_ids"], y["token_ids"], x["logprobs"], y["logprobs"]):
            if t != u:
                break
            dlp = max(dlp, abs(float.fromhex(lx) - float.fromhex(ly)))
            n += 1
        agree += n
        same += x["token_ids"] == y["token_ids"]
    return {"prompts": len(a), "prompts_same_tokens": same, "tokens": total, "tokens_agreeing": agree,
            "max_abs_dlogprob": dlp}


def ratio(num: list[float], den: list[float]) -> dict | None:
    rs = [x / y for x, y in zip(num, den) if x and y]
    if not rs:
        return None
    m = statistics.fmean(rs)
    out = {"mean": m, "n": len(rs)}
    if len(rs) >= 2:
        h = T95.get(len(rs) - 1, 1.96) * statistics.stdev(rs) / math.sqrt(len(rs))
        out["ci95"] = [m - h, m + h]
    return out


def main() -> int:
    out = Path(sys.argv[1])
    by = arms(out)
    fails: list[str] = []
    report: dict = {"arms": {a: len(v) for a, v in by.items()}, "gate": {}, "points": []}
    for a in ("pstack", "stack"):
        for i, r in enumerate(by.get(a, [])):
            g = r.get("install", {}).get("gates")
            if not g or not g.get("layers"):
                fails.append(f"{a}[{i}]: no install gates")
    ref = by.get("pstack", [None])[0]
    if ref is None or not ref.get("gate"):
        fails.append("no pstack gate generation")
    else:
        for a in ("pstack", "stack"):
            for i, r in enumerate(by.get(a, [])):
                if a == "pstack" and i == 0:
                    continue
                ok = r.get("gate") == ref["gate"]
                report["gate"][f"{a}[{i}] == pstack[0]"] = ok
                if not ok:
                    fails.append(f"{a}[{i}]: gate generation differs from pstack[0]'s: {against(r.get('gate', []), ref['gate'])}")
        for i, r in enumerate(by.get("stock", [])):
            report["gate"][f"stock[{i}] vs pstack[0]"] = against(r["gate"], ref["gate"])
        if by.get("stock") and by.get("stack"):
            report["gate"]["stock[0] vs stack[0]"] = against(by["stock"][0]["gate"], by["stack"][0]["gate"])
    keys = sorted({(p["batch"], p["context"]) for v in by.values() for r in v for p in r.get("points", [])})
    for b, ctx in keys:
        row: dict = {"batch": b, "context": ctx, "step_ms": {}, "ttft_ms": {}}
        for a, v in by.items():
            pts = [next((p for p in r.get("points", []) if (p["batch"], p["context"]) == (b, ctx)), {}) for r in v]
            row["step_ms"][a] = [p.get("step_ms") for p in pts]
            row["ttft_ms"][a] = [p.get("ttft_ms") for p in pts]
        for x, y in PAIRS:
            if x in by and y in by:
                row[f"{x}/{y}"] = ratio(row["step_ms"][x], row["step_ms"][y])
                t = ratio(row["ttft_ms"][x], row["ttft_ms"][y])
                if t:
                    row[f"ttft {x}/{y}"] = t
        report["points"].append(row)
        print(json.dumps(row))
    report["fails"] = fails
    (out / "stack-points.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["gate"], indent=1))
    for f in fails:
        print("FAIL", f)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
