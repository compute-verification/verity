"""The audited stack's overhead over stock serving, by term, from `kv_filler/step_profile.py`'s `step_profile.json` of a
stock run and of encoded-stack runs at the same shapes: each step's kernel time split into the weights (the linears'
GEMMs, with the stack's codec decode, split-K reduce and chunk decode), attention (its KV reads, decoded inside the
kernel when encoded), the KV encode (vLLM's cache write, the stack's append), receipts and the rest; each term's added
time over stock; and attention's KV payload rate against `--R`. Writes `budget.json` under `--out`.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TERMS = ("weights", "attention", "kv_encode", "receipts", "other")
RULES = (
    ("kv_encode", re.compile(r"reshape_and_cache|pg_append")),
    ("receipts", re.compile(r"\brb::|pg_receipt|rc_a_kernel|sketch")),
    ("attention", re.compile(r"flash_fwd|pg_attn|pg_combine|pg_gather")),
    ("weights", re.compile(r"cutlass|gemv|gemm|wt::|pg_reduce|_int_mm|cublas", re.IGNORECASE)),
)


def term(name: str) -> str:
    return next((t for t, rx in RULES if rx.search(name)), "other")


def split(kernels: list[dict]) -> dict[str, float]:
    ms = dict.fromkeys(TERMS, 0.0)
    for k in kernels:
        ms[term(k["name"])] += k["ms"]
    return ms


def label(p: dict) -> str:
    a = p["args"]
    return f"{a.get('stack') or 'stock'}" + (f" k{a['stack_k']}" if a.get("stack") else "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stock", type=Path, help="the stock run's step_profile.json")
    ap.add_argument("encoded", type=Path, nargs="+", help="encoded-stack runs' step_profile.json")
    ap.add_argument("--R", type=float, default=1539.0, help="GB/s, the card's measured read rate")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    stock = json.loads(a.stock.read_text())
    base = {(s["batch"], s["ctx"]): s for s in stock["shapes"]}
    base_pre = {(p["chunk"], p["context"]): p for p in stock.get("prefill", [])}
    res: dict = {"format": "verity-serve-bandwidth/budget/v0", "R_GBps": a.R, "stock": str(a.stock), "runs": []}
    for path in a.encoded:
        enc = json.loads(path.read_text())
        run: dict = {"label": label(enc), "file": str(path), "decode": [], "prefill": []}
        for s in enc["shapes"]:
            b = base.get((s["batch"], s["ctx"]))
            if b is None:
                continue
            ms0, ms1 = split(b["kernels"]), split(s["kernels"])
            kv_gb = s["kv_bytes_per_step"] / 1e9
            run["decode"].append({
                "batch": s["batch"], "ctx": s["ctx"],
                "stock_kernel_ms": round(sum(ms0.values()), 3), "kernel_ms": round(sum(ms1.values()), 3),
                "stock_span_ms": round(b["span_ms"], 3), "span_ms": round(s["span_ms"], 3),
                "ms": {t: round(ms1[t], 3) for t in TERMS},
                "added_ms": {t: round(ms1[t] - ms0[t], 3) for t in TERMS},
                "attention_over_stock": round(ms1["attention"] / ms0["attention"], 3) if ms0["attention"] else None,
                "attention_kv_of_R": round(kv_gb / (a.R * ms1["attention"] / 1e3), 3) if ms1["attention"] else None,
                "stock_attention_kv_of_R": round(kv_gb / (a.R * ms0["attention"] / 1e3), 3) if ms0["attention"] else None,
            })
        for p in enc.get("prefill", []):
            b = base_pre.get((p["chunk"], p["context"]))
            if b is None:
                continue
            ms0, ms1 = split(b["kernels"]), split(p["kernels"])
            run["prefill"].append({
                "chunk": p["chunk"], "context": p["context"],
                "stock_kernel_ms": round(sum(ms0.values()), 3), "kernel_ms": round(sum(ms1.values()), 3),
                "stock_wall_ms": round(b["wall_ms"], 3), "wall_ms": round(p["wall_ms"], 3),
                "ms": {t: round(ms1[t], 3) for t in TERMS},
                "added_ms": {t: round(ms1[t] - ms0[t], 3) for t in TERMS},
            })
        res["runs"].append(run)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "budget.json").write_text(json.dumps(res, indent=1))
    for run in res["runs"]:
        for d in run["decode"]:
            add = " ".join(f"{t} {v:+.2f}" for t, v in d["added_ms"].items())
            print(f"{run['label']:10s} decode B{d['batch']} ctx{d['ctx']}: kernels {d['stock_kernel_ms']:.2f} -> "
                  f"{d['kernel_ms']:.2f} ms ({add}); attention x{d['attention_over_stock']}, KV at "
                  f"{d['attention_kv_of_R']} R (stock {d['stock_attention_kv_of_R']} R)")
        for p in run["prefill"]:
            add = " ".join(f"{t} {v:+.2f}" for t, v in p["added_ms"].items())
            print(f"{run['label']:10s} prefill {p['chunk']}@{p['context']}: kernels {p['stock_kernel_ms']:.1f} -> "
                  f"{p['kernel_ms']:.1f} ms ({add})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
