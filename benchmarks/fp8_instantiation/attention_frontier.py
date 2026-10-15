"""The frontier of `attention.py`'s results: for each context and family of edits, the edit with the most entropy per
entry within 1% and within 10% of bf16 (and of the same edit without noise), plus the cost of its limbs.

An edit's entropy is the least over its four operands (Q, K, P, V) of the bits per entry of its noise: the operand a
cheating prover would attack.  Per-operand frontiers (noise on one matmul operand group alone) are listed beside it.
Cost: the exact int8 products per matmul, and the extra MACs over one product each as a share of the model's MACs per
token (Llama-3.1-8B: 218.1M linear MACs per layer and token, lm_head 525.3M per token, causal attention
H d (s + 1) per layer and token averaged over a window of s tokens).

    python attention_frontier.py RESULTS.jsonl...
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import attention as A

LAYERS, LINEAR, LM_HEAD, HD = 32, 218_103_808, 525_336_576, 4096


def family(x: dict) -> str | None:
    if x["grid"] != "int8":
        return None
    n = x["n"]
    limbs = ("P,V,Q,K on two limbs" if x["qk2"] else "P,V on two limbs" if x["v"] == "vchan2"
             else "P on two limbs" if x["p"] == "prow2" else "one limb")
    return f"{limbs}: noise on all four" if all(v != "none" for v in n.values()) else None


def alone(x: dict) -> str | None:
    if x["grid"] != "int8":
        return None
    n = x["n"]
    on = [o for o in A.OPERANDS if n[o] != "none"]
    if on == ["q", "k"] and n["q"] == n["k"]:
        group = "Q,K"
    elif on in (["p"], ["v"]):
        group = on[0].upper()
    else:
        return None
    limbs = "two limbs" if {"Q,K": x["qk2"], "P": x["p"] == "prow2", "V": x["v"] == "vchan2"}[group] else "one limb"
    return f"{group} alone, {limbs}"


def min_bits(x: dict) -> float:
    return min(x["info"][o]["bits"] for o in A.OPERANDS)


def products(x: dict) -> dict:
    return {"qk": 4 if x["qk2"] else 1, "pv": (2 if x["p"] == "prow2" else 1) * (2 if x["v"] == "vchan2" else 1)}


def extra_cost(x: dict) -> float:
    """Extra MACs of the edit's limbs as a share of the unedited model's MACs per token."""
    pr = products(x)
    attn = HD * (x["ctx"] + 1) / 2
    total = LAYERS * (LINEAR + 2 * attn) + LM_HEAD
    return LAYERS * attn * ((pr["qk"] - 1) + (pr["pv"] - 1)) / total


def frontier(rows: list[dict]) -> str:
    out = ["| ctx | family | within, of | edit | entropy, bits/entry (min over Q, K, P, V) | perplexity (seeds) | Δ% vs bf16 "
           "| Δ% vs grid | int8 products (QK, PV) | extra MACs, % of model |", "|---:|---|---|---|---:|---:|---:|---:|---|---:|"]
    for ctx in sorted({x["ctx"] for x in rows}):
        fams: dict = {}
        for x in rows:
            if x["ctx"] == ctx and x["d_bf16"] is not None:
                for f, ent in ((family(x), min_bits(x)), (alone(x), max(x["info"][o]["bits"] for o in A.OPERANDS))):
                    if f:
                        fams.setdefault(f, []).append((ent, x))
        for f in sorted(fams):
            for lim, ref in ((1, "bf16"), (10, "bf16"), (1, "grid"), (10, "grid")):
                ok = [(e, x) for e, x in fams[f] if x[f"d_{ref}"] is not None and x[f"d_{ref}"] < lim]
                if not ok:
                    out.append(f"| {ctx} | {f} | {lim}%, {ref} | none | - | - | - | - | - | - |")
                    continue
                e, x = max(ok, key=lambda t: (t[0], sum(t[1]["info"][o]["bits"] for o in A.OPERANDS), -t[1]["d_bf16"]))
                pr = products(x)
                edit = x["grid"] + (", qk2" if x["qk2"] else "") + f", {x['p']}, {x['v']}" + (", rot" if x["rot"] else "")
                dg = "-" if x["d_grid"] is None else f"{x['d_grid']:+.2f}"
                out.append(f"| {ctx} | {f} | {lim}%, {ref} | {edit}; {x['label']} | {e:.3f} | {x['ppl']:.3f} ({x['seeds']}) | "
                           f"{x['d_bf16']:+.2f} | {dg} | {pr['qk']}, {pr['pv']} | {100 * extra_cost(x):.1f} |")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    results = [json.loads(line) for p in argv for line in Path(p).read_text().splitlines() if line.startswith("{")]
    print(frontier(A.rows(results)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
