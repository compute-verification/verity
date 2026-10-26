"""The stacked share f under each randomization, from the cheater's measured squarings (`cheater.py`'s JSON) at the
census's operating points, and the honest side's measured accuracy (`honest.py`'s results), as Markdown tables.

The stacking law (`benchmarks/stacked_share/README.md`): `f = (128·R_cap·L/E + R_in·L) / M_dyn`. A mechanism moves `E`
by the ratio of its cheapest measured squaring to the reference construction of the same family (`anyfn_seam`, 160,
for the restricted constructions; `anyfn_fixup_b8`, the same greedy pass without noise, for the any-function ones), so
`E' = E · U/U_ref` at each `E` the census tabulates (modeled: the conjecture's `e` is assumed to scale as the cheapest
construction does).

A second column bounds the encoder by the node's int8 MAC peak too: words a second `≤ (MAC_peak − honest MACs) /
(MACs a squaring · 3,063)`. Nothing in property/v1 or the device model bounds MACs today (they are free), so that
column holds only if the accounting adds that bound; it is the column where accumulator noise and rotations bite."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

#: The census (art:dc25549e6f14f82f2ece12f1e65054ee48d6008be4b6c9fd97cd606d2fa8f500, share.json `models.*`, as built):
#: R_cap the peak 1 s window's property/v1 units a second, M_dyn vLLM's KV pool, units a token. `linear_macs` is the
#: model's linear weights (its MACs a token, modeled from its config: 8B 32 × 218.1M + 525M LM head, 70B 80 × 855.6M +
#: 1.05G LM head).
CENSUS = {
    "Llama-3.1-8B (8 engines)": {"R_cap": 9072278955625.574, "M_dyn": 609801469952, "units_per_token": 141241088.40276918,
                                 "linear_macs": 7.50e9},
    "Llama-3.1-70B TP8": {"R_cap": 2181007198836.8477, "M_dyn": 608777011200, "units_per_token": 584467165.3898581,
                          "linear_macs": 6.95e10},
}
E_WORD = {"conjectured": 364497, "red team (#2420)": 490080}
LIFETIMES = {"10 min": 600, "1 h": 3600}
#: the census headline's run-input rate (`stacked_share/share.py`'s first `R_INS`), so the `none` row is the census's f
R_IN = 1e5
SQUARINGS_PER_WORD = 3063
GPUS = 8
#: H100 SXM dense int8, 1,979 TOPS (NVIDIA's datasheet; assumed: the census's served trace ran on H100s)
MAC_PEAK_GPU = 1979e12 / 2


def mac_free(model: str) -> float:
    c = CENSUS[model]
    return GPUS * MAC_PEAK_GPU - c["linear_macs"] * c["R_cap"] / c["units_per_token"]


def f(model: str, L: float, E: float, macs_sq: float | None = None) -> float:
    c = CENSUS[model]
    words = c["R_cap"] / E
    if macs_sq is not None:
        words = min(words, mac_free(model) / (macs_sq * SQUARINGS_PER_WORD))
    return (128 * words * L + R_IN * L) / c["M_dyn"]


def fmt_f(x: float) -> str:
    return f"{x:.2f}x" if x >= 1 else f"{100 * x:.1f}%"


def mechanisms(rows: dict) -> list[dict]:
    """Each mechanism's cheapest construction (fewest units), and the one best for the cheater under the MAC bound."""
    def ok(r):
        return "units" in r and r["right"] == r["trials"] and r["proven"] and r["operand_bits_128"]

    seam = rows["anyfn_seam"]["units"]
    fix0 = rows["anyfn_fixup_b8"]["units"]
    out = [
        {"mechanism": "none (property/v1 as is)", "units": seam, "ratio": 1.0, "macs": rows["anyfn_seam"]["macs"],
         "how": "anyfn_seam"},
        {"mechanism": "SR or dither, the format keeping a deterministic integer narrowing (mod, and, byte, I32Div)",
         "units": seam, "ratio": 1.0, "macs": rows["anyfn_seam"]["macs"], "how": "anyfn_seam on those primitives"},
        {"mechanism": "signed permutation (any)", "units": seam, "ratio": 1.0, "macs": rows["anyfn_seam"]["macs"],
         "how": "exact: commutes with every exact dot"},
        {"mechanism": "fresh Hadamard rotation of every dot (block 128)", "units": seam, "ratio": 1.0,
         "macs": rows["anyfn_seam"]["macs"] * 128, "how": "one nonzero a block: exact, 128x the MACs"},
        {"mechanism": "restricted primitives only (det floors, clamps; no randomization)",
         "units": rows["restricted_det_b8"]["units"], "ratio": rows["restricted_det_b8"]["units"] / seam,
         "macs": rows["restricted_det_b8"]["macs"], "how": "restricted_det_b8"},
        {"mechanism": "SR or dither on every rounding, restricted primitives", "units": rows["restricted_sr_b7"]["units"],
         "ratio": rows["restricted_sr_b7"]["units"] / seam, "macs": rows["restricted_sr_b7"]["macs"],
         "how": "restricted_sr_b7"},
        {"mechanism": "±1 LSB noise on every rounding, restricted primitives",
         "units": rows["restricted_noise1_b6"]["units"], "ratio": rows["restricted_noise1_b6"]["units"] / seam,
         "macs": rows["restricted_noise1_b6"]["macs"], "how": "restricted_noise1_b6"},
    ]
    for g in sorted({r["noise_g"] for r in rows.values() if r.get("noise_g")}):
        for fam, ref, label in (("anyfn_fixup", fix0, f"accumulator noise 2^{g} (deterministic roundings)"),
                                ("restricted_acc_sr", seam, f"accumulator noise 2^{g} + SR, restricted")):
            cands = [r for n, r in rows.items() if n.startswith(fam) and r.get("noise_g") == g and ok(r)]
            if not cands:
                continue
            best = min(cands, key=lambda r: r["units"])
            out.append({"mechanism": label, "units": best["units"], "ratio": best["units"] / ref,
                        "macs": best["macs"], "how": best["name"], "candidates": [
                            {"how": r["name"], "ratio": r["units"] / ref, "macs": r["macs"]} for r in cands]})
    return out


def table(mechs: list[dict]) -> str:
    head = ["mechanism", "cheapest squaring", "units (x)", "MACs a squaring"]
    for m in CENSUS:
        for L in LIFETIMES:
            head.append(f"{m.split(' ')[0]} {L}")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for x in mechs:
        cells = [x["mechanism"], x["how"], f"{x['units']} ({x['ratio']:.2f}x)", f"{x['macs']:,}"]
        for m in CENSUS:
            for L, s in LIFETIMES.items():
                E = E_WORD["conjectured"] * x["ratio"]
                fu = f(m, s, E)
                # under the MAC bound the cheater picks its best candidate, not the fewest units
                cands = x.get("candidates") or [{"ratio": x["ratio"], "macs": x["macs"]}]
                fm = max(f(m, s, E_WORD["conjectured"] * c["ratio"], c["macs"]) for c in cands)
                cells.append(f"{fmt_f(fu)} / {fmt_f(fm)}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def honest(results: Path) -> str:
    """Per arm: perplexity (mean over seeds, and its range), its change from `rtn`, MMLU and GSM8K accuracy and their
    paired change from `rtn` (questions right under one and not the other)."""
    recs = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
    ppl: dict = {}
    task: dict = {}
    for r in recs:
        if r["kind"] == "ppl":
            ppl.setdefault((r["set"], r["arm"]), []).append(r["ppl"])
        elif r["kind"] in ("mmlu", "gsm8k"):
            task[(r["kind"], r["arm"])] = r
    sets = sorted({s for s, _ in ppl})
    arms = list(dict.fromkeys([a for _, a in ppl] + [a for _, a in task]))
    head = ["arm"] + [f"{s} ppl (Δ vs rtn)" for s in sets] + ["MMLU (Δ, n)", "GSM8K (Δ, n)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for arm in arms:
        cells = [arm]
        for s in sets:
            v = ppl.get((s, arm))
            ref = ppl.get((s, "rtn"))
            if not v:
                cells.append("")
                continue
            mean = sum(v) / len(v)
            rng = f" [{min(v):.4f}, {max(v):.4f}]" if len(v) > 1 else ""
            d = f" ({100 * (mean / (sum(ref) / len(ref)) - 1):+.2f}%)" if ref else ""
            cells.append(f"{mean:.4f}{rng}{d}")
        for kind in ("mmlu", "gsm8k"):
            r, ref = task.get((kind, arm)), task.get((kind, "rtn"))
            if not r:
                cells.append("")
                continue
            d = ""
            if ref:
                gain = sum(a > b for a, b in zip(r["right"], ref["right"]))
                loss = sum(a < b for a, b in zip(r["right"], ref["right"]))
                d = f" ({100 * (r['acc'] - ref['acc']):+.1f} pt, +{gain}/−{loss})"
            cells.append(f"{100 * r['acc']:.1f}%{d}, {r['n']}")
        lines.append("| " + " | ".join(cells) + " |")
    stats = [r for r in recs if r["kind"] == "ppl" and "log2_rms_acc" in r]
    if stats:
        s = stats[0]
        lines.append("")
        lines.append("RTN's accumulators, log2 of the rms, median over dots (p10, p90): " + ", ".join(
            f"{k} {v:.1f} ({s['log2_rms_acc_p10_p90'][k][0]:.1f}, {s['log2_rms_acc_p10_p90'][k][1]:.1f})"
            for k, v in s["log2_rms_acc"].items()))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cheater", type=Path, help="cheater.py's JSON")
    ap.add_argument("--honest", type=Path, nargs="*", default=[], help="honest.py's results.jsonl files")
    ap.add_argument("--json", type=Path, help="write the mechanisms and f values here too")
    a = ap.parse_args()
    if a.cheater:
        rows = json.loads(a.cheater.read_text())["constructions"]
        mechs = mechanisms(rows)
        print("f at E conjectured x the mechanism's units ratio: units only / with the MAC-peak bound\n")
        print(table(mechs))
        if a.json:
            out = []
            for x in mechs:
                y = dict(x)
                y["f"] = {m: {L: {"units": f(m, s, E_WORD["conjectured"] * x["ratio"]),
                                  "mac_bound": max(f(m, s, E_WORD["conjectured"] * c["ratio"], c["macs"])
                                                   for c in (x.get("candidates") or [x]))}
                              for L, s in LIFETIMES.items()} for m in CENSUS}
                out.append(y)
            a.json.write_text(json.dumps({"mechanisms": out, "mac_free": {m: mac_free(m) for m in CENSUS}}, indent=1))
    for h in a.honest:
        print(f"\n{h}\n")
        print(honest(h))


if __name__ == "__main__":
    main()
