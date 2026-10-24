"""coverage(k), phi and eta(k, c) from the units files ``sessions.py`` writes.

    python measure.py UNITS_DIR --out OUT     # OUT/tables.md and OUT/tables.json

- **Units.** Rules 1 and 2 count per call (each request is its own block); rule 3 counts per session (the session's
  earlier turns are inside its block).
- **Unregistered tokens per unit.** Rule 1: the whole context. Rule 2: context minus the registered prefix. Rule 3: the
  session's new user and tool tokens (its first call: context minus registered prefix).
- **FLOPs per unit** (``flops.py``). Rule 1: prefill of the whole context plus decode. Rule 2: prefill after the registered
  prefix (attention still reads it) plus decode. Rule 3: per call, prefill of its new tokens plus decode, summed over the
  session.
- **coverage(k)**: fraction of units whose unregistered tokens times bits per token are at most k bits.
- **phi**: total FLOPs over total unregistered bits (traffic-weighted, per source); a class averages its sources with equal
  weight, every source under every rule (``eff``: a source without prefix or session structure counts under the rule
  below, marked * in the tables). **eta(k, c)** = c / (k phi).
- **c matched to typical traffic**: c = k phi_typ, with phi_typ the chat and API classes' phi pooled (equal weight per
  source), so eta = phi_typ / phi for each class at every k.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flops  # noqa: E402
import units  # noqa: E402

KS = [("1 kB", 1e3), ("10 kB", 1e4), ("100 kB", 1e5), ("1 MB", 1e6), ("10 MB", 1e7)]
BITS = {"raw": 17.0, "coded": 3.0}
C_FIXED = 1.4e17
CLASSES = ["chat", "api", "reasoning", "agentic", "rag"]
RULES = {"R1": "recompute", "R2": "registered prefixes", "R3": "per-session blocks"}


def per_unit(d: dict[str, np.ndarray], rule: str) -> dict[str, np.ndarray]:
    ctx, reg, new, out, sess = d["ctx"], d["reg"], d["new"], d["out"], d["sess"]
    if rule == "R1":
        tok, start = ctx, np.zeros_like(ctx)
    elif rule == "R2":
        tok, start = ctx - reg, reg
    else:
        tok, start = new, ctx - new
    f = {m: flops.call_flops(m, start, ctx, out) for m in flops.MODELS}
    if rule != "R3":
        return {"tok": tok.astype(np.float64), **f}
    _, inv = np.unique(sess, return_inverse=True)
    return {"tok": np.bincount(inv, weights=tok.astype(np.float64)),
            **{m: np.bincount(inv, weights=v) for m, v in f.items()}}


def summarize(u: dict[str, np.ndarray]) -> dict:
    tok = u["tok"]
    fit = {b: {kn: tok * bits <= 8 * kv for kn, kv in KS} for b, bits in BITS.items()}
    # the units at k's scale: unregistered bits in (k/10, k]
    band = {b: {kn: fit[b][kn] & (tok * bits > 0.8 * kv) for kn, kv in KS} for b, bits in BITS.items()}
    s = {"units": int(len(tok)), "tok_mean": float(tok.mean()),
         "tok_q": {p: float(np.percentile(tok, p)) for p in (50, 90, 99)}, "tok_max": float(tok.max()),
         "cov": {b: {kn: float(fit[b][kn].mean()) for kn, _ in KS} for b in BITS},
         "cov_flops": {m: {b: {kn: float(u[m][fit[b][kn]].sum() / u[m].sum()) for kn, _ in KS} for b in BITS}
                       for m in flops.MODELS},
         "phi_band": {m: {b: {kn: (float(u[m][band[b][kn]].sum() / (tok[band[b][kn]].sum() * BITS[b]))
                                   if band[b][kn].any() else None) for kn, _ in KS} for b in BITS}
                      for m in flops.MODELS},
         "band_units": {b: {kn: int(band[b][kn].sum()) for kn, _ in KS} for b in BITS},
         "flops_p99_fit": {m: {b: {kn: (float(np.percentile(u[m][fit[b][kn]], 99)) if fit[b][kn].any() else None)
                                   for kn, _ in KS} for b in BITS} for m in flops.MODELS},
         "flops_mean": {m: float(u[m].mean()) for m in flops.MODELS},
         "fits_c": {m: float((u[m] <= C_FIXED).mean()) for m in flops.MODELS}}
    return s


def phi(s: dict, model: str, b: str) -> float:
    return s["flops_mean"][model] / (s["tok_mean"] * BITS[b]) if s["tok_mean"] else float("inf")


def pooled(sums: list[dict], model: str, b: str) -> float:
    return np.mean([s["flops_mean"][model] for s in sums]) / (np.mean([s["tok_mean"] for s in sums]) * BITS[b])


def eff(v: dict, rule: str) -> dict:
    """A source's summary under ``rule``, or under the nearest rule below it that the source can speak to: without prefix
    structure nothing is known to be registered (R2 = R1), and without sessions each request is its own (R3 = R2). Both
    fallbacks can only overstate unregistered input."""
    order = list(RULES)
    for r in reversed(order[: order.index(rule) + 1]):
        if r in v["rules"]:
            return v["rules"][r]
    raise KeyError(rule)


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("units_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    src: dict[str, dict] = {}
    raw: dict[str, dict] = {}
    for p in sorted(args.units_dir.glob("*.npz")):
        meta, d = units.load(p)
        raw[meta["src"]] = {r: per_unit(d, r) for r in meta["rules"]}
        src[meta["src"]] = {"meta": meta, "rules": {r: summarize(u) for r, u in raw[meta["src"]].items()}}
    cls: dict[str, dict] = {}
    for c in CLASSES:
        cls[c] = {}
        for r in RULES:
            sums = [eff(v, r) for v in src.values() if v["meta"]["cls"] == c]
            if not sums:
                continue
            cls[c][r] = {"sources": [k + ("" if r in v["rules"] else "*") for k, v in src.items()
                                     if v["meta"]["cls"] == c],
                         "cov": {b: {kn: float(np.mean([s["cov"][b][kn] for s in sums])) for kn, _ in KS}
                                 for b in BITS},
                         "cov_flops": {m: {b: {kn: float(np.mean([s["cov_flops"][m][b][kn] for s in sums]))
                                               for kn, _ in KS} for b in BITS} for m in flops.MODELS},
                         "phi": {m: {b: float(pooled(sums, m, b)) for b in BITS} for m in flops.MODELS},
                         "phi_band": {m: {b: {kn: _mean([s["phi_band"][m][b][kn] for s in sums]) for kn, _ in KS}
                                          for b in BITS} for m in flops.MODELS},
                         "fits_c": {m: float(np.mean([s["fits_c"][m] for s in sums])) for m in flops.MODELS}}
    typ, c_p99 = {}, {}
    for r in RULES:
        sums = [eff(v, r) for v in src.values() if v["meta"]["cls"] in ("chat", "api")]
        typ[r] = {m: {b: float(pooled(sums, m, b)) for b in BITS} for m in flops.MODELS} if sums else None
        c_p99[r] = {m: {b: {kn: max((s["flops_p99_fit"][m][b][kn] or 0.0) for s in sums) for kn, _ in KS}
                        for b in BITS} for m in flops.MODELS} if sums else None
    # coverage when c is matched to typical traffic: a unit fits when its bits are at most k and its FLOPs at most k phi_typ
    for c, rr in cls.items():
        for r, v in rr.items():
            v["cov_matched"] = {m: {b: {kn: float(np.mean([
                ((u["tok"] * BITS[b] <= 8 * kv) & (u[m] <= 8 * kv * typ[r][m][b])).mean()
                for k2, w in src.items() if w["meta"]["cls"] == c
                for u in [raw[k2][next(rx for rx in reversed(list(RULES)[: list(RULES).index(r) + 1])
                                      if rx in raw[k2])]]])) for kn, kv in KS} for b in BITS} for m in flops.MODELS}
    result = {"sources": src, "classes": cls, "phi_typ": typ, "c_p99_typ": c_p99, "c_fixed": C_FIXED, "ks": dict(KS),
              "bits": BITS}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tables.json").write_text(json.dumps(result, indent=1))
    (args.out / "tables.md").write_text(render(result))
    print(render(result))


def _pct(x: float) -> str:
    return f"{100 * x:.0f}" if x >= 0.995 or x <= 0.005 or 0.1 <= x <= 0.9 else f"{100 * x:.1f}"


def render(res: dict) -> str:
    src, cls, typ = res["sources"], res["classes"], res["phi_typ"]
    out = []
    kh = " | ".join(kn for kn, _ in KS)
    for b, bits in BITS.items():
        out.append(f"\n### coverage(k), % of units, {bits:g} bits per token ({b})\n")
        out.append(f"| class | rule | {kh} | sources |")
        out.append("|---|---|" + "---|" * len(KS) + "---|")
        for c, rr in cls.items():
            for r, v in rr.items():
                out.append(f"| {c} | {r} | " + " | ".join(_pct(v["cov"][b][kn]) for kn, _ in KS)
                           + f" | {', '.join(v['sources'])} |")
    out.append("\n### coverage(k) by source, % of units, 17 bits per token (3 bits per token in brackets)\n")
    out.append(f"| source | class | rule | units | p50 / p90 / p99 / max tokens | {kh} |")
    out.append("|---|---|---|---|---|" + "---|" * len(KS))
    for k, v in src.items():
        for r, s in v["rules"].items():
            q = s["tok_q"]
            out.append(f"| {k} | {v['meta']['cls']} | {r} | {s['units']} | {q[50]:.0f} / {q[90]:.0f} / {q[99]:.0f} / "
                       f"{s['tok_max']:.0f} | " + " | ".join(
                           f"{_pct(s['cov']['raw'][kn])} ({_pct(s['cov']['coded'][kn])})" for kn, _ in KS) + " |")
    out.append("\n### phi, FLOPs per unregistered input bit, 17 bits per token (x 17/3 = 5.67 at 3 bits per token)\n")
    out.append("| class | rule | DSV4-Pro-like | 7B dense | units within c = 1.4e17 (DSV4) |")
    out.append("|---|---|---|---|---|")
    for c, rr in cls.items():
        for r, v in rr.items():
            out.append(f"| {c} | {r} | {v['phi']['dsv4']['raw']:.2e} | {v['phi']['dense7b']['raw']:.2e} | "
                       f"{_pct(v['fits_c']['dsv4'])}% |")
    for r, t in typ.items():
        if t:
            out.append(f"| typical (chat + API) | {r} | {t['dsv4']['raw']:.2e} | {t['dense7b']['raw']:.2e} | |")
    p99 = res["c_p99_typ"]
    f = lambda x: "-" if x is None else f"{x:.2e}"
    for model in flops.MODELS:
        out.append(f"\n### phi at k's scale (units with unregistered input in (k/10, k]), {flops.MODELS[model]['label']}, "
                   "17 bits per token\n")
        out.append(f"| class | rule | {kh} |")
        out.append("|---|---|" + "---|" * len(KS))
        for c, rr in cls.items():
            for r, v in rr.items():
                out.append(f"| {c} | {r} | " + " | ".join(f(v["phi_band"][model]["raw"][kn]) for kn, _ in KS) + " |")
        out.append(f"\n### coverage(k, c = k phi_typ), % of units within both k bits and c FLOPs, "
                   f"{flops.MODELS[model]['label']}, 17 bits per token\n")
        out.append(f"| class | rule | {kh} |")
        out.append("|---|---|" + "---|" * len(KS))
        for c, rr in cls.items():
            for r, v in rr.items():
                out.append(f"| {c} | {r} | " + " | ".join(_pct(v["cov_matched"][model]["raw"][kn]) for kn, _ in KS)
                           + " |")
        out.append(f"\n### FLOP-weighted coverage(k), % of honest FLOPs in units that fit, {flops.MODELS[model]['label']}, "
                   "17 bits per token\n")
        out.append(f"| class | rule | {kh} |")
        out.append("|---|---|" + "---|" * len(KS))
        for c, rr in cls.items():
            for r, v in rr.items():
                out.append(f"| {c} | {r} | " + " | ".join(_pct(v["cov_flops"][model]["raw"][kn]) for kn, _ in KS) + " |")
        for b, bits in BITS.items():
            out.append(f"\n### eta(k, c = 1.4e17) = c / (k phi), {flops.MODELS[model]['label']}, {bits:g} bits per token\n")
            out.append(f"| class | rule | {kh} | eta at c = k phi_typ |")
            out.append("|---|---|" + "---|" * len(KS) + "---|")
            for c, rr in cls.items():
                for r, v in rr.items():
                    p = v["phi"][model][b]
                    em = typ[r][model][b] / p if typ.get(r) else float("nan")
                    out.append(f"| {c} | {r} | " + " | ".join(f"{C_FIXED / (8 * kv * p):.3g}" for _, kv in KS)
                               + f" | {em:.2f} |")
            out.append(f"\n### eta(k, c = c99(k)), c99 = the 99th-percentile FLOPs of chat and API units that fit in k "
                       f"(the most of any one source), {flops.MODELS[model]['label']}, {bits:g} bits per token\n")
            out.append(f"| class | rule | {kh} |")
            out.append("|---|---|" + "---|" * len(KS))
            for r, t in p99.items():
                if t:
                    out.append(f"| c99 | {r} | " + " | ".join(f"{t[model][b][kn]:.2e}" for kn, _ in KS) + " |")
            for c, rr in cls.items():
                for r, v in rr.items():
                    if not p99.get(r):
                        continue
                    p = v["phi"][model][b]
                    out.append(f"| {c} | {r} | " + " | ".join(f"{p99[r][model][b][kn] / (8 * kv * p):.3g}"
                                                             for kn, kv in KS) + " |")
            cm = " ; ".join(f"{r}: {8 * 1e4 * t[model][b]:.2e}" for r, t in typ.items() if t)
            out.append(f"\nc = k phi_typ at 10 kB (it scales with k): {cm}\n")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
