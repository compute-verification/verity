"""The codec table: each codec's decode cost, encode floor, partial-completion margin, and the γ and step it gives.

Computed from recorded runs, nothing measured here:
- `--attack`: `mask_attack.jsonl` (`run.sh`): the margins of the masked word and of the k = 1 wrap at the floor.
- `--rates`: word_codec's `kv-codec.json` (`run.sh` with KV=1, kv_bench.py): decode, encode and root rates a library
  and k, plain and masked (`-m`), at 512, 768 and 1024 bits.
- `--exhaust` (repeated): word_codec's `kv-exhaust.json` (kv_exhaust.py at 8B's shape): attention's overhead a KV
  byte over the read at mma-ws, B160 × 4k, plain and masked (`-h32`, `-h32m`), k = 2 and 3; p512's is required.

The step model. ra-codec-tensor's step at p512 (`STEP`, its runs) splits d into the plain step, the linears' decode and
receipt (W), attention (A) and the append (E) at k = 1, 2, 3. A codec is placed on that k axis by its decode time a
byte: κ is the p512 round count whose decode (thread-per-word, `rates`) takes as long a byte, piecewise linear in k,
and W and A are read off p512's curves at κ (the slope from 2 to 3 past 3). E scales with the honest encode a byte,
k·x₁. Wherever a codec's attention was measured (`--exhaust`, 8B's shape, the best encoded form), its κ projects
attention from plain p512's best form at k = 2 and 3, and A at both points is scaled by measured over projected (`check`
lists each); such a row is built. The rest (p640, a tensor-core layout at 768, the permutation, the crossed halves) are
projections, and the frontier is given over built rows and over all.

γ_codec = μ_chip 0.28 + μ_sample 0.10 + μ_rebuild + risk, before μ_host, μ_link and λ (`room` is what 1% leaves them):
μ_rebuild = 100·d/(k·x₁,adv), x₁,adv = 175 at p512 (the red team's) scaled by root bytes a second, at the measured
squaring and at the red team's 1.2× and 1.45× ceilings; risk = μ_rebuild·(1/m − 1) at the margin's low end m < 1.

`python table.py --attack A.jsonl --rates kv-codec.json --exhaust kv-exhaust.json [--exhaust ...] [--json OUT]`
"""

from __future__ import annotations

import argparse
import json
import sys

MU_FIXED = 0.28 + 0.10
X1_P512 = 175.0
CEILINGS = (1.0, 1.2, 1.45)
#: μ_host + μ_link + λ: none, the red team's fix (b) at 0.1% and at 0.26%.
ALLOWANCES = (0.0, 0.1, 0.26)

#: ra-codec-tensor's step at p512, in native passes (its report's table): d, W = linears' decode + receipt, A =
#: attention, E = the append, at k = 1, 2, 3.
STEP = {
    "8B B80 ctx 8,256": {"run": "r20261012-175906-b479", "d": (1.207, 1.360, 1.875), "W": (0.086, 0.114, 0.137),
                         "A": (0.022, 0.127, 0.594), "E": (0.026, 0.052, 0.077)},
    "70B B8 ctx 9,280": {"run": "r20261012-175923-0080", "d": (1.506, 1.813, 2.191), "W": (0.390, 0.640, 0.881),
                         "A": (0.011, 0.065, 0.197), "E": (0.007, 0.015, 0.022)},
}

#: The red team's GPU-floor margins for the plain word at k = 2 (centre, low, high; r20261012-170958-f49a), and k = 1's.
PLAIN_K2 = {512: (1.2, 0.75, 2.7), 640: (0.8, 0.50, 1.8), 768: (0.6, 0.37, 1.3), 1024: (0.3, 0.23, 0.84)}


def interp(ys, kappa: float) -> float:
    if kappa <= 2:
        return ys[0] + (ys[1] - ys[0]) * (kappa - 1)
    return ys[1] + (ys[2] - ys[1]) * (kappa - 2)


def frontier(rates: dict) -> dict:
    return {(r["lib"], r["k"]): r for r in rates["frontier_rates"]}


def ns_a_kb(r: dict) -> float:
    return 1e12 / (r["decode_words_per_s"] * r["b"] / 8)


def kappa_of(fr: dict, lib: str, k: int) -> float:
    t = ns_a_kb(fr[(lib, k)])
    t2, t3 = ns_a_kb(fr[("p512", 2)]), ns_a_kb(fr[("p512", 3)])
    return 2 + (t - t2) / (t3 - t2)


def x1_of(fr: dict, b: int) -> float:
    """x₁,adv at b bits: the adversary's root is the fastest any build of the width ran (the mask adds no root)."""
    best = lambda w: max(r["roots_per_s"] * w / 8 for r in fr.values() if r["b"] == w)  # noqa: E731
    return X1_P512 * best(512) / best(b)


def attack_margins(lines: list[dict]) -> dict:
    out: dict = {}
    for d in lines:
        f, e = d["field"], d["experiment"]
        if e == "floor":
            out.setdefault(f, {}).update(peel_floor=d["margin"], wrap_floor=d["wrap_margin"])
        elif e == "copper":
            c = out.setdefault(f, {}).setdefault("copper", {})
            c[d["m"]] = (d["x_over_b"], d["margin"], d["margin_floor"], d["completed"], d["tried"])
        elif e == "quartic":
            out.setdefault(f, {})[f"quartic_{d['mask']}"] = (d["completed"], d["words"])
        elif e == "advice":
            out.setdefault(f, {})["advice_bits"] = (d["stored_bits"], d["bits"])
    return out


def row(name, b, k, *, fr, lib=None, kappa=None, x1=None, margin, note, attn=None, built=None):
    kappa = kappa if kappa is not None else kappa_of(fr, lib, k)
    x1 = x1 if x1 is not None else x1_of(fr, b)
    out = {"codec": name, "b": b, "k": k, "kappa": kappa, "x1_adv": x1, "margin": margin, "note": note, "points": {},
           "built": attn is not None if built is None else built}
    if lib:
        r = fr[(lib, k)]
        out.update(decode_gbps=r["decode_words_per_s"] * b / 8e9, encode_gbps=r["encode_words_per_s"] * b / 8e9,
                   encode_over_floor=r["root_over_squaring_floor"])
    scale = 1.0
    if attn is not None:
        out["attn_b160_4k"] = attn
        scale = attn["measured"] / attn["projected"]
    m_low = margin[1] if margin else None
    for p, s in STEP.items():
        base = s["d"][1] - s["W"][1] - s["A"][1] - s["E"][1]
        e = s["E"][0] * k * x1 / X1_P512
        d = base + interp(s["W"], kappa) + scale * interp(s["A"], kappa) + e
        g = {}
        for c in CEILINGS:
            mu = 100 * d / (k * x1 / c)
            risk = mu * (1 / m_low - 1) if m_low is not None and m_low < 1 else 0.0
            g[c] = {"mu_rebuild": mu, "risk": risk, "gamma": MU_FIXED + mu + risk, "room": 1 - (MU_FIXED + mu + risk)}
        out["points"][p] = {"d": d, "gamma": g}
    return out


def build(attack: list[dict], rates: dict, exhausts: list[dict]) -> dict:
    fr = frontier(rates)
    am = attack_margins(attack)
    best: dict = {}
    for e in exhausts:
        for n, L in e["libs"].items():
            lib = n.split("-")[0] + ("-m" if n.endswith("m") else "")
            for form, ks in L["exhaust"]["b160_ctx4096"]["forms"].items():
                for k, v in ks.items():
                    if k.isdigit() and int(k) > 1 and v["overhead_vs_read"] < best.get((lib, int(k)), (9e9,))[0]:
                        best[(lib, int(k))] = (v["overhead_vs_read"], form)
    a2, a3 = best[("p512", 2)][0], best[("p512", 3)][0]

    def attn(lib, k):
        if (lib, k) not in best:
            return None
        m, form = best[(lib, k)]
        kap = kappa_of(fr, lib, k)
        return {"measured": m, "form": form, "kappa": kap, "projected": a2 + (a3 - a2) * (kap - 2)}

    check = [{"lib": lib, "k": k} | attn(lib, k) for lib, k in sorted(best) if (lib, k) not in (("p512", 2), ("p512", 3))]

    def masked(b):
        f = am[f"p{b}"]
        c = f.get("copper", {})
        free = min((v[2] for v in c.values()), default=None)
        return (f["peel_floor"], 1.0, f["peel_floor"]), (
            f"peel floor {f['peel_floor']:.2f}; quartic {f['quartic_dense'][0]}/{f['quartic_dense'][1]} (plain "
            f"{f['quartic_plain'][0]}/{f['quartic_plain'][1]}); free Coppersmith to b/2 → 1.0"
            + (f", m ≤ 6 floor ≥ {free:.2f}" if free else ""))

    rows = []
    w = am["p512"]["wrap_floor"]
    rows.append(row("p512 k=1, plain or any cheap public wrap (Feistel, ARX, XOR, permutation)", 512, 1, fr=fr,
                    kappa=1.0, x1=X1_P512, margin=(w, w, w), built=True, note="F27; the wrap is absorbed (floor "
                    f"margin {w:.3f} at 512, {am['p2048']['wrap_floor']:.3f} at 2048)"))
    rows.append(row("p512 k=2 plain", 512, 2, fr=fr, lib="p512", kappa=2.0, margin=PLAIN_K2[512],
                    note="red team's 4D floor, assumption 7", attn=attn("p512", 2)))
    m, n = masked(512)
    rows.append(row("p512 k=2 masked (σ_K)", 512, 2, fr=fr, lib="p512-m", margin=m, note=n,
                    attn=attn("p512-m", 2)))
    rows.append(row("p512 k=3 plain", 512, 3, fr=fr, lib="p512", kappa=3.0, margin=None,
                    note="direct degree-8 relation not priced", attn=attn("p512", 3)))
    rows.append(row("p512 k=3 masked", 512, 3, fr=fr, lib="p512-m", margin=(2.0, 1.33, 2.0),
                    note="peel two roots: 2.02/(3·x/b) at b/3, 2/(1.5) at free b/2",
                    attn=attn("p512-m", 3)))
    m, n = masked(640)
    k640 = (kappa_of(fr, "p512-m", 2) + kappa_of(fr, "p768-m", 2)) / 2
    x640 = (x1_of(fr, 512) * x1_of(fr, 768)) ** 0.5
    no_mma = "; kv.cu's tensor-core attention needs a 256-byte head to hold whole words (512, 1024 bits), so this d " \
             "assumes a layout not yet built"
    as_built = "; kv.cu's tensor-core attention needs a 256-byte head to hold whole words (512, 1024 bits), so d " \
               "takes attention as built, on the CUDA cores"
    rows.append(row("p640 k=2 masked (σ_K), interpolated", 640, 2, fr=fr, kappa=k640, x1=x640, margin=m,
                    note=n + "; not built: κ and x₁ midway (x₁ geometric) between masked 512 and 768" + no_mma))
    for b in (768, 1024):
        wide = "" if b % 1024 == 0 else as_built if attn(f"p{b}", 2) else no_mma
        rows.append(row(f"p{b} k=2 plain", b, 2, fr=fr, lib=f"p{b}", margin=PLAIN_K2[b],
                        note="red team's 4D floor" + wide, attn=attn(f"p{b}", 2)))
        m, n = masked(b)
        rows.append(row(f"p{b} k=2 masked (σ_K)", b, 2, fr=fr, lib=f"p{b}-m", margin=m,
                        note=n + ("" if attn(f"p{b}-m", 2) else "; projected step") + wide, attn=attn(f"p{b}-m", 2)))
    if attn("p768-m", 2):
        m, n = masked(768)
        rows.append(row("p768 k=2 masked (σ_K), tensor-core layout (projected)", 768, 2, fr=fr, lib="p768-m",
                        margin=m, note=n + "; d with attention at its κ projection" + no_mma))
    kb = kappa_of(fr, "p512", 2)
    rows.append(row("p512 k=2, bit permutation between rounds", 512, 2, fr=fr, kappa=kb + 1.0, x1=X1_P512,
                    margin=rows[2]["margin"], note="σ_K's margin; a Beneš network ≈ a squaring a boundary (κ + 1)"))
    rows.append(row("two p256 halves crossed, k=2", 512, 2, fr=fr, kappa=1.5, x1=X1_P512 / 2 ** 1.75,
                    margin=None, note="k = 2 at 256 bits: x₁ scales as b^1.75 (measured 1.71-1.78 from 512 to 1024)"))
    return {"rows": rows, "check": check, "attack": {f: {k: v for k, v in d.items() if k != "copper"} | (
        {"copper": {str(m): v for m, v in d.get("copper", {}).items()}}) for f, d in am.items()}}


def render(t: dict) -> str:
    pts = list(STEP)
    head = ["codec", "κ", "decode GB/s", "attn /KV B, measured (projected)", "encode % floor", "x₁,adv",
            "margin (low)"]
    for p in pts:
        head += [f"d {p}", f"γ {p} (1× / 1.2× / 1.45×)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in t["rows"]:
        mg = "—" if not r["margin"] else f"{r['margin'][0]:.2f} ({r['margin'][1]:.2f})"
        cells = [r["codec"], f"{r['kappa']:.2f}", f"{r['decode_gbps']:.0f}" if "decode_gbps" in r else "—",
                 (lambda a: f"{a['measured']:+.3f} {a['form']} ({a['projected']:+.3f})")(r["attn_b160_4k"])
                 if "attn_b160_4k" in r else "—",
                 f"{100 * r['encode_over_floor']:.0f}" if "encode_over_floor" in r else "—", f"{r['x1_adv']:.0f}", mg]
        for p in pts:
            q = r["points"][p]
            cells += [f"{q['d']:.3f}", " / ".join(f"{q['gamma'][c]['gamma']:.2f}" for c in CEILINGS)]
        lines.append("| " + " | ".join(cells) + " |")
    priced = [r for r in t["rows"] if r["margin"]]
    for title, rows in (("built (attention measured)", [r for r in priced if r["built"]]),
                        ("counting the projected rows", priced)):
        lines += ["", f"The fastest codec (least d) within 1% with a priced margin, {title}, given an allowance for "
                  "μ_host + μ_link + λ:", "", "| point | allowance | 1× | 1.2× | 1.45× |", "|---|---|---|---|---|"]
        for p in pts:
            for allow in ALLOWANCES:
                cells = []
                for c in CEILINGS:
                    ok = [r for r in rows if r["points"][p]["gamma"][c]["gamma"] + allow <= 1.0]
                    best = min(ok, key=lambda r: r["points"][p]["d"], default=None)
                    cells.append("none" if best is None else f"{best['codec']} (d {best['points'][p]['d']:.3f})")
                lines.append(f"| {p} | {allow:.2f} | " + " | ".join(cells) + " |")
    lines.append("")
    for c in t["check"]:
        lines.append(f"Check: {c['lib']} k={c['k']} attention, B160 ctx 4k: projected {c['projected']:+.3f} a KV byte "
                     f"(plain p512's best form at κ {c['kappa']:.2f}), measured {c['measured']:+.3f} ({c['form']}); d "
                     "scales the step's attention by the ratio.")
    for r in t["rows"]:
        lines.append(f"- {r['codec']}: {r['note']}")
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--attack", required=True)
    ap.add_argument("--rates", required=True)
    ap.add_argument("--exhaust", action="append", required=True)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    attack = [json.loads(s) for s in open(a.attack) if s.strip()]
    t = build(attack, json.load(open(a.rates)), [json.load(open(p)) for p in a.exhaust])
    print(render(t))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(t, f, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
