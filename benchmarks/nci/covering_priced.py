"""Kimi K2's compute dampener rho under the covering bound's forms, at bf16, FP8 and FP4, with a price per weight.

    python benchmarks/nci/covering_priced.py [--prices FILE.json] [--json OUT.json]

The forms, each a lower bound rho >= (chained charge per gate) * s / (charge one inference block can cover):

* ``one c`` (``Nci.Guarantees.CoveringBound``, on main): one cost ``c = r b`` for a weight and an activation, per block
  ``(k/c)^2 + k``. ``c`` is at most the cheapest updated weight's cost, since parameter-hardness holds for every block.
* ``two c's`` (``Nci.Guarantees.CoveringBoundTwoCosts``, on main): ``c_w`` per weight (the cheapest class again),
  ``c_a`` per activation, per block ``(k/c_w)(k/c_a) + k``.
* ``priced`` (in progress): weight ``i`` costs ``p_i`` bits, ``sum_{i in theta'_S} p_i <= |in(S) \\ theta|``. A block
  covers chained products of price sum at most ``k * (k/c_a) + k * p_max``, so the numerator is the price sum over the
  chained products: the product-weighted mean price replaces the cheapest class's ``c_w``.
* ``priced additive``: the priced form with the weights' bits paid outside the token computations (the conditional
  premise of ``Nci.Cover.Guarantees.CoveringBoundAdditive``, PR #3039), per block ``k^2/(4 c_a) + k p_max``.

Chaining (``chainedBlocksAt``) is decided per product stage of one token's forward pass from the count ``n`` of distinct
updated weights upstream inside the token's computation (``depth.py`` of the realism note: routed experts and the router
only, the token's own key and value, one coordinate at a time). A count form chains a stage iff
``n >= floor(k/c_w) + ceil(c_a)``. The priced form chains it either by price mass, ``sum p >= k + ceil(c_a) p_max``
(variant ``A``, what a priced proof needs), or by count at ``p_min D > k`` (variant ``B``, the two-cost hypothesis with
``c_w = p_min``).

Conventions, as the realism note's (``note:training-20261022T1045Z-cover-assumptions-realism``):

* ``s = 4.52e16`` and ``k`` in bits (1 MB = 8e6), ``c_a = b`` (16 / 8 / 4);
* ``M*`` holds one block per (updated weight, token): the forward products of kappa's ``N`` (every matrix, the router
  and the experts; not the LM head or the norm gammas), ``alpha0 = N / gates`` over K2's schedule (``kappa.py``).
  ``--dgrad`` adds the backward products into activations, ``W^T delta``, and the LM head's;
* prices are per weight class; a block that picks the cheapest weights is what the uniform forms' ``c_w`` prices.
  Pythia's (``PYTHIA``) map onto K2's matrices by role (``ROLE``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import kappa as K  # noqa: E402

S = 4.52e16
KS = {"1 MB": 8e6, "100 MB": 8e8, "1 GB": 8e9}
#: weight and activation width b per precision; c_a = b
FMTS = {"bf16": 16, "fp8": 8, "fp4": 4}

#: K2's weight classes: (label, role, in kappa's N)
CLASSES = {
    "mla_in": ("MLA q_a, q_b, kv_a, kv_b", "qkv", True),
    "mla_out": ("MLA o_proj", "attn_out", True),
    "dense_in": ("dense MLP gate, up (layer 0)", "mlp_in", True),
    "dense_out": ("dense MLP down (layer 0)", "mlp_out", True),
    "expert_in": ("routed experts gate, up", "mlp_in", True),
    "expert_out": ("routed experts down", "mlp_out", True),
    "shared_in": ("shared expert gate, up", "mlp_in", True),
    "shared_out": ("shared expert down", "mlp_out", True),
    "router": ("router", "router", True),
    "head": ("LM head", "head", False),
    "norm": ("norm gammas", "norm", False),
    "embed": ("embedding (a lookup)", "embed", False),
}

#: Pythia's per-tensor-type prices at 1,000 steps after theta, bits per weight: the lowest codec length over every
#: codec (theta's forward codes, and theta's fp32 master and Adam moments), as `rweights/ckpt_deltas.py` measured them
#: (art:044580cc14c18dcb55b1e69da10d0e0f3f6ef1321ed11375c50953a1af8dc7d2, `rweights/ckpt_160m_late.jsonl` and
#: `ckpt_410m.jsonl`). qkv: query_key_value; attn_out: dense; mlp_in: dense_h_to_4h; mlp_out: dense_4h_to_h; head:
#: embed_out; embed: embed_in. The forward-codes-only values differ only at FP4 (about 0.2 bits higher).
PYTHIA = {
    "160m@100k": {
        "bf16": {"qkv": 6.694, "attn_out": 7.314, "mlp_in": 6.667, "mlp_out": 6.665, "head": 5.912, "embed": 7.268},
        "fp8": {"qkv": 2.767, "attn_out": 3.286, "mlp_in": 2.718, "mlp_out": 2.733, "head": 2.095, "embed": 3.161},
        "fp4": {"qkv": 0.695, "attn_out": 0.972, "mlp_in": 0.663, "mlp_out": 0.662, "head": 0.540, "embed": 0.931},
    },
    "160m@140k": {
        "bf16": {"qkv": 5.817, "attn_out": 6.750, "mlp_in": 5.988, "mlp_out": 5.955, "head": 1.583, "embed": 6.474},
        "fp8": {"qkv": 2.078, "attn_out": 2.813, "mlp_in": 2.137, "mlp_out": 2.137, "head": 0.264, "embed": 2.501},
        "fp4": {"qkv": 0.418, "attn_out": 0.717, "mlp_in": 0.431, "mlp_out": 0.433, "head": 0.127, "embed": 0.590},
    },
    "410m@100k": {
        "bf16": {"qkv": 6.124, "attn_out": 6.338, "mlp_in": 6.194, "mlp_out": 6.028, "head": 6.359, "embed": 6.531},
        "fp8": {"qkv": 2.275, "attn_out": 2.438, "mlp_in": 2.284, "mlp_out": 2.169, "head": 2.412, "embed": 2.543},
        "fp4": {"qkv": 0.486, "attn_out": 0.548, "mlp_in": 0.488, "mlp_out": 0.440, "head": 0.559, "embed": 0.610},
    },
    "410m@140k": {
        "bf16": {"qkv": 5.040, "attn_out": 5.483, "mlp_in": 5.228, "mlp_out": 5.037, "head": 5.357, "embed": 5.571},
        "fp8": {"qkv": 1.492, "attn_out": 1.770, "mlp_in": 1.569, "mlp_out": 1.460, "head": 1.674, "embed": 1.811},
        "fp4": {"qkv": 0.249, "attn_out": 0.322, "mlp_in": 0.263, "mlp_out": 0.234, "head": 0.298, "embed": 0.335},
    },
}

#: the uniform forms' c_w: the mean cost of the cheapest theta-defined classes (theta's code x its Adam-v decile) of the
#: transformer-block matrices holding 1% (and 0.1%) of the weights, `rweights/classes.out` (Pythia-160m only)
CHEAPEST = {
    "160m@100k": {"1%": {"bf16": 3.425, "fp8": 0.741, "fp4": 0.188},
                  "0.1%": {"bf16": 2.786, "fp8": 0.574, "fp4": 0.101}},
    "160m@140k": {"1%": {"bf16": 2.265, "fp8": 0.453, "fp4": 0.121},
                  "0.1%": {"bf16": 1.540, "fp8": 0.314, "fp4": 0.075}},
}

#: HF tensor names a price file may use, onto the roles (an equal-weight mean where several share one)
HF_ROLES = {"q_proj": "qkv", "k_proj": "qkv", "v_proj": "qkv", "o_proj": "attn_out", "gate_proj": "mlp_in",
            "up_proj": "mlp_in", "down_proj": "mlp_out", "lm_head": "head", "embed_tokens": "embed"}


def k2() -> dict:
    return K.MODELS["kimi-k2"]


def shapes(m: dict) -> dict[str, tuple[float, float, float]]:
    """Per class: (weights, forward products per token, copies of each weight per token)."""
    d, a, L, Lm = m["d"], m["attention"], m["layers"], m["layers"] - m["dense_layers"]
    h, eff = a["heads"], m["expert_ffn"]
    mla_in = a["q_rank"] * d + h * (a["nope"] + a["rope"]) * a["q_rank"] + (a["kv_rank"] + a["rope"]) * d \
        + h * (a["nope"] + a["v"]) * a["kv_rank"]
    mla_out = d * h * a["v"]
    expert = eff * d
    norms = L * (2 * d + a["q_rank"] + a["kv_rank"]) + d
    head = m["vocab"] * d
    use = m["active"] / m["experts"]
    return {
        "mla_in": (L * mla_in, L * mla_in, 1),
        "mla_out": (L * mla_out, L * mla_out, 1),
        "dense_in": (m["dense_layers"] * 2 * m["ffn"] * d, m["dense_layers"] * 2 * m["ffn"] * d, 1),
        "dense_out": (m["dense_layers"] * m["ffn"] * d, m["dense_layers"] * m["ffn"] * d, 1),
        "expert_in": (Lm * m["experts"] * 2 * expert, Lm * m["active"] * 2 * expert, use),
        "expert_out": (Lm * m["experts"] * expert, Lm * m["active"] * expert, use),
        "shared_in": (Lm * m["shared"] * 2 * expert, Lm * m["shared"] * 2 * expert, 1),
        "shared_out": (Lm * m["shared"] * expert, Lm * m["shared"] * expert, 1),
        "router": (Lm * m["experts"] * d, Lm * m["experts"] * d, 1),
        "head": (head, head, 1),
        "norm": (norms, norms, 1),
        "embed": (head, 0, 0),
    }


def stages(m: dict) -> list[dict]:
    """Every product stage of one token's forward pass, in order: its layer, name, class, products per token, and the
    updated weights upstream inside the token's computation by class (n per coordinate, bf16 granularity: K2's scale
    blocks change no chaining verdict, `depth.py`)."""
    d, a, V = m["d"], m["attention"], m["vocab"]
    h, qr, kr, nope, rope, v = a["heads"], a["q_rank"], a["kv_rank"], a["nope"], a["rope"], a["v"]
    qa, qb, kva, kvb, o = qr * d, h * (nope + rope) * qr, (kr + rope) * d, h * (nope + v) * kr, d * h * v
    E, act, sh, eff, ffn = m["experts"], m["active"], m["shared"], m["expert_ffn"], m["ffn"]
    below: dict[str, float] = {}
    out = []

    def up(*parts):
        u = dict(below)
        for p in parts:
            for c, n in p.items():
                u[c] = u.get(c, 0) + n
        return u

    def emit(layer, name, cls, prods, upstream):
        out.append({"layer": layer, "stage": name, "class": cls, "products": prods, "up": upstream})

    for ell in range(m["layers"]):
        emit(ell, "q_a", "mla_in", qa, up({"norm": 1}))
        emit(ell, "kv_a", "mla_in", kva, up({"norm": 1}))
        emit(ell, "q_b", "mla_in", qb, up({"mla_in": qa, "norm": d + 1}))
        emit(ell, "kv_b", "mla_in", kvb, up({"mla_in": kr * d, "norm": d + 1}))
        emit(ell, "o_proj", "mla_out", o,
             up({"mla_in": qa + (nope + rope) * qr + kva + (nope + 1) * kr, "norm": d + qr + kr}))
        attn = {"mla_in": qa + qb + kva + kvb, "mla_out": o, "norm": d + qr + kr}
        if ell < m["dense_layers"]:
            emit(ell, "gate/up", "dense_in", 2 * ffn * d, up(attn, {"norm": 1}))
            emit(ell, "down", "dense_out", ffn * d, up(attn, {"dense_in": 2 * d, "norm": d}))
            layer = {"dense_in": 2 * ffn * d, "dense_out": ffn * d}
        else:
            emit(ell, "router", "router", E * d, up(attn, {"norm": 1}))
            emit(ell, "routed gate/up", "expert_in", act * 2 * eff * d, up(attn, {"router": E * d, "norm": 1}))
            emit(ell, "shared gate/up", "shared_in", sh * 2 * eff * d, up(attn, {"norm": 1}))
            emit(ell, "routed down", "expert_out", act * eff * d,
                 up(attn, {"router": E * d, "expert_in": 2 * d, "norm": d}))
            emit(ell, "shared down", "shared_out", sh * eff * d, up(attn, {"shared_in": 2 * d, "norm": d}))
            layer = {"router": E * d, "expert_in": act * 2 * eff * d, "expert_out": act * eff * d,
                     "shared_in": sh * 2 * eff * d, "shared_out": sh * eff * d}
        for c, n in list(attn.items()) + list(layer.items()) + [("norm", d)]:
            below[c] = below.get(c, 0) + n
    emit(m["layers"], "LM head", "head", V * d, up({"norm": 1}))
    return out


def dgrad_stages(m: dict) -> list[dict]:
    """The backward products into activations, one stage per class: an activation gradient reads the loss, so every
    updated weight of the token's forward pass is upstream of it."""
    st = stages(m)
    every = dict(st[-1]["up"])
    every["head"] = every.get("head", 0) + st[-1]["products"]
    by: dict[str, float] = {}
    for s in st:
        if CLASSES[s["class"]][2] or s["class"] == "head":
            by[s["class"]] = by.get(s["class"], 0) + s["products"]
    return [{"layer": "bwd", "stage": f"dgrad {c}", "class": c, "products": n, "up": every} for c, n in by.items()]


def prices(table: dict[str, float]) -> dict[str, float]:
    """K2's class prices from one point's role prices: the router has no measured analogue and takes the cheapest
    role's price; the norm gammas are unmeasured and priced 0."""
    roles = dict(table)
    roles.setdefault("router", min(roles[r] for r in ("qkv", "attn_out", "mlp_in", "mlp_out", "head") if r in roles))
    roles.setdefault("norm", 0.0)
    roles.setdefault("embed", roles.get("mlp_in", 0.0))
    return {c: roles[role] for c, (_, role, _) in CLASSES.items()}


def counted(st: list[dict], head: bool) -> list[dict]:
    return [s for s in st if CLASSES[s["class"]][2] or (head and s["class"] == "head")]


def n_up(s: dict) -> float:
    return sum(s["up"].values())


def mass(s: dict, p: dict[str, float]) -> float:
    return sum(n * p[c] for c, n in s["up"].items())


def rho(form: str, st: list[dict], p: dict[str, float], b: int, k: float, cw: float, gates: float,
        chain: str = "A") -> dict:
    """One form's bound at one precision and k: the chained share of the counted products, the first chained stage,
    the numerator (products, or their price sum, per token) and rho."""
    ca, pmax = float(b), float(b)
    if form == "one c":
        c = min(cw, ca)
        ok = [n_up(s) >= math.floor(k / c) + math.ceil(c) for s in st]
        denom = (k / c) ** 2 + k
    elif form == "two c's":
        ok = [n_up(s) >= math.floor(k / cw) + math.ceil(ca) for s in st]
        denom = (k / cw) * (k / ca) + k
    else:
        if chain == "A":
            ok = [mass(s, p) >= k + math.ceil(ca) * pmax for s in st]
        else:
            ok = [n_up(s) >= math.floor(k / cw) + math.ceil(ca) for s in st]
        denom = k * k / ca + k * pmax if form == "priced" else k * k / (4 * ca) + k * pmax
    tot = sum(s["products"] for s in st)
    chained = sum(s["products"] for s, y in zip(st, ok) if y)
    if form in ("one c", "two c's"):
        num = chained
    else:
        num = sum(s["products"] * p[s["class"]] for s, y in zip(st, ok) if y)
    first = next((f"layer {s['layer']} {s['stage']}" for s, y in zip(st, ok) if y), None)
    mean = num / chained if chained and form not in ("one c", "two c's") else None
    return {"rho": num / gates * S / denom, "chained": chained / tot, "first": first, "mean_price": mean}


FORMS = ("one c", "two c's", "priced", "priced additive")


def grid(st: list[dict], pts: dict, cheapest: dict, gates: float, tier: str = "1%", chain: str = "A") -> dict:
    """rho[point][fmt][k][form] for every point with a cheapest-class cost (the uniform forms' c_w)."""
    out = {}
    for pt, tab in pts.items():
        if pt not in cheapest:
            continue
        out[pt] = {}
        for fmt, b in FMTS.items():
            p = prices(tab[fmt])
            cw = cheapest[pt][tier][fmt]
            out[pt][fmt] = {kn: {f: rho(f, st, p, b, k, cw, gates, chain) for f in FORMS} for kn, k in KS.items()}
    return out


def mean_price(st: list[dict], p: dict[str, float]) -> float:
    tot = sum(s["products"] for s in st)
    return sum(s["products"] * p[s["class"]] for s in st) / tot


def load_prices(path: str) -> tuple[dict, dict]:
    """A price file: ``{"points": {name: {fmt: {role or HF tensor name: bits}}}}``, and optionally the uniform forms'
    ``c_w`` as ``{"cheapest": {name: {"1%": {fmt: bits}}}}``."""
    raw = json.loads(Path(path).read_text())
    pts = {}
    for name, fmts in raw["points"].items():
        pts[name] = {}
        for fmt, tab in fmts.items():
            roles: dict[str, list[float]] = {}
            for key, val in tab.items():
                roles.setdefault(HF_ROLES.get(key, key), []).append(val)
            pts[name][fmt] = {r: sum(v) / len(v) for r, v in roles.items()}
    return pts, raw.get("cheapest", {})


def fmt_n(x: float) -> str:
    if x == 0:
        return "0"
    if abs(x) >= 1e4 or abs(x) < 1e-2:
        m, e = f"{x:.2e}".split("e")
        return f"{m}e{int(e)}"
    if abs(x) >= 100:
        return f"{x:.0f}"
    return f"{x:.3g}"


def span(xs: list[float]) -> str:
    lo, hi = min(xs), max(xs)
    return fmt_n(lo) if fmt_n(lo) == fmt_n(hi) else f"{fmt_n(lo)}–{fmt_n(hi)}"


def report(pts: dict, cheapest: dict, label: str, dgrad: bool = False) -> dict:
    m = k2()
    sh = shapes(m)
    n_tot, gates = K.active_weights(m), K.schedule(m)[1]
    st = stages(m)
    base = counted(st, head=False)
    cst = base + (dgrad_stages(m) if dgrad else [])
    batch = m["batch_tokens"]
    res = {"label": label, "s": S, "gates_per_token": gates, "alpha0": n_tot / gates, "batch_tokens": batch,
           "dgrad": dgrad}
    assert abs(sum(s["products"] for s in base) - n_tot) < 1, "the walk's counted products are kappa's N"

    print(f"# Kimi K2 under the covering bound's forms ({label})\n")
    print(f"s = {S:.3g}, gates per token {gates:.4g} (kappa.py, K2's schedule), alpha0 = N / gates = "
          f"{n_tot / gates:.4f}; {'forward and backward-into-activation' if dgrad else 'forward'} products; "
          f"{batch:.3g} tokens per step.\n")

    print("## Products per weight class (forward; the backward into activations adds as many, plus the LM head's)\n")
    print("| class | weights | share of weights | products per token | share of N | per step | per weight per step | "
          "in M* |")
    print("|---|---|---|---|---|---|---|---|")
    allw = sum(w for w, _, _ in sh.values())
    rows = {}
    for c, (lab, _, inN) in CLASSES.items():
        w, pr, use = sh[c]
        rows[c] = {"weights": w, "products_per_token": pr, "products_per_step": pr * batch, "use": use}
        print(f"| {lab} | {fmt_n(w)} | {100 * w / allw:.2f}% | {fmt_n(pr)} | "
              f"{100 * pr / n_tot:.2f}% | {fmt_n(pr * batch)} | {fmt_n(use * batch)} | "
              f"{'yes' if inN else 'no'} |")
    print(f"| total (N: the classes in M*) | {fmt_n(allw)} | | {fmt_n(n_tot)} | 100% | {fmt_n(n_tot * batch)} | | |\n")
    res["classes"] = rows

    print("## Prices (bits per weight) and the product-weighted mean over M*\n")
    print("| point | fmt | MLA in | o_proj | MLP in | MLP down | LM head | router (cheapest role) | mean over M* | "
          "with LM head | cheapest 1% (uniform c_w) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    res["prices"] = {}
    for pt, tab in pts.items():
        for fmt in FMTS:
            p = prices(tab[fmt])
            mp, mph = mean_price(base, p), mean_price(counted(st, head=True), p)
            cw = cheapest.get(pt, {}).get("1%", {}).get(fmt)
            res["prices"][f"{pt}|{fmt}"] = {"class": p, "mean": mp, "mean_with_head": mph, "cheapest_1pct": cw}
            print(f"| {pt} | {fmt} | {p['mla_in']:.3g} | {p['mla_out']:.3g} | {p['expert_in']:.3g} | "
                  f"{p['expert_out']:.3g} | {p['head']:.3g} | {p['router']:.3g} | {mp:.3g} | {mph:.3g} | "
                  f"{'' if cw is None else f'{cw:.3g}'} |")
    print()

    g = grid(cst, pts, cheapest, gates)
    res["grid"] = g
    pts_u = list(g)
    low = min(pts_u, key=lambda pt: cheapest[pt]["1%"]["fp4"])
    print(f"## rho by form, k and precision (range over {', '.join(pts_u)}; c_a = b; uniform c_w = cheapest 1%)\n")
    print(f"| form | k | bf16 | FP8 | FP4 | chained share of M* | first chained stage, bf16 / FP4 ({low}) |")
    print("|---|---|---|---|---|---|---|")
    for f in FORMS:
        for kn in KS:
            cells = [span([g[pt][fmt][kn][f]["rho"] for pt in pts_u]) for fmt in FMTS]
            ch = span([g[pt][fmt][kn][f]["chained"] for pt in pts_u for fmt in FMTS])
            first = " / ".join(str(g[low][fmt][kn][f]["first"] or "none (vacuous)") for fmt in ("bf16", "fp4"))
            print(f"| {f} | {kn} | {' | '.join(cells)} | {ch} | {first} |")
    print()

    print("## Gain of the priced form over the uniform two-cost form (same k, same point)\n")
    print("| k | bf16 | FP8 | FP4 |")
    print("|---|---|---|---|")
    for kn in KS:
        cells = []
        for fmt in FMTS:
            pairs = [(g[pt][fmt][kn]["priced"]["rho"], g[pt][fmt][kn]["two c's"]["rho"]) for pt in pts_u]
            if any(u == 0 for _, u in pairs):
                cells.append("uniform vacuous" + ("" if all(u == 0 for _, u in pairs) else " at one point"))
            else:
                cells.append(f"×{span([p / u for p, u in pairs])}")
        print(f"| {kn} | {' | '.join(cells)} |")
    print()

    gb = grid(cst, pts, cheapest, gates, chain="B")
    res["grid_chain_B"] = gb
    print("## Priced form, chained by count at p_min D > k (variant B, p_min = the cheapest 1%) or price mass (A)\n")
    print("| k | bf16 A | bf16 B | FP8 A | FP8 B | FP4 A | FP4 B |")
    print("|---|---|---|---|---|---|---|")
    for kn in KS:
        cells = []
        for fmt in FMTS:
            cells += [span([g[pt][fmt][kn]["priced"]["rho"] for pt in pts_u]),
                      span([gb[pt][fmt][kn]["priced"]["rho"] for pt in pts_u])]
        print(f"| {kn} | {' | '.join(cells)} |")
    print()

    print("## Priced form at 1 MB at every point with prices (no uniform c_w needed)\n")
    print("| point | bf16 | FP8 | FP4 | priced additive bf16 | FP8 | FP4 |")
    print("|---|---|---|---|---|---|---|")
    k1 = KS["1 MB"]
    res["priced_1mb"] = {}
    for pt, tab in pts.items():
        r = {fmt: {f: rho(f, cst, prices(tab[fmt]), b, k1, 1.0, gates)["rho"] for f in ("priced", "priced additive")}
             for fmt, b in FMTS.items()}
        res["priced_1mb"][pt] = r
        print(f"| {pt} | " + " | ".join(fmt_n(r[fmt]["priced"]) for fmt in FMTS) + " | "
              + " | ".join(fmt_n(r[fmt]["priced additive"]) for fmt in FMTS) + " |")
    print()
    return res


def sensitivity(pts: dict, cheapest: dict) -> dict:
    """Where the priced form is weakest: the routed experts' price, a cheap router or norm, the LM head counted."""
    m = k2()
    gates = K.schedule(m)[1]
    st = stages(m)
    base = counted(st, head=False)
    k = KS["1 MB"]
    out = {}
    print("## Sensitivity of the priced form at 1 MB (rho relative to the baseline, mean over the points)\n")
    print("| change | bf16 | FP8 | FP4 |")
    print("|---|---|---|---|")
    cases = {
        "routed experts at half price": ("scale", {"expert_in": 0.5, "expert_out": 0.5}, base),
        "routed experts at the cheapest 1% class": ("cheapest", {"expert_in", "expert_out"}, base),
        "router at price 0": ("scale", {"router": 0.0}, base),
        "attention (MLA) at price 0": ("scale", {"mla_in": 0.0, "mla_out": 0.0}, base),
        "LM head's products counted (at its price)": ("scale", {}, counted(st, head=True)),
        "chaining by count at p_min = 0.001 (Pythia's late FP4 head, cheapest class)": ("chain B", 0.001, base),
    }
    for name, (kind, arg, sts) in cases.items():
        cells = []
        for fmt, b in FMTS.items():
            rel = []
            for pt, tab in pts.items():
                p = prices(tab[fmt])
                r0 = rho("priced", base, p, b, k, 1.0, gates)["rho"]
                q = dict(p)
                if kind == "chain B":
                    rel.append(rho("priced", sts, q, b, k, arg, gates, chain="B")["rho"] / r0)
                    continue
                if kind == "scale":
                    for c, f in arg.items():
                        q[c] = p[c] * f
                else:
                    if pt not in cheapest:
                        continue
                    for c in arg:
                        q[c] = cheapest[pt]["1%"][fmt]
                rel.append(rho("priced", sts, q, b, k, 1.0, gates)["rho"] / r0)
            cells.append(f"×{sum(rel) / len(rel):.3f}")
        out[name] = cells
        print(f"| {name} | {' | '.join(cells)} |")
    print()
    return out


def check_daniel() -> dict:
    """Daniel's marginal-dampener table (covering bound write-up): Kimi K2 at k = 2e7, w = 2.5, raw: the base tier
    k^2/w + k gives 41, the additive charge 166, a-bit activations (a = 4) 660."""
    m = k2()
    gates = K.schedule(m)[1]
    a0 = K.active_weights(m) / gates
    k, w = 2e7, 2.5
    return {"base": a0 * S / (k * k / w + k), "additive": a0 * S / (k * k / (4 * w) + k),
            "a=4": a0 * S / (k * k / (4 * w * 4) + k)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prices", help="a price file (load_prices) in place of Pythia's")
    ap.add_argument("--dgrad", action="store_true", help="count the backward products into activations too")
    ap.add_argument("--json", help="write every number here")
    args = ap.parse_args(argv)
    if args.prices:
        pts, cheapest = load_prices(args.prices)
        label = Path(args.prices).stem
    else:
        pts, cheapest, label = PYTHIA, CHEAPEST, "Pythia prices"
    res = report(pts, cheapest, label, dgrad=args.dgrad)
    res["sensitivity"] = sensitivity(pts, cheapest)
    res["daniel_check"] = check_daniel()
    print("Daniel's table at k = 2e7, w = 2.5 (41 / 166 / 660): "
          + ", ".join(f"{n} {v:.0f}" for n, v in res["daniel_check"].items()))
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
