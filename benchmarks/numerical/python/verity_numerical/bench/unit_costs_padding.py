"""Front F6's padding table: what proving an audit's drawn units in fixed-shape sessions costs over their own nonzeros, per
workload and bucket menu, at unit = one subcircuit with :mod:`unit_costs`' per-unit nonzeros (registered lowering where one
exists).

    python -m verity_numerical.bench.unit_costs_padding --inventory unit-costs.json --out DIR

**Shapes.** A session proves B slots, each holding any unit of at most Z nonzeros: its shape is (B, Z). A menu is a set of
Z buckets; a drawn unit goes to the least bucket at or above its nonzeros, and a bucket with X drawn units takes
ceil(X / B) sessions of B slots. A unit's nonzeros for placement are its maximum (an attention stratum's at its largest T),
and for the actual cost its mean (at the stratum's mean T).

**Menus** (over the body, see tails): ``max`` one bucket at the largest Z; ``pow2`` the powers of two; ``dp{k}`` the k
buckets minimising the expected padded nonzeros, Σ_b Z_b E[X_b], by dynamic programming over the sorted distinct Z with
each unit type weighted by its draw mean (:func:`menu_dp`); ``natural`` one bucket per distinct Z (a GEMM-like unit per K,
attention per NB stratum, every other binding its own).

**Tails.** The sampler (token-select, gumbel) and every unit type over :data:`TAIL_FACTOR` x the body's 99th-percentile
draw (the Z under which 99% of drawn units fall) are the tail, at B = 1 and never in a menu: each sampler binding its own
bucket, the other large units one bucket at their largest Z (``eval_per_stratum``: one bucket per distinct Z instead), its
cost shown separately (``tail``). token-select is priced at its argmax tree (``unit_costs.retree``, applied here to an
inventory without it), which supersedes the generic scan count; ``eval_step_model`` prices the tail again with it at
(V − 1) ArgmaxStep units.

**The draw.** K = ``record_k`` units uniform without replacement from the workload's N subcircuits; a bucket with n of them
holds X ~ Hypergeometric(N, n, K), its pmf summed exactly (:func:`hyper`). Per workload x menu x B:

- ``zpad``: Σ_b Z_b E[X_b] / E[actual], the padding from rounding each unit's nonzeros up to its bucket;
- ``pad_mean``: Σ_b B Z_b E[ceil(X_b / B)] / E[actual], with sessions rounded up; ``round_extra`` = pad_mean − zpad;
- ``public``: Σ_b B Z_b ceil(q_b / B) / E[actual], each shape's session count fixed at its bucket's 1 − 2⁻⁴⁰ quantile q_b,
  a public number independent of the drawn mix (enough slots with probability ≥ 1 − m 2⁻⁴⁰ over m shapes);
  ``hide_extra`` = public − pad_mean, what hiding the mix costs;
- ``pad_q``: an upper bound on the 1 − 2⁻⁴⁰ quantile of padded / actual: every bucket at its upper and every unit type at
  its lower quantile with 2⁻⁴⁰ split evenly over them (a union bound).

``best_B`` is the largest B whose pad_mean is within :data:`B_TOLERANCE` of B = 1's (fewer, fuller sessions at the same
work). ``large`` lists, per Z in :data:`LARGE_Z`, the unit types over it (which must be split) with their draw share.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

DELTA = 2.0 ** -40
BS = (1, 16, 256, 4096)
KS = (2, 3, 4, 5, 6)
LARGE_Z = (2 ** 16, 2 ** 18, 2 ** 20, 2 ** 23)
SAMPLERS = ("token-select-greedy", "gumbel-top-p-token-select")
TAIL_FACTOR = 16
B_TOLERANCE = 0.05


@dataclass(frozen=True)
class Unit:
    label: str
    template: str
    n: int           # of the workload's N subcircuits
    z: float         # nonzeros for placement (the maximum)
    z_mean: float    # mean nonzeros
    source: str = ""
    z_step: float | None = None   # token-select's ``nnz_step_model`` (a cut scan), when it has one


def units_of(inv: Mapping[str, Any], workload: str) -> tuple[int, list[Unit], list[str]]:
    """The workload's population N, its unit types with nonzeros, and the labels of those without (excluded or unbuilt)."""
    N, out, skipped = 0, [], []
    for r in inv["rows"]:
        if r["workload"] != workload:
            continue
        g = r["subcircuit"]
        N = g["population"]
        c = g.get("per_unit")
        if not c:
            skipped.append(r["unit_type"])
            continue
        z = (g.get("nnz_range") or {}).get("max", c["nnz"])
        step = (r.get("nnz_step_model") or {}).get("nnz")
        out.append(Unit(r["unit_type"], r["template"], g["units"], float(max(z, c["nnz"])), float(c["nnz"]), r.get("source", ""),
                        None if step is None else float(step)))
    return N, out, skipped


# --- the draw --------------------------------------------------------------------------------------------------------------

_PMF: dict[tuple[int, int, int], tuple[int, list[float]]] = {}


def hyper(N: int, n: int, K: int) -> tuple[int, list[float]]:
    """Hypergeometric(N, n, K)'s pmf as ``(k0, [P(X = k0), P(X = k0 + 1), ..])``: from the mode outward by the term ratio,
    truncated where a term falls below 1e-60 of the mode's, normalised."""
    if not 0 <= n <= N or not 0 <= K <= N:
        raise ValueError(f"N = {N}, n = {n}, K = {K}")
    key = (N, n, K)
    if key in _PMF:
        return _PMF[key]
    lo, hi = max(0, K - (N - n)), min(n, K)
    mode = min(hi, max(lo, (K + 1) * (n + 1) // (N + 2)))
    up, p = [1.0], 1.0
    for k in range(mode, hi):
        p *= (n - k) * (K - k) / ((k + 1) * (N - n - K + k + 1))
        if p < 1e-60:
            break
        up.append(p)
    down, p = [], 1.0
    for k in range(mode, lo, -1):
        p *= k * (N - n - K + k) / ((n - k + 1) * (K - k + 1))
        if p < 1e-60:
            break
        down.append(p)
    probs = down[::-1] + up
    s = sum(probs)
    _PMF[key] = (mode - len(down), [x / s for x in probs])
    return _PMF[key]


def upper_q(N: int, n: int, K: int, tail: float = DELTA) -> int:
    """The least k with P(X > k) <= tail."""
    k0, p = hyper(N, n, K)
    acc = 0.0
    for j in range(len(p) - 1, -1, -1):
        if acc + p[j] > tail:
            return k0 + j
        acc += p[j]
    return k0


def lower_q(N: int, n: int, K: int, tail: float = DELTA) -> int:
    """The largest k with P(X < k) <= tail."""
    k0, p = hyper(N, n, K)
    acc = 0.0
    for j in range(len(p)):
        if acc + p[j] > tail:
            return k0 + j
        acc += p[j]
    return k0 + len(p) - 1


def mean_sessions(N: int, n: int, K: int, B: int) -> float:
    """E[ceil(X / B)]."""
    k0, p = hyper(N, n, K)
    return sum(pj * -(-(k0 + j) // B) for j, pj in enumerate(p))


# --- menus -----------------------------------------------------------------------------------------------------------------

def tail_split(units: Sequence[Unit], factor: float = TAIL_FACTOR) -> tuple[list[Unit], list[Unit], float]:
    """(body, tail, threshold): the samplers, and every unit over ``factor`` x the 99th-percentile Z of the others' draws."""
    rest = sorted((u for u in units if u.template not in SAMPLERS), key=lambda u: u.z)
    total, acc, z99 = sum(u.n for u in rest), 0, 0.0
    for u in rest:
        acc += u.n
        z99 = u.z
        if acc >= 0.99 * total:
            break
    thr = factor * z99
    body = [u for u in rest if u.z <= thr]
    return body, [u for u in units if u not in body], thr


def menu_max(units: Sequence[Unit]) -> list[float]:
    return [max(u.z for u in units)]


def menu_pow2(units: Sequence[Unit]) -> list[float]:
    return sorted({float(2 ** math.ceil(math.log2(u.z))) for u in units})


def menu_natural(units: Sequence[Unit]) -> list[float]:
    return sorted({u.z for u in units})


def menu_dp(units: Sequence[Unit], k: int) -> list[float]:
    """The at most k bucket tops minimising Σ_b Z_b W_b, W_b the draw weight (units) of the types in bucket b: exact, over
    the sorted distinct Z (a bucket's top is the largest Z it holds)."""
    zs = sorted({u.z for u in units})
    w = [sum(u.n for u in units if u.z == z) for z in zs]
    m = len(zs)
    pre = [0]
    for x in w:
        pre.append(pre[-1] + x)
    INF = float("inf")
    # best[j][i]: least cost covering zs[:i] with j buckets, the last topped at zs[i - 1]
    best = [[INF] * (m + 1) for _ in range(k + 1)]
    arg = [[0] * (m + 1) for _ in range(k + 1)]
    best[0][0] = 0
    for j in range(1, k + 1):
        for i in range(1, m + 1):
            for s in range(j - 1, i):
                c = best[j - 1][s] + zs[i - 1] * (pre[i] - pre[s])
                if c < best[j][i]:
                    best[j][i], arg[j][i] = c, s
    j = min(range(1, k + 1), key=lambda j: best[j][m])
    tops, i = [], m
    while i > 0:
        tops.append(zs[i - 1])
        i, j = arg[j][i], j - 1
    return sorted(tops)


def assign(units: Sequence[Unit], menu: Sequence[float]) -> dict[float, list[Unit]]:
    out: dict[float, list[Unit]] = {Z: [] for Z in sorted(menu)}
    for u in units:
        out[next(Z for Z in sorted(menu) if Z >= u.z)].append(u)
    return {Z: us for Z, us in out.items() if us}


# --- evaluation ------------------------------------------------------------------------------------------------------------

def evaluate(N: int, K: int, units: Sequence[Unit], menu: Sequence[float], B: int, delta: float = DELTA) -> dict[str, Any]:
    """One menu at B over ``units``: the expected actual nonzeros and every padded figure in the module docstring."""
    buckets = assign(units, menu)
    actual = sum(K * u.n / N * u.z_mean for u in units)
    m = len(buckets) + len(units)
    shapes, frac, mean, public, hi = [], 0.0, 0.0, 0.0, 0.0
    for Z, us in buckets.items():
        n = sum(u.n for u in us)
        xm, q = K * n / N, upper_q(N, n, K, delta)
        s = mean_sessions(N, n, K, B)
        frac += Z * xm
        mean += B * Z * s
        public += B * Z * -(-q // B)
        hi += B * Z * -(-upper_q(N, n, K, delta / m) // B)
        shapes.append({"Z": Z, "types": len(us), "units": n, "drawn_mean": xm, "drawn_q": q, "sessions_mean": s,
                       "sessions_public": -(-q // B)})
    low = sum(lower_q(N, u.n, K, delta / m) * u.z_mean for u in units)
    r = lambda x: x / actual if actual else float("nan")  # noqa: E731
    return {"B": B, "buckets": len(buckets), "actual_mean": actual, "padded_mean": mean, "padded_public": public,
            "zpad": r(frac), "pad_mean": r(mean), "round_extra": r(mean - frac), "public": r(public),
            "hide_extra": r(public - mean), "pad_q": hi / low if low else None,
            "sessions_mean": sum(s["sessions_mean"] for s in shapes), "sessions_public": sum(s["sessions_public"] for s in shapes),
            "shapes": shapes}


def large(N: int, units: Sequence[Unit], K: int) -> list[dict[str, Any]]:
    out = []
    for Z in LARGE_Z:
        over = sorted((u for u in units if u.z > Z), key=lambda u: -u.n)
        out.append({"Z": Z, "types_over": len(over), "types": len(units), "draw_share_over": sum(u.n for u in over) / N,
                    "pieces_mean": (sum(u.n * math.ceil(u.z / Z) for u in over) / sum(u.n for u in over)) if over else 0,
                    "units": [{"unit_type": u.label, "nnz": u.z, "draw_share": u.n / N, "pieces": math.ceil(u.z / Z)} for u in over]})
    return out


def workload_table(inv: Mapping[str, Any], workload: str) -> dict[str, Any]:
    K = int(inv["K"])
    N, units, skipped = units_of(inv, workload)
    body, tail, thr = tail_split(units)
    menus: dict[str, list[float]] = {"max": menu_max(body), "pow2": menu_pow2(body)}
    for k in KS:
        menus[f"dp{k}"] = menu_dp(body, k)
    menus["natural"] = menu_natural(body)
    rows = []
    for name, menu in menus.items():
        by_b = {B: evaluate(N, K, body, menu, B) for B in BS}
        base = by_b[1]["pad_mean"]
        best = max(B for B in BS if by_b[B]["pad_mean"] <= (1 + B_TOLERANCE) * base)
        rows.append({"menu": name, "Z": menu, "best_B": best, "by_B": {str(B): v for B, v in by_b.items()}})
    big = [u for u in tail if u.template not in SAMPLERS]

    def tail_eval(samplers: list[Unit], one_bucket: bool = True) -> dict[str, Any] | None:
        parts = [evaluate(N, K, samplers, menu_natural(samplers), 1)] if samplers else []
        if big:
            parts.append(evaluate(N, K, big, menu_max(big) if one_bucket else menu_natural(big), 1))
        if not parts:
            return None
        keys = ("actual_mean", "padded_mean", "padded_public", "sessions_mean", "sessions_public")
        out: dict[str, Any] = {k: sum(p[k] for p in parts) for k in keys}
        out.update(buckets=sum(p["buckets"] for p in parts), shapes=[s for p in parts for s in p["shapes"]])
        return out

    samplers = [u for u in tail if u.template in SAMPLERS]
    tl = tail_eval(samplers)
    ts = tail_eval([replace(u, z=u.z_step, z_mean=u.z_step) if u.z_step else u for u in samplers])
    tn = tail_eval(samplers, one_bucket=False)
    return {"workload": workload, "N": N, "K": K, "skipped": skipped, "tail_threshold": thr,
            "body": {"types": len(body), "draw_share": sum(u.n for u in body) / N},
            "tail": {"types": [{"unit_type": u.label, "nnz": u.z, "nnz_mean": u.z_mean, "draw_share": u.n / N, "source": u.source}
                               for u in tail], "draw_share": sum(u.n for u in tail) / N, "eval": tl, "eval_step_model": ts,
                     "eval_per_stratum": tn},
            "menus": rows, "large": large(N, units, K)}


def padding(inv: Mapping[str, Any]) -> dict[str, Any]:
    if not inv["rows"] or "per_unit" not in next((r["subcircuit"] for r in inv["rows"] if r["subcircuit"].get("per_unit")), {}):
        raise ValueError("the inventory has no per-unit nonzeros")
    return {"format": "verity/f6-padding/v1", "unit": "subcircuit", "K": inv["K"], "delta": "2^-40", "Bs": list(BS),
            "tail_factor": TAIL_FACTOR, "b_tolerance": B_TOLERANCE, "nnz": inv.get("nnz"),
            "workloads": [workload_table(inv, w) for w in dict.fromkeys(r["workload"] for r in inv["rows"])]}


# --- rendering -------------------------------------------------------------------------------------------------------------

def _si(v: float | None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    for unit, d in (("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(v) >= d:
            return f"{v / d:.3g}{unit}"
    return f"{v:.3g}"


def _short(w: str) -> str:
    p = w.split("/")[-1].split("__")
    return "/".join(x for x in (p[0], *p[4:6], *(q for q in p[7:8] if q != "mixed")) if x)


def render(pad: Mapping[str, Any]) -> str:
    L = [f"# F6 padding: unit = one subcircuit, K = {pad['K']} draws, δ = {pad['delta']}", "",
         "Per-unit nonzeros: " + str(pad.get("nnz")), "",
         "zpad: Z rounding only; pad_mean: with sessions of B slots rounded up; public: each shape's session count fixed at "
         "its 1 − 2⁻⁴⁰ quantile (hides the drawn mix); pad_q: bound on the 1 − 2⁻⁴⁰ quantile of padded / actual. All over the "
         "body's expected actual nonzeros. Tail: the samplers and units over "
         f"{pad['tail_factor']}x the body's 99th-percentile Z, each its own bucket at B = 1.", "",
         "## Compact: each workload x menu at its best B", "",
         f"best B: the largest B within {pad['b_tolerance']:.0%} of B = 1's pad_mean.", "",
         "| workload | menu | buckets | best B | zpad | pad_mean | round extra | public | hide extra | pad_q | sessions mean / "
         "public | tail share of draws | tail actual / body actual | tail public / body actual | same, token-select at its step "
         "model |", "|" + "---|" * 15]
    for w in pad["workloads"]:
        t, ts = w["tail"]["eval"], w["tail"]["eval_step_model"] or w["tail"]["eval"]
        for m in w["menus"]:
            e = m["by_B"][str(m["best_B"])]
            L.append("| " + " | ".join([
                _short(w["workload"]), m["menu"], str(e["buckets"]), str(m["best_B"]), f"{e['zpad']:.3f}", f"{e['pad_mean']:.3f}",
                f"{e['round_extra']:.4f}", f"{e['public']:.3f}", f"{e['hide_extra']:.4f}",
                f"{e['pad_q']:.3f}" if e["pad_q"] else "–", f"{e['sessions_mean']:.1f} / {e['sessions_public']}",
                f"{w['tail']['draw_share']:.2e}", f"{t['actual_mean'] / e['actual_mean']:.3g}" if t else "0",
                f"{t['padded_public'] / e['actual_mean']:.3g}" if t else "0",
                f"{ts['actual_mean'] / e['actual_mean']:.3g} / {ts['padded_public'] / e['actual_mean']:.3g}" if ts else "0"]) + " |")
    L += ["", "## B sensitivity: pad_mean and public at each B", "",
          "| workload | menu | " + " | ".join(f"B={B} mean / public" for B in pad["Bs"]) + " |", "|" + "---|" * (2 + len(pad["Bs"]))]
    for w in pad["workloads"]:
        for m in w["menus"]:
            L.append(f"| {_short(w['workload'])} | {m['menu']} | " + " | ".join(
                f"{m['by_B'][str(B)]['pad_mean']:.3f} / {m['by_B'][str(B)]['public']:.3f}" for B in pad["Bs"]) + " |")
    L += ["", "## Tails", "", "| workload | tail unit | src | nnz | draw share |", "|---|---|---|---|---|"]
    for w in pad["workloads"]:
        for u in sorted(w["tail"]["types"], key=lambda u: -u["nnz"]):
            L.append(f"| {_short(w['workload'])} | {u['unit_type']} | {u['source'][:1].upper()} | {_si(u['nnz'])} | {u['draw_share']:.2e} |")
    L += ["", "## Large units: unit types over Z (must be split)", "",
          "| workload | Z | types over / all | draw share over | mean pieces | largest by share |", "|---|---|---|---|---|---|"]
    for w in pad["workloads"]:
        for g in w["large"]:
            top = ", ".join(f"{u['unit_type']} ({u['draw_share']:.1%}, {u['pieces']} pieces)" for u in g["units"][:3])
            L.append(f"| {_short(w['workload'])} | 2^{int(math.log2(g['Z']))} | {g['types_over']} / {g['types']} | "
                     f"{g['draw_share_over']:.4%} | {g['pieces_mean']:.1f} | {top} |")
    L += ["", "## Menus", "", "| workload | menu | Z buckets |", "|---|---|---|"]
    for w in pad["workloads"]:
        for m in w["menus"]:
            L.append(f"| {_short(w['workload'])} | {m['menu']} | {', '.join(_si(z) for z in m['Z'])} |")
    return "\n".join(L) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--inventory", required=True, help="unit_costs' unit-costs.json")
    ap.add_argument("--out", required=True, help="directory for padding.json and padding.md")
    a = ap.parse_args(argv)
    from pathlib import Path

    from .unit_costs import retree

    inv = json.loads(Path(a.inventory).read_text())
    if "token_select" not in inv:
        retree(inv)
    pad = padding(inv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "padding.json").write_text(json.dumps(pad, indent=1) + "\n")
    (out / "padding.md").write_text(render(pad))
    print(f"wrote {out / 'padding.json'} and {out / 'padding.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
