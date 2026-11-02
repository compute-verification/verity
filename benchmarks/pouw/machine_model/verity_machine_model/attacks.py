"""The attack families at a vector's prices: an attack's X, the cheapest attacks of a design, and the check every vector
passes.

An attack is a program of warp instructions a tensor MAC replaced (`data/attacks/<family>.jsonl`); X is its price over
the MACs it replaces at the vector's MAC price, so X ≥ 1 means it costs at least the honest tensor MACs.
"""
from __future__ import annotations

from collections import Counter

from verity_machine_model.lp import MACS, Solution, cost, price
from verity_machine_model.model import Attack, Census, Floor, Model, measured_cost

EPS = 1e-9


def x_of(census: Census, sol: Solution, prog: dict[str, float]) -> float:
    return cost(census, sol.lam, prog) / (price(census, sol.lam, "MMA") / MACS)


def per_word(census: Census, sol: Solution, kind: str, words: int) -> float:
    """A kind's price in tensor MACs a word it writes."""
    return x_of(census, sol, {kind: 1}) / words


def under_floor(census: Census, sol: Solution, floors: list[Floor]) -> dict[str, tuple[int, list[str], list[str]]]:
    """Per claim: how many of its kinds the census has, each priced under its floor, and each the census lacks. A claim
    holds over the census when nothing is under its floor; a kind the census lacks is a class the epoch didn't split
    out, priced as the class that holds it."""
    out: dict[str, tuple[int, list[str], list[str]]] = {}
    for f in floors:
        n, below, absent = out.get(f.claim, (0, [], []))
        if f.kind not in census:
            absent.append(f.kind)
        else:
            n += 1
            if (x := per_word(census, sol, f.kind, f.words)) < f.floor * (1 - EPS):
                below.append(f"{f.kind} {x:.2f} a word < {f.floor:g}")
        out[f.claim] = (n, below, absent)
    return out


def priced(census: Census, sol: Solution, floors: list[Floor], claim: str) -> list[tuple[str, int, float, float, str]]:
    """A claim's kinds the census has, cheapest a word first: (kind, words, price a word, measured cost a word, what),
    in MACs at the vector's MAC price. The measured cost a word is the most a sound price can charge."""
    mac = price(census, sol.lam, "MMA") / sol.sigma / MACS
    mc = measured_cost(census)
    return sorted(((f.kind, f.words, per_word(census, sol, f.kind, f.words), mc[f.kind] / mac / f.words, f.what)
                   for f in floors if f.claim == claim and f.kind in census), key=lambda t: (round(t[2], 2), t[0]))


def cheapest(census: Census, sol: Solution, attacks: list[Attack], design: str, n: int = 3):
    """The `n` cheapest (X, family, name) of `design`, and the cheapest three-piece (iv) attack's X (None without one)."""
    xs = sorted((x_of(census, sol, a.program), a.family, a.name) for a in attacks
                if a.design == design and not set(a.program) & set(sol.refused))
    tp = min((x for x, fam, _ in xs if fam.startswith("(iv)")), default=None)
    return xs[:n], tp


def margin(census: Census, sol: Solution, attacks: list[Attack], design: str) -> float:
    """The cheapest attack's X in `design`: how far the cheapest known attack is above the honest MACs it replaces."""
    return cheapest(census, sol, attacks, design, 1)[0][0][0]


def families(attacks: list[Attack]) -> Counter:
    return Counter(a.family for a in attacks)


def check(model: Model, census: Census, sol: Solution, credits, attacks: list[Attack], x_min: float,
          rho: float, designs: set[str] | None = None) -> list[str]:
    """What the vector fails, empty when it passes: Σθ = 1, every price at or below its kind's measured cost (a convex
    combination of rows is at most its largest row), ρ on every CUDA lane kind, every credit paid, every attack at
    X ≥ x_min, and no kind refused."""
    out = []
    pr = {k: price(census, sol.lam, k) / sol.sigma for k in census}
    if abs(sum(sol.theta.values()) - 1) >= EPS:
        out.append(f"Σθ = {sum(sol.theta.values())!r}, not 1")
    mc = measured_cost(census)
    out += [f"{k}'s price {pr[k]:.6g} u exceeds its measured cost {mc[k]:.6g} u" for k in census
            if not pr[k] <= mc[k] * (1 + EPS)]
    out += [f"(i) ρ_{k} = {4 * pr[k] / pr['MMA']:.6g} < {rho}" for k in model.cuda if k in census
            and not 4 * pr[k] / pr["MMA"] >= rho - EPS]
    out += [f"credit {k} ≥ {cr} MACs unpaid: {pr[k] / (pr['MMA'] / MACS):.6g}" for k, cr in credits
            if not pr[k] >= cr * pr["MMA"] / MACS * (1 - EPS)]
    for a in attacks:
        if a.on and (designs is None or a.design in designs):
            if set(a.program) - set(census):
                out.append(f"attack {a.design} / {a.family} / {a.name} uses kinds the census lacks: "
                           f"{sorted(set(a.program) - set(census))}")
            elif (x := x_of(census, sol, a.program)) < x_min * (1 - EPS):
                out.append(f"attack {a.design} / {a.family} / {a.name} costs X = {x:.6g} < {x_min}")
    if sol.refused:
        out.append(f"kinds refused (no coefficient on the vector's rows): {', '.join(sol.refused)}")
    return out
