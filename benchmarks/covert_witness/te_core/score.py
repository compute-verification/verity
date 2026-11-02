"""The slowdown ρ the theorem gives a submission, and the effective slowdown over the frontier.

For a block of c MACs that imports k bits (in(S) in the sender form), an annotation whose updated MACs each import at
least β bits (update hardness, T1) and whose share φ of |C| is updated MACs with at least D ≥ k/β updated MACs upstream
(deep use, T2) gives

    ρ = max(1, φ c β / (k (k + β)))   if D ≥ k/β,   else 1.

The scorer takes the largest ρ over thresholds τ (`score`). For each τ in {p/r} and the covered steps' β_t below p/r, the
annotation U_τ holds the uncovered steps' U and the covered steps' with β_t ≥ τ, with β = min(p/r, τ) and φ recomputed on
U_τ (the same |C|). Each U_τ is valid: no verified counterexample family shows a step of it below β. So a cheap covered
step costs only its own share of φ, and covering every step at an honest β_t ≥ p/r leaves ρ where it was. τ = the least
β_t is the annotation of all of U at that β; τ = p/r drops only the steps covered below p/r.

p is the bits per updated value (default: the narrowest width an updated weight is multiplied at), r in `R` its assumed
compressibility, k in `K_BITS`, and c = `C_PER_FORWARD_TOKEN` times the start model's forward MACs per token.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

from .annotate import Annotation, annotate

K_BITS = {"10kB": 80_000, "100kB": 800_000, "1MB": 8_000_000}
R = (1, 10, 100)
C_PER_FORWARD_TOKEN = 4096


def block_c(forward_macs_per_token: float) -> float:
    return C_PER_FORWARD_TOKEN * forward_macs_per_token


def cell(k: str, r: int) -> str:
    return f"{k}/r{r}"


def rho_of(phi: float, depth: float, beta: float, k: float, c: float) -> float:
    if beta <= 0 or depth < k / beta:
        return 1.0
    return max(1.0, phi * c * beta / (k * (k + beta)))


@dataclass(frozen=True)
class Option:
    tau: float
    steps: int  # the steps whose U it holds
    beta: float
    d_min: float
    depth: int
    phi: float
    rho: float


class Thresholds:
    """An annotation's steps grouped by what they add to φ and D (their profile and depth), each group with its uncovered
    count and its covered steps' sorted β_t, so U_τ costs a lookup per group rather than a pass over every step."""

    def __init__(self, ann: Annotation, cov: dict[int, float]):
        self.macs = ann.macs
        groups: dict[tuple, tuple[list[int], list[float]]] = {}
        for s in ann.steps:
            free, betas = groups.setdefault((s.depth if s.updated_macs else 0, s.profile), ([0], []))
            if s.step in cov:
                betas.append(cov[s.step])
            else:
                free[0] += 1
        self.groups = []
        for (depth, profile), (free, betas) in groups.items():
            prof = sorted(profile)
            suffix = [0] * (len(prof) + 1)
            for i in range(len(prof) - 1, -1, -1):
                suffix[i] = suffix[i + 1] + prof[i][1]
            self.groups.append((depth, [up for up, _ in prof], suffix, free[0], sorted(betas)))

    def option(self, tau: float, beta: float, k: float, c: float) -> Option:
        d_min = k / beta if beta > 0 else math.inf
        steps = depth = deep = 0
        for g_depth, ups, suffix, free, betas in self.groups:
            n = free + len(betas) - bisect.bisect_left(betas, tau)
            if n:
                steps += n
                depth = max(depth, g_depth)
                deep += n * suffix[bisect.bisect_left(ups, d_min)]
        phi = deep / self.macs if self.macs > 0 else 0.0
        return Option(tau, steps, beta, d_min, depth, phi, rho_of(phi, depth, beta, k, c))


def score(ann: Annotation, covered: dict[int, float], k: float, r: int, p: float, c: float) -> dict:
    """ρ at one (k, r): `covered` maps each step a verified family covers to its β_t; the largest ρ over the thresholds τ
    (the module docstring), ties to the larger τ."""
    base = p / r
    here = {s.step for s in ann.steps}
    cov = {t: b for t, b in covered.items() if t in here}
    th = Thresholds(ann, cov)
    best = None
    for tau in sorted({base, *(b for b in cov.values() if b < base)}, reverse=True):
        o = th.option(tau, min(base, tau), k, c)
        if best is None or o.rho > best.rho:
            best = o
    return {"rho": best.rho, "option": f"tau={best.tau:.4g}", "tau": best.tau, "steps": best.steps, "beta": best.beta,
            "phi": best.phi, "depth": best.depth, "d_min": best.d_min}


def score_grid(ann: Annotation, covered: dict[int, float], p: float, c: float) -> dict[str, dict]:
    return {cell(name, r): score(ann, covered, k, r, p, c) for name, k in K_BITS.items() for r in R}


def checkpoints(record: dict, covered: dict[int, float], c: float, p: float | None = None) -> list[dict]:
    """Every checkpoint of a run as its own submission: the prefix of C through its step, with the coverage of the steps
    it holds."""
    out = []
    last = record["steps"][-1]["step"] if record["steps"] else -1
    for cp in record.get("checkpoints", []):
        ann = annotate(record, through=None if cp["step"] >= last else cp["step"])
        pp = p if p is not None else (ann.bits() or 16)
        grid = score_grid(ann, covered, pp, c)
        out.append({"checkpoint": cp, "macs": ann.macs, "p": pp, "c": c, "annotation": ann.summary(), "grid": grid})
    return out


def w_star(subs: list[dict], l_star: float, key: str) -> float | None:
    """W*(l*): the least |C| ρ over submissions that reach `l_star` bits per byte (val)."""
    vals = [s["macs"] * s["rho"][key] for s in subs if s.get("val_bpb") is not None and s["val_bpb"] <= l_star]
    return min(vals, default=None)


def honest(refs: list[dict], l_star: float) -> float | None:
    """The least |C| over the reference runs' checkpoints that reach `l_star`."""
    vals = [s["macs"] for s in refs if s.get("val_bpb") is not None and s["val_bpb"] <= l_star]
    return min(vals, default=None)


def effective_slowdown(subs: list[dict], refs: list[dict], l_star: float, key: str) -> float | None:
    w, h = w_star(subs, l_star, key), honest(refs, l_star)
    return None if w is None or not h else w / h
