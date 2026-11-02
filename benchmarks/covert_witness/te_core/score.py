"""The slowdown ρ the theorem gives a submission, and the effective slowdown over the frontier.

For a block of c MACs that imports k bits (in(S) in the sender form), an annotation whose updated MACs each import at
least β bits (update hardness, T1) and whose share φ of |C| is updated MACs with at least D ≥ k/β updated MACs upstream
(deep use, T2) gives

    ρ = max(1, φ c β / (k (k + β)))   if D ≥ k/β,   else 1.

A step's β_t is the least of its witnesses (`step_betas`): the narrowest cut of its traced circuit between its
gradient-side computation and its new persistent values, per updated MAC per token block (`StepAnnotation.beta`,
te_core.capture), and the β_t of every verified family covering it. A step with no updated MAC, or with no witness (a
step the capture didn't trace and no family covers), is in no U. The scorer takes the largest ρ over thresholds τ
(`score`): for each τ among the steps' β_t, the annotation U_τ holds the steps with β_t ≥ τ, at β = τ, with φ recomputed
on U_τ (the same |C|), ties to the larger τ. Each U_τ is valid: no witness shows a step of it below τ. So a cheap step
costs only its own share of φ.

k is in `K_BITS`, and c = `C_PER_FORWARD_TOKEN` times the start model's forward MACs per token.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from .annotate import Annotation, annotate

K_BITS = {"10kB": 80_000, "100kB": 800_000, "1MB": 8_000_000}
C_PER_FORWARD_TOKEN = 4096


def block_c(forward_macs_per_token: float) -> float:
    return C_PER_FORWARD_TOKEN * forward_macs_per_token


def rho_of(phi: float, depth: float, beta: float, k: float, c: float) -> float:
    if beta <= 0 or depth < k / beta:
        return 1.0
    return max(1.0, phi * c * beta / (k * (k + beta)))


def step_betas(ann: Annotation, covered: dict[int, float]) -> dict[int, float]:
    """Each step's β_t: the least of its cut and the β_t of the families covering it (`covered`: step -> least β_t);
    a step with no updated MAC or no witness is left out."""
    out = {}
    for s in ann.steps:
        ws = [b for b in (s.beta, covered.get(s.step)) if b is not None]
        if s.updated_macs and ws:
            out[s.step] = min(ws)
    return out


@dataclass(frozen=True)
class Option:
    tau: float
    steps: int  # the steps whose U it holds
    beta: float
    d_min: float | None
    depth: int
    phi: float
    rho: float


class Thresholds:
    """An annotation's steps grouped by what they add to φ and D (their profile and depth), each group with its steps'
    sorted β_t, so U_τ costs a lookup per group rather than a pass over every step."""

    def __init__(self, ann: Annotation, betas: dict[int, float]):
        self.macs = ann.macs
        groups: dict[tuple, list[float]] = {}
        for s in ann.steps:
            if s.step in betas:
                groups.setdefault((s.depth, s.profile), []).append(betas[s.step])
        self.groups = []
        for (depth, profile), bs in groups.items():
            prof = sorted(profile)
            suffix = [0] * (len(prof) + 1)
            for i in range(len(prof) - 1, -1, -1):
                suffix[i] = suffix[i + 1] + prof[i][1]
            self.groups.append((depth, [up for up, _ in prof], suffix, sorted(bs)))

    def option(self, tau: float, k: float, c: float) -> Option:
        d_min = k / tau
        steps = depth = deep = 0
        for g_depth, ups, suffix, betas in self.groups:
            n = len(betas) - bisect.bisect_left(betas, tau)
            if n:
                steps += n
                depth = max(depth, g_depth)
                deep += n * suffix[bisect.bisect_left(ups, d_min)]
        phi = deep / self.macs if self.macs > 0 else 0.0
        return Option(tau, steps, tau, d_min, depth, phi, rho_of(phi, depth, tau, k, c))


def score(ann: Annotation, covered: dict[int, float], k: float, c: float) -> dict:
    """ρ at one k: `covered` maps each step a verified family covers to its β_t; the largest ρ over the thresholds τ
    (the module docstring), ties to the larger τ."""
    betas = step_betas(ann, covered)
    th = Thresholds(ann, betas)
    best = Option(0.0, 0, 0.0, None, 0, 0.0, 1.0)
    for tau in sorted({b for b in betas.values() if b > 0}, reverse=True):
        o = th.option(tau, k, c)
        if best.steps == 0 or o.rho > best.rho:
            best = o
    return {"rho": best.rho, "option": f"tau={best.tau:.4g}" if best.steps else "no U", "tau": best.tau,
            "steps": best.steps, "beta": best.beta, "phi": best.phi, "depth": best.depth, "d_min": best.d_min}


def score_grid(ann: Annotation, covered: dict[int, float], c: float) -> dict[str, dict]:
    return {name: score(ann, covered, k, c) for name, k in K_BITS.items()}


def checkpoints(record: dict, covered: dict[int, float], c: float) -> list[dict]:
    """Every checkpoint of a run as its own submission: the prefix of C through its step, with the coverage of the steps
    it holds."""
    out = []
    last = record["steps"][-1]["step"] if record["steps"] else -1
    for cp in record.get("checkpoints", []):
        ann = annotate(record, through=None if cp["step"] >= last else cp["step"])
        here = {s.step for s in ann.steps}
        grid = score_grid(ann, {t: b for t, b in covered.items() if t in here}, c)
        out.append({"checkpoint": cp, "macs": ann.macs, "c": c, "annotation": ann.summary(), "grid": grid})
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
