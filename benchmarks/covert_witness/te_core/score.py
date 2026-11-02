"""The slowdown ρ the theorem gives a submission, and the effective slowdown over the frontier.

For a block of c MACs that imports k bits (in(S) in the sender form), an annotation whose updated MACs each import at
least β bits (update hardness, T1) and whose share φ of |C| is updated MACs with at least D ≥ k/β updated MACs upstream
(deep use, T2) gives

    ρ = max(1, φ c β / (k (k + β)))   if D ≥ k/β,   else 1.

The scorer takes the larger of two valid annotations (`score`):

- (a) all of U, with β = min(p/r, min over covered steps of β_t): a verified counterexample family lowers β to what it
  shows;
- (b) only the uncovered steps' U, with β = p/r and φ recomputed on that subset (the same |C|): steps a family covers are
  left out, so one cheap step can't lower the β of the rest.

p is the bits per updated value (default: the narrowest width an updated weight is multiplied at), r in `R` its assumed
compressibility, k in `K_BITS`, and c = `C_PER_FORWARD_TOKEN` times the start model's forward MACs per token.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

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
    name: str
    steps: int  # the steps whose U it holds
    beta: float
    d_min: float
    depth: int
    phi: float
    rho: float


def option(name: str, ann: Annotation, steps, beta: float, k: float, c: float) -> Option:
    d_min = k / beta if beta > 0 else math.inf
    sel = ann.select(steps)
    depth, phi = ann.depth(steps), ann.phi(d_min, steps)
    return Option(name, len(sel), beta, d_min, depth, phi, rho_of(phi, depth, beta, k, c))


def score(ann: Annotation, covered: dict[int, float], k: float, r: int, p: float, c: float) -> dict:
    """ρ at one (k, r): `covered` maps each step a verified family covers to its β_t."""
    base = p / r
    here = {s.step for s in ann.steps}
    cov = {t: b for t, b in covered.items() if t in here}
    a = option("a", ann, None, min([base, *cov.values()]), k, c)
    b = option("b", ann, [t for t in sorted(here) if t not in cov], base, k, c)
    best = a if a.rho >= b.rho else b
    return {"rho": best.rho, "option": best.name, "beta": best.beta, "phi": best.phi, "depth": best.depth,
            "d_min": best.d_min, "a": asdict(a), "b": asdict(b)}


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
