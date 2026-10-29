"""Flock's cost model, fitted from measurements, for the sampling planner (`optimize.CostModel`).

Three regressions on a template's features (`FEATURES`: one, ands, ands staged, ands drawn), each a least squares in
relative error with non-negative coefficients over the observations that measure it:

    prove   GPU-s a session      = f0 + f1 ands + f2 ands staged + b ands drawn
    verify  CPU-s a statement    = v0 + v1 ands + v2 ands staged + v3 ands drawn
    stage   CPU-s a statement    = s0 + s1 ands + s2 ands staged

each on the features its caller gives.  `ands` is one proof block's in-circuit ANDs, its rows' openings included
(`optimize.row_ands`), `staged` the statement's proof blocks and `drawn` the session's.  A coefficient another fit measures
better is pinned (`fixed`), and its source says so.  `model` puts the three in the planner's currency, GPU-seconds, at
`cpu` GPU-seconds a CPU-second (the price ratio): each feature's rate is its prove coefficient plus its verify and stage
coefficients at `cpu`, `one` and `ands` per touched statement, `ands*staged` per staged proof block and `ands*drawn` per proved
proof block.  A fit's `sigma` is its relative residual, at least `floor` when it has no spare observation; the model's is the
largest.  `record` is the cost-model/v1 payload (`research data put --kind cost-model/v1`), keyed by the prover's build,
the verifier's and the hardware, so a plan reads the fit of the builds it will run.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from math import sqrt

from verity.core.protocols.sampled_proofs.optimize import CostModel

FEATURES = ("one", "ands", "ands*drawn", "ands*staged")
TARGETS = ("prove", "verify", "stage")


@dataclass(frozen=True)
class Observation:
    """One measurement from `source` (a run or `art:` id): a session sampling `drawn` proof blocks from a statement of `staged`
    proof blocks of `template`, `ands` ANDs a proof block, with what it measured: `prove` (GPU-s, the session), `verify` and `stage`
    (CPU-s, the statement)."""

    source: str
    template: str
    ands: int
    staged: int
    drawn: int
    prove: float | None = None
    verify: float | None = None
    stage: float | None = None

    def feature(self, name: str) -> float:
        return {"one": 1.0, "ands": float(self.ands), "ands*drawn": float(self.ands * self.drawn),
                "ands*staged": float(self.ands * self.staged)}[name]


@dataclass(frozen=True)
class Fit:
    """`target` = sum of `coef[f]` times feature f, over the observations from `sources`; `pinned` names the coefficients
    another fit gave."""

    target: str
    coef: Mapping[str, float]
    sigma: float
    points: int
    sources: tuple[str, ...]
    pinned: tuple[str, ...] = ()

    def __call__(self, o: Observation) -> float:
        return sum(c * o.feature(f) for f, c in self.coef.items())


def _solve(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    m = [row[:] + [v] for row, v in zip(a, b)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        if abs(m[p][i]) < 1e-300:
            return None
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                m[r] = [x - f * y for x, y in zip(m[r], m[i])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit(observations: Sequence[Observation], target: str, features: Sequence[str], *,
        fixed: Mapping[str, float] | None = None, floor: float = 0.5) -> Fit:
    """The non-negative least squares in relative error of `target` on `features` (each of `FEATURES`), over the
    observations that measure it, the `fixed` coefficients held."""
    if target not in TARGETS or not set(features) <= set(FEATURES):
        raise ValueError(f"a fit is of one of {TARGETS} on features among {FEATURES}")
    fixed = dict(fixed or {})
    rows = [o for o in observations if getattr(o, target) is not None]
    if not rows:
        raise ValueError(f"no observation measures {target}")
    free = [f for f in features if f not in fixed]
    ys = [getattr(o, target) - sum(c * o.feature(f) for f, c in fixed.items()) for o in rows]
    ws = [1 / getattr(o, target) ** 2 for o in rows]
    best: tuple[float, dict[str, float]] | None = None
    for r in range(len(free), -1, -1):
        for subset in combinations(free, r):
            xs = [[o.feature(f) for f in subset] for o in rows]
            if subset:
                scale = [max(abs(x[j]) for x in xs) or 1.0 for j in range(len(subset))]
                a = [[sum(w * x[i] * x[j] / (scale[i] * scale[j]) for x, w in zip(xs, ws)) for j in range(len(subset))]
                     for i in range(len(subset))]
                rhs = [sum(w * x[i] * y / scale[i] for x, y, w in zip(xs, ys, ws)) for i in range(len(subset))]
                sol = _solve(a, rhs)
                if sol is None or min(sol) < 0:
                    continue
                coef = {f: c / s for f, c, s in zip(subset, sol, scale)}
            else:
                coef = {}
            res = sum(w * (y - sum(coef.get(f, 0.0) * o.feature(f) for f in subset)) ** 2
                      for o, y, w in zip(rows, ys, ws))
            if best is None or res < best[0] * (1 - 1e-12):
                best = (res, coef)
    res, coef = best
    spare = len(rows) - len(free)
    sigma = max(sqrt(res / spare), floor if spare <= 1 else 0.0) if spare > 0 else floor
    return Fit(target, {**{f: coef.get(f, 0.0) for f in free}, **fixed}, sigma, len(rows),
               tuple(sorted({o.source for o in rows})), tuple(sorted(fixed)))


@dataclass(frozen=True)
class Fitted:
    """The three fits and the price ratio, keyed by the builds and hardware they measured."""

    prove: Fit
    verify: Fit
    stage: Fit
    cpu: float
    key: Mapping[str, str] = field(default_factory=dict)

    def model(self, *, source: str = "") -> CostModel:
        p, v, s = self.prove.coef, self.verify.coef, self.stage.coef

        def rate(f: str) -> float:
            return p.get(f, 0.0) + (v.get(f, 0.0) + s.get(f, 0.0)) * self.cpu

        return CostModel(per_and=rate("ands*drawn"), statement_fixed=rate("one"), statement_per_and=rate("ands"),
                         staged_per_and=rate("ands*staged"),
                         sigma=max(self.prove.sigma, self.verify.sigma, self.stage.sigma), source=source)

    def record(self, observations: Sequence[Observation]) -> dict:
        return {"kind": "cost-model/v1", "key": dict(self.key), "cpu": self.cpu,
                "fits": {f.target: {"coef": dict(f.coef), "sigma": f.sigma, "points": f.points,
                                    "sources": list(f.sources), "pinned": list(f.pinned)}
                         for f in (self.prove, self.verify, self.stage)},
                "observations": [asdict(o) for o in observations]}


def from_record(rec: Mapping) -> Fitted:
    if rec.get("kind") != "cost-model/v1":
        raise ValueError(f"not a cost-model/v1 record: {rec.get('kind')!r}")
    fits = {t: Fit(t, dict(f["coef"]), f["sigma"], f["points"], tuple(f["sources"]), tuple(f.get("pinned", ())))
            for t, f in rec["fits"].items()}
    return Fitted(fits["prove"], fits["verify"], fits["stage"], rec["cpu"], dict(rec.get("key", {})))


__all__ = ["FEATURES", "Fit", "Fitted", "Observation", "TARGETS", "fit", "from_record"]
