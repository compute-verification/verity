"""n(H)(m, n, k): an honest kernel's executed warp instructions by census class as a closed form in the GEMM's shape
(data/counts/NAME.toml), its check against ncu, and the model bound it certifies at any shape.

The kernel (egemm.cu's e_body) runs C persistent CTAs over the T = (m/BM)(n/BN) tiles, each tile k/BK stages and a peel,
and what it issues depends on the shape alone, so each class c's count is

    n_c(m, n, k) = stage_c · T · k/BK + tile_c · T + cross_c · X + launch_c

with stage_c a stage's instructions over a CTA's warps, tile_c a tile's outside its stages (the peel's included), cross_c
a raster group crossing's and launch_c the grid's outside its tiles, one set a variant: k ≤ k3_above runs the kernel, a
longer k its _k3 twin (the 3-atom peel). X = `crossings(m, n)`: a CTA's tile place advances its group row by a loop of
subtractions (EL::place_step, the same code in both variants, so they share cross_c), one pass a group of 8 tile rows
crossed. What a warp issues because of time is left out: WAIT, a barrier's retry loop, and the re-issue of a per-tile
block when a warp's lanes leave a barrier's poll apart (8237 against 8224 executions of an 8192-execution block at one
shape in two runs). ncu's counts scatter around n by up to their class's `tolerance`; MMA, LDSM and TMA, outside any such
block, match it exactly. The coefficients are `fit`'s on the observations, which `check` holds them to.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verity_machine_model.eps import MMA, IssueCosts, binding, honest
from verity_machine_model.model import DATA

TERMS = ("stage", "tile", "cross", "launch")
GROUP = 8   # the raster's group: tile rows a group


def crossings(m: int, n: int, tile: tuple[int, int, int], ctas: int) -> int:
    """The raster's group crossings over the grid: each CTA's group row at its last tile (it starts at 0, and its tiles
    are its index plus multiples of the grid)."""
    tn = n // tile[1]
    t = m // tile[0] * tn
    return sum((b + ctas * ((t - 1 - b) // ctas)) // (GROUP * tn) for b in range(ctas))


@dataclass(frozen=True)
class Obs:
    """One ncu profile of the kernel: its run, shape and executed warp instructions by class, WAIT left out."""
    run: str
    shape: tuple[int, int, int]
    counts: dict[str, int]


@dataclass(frozen=True)
class Counts:
    name: str
    kernel: str
    source: str
    ctas: int
    tile: tuple[int, int, int]                       # BM, BN, BK
    k3_above: int
    shapes: tuple[int, int, int, int]                # m, n and k multiples, and k's largest: the shapes the kernel runs
    tolerance: dict[str, float]                      # |ncu − n| ≤ tolerance · n by class (else `default`)
    exact: tuple[str, ...]                           # the classes ncu matches exactly
    bound_tolerance: float                           # the model bound at n against at ncu's counts, relative
    variants: dict[str, dict[str, dict[str, float]]]  # variant -> term -> class -> coefficient
    obs: tuple[Obs, ...]
    form: str = MMA                                  # the census kind its MMAs are priced as (its honest profile's)

    def variant(self, k: int) -> str:
        return "k3" if k > self.k3_above else "k2"

    def terms(self, m: int, n: int, k: int) -> dict[str, int]:
        bm, bn, bk = self.tile
        mm, mn, mk, kmax = self.shapes
        if m % mm or n % mn or k % mk or not 0 < k <= kmax:
            raise SystemExit(f"{self.kernel} runs m % {mm}, n % {mn}, k % {mk} == 0 and k <= {kmax}, not {m} x {n} x {k}")
        t = m // bm * (n // bn)
        if t < self.ctas:
            raise SystemExit(f"{self.name}: {t} tiles leave some of the {self.ctas} CTAs idle, which n does not cover")
        return {"stage": t * (k // bk), "tile": t, "cross": crossings(m, n, self.tile, self.ctas), "launch": 1}

    def n(self, m: int, n: int, k: int) -> dict[str, float]:
        """Each class's executed warp instructions at m x n x k."""
        x, v = self.terms(m, n, k), self.variants[self.variant(k)]
        classes = sorted({c for term in TERMS for c in v[term]})
        return {c: sum(v[term].get(c, 0) * x[term] for term in TERMS) for c in classes}

    def per_mma(self, m: int, n: int, k: int) -> dict[str, float]:
        c = self.n(m, n, k)
        return {k_: v / c[MMA] for k_, v in c.items()}

    def priced(self, per: dict[str, float]) -> dict[str, float]:
        """Counts by class with the MMAs under their form's class, as the ε LP prices them."""
        return {self.form if c == MMA else c: v for c, v in per.items()}

    def bound(self, pi: IssueCosts, m: int, n: int, k: int) -> tuple[float, tuple[str, ...]]:
        """The model's time for the kernel at m x n x k, SMSP-cycles a MMA, and its binding rows."""
        load = pi.load(self.priced(self.per_mma(m, n, k)))
        t = max(load.values())
        return t, binding(load, t)

    def check(self) -> list[str]:
        """Every observation against n: what is out of tolerance, or inexact in an exact class."""
        bad = []
        for o in self.obs:
            want = self.n(*o.shape)
            for c in sorted(set(want) | set(o.counts)):
                got, w = o.counts.get(c, 0), want.get(c, 0)
                tol = self.tolerance.get(c, self.tolerance["default"])
                if (c in self.exact and abs(got - w) >= 0.5) or abs(got - w) > tol * max(w, 1):
                    bad.append(f"{o.run} {'x'.join(map(str, o.shape))} {c}: ncu {got}, n {w} ({(got - w) / max(w, 1):+.2e})")
        return bad

    def check_bound(self, pi: IssueCosts) -> list[str]:
        """Every observation's model bound at n against at ncu's counts: what is out of `bound_tolerance`."""
        bad = []
        for o in self.obs:
            want, _ = self.bound(pi, *o.shape)
            load = pi.load(self.priced({c: v / o.counts[MMA] for c, v in o.counts.items()}))
            got = max(load.values())
            if abs(want - got) > self.bound_tolerance * got:
                bad.append(f"{o.run} {'x'.join(map(str, o.shape))}: bound at n {want:.4f}, at ncu's counts {got:.4f}")
        return bad


def fit(obs: list[Obs], tile: tuple[int, int, int], ctas: int,
        cross: dict[str, float] | None = None) -> dict[str, dict[str, float]]:
    """One variant's coefficients from its observations: least squares on (T k/BK, T, X, 1), stage_c, tile_c and
    cross_c to 0.01, launch_c to 1; with `cross` given (another variant's), cross_c is that and the rest is fitted."""
    bm, bn, bk = tile
    terms = TERMS if cross is None else tuple(t for t in TERMS if t != "cross")
    rows = []
    for m, n, k in (o.shape for o in obs):
        t = m // bm * (n // bn)
        x = {"stage": t * (k // bk), "tile": t, "cross": crossings(m, n, tile, ctas), "launch": 1}
        rows.append((x["cross"], [x[term] for term in terms]))
    a_ = np.array([r for _, r in rows], float)
    out: dict[str, dict[str, float]] = {t: {} for t in TERMS}
    if cross is not None:
        out["cross"] = dict(cross)
    for c in sorted({c for o in obs for c in o.counts}):
        y = np.array([o.counts.get(c, 0) - (cross or {}).get(c, 0) * x for o, (x, _) in zip(obs, rows)], float)
        sol, *_ = np.linalg.lstsq(a_, y, rcond=None)
        for t, a in zip(terms, sol):
            a = round(float(a), 0 if t == "launch" else 2)
            if a:
                out[t][c] = int(a) if a == int(a) else a
    return out


def load(name: str, data: Path = DATA) -> Counts:
    p = data / "counts" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no closed-form counts {name!r}: one of {', '.join(names(data))}")
    t = tomllib.loads(p.read_text())
    obs = tuple(Obs(run=o["run"], shape=tuple(o["shape"]), counts=dict(o["counts"])) for o in t["obs"])
    return Counts(name=t["name"], kernel=t["kernel"], source=t["source"], ctas=t["ctas"], tile=tuple(t["tile"]),
                  k3_above=t["k3_above"], shapes=tuple(t["shapes"]), tolerance=dict(t["tolerance"]),
                  exact=tuple(t["exact"]), bound_tolerance=t["bound_tolerance"],
                  variants={v: {term: dict(t["variant"][v].get(term, {})) for term in TERMS} for v in t["variant"]},
                  obs=obs, form=honest(t["name"], data).form if (data / "honest" / f"{t['name']}.toml").is_file() else MMA)


def names(data: Path = DATA) -> list[str]:
    return sorted(p.stem for p in (data / "counts").glob("*.toml"))
