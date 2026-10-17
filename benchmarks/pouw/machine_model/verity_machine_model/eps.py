"""ε for a PoUW design in the tensor–scalar model (note:20261017T0310Z-report-eps1-program §2), every term from data:

    1 − ε = (1 − ε_skip − ε_mand − ε_machine − ε_luck) · (1 − ε_sched) · (1 − ε_clock)

each term a fraction of honest time.

- **Π** is a named vector's rows at its epoch, as the price LP reads them: row r holds Σ_k c_kr·n_k SMSP-cycles for n_k warp
  instructions of class k, and a program takes at least its largest row.
- **ε_machine** is one LP over Π. A program writes every committed word, folds every word and does the filler, and takes
  T ≥ every row. A word is written on one of three routes: `tensor` (the honest kernel's cost a MMA, folded on lanes),
  `lanes` (q lane instructions a word in any mix of the CUDA classes, folded on lanes), `baseline` (the design's
  `[baseline]` program, dp4a, folded on lanes) or `uniform` (q_u uniform instructions a word, folded there too). The
  filler runs on lanes (its class mix) or on the uniform datapath (one uniform instruction a lane instruction). The
  cheater minimizes T at q = q* over every route. Honest is the same LP with its own routes: the tensor route, the
  baseline program (whose words belong to the baseline, not to ε: the brief §2) and its filler where it runs it.
  ε_machine = 1 − T_cheat/T_honest, and against honest as built (the tensor route alone) beside it.
- **ε_sched** is honest's measured time above its model time (`[measured]` in data/honest/), when the design runs the
  profile's fold at its L. Unmeasured, it is 0: honest is certified at the model's bound, which moves it into γ.
- **ε_skip**, **ε_mand** and **ε_clock** are the design's accounting (data/designs/), each part with its source.
- **ε_luck** is Theorem S (note:20261016T0005Z-draft-luck-loss-round4): ln(1/δ)/((1 − p)·S) over S spot checks.

A warp instruction does 32 lane instructions; a uniform instruction does one. Every count is per committed word.
"""
from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from scipy.optimize import linprog

from verity_machine_model.model import DATA, census, model, vector

LANES = 32
UNI = "UNI"
ROUTES = ("tensor", "lanes", "baseline", "uniform")


@dataclass(frozen=True)
class Honest:
    """The honest kernel's instruction profile a MMA (data/honest/NAME.toml)."""
    name: str
    status: str                  # `provisional` until PROFILE's measured profile replaces it
    source: str
    L: int                       # the commit spacing the profile was taken at
    mma: str
    words: int                   # output words one MMA writes
    depth: int                   # k of one MMA
    per_mma: dict[str, float]    # warp instructions a MMA by class, everything the kernel issues, the fold included
    fold: dict[str, float]       # the fold inside per_mma: lane instructions a committed word by class
    fold_name: str
    measured: float | None       # honest's measured SMSP-cycles a MMA
    measured_source: str

    def base(self) -> dict[str, float]:
        """Warp instructions a MMA without the fold."""
        per_mma = self.words / LANES * self.depth / self.L
        out = {k: v - self.fold.get(k, 0.0) * per_mma for k, v in self.per_mma.items()}
        if any(v < -1e-12 for v in out.values()) or set(self.fold) - set(self.per_mma):
            raise SystemExit(f"{self.name}: the fold is not inside per_mma")
        return {k: v for k, v in out.items() if v > 1e-12}


@dataclass(frozen=True)
class Part:
    eps: float
    source: str


@dataclass(frozen=True)
class Filler:
    """Verified proof-of-work filler: `per_word` lane instructions a committed word (or "saturate": enough to fill the
    uniform datapath for honest's time), in the class mix `mix` on lanes, run by honest on `route`."""
    per_word: float | str = 0.0
    mix: dict[str, float] | None = None
    route: str = "lanes"         # lanes or uniform
    source: str = ""


UPFILL = Filler(per_word="saturate", mix={"ALU": 1.0, "IMAD": 1.0}, route="uniform",
                source="CALC: a uniform duo-style chain a warp, receipts checked like the fold's (candidate)")


@dataclass(frozen=True)
class Design:
    """A protocol's parameters (data/designs/NAME.toml)."""
    name: str
    summary: str
    honest: str
    L: int
    work: float                  # honest MMAs a useful MMA (NCP's three copies: 3)
    fold: dict[str, float]       # lane instructions a committed word by class
    fold_name: str
    filler: Filler
    skip: dict[str, Part]
    mand: dict[str, Part]
    clock: Part
    delta: float
    p: float
    checks: float
    luck_source: str
    baseline: dict[str, float]   # honest's own lane program for a word (dp4a), lane instructions by class; {}: none
    baseline_name: str
    baseline_source: str
    lanes: bool = True
    uniform: bool = True
    q_uniform: float | None = None   # uniform instructions a word on the uniform datapath; None: q*

    def fold_cost(self) -> float:
        return sum(self.fold.values())

    def with_fold_cost(self, r: float) -> Design:
        """The design with its fold scaled to r lane instructions a word, in the same mix of classes."""
        s = r / self.fold_cost()
        return replace(self, name=f"{self.name}@r{r:g}", fold={k: v * s for k, v in self.fold.items()},
                       fold_name=f"{self.fold_name}×{s:.3g}")

    def with_upfill(self) -> Design:
        """The design with UPFILL, verified filler keeping the uniform datapath busy for honest's time."""
        if self.filler.per_word:
            raise SystemExit(f"{self.name} has a filler already")
        return replace(self, name=f"{self.name}+up", filler=UPFILL)


def _parts(t: dict) -> dict[str, Part]:
    return {k: Part(float(v["eps"]), v["source"]) for k, v in t.items()}


def honest(name: str, data: Path = DATA) -> Honest:
    p = data / "honest" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no honest profile {name!r}: one of {', '.join(profiles(data))}")
    t = tomllib.loads(p.read_text())
    fold = dict(t["fold"])
    m = t.get("measured", {})
    return Honest(name=t["name"], status=t["status"], source=t["source"], L=t["L"], mma=t["mma"]["kind"],
                  words=t["mma"]["words"], depth=t["mma"]["depth"], per_mma=dict(t["per_mma"]),
                  fold_name=fold.pop("name"), fold=fold, measured=m.get("cycles_per_mma"),
                  measured_source=m.get("source", ""))


def design(name: str, data: Path = DATA) -> Design:
    p = data / "designs" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no design {name!r}: one of {', '.join(designs(data))}")
    t = tomllib.loads(p.read_text())
    h = honest(t["honest"], data)
    fold = dict(t.get("fold", {"name": h.fold_name, **h.fold}))
    luck, routes, base = t["luck"], t.get("routes", {}), t.get("baseline", {})
    f = t.get("filler", {})
    filler = Filler(per_word=f.get("per_word", 0.0), mix=f.get("mix"), route=f.get("route", "lanes"),
                    source=f.get("source", ""))
    if filler.route not in ("lanes", "uniform") or (filler.per_word and not filler.mix):
        raise SystemExit(f"{name}: the filler runs on lanes or uniform, with its class mix on lanes")
    return Design(name=t["name"], summary=t["summary"], honest=t["honest"], L=t["L"], work=t.get("work", 1.0),
                  fold_name=fold.pop("name"), fold=fold, filler=filler, skip=_parts(t.get("skip", {})),
                  mand=_parts(t.get("mand", {})), clock=Part(float(t["clock"]["eps"]), t["clock"]["source"]),
                  delta=luck["delta"], p=luck["p"], checks=luck["checks"], luck_source=luck["source"],
                  baseline=dict(base.get("program", {})), baseline_name=base.get("name", ""),
                  baseline_source=base.get("source", ""), lanes=routes.get("lanes", True),
                  uniform=routes.get("uniform", True), q_uniform=routes.get("q_uniform"))


def profiles(data: Path = DATA) -> list[str]:
    return sorted(p.stem for p in (data / "honest").glob("*.toml"))


def designs(data: Path = DATA) -> list[str]:
    return sorted(p.stem for p in (data / "designs").glob("*.toml"))


@dataclass(frozen=True)
class Machine:
    """Π: each class's SMSP-cycles a warp instruction on each of a vector's rows, at its epoch."""
    vector: str
    rows: tuple[str, ...]
    c: dict[str, dict[str, float]]
    cuda: tuple[str, ...]        # the CUDA lane classes a word on lanes may use

    def load(self, n: dict[str, float]) -> dict[str, float]:
        """Each row's SMSP-cycles for n_k warp instructions of each class k."""
        missing = set(n) - set(self.c)
        if missing:
            raise SystemExit(f"{self.vector}: no census line for {', '.join(sorted(missing))}")
        return {r: sum(v * self.c[k].get(r, 0.0) for k, v in n.items()) for r in self.rows}


def machine(name: str = "", data: Path = DATA) -> Machine:
    m = model(data)
    v = vector(name or m.current, data)
    cen = census(m, v.epoch, data)
    c = {k: {r: u / m.cycle_u for r, u in rs.items() if r in v.rows} for k, rs in cen.items()}
    cuda = tuple(k for k in m.cuda if k in c and any(c[k].values()))
    return Machine(vector=v.name, rows=v.rows, c=c, cuda=cuda)


def _add(n: dict[str, float], prog: dict[str, float], x: float) -> None:
    for k, v in prog.items():
        n[k] = n.get(k, 0.0) + v * x


def binding(load: dict[str, float], T: float) -> tuple[str, ...]:
    return tuple(r for r, x in sorted(load.items(), key=lambda t: -t[1]) if x >= T * (1 - 1e-7))


@dataclass(frozen=True)
class Mix:
    T: float                          # SMSP-cycles a committed word
    words: dict[str, float]           # the share of words on each route
    filler_uniform: float             # the share of the filler on the uniform datapath
    n: dict[str, float]               # warp (or uniform) instructions a word by class
    binding: tuple[str, ...]


def fastest(d: Design, h: Honest, pi: Machine, *, q: float | None, q_uniform: float | None, baseline: bool,
            filler_per_word: float, filler_routes: tuple[str, ...]) -> Mix:
    """The fastest mix over Π: min T over [T, w_tensor, w_lanes, w_baseline, w_uniform, s_k (k in pi.cuda), filler on
    lanes, filler on the uniform datapath]. q None closes the lane route, q_uniform None the uniform route, and
    `baseline` opens the design's baseline program."""
    tensor: dict[str, float] = {}
    _add(tensor, h.base(), d.L / (h.depth * h.words))
    _add(tensor, d.fold, 1 / LANES)
    on_lanes = {k: v / LANES for k, v in d.fold.items()}
    base: dict[str, float] = {}
    _add(base, d.baseline, 1 / LANES)
    _add(base, d.fold, 1 / LANES)
    fmix = d.filler.mix or {}
    fsum = sum(fmix.values()) or 1.0
    filler = {k: filler_per_word * v / fsum / LANES for k, v in fmix.items()}
    S = list(pi.cuda)
    nv = 5 + len(S) + 2
    iT, it, il, ib, iu, i_fv, i_fu = 0, 1, 2, 3, 4, 5 + len(S), 6 + len(S)

    def col(k: str) -> list[float]:
        a = [0.0] * nv
        a[it] = tensor.get(k, 0.0)
        a[il] = on_lanes.get(k, 0.0)
        a[ib] = base.get(k, 0.0)
        if k in S:
            a[5 + S.index(k)] = 1.0
        a[i_fv] = filler.get(k, 0.0)
        if k == UNI:
            a[iu] = (q_uniform or 0.0) + d.fold_cost()
            a[i_fu] = filler_per_word
        return a

    cols = {k: col(k) for k in set(tensor) | set(on_lanes) | set(base) | set(S) | set(filler) | {UNI}}
    missing = set(cols) - set(pi.c)
    if missing:
        raise SystemExit(f"{pi.vector}: no census line for {', '.join(sorted(missing))}")
    A_ub, b_ub = [], []
    for row in pi.rows:
        a = [0.0] * nv
        for k, ck in cols.items():
            w = pi.c[k].get(row, 0.0)
            if w:
                a = [x + w * y for x, y in zip(a, ck)]
        if any(a):
            a[iT] = -1.0
            A_ub.append(a)
            b_ub.append(0.0)
    eq = [[0.0] * nv for _ in range(3)]
    eq[0][it] = eq[0][il] = eq[0][ib] = eq[0][iu] = 1.0   # every word written once
    eq[1][i_fv] = eq[1][i_fu] = 1.0                       # the filler done once
    for i in range(len(S)):
        eq[2][5 + i] = 1.0
    eq[2][il] = -(q or 0.0) / LANES                       # a word on lanes: q lane instructions in any CUDA mix
    bounds = [(0, None)] * nv
    if q is None:
        bounds[il] = (0, 0)
    if not baseline or not d.baseline:
        bounds[ib] = (0, 0)
    if q_uniform is None:
        bounds[iu] = (0, 0)
    if "lanes" not in filler_routes:
        bounds[i_fv] = (0, 0)
    if "uniform" not in filler_routes:
        bounds[i_fu] = (0, 0)
    obj = [0.0] * nv
    obj[iT] = 1.0
    s = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=eq, b_eq=[1.0, 1.0, 0.0], bounds=bounds, method="highs")
    if s.status:
        raise SystemExit(f"{d.name}: the LP failed ({s.message})")
    x = [float(v) for v in s.x]
    n = {k: sum(a * y for a, y in zip(ck, x)) for k, ck in cols.items()}
    n = {k: v for k, v in n.items() if v > 1e-12}
    return Mix(T=x[iT], words={"tensor": x[it], "lanes": x[il], "baseline": x[ib], "uniform": x[iu]},
               filler_uniform=x[i_fu], n=n, binding=binding(pi.load(n), x[iT]))


@dataclass(frozen=True)
class Eps:
    design: Design
    honest: Honest
    machine: str
    q: float
    as_built: Mix                    # honest's kernel: the tensor route alone
    baseline: Mix                    # honest's fastest program: the tensor route and the baseline program
    cheat: Mix
    filler_per_word: float
    T_honest_measured: float | None  # measured SMSP-cycles a committed word
    gamma: float                     # the baseline's time a useful MMA over the plain GEMM's fastest, times `work`
    sched: float | None
    skip: float
    mand: float
    machine_eps: float               # 1 − T_cheat/T_baseline
    machine_as_built: float          # 1 − T_cheat/T_as_built
    luck: float
    clock: float

    def total(self, sched: bool = True) -> float:
        s = (self.sched or 0.0) if sched else 0.0
        return 1 - (1 - self.skip - self.mand - self.machine_eps - self.luck) * (1 - s) * (1 - self.clock)

    def terms(self) -> dict[str, float | None]:
        return {"sched": self.sched, "skip": self.skip, "mand": self.mand, "machine": self.machine_eps,
                "luck": self.luck, "clock": self.clock}


def luck(d: Design) -> float:
    """Theorem S: Pr[accept ∧ deficiency ≥ D] ≤ e^{−(1−p)D}, so at acceptance δ the loss is ln(1/δ)/((1 − p)·S)."""
    return math.log(1 / d.delta) / ((1 - d.p) * d.checks)


def eps(d: Design, pi: Machine, q: float, data: Path = DATA, *, lanes: bool | None = None,
        uniform: bool | None = None, q_uniform: float | None = None) -> Eps:
    h = honest(d.honest, data)
    if d.L % h.depth:
        raise SystemExit(f"{d.name}: L = {d.L} is not a multiple of the MMA's depth {h.depth}")
    lanes = d.lanes if lanes is None else lanes
    uniform = d.uniform if uniform is None else uniform
    qu = q_uniform if q_uniform is not None else (d.q_uniform if d.q_uniform is not None else q)
    hon = (d.filler.route,)
    fpw = d.filler.per_word
    if fpw == "saturate":
        if d.filler.route != "uniform":
            raise SystemExit(f"{d.name}: only a uniform filler saturates")
        t = fastest(d, h, pi, q=None, q_uniform=None, baseline=True, filler_per_word=0.0, filler_routes=hon).T
        fpw = t / pi.c[UNI]["UP"]
    fpw = float(fpw)
    built = fastest(d, h, pi, q=None, q_uniform=None, baseline=False, filler_per_word=fpw, filler_routes=hon)
    base = fastest(d, h, pi, q=None, q_uniform=None, baseline=True, filler_per_word=fpw, filler_routes=hon)
    c = fastest(d, h, pi, q=q if lanes else None, q_uniform=qu if uniform else None, baseline=True,
                filler_per_word=fpw, filler_routes=("lanes", "uniform"))
    words_per_mma = h.words * h.depth / d.L
    bare = replace(d, fold={}, filler=Filler(), L=h.depth)
    plain = fastest(bare, h, pi, q=None, q_uniform=None, baseline=True, filler_per_word=0.0, filler_routes=("lanes",))
    plain_mma = plain.T * h.words
    same = d.L == h.L and d.fold == h.fold and not fpw
    measured = h.measured / words_per_mma if (h.measured is not None and same) else None
    return Eps(design=d, honest=h, machine=pi.vector, q=q, as_built=built, baseline=base, cheat=c,
               filler_per_word=fpw, T_honest_measured=measured, gamma=d.work * base.T * words_per_mma / plain_mma,
               sched=(1 - built.T / measured) if measured else None, skip=sum(p.eps for p in d.skip.values()),
               mand=sum(p.eps for p in d.mand.values()), machine_eps=max(0.0, 1 - c.T / base.T),
               machine_as_built=max(0.0, 1 - c.T / built.T), luck=luck(d), clock=d.clock.eps)


def threshold(d: Design, pi: Machine, data: Path = DATA, hi: float = 64.0, tol: float = 1e-9,
              **routes) -> float | None:
    """The least q* at which ε_machine is 0 (to 1e-6), or None when it isn't 0 at q* = hi."""
    f = lambda q: eps(d, pi, q, data, **routes).machine_eps
    if f(hi) > tol:
        return None
    lo = 0.0
    if f(lo) <= tol:
        return lo
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) <= tol else (mid, hi)
    return hi


QS = [1 + 0.5 * i for i in range(31)]   # q* in [1, 16]
