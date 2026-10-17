"""ε for a PoUW design in the tensor–scalar model (note:20261017T0310Z-report-eps1-program §2), every term from data:

    1 − ε = (1 − ε_skip − ε_mand − ε_machine − ε_luck) · (1 − ε_sched) · (1 − ε_clock)

each term a fraction of honest time.

- **Π** is a named vector's rows at its epoch, as the price LP reads them: row r holds Σ_k c_kr·n_k SMSP-cycles for n_k warp
  instructions of class k, and a program takes at least its largest row.
- **ε_machine** is one LP over Π (SPEC's §0.8, note:20261017T0324Z-draft-eps1-spec). A program writes every committed
  word, folds every word and does the filler, and takes T ≥ every row. A word is written on one of four routes: `tensor`
  (1/128 of the MMA with its operand delivery, folded on lanes), `lanes` (q lane instructions a word in any mix of the
  CUDA classes, NOROW's componentwise minimum among them, folded on lanes), `baseline` (the design's `[baseline]`
  program, dp4a, folded on lanes) or `uniform` (q_u uniform instructions a word, folded there too). Each filler runs on
  lanes (its class mix) or on the uniform datapath (one uniform instruction a lane instruction). The cheater minimizes
  T at q = q* over every route. Honest is the same LP with its own routes: the tensor route at its own operand cost, the
  baseline program (whose words belong to the baseline, not to ε: the brief §2) and its filler where it runs it.
  ε_machine = 1 − T_cheat/T_honest, and against honest as built (the tensor route alone) beside it. The cheater's
  operand delivery is priced 0 (`free`, SPEC's conservative default), at the noise's floor φ_del (`floor`: the
  design's `[operands]`, else the profile's), or at honest's own (`honest`). The LP's duals are the certificate π*, at
  which ε_machine splits exactly into ε_cert (1 − π*·n(H)/B(H)) and ε_route. With the scalar routes closed it is
  ε_machine's floor, which no q* lowers; the threshold is the least q* at which ε_machine reaches it.
- **The credit** is honest's kernel (`H`, the master inequality) or the floor program ♭ (`floor`, SPEC §0.7a's
  Corollary D3): the same LP restricted to honest's routes with the tensor route at φ_del and no overhead. Under ♭,
  ε_sched, ε_clock and honest's overhead above ♭ leave ε for γ, as the factor T_H/B_♭.
- **ε_luck** is Theorem S inside the same LP: an accepted cheater writes (1 − d̄) of the words and folds them all, d̄ =
  ln(1/δ)/((1 − p)·S) over S spot checks (note:20261016T0005Z-draft-luck-loss-round4), and ε_luck is what that saves
  it, d̄·w*/B(H) at π*. The terms add up to the LP exactly.
- **ε_sched** is honest's measured time above its model time (`[measured]` in data/honest/), when the design runs the
  profile's fold at its L (with its fillers, the measured time scaled by their model cost). Unmeasured, it is 0: honest
  is certified at the model's bound, which moves it into γ.
- **γ** is honest's time a useful MMA (`work` folded MMAs and `plain` unfolded ones) over the plain GEMM's model time.
  Honest's time is measured when the design runs the profile's kernel, scaled by the model's cost of the design's
  fillers when it adds some, and its fastest program's model time otherwise.
- **ε_skip**, **ε_mand** and **ε_clock** are the design's accounting (data/designs/), each part with its source.

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
MMA = "MMA"
ROUTES = ("tensor", "lanes", "baseline", "uniform")
OPERANDS = ("free", "floor", "honest")
CREDITS = ("H", "floor")


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
    floor: dict[str, float]      # the least operand delivery any program pays a MMA, warp instructions by class
    floor_source: str

    def base(self) -> dict[str, float]:
        """Warp instructions a MMA without the fold."""
        per_mma = self.words / LANES * self.depth / self.L
        out = {k: v - self.fold.get(k, 0.0) * per_mma for k, v in self.per_mma.items()}
        if any(v < -1e-12 for v in out.values()) or set(self.fold) - set(self.per_mma):
            raise SystemExit(f"{self.name}: the fold is not inside per_mma")
        return {k: v for k, v in out.items() if v > 1e-12}

    def tensor(self, operands: str, floor: dict[str, float] | None = None) -> dict[str, float]:
        """The tensor route a MMA: the MMA with its operand delivery free, at the floor (the design's, else this
        profile's), or at honest's."""
        if operands == "honest":
            return self.base()
        if operands not in OPERANDS:
            raise SystemExit(f"operands: one of {', '.join(OPERANDS)}")
        floor = self.floor if floor is None else floor
        if operands == "floor" and not floor:
            raise SystemExit(f"{self.name}: no [operands] floor")
        out = {MMA: self.per_mma[MMA]}
        _add(out, floor if operands == "floor" else {}, 1.0)
        return out


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


def lane_fill(x: float) -> Filler:
    """x lane instructions a word of verified filler on lanes, LOP3 and IMAD alternating (FILL-OPT)."""
    return Filler(per_word=x, mix={"ALU": 1.0, "IMAD": 1.0}, route="lanes",
                  source="CALC: a lane chain of LOP3 and IMAD, receipts checked like the fold's (candidate)")


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
    filler: tuple[Filler, ...]
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
    floor: dict[str, float]      # the noise's operand-delivery floor φ_del a MMA, warp instructions by class
    floor_source: str
    plain: float = 0.0           # plain MMAs a useful MMA beside `work`, unfolded (CO-Dither's strip): γ only
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
        if any(f.per_word == "saturate" for f in self.filler):
            raise SystemExit(f"{self.name} has UPFILL already")
        return replace(self, name=f"{self.name}+up", filler=(*self.filler, UPFILL))

    def with_fill(self, x: float) -> Design:
        """The design with x lane instructions a word of verified filler on lanes (`lane_fill`)."""
        return replace(self, name=f"{self.name}+fill{x:g}", filler=(*self.filler, lane_fill(x)))


def _parts(t: dict) -> dict[str, Part]:
    return {k: Part(float(v["eps"]), v["source"]) for k, v in t.items()}


def honest(name: str, data: Path = DATA) -> Honest:
    p = data / "honest" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no honest profile {name!r}: one of {', '.join(profiles(data))}")
    t = tomllib.loads(p.read_text())
    fold = dict(t["fold"])
    m, ops = t.get("measured", {}), t.get("operands", {})
    if MMA not in t["per_mma"]:
        raise SystemExit(f"{name}: per_mma has no {MMA}")
    return Honest(name=t["name"], status=t["status"], source=t["source"], L=t["L"], mma=t["mma"]["kind"],
                  words=t["mma"]["words"], depth=t["mma"]["depth"], per_mma=dict(t["per_mma"]),
                  fold_name=fold.pop("name"), fold=fold, measured=m.get("cycles_per_mma"),
                  measured_source=m.get("source", ""), floor=dict(ops.get("floor", {})),
                  floor_source=ops.get("source", ""))


def design(name: str, data: Path = DATA) -> Design:
    p = data / "designs" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no design {name!r}: one of {', '.join(designs(data))}")
    t = tomllib.loads(p.read_text())
    h = honest(t["honest"], data)
    fold = dict(t.get("fold", {"name": h.fold_name, **h.fold}))
    luck, routes, base = t["luck"], t.get("routes", {}), t.get("baseline", {})
    ops = t.get("operands", {"floor": h.floor, "source": h.floor_source})
    f = t.get("filler", {})
    filler = Filler(per_word=f.get("per_word", 0.0), mix=f.get("mix"), route=f.get("route", "lanes"),
                    source=f.get("source", ""))
    if filler.route not in ("lanes", "uniform") or (filler.per_word and not filler.mix):
        raise SystemExit(f"{name}: the filler runs on lanes or uniform, with its class mix on lanes")
    return Design(name=t["name"], summary=t["summary"], honest=t["honest"], L=t["L"], work=t.get("work", 1.0),
                  fold_name=fold.pop("name"), fold=fold, filler=(filler,) if filler.per_word else (),
                  skip=_parts(t.get("skip", {})),
                  mand=_parts(t.get("mand", {})), clock=Part(float(t["clock"]["eps"]), t["clock"]["source"]),
                  delta=luck["delta"], p=luck["p"], checks=luck["checks"], luck_source=luck["source"],
                  baseline=dict(base.get("program", {})), baseline_name=base.get("name", ""),
                  baseline_source=base.get("source", ""), floor=dict(ops.get("floor", {})),
                  floor_source=ops.get("source", ""), plain=t.get("plain", 0.0), lanes=routes.get("lanes", True),
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
    filler_uniform: tuple[float, ...]  # each filler's share on the uniform datapath
    n: dict[str, float]               # warp (or uniform) instructions a word by class
    binding: tuple[str, ...]
    pi: dict[str, float]              # the certificate: each row's dual weight, summing to 1

    def price(self, pi: Machine, n: dict[str, float]) -> float:
        """π·n: the certificate's price of n, in SMSP-cycles."""
        load = pi.load(n)
        return sum(w * load[r] for r, w in self.pi.items())


def fastest(d: Design, h: Honest, pi: Machine, *, q: float | None, q_uniform: float | None, baseline: bool,
            fillers: tuple[tuple[Filler, float], ...] = (), cheat: bool = False,
            tensor_mma: dict[str, float] | None = None, deficit: float = 0.0) -> Mix:
    """The fastest mix over Π: min T over [T, w_tensor, w_lanes, w_baseline, w_uniform, s_k (k in pi.cuda), a fold
    alone on lanes, a fold alone on the uniform datapath, then each filler on lanes and on the uniform datapath]. The
    routes write 1 − `deficit` of the words and every word is folded, the rest alone. Each filler (Filler, lane
    instructions a word) runs once, on its own route unless `cheat`. The tensor route pays `tensor_mma` a MMA (default
    honest's). q None closes the lane route, q_uniform None the uniform route (and the fold alone there), and
    `baseline` opens the design's baseline program."""
    tensor: dict[str, float] = {}
    _add(tensor, h.base() if tensor_mma is None else tensor_mma, d.L / (h.depth * h.words))
    _add(tensor, d.fold, 1 / LANES)
    on_lanes = {k: v / LANES for k, v in d.fold.items()}
    base: dict[str, float] = {}
    _add(base, d.baseline, 1 / LANES)
    _add(base, d.fold, 1 / LANES)
    fl = [{k: pw * v / sum(f.mix.values()) / LANES for k, v in f.mix.items()} for f, pw in fillers]
    S = list(pi.cuda)
    nv = 7 + len(S) + 2 * len(fillers)
    iT, it, il, ib, iu, i_xv, i_xu = 0, 1, 2, 3, 4, 5 + len(S), 6 + len(S)
    fv = [7 + len(S) + 2 * j for j in range(len(fillers))]
    fu = [8 + len(S) + 2 * j for j in range(len(fillers))]

    def col(k: str) -> list[float]:
        a = [0.0] * nv
        a[it] = tensor.get(k, 0.0)
        a[il] = a[i_xv] = on_lanes.get(k, 0.0)
        a[ib] = base.get(k, 0.0)
        if k in S:
            a[5 + S.index(k)] = 1.0
        for j, (_, pw) in enumerate(fillers):
            a[fv[j]] = fl[j].get(k, 0.0)
            if k == UNI:
                a[fu[j]] = pw
        if k == UNI:
            a[iu] = (q_uniform or 0.0) + d.fold_cost()
            a[i_xu] = d.fold_cost()
        return a

    cols = {k: col(k) for k in set(tensor) | set(on_lanes) | set(base) | set(S) | {k for f in fl for k in f} | {UNI}}
    missing = set(cols) - set(pi.c)
    if missing:
        raise SystemExit(f"{pi.vector}: no census line for {', '.join(sorted(missing))}")
    A_ub, b_ub, rows = [], [], []
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
            rows.append(row)
    eq = [[0.0] * nv for _ in range(3 + len(fillers))]
    eq[0][it] = eq[0][il] = eq[0][ib] = eq[0][iu] = 1.0   # 1 − d̄ of the words written
    eq[1][i_xv] = eq[1][i_xu] = 1.0                       # the rest folded alone
    for i in range(len(S)):
        eq[2][5 + i] = 1.0
    eq[2][il] = -(q or 0.0) / LANES                       # a word on lanes: q lane instructions in any CUDA mix
    for j in range(len(fillers)):
        eq[3 + j][fv[j]] = eq[3 + j][fu[j]] = 1.0         # each filler done once
    bounds = [(0, None)] * nv
    if q is None:
        bounds[il] = (0, 0)
    if not baseline or not d.baseline:
        bounds[ib] = (0, 0)
    if q_uniform is None:
        bounds[iu] = bounds[i_xu] = (0, 0)
    for j, (f, _) in enumerate(fillers):
        if not cheat:
            bounds[fu[j] if f.route == "lanes" else fv[j]] = (0, 0)
    obj = [0.0] * nv
    obj[iT] = 1.0
    s = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=eq, b_eq=[1.0 - deficit, deficit, 0.0] + [1.0] * len(fillers), bounds=bounds,
                method="highs")
    if s.status:
        raise SystemExit(f"{d.name}: the LP failed ({s.message})")
    x = [float(v) for v in s.x]
    n = {k: sum(a * y for a, y in zip(ck, x)) for k, ck in cols.items()}
    n = {k: v for k, v in n.items() if v > 1e-12}
    lam = {r: -float(m) for r, m in zip(rows, s.ineqlin.marginals) if -m > 1e-9}
    return Mix(T=x[iT], words={"tensor": x[it], "lanes": x[il], "baseline": x[ib], "uniform": x[iu]},
               filler_uniform=tuple(x[i] for i in fu), n=n, binding=binding(pi.load(n), x[iT]), pi=lam)


def deficit(d: Design) -> float:
    """Theorem S: Pr[accept ∧ deficiency ≥ D] ≤ e^{−(1−p)D}, so at acceptance δ an accepted cheater skips at most
    d̄ = ln(1/δ)/((1 − p)·S) of the words."""
    return math.log(1 / d.delta) / ((1 - d.p) * d.checks)


@dataclass(frozen=True)
class Setup:
    """A design on Π with its credited side solved: the cheater's LP at any q* reads it."""
    design: Design
    honest: Honest
    machine: Machine
    operands: str
    credit: str                      # H (honest's kernel) or floor (the floor program ♭)
    fillers: tuple[tuple[Filler, float], ...]   # each filler with its lane instructions a word
    as_built: Mix                    # the credited kernel: the tensor route alone
    baseline: Mix                    # the credited program: the tensor route and the baseline program

    def cheat(self, q: float, *, lanes: bool | None = None, uniform: bool | None = None,
              q_uniform: float | None = None, luck: bool = False) -> Mix:
        d = self.design
        lanes = d.lanes if lanes is None else lanes
        uniform = d.uniform if uniform is None else uniform
        qu = q_uniform if q_uniform is not None else (d.q_uniform if d.q_uniform is not None else q)
        return fastest(d, self.honest, self.machine, q=q if lanes else None, q_uniform=qu if uniform else None,
                       baseline=True, fillers=self.fillers, cheat=True,
                       tensor_mma=self.honest.tensor(self.operands, d.floor), deficit=deficit(d) if luck else 0.0)

    def machine_eps(self, q: float, **routes) -> float:
        return max(0.0, 1 - self.cheat(q, **routes).T / self.baseline.T)

    def floor(self) -> float:
        """ε_machine with the scalar routes closed: what operand delivery and the baseline leave at every q*."""
        return self.machine_eps(0.0, lanes=False, uniform=False)


def setup(d: Design, pi: Machine, data: Path = DATA, operands: str = "free", credit: str = "H") -> Setup:
    """`credit` H compares the cheater with honest's kernel; floor with the floor program ♭ (SPEC §0.7a): the tensor
    route at the noise's φ_del, the fold, the fillers on their own routes and the baseline program, with no overhead.
    A saturating filler fills the uniform pipe's slack in the credited program's time."""
    h = honest(d.honest, data)
    if d.L % h.depth:
        raise SystemExit(f"{d.name}: L = {d.L} is not a multiple of the MMA's depth {h.depth}")
    if credit not in CREDITS:
        raise SystemExit(f"credit: one of {', '.join(CREDITS)}")
    if credit == "floor" and operands == "honest":
        raise SystemExit("the floor program delivers operands at φ_del: --operands honest reads honest's credit only")
    tm = h.tensor("floor", d.floor) if credit == "floor" else None
    fixed = tuple((f, float(f.per_word)) for f in d.filler if f.per_word != "saturate")
    fillers = []
    for f in d.filler:
        if f.per_word == "saturate":
            if f.route != "uniform":
                raise SystemExit(f"{d.name}: only a uniform filler saturates")
            m = fastest(d, h, pi, q=None, q_uniform=None, baseline=True, fillers=fixed, tensor_mma=tm)
            fillers.append((f, max(0.0, m.T - pi.load(m.n)["UP"]) / pi.c[UNI]["UP"]))
        else:
            fillers.append((f, float(f.per_word)))
    fs = tuple(fillers)
    built = fastest(d, h, pi, q=None, q_uniform=None, baseline=False, fillers=fs, tensor_mma=tm)
    base = fastest(d, h, pi, q=None, q_uniform=None, baseline=True, fillers=fs, tensor_mma=tm)
    return Setup(design=d, honest=h, machine=pi, operands=operands, credit=credit, fillers=fs, as_built=built,
                 baseline=base)


@dataclass(frozen=True)
class Eps:
    design: Design
    honest: Honest
    machine: str
    operands: str
    credit: str
    q: float
    as_built: Mix
    baseline: Mix                    # the credited program: ε's denominator
    cheat: Mix                       # the cheater writing every word: ε_machine
    lucky: Mix                       # the cheater writing 1 − d̄ of them: ε_luck
    fillers: tuple[tuple[Filler, float], ...]
    T_honest_measured: float | None  # measured SMSP-cycles a committed word
    T_honest: float                  # honest's SMSP-cycles a committed word: measured, else its fastest program's model
    T_honest_source: str
    gamma: float                     # honest's time a useful MMA (`work` of them, `plain` plain) over the plain GEMM's
    sched: float | None
    skip: float
    mand: float
    machine_eps: float               # 1 − T_cheat/T_baseline
    machine_as_built: float          # 1 − T_cheat/T_as_built
    cert: float                      # ε_cert(π*): 1 − π*·n(H)/B(H), π* the cheater's certificate
    floor: float                     # ε_machine with the scalar routes closed
    deficit: float                   # d̄
    luck: float                      # (T_cheat − T_lucky)/T_baseline
    clock: float

    @property
    def route(self) -> float:
        """ε_route(π*) = ε_machine − ε_cert(π*)."""
        return self.machine_eps - self.cert

    @property
    def credit_ratio(self) -> float:
        """Honest's time over the credited program's: under the floor credit, the factor that leaves ε for γ."""
        return self.T_honest / self.baseline.T

    def total(self, sched: bool = True) -> float:
        s = (self.sched or 0.0) if sched else 0.0
        return 1 - (1 - self.skip - self.mand - self.machine_eps - self.luck) * (1 - s) * (1 - self.clock)

    def terms(self) -> dict[str, float | None]:
        return {"sched": self.sched, "skip": self.skip, "mand": self.mand, "machine": self.machine_eps,
                "luck": self.luck, "clock": self.clock}


def eps(d: Design, pi: Machine, q: float, data: Path = DATA, *, operands: str = "free", credit: str = "H",
        lanes: bool | None = None, uniform: bool | None = None, q_uniform: float | None = None) -> Eps:
    """Under the floor credit ε_sched and ε_clock leave ε (honest's time above ♭ is in γ, `credit_ratio`)."""
    s = setup(d, pi, data, operands, credit)
    hs = s if credit == "H" else setup(d, pi, data, "free")
    h, base = s.honest, s.baseline
    routes = {"lanes": lanes, "uniform": uniform, "q_uniform": q_uniform}
    c, lucky = s.cheat(q, **routes), s.cheat(q, luck=True, **routes)
    words_per_mma = h.words * h.depth / d.L
    bare = replace(d, fold={}, filler=(), L=h.depth)
    plain = fastest(bare, h, pi, q=None, q_uniform=None, baseline=True)
    plain_mma = plain.T * h.words
    kernel = d.L == h.L and d.fold == h.fold and h.measured is not None
    measured = h.measured / words_per_mma if (kernel and not d.filler) else None
    if kernel and d.filler:
        bare_fill = setup(replace(d, filler=()), pi, data).baseline.T
        t_h, t_src = h.measured / words_per_mma * hs.baseline.T / bare_fill, "measured × the model's filler cost"
    else:
        t_h, t_src = (measured, "measured") if measured else (hs.baseline.T, "model")
    machine_eps = max(0.0, 1 - c.T / base.T)
    return Eps(design=d, honest=h, machine=pi.vector, operands=operands, credit=credit, q=q, as_built=s.as_built,
               baseline=base, cheat=c, lucky=lucky, fillers=s.fillers, T_honest_measured=measured, T_honest=t_h,
               T_honest_source=t_src,
               gamma=d.work * t_h * words_per_mma / plain_mma + d.plain,
               sched=(1 - hs.as_built.T / t_h) if (kernel and credit == "H") else None,
               skip=sum(p.eps for p in d.skip.values()),
               mand=sum(p.eps for p in d.mand.values()), machine_eps=machine_eps,
               machine_as_built=max(0.0, 1 - c.T / s.as_built.T),
               cert=min(machine_eps, max(0.0, 1 - c.price(pi, base.n) / base.T)), floor=s.floor(),
               deficit=deficit(d), luck=max(0.0, (c.T - lucky.T) / base.T),
               clock=d.clock.eps if credit == "H" else 0.0)


def threshold(s: Setup, hi: float = 64.0, tol: float = 1e-9, **routes) -> float | None:
    """The least q* at which ε_machine reaches its floor (to 1e-6), or None when it hasn't at q* = hi."""
    floor = s.floor()
    f = lambda q: s.machine_eps(q, **routes) - floor
    if f(hi) > tol:
        return None
    lo = 0.0
    if f(lo) <= tol:
        return lo
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if f(mid) <= tol else (mid, hi)
    return hi


QS = [0.5 * i for i in range(1, 33)]   # q* in [0.5, 16]

