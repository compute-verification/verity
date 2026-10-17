"""ε for a PoUW design in the tensor–scalar model (note:20261017T0310Z-report-eps1-program §2), every term from data,
in two layers that are never mixed (Daniel, 17 Oct 06:29Z). The protocol with 100% checking, every word and operand
verified, has the core ε

    1 − ε_core = (1 − ε_skip − ε_mand − ε_machine) · (1 − ε_sched) · (1 − ε_clock)

and the sampling compiler adds ε_sample = ε_luck + ε_adapt, a function of the proof unit, its budget and its selection
rule, 0 at 100% sampling. Each term is a fraction of honest time.

- **Π** is a named vector's rows at its epoch, as the price LP reads them: row r holds Σ_k c_kr·n_k SMSP-cycles for n_k warp
  instructions of class k, and a program takes at least its largest row.
- **ε_machine** is one LP over Π (SPEC's §0.8, note:20261017T0324Z-draft-eps1-spec). A program writes every committed
  word, folds every word and does the filler, and takes T ≥ every row. A word is written on one of four routes: `tensor`
  (1/128 of the MMA with its operand delivery, folded on lanes), `lanes` (q lane instructions a word in any mix of the
  CUDA classes, NOROW's componentwise minimum among them, folded on lanes), `baseline` (the design's `[baseline]`
  program, dp4a, folded on lanes) or `uniform` (q_u uniform instructions a word, folded there too); the cheater may
  have a fifth, `table` (QSTAR's register-indexed lookups, data/table.toml: q_T lane instructions from the lanes' pool
  and 8 B of L2 a word against an L2 facet at the card's bandwidth, honest's L2 uncharged). Each filler runs on
  lanes (its class mix) or on the uniform datapath (one uniform instruction a lane instruction). The cheater minimizes
  T at q = q* over every route. Honest is the same LP with its own routes: the tensor route at its own operand cost, the
  baseline program (whose words belong to the baseline, not to ε: the brief §2) and its filler where it runs it.
  ε_machine = 1 − T_cheat/T_honest, and against honest as built (the tensor route alone) beside it. The cheater's
  operand delivery is priced 0 (`free`, SPEC's conservative default), at the noise's floor φ_del (`floor`: the
  design's `[operands]`, else the profile's), or at honest's own (`honest`). The LP's duals are the certificate π*, at
  which ε_machine splits exactly into ε_cert (1 − π*·n(H)/B(H)) and ε_route. With the scalar routes closed it is
  ε_machine's floor, which no q* lowers; the threshold is the least q* at which ε_machine reaches it.
- **The credit** is honest's kernel (`H`, the master inequality), its model bound B(H)/f_max (`model`, PROFILE's
  reading: ε_sched and ε_clock leave ε for γ), or the floor program ♭ (`floor`, SPEC §0.7a's Corollary D3): the same LP
  restricted to honest's routes with the tensor route at φ_del and no overhead. Under ♭, ε_sched, ε_clock and honest's
  overhead above ♭ leave ε for γ, as the factor T_H/B_♭ · f_max/f_H (`credit_ratio`; f_max/f_H = 1/(1 − ε_clock)).
- **ε_luck** is Theorem S inside the same LP: an accepted cheater writes (1 − d̄) of the words and folds them all, d̄ =
  ln(1/δ)/((1 − p)·S) over S spot checks (note:20261016T0005Z-draft-luck-loss-round4), and ε_luck is what that saves
  it, d̄·w*/B(H) at π*. The terms add up to the LP exactly.
- **ε_sched** is honest's measured time above its model time (`[measured]` in data/honest/), when the design runs the
  profile's fold at its L (with its fillers, the measured time plus their model cost). Unmeasured, it is 0: honest is
  certified at the model's bound, which moves it into γ.
- **γ** is honest's time a useful MMA (`work` folded MMAs and `plain` unfolded ones) over the plain GEMM's model time.
  Honest's time is measured when the design runs the profile's kernel; otherwise, when the profile is measured, the
  measured time plus the model's difference between the design and the profile's kernel, and else its fastest
  program's model time.
- **ε_skip**, **ε_mand** and **ε_clock** are the design's accounting (data/designs/), each part with its source.
- **ε_adapt** is the proof unit's (data/adapt.toml, SPEC's Theorem D): what steering instances in unproved regions gains
  when σ is drawn after the commitment and before evaluation.

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
ROUTES = ("tensor", "lanes", "baseline", "uniform", "table")
OPERANDS = ("free", "floor", "honest")
CREDITS = ("H", "model", "floor")


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
    cls: str = MMA               # the MMA's census class: sm_120's MMA, a SASS form on another device (IMMA16816)

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
        out = {self.cls: self.per_mma[self.cls]}
        _add(out, floor if operands == "floor" else {}, 1.0)
        return out


@dataclass(frozen=True)
class Part:
    eps: float
    source: str


@dataclass(frozen=True)
class Filler:
    """Verified proof-of-work filler: `per_word` lane instructions a committed word (or "saturate": enough to fill the
    uniform datapath for honest's time), in the class mix `mix` on lanes, run by honest on `route`; a deficient block
    of it passes its check with probability ≤ `p` (SPEC §5: 0.854 for both fillers, any K)."""
    per_word: float | str = 0.0
    mix: dict[str, float] | None = None
    route: str = "lanes"         # lanes or uniform
    source: str = ""
    p: float = 0.854


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
    cls = t["mma"].get("class", MMA)
    if cls not in t["per_mma"]:
        raise SystemExit(f"{name}: per_mma has no {cls}")
    return Honest(name=t["name"], status=t["status"], source=t["source"], L=t["L"], mma=t["mma"]["kind"],
                  words=t["mma"]["words"], depth=t["mma"]["depth"], per_mma=dict(t["per_mma"]),
                  fold_name=fold.pop("name"), fold=fold, measured=m.get("cycles_per_mma"),
                  measured_source=m.get("source", ""), floor=dict(ops.get("floor", {})),
                  floor_source=ops.get("source", ""), cls=cls)


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
                    source=f.get("source", ""), p=f.get("p", Filler.p))
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


@dataclass(frozen=True)
class Table:
    """The table route (data/table.toml): q lane instructions and `bytes` of L2 reads a word, L2 at `l2` bytes a
    second over `sms` SMs at `mhz`; `l2` None (unmeasured on the device) leaves the route bound by its slots alone."""
    q: float
    bytes: float
    l2: float | None
    sms: int
    mhz: float
    source: str

    def per_cycle(self) -> float | None:
        """L2 bytes a SMSP-cycle (4 SMSPs an SM)."""
        return None if self.l2 is None else self.l2 / (self.sms * 4 * self.mhz * 1e6)


def table(data: Path = DATA, q: float | None = None, l2: float | None = None, nbytes: float | None = None) -> Table:
    t = tomllib.loads((data / "table.toml").read_text())
    return Table(q=t["q"] if q is None else q, bytes=t["bytes"] if nbytes is None else nbytes,
                 l2=t.get("l2") if l2 is None else l2, sms=t["sms"], mhz=t["mhz"], source=t["source"])


@dataclass(frozen=True)
class Adapt:
    """ε_adapt for a proof unit (`data/adapt.toml`): what steering in unproved regions gains under the one-σ order,
    with the unit's selection rule."""
    name: str
    eps: float
    source: str
    selection: str = ""


def _adapt_file(data: Path) -> Path:
    """A proof unit's ε_adapt doesn't depend on the device: a device directory reads data/adapt.toml."""
    p = data / "adapt.toml"
    return p if p.is_file() else DATA / "adapt.toml"


def adapt(name: str = "", data: Path = DATA) -> Adapt:
    t = tomllib.loads(_adapt_file(data).read_text())
    name = name or t["default"]
    if name not in t or not isinstance(t[name], dict):
        raise SystemExit(f"no proof unit {name!r} in adapt.toml: one of {', '.join(adapts(data))}")
    return Adapt(name=name, eps=t[name]["eps"], source=t[name]["source"], selection=t[name].get("selection", ""))


def adapts(data: Path = DATA) -> list[str]:
    return [k for k, v in tomllib.loads(_adapt_file(data).read_text()).items() if isinstance(v, dict)]


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


def uniform_on_issue(pi: Machine, cycles: float) -> Machine:
    """Π with each uniform class (UNI, UNI64) also on the issue rows at `cycles` times an ALU instruction's cost there:
    `issue`, and the joint row pricing every other vector-pipe instruction (JI3e_0, JI3n_0), where the census since VF3
    has it on UP alone."""
    c = {k: dict(v) for k, v in pi.c.items()}
    rows = [r for r in pi.rows if r == "issue" or (r.startswith("JI") and r.endswith("_0"))]
    for k in ("UNI", "UNI64"):
        for r in rows:
            if k in c and r in c["ALU"]:
                c[k][r] = cycles * c["ALU"][r]
    return replace(pi, vector=f"{pi.vector} (UNI on issue at {cycles:g})", c=c)


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
    short: dict[str, float]           # the share of words, and of each filler, left undone (Theorem S's deficit)

    def price(self, pi: Machine, n: dict[str, float]) -> float:
        """π·n: the certificate's price of n, in SMSP-cycles (n's L2 bytes are not counted: honest's aren't charged)."""
        load = pi.load(n)
        return sum(w * load.get(r, 0.0) for r, w in self.pi.items())


def fastest(d: Design, h: Honest, pi: Machine, *, q: float | None, q_uniform: float | None, baseline: bool,
            fillers: tuple[tuple[Filler, float], ...] = (), cheat: bool = False,
            tensor_mma: dict[str, float] | None = None, budget: float = 0.0, table: Table | None = None) -> Mix:
    """The fastest mix over Π: min T over [T, w_tensor, w_lanes, w_baseline, w_uniform, s_k (k in pi.cuda), a fold
    alone on lanes, a fold alone on the uniform datapath, each filler on lanes and on the uniform datapath, the words'
    shortfall, each filler's shortfall, w_table]. The routes write every word but the shortfall, and every word is
    folded, the rest alone. Each filler (Filler, lane instructions a word) runs once but for its shortfall, on its own
    route unless `cheat`. Theorem S bounds the shortfalls together: (1 − p)·words' + Σ (1 − p_f)·filler f's ≤ `budget`
    (ln(1/δ)/S; 0 leaves none). The tensor route pays `tensor_mma` a MMA (default honest's). q None closes the lane
    route, q_uniform None the uniform route (and the fold alone there), `baseline` opens the design's baseline
    program, and `table` the table route: table.q lane instructions from the lanes' class pool and table.bytes on an
    L2 facet a word, folded on lanes. A census with no uniform line (sm_89's: ptxas keeps uniform chains on lanes)
    closes the uniform route."""
    uni = UNI in pi.c
    if not uni:
        q_uniform = None
        if any(f.route == "uniform" for f, _ in fillers):
            raise SystemExit(f"{pi.vector}: no {UNI} census line for a filler on the uniform datapath")
    tensor: dict[str, float] = {}
    _add(tensor, h.base() if tensor_mma is None else tensor_mma, d.L / (h.depth * h.words))
    _add(tensor, d.fold, 1 / LANES)
    on_lanes = {k: v / LANES for k, v in d.fold.items()}
    base: dict[str, float] = {}
    _add(base, d.baseline, 1 / LANES)
    _add(base, d.fold, 1 / LANES)
    fl = [{k: pw * v / sum(f.mix.values()) / LANES for k, v in f.mix.items()} for f, pw in fillers]
    S = list(pi.cuda)
    nf = len(fillers)
    nv = 9 + len(S) + 3 * nf
    iT, it, il, ib, iu, i_xv, i_xu = 0, 1, 2, 3, 4, 5 + len(S), 6 + len(S)
    fv = [7 + len(S) + 2 * j for j in range(nf)]
    fu = [8 + len(S) + 2 * j for j in range(nf)]
    i_dw = 7 + len(S) + 2 * nf
    df = [i_dw + 1 + j for j in range(nf)]
    i_tb = nv - 1

    def col(k: str) -> list[float]:
        a = [0.0] * nv
        a[it] = tensor.get(k, 0.0)
        a[il] = a[i_xv] = a[i_tb] = on_lanes.get(k, 0.0)
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

    cols = {k: col(k) for k in set(tensor) | set(on_lanes) | set(base) | set(S) | {k for f in fl for k in f}
            | ({UNI} if uni else set())}
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
    if table is not None and table.l2 is not None:
        a = [0.0] * nv
        a[i_tb], a[iT] = table.bytes / table.per_cycle(), -1.0
        A_ub.append(a)
        b_ub.append(0.0)
        rows.append("L2")
    a = [0.0] * nv
    a[i_dw] = 1 - d.p
    for j, (f, _) in enumerate(fillers):
        a[df[j]] = 1 - f.p
    A_ub.append(a)                                        # Theorem S's budget, after the facets
    b_ub.append(budget)
    eq = [[0.0] * nv for _ in range(3 + nf)]
    eq[0][it] = eq[0][il] = eq[0][ib] = eq[0][iu] = eq[0][i_dw] = eq[0][i_tb] = 1.0   # every word but the shortfall
    eq[1][i_xv] = eq[1][i_xu] = 1.0                       # the shortfall folded alone
    eq[1][i_dw] = -1.0
    for i in range(len(S)):
        eq[2][5 + i] = 1.0
    eq[2][il] = -(q or 0.0) / LANES                       # a word on lanes: q lane instructions in any CUDA mix
    eq[2][i_tb] = -(table.q if table else 0.0) / LANES
    for j in range(nf):
        eq[3 + j][fv[j]] = eq[3 + j][fu[j]] = eq[3 + j][df[j]] = 1.0   # each filler done once but its shortfall
    bounds = [(0, None)] * nv
    for i in (i_dw, *df):
        bounds[i] = (0, 1)
    if q is None:
        bounds[il] = (0, 0)
    if table is None:
        bounds[i_tb] = (0, 0)
    if not baseline or not d.baseline:
        bounds[ib] = (0, 0)
    if q_uniform is None:
        bounds[iu] = bounds[i_xu] = (0, 0)
    for j, (f, _) in enumerate(fillers):
        if not cheat:
            bounds[fu[j] if f.route == "lanes" else fv[j]] = (0, 0)
        if not uni:
            bounds[fu[j]] = (0, 0)
    obj = [0.0] * nv
    obj[iT] = 1.0
    s = linprog(obj, A_ub=A_ub, b_ub=b_ub, A_eq=eq, b_eq=[1.0, 0.0, 0.0] + [1.0] * nf, bounds=bounds, method="highs")
    if s.status:
        raise SystemExit(f"{d.name}: the LP failed ({s.message})")
    x = [float(v) for v in s.x]
    n = {k: sum(a * y for a, y in zip(ck, x)) for k, ck in cols.items()}
    n = {k: v for k, v in n.items() if v > 1e-12}
    lam = {r: -float(m) for r, m in zip(rows, s.ineqlin.marginals) if -m > 1e-9}
    load = pi.load(n)
    if table is not None and table.l2 is not None:
        load["L2"] = x[i_tb] * table.bytes / table.per_cycle()
    return Mix(T=x[iT], words={"tensor": x[it], "lanes": x[il], "baseline": x[ib], "uniform": x[iu], "table": x[i_tb]},
               filler_uniform=tuple(x[i] for i in fu), n=n, binding=binding(load, x[iT]), pi=lam,
               short={"words": x[i_dw], **{f"filler {j + 1}": x[df[j]] for j in range(nf)}})


def budget(d: Design) -> float:
    """Theorem S with each block at its own p: Pr[accept ∧ Σ_w (1 − p_w)·d_w ≥ B] ≤ e^{−B}, so at acceptance δ an
    accepted cheater's deficient blocks, each weighted by 1 − p, are at most ln(1/δ)/S of the S checked ones. This is
    CALC's reading of the theorem's proof (a window passes w.p. ≤ 1 − (1 − p_w)·d_w ≤ e^{−(1 − p_w)·d_w}), which states
    one p; with one p it is the theorem."""
    return math.log(1 / d.delta) / d.checks


def deficit(d: Design) -> float:
    """d̄ = ln(1/δ)/((1 − p)·S): the words an accepted cheater skips when it skips words alone."""
    return budget(d) / (1 - d.p)


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
    table: Table | None = None       # the cheater's table route, when open

    def cheat(self, q: float, *, lanes: bool | None = None, uniform: bool | None = None,
              q_uniform: float | None = None, luck: bool = False) -> Mix:
        d = self.design
        lanes = d.lanes if lanes is None else lanes
        uniform = d.uniform if uniform is None else uniform
        qu = q_uniform if q_uniform is not None else (d.q_uniform if d.q_uniform is not None else q)
        return fastest(d, self.honest, self.machine, q=q if lanes else None, q_uniform=qu if uniform else None,
                       baseline=True, fillers=self.fillers, cheat=True,
                       tensor_mma=self.honest.tensor(self.operands, d.floor), budget=budget(d) if luck else 0.0,
                       table=self.table)

    def machine_eps(self, q: float, **routes) -> float:
        return max(0.0, 1 - self.cheat(q, **routes).T / self.baseline.T)

    def floor(self) -> float:
        """ε_machine with the lane and uniform routes closed: what operand delivery, the baseline and the table route
        (priced by its L2 bytes, not by q*) leave at every q*."""
        return self.machine_eps(0.0, lanes=False, uniform=False)


def setup(d: Design, pi: Machine, data: Path = DATA, operands: str = "free", credit: str = "H",
          table: Table | None = None) -> Setup:
    """`credit` H and model compare the cheater with honest's kernel; floor with the floor program ♭ (SPEC §0.7a):
    the tensor route at the noise's φ_del, the fold, the fillers on their own routes and the baseline program, with no
    overhead. A saturating filler fills the uniform pipe's slack in the credited program's time. `table` opens the
    cheater's table route."""
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
                 baseline=base, table=table)


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
    T_honest: float                  # honest's SMSP-cycles a committed word: measured, else the measured profile's
                                     # plus the model's difference to it, else its fastest program's model
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
    adapt: Adapt | None = None       # the proof unit's ε_adapt; None charges none

    @property
    def route(self) -> float:
        """ε_route(π*) = ε_machine − ε_cert(π*)."""
        return self.machine_eps - self.cert

    @property
    def gamma_clock(self) -> float:
        """γ with the clock gap: honest's time a useful MMA at f_H over the plain GEMM's model time at f_max."""
        return self.gamma / (1 - self.design.clock.eps)

    @property
    def credit_ratio(self) -> float:
        """Honest's time over the credited program's, the credit at f_max under the model and floor credits: the factor
        by which ε_sched, ε_clock and honest's overhead leave ε for γ (SPEC's γ_♭)."""
        r = self.T_honest / self.baseline.T
        return r if self.credit == "H" else r / (1 - self.design.clock.eps)

    def core(self, sched: bool = True) -> float:
        """ε with 100% checking, every word and operand verified (Daniel, 17 Oct 06:29Z): ε_skip, ε_mand, ε_machine and
        the credit terms, with ε_luck = ε_adapt = 0."""
        s = (self.sched or 0.0) if sched else 0.0
        return 1 - (1 - self.skip - self.mand - self.machine_eps) * (1 - s) * (1 - self.clock)

    def sample(self) -> float:
        """ε_sample = ε_luck + ε_adapt, the sampling compiler's: the proof unit's, its budget's and its selection rule's,
        0 at 100% sampling. It is reported beside `core`, never inside it."""
        return self.luck + (self.adapt.eps if self.adapt else 0.0)

    def total(self, sched: bool = True) -> float:
        """The compiled protocol's ε, core and sample in one product; for the record, never for comparing designs."""
        s = (self.sched or 0.0) if sched else 0.0
        a = self.adapt.eps if self.adapt else 0.0
        return 1 - (1 - self.skip - self.mand - self.machine_eps - self.luck - a) * (1 - s) * (1 - self.clock)

    def terms(self) -> dict[str, float | None]:
        return {"sched": self.sched, "skip": self.skip, "mand": self.mand, "machine": self.machine_eps,
                "luck": self.luck, "adapt": self.adapt.eps if self.adapt else None, "clock": self.clock}


def eps(d: Design, pi: Machine, q: float, data: Path = DATA, *, operands: str = "free", credit: str = "H",
        lanes: bool | None = None, uniform: bool | None = None, q_uniform: float | None = None,
        table: Table | None = None, adapt: Adapt | None = None) -> Eps:
    """Under the model and floor credits ε_sched and ε_clock leave ε (honest's time above the credit is in γ,
    `credit_ratio`). `table` opens the cheater's table route; `adapt` charges a proof unit's ε_adapt."""
    s = setup(d, pi, data, operands, credit, table)
    hs = s if credit != "floor" else setup(d, pi, data, "free")
    h, base = s.honest, s.baseline
    routes = {"lanes": lanes, "uniform": uniform, "q_uniform": q_uniform}
    c, lucky = s.cheat(q, **routes), s.cheat(q, luck=True, **routes)
    words_per_mma = h.words * h.depth / d.L
    bare = replace(d, fold={}, filler=(), L=h.depth)
    plain = fastest(bare, h, pi, q=None, q_uniform=None, baseline=True)
    plain_mma = plain.T * h.words
    kernel = d.L == h.L and d.fold == h.fold and h.measured is not None
    measured = h.measured / words_per_mma if (kernel and not d.filler) else None
    if measured:
        t_h, t_src = measured, "measured"
    elif h.measured is not None:
        own = setup(replace(d, L=h.L, fold=h.fold, filler=()), pi, data).baseline.T * h.words * h.depth / h.L
        t_h, t_src = (h.measured - own) / words_per_mma + hs.baseline.T, "measured + the model's difference"
    else:
        t_h, t_src = hs.baseline.T, "model"
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
               clock=d.clock.eps if credit == "H" else 0.0, adapt=adapt)


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

