"""ε for a PoUW design in the tensor–scalar model, every term from data, in two layers that are never mixed. The
protocol with 100% checking, every word and operand verified, has the core ε, Corollary D3's

    1 − ε_core = (1 − ε_skip − ε_mand − ε_machine − ε_luck) · (1 − ε_sched) · (1 − ε_clock)

ε_luck in it: every check is one draw, so an accepted cheater spends Theorem S's budget with every block checked
(LEDGER; JL is not assumed). The sampling compiler adds ε_sample = ε_adapt, a function of the proof block and its
selection rule, 0 at 100% sampling. Each term is a fraction of honest time.

- **Π** is a named vector's rows at its epoch, as the price LP reads them: row r holds Σ_k c_kr·n_k SMSP-cycles for n_k warp
  instructions of class k, and a program takes at least its largest row.
- **ε_machine** is one LP over Π (SPEC's §0.8). A program writes every committed
  word, folds every word and does the filler, and takes T ≥ every row. A word is written on one of four routes: `tensor`
  (on any MMA form of data/forms.toml, L/(depth·words) of that form's instructions with honest's operand delivery a
  word, folded on lanes; honest writes on its own form), `lanes` (q lane instructions a word in any mix of the
  CUDA classes, NOROW's componentwise minimum among them, folded on lanes), `baseline` (the design's `[baseline]`
  program, dp4a, folded on lanes) or `uniform` (q_u uniform instructions a word, folded there too); the cheater may
  have a fifth, `table` (QSTAR's register-indexed lookups, data/table.toml: q_T lane instructions from the lanes' pool
  and 8 B of L2 a word against an L2 facet at the card's bandwidth, honest's L2 uncharged). Honest runs each filler on
  lanes (its class mix); the cheater may run it on the uniform datapath (one uniform instruction a lane instruction)
  when every class of its mix has a uniform form there (the device's `uniform_forms`; sm_90's datapath has no FFMA).
  A design's `[fold_routes]` are its fold's other bit-exact programs (the check reads the fold's result, not its
  program): the cheater folds on lanes by any mix of them and the design's fold, ♭ by the built ones alone (its
  `counted` routes, which no run has built, are the cheater's only), honest by the design's fold (its kernel's). The
  cheater minimizes
  T at q = q* over every route. Honest is the same LP with its own routes: the tensor route at its own operand cost, the
  baseline program (whose words belong to the baseline, not to ε: the brief §2) and its filler where it runs it.
  ε_machine = 1 − T_cheat/T_honest, and against honest as built (the tensor route alone) beside it. The cheater's
  operand delivery is priced 0 (`free`, SPEC's conservative default), at the noise's floor φ_del (`floor`: the
  design's `[operands]`, else the profile's), or at honest's own (`honest`). The LP's duals are the certificate π*, at
  which ε_machine splits exactly into ε_cert (1 − π*·n(H)/B(H)) and ε_route. With the scalar routes closed it is
  ε_machine's floor, which no q* lowers; the threshold is the least q* at which ε_machine reaches it. q* is (I3)
  ScalarRoute's floor: ½ is Lean's (SheetScalarRouteHalf, from A6), and 1.22, decision 89's reading, is the 1.22-slot
  issue floor (Theorem C, unproved in Lean).
- **The credit** is honest's kernel (`H`, the master inequality), its model bound B(H)/f_max (`model`, PROFILE's
  reading: ε_sched and ε_clock leave ε for γ), or the floor program ♭ (`floor`, SPEC §0.7a's Corollary D3): the same LP
  restricted to honest's routes with the tensor route at φ_del and no overhead. Under ♭, ε_sched, ε_clock and honest's
  overhead above ♭ leave ε for γ, as the factor T_H/B_♭ · f_max/f_H (`credit_ratio`; f_max/f_H = 1/(1 − ε_clock)).
- **ε_luck** is Theorem S inside the same LP: an accepted cheater writes (1 − d̄) of the words and folds them all, d̄ =
  ln(1/δ)/((1 − p)·S) over S spot checks, each filler's blocks at their
  own p (`budget`), and ε_luck is what that saves it, d̄·w*/B(H) at π*. The terms add up to the LP exactly.
- **ε_sched** is honest's measured time above its model time (`[measured]` in data/honest/), when the design runs the
  profile's fold at its L (with its fillers, the measured time plus their model cost). Unmeasured, it is 0: honest is
  certified at the model's bound, which moves it into γ.
- **γ** is honest's time a useful MMA (`work` folded MMAs and `plain` unfolded ones) over the plain GEMM's model time.
  Honest's time is measured when the design runs the profile's kernel; otherwise, when the profile is measured, the
  measured time plus the model's difference between the design and the profile's kernel, and else its fastest
  program's model time.
- **ε_skip** is SKIP's calculator's number for the design's noise (benchmarks/pouw/eps_budget.py, `design_skip`), which
  its `[skip.served]` records at the device's current vector, and at another vector V as `at = { V = … }` (♭ moves
  with Π); **ε_mand** and **ε_clock** are the design's accounting (data/designs/), each part with its source.
- **ε_adapt** is the proof block's (data/adapt.toml, SPEC's Theorem D): what steering instances in unproved regions gains
  when σ is drawn after the commitment and before evaluation. A design names its unit (`unit`).

A design is one file in data/designs/, read on top of its `base` and of data/designs.toml (`design`): its name (decision
79's, with the names tonight's notes and data used as `aliases`), the noise on each product (`[noise]`, a leaf of
NOISE for each of PRODUCTS), the fold, the filler, the proof block, the coins and its perplexity arm (`ppl`, a spec in
benchmarks/ppl/specs/), with the ε terms' accounting beside them.

- **The MMA forms** are census kinds, each priced on every row by its own line; a row a form has no line on (VF3q-16's
  EFFMA rows, every row before VF3n-16) reads the generic MMA's there. The generic MMA itself is no route: honest's
  profile names its form (IMMA16832 on sm_120), and the cheater mixes forms.

A warp instruction does 32 lane instructions; a uniform instruction does one. Every count is per committed word.
"""
from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from scipy.optimize import linprog

from verity_machine_model.model import DATA, census, model, vector

LANES = 32
UNI = "UNI"
MMA = "MMA"
ROUTES = ("tensor", "lanes", "baseline", "uniform", "table")
OPERANDS = ("free", "floor", "honest")
CREDITS = ("H", "model", "floor")
PRODUCTS = ("linears", "lm_head", "qk", "pv")     # a served model's products: the linears, lm_head, QKᵀ and P·V
NOISE = {                                          # how a product is noised and served, and what of it is unchecked
    "one-sided": "one operand dithered with the low-rank U, the clean product served: the peeled strip (L of k products "
                 "a word) unchecked",
    "two-sided": "both operands dithered with fresh bits, no U, the last checked word served: nothing unchecked",
    "co32c": "CO-Dither at r = 32, the residue sums' base-128 digits committed: every limb a checked word",
    "cohalf": "CO-Dither at r = k/2 with a 7-bit served side: the strip one checked limb",
    "peel": "P's w64 peel with U: the strip unchecked, all of it free at a one-hot P row (CHECKED-STRIP Result 3)",
    "pv": "P served with a fresh bit, V served, Ycum's base-128 digits committed: every product checked",
    "clean": "served plain, never proved",
}
SUMS = {                                           # the kernel forming a peel's sums DG, S and X(0) (eps_budget prices it)
    "vsum": "BC-KERNEL r2's linear vsum, 35 per-thread instructions an entry",
    "vfold": "D2S-KERNEL r3's vfold, the V producers in one pass: 0.69–0.73 warp instructions an entry",
}
OWN = ("name", "title", "summary")                 # what a design file states itself, never from its base


@dataclass(frozen=True)
class Form:
    """An MMA form a committed word may be written on (data/forms.toml): its census kind, the fresh words and the s8
    k-steps one warp instruction of it writes."""
    kind: str
    words: int
    depth: float                 # k_dense/split: k (k/2 when 2:4 sparse) over the k-slots an exact s8 product takes
    split: int = 1
    sparse: bool = False
    exact: str = ""

    def per_word(self, L: int) -> float:
        """Its warp instructions a committed word at L."""
        return L / (self.depth * self.words)


def forms(data: Path = DATA) -> tuple[Form, ...]:
    """The forms of data/forms.toml; none in a device directory that has no file (its MMA its honest's class)."""
    p = data / "forms.toml"
    if not p.is_file():
        return ()
    t = tomllib.loads(p.read_text())
    out = []
    for kind, f in t.items():
        if not isinstance(f, dict):
            continue
        m, n, k = f["shape"]
        sparse = f.get("sparse", False)
        out.append(Form(kind=kind, words=m * n, depth=(k // 2 if sparse else k) / f["split"], split=f["split"],
                        sparse=sparse, exact=f["exact"]))
    return tuple(out)


@dataclass(frozen=True)
class Honest:
    """The honest kernel's instruction profile a MMA (data/honest/NAME.toml)."""
    name: str
    status: str                  # `provisional` until PROFILE's measured profile replaces it
    source: str
    L: int                       # the commit spacing the profile was taken at
    mma: str
    form: str                    # the MMA's census kind (data/forms.toml), the class per_mma counts it under
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
        out = {self.form: self.per_mma[self.form]}
        _add(out, floor if operands == "floor" else {}, 1.0)
        return out


@dataclass(frozen=True)
class Part:
    eps: float                   # at the device's current vector
    source: str
    at: dict[str, float] = field(default_factory=dict)   # at other vectors, by name


PINNED = ("Accounting.Guarantees.FillerLuck.PinnedFillBlocksDeficit (and PinnedFillLuckAt): the check recomputes the "
          "drawn tile's receipt from committed data and the round's coins and compares it with the committed one, so a "
          "wrong receipt fails whenever its tile is drawn and nothing is left to guess; what a right receipt cost is "
          "FILL-OPT's")
PFILL = ("Accounting.Guarantees.FillerLuck.FillerBlocksDeficit (and FillerLuckAt): pfill's cell check, a short cell "
         "right with probability at most ½ + 2^(−3/2)")


@dataclass(frozen=True)
class Filler:
    """Verified proof-of-work filler: `per_word` lane instructions a committed word in the class mix `mix`, run by
    honest on `route`; a deficient block of it passes its check with probability ≤ `p`, as the theorem `luck` names
    proves. Without one p is 1, no luck credit: the cheater leaves the filler undone at no cost to Theorem S's budget,
    even with every block checked, and the floor program ♭ leaves it out too (decision 99: a filler's credit is the
    theorem its check meets), while honest's time, γ, still counts it. A receipt the check recomputes and compares
    with the committed one has p = 0 (`PINNED`), pfill's cells 0.854 (`PFILL`)."""
    per_word: float = 0.0
    mix: dict[str, float] | None = None
    route: str = "lanes"         # lanes or uniform
    source: str = ""
    p: float = 1.0
    luck: str = ""               # the theorem that proves p < 1


def lane_fill(x: float, mix: dict[str, float] | None = None) -> Filler:
    """x lane instructions a word of verified filler on lanes, LOP3 and IMAD alternating (FILL-OPT), or in `mix`, its
    receipts recomputed by the check (p = 0, `PINNED`)."""
    if mix is None:
        return Filler(per_word=x, mix={"ALU": 1.0, "IMAD": 1.0}, route="lanes",
                      source="CALC: a lane chain of LOP3 and IMAD, receipts checked like the fold's (candidate)",
                      p=0.0, luck=PINNED)
    return Filler(per_word=x, mix=dict(mix), route="lanes",
                  source=f"a lane chain in the mix {', '.join(f'{k} {v:g}' for k, v in mix.items())} (candidate)",
                  p=0.0, luck=PINNED)


@dataclass(frozen=True)
class Design:
    """A protocol's parameters (data/designs/NAME.toml, on its base and data/designs.toml)."""
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
    index_bits: int = 1              # bits a noise entry on the operand a table route indexes (`[table] bits`, else
                                     # the coins')
    title: str = ""              # decision 79's name
    aliases: tuple[str, ...] = ()    # the names tonight's notes and data used
    base: str = ""               # the design this one varies
    ppl: str = ""                # its perplexity arm, benchmarks/ppl/specs/PPL.toml; "": none on main
    noise: dict[str, str] | None = None   # each of PRODUCTS: a leaf of NOISE
    sums: str = "vsum"           # a peel's sums kernel, a key of SUMS
    unit: str = ""               # the proof block, a table of data/adapt.toml
    coin_bits: int = 1           # a noised entry's coin on {0, …, 2^coin_bits − 1}
    coins_source: str = ""
    fold_routes: dict[str, dict[str, float]] = field(default_factory=dict)   # the fold's other bit-exact programs,
                                                                             # lane instructions a word (the cheater's
                                                                             # and, the built ones, ♭'s; not honest's)
    fold_routes_source: str = ""
    counted_routes: tuple[str, ...] = ()     # the fold routes no run has built: the cheater's alone, never ♭'s

    def fold_cost(self) -> float:
        return sum(self.fold.values())

    def built_routes(self) -> dict[str, dict[str, float]]:
        """The fold routes ♭ may fold by: every one a run has built, at its built (or measured) cost."""
        return {n: r for n, r in self.fold_routes.items() if n not in self.counted_routes}

    def with_fold_cost(self, r: float) -> Design:
        """The design with its fold scaled to r lane instructions a word, in the same mix of classes, and each of its
        fold routes by the same factor."""
        s = r / self.fold_cost()
        return replace(self, name=f"{self.name}@r{r:g}", fold={k: v * s for k, v in self.fold.items()},
                       fold_name=f"{self.fold_name}×{s:.3g}",
                       fold_routes={n: {k: v * s for k, v in f.items()} for n, f in self.fold_routes.items()})

    def with_fill(self, x: float, mix: dict[str, float] | None = None) -> Design:
        """The design with x lane instructions a word of verified filler on lanes (`lane_fill`)."""
        return replace(self, name=f"{self.name}+fill{x:g}", filler=(*self.filler, lane_fill(x, mix)))

    def with_index_bits(self, b: int) -> Design:
        """The design with b-bit noise entries on the operand the table route indexes, the other operand as it is."""
        return replace(self, name=f"{self.name}@b{b}", index_bits=b)

    def table(self, t: Table | None) -> Table | None:
        """The table route against this design: b-bit entries on the indexed operand cost b times the 1-bit route's
        lane instructions and bytes a word (QSTAR's b-bit count: ≥ 1.25·b and 8·b B, SPEC §4.2a)."""
        b = self.index_bits
        return t if t is None or b == 1 else replace(t, q=t.q * b, bytes=t.bytes * b)


def _parts(t: dict) -> dict[str, Part]:
    return {k: Part(float(v["eps"]), v["source"], {n: float(x) for n, x in v.get("at", {}).items()})
            for k, v in t.items()}


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
    mma, fs = t["mma"], {f.kind: f for f in forms(data)}
    if fs or "form" in mma:
        f = fs.get(mma.get("form", ""))
        if f is None or (f.words, f.depth) != (mma["words"], mma["depth"]):
            raise SystemExit(f"{name}: [mma] form names its census kind, one of {', '.join(fs)}, with its words and "
                             "depth")
        form = f.kind
    else:
        form = cls
    return Honest(name=t["name"], status=t["status"], source=t["source"], L=t["L"], mma=mma["kind"], form=form,
                  words=mma["words"], depth=mma["depth"],
                  per_mma={form if k == cls else k: v for k, v in t["per_mma"].items()},
                  fold_name=fold.pop("name"), fold=fold, measured=m.get("cycles_per_mma"),
                  measured_source=m.get("source", ""), floor=dict(ops.get("floor", {})),
                  floor_source=ops.get("source", ""), cls=cls)


def _table(name: str, data: Path, seen: tuple[str, ...] = ()) -> dict:
    """A design file read on top of its base's, and the first on top of data/designs.toml: each top-level key of the
    later replaces the earlier's whole, but OWN and `aliases` are never inherited."""
    p = data / "designs" / f"{name}.toml"
    if not p.is_file():
        raise SystemExit(f"no design {name!r}: one of {', '.join(designs(data))} (or an alias: "
                         f"{', '.join(sorted(aliases(data)))})")
    if name in seen:
        raise SystemExit(f"design {name}: its bases loop ({' → '.join((*seen, name))})")
    t = tomllib.loads(p.read_text())
    missing = [k for k in OWN if k not in t]
    if missing or t["name"] != name:
        raise SystemExit(f"designs/{name}.toml: name it {name!r} and give it its own {', '.join(OWN)}")
    under = _table(t["base"], data, (*seen, name)) if t.get("base") else tomllib.loads(
        (data / "designs.toml").read_text())
    return {**{k: v for k, v in under.items() if k not in (*OWN, "aliases", "base")}, **t}


def design(name: str, data: Path = DATA) -> Design:
    """The design NAME, or the design with NAME among its aliases."""
    name = aliases(data).get(name, name)
    t = _table(name, data)
    h = honest(t["honest"], data)
    fold = dict(t.get("fold", {"name": h.fold_name, **h.fold}))
    luck, routes, base = t["luck"], t.get("routes", {}), t.get("baseline", {})
    ops = t.get("operands", {"floor": h.floor, "source": h.floor_source})
    f = t.get("filler", {})
    filler = Filler(per_word=f.get("per_word", 0.0), mix=f.get("mix"), route=f.get("route", "lanes"),
                    source=f.get("source", ""), p=f.get("p", Filler.p), luck=f.get("luck", ""))
    if filler.route != "lanes" or (filler.per_word and not filler.mix):
        raise SystemExit(f"{name}: honest runs a filler on lanes, with its class mix (decision 73: a uniform "
                         "instruction takes an issue slot, so no design fills the uniform datapath)")
    if filler.per_word and ("p" not in f or not 0 <= filler.p <= 1 or (filler.p < 1 and not filler.luck)):
        raise SystemExit(f"{name}: [filler] p is the probability a deficient block passes its check, and `luck` the "
                         f"theorem that proves it: 0 for a receipt the check recomputes ({PINNED.split(' ')[0]}), "
                         f"0.854 for pfill's cells ({PFILL.split(' ')[0]}), or p = 1 with no `luck` (no luck credit)")
    noise = dict(t.get("noise", {}))
    if set(noise) != set(PRODUCTS) or set(noise.values()) - set(NOISE):
        raise SystemExit(f"{name}: [noise] names a leaf of NOISE ({', '.join(NOISE)}) for each of {', '.join(PRODUCTS)}")
    if t.get("sums", "vsum") not in SUMS:
        raise SystemExit(f"{name}: sums {t['sums']!r} is not a peel sums kernel ({', '.join(SUMS)})")
    if t["unit"] not in adapts(data):
        raise SystemExit(f"{name}: unit {t['unit']!r} is not a proof block of adapt.toml ({', '.join(adapts(data))})")
    coins = t["coins"]
    index_bits = t.get("table", {}).get("bits", coins["bits"])
    if index_bits < coins["bits"]:
        raise SystemExit(f"{name}: [table] bits {index_bits} is below the design's {coins['bits']}-bit coins: an entry "
                         "on the indexed operand has at least the coins' bits")
    fr = dict(t.get("fold_routes", {}))
    fr_source = fr.pop("source", "")
    counted = tuple(fr.pop("counted", ()))
    if (fr and not fr_source) or not all(isinstance(v, dict) and v for v in fr.values()):
        raise SystemExit(f"{name}: [fold_routes] names each other bit-exact program of the fold as a table of lane "
                         "instructions a word by class, with one `source`")
    if set(counted) - set(fr) or len(set(counted)) != len(counted):
        raise SystemExit(f"{name}: [fold_routes] `counted` lists the routes no run has built, each once, by their names "
                         f"in the table ({sorted(set(counted) - set(fr))} aren't)")
    return Design(name=t["name"], summary=t["summary"], honest=t["honest"], L=t["L"], work=t.get("work", 1.0),
                  fold_name=fold.pop("name"), fold=fold, filler=(filler,) if filler.per_word else (),
                  skip=_parts(t.get("skip", {})),
                  mand=_parts(t.get("mand", {})), clock=Part(float(t["clock"]["eps"]), t["clock"]["source"]),
                  delta=luck["delta"], p=luck["p"], checks=luck["checks"], luck_source=luck["source"],
                  baseline=dict(base.get("program", {})), baseline_name=base.get("name", ""),
                  baseline_source=base.get("source", ""), floor=dict(ops.get("floor", {})),
                  floor_source=ops.get("source", ""), plain=t.get("plain", 0.0), lanes=routes.get("lanes", True),
                  uniform=routes.get("uniform", True), q_uniform=routes.get("q_uniform"),
                  index_bits=index_bits, title=t["title"],
                  aliases=tuple(t.get("aliases", ())), base=t.get("base", ""), ppl=t.get("ppl", ""), noise=noise,
                  sums=t.get("sums", "vsum"), unit=t["unit"], coin_bits=coins["bits"], coins_source=coins["source"],
                  fold_routes={k: {c: float(x) for c, x in v.items()} for k, v in fr.items()},
                  fold_routes_source=fr_source, counted_routes=counted)


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
    """ε_adapt for a proof block (`data/adapt.toml`): what steering in unproved regions gains under the one-σ order,
    with the unit's selection rule."""
    name: str
    eps: float
    source: str
    selection: str = ""


def _adapt_file(data: Path) -> Path:
    """A proof block's ε_adapt doesn't depend on the device: a device directory reads data/adapt.toml."""
    p = data / "adapt.toml"
    return p if p.is_file() else DATA / "adapt.toml"


def adapt(name: str, data: Path = DATA) -> Adapt:
    t = tomllib.loads(_adapt_file(data).read_text())
    if name not in t or not isinstance(t[name], dict):
        raise SystemExit(f"no proof block {name!r} in adapt.toml: one of {', '.join(adapts(data))}")
    return Adapt(name=name, eps=t[name]["eps"], source=t[name]["source"], selection=t[name].get("selection", ""))


def adapts(data: Path = DATA) -> list[str]:
    return [k for k, v in tomllib.loads(_adapt_file(data).read_text()).items() if isinstance(v, dict)]


def designs(data: Path = DATA) -> list[str]:
    """The design names, their numbers in numeric order (dither-1s-l32 before dither-1s-l128)."""
    key = lambda n: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", n)]
    return sorted((p.stem for p in (data / "designs").glob("*.toml")), key=key)


def aliases(data: Path = DATA) -> dict[str, str]:
    """Each alias, a name tonight's notes and data used, with the design it names."""
    out: dict[str, str] = {}
    for n in designs(data):
        for a in tomllib.loads((data / "designs" / f"{n}.toml").read_text()).get("aliases", ()):
            if a in out or (data / "designs" / f"{a}.toml").is_file():
                raise SystemExit(f"alias {a!r} names two designs ({out.get(a, a)}, {n})")
            out[a] = n
    return out


@dataclass(frozen=True)
class IssueCosts:
    """Π: each class's SMSP-cycles a warp instruction on each of a vector's rows, at its epoch."""
    vector: str
    rows: tuple[str, ...]
    c: dict[str, dict[str, float]]
    cuda: tuple[str, ...]        # the CUDA lane classes a word on lanes may use
    uniform_forms: tuple[str, ...] | None = None   # the lane classes with a uniform-datapath form; None: every one

    def uniform_form(self, f: Filler) -> bool:
        """Whether every class of the filler's mix has a uniform-datapath form, so its work can run there."""
        return self.uniform_forms is None or not set(f.mix or {}) - set(self.uniform_forms)

    def load(self, n: dict[str, float]) -> dict[str, float]:
        """Each row's SMSP-cycles for n_k warp instructions of each class k."""
        missing = set(n) - set(self.c)
        if missing:
            raise SystemExit(f"{self.vector}: no census line for {', '.join(sorted(missing))}")
        return {r: sum(v * self.c[k].get(r, 0.0) for k, v in n.items()) for r in self.rows}


def machine(name: str = "", data: Path = DATA) -> IssueCosts:
    """Π at a vector's epoch, each MMA form on the generic MMA's line wherever it has none of its own."""
    m = model(data)
    v = vector(name or m.current, data)
    cen = census(m, v.epoch, data)
    c = {k: {r: u / m.cycle_u for r, u in rs.items() if r in v.rows} for k, rs in cen.items()}
    for f in forms(data):
        c[f.kind] = {**c.get(MMA, {}), **c.get(f.kind, {})}
    cuda = tuple(k for k in m.cuda if k in c and any(c[k].values()))
    return IssueCosts(vector=v.name, rows=v.rows, c=c, cuda=cuda, uniform_forms=m.uniform_forms)


def uniform_issue_charged(pi: IssueCosts) -> bool:
    """Whether Π's census already charges a uniform instruction on an issue row (VF3p-16 on, decision 73; a DS vector's
    `DS_issue`, A2u)."""
    return any(v > 0 and (r == "issue" or r.endswith("_issue") or r.startswith("JI")) for r, v in pi.c.get("UNI", {}).items())


def uniform_on_issue(pi: IssueCosts, cycles: float) -> IssueCosts:
    """Π with each uniform class (UNI, UNI64) also on the issue rows at `cycles` times an ALU instruction's cost there:
    `issue`, and the joint row pricing every other vector-pipe instruction (JI3e_0, JI3n_0), where the census from VF3
    to VF3o-16 has it on UP alone. Π itself where its census charges it there already, so it is never charged twice."""
    if uniform_issue_charged(pi):
        return pi
    c = {k: dict(v) for k, v in pi.c.items()}
    rows = [r for r in pi.rows if r == "issue" or r.endswith("_issue") or (r.startswith("JI") and r.endswith("_0"))]
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
    forms: dict[str, float]           # the share of words on each MMA form's tensor route
    folds: dict[str, float] = field(default_factory=dict)   # the share of words folded by each of the fold's routes

    def price(self, pi: IssueCosts, n: dict[str, float]) -> float:
        """π·n: the certificate's price of n, in SMSP-cycles (n's L2 bytes are not counted: honest's aren't charged)."""
        load = pi.load(n)
        return sum(w * load.get(r, 0.0) for r, w in self.pi.items())


def fastest(d: Design, h: Honest, pi: IssueCosts, *, q: float | None, q_uniform: float | None, baseline: bool,
            fillers: tuple[tuple[Filler, float], ...] = (), cheat: bool = False,
            tensor_mma: dict[str, float] | None = None, budget: float = 0.0, table: Table | None = None,
            forms: tuple[Form, ...] = (), fold_menu: bool = False) -> Mix:
    """The fastest mix over Π: min T over [T, w_tensor on forms[0], w_lanes, w_baseline, w_uniform, s_k (k in
    pi.cuda), a fold alone on lanes, a fold alone on the uniform datapath, each filler on lanes and on the uniform
    datapath, the words' shortfall, each filler's shortfall, w_table, w_tensor on each later form]. The routes write
    every word but the shortfall, and every word is folded, the rest alone. Each filler (Filler, lane instructions a
    word) runs once but for its shortfall, on its own route unless `cheat`. Theorem S bounds the shortfalls together:
    (1 − p)·words' + Σ (1 − p_f)·filler f's ≤ `budget` (ln(1/δ)/S; 0 leaves none). The tensor route pays `tensor_mma` a
    MMA (default honest's), its MMA on any of `forms` (default honest's own) at that form's instructions a word. q None
    closes the lane route, q_uniform None the uniform route (and the fold alone there), `baseline` opens the design's
    baseline program, and `table` the table route: table.q lane instructions from the lanes' class pool and table.bytes
    on an L2 facet a word, folded on lanes. A census with no uniform line closes the uniform route. The cheater folds a
    word on lanes by the design's fold or any of its `fold_routes`, in any mix (the words folded by each route, after
    the rest: [z_j]), and alone on the uniform datapath at the least of their instruction counts; ♭ (`fold_menu`) folds
    the same way by the built ones alone (`Design.built_routes`), and honest by the design's fold."""
    uni = UNI in pi.c
    if not uni:
        q_uniform = None
        if any(f.route == "uniform" for f, _ in fillers):
            raise SystemExit(f"{pi.vector}: no {UNI} census line for a filler on the uniform datapath")
    own = h.base() if tensor_mma is None else tensor_mma
    rest = {k: v for k, v in own.items() if k != h.form}
    forms = forms or (Form(kind=h.form, words=h.words, depth=h.depth),)
    tensors = []
    for f in forms:
        t: dict[str, float] = {}
        _add(t, rest, d.L / (h.depth * h.words))
        _add(t, {f.kind: own.get(h.form, 0.0)}, f.per_word(d.L))
        _add(t, d.fold, 1 / LANES)
        tensors.append(t)
    on_lanes = {k: v / LANES for k, v in d.fold.items()}
    base: dict[str, float] = {}
    _add(base, d.baseline, 1 / LANES)
    _add(base, d.fold, 1 / LANES)
    fl = [{k: pw * v / sum(f.mix.values()) / LANES for k, v in f.mix.items()} for f, pw in fillers]
    S = list(pi.cuda)
    nf = len(fillers)
    menu = d.fold_routes if cheat else d.built_routes() if fold_menu else {}
    fold_u = min((d.fold_cost(), *(sum(r.values()) for r in menu.values())))
    nv0 = 8 + len(S) + 3 * nf + len(forms)
    nv = nv0 + len(menu)
    iT, it, il, ib, iu, i_xv, i_xu = 0, 1, 2, 3, 4, 5 + len(S), 6 + len(S)
    fv = [7 + len(S) + 2 * j for j in range(nf)]
    fu = [8 + len(S) + 2 * j for j in range(nf)]
    i_dw = 7 + len(S) + 2 * nf
    df = [i_dw + 1 + j for j in range(nf)]
    i_tb = i_dw + 1 + nf
    i_f = [it, *range(i_tb + 1, nv0)]
    iz = list(range(nv0, nv))

    def col(k: str) -> list[float]:
        a = [0.0] * nv
        for i, t in zip(i_f, tensors):
            a[i] = t.get(k, 0.0)
        a[il] = a[i_xv] = a[i_tb] = on_lanes.get(k, 0.0)
        a[ib] = base.get(k, 0.0)
        if k in S:
            a[5 + S.index(k)] = 1.0
        for j, (_, pw) in enumerate(fillers):
            a[fv[j]] = fl[j].get(k, 0.0)
            if k == UNI:
                a[fu[j]] = pw
        if k == UNI:
            a[iu] = (q_uniform or 0.0) + fold_u
            a[i_xu] = fold_u
        for i, r in zip(iz, menu.values()):                # a word folded by route j in place of the design's fold
            a[i] = (r.get(k, 0.0) - d.fold.get(k, 0.0)) / LANES
        return a

    # Sorted: each facet's row sums the classes in this order, and in a set's (string-hash) order the last bits, and on a
    # tie face the LP's vertex, would vary from one interpreter to the next.
    cols = {k: col(k) for k in sorted({k for t in tensors for k in t} | set(on_lanes) | set(base) | set(S)
                                      | {k for f in fl for k in f} | ({UNI} if uni else set())
                                      | {k for r in menu.values() for k in r})}
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
    if menu:
        a = [0.0] * nv
        for i in (*i_f, il, ib, i_tb, i_xv):
            a[i] = -1.0
        for i in iz:
            a[i] = 1.0
        A_ub.append(a)                                    # no more words on the fold's routes than are folded on lanes
        b_ub.append(0.0)
    eq = [[0.0] * nv for _ in range(3 + nf)]
    for i in (*i_f, il, ib, iu, i_dw, i_tb):              # every word but the shortfall
        eq[0][i] = 1.0
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
        if not uni or not pi.uniform_form(f):
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
    return Mix(T=x[iT], words={"tensor": sum(x[i] for i in i_f), "lanes": x[il], "baseline": x[ib], "uniform": x[iu],
                               "table": x[i_tb]},
               filler_uniform=tuple(x[i] for i in fu), n=n, binding=binding(load, x[iT]), pi=lam,
               short={"words": x[i_dw], **{f"filler {j + 1}": x[df[j]] for j in range(nf)}},
               forms={f.kind: x[i] for f, i in zip(forms, i_f) if x[i] > 1e-12},
               folds={r: x[i] for r, i in zip(menu, iz) if x[i] > 1e-12})


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
    machine: IssueCosts
    operands: str
    credit: str                      # H (honest's kernel) or floor (the floor program ♭)
    fillers: tuple[tuple[Filler, float], ...]   # each filler with its lane instructions a word
    as_built: Mix                    # the credited kernel: the tensor route alone
    baseline: Mix                    # the credited program: the tensor route and the baseline program
    table: Table | None = None       # the cheater's table route, when open
    forms: tuple[Form, ...] = ()     # the MMA forms the cheater's tensor route mixes, `undominated`; () honest's alone

    def cheat(self, q: float, *, lanes: bool | None = None, uniform: bool | None = None,
              q_uniform: float | None = None, luck: bool = False) -> Mix:
        d = self.design
        lanes = d.lanes if lanes is None else lanes
        uniform = d.uniform if uniform is None else uniform
        qu = q_uniform if q_uniform is not None else (d.q_uniform if d.q_uniform is not None else q)
        return fastest(d, self.honest, self.machine, q=q if lanes else None, q_uniform=qu if uniform else None,
                       baseline=True, fillers=self.fillers, cheat=True,
                       tensor_mma=self.honest.tensor(self.operands, d.floor), budget=budget(d) if luck else 0.0,
                       table=self.table, forms=self.forms)

    def machine_eps(self, q: float, **routes) -> float:
        return max(0.0, 1 - self.cheat(q, **routes).T / self.baseline.T)

    def floor(self) -> float:
        """ε_machine with the lane and uniform routes closed: what operand delivery, the baseline and the table route
        (priced by its L2 bytes, not by q*) leave at every q*."""
        return self.machine_eps(0.0, lanes=False, uniform=False)


def undominated(fs: list[Form], pi: IssueCosts) -> tuple[Form, ...]:
    """The forms no other writes a word on as cheaply on every row of Π, the first of equals kept. The tensor routes
    differ only in their MMA, so dropping the rest leaves the LP's optimum where it is."""
    v = [[pi.c[f.kind].get(r, 0.0) / (f.depth * f.words) for r in pi.rows] for f in fs]
    return tuple(f for i, f in enumerate(fs)
                 if not any(j != i and all(a <= b for a, b in zip(v[j], v[i])) and (j < i or v[j] != v[i])
                            for j in range(len(fs))))


def setup(d: Design, pi: IssueCosts, data: Path = DATA, operands: str = "free", credit: str = "H",
          table: Table | None = None) -> Setup:
    """`credit` H and model compare the cheater with honest's kernel; floor with the floor program ♭ (SPEC §0.7a):
    the tensor route at the noise's φ_del, the fold at the cheapest mix of the design's fold and its built
    `fold_routes` on Π (the check reads the fold's result, not its program; a counted route is the cheater's alone),
    the fillers on their own routes and the baseline program,
    with no overhead. `table` opens the cheater's table route, priced for the design's `index_bits`. The cheater writes
    on any MMA form Π prices, honest on its own."""
    table = d.table(table)
    h = honest(d.honest, data)
    fms = undominated(sorted((f for f in forms(data) if f.kind in pi.c), key=lambda f: f.kind != h.form), pi)
    if d.L % h.depth:
        raise SystemExit(f"{d.name}: L = {d.L} is not a multiple of the MMA's depth {h.depth}")
    if credit not in CREDITS:
        raise SystemExit(f"credit: one of {', '.join(CREDITS)}")
    if credit == "floor" and operands == "honest":
        raise SystemExit("the floor program delivers operands at φ_del: --operands honest reads honest's credit only")
    tm = h.tensor("floor", d.floor) if credit == "floor" else None
    fs = tuple((f, float(f.per_word)) for f in d.filler)
    fm = credit == "floor"
    built = fastest(d, h, pi, q=None, q_uniform=None, baseline=False, fillers=fs, tensor_mma=tm, fold_menu=fm)
    base = fastest(d, h, pi, q=None, q_uniform=None, baseline=True, fillers=fs, tensor_mma=tm, fold_menu=fm)
    return Setup(design=d, honest=h, machine=pi, operands=operands, credit=credit, fillers=fs, as_built=built,
                 baseline=base, table=table, forms=tuple(fms))


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
    adapt: Adapt | None = None       # the proof block's ε_adapt; None charges none

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
        """ε with 100% checking, every word and operand verified, Corollary D3's: ε_skip, ε_mand, ε_machine, ε_luck
        (each check one draw, Theorem S at each block's own p) and the credit terms, with ε_adapt = 0."""
        s = (self.sched or 0.0) if sched else 0.0
        return 1 - (1 - self.skip - self.mand - self.machine_eps - self.luck) * (1 - s) * (1 - self.clock)

    def sample(self) -> float:
        """ε_sample = ε_adapt, the sampling compiler's: the proof block's and its selection rule's, 0 at 100% sampling.
        It is reported beside `core`, never inside it."""
        return self.adapt.eps if self.adapt else 0.0

    def total(self, sched: bool = True) -> float:
        """The compiled protocol's ε, core and sample in one product; for the record, never for comparing designs."""
        s = (self.sched or 0.0) if sched else 0.0
        a = self.adapt.eps if self.adapt else 0.0
        return 1 - (1 - self.skip - self.mand - self.machine_eps - self.luck - a) * (1 - s) * (1 - self.clock)

    def terms(self) -> dict[str, float | None]:
        return {"sched": self.sched, "skip": self.skip, "mand": self.mand, "machine": self.machine_eps,
                "luck": self.luck, "adapt": self.adapt.eps if self.adapt else None, "clock": self.clock}


def eps(d: Design, pi: IssueCosts, q: float, data: Path = DATA, *, operands: str = "free", credit: str = "H",
        lanes: bool | None = None, uniform: bool | None = None, q_uniform: float | None = None,
        table: Table | None = None, adapt: Adapt | None = None) -> Eps:
    """Under the model and floor credits ε_sched and ε_clock leave ε (honest's time above the credit is in γ,
    `credit_ratio`). `table` opens the cheater's table route; `adapt` charges a proof block's ε_adapt."""
    s = setup(d, pi, data, operands, credit, table)
    run = replace(d, filler=tuple(replace(f, p=0.0) for f in d.filler))   # honest runs every filler, credited or not
    hs = setup(run, pi, data, "free") if credit == "floor" or any(f.p >= 1 for f in d.filler) else s
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
               skip=sum(p.at.get(pi.vector, p.eps) for p in d.skip.values()),
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

