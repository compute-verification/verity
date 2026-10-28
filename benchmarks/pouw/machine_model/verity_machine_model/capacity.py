"""Certified capacity in the tensor–scalar model (FILLER, decision 77).

A window's certified work is a load on Π: n_k warp instructions of class k a SMSP-cycle, counted at the work's floor
(every MMA it ran and the instructions its mandatory fold costs under FOLD-OPT, never honest's overhead). What it
certifies is the GPU's capacity: the useful work that any program could still have done beside it is

    hidden(n) = max Σ_k μ_k h_k   subject to   Σ_k c_kr (n_k + h_k) ≤ 1 on every row r of Π,  h ≥ 0,

with μ_k the useful work one warp instruction of class k does, in tensor slots (an MMA's 4096 int8 MACs = 1; IDP4A's
128 = 1/32; HFMA2's 64 = 1/64; FFMA's 32 = 1/128), operands free. ε_cap = hidden(n) / hidden(0) is the share of the
empty GPU's useful capacity that the certificate leaves open, and σ = 1 − ε_cap is the certified share. A window is a
set of groups (a share of the cell's SMSP-time and that share's certified load). A receipt says neither where nor when
in its window and node its work ran, and hidden is concave in n, so the prover's best placement spreads the certified
work evenly: σ is read on the pooled load Σ w_g n_g, never as Σ w_g hidden(n_g), which is smaller and would credit a
placement nobody attests. Each group's own σ is reported beside it as that placement-attested reading.

Beside σ it reports the time-credit reading B(n) = max_r load_r (the share of the window the floor program needs, the
floor credit's reading of decision 74), which is never below σ: B credits a saturated facet whole, while σ asks what a
second program could still issue on the facets that aren't.

A tensor filler a MMA writes `words` committed words (128 for m16n8k32 at L = 32) and folds each `rounds` times with
FOLDS[fold] lane instructions a word; a lane filler is verified scalar work (FILL-OPT) at `rate` warp instructions a
SMSP-cycle in its class mix.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from scipy.optimize import linprog

from verity_machine_model.eps import DATA, LANES, MMA, IssueCosts, honest, profiles

TENSOR_CYCLES = 16.0  # every MMA kind holds the tensor pipe 16 SMSP-cycles (row M)

#: useful work a warp instruction, in tensor slots (4096 int8 MACs): the classes a hidden program could compute with,
#: sm_120's; `useful_of` adds a device's own tensor classes
USEFUL = {MMA: 1.0, "IMAD": 128 / 4096, "HFMA2": 64 / 4096, "FFMA": 32 / 4096}


def useful_of(data=DATA) -> dict[str, float]:
    """USEFUL with each honest profile's MMA class in `data` at its MACs a warp instruction over 4096. A device whose
    tensor class isn't sm_120's MMA (UTCIMMA_M128_N256 on sm_100, 256 slots; IGMMA_S8_N256 on sm_90, 32) would
    otherwise offer a hidden program no tensor pipe at all."""
    u = dict(USEFUL)
    for p in profiles(data):
        h = honest(p, data)
        u[h.cls] = max(u.get(h.cls, 0.0), h.words * h.depth / 4096)
    return u

#: a fold's lane instructions a committed word by class (pfill.cuh, receipt_folds.cuh)
FOLDS = {
    "none": {},
    "duo": {"ALU": 2.0, "IMAD": 1.0},     # u = y ^ p ^ k (LOP3); y = u a + z (IMAD); z = rotl(u, r) + y (LEA.HI)
    "xin13": {"ALU": 1.0, "IMAD": 0.5},   # a pair: SHF, LOP3, IMAD
    "li": {"ALU": 0.5, "IMAD": 0.5},      # a pair: LOP3, IMAD
    "arx": {"ALU": 1.0},                  # a pair: LOP3, LEA.HI (pfill's NC 9: a cost probe, not a receipt fold)
}
LANE_MIX = {"ALU": 0.5, "IMAD": 0.5}      # a lane filler's LOP3 and IMAD alternating (FILL-OPT, CALC's lane_fill)


@dataclass(frozen=True)
class Work:
    """Certified work: warp instructions a SMSP-cycle by class."""
    name: str
    n: dict[str, float]
    source: str = ""

    def __add__(self, o: Work) -> Work:
        n = dict(self.n)
        for k, v in o.n.items():
            n[k] = n.get(k, 0.0) + v
        return Work(f"{self.name} + {o.name}", n, "; ".join(s for s in (self.source, o.source) if s))

    def tensor_share(self) -> float:
        return self.n.get(MMA, 0.0) * TENSOR_CYCLES


def _mmas(share: float, fold: str, rounds: float, words: int, extra: dict | None) -> dict[str, float]:
    if fold not in FOLDS:
        raise SystemExit(f"fold: one of {', '.join(FOLDS)}")
    m = share / TENSOR_CYCLES
    n = {MMA: m}
    for k, v in FOLDS[fold].items():
        n[k] = n.get(k, 0.0) + m * v * rounds * words / LANES
    for k, v in (extra or {}).items():
        n[k] = n.get(k, 0.0) + m * v
    return n


def tensor_filler(share: float, fold: str = "duo", rounds: float = 1.0, words: int = 128, noise: dict | None = None,
                  source: str = "") -> Work:
    """A noise-GEMM filler at `share` of the tensor peak, each MMA's `words` committed words folded `rounds` times;
    `noise` adds certified operand work a MMA (warp instructions by class, e.g. SHA-512 under a named SHA-OPT)."""
    return Work(f"{fold}×{rounds:g} at {share:.3g}", _mmas(share, fold, rounds, words, noise), source)


def lane_filler(rate: float, mix: dict[str, float] | None = None, source: str = "") -> Work:
    mix = mix or LANE_MIX
    t = sum(mix.values())
    return Work(f"lane {rate:.3g}", {k: rate * v / t for k, v in mix.items()}, source)


def served(share: float, fold: str = "duo", words: int = 128, per_mma: dict[str, float] | None = None,
           source: str = "") -> Work:
    """Certified served products at `share` of the tensor peak (the core protocol's checked words, each folded once
    into its receipt), with `per_mma` any other certified work a MMA (warp instructions by class)."""
    return Work(f"served {fold} at {share:.3g}", _mmas(share, fold, 1.0, words, per_mma), source)


def parse(spec: str) -> Work:
    """A load from terms joined by '+': FOLD:SHARE[*ROUNDS] (a tensor filler), lane:RATE, served:SHARE[:FOLD],
    CLASS=RATE (raw warp instructions a SMSP-cycle)."""
    w = Work("", {})
    for t in spec.split("+"):
        t = t.strip()
        if "=" in t:
            k, v = t.split("=", 1)
            x = Work(t, {k: float(v)})
        elif t.startswith("lane:"):
            x = lane_filler(float(t[5:]))
        elif t.startswith("served:"):
            p = t.split(":")
            x = served(float(p[1]), p[2] if len(p) > 2 else "duo")
        elif ":" in t:
            fold, rest = t.split(":", 1)
            share, _, rounds = rest.partition("*")
            x = tensor_filler(float(share), fold, float(rounds or 1))
        else:
            raise SystemExit(f"capacity: can't read the term {t!r} (FOLD:SHARE[*R], lane:RATE, served:SHARE[:FOLD], "
                             "CLASS=RATE)")
        w = x if not w.name else w + x
    return Work(spec, w.n, w.source)


@dataclass(frozen=True)
class Hidden:
    work: Work
    load: dict[str, float]          # each row's load from the certified work, SMSP-cycles a SMSP-cycle
    hidden: float                   # tensor slots a SMSP-cycle a second program could add
    mix: dict[str, float]           # its warp instructions a SMSP-cycle by class
    binding: tuple[str, ...]        # rows the second program fills
    over: tuple[str, ...]           # rows the certified work alone exceeds (a model violation or a miscount)

    def credit(self) -> float:
        return max(self.load.values()) if self.load else 0.0


def hidden(pi: IssueCosts, w: Work, useful: dict[str, float] | None = None) -> Hidden:
    useful = {k: v for k, v in (useful or USEFUL).items() if k in pi.c}
    load = pi.load(w.n)
    over = tuple(r for r, x in load.items() if x > 1 + 1e-9)
    ks = list(useful)
    rows = [r for r in pi.rows if any(pi.c[k].get(r, 0.0) for k in ks)]
    if over:
        return Hidden(w, load, 0.0, {}, (), over)
    A = [[pi.c[k].get(r, 0.0) for k in ks] for r in rows]
    b = [max(0.0, 1.0 - load[r]) for r in rows]
    res = linprog([-useful[k] for k in ks], A_ub=A, b_ub=b, bounds=[(0, None)] * len(ks), method="highs")
    if res.status != 0:
        raise SystemExit(f"capacity LP: {res.message}")
    h = {k: float(x) for k, x in zip(ks, res.x) if x > 1e-12}
    slack = res.ineqlin.marginals
    binding = tuple(r for r, s in zip(rows, slack) if s < -1e-9)
    return Hidden(w, load, float(-res.fun), h, binding, ())


def pool(groups: list[tuple[float, Work]] | tuple[tuple[float, Work], ...]) -> Work:
    n: dict[str, float] = {}
    for w, g in groups:
        for k, v in g.n.items():
            n[k] = n.get(k, 0.0) + w * v
    return Work(" | ".join(f"{w:.3g}×({g.name})" for w, g in groups), n,
                "; ".join(g.source for _, g in groups if g.source))


@dataclass(frozen=True)
class WindowCell:
    """A window and node cell: groups of (share of its SMSP-time, certified work), read on their pooled load."""
    groups: tuple[tuple[float, Work], ...]
    vector: str
    empty: float                    # hidden(0): the empty GPU's useful tensor slots a SMSP-cycle
    pooled: Hidden | None = None
    parts: tuple[Hidden, ...] = field(default=())

    def eps_cap(self) -> float:
        return self.pooled.hidden / self.empty

    def sigma(self) -> float:
        return 1.0 - self.eps_cap()

    def sigma_placed(self) -> float:
        """σ if each group's placement were attested (Σ w_g hidden(n_g)): an upper reading, not a certificate."""
        return 1.0 - sum(w * h.hidden for (w, _), h in zip(self.groups, self.parts)) / self.empty

    def credit(self) -> float:
        return min(1.0, self.pooled.credit())

    def tensor(self) -> float:
        return min(1.0, self.pooled.work.tensor_share())

    def record(self) -> dict:
        return {"vector": self.vector, "sigma": self.sigma(), "eps_cap": self.eps_cap(), "credit": self.credit(),
                "tensor": self.tensor(), "sigma_placed": self.sigma_placed(),
                "empty_slots_per_smsp_cycle": self.empty, "n": self.pooled.work.n, "load": self.pooled.load,
                "mix": self.pooled.mix, "binding": list(self.pooled.binding), "over": list(self.pooled.over),
                "groups": [{"weight": w, "work": g.name, "n": g.n, "source": g.source, "hidden": h.hidden,
                            "sigma": 1 - h.hidden / self.empty, "credit": h.credit(), "tensor": g.tensor_share(),
                            "mix": h.mix, "binding": list(h.binding), "over": list(h.over)}
                           for (w, g), h in zip(self.groups, self.parts)]}


def window(pi: IssueCosts, groups: list[tuple[float, Work]] | tuple[tuple[float, Work], ...],
           useful: dict[str, float] | None = None) -> WindowCell:
    tot = sum(w for w, _ in groups)
    if abs(tot - 1.0) > 1e-6:
        raise SystemExit(f"capacity: the groups' shares of SMSP-time sum to {tot:g}, not 1")
    empty = hidden(pi, Work("empty", {}), useful).hidden
    return WindowCell(tuple(groups), pi.vector, empty, hidden(pi, pool(groups), useful),
                  tuple(hidden(pi, g, useful) for _, g in groups))
