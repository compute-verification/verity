"""The proof draw under the one-σ order: the cheater's best response and the minimax draw (SAMPLING).

A round's s proofs are whole dot products (DPs), drawn i.i.d. from a distribution ρ over the round's DPs. The cheater
knows σ before it evaluates (SPEC §2.6) and picks, in each round:

- deficient DPs, each of which passes a drawn proof with probability ≤ p over σ (Theorem S's luck);
- steering attacks: a producer row's noise-active coordinates (R_i ≠ 0, plus one outlier) written wrong so that the
  consumer's noised operand is 0, which makes every DP of the consumer group pass for free (SPEC's outlier trick).

Proposition E2 with s draws bounds Σ_rounds [c_luck(s)·ρ(deficient) + s·ρ(wrong)] by L = ln(1/δ) off a δ event, so the
cheater's best response to a sample is a fractional knapsack over these attacks with per-round supplies
(`best_response`). The verifier's minimax draw is a covering LP whose optimum is closed-form: a DP's weight is its MACs
times max(1/c_luck(s), (1 + F)/s), F the largest fan-out of the levers its row carries (`optimal`); knowing σ (public
before the sample), the verifier puts each row's lever excess only on the coordinates every lever needs (`aware`), which
attains `capped_optimum`, the exact minimax against the cheater with per-round supplies, solved as one LP. A
closure, r producer DPs proved at random noise-active coordinates of each drawn consumer's operand row, detects a
steered consumer as if it had failed (`closure`); at any coordinates of the row, it detects with the row's wrong
fraction (`closure_any`). ε_luck + ε_adapt = luck factor × (saving / S).

Inputs are SPEC's and SKIP's (note:20261017T0340Z-draft-skip) shapes and Theorem S's constants. The proof cost is the
proofs team's (PRF-57): C-Flock's `--zk` CPU prover by session size, one session at a time per site, one per table
until PRF-48. A round is 12.6 s, so proving keeps up only if a session batches enough rounds (`Cost.batch`); a round
that waits for its own proofs is longer, and ε grows with it (`unbatched`). Run `uv run
benchmarks/pouw/proof_sampling.py` for the table.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import numpy as np

LN_INV_DELTA = 40 * math.log(2)
ROUNDS = 48_000
ROUND_S = 12.6
P_PASS = 0.75
LUCK_FACTOR = 0.403

COMPONENTS = ("qkv", "qk", "pv", "o", "gu", "down", "lm")

# One `flock-circuit --zk` session on a 32-thread CPU prover (PRF-57, note:proofs/20261017T0552Z-report-proof-service-
# cost-model; Pc4Tile_v1 at K = 4096): (whole DPs of length 4,096, prove s, process wall s). A DP's marginal cost is
# linear in its length; below 16 DPs the 16–256 line runs down to the session's fixed cost (1.8 s prove, 12.1 s wall).
# Sessions run one at a time per site, and until PRF-48 each DP shape (a padded length class) is its own session.
ZK_SESSION = ((16, 2.4, 14.0), (256, 11.3, 42.0), (4096, 283.0, 950.0))
DP_K = 4096


@dataclass(frozen=True)
class Model:
    name: str
    d: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    vocab: int


LLAMA_8B = Model("8B", 4096, 32, 32, 8, 128, 14336, 128256)
LLAMA_70B = Model("70B", 8192, 80, 64, 8, 128, 28672, 128256)


@dataclass(frozen=True)
class Design:
    """What the sample sees of a design: Pr[R_i ≠ 0] of its activation-side noise and its QKᵀ MACs per model MAC."""

    name: str
    pi: float
    qk_mult: float


DESIGN_A = Design("A", 0.5, 1.0)
DESIGN_BC = Design("Bc", 0.5, 1.5)
DECODE_STEPS = {"8B": 1575, "70B": 315}


@dataclass
class Cell:
    """A round's DPs by subclass (a class split by which levers' consumers see R_i ≠ 0 there), and the levers.

    A lever is one way to steer a consumer group from one producer row: `lev_forced` noise-active producer DPs plus one
    outlier are written wrong, and the consumer's `lev_consumer` MACs pass for free. `lev_rows` rows a round carry it.
    """

    label: str
    comp: np.ndarray
    count: np.ndarray
    k: np.ndarray
    parent: np.ndarray
    bits: np.ndarray
    lev_class: np.ndarray
    lev_bit: np.ndarray
    lev_consumer: np.ndarray
    lev_forced: np.ndarray
    lev_rows: np.ndarray
    lev_k: np.ndarray
    lev_pos: np.ndarray
    lev_label: list[str]
    class_row: np.ndarray
    class_dpc: np.ndarray
    class_nlev: np.ndarray
    pi: float

    @property
    def macs(self) -> float:
        return float((self.count * self.k).sum())

    def fan(self) -> np.ndarray:
        """SPEC's Fan per lever: consumer MACs over the wrong producer DPs' MACs."""
        return self.lev_consumer / ((self.lev_forced + 1) * self.lev_k)

    def share(self, comp: str) -> float:
        sel = self.comp == COMPONENTS.index(comp)
        return float((self.count[sel] * self.k[sel]).sum()) / self.macs


@dataclass
class _Builder:
    pi: float
    comp: list = field(default_factory=list)
    count: list = field(default_factory=list)
    k: list = field(default_factory=list)
    row: list = field(default_factory=list)
    dpc: list = field(default_factory=list)
    pos: list = field(default_factory=list)
    levers: list = field(default_factory=list)

    def add(self, comp, count, k, row=1, dpc=1, levers=(), pos=0):
        """Add a class (arrays over positions broadcast); each lever is (label, consumer MACs, forced wrong DPs)."""
        shapes = [np.shape(x) for x in (count, k, row, pos)] + [np.shape(x) for _, c, f in levers for x in (c, f)]
        n = int(np.prod(np.broadcast_shapes(*shapes)))
        count, k, row, pos = (np.broadcast_to(np.asarray(x, float), (n,)).copy() for x in (count, k, row, pos))
        if not count.any():
            return
        self.comp.append(np.full(n, COMPONENTS.index(comp)))
        self.count.append(count)
        self.k.append(k)
        self.row.append(row)
        self.dpc.append(np.full(n, float(dpc)))
        self.pos.append(pos)
        self.levers.append([(lab, *(np.broadcast_to(np.asarray(x, float), (n,)) for x in (c, f)))
                            for lab, c, f in levers])

    def build(self, label: str) -> Cell:
        sub = {key: [] for key in ("comp", "count", "k", "parent", "bits")}
        lev = {key: [] for key in ("cls", "bit", "C", "forced", "rows", "k", "pos")}
        labels: list[str] = []
        base = 0
        nlev_all = []
        for comp, count, k, row, dpc, pos, levers in zip(self.comp, self.count, self.k, self.row, self.dpc, self.pos,
                                                         self.levers):
            n, nlev = count.size, len(levers)
            nlev_all.append(np.full(n, nlev))
            idx = base + np.arange(n)
            for b in range(2**nlev):
                frac = 1.0
                for i in range(nlev):
                    frac *= self.pi if b >> i & 1 else 1 - self.pi
                sub["comp"].append(comp)
                sub["count"].append(count * frac)
                sub["k"].append(k)
                sub["parent"].append(idx)
                sub["bits"].append(np.full(n, b))
            for i, (lab, consumer, forced) in enumerate(levers):
                keep = consumer > 0
                lev["cls"].append(idx[keep])
                lev["bit"].append(np.full(keep.sum(), i))
                lev["C"].append(consumer[keep])
                lev["forced"].append(forced[keep])
                lev["rows"].append(count[keep] / row[keep])
                lev["k"].append(k[keep])
                lev["pos"].append(pos[keep])
                labels += [lab] * int(keep.sum())
            base += n
        cat = np.concatenate
        return Cell(
            label=label,
            comp=cat(sub["comp"]),
            count=cat(sub["count"]),
            k=cat(sub["k"]),
            parent=cat(sub["parent"]),
            bits=cat(sub["bits"]),
            lev_class=cat(lev["cls"]),
            lev_bit=cat(lev["bit"]),
            lev_consumer=cat(lev["C"]),
            lev_forced=cat(lev["forced"]),
            lev_rows=cat(lev["rows"]),
            lev_k=cat(lev["k"]),
            lev_pos=cat(lev["pos"]),
            lev_label=labels,
            class_row=cat(self.row),
            class_dpc=cat(self.dpc),
            class_nlev=cat(nlev_all),
            pi=self.pi,
        )


def cell(model: Model, phase: str, n: int, design: Design = DESIGN_A, steps: int | None = None) -> Cell:
    """One round's DPs: a prefill of n tokens, or `steps` decode steps of one sequence after an n-token prompt.

    Per token and layer: q, k, v (qkv), the scores (QKᵀ), P·V, o_proj, gate/up, down, and lm_head where logits are
    computed (the prefill's last token, every decode step). Prefill keys are causal; a decode step reads the prompt's
    committed (pre-σ, so unsteerable) K/V and the round's own.
    """
    m, pi = model, design.pi
    hd, d, kv = m.head_dim, m.d, m.kv_heads * m.head_dim
    group = m.heads // m.kv_heads
    if phase == "prefill":
        keys = np.arange(1, n + 1, dtype=float)
        future = n - keys + 1
        lm_tokens = 1
        label = f"{m.name} prefill {n // 1024}K"
    else:
        steps = steps or DECODE_STEPS[m.name]
        keys = n + np.arange(1, steps + 1, dtype=float)
        future = steps - np.arange(1, steps + 1, dtype=float) + 1
        lm_tokens = steps
        label = f"{m.name} decode {n // 1024}K"
    tokens = keys.size
    t = np.arange(1, tokens + 1, dtype=float)
    L = m.layers
    qk = design.qk_mult
    f_head = pi * hd
    b = _Builder(pi)
    # q of token t steers its scores, a head row of 128 coordinates: keys·128 consumer MACs each.
    b.add("qkv", L * d, d, row=hd, levers=[("q→QKᵀ", keys * hd * qk, f_head)], pos=t)
    b.add("qkv", L * kv, d, row=hd, levers=[("k→QKᵀ", group * hd * future * qk, f_head)], pos=t)
    b.add("qkv", L * kv, d, row=hd, levers=[("v→P·V", group * hd * future, f_head)], pos=t)
    # The scores of (t, head) steer P̂·V there; P·V of token t steers o_proj's whole row.
    b.add("qk", L * m.heads * keys, hd * qk, row=keys, levers=[("QKᵀ→P·V", hd * keys, pi * keys)],
          pos=t)
    b.add("pv", L * d, keys, row=d, levers=[("P·V→o", d * d, pi * d)], pos=t)
    qkv_next = (d + 2 * kv) * d
    gu, down_c, lm_c = 2 * m.ffn * d, d * m.ffn, m.vocab * d
    mid = [("o→gate/up", gu, pi * d), ("o→qkv′", qkv_next, pi * d)]
    b.add("o", (L - 1) * tokens * d, d, row=d, levers=mid)
    b.add("o", (tokens - lm_tokens) * d, d, row=d, levers=[("o→gate/up", gu, pi * d)])
    b.add("o", lm_tokens * d, d, row=d, levers=[("o→gate/up", gu, pi * d), ("o→lm_head", lm_c, pi * d)])
    b.add("gu", L * tokens * 2 * m.ffn, d, row=2 * m.ffn, dpc=2, levers=[("gate/up→down", down_c, pi * m.ffn)])
    b.add("down", (L - 1) * tokens * d, m.ffn, row=d,
          levers=[("down→qkv′", qkv_next, pi * d), ("down→gate/up′", gu, pi * d)])
    b.add("down", (tokens - lm_tokens) * d, m.ffn, row=d)
    b.add("down", lm_tokens * d, m.ffn, row=d, levers=[("down→lm_head", lm_c, pi * d)])
    b.add("lm", lm_tokens * m.vocab, d)
    return b.build(f"{label} ({design.name})")


def c_luck(s: int, p: float = P_PASS) -> float:
    """Luck's detection rate per unit of deficient ρ-mass a round with s draws sharing one σ.

    A round with deficient mass d passes with probability ≤ p + (1 − p)(1 − d)^s, so −ln of it is at least c·d with
    c = inf over d of its ratio: 1 − p at one sample (the tangent at 0) and ln(1/p) once (s − 1)p ≥ 1 − p (the chord).
    """
    if s == 1:
        return 1 - p
    d = np.geomspace(1e-9, 1.0, 4001)
    return float((-np.log(p + (1 - p) * (1 - d) ** s) / d).min())


# ---- strategies: a weight per DP of each subclass (ρ is the weight normalized over the round) ----


def _lever_class_max(c: Cell, values: np.ndarray, fill: float = 0.0) -> np.ndarray:
    out = np.full(c.class_row.size, fill)
    np.maximum.at(out, c.lev_class, values)
    return out


def work(c: Cell, s: int) -> np.ndarray:
    return c.k.copy()


def uniform(c: Cell, s: int) -> np.ndarray:
    return np.ones_like(c.k)


def stratified(c: Cell, s: int) -> np.ndarray:
    """Equal draw mass per component, work-weighted within it."""
    mass = np.zeros(len(COMPONENTS))
    np.add.at(mass, c.comp, c.count * c.k)
    return c.k / mass[c.comp]


def fanout(c: Cell, s: int) -> np.ndarray:
    """Work times 1 + the row's largest Fan."""
    return c.k * (1 + _lever_class_max(c, c.fan())[c.parent])


def optimal(c: Cell, s: int) -> np.ndarray:
    """The minimax draw that ignores σ: work times max(1/c_luck(s), (1 + F)/s)."""
    return c.k * np.maximum(1 / c_luck(s), (1 + _lever_class_max(c, c.fan())[c.parent]) / s)


def aware(c: Cell, s: int) -> np.ndarray:
    """The minimax draw given σ: luck's work/c_luck everywhere, and each row's lever excess on the coordinates that
    every lever needing one forces wrong.

    Lever i on a row needs e_i = (C_i + m_i·k)/s − m_i·k/c_luck more weight on its forced DPs; putting max_i e_i on the
    DPs active for every such lever meets them all, and no cover of the row does with less.
    """
    cl = c_luck(s)
    m = c.lev_forced + 1
    excess = (c.lev_consumer + m * c.lev_k) / s - m * c.lev_k / cl
    need = excess > 0
    nclass = c.class_row.size
    e_max = np.zeros(nclass)
    np.maximum.at(e_max, c.lev_class[need], excess[need])
    mask = np.zeros(nclass, dtype=np.int64)
    np.bitwise_or.at(mask, c.lev_class[need], 1 << c.lev_bit[need])
    nbits = np.zeros(nclass)
    np.add.at(nbits, c.lev_class[need], 1.0)
    w = c.k / cl
    on = (e_max[c.parent] > 0) & ((c.bits & mask[c.parent]) == mask[c.parent])
    dpc, row = c.class_dpc[c.parent], c.class_row[c.parent]
    covered = row * c.pi ** nbits[c.parent]
    w = w + np.where(on, e_max[c.parent] * dpc / np.where(on, covered, 1.0), 0.0)
    return w


STRATEGIES = {"work": work, "uniform": uniform, "stratified": stratified, "fanout": fanout, "optimal": optimal,
              "aware": aware}


def capped_optimum(c: Cell, s: int) -> tuple[float, np.ndarray]:
    """The exact minimax draw against the capped cheater, as one LP, and its ε.

    The cheater's knapsack (max Σ y·save, Σ y·det(ρ) ≤ L, 0 ≤ y ≤ S·supply) has the dual
    min L·μ + S·Σ supply·ν subject to μ·det_a(ρ) + ν_a ≥ save_a. With x = μρ the verifier's choice of ρ joins it as one
    LP over x ≥ 0, ν ≥ 0, and z_class ≤ x (the outlier, the row's cheapest DP). It is solved in MAC units, since
    shares of a round's MACs fall below the solver's tolerances.
    """
    from scipy import sparse
    from scipy.optimize import linprog

    cl, nsub, nlev, ncls = c_luck(s), c.k.size, c.lev_k.size, c.class_row.size
    ix, iz, ind, inl = 0, nsub, nsub + ncls, 2 * nsub + ncls
    cost = np.concatenate([LN_INV_DELTA * c.count, np.zeros(ncls), ROUNDS * c.count, ROUNDS * c.lev_rows])
    sub = np.arange(nsub)
    rows = [sub, sub]
    cols = [ix + sub, ind + sub]
    vals = [np.full(nsub, -cl), np.full(nsub, -1.0)]
    act_n = np.zeros((ncls, 3))
    for i in range(3):
        on = (c.bits >> i & 1).astype(bool)
        np.add.at(act_n[:, i], c.parent[on], c.count[on])
    order = np.argsort(c.parent, kind="stable")
    start = np.searchsorted(c.parent[order], np.arange(ncls))
    width = np.searchsorted(c.parent[order], np.arange(ncls), side="right") - start
    lev = np.arange(nlev)
    for o in range(int(width.max())):
        ok = o < width[c.lev_class]
        j = lev[ok]
        u = order[start[c.lev_class[j]] + o]
        on = (c.bits[u] >> c.lev_bit[j] & 1).astype(bool)
        j, u = j[on], u[on]
        rows.append(nsub + j)
        cols.append(ix + u)
        vals.append(-s * c.lev_forced[j] * c.count[u] / act_n[c.lev_class[j], c.lev_bit[j]])
    rows += [nsub + lev, nsub + lev, nsub + nlev + sub, nsub + nlev + sub]
    cols += [iz + c.lev_class, inl + lev, iz + c.parent, ix + sub]
    vals += [np.full(nlev, -float(s)), np.full(nlev, -1.0), np.ones(nsub), -np.ones(nsub)]
    rhs = np.concatenate([-c.k, -(c.lev_consumer + (c.lev_forced + 1) * c.lev_k), np.zeros(nsub)])
    a = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(rhs.size, cost.size))
    res = linprog(cost, A_ub=a, b_ub=rhs, bounds=(0, None), method="highs")
    if res.status != 0:
        raise RuntimeError(f"capped minimax LP: {res.message}")
    return LUCK_FACTOR * res.fun / c.macs / ROUNDS, res.x[ix:iz]


# ---- the cheater's best response ----


@dataclass
class Response:
    """The best response's ε and Λ (its saving in units of ln(1/δ)/S), its saving by attack, the sample, the closed
    levers, and the rows of each lever it steers over the window."""

    eps: float
    lam: float
    by_attack: dict[str, float]
    rho: np.ndarray
    closure: np.ndarray
    lev_used: np.ndarray


def best_response(c: Cell, w: np.ndarray, s: int, closure_r: int = 0, caps: bool = True,
                  levers: np.ndarray | None = None, closure_any: bool = False) -> Response:
    """The cheater's saving-maximizing attacks against draw weights w at s draws a round (fractional knapsack).

    Each attack saves MACs and spends E2's detection budget L = ln(1/δ) over the window: c_luck(s)·ρ per deficient DP,
    s·ρ per wrong DP, and with a closure s·q·ρ(consumer group) for a steered consumer whose lever the closure covers. q is
    1 for producers sampled at noise-active coordinates, and 1 − (1 − m/row)^r for `closure_any`'s producers sampled at
    any coordinate of the row (a closure map that reads only the sampling coin and the Program, not σ).
    With `caps` an attack is available once a round per row (or DP); without, its best ratio takes the whole budget.
    """
    rho = w / (c.count * w).sum()
    M = c.macs
    cl = c_luck(s)
    nclass = c.class_row.size
    # Per class: the mean ρ of its DPs active for each lever bit, and its cheapest DP (the outlier).
    act_mass = np.zeros((nclass, 3))
    act_n = np.zeros((nclass, 3))
    for i in range(3):
        on = (c.bits >> i & 1).astype(bool)
        np.add.at(act_mass[:, i], c.parent[on], (c.count * rho)[on])
        np.add.at(act_n[:, i], c.parent[on], c.count[on])
    rho_min = np.full(nclass, np.inf)
    np.minimum.at(rho_min, c.parent, rho)
    mean_act = act_mass[c.lev_class, c.lev_bit] / act_n[c.lev_class, c.lev_bit]
    m = c.lev_forced + 1
    fan = c.fan()
    sel = np.ones(fan.size, bool) if levers is None else levers
    closed = np.zeros(fan.size, bool)
    q = 1.0
    if closure_r:
        closed = (1 + fan) / s > 1 / cl
        if closure_any:
            q = 1 - (1 - np.minimum(1.0, m / c.class_row[c.lev_class])) ** closure_r
    det_lev = s * (c.lev_forced * mean_act + rho_min[c.lev_class] + np.where(closed, q * c.lev_consumer / M, 0.0))
    save_lev = (c.lev_consumer + m * c.lev_k) / M
    save = np.concatenate([c.k / M, save_lev[sel]])
    det = np.concatenate([cl * rho, det_lev[sel]])
    supply = np.concatenate([c.count, c.lev_rows[sel]])
    labels = np.concatenate([np.full(c.k.size, -1), np.flatnonzero(sel)])
    ratio = save / det
    order = np.argsort(-ratio, kind="stable")
    cap = ROUNDS * supply[order] * det[order] if caps else np.full(order.size, np.inf)
    spent = np.minimum(np.cumsum(cap), LN_INV_DELTA)
    taken = np.diff(np.concatenate([[0.0], spent]))
    gain = taken * ratio[order]
    by: dict[str, float] = {}
    for j, g in zip(labels[order][taken > 0], gain[taken > 0]):
        key = "luck" if j < 0 else c.lev_label[j]
        by[key] = by.get(key, 0.0) + float(g)
    used = np.zeros(fan.size)
    lev = labels[order] >= 0
    np.add.at(used, labels[order][lev], (taken / det[order])[lev])
    saving = float(gain.sum())
    return Response(LUCK_FACTOR * saving / ROUNDS, saving / LN_INV_DELTA, by, rho, closed, used)


# ---- proof cost ----


def session_s(dps: float) -> tuple[float, float]:
    """A session's prove and process wall seconds at `dps` whole DPs of length 4,096 (or their MACs in longer ones)."""
    (n0, p0, w0), (n1, p1, w1), (n2, p2, w2) = ZK_SESSION
    if dps <= n1:
        t = (dps - n0) / (n1 - n0)
        return p0 + (p1 - p0) * t, w0 + (w1 - w0) * t
    t = (dps - n1) / (n2 - n1)
    return p1 + (p2 - p1) * t, w1 + (w2 - w1) * t


def sessions_s(dps: float) -> float:
    """Wall seconds to prove `dps` DPs in back-to-back sessions of at most 256 (PRF-57's sweet spot)."""
    full, rest = divmod(dps, ZK_SESSION[1][0])
    return full * ZK_SESSION[1][2] + (session_s(rest)[1] if rest > 0 else 0.0)


def shape(k: np.ndarray) -> np.ndarray:
    """A DP's table: its length padded to a power of two (P·V's lengths, one per position, in ≈ 9 classes)."""
    return 2 ** np.ceil(np.log2(k))


@dataclass
class Cost:
    """A round's expected proofs: MACs and proofs per table (padded length class)."""

    macs: dict[float, float]
    draws: dict[float, float]

    @property
    def dps(self) -> float:
        """Proved MACs in whole DPs of length 4,096."""
        return sum(self.macs.values()) / DP_K

    def n_tables(self, rounds: int = 1) -> float:
        """Expected tables drawn over `rounds` rounds: each table's chance of at least one proof."""
        return float(sum(1 - math.exp(-rounds * n) for n in self.draws.values()))

    @property
    def prove_s(self) -> float:
        return session_s(self.dps)[0]

    @property
    def wall_s(self) -> float:
        """One multi-table session a round (PRF-48)."""
        return session_s(self.dps)[1]

    def keep_up_s(self, rounds: int, today: bool = False) -> float:
        """Wall seconds to prove `rounds` rounds' draws: as one multi-table batch (PRF-48), or `today` one per table."""
        if not today:
            return sessions_s(rounds * self.dps)
        out = 0.0
        for t, m in self.macs.items():
            hit = 1 - math.exp(-rounds * self.draws[t])
            if hit > 0:
                out += hit * sessions_s(rounds * m / DP_K / hit)
        return out

    def batch(self, today: bool = False, most: int = 256) -> int | None:
        """The fewest rounds a batch must cover so that proving keeps up with 12.6 s rounds (None: it never does)."""
        for b in range(1, most + 1):
            if self.keep_up_s(b, today) <= b * ROUND_S:
                return b
        return None


def proof_cost(c: Cell, r: Response, s: int, closure_r: int = 0) -> Cost:
    """A round's expected proved MACs, by table, with the closure's producers."""
    macs: dict[float, float] = {}
    draws: dict[float, float] = {}

    def put(k: np.ndarray, n: np.ndarray) -> None:
        for t, kk, nn in zip(shape(k), k, n):
            macs[t] = macs.get(t, 0.0) + float(nn * kk)
            draws[t] = draws.get(t, 0.0) + float(nn)

    put(c.k, s * c.count * r.rho)
    if closure_r:
        put(c.lev_k, s * closure_r * c.lev_rows * c.lev_consumer / c.macs * r.closure)
    return Cost(macs, draws)


# ---- the table ----


def evaluate(c: Cell, strategy: str, s: int, closure_r: int = 0) -> tuple[Response, Cost]:
    """`closure` and `closure_any` are the work sample with producers at noise-active or at any coordinates."""
    closed = strategy.startswith("closure")
    w = work(c, s) if closed else STRATEGIES[strategy](c, s)
    r = best_response(c, w, s, closure_r=closure_r if closed else 0, closure_any=strategy == "closure_any")
    return r, proof_cost(c, r, s, closure_r if closed else 0)


def unbatched(eps: float, wall_s: float) -> float:
    """ε when each round lasts at least its own proofs' wall time: S shrinks, and ε ∝ 1/S."""
    return eps * max(1.0, wall_s / ROUND_S)


def cells(designs=(DESIGN_A, DESIGN_BC)) -> list[Cell]:
    out = []
    for design in designs:
        for model in (LLAMA_8B, LLAMA_70B):
            for n in (2048, 8192, 32768):
                out.append(cell(model, "prefill", n, design))
            for n in (2048, 32768):
                out.append(cell(model, "decode", n, design))
    return out


def eps_luck(s: int) -> float:
    return LUCK_FACTOR * LN_INV_DELTA / c_luck(s) / ROUNDS


def table(cs: list[Cell], strategies: list[str], draws: list[int], closure_r: int) -> str:
    """ε and Λ, the worst cell per model and design, against the round's proof cost (each column's worst cell).

    ε is at S = 48,000 rounds of 12.6 s, which holds when sessions batch enough rounds to keep up (the batch columns);
    "unbatched" is ε when every round waits for its own session (PRF-48's one multi-table session a round).
    """
    groups: dict[str, list[Cell]] = {}
    for c in cs:
        groups.setdefault(f"{c.label.split()[0]} {c.label.split('(')[1].rstrip(')')}", []).append(c)
    lines = ["| strategy | s | " + " | ".join(f"ε {g}" for g in groups) +
             " | Λ (max) | worst cell: its top attack | 4K DPs a round | tables a round | prove s | wall s"
             " | ε unbatched | rounds a batch: PRF-48, today |",
             "|---|---:|" + "---:|" * len(groups) + "---:|---|---:|---:|---:|---:|---:|---:|"]
    for st in strategies:
        for s in draws:
            got = {c.label: evaluate(c, st, s, closure_r) for c in cs}
            eps = [max(got[c.label][0].eps for c in cl) for cl in groups.values()]
            worst = max(got, key=lambda lab: got[lab][0].eps)
            r = got[worst][0]
            costs = [g[1] for g in got.values()]
            name = f"{st} r={closure_r}" if st.startswith("closure") else st
            batches = [[x.batch(today) for x in costs] for today in (False, True)]
            batch = ", ".join("—" if None in b else str(max(b)) for b in batches)
            lines.append(f"| {name} | {s} | " + " | ".join(f"{100 * e:.3f}%" for e in eps) +
                         f" | {max(g[0].lam for g in got.values()):.3f} | {worst}: {max(r.by_attack, key=r.by_attack.get)}"
                         f" | {max(x.dps for x in costs):.1f} | {max(x.n_tables() for x in costs):.1f}"
                         f" | {max(x.prove_s for x in costs):.1f} | {max(x.wall_s for x in costs):.1f}"
                         f" | {100 * max(unbatched(g[0].eps, g[1].wall_s) for g in got.values()):.3f}% | {batch} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--strategies", default="work,uniform,stratified,fanout,optimal,aware,closure,closure_any")
    ap.add_argument("--draws", default="1,2,4,16,64,256")
    ap.add_argument("--closure-r", type=int, default=1)
    ap.add_argument("--designs", default="A,Bc")
    a = ap.parse_args()
    designs = [{"A": DESIGN_A, "Bc": DESIGN_BC}[x] for x in a.designs.split(",")]
    draws = [int(x) for x in a.draws.split(",")]
    print(f"eps_luck: s=1 {100 * eps_luck(1):.4f}%, s>=2 {100 * eps_luck(2):.4f}%")
    print(table(cells(designs), a.strategies.split(","), draws, a.closure_r))


if __name__ == "__main__":
    main()
