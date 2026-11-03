"""The I/O bound on a training step's slowdown ρ: how many of training formalism v2's blocks (at most c gates, at most
k bits of input outside θ) a step's products need, from the Loomis–Whitney inequality, and the step's products as the
bound reads them off a captured step (te_core.capture).

    python -m te_core.io_bound vectors [--out FILE]               # the test vectors (tests/io_bound_vectors.json)
    python -m te_core.io_bound steps [--only NAME,...] [--out FILE]   # capture the s-shape steps (`STEPS`), bound each

Run from `benchmarks/covert_witness/`. |C|, c and g are Boolean gates (te_core.gates); every other quantity is bits.

**One product.** C = A·B, A ∈ I×L entries of b_A bits, B ∈ L×J of b_B bits, each output entry accumulated along its
chain over L in a b_C-bit partial sum (`ACC_BITS`: the GEMM gadget accumulates from a zero 32-bit word). A block's MACs
S ⊆ I×L×J satisfy |S|² ≤ |π_IL S|·|π_LJ S|·|π_IJ S| (Loomis–Whitney: slice S by l; a slice has at most
min(|A_l|·|B_l|, |π_IJ S|) ≤ √(|A_l|·|B_l|·|π_IJ S|) MACs; Cauchy–Schwarz over l). Under P-width (below) each A or B
entry the block uses costs it b_A or b_B input bits of its own, and each output entry it touches either starts its
chain in the block (s of them) or has its partial sum read (b_C bits). With x·b_A + y·b_B + t ≤ k and E = s·b_C,
|S| ≤ √(x·y·(t/b_C + s)), whose most over the split is h(E) = ((k + E)/3)^{3/2}/√(b_A·b_B·b_C) for E ≤ k/2 and
(k/2)·√(E/(b_A·b_B·b_C)) beyond. Two bounds on h sum over blocks:

    h ≤ α + β·√s        h ≤ α + μ·E
    α = (k/3)^{3/2} / √(b_A·b_B·b_C)        β = (k/2) / √(b_A·b_B)        μ = (3√3/16)·√k / √(b_A·b_B·b_C)

The first splits the root; the second is h's concave envelope, the line from h(0) = α tangent to h at E = 16k/27. Each
output's chain starts in one block, so over N blocks Σ s ≤ |O|: M ≤ N·α + β·√(N·|O|) and M ≤ N·α + μ·b_C·|O|.

**A step.** Over the counted products P (`Row.kind` "fresh") at g_P gates a MAC, a block splitting its budget as
k = Σ k_P has Σ g_P·α_P(k_P) ≤ max_P g_P·α_P(k) (α grows as k^{3/2}), Σ g_P·β_P(k_P)·√s_P ≤
(k/2)·√(Σ g_P²·s_P/(b_A·b_B)) and μ_P(k_P) ≤ μ_P(k), so with Cauchy–Schwarz over the blocks for the first form

    G = Σ g_P·M_P ≤ Λ·(N·Γ₀ + (k/2)·√(N·Q)),    G ≤ Λ·(N·Γ₀ + L),
    Γ₀ = max_P g_P·α_P,    Q = Σ_P g_P²·|O_P|/(b_A·b_B),    L = Σ_P g_P·μ_P·b_C·|O_P|.

Both hold on any subset of the products, with Γ₀ and Λ at most the whole set's. The linear form's best subset keeps
the products whose work exceeds their chain starts' allowance, G_P > Λ·L_P (chains longer than Λ·μ·b_C), so N is at
least the larger of ((√(B² + 4ΓG) − B)/(2Γ))² and Σ_P (G_P − Λ·L_P)⁺/Γ, with Γ = Λ·Γ₀ and B = Λ·(k/2)·√Q (`blocks`),
and ρ = N·c/|C| at the best partition is at least that times c/|C| (`bound`). The linear form binds where blocks start
few chains each (long chains: a dense step's), the root form where they start many (a rank-16 adapter's). Dropping
both gives ρ ≥ σ·c/Γ, σ = G/|C| the counted share (`rho_main`).

Λ covers a block that does several products at once. Products sharing their arrays (a linear layer's forward, dX and
dW read X, W and dY two times each) do at most 2/√3 of the single-product cap (`SHARE`: Cauchy–Schwarz, then
xy + yz + zx ≤ (x + y + z)²/3). A partial sum read in and finished into the next product's operand pays both faces by
splitting its b_C bits: `feed` = √(b_C/(b_C − b)) for b ≤ b_C/2, else 2√(b/b_C). Operand entries a block makes whole
from chains it runs itself, their own operands read or made whole in turn, number at most e·k/b, e the least root of
e = r·(1 + e)² with r = k/(3·K²·b) (LW on each source product, 4/3 for sources sharing arrays), worth `whole_chain` =
(1 + e)^{3/2}. That one-class rule holds only where no shorter chain is nested below a counted operand, and on a
transformer's step attention's d_h-long chains are, so the proved factor is the step's whole-making system's
(`chain_classes`, `whole_chain_of`): every product with both operands affected in a class by its chain K, with
r_j = c_j·k/(K_j²·b), μ_j ≤ r_j·(P_j + Σ_i A_ij·μ_i)² at its least fixed point (exact class by class in feeding
order, a cycle iterated) and e = Σ_i A_i,counted·μ_i, void where a cycle with no fixed point feeds a counted product.
Proved: Λ = SHARE·feed·whole_chain, at the kept products' widest operand and the whole-making factor of the classes
that feed them, so a subset that drops the products a high-μ class feeds has its own smaller factor (`bound`). Each
factor bounds both forms: shared arrays scale √(x·y·(t/b_C + s)) itself, a fed partial sum lowers only the read face's
width (raising α by `feed` and μ·b_C by less), and whole-making raises k (α by (1 + e)^{3/2}, μ by less). Found by
search over small chained triples: Λ = 2/√3 (the "numerical" figures).

**Premises.**
- P-chain: the circuit accumulates each output entry along one chain of gadgets from a zero word, so Σ s ≤ |O|. A
  split-K reduction starts more chains: `outputs` is then the chains' count.
- P-cell: a block holds whole GEMM gadgets (`Row.cell_gates`: the dot product of `te_core.gates.PRICES`). Without it
  each split gadget costs some block one input bit of its own, so at most N·k are split, and Γ grows by k·cell_gates
  (the "no_cell" figures, which need no premise on the partition).
- P-width: every operand entry of a counted product that a block uses is paid by input bits of its own, at least its
  passed width (a partial sum may pay b of its b_C bits and keep the rest for its own face; fan-out 1). It is a premise
  on a reference training circuit, false on a declared one, whose wiring the prover controls. It is stronger than
  Security's `TrainingIncompressible`, which charges a received partial sum b_a, takes the larger of the input and
  partial-sum counts rather than their sum, and whose `Chain` has no product of two activations. `rows_of` checks its
  dataflow side (`_Rates`) and counts only products whose operands both pass; the gate-level side is assumed.
- P-width extended: every value a block reads while making operands whole costs its passed width, attention's and the
  optimizer's included, not only a counted product's operands.
- P-match: the elementwise steps between a product's outputs and the next products' operands map operand entries one
  to one, so fan-out is at most fan-in (te_core.capture.products checks it on a traced step).
- P-share: counted products share arrays only as forward/dX/dW triples. A product of an array with itself (`self`)
  can do 2α and is not counted.
- For every method: no product is computed by a bilinear algorithm with fewer multiplications than its MACs (a
  Strassen-type block does k^{ω₀/2}), and no operand is read through a code shorter than its passed width.

Attention's products (two token-indexed dims among m, k and n) are not counted: their composite cap is k²/(3072·d_h)
MACs a block (Q, K and V rows at 16 bits, the output's 32-bit accumulators), below α while k < 175,000 at d_h = 64, so
counting them would need their own cap. Elementwise ops, norms and reductions reuse each entry O(1) times, so k caps
them only through the GEMM outputs a block finishes; they enter only through the widths they pass.
"""
from __future__ import annotations

import argparse
import json
import math
import operator
import sys
from dataclasses import dataclass
from pathlib import Path

from .policy import CONSOLE, REFERENCE_K_BITS as PANEL_K, SESSION_TOKENS as PANEL_TOKENS

ACC_BITS = 32
SHARE = 2 / math.sqrt(3)
KINDS = ("fresh", "theta", "expansion", "attention", "self", "unaffected")
#: the console's policy (c gates, k bits) is te_core.policy's `CONSOLE`; the panels' c is `PANEL_TOKENS` forward-token
#: gates of the start, at k = `PANEL_K`
VECTORS = Path(__file__).resolve().parent / "tests" / "io_bound_vectors.json"


# -- the cap and the term ---------------------------------------------------------------------------------------------

def alpha(k: float, b_a: float, b_b: float, b_c: float = ACC_BITS) -> float:
    """The single-product cap's chain-reading term: MACs a block does from k bits, its chains entering as partial sums."""
    return (k / 3) ** 1.5 / math.sqrt(b_a * b_b * b_c)


def beta(k: float, b_a: float, b_b: float) -> float:
    """The coefficient of √s, s the chains the block starts."""
    return (k / 2) / math.sqrt(b_a * b_b)


def mu(k: float, b_a: float, b_b: float, b_c: float = ACC_BITS) -> float:
    """The linear form's slope: MACs a block adds for each bit of b_C·s, s the chains it starts."""
    return 3 * math.sqrt(3) / 16 * math.sqrt(k) / math.sqrt(b_a * b_b * b_c)


def feed(b: float, b_c: float = ACC_BITS) -> float:
    """Λ's factor for a partial sum that pays both its own face and, finished, the next product's b-bit operand."""
    return math.sqrt(b_c / (b_c - b)) if 2 * b <= b_c else 2 * math.sqrt(b / b_c)


def whole_chain(k: float, k_min: float, b_min: float) -> float:
    """Λ's factor for operand entries made whole in the block from chains of length ≥ `k_min` (module docstring); ∞
    where no bound exists (r > 1/4)."""
    r = k / (3 * k_min ** 2 * b_min)
    if r > 0.25:
        return math.inf
    return (1 + 2 * r / (1 - 2 * r + math.sqrt(1 - 4 * r))) ** 1.5


def blocks(G: float, gamma: float, B: float) -> float:
    """The least N with N·Γ + B·√N ≥ G."""
    if G <= 0:
        return 0.0
    s = 2 * G / (math.sqrt(B * B + 4 * gamma * G) + B)
    return s * s


def _lam_proved(ps: list[dict], k: float, whole: float | None = None) -> float:
    """Λ proved on the products `ps`: SHARE·feed·whole_chain at their widest operand, `whole_chain` the step's
    whole-making system's (`whole`, from `whole_chain_of`) or, without it, the one-class rule at their shortest
    whole-making chain; ∞ where void."""
    fd = max(feed(max(p["b_a"], p["b_b"]), p["b_c"]) for p in ps)
    if whole is not None:
        return SHARE * fd * whole
    ks = [p["k_whole"] for p in ps if p.get("k_whole")]
    wc = whole_chain(k, min(ks), min(min(p["b_a"], p["b_b"]) for p in ps)) if ks else 1.0
    return SHARE * fd * wc


def _figures(ps: list[dict], lam: float, c: float, k: float, C: float) -> dict:
    """The blocks and ρ at Λ = `lam` on the products `ps` of a step of |C| = `C` gates (`bound`)."""
    G = sum(p["gates_per_mac"] * p["macs"] for p in ps)
    lin = [(p["gates_per_mac"] * p["macs"], p["gates_per_mac"] * mu(k, p["b_a"], p["b_b"], p["b_c"]) * p["b_c"] *
            p["outputs"]) for p in ps]
    gamma0 = max(p["gates_per_mac"] * alpha(k, p["b_a"], p["b_b"], p["b_c"]) for p in ps)
    Q = sum(p["gates_per_mac"] ** 2 * p["outputs"] / (p["b_a"] * p["b_b"]) for p in ps)
    cell = max(p["cell_gates"] for p in ps)
    gamma, B = lam * gamma0, lam * (k / 2) * math.sqrt(Q)
    net = sum(max(0.0, g - lam * x) for g, x in lin)
    root, linear = blocks(G, gamma, B), net / gamma
    n = max(root, linear)
    n_cell = max(blocks(G, gamma + k * cell, B), net / (gamma + k * cell))
    return {"lam": lam, "gamma": gamma, "B": B, "blocks_root": root, "blocks_linear": linear, "blocks": n,
            "rho": n * c / C, "rho_main": G / gamma * c / C, "blocks_no_cell": n_cell, "rho_no_cell": n_cell * c / C}


def _e_of(fed: list[dict], made_whole: dict) -> float:
    """e on products whose operands' feeds are `fed` ({K: A}): Σ_K μ_K·max_P A_K,P over the classes in `made_whole`."""
    top: dict = {}
    for f in fed:
        for K, a in f.items():
            if a > 0 and K in made_whole:
                top[K] = max(top.get(K, 0.0), a)
    return sum(a * made_whole[K] for K, a in top.items())


def bound(products: list[dict], c: float, k: float, C: float, whole: float | None = None,
          made_whole: dict | None = None) -> dict:
    """The bound (module docstring) at policy (c, k) on a step of |C| = `C` gates whose counted products are `products`,
    each {"macs", "outputs", "k_dim", "k_whole", "b_a", "b_b", "b_c", "gates_per_mac", "cell_gates"} (`k_whole` None:
    no chain makes an operand whole by itself): the cap's terms on them all (`whole_chain` None where void), Λ proved
    and numerical, and for each the blocks and ρ with P-cell (the larger of the root form's, `blocks_root`, and the
    linear form's on its best subset, `blocks_linear`), without the chain-start terms (`rho_main`) and without P-cell.

    Both forms hold on any subset of the products, so the proved figures are each the best over the subsets that keep
    the products whose whole-making chain is at least a threshold, one threshold for each `k_whole` among them and one
    keeping only those with none, every product's gates still in |C| (a product with a short chain would otherwise
    void Λ for the whole step): `subset` (with P-cell) and `subset_no_cell` give the threshold (`k_whole_min`, None:
    only products with no whole-making chain), the products kept (`kept`, indices into the counted ones), how many are
    dropped and their share of the counted MACs, and that subset's Λ. Where every subset's Λ is void (`whole_chain`
    past r = 1/4) the proved figures are None, never 0; with no counted product they are 0.

    `whole` is the step's `whole_chain` from its whole-making system (`whole_chain_of(chain_classes(graph, k), k)`),
    one factor for every subset, ∞ where a cycle with no fixed point feeds a counted product (void). Without it Λ takes
    the one-class rule at each subset's least `k_whole`, which holds only where no shorter chain is nested below a kept
    operand: attention's d_h-long chains are, on a transformer's step.

    `made_whole` is the system's made-whole shares by class (`whole_chain_of`'s `mu`, ∞ where a diverged cycle reaches
    the class), with each product's `fed` ([K, A] pairs: the largest factor from class K's outputs into its operands);
    `whole` is then not read. Each subset's whole-making factor is its own, (1 + e_S)^{3/2} with e_S = Σ_K μ_K·max over
    the subset's products of their A_K (a block making a kept operand whole runs only the chains that feed it), and the
    subsets also drop the products whose own e is largest, or that are fed by the classes of largest μ, one threshold
    for each value among them: so where a short chain's class makes e large (an adapter's rank-r class), the term is
    never below what the products it doesn't feed give alone. Each subset's `rule` says which threshold kept it
    (``k_whole``, ``e`` or ``mu``), with its `threshold`, its `e` and the classes only its dropped products read
    (`dropped_fed`)."""
    ps = [p for p in products if p["macs"]]
    G = sum(p["gates_per_mac"] * p["macs"] for p in ps)
    out = {"c": c, "k": k, "C_gates": C, "G": G, "share": G / C if C else 0.0}
    zero = {"lam": 0.0, "gamma": 0.0, "B": 0.0, "blocks_root": 0.0, "blocks_linear": 0.0, "blocks": 0.0, "rho": 0.0,
            "rho_main": 0.0, "blocks_no_cell": 0.0, "rho_no_cell": 0.0}
    if not ps:
        return out | {"gamma0": 0.0, "Q": 0.0, "L": 0.0, "feed": 1.0, "whole_chain": 1.0,
                      "proved": zero | {"subset": None, "subset_no_cell": None}, "numerical": zero}
    if made_whole is not None and not all("fed" in p for p in ps):
        raise ValueError("bound(made_whole=...) reads each product's `fed` (te_core.capture.products' rows from te_core 4.12.1 "
                         "on); without them pass the step's `whole`")
    fed = [{K: a for K, a in p["fed"]} for p in ps] if made_whole is not None else None
    ks = [p["k_whole"] for p in ps if p.get("k_whole")]
    if made_whole is not None:
        e_all = _e_of(fed, made_whole)
        wc = (1 + e_all) ** 1.5 if math.isfinite(e_all) else math.inf
    else:
        wc = whole if whole is not None else \
            whole_chain(k, min(ks), min(min(p["b_a"], p["b_b"]) for p in ps)) if ks else 1.0
    out |= {"gamma0": max(p["gates_per_mac"] * alpha(k, p["b_a"], p["b_b"], p["b_c"]) for p in ps),
            "Q": sum(p["gates_per_mac"] ** 2 * p["outputs"] / (p["b_a"] * p["b_b"]) for p in ps),
            "L": sum(p["gates_per_mac"] * mu(k, p["b_a"], p["b_b"], p["b_c"]) * p["b_c"] * p["outputs"] for p in ps),
            "feed": max(feed(max(p["b_a"], p["b_b"]), p["b_c"]) for p in ps),
            "whole_chain": wc if math.isfinite(wc) else None, "numerical": _figures(ps, SHARE, c, k, C)}
    macs = sum(p["macs"] for p in ps)
    subsets: dict[tuple, tuple[str, float | None]] = {}
    for t in [*sorted(set(ks)), math.inf]:
        kept = tuple(i for i, p in enumerate(ps) if (p.get("k_whole") or math.inf) >= t)
        subsets.setdefault(kept, ("k_whole", t if math.isfinite(t) else None))
    if made_whole is not None:
        own = [_e_of([f], made_whole) for f in fed]
        top = [max((made_whole[K] for K, a in f.items() if a > 0 and K in made_whole), default=0.0) for f in fed]
        for rule, of in (("e", own), ("mu", top)):
            for t in sorted({x for x in of if math.isfinite(x)}):
                subsets.setdefault(tuple(i for i, x in enumerate(of) if x <= t), (rule, t))
    best: dict[str, dict] = {}
    for kept, (rule, t) in subsets.items():
        if not kept:
            continue
        sub = [ps[i] for i in kept]
        if made_whole is None:
            lam, e = _lam_proved(sub, k, whole), None
        else:
            e = _e_of([fed[i] for i in kept], made_whole)
            lam = SHARE * max(feed(max(p["b_a"], p["b_b"]), p["b_c"]) for p in sub) * (1 + e) ** 1.5 \
                if math.isfinite(e) else math.inf
        if not math.isfinite(lam):
            continue
        f = _figures(sub, lam, c, k, C)
        least = [p["k_whole"] for p in sub if p.get("k_whole")]
        f["subset"] = {"k_whole_min": min(least) if least else None, "kept": list(kept), "dropped": len(ps) - len(kept),
                       "dropped_macs_share": 1 - sum(p["macs"] for p in sub) / macs, "lam": lam}
        if made_whole is not None:
            reads = {K for i in kept for K, a in fed[i].items() if a > 0 and K in made_whole}
            gone = {K for i in set(range(len(ps))) - set(kept) for K, a in fed[i].items() if a > 0 and K in made_whole}
            f["subset"] |= {"rule": rule, "threshold": t, "e": e, "dropped_fed": sorted(gone - reads)}
        for fig in ("blocks", "blocks_no_cell"):
            if fig not in best or f[fig] > best[fig][fig]:
                best[fig] = f
    if not best:
        out["proved"] = dict.fromkeys(zero) | {"subset": None, "subset_no_cell": None}
        return out
    cell, no_cell = best["blocks"], best["blocks_no_cell"]
    out["proved"] = {x: cell[x] for x in zero if x not in ("blocks_no_cell", "rho_no_cell")} | {
        "blocks_no_cell": no_cell["blocks_no_cell"], "rho_no_cell": no_cell["rho_no_cell"],
        "subset": cell["subset"], "subset_no_cell": no_cell["subset"]}
    return out


# -- a captured step's products -----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """One GEMM of a captured step as the bound reads it (`rows_of`)."""
    node: str
    part: str
    phase: str
    op: str
    dims: dict
    macs: int
    outputs: int
    k_dim: int
    gates_per_mac: float
    cell_gates: float
    affected: tuple[bool, bool]
    widths: tuple[int, int]
    rates: tuple[float, float]
    kind: str
    step: int = 0
    #: the shortest chain that makes an operand entry whole by itself (`_chain_ks`), ∞ where none does
    k_whole: float = math.inf

    def product(self) -> dict:
        return {"macs": self.macs, "outputs": self.outputs, "k_dim": self.k_dim,
                "k_whole": self.k_whole if math.isfinite(self.k_whole) else None, "b_a": self.widths[0],
                "b_b": self.widths[1], "b_c": ACC_BITS, "gates_per_mac": self.gates_per_mac, "cell_gates": self.cell_gates}


class _Step:
    """A captured step's dataflow as te_core.capture.structure builds it, with its dependent slots (`StepGraph`'s), so
    each value is affected exactly when the capture's update-dependence says so."""

    def __init__(self, graph) -> None:
        from te_core.capture.structure import _closure, _Flow, _phases, _shape

        base, varied, _ = graph._inputs
        if base.region is not None:
            raise ValueError("io_bound reads a step captured by capture_step or capture_fn, not a recorded program")
        gm = self.gm = base.gm
        self.nodes = list(gm.graph.nodes)
        pos = {n: i for i, n in enumerate(self.nodes)}
        vnodes = [list(t.gm.graph.nodes) for t in varied]

        def vshape(n, idx, vs):
            v = (vs[pos[n]] if vs is not None else n).meta.get("val")
            if idx is not None:
                v = v[idx] if isinstance(v, (list, tuple)) and idx < len(v) else None
            return _shape(v)

        def tok(n, idx) -> bool:
            s = vshape(n, idx, None)
            return s is not None and any(vshape(n, idx, vs) != s for vs in vnodes)

        self.phases = _phases(gm)
        flow = self.flow = _Flow(gm, self.phases, tok)
        self.attrs = base.attrs
        params = {t: v for t, v in base.attrs.items() if v[0] == "parameter"}
        entry = {key: v for v, key in flow.source.items() if key[0] == "attr"}
        writes: dict = {}
        for key, v, _ in flow.writes:
            if key[0] == "attr":
                writes.setdefault(key, []).append(v)
        before = {("attr", t) for t in base.written_before if ("attr", t) in entry}
        seeded = {key for key in entry if key[1] in params and params[key[1]][1] in graph.updated}
        slots = {key for key in set(writes) | before if key[1] not in params}
        while True:
            seeds = {entry[key] for key in slots | seeded if key in entry}
            dep = _closure(flow, seeds.__contains__)
            kept = {key for key in slots if key not in writes or any(dep[v] for v in writes[key])}
            if kept == slots:
                break
            slots = kept
        self.dep = dep
        self.updated = graph.updated
        self.records = {(m.node, m.part): m for m in graph.matmuls}

    def values(self, n) -> list[int]:
        v = self.flow.val.get(n)
        return [v] if isinstance(v, int) else [x for x in v if isinstance(x, int)] if isinstance(v, list) else []

    def affected(self, n) -> bool:
        return any(self.dep[v] for v in self.values(n))

    def reads_affected(self, n, a) -> bool:
        """Whether node `n` read an affected value through its argument `a` (storage-versioned, as the capture reads)."""
        return any(self.dep[x] for x in self.flow.reads.get((n, a), ()))

    def reads_tokens(self, n, a) -> bool:
        return any(self.flow.tok[x] for x in self.flow.reads.get((n, a), ()))

    def updated_param(self, a) -> bool:
        s = _root(a, scale=True)
        if not (hasattr(s, "op") and s.op == "get_attr"):
            return False
        hit = self.attrs.get(s.target)
        return hit is not None and hit[0] == "parameter" and hit[1] in self.updated


def _width(n) -> int:
    from te_core.capture.structure import _dtype_bits

    v = n.meta.get("val") if hasattr(n, "meta") else None
    if isinstance(v, (list, tuple)):
        v = next((x for x in v if hasattr(x, "dtype")), None)
    return _dtype_bits(v.dtype) if hasattr(v, "dtype") else 64


def _numel(n) -> int:
    v = n.meta.get("val") if hasattr(n, "meta") else None
    vs = v if isinstance(v, (list, tuple)) else [v]
    return sum(x.numel() for x in vs if hasattr(x, "numel"))


def _tensor_args(n) -> list:
    from te_core.capture.structure import _arg_nodes

    return [a for a in _arg_nodes(n) if hasattr(a.meta.get("val"), "dtype")]


def _root(a, scale: bool = False):
    """The node a value is read from through views and casts (and, with `scale`, scalings by a constant)."""
    from te_core.capture.structure import _SCALE, _WALK, _pk

    for _ in range(10_000):
        if not (hasattr(a, "op") and a.op == "call_function"):
            return a
        if a.target is operator.getitem:
            a = a.args[0]
            continue
        name = _pk(a.target)
        one = len(_tensor_args(a)) == 1 and hasattr(a.args[0], "op")
        if name in _WALK and hasattr(a.args[0], "op") or scale and name in _SCALE and one:
            a = a.args[0]
            continue
        return a
    return a


#: views that repeat their argument's entries
_REPEATS = {"expand", "expand_as", "repeat", "broadcast_to"}
#: ops whose result is their tensor arguments laid side by side
_CONCAT = {"cat", "stack", "concat", "hstack", "vstack"}


class _Rates:
    """The bits of input a block must pay per entry of each affected value (P-width's check; `rows_of`):

    - an affected graph attribute (a parameter or state the run produced) and a sum over tokens (a parameter-indexed
      GEMM result) are paid at their width;
    - a gather by token data is 0: its rows repeat over the tokens that share an index;
    - a token-indexed GEMM result with an affected parameter side is 4·K²·b_tok·b_par/k at policy k, its sides at their
      passed widths (`_passed_width`): by LW, x rows and y columns of K entries read in k bits make at most x·y entries, so a block making E of
      them whole pays 2K·√(b_tok·b_par·E) bits, at least that an entry within k (the result's width b once
      K² ≥ k/(4b): K ≥ 32 at the console's k and 16 bits, which a rank-16 adapter's second product is not). A token side
      cheaper than its width (a gather's rows) makes whole columns of the result, which a consumer's chains use at most
      k²/(K·b·b_C) MACs a block, within its α once K ≥ 3^{3/2}·√(k/b_C) (235 at the console's k); below that the
      result is priced as if the weight were θ. With a parameter side outside the update it is its token side's rate
      times K/n (a frozen weight widening n > K passes fewer bits an entry than its width, narrowing restores them).
      A token side's rate is compared with its passed width, as `paid` compares an operand's;
    - an attention product is rate(A)·K/n + rate(B)·K/m;
    - a view or cast keeps its argument's rate, a repeating view spreads it, a concatenation sums its parts', and any
      other op takes the largest rate(input)·numel(input)/numel(output) over its affected inputs (a broadcast gain passes
      almost none, a reduction restores);

    each capped at the value's width. Unaffected values cost nothing: a block computes them from θ and token data."""

    def __init__(self, st: _Step, k: float) -> None:
        from te_core.capture.structure import _GATHERS, _WALK, _gemms, _pk

        self.st, self.k = st, k
        rate: dict = {}
        for n in st.nodes:
            if not st.affected(n):
                rate[n] = 0.0
                continue
            w = _width(n)
            if n.op != "call_function":
                rate[n] = float(w)
                continue
            if n.target is operator.getitem:
                rate[n] = min(w, rate.get(n.args[0], float(w)))
                continue
            name, gs = _pk(n.target), _gemms(n)
            if gs:
                rate[n] = min(float(w), max(self._gemm(n, g, rate) for g in gs))
                continue
            if name in _GATHERS and len(n.args) > 1 and hasattr(n.args[1], "op") and not st.reads_affected(n, n.args[1]):
                rate[n] = 0.0
                continue
            ins = [a for a in _tensor_args(n) if st.reads_affected(n, a)]
            out = max(_numel(n), 1)
            if not ins:
                r = float(w)
            elif name in _WALK and name not in _REPEATS and ins[0] is n.args[0]:
                r = rate.get(ins[0], float(w))
            elif name in _CONCAT:
                r = sum(rate.get(a, 0.0) * _numel(a) for a in ins) / out
            else:
                r = max(rate.get(a, 0.0) * _numel(a) / out for a in ins)
            rate[n] = min(float(w), r)
        self.rate = rate

    def _gemm(self, n, g, rate) -> float:
        st = self.st
        A, B = g.operands[0], g.operands[1]
        d = g.dims
        rec = st.records.get((n.name, g.part))
        td = rec.token_dims if rec is not None else {}
        if not any(x in td for x in ("batch", "m", "n")):
            return math.inf
        ra = rate.get(A, 0.0) if st.reads_affected(n, A) else 0.0
        rb = rate.get(B, 0.0) if st.reads_affected(n, B) else 0.0
        if sum(x in td for x in ("m", "k", "n")) >= 2:
            return ra * d["k"] / max(d["n"], 1) + rb * d["k"] / max(d["m"], 1)
        token_a = "m" in td or "n" not in td
        param, r_tok, spread = (B, ra, d["n"]) if token_a else (A, rb, d["m"])
        tok = A if token_a else B
        b_tok, b_par = _passed_width(st, n, tok), _passed_width(st, n, param)
        if st.reads_affected(n, param) and (is_paid(r_tok, b_tok) or d["k"] >= 3 ** 1.5 * math.sqrt(self.k / ACC_BITS)):
            return 4 * d["k"] ** 2 * b_tok * b_par / self.k
        return r_tok * d["k"] / max(spread, 1)

    def paid(self, n, a) -> bool:
        return is_paid(self.rate.get(a, 0.0), _passed_width(self.st, n, a))


def is_paid(rate: float, width: float) -> bool:
    """Whether an operand a block pays `rate` bits an entry of is paid at its passed width `width` (P-width): a cast
    keeps its argument's rate, and a block can cut before it, so a widened operand is paid at the narrower width."""
    return rate >= 0.999 * width


def _passed_width(st: _Step, n, a) -> int:
    """The narrowest width a value has on its way into operand `a` of node `n`: back through views, casts, scalings and
    ops with one same-size affected tensor argument (others broadcast or unaffected), which a block can apply itself."""
    from te_core.capture.structure import _pk

    b = _width(a)
    for _ in range(10_000):
        if not (hasattr(a, "op") and a.op == "call_function"):
            break
        if a.target is operator.getitem:
            a = a.args[0]
            b = min(b, _width(a))
            continue
        same = [x for x in _tensor_args(a) if _numel(x) == _numel(a) and st.reads_affected(a, x)]
        if len(same) != 1 or _pk(a.target) in _CONCAT:
            break
        a = same[0]
        b = min(b, _width(a))
    return b


def _chain_ks(st: _Step) -> dict:
    """For each value, the longest chain among those a block must run to make an entry of it whole (`Row.k_whole`): a
    GEMM with both operands affected is its K (its longest part's); a value read at its width (an updated parameter or
    state, a GEMM's output with an operand outside the update) is ∞, an unaffected one 0; any other op takes the largest
    over its affected arguments at least its size (each is needed; a smaller one, a broadcast gain or a row's
    statistic, is shared by many entries)."""
    from te_core.capture.structure import _GATHERS, _gemms, _pk

    out: dict = {}
    for n in st.nodes:
        if not st.affected(n):
            out[n] = 0.0
        elif n.op != "call_function":
            out[n] = math.inf
        elif n.target is operator.getitem:
            out[n] = out.get(n.args[0], math.inf)
        elif gs := _gemms(n):
            both = all(st.reads_affected(n, g.operands[0]) and st.reads_affected(n, g.operands[1]) for g in gs)
            out[n] = float(max(g.dims["k"] for g in gs)) if both else math.inf
        elif _pk(n.target) in _GATHERS and len(n.args) > 1 and hasattr(n.args[1], "op") and \
                not st.reads_affected(n, n.args[1]):
            # a gather's rows repeat over the tokens that share an index (rate 0): read, never made whole by a chain
            out[n] = 0.0
        else:
            cat = _pk(n.target) in _CONCAT
            ins = [a for a in _tensor_args(n) if st.reads_affected(n, a) and (cat or _numel(a) >= _numel(n))]
            out[n] = max((out.get(a, math.inf) for a in ins), default=0.0)
    return out


#: a class's c_j (`chain_classes`): products of two arrays, which under P-share form triangles at most, so
#: Σ_edges x_i·x_j ≤ a²/3 (Motzkin–Straus at clique number 3); a product of an array with itself has at most x²/K² outputs
C_PAIR, C_SELF = 1 / 3, 1.0


def _whole_making(st: _Step, rates: _Rates) -> tuple[dict, dict]:
    """For each value, the chain sources whose outputs reach it ({GEMM node with both operands affected: the largest
    factor along a path}) and the largest factor any path into it has for paid entries (`chain_classes`)."""
    from te_core.capture.structure import _gemms, _pk

    def both(n) -> bool:
        gs = _gemms(n)
        return bool(gs) and all(st.reads_affected(n, x.operands[0]) and st.reads_affected(n, x.operands[1]) for x in gs)

    def gk(n) -> int:
        return max(x.dims["k"] for x in _gemms(n))

    feeds: dict = {}
    paid: dict = {}
    for n in st.nodes:
        res: dict = {}
        best = 1.0
        if not st.affected(n) or n.op != "call_function":
            pass
        elif both(n):
            res = {n: 1.0}
        elif _gemms(n):
            for x in _gemms(n):
                for a in (x.operands[0], x.operands[1]):
                    if st.reads_affected(n, a):
                        f = x.dims["n"] / x.dims["k"] if a is x.operands[0] else x.dims["m"] / x.dims["k"]
                        for s, v in feeds.get(a, {}).items():
                            res[s] = max(res.get(s, 0.0), v * f)
                        best = max(best, f * paid.get(a, 1.0), f)
        else:
            cat = n.target is not operator.getitem and _pk(n.target) in _CONCAT
            ins = [n.args[0]] if n.target is operator.getitem else [
                a for a in _tensor_args(n) if st.reads_affected(n, a) and (cat or _numel(a) >= _numel(n))]
            for a in ins:
                best = max(best, paid.get(a, 1.0))
            # every same-size argument's entry is needed, so one argument paid at its width bounds the output (the one
            # fed by the least mass); a concatenation's parts, or arguments none of which is paid at its width (a
            # gather's rows, a widening product's), take the union
            whole = [a for a in ins if is_paid(rates.rate.get(a, 0.0), _width(a))]
            if not cat and len(ins) > 1 and whole:
                ins = [min(whole, key=lambda a: sum(v * v / gk(s) ** 2 for s, v in feeds.get(a, {}).items()))]
            for a in ins:
                for s, v in feeds.get(a, {}).items():
                    res[s] = max(res.get(s, 0.0), v)
        feeds[n] = res
        paid[n] = 1.0 if both(n) else best
    return feeds, paid


def chain_classes(graph, k: float = CONSOLE[1], gemm_dtype: str | None = "bfloat16") -> dict:
    """A captured step's whole-making system at policy `k` (module docstring, `whole_chain_of`): every GEMM with both
    operands affected (counted, dropped, attention's, a product of an array with itself) in a class by its chain K,
    each class with c (`C_PAIR`, or `C_SELF` where it holds a self product), P (the largest factor any path into its
    operands has, for paid entries) and feeds ({K_i: A_ij}, the largest factor from class i's outputs into its
    operands: 1 through elementwise ops, n/K through a product with one operand outside the update); `counted` is the
    same for the counted products' operands, and `b` the least width of a class's operand."""
    from te_core.capture.structure import _gemms

    st = _Step(graph)
    rows = {(r.node, r.part): r for r in rows_of(graph, gemm_dtype, k)}
    feeds, paid = _whole_making(st, _Rates(st, k))
    by_name = {n.name: n for n in st.nodes}
    classes: dict = {}
    counted: dict = {}
    b = math.inf
    for m in graph.matmuls:
        n = by_name[m.node]
        gs = _gemms(n)
        x = next(y for y in gs if y.part == m.part)
        row = rows[(n.name, x.part)]
        ops = [a for a in (x.operands[0], x.operands[1]) if st.reads_affected(n, a)]
        src: dict = {}
        for a in ops:
            for s, v in feeds.get(a, {}).items():
                K = max(y.dims["k"] for y in _gemms(s))
                src[K] = max(src.get(K, 0.0), v)
        if all(st.reads_affected(n, y.operands[0]) and st.reads_affected(n, y.operands[1]) for y in gs):
            e = classes.setdefault(x.dims["k"], {"c": C_PAIR, "P": 1.0, "feeds": {}})
            e["c"] = max(e["c"], C_SELF if row.kind == "self" else C_PAIR)
            e["P"] = max([e["P"], *(paid.get(a, 1.0) for a in ops)])
            for K, v in src.items():
                e["feeds"][K] = max(e["feeds"].get(K, 0.0), v)
            b = min(b, *row.widths)
        if row.kind == "fresh":
            for K, v in src.items():
                counted[K] = max(counted.get(K, 0.0), v)
    return {"classes": classes, "counted": counted, "b": b if math.isfinite(b) else 16}


def _components(feeds: dict) -> list[list]:
    """The strongly connected components of the graph whose edges are i → j for each class i in `feeds[j]` (Tarjan),
    feeding components before the ones they feed."""
    index: dict = {}
    low: dict = {}
    stack: list = []
    on: set = set()
    out: list = []

    def visit(v):
        index[v] = low[v] = len(index)
        stack.append(v)
        on.add(v)
        for w in (j for j, f in feeds.items() if v in f):
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            out.append(sorted(comp))

    for v in sorted(feeds):
        if v not in index:
            visit(v)
    return out[::-1]


def whole_chain_of(system: dict, k: float, iters: int = 100_000, tol: float = 1e-12) -> dict:
    """The least fixed point of `system` (`chain_classes`) at policy `k`: each class j's output entries made whole, in
    units of k/b, μ_j ≤ r_j·(P_j + Σ_i A_ij·μ_i)² with r_j = c_j·k/(K_j²·b). The classes are solved one strongly
    connected component at a time, feeders first (`_components`). A class on no cycle is its right side exactly, from
    the classes that feed it, however large. A cycle (`cycles`) is iterated from 0 (induction on nesting depth bounds μ
    by any post-fixed point) to a post-fixed point, which it then is; or it diverges, which is proved once an iterate's
    increase stops shrinking (the right side is monotone and convex, so the increases never shrink again and μ grows
    without bound) or where no post-fixed point is reached in `iters` rounds. A diverged class, and every class it
    feeds, has μ ∞ (`diverged`). e = Σ_i A_i,counted·μ_i and `whole_chain` = (1 + e)^{3/2}; e is None and `whole_chain`
    ∞ (the proved bound void on the whole step) only where a diverged class feeds a counted product. `bound` with
    `made_whole` = `mu` takes each subset's own e, so a subset that drops those products keeps a proved Λ."""
    cls, b = system["classes"], system["b"]
    r = {K: e["c"] * k / (K * K * b) for K, e in cls.items()}
    feeds = {K: {i: v for i, v in e["feeds"].items() if i in cls and v > 0} for K, e in cls.items()}
    mu: dict = {}
    cycles = []

    def rhs(K, at: dict) -> float:
        if not r[K]:
            return 0.0
        x = cls[K]["P"] + sum(v * at[i] for i, v in feeds[K].items())
        return r[K] * x * x

    for comp in _components(feeds):
        if len(comp) == 1 and comp[0] not in feeds[comp[0]]:
            mu[comp[0]] = rhs(comp[0], mu)
            continue
        cycles.append(comp)
        at = mu | dict.fromkeys(comp, 0.0)
        last: dict | None = None
        fixed = False
        for _ in range(iters):
            new = {K: rhs(K, at) for K in comp}
            if not all(math.isfinite(v) for v in new.values()):
                break
            d = {K: new[K] - at[K] for K in comp}
            if last is not None and any(last[K] > tol * max(1.0, at[K]) for K in comp) and \
                    all(d[K] >= last[K] for K in comp):
                break
            last = d
            at |= new
            if all(d[K] <= tol * max(1.0, new[K]) for K in comp):
                up = at | {K: new[K] * (1 + 1e-9) for K in comp}
                if all(rhs(K, up) <= up[K] for K in comp):
                    at, fixed = up, True
                    break
        mu |= {K: at[K] for K in comp} if fixed else dict.fromkeys(comp, math.inf)
    diverged = sorted(K for K, v in mu.items() if math.isinf(v))
    e = sum(v * mu[i] for i, v in system["counted"].items() if i in mu and v > 0)
    if math.isinf(e):
        return {"r": r, "mu": mu, "e": None, "whole_chain": math.inf, "diverged": diverged, "cycles": cycles}
    return {"r": r, "mu": mu, "e": e, "whole_chain": (1 + e) ** 1.5, "diverged": diverged, "cycles": cycles}


def p_match(graph, k: float = CONSOLE[1]) -> list[str]:
    """P-match's check within one op: the nodes on the way into a class's operand (`chain_classes`) that hand one entry
    made whole to several, a gather by token data of a table some chain's outputs reach (its rows repeat over the
    tokens that share an index) or a repeating view of a value some chain's outputs reach whose result is carried on,
    as te_core.capture.products' check has them. A gather of a table no chain reaches (an embedding of θ′, read at its
    width) hands on no entry made whole: its rows' repeats are the bound's to drop, every product reading them before
    an op makes them token-specific (te_core.capture.products' `gathered`, te_core.formalism). The system carries
    made-whole entries through an op at factor 1 by its same-size arguments, charging a smaller one (a row's
    statistic, a gain) nothing, which a same-size argument bounds. A value read by several ops is matched through the
    other arguments of the ops that read it (a residual add's), by structure, and isn't checked here."""
    from te_core.capture.structure import _GATHERS, _gemms, _pk

    st = _Step(graph)
    feeds, _ = _whole_making(st, _Rates(st, k))
    reach: set = set()
    for n in reversed(st.nodes):
        if n.op != "call_function":
            continue
        gs = _gemms(n)
        if gs:
            if all(st.reads_affected(n, x.operands[0]) and st.reads_affected(n, x.operands[1]) for x in gs):
                reach.update(a for x in gs for a in (x.operands[0], x.operands[1]))
        elif n in reach:
            reach.update([n.args[0]] if n.target is operator.getitem else
                         [a for a in _tensor_args(n) if st.reads_affected(n, a)])
    bad = []
    for n in st.nodes:
        if n not in reach or n.op != "call_function" or n.target is operator.getitem or not st.affected(n):
            continue
        name = _pk(n.target)
        a = n.args[0] if n.args and hasattr(n.args[0], "op") else None
        if name in _GATHERS and len(n.args) > 1 and hasattr(n.args[1], "op") and not st.reads_affected(n, n.args[1]) \
                and a is not None and feeds.get(a) \
                or name in _REPEATS and a is not None and _numel(n) > _numel(a) and feeds.get(n):
            bad.append(n.name)
    return bad


def _gemm_price(phase: str, op: str, dtype: str | None, gemm_dtype: str | None) -> tuple[float, float]:
    """(gates a MAC, gates a gadget) of a GEMM: its operands' dtype's Definition, or `gemm_dtype`'s for a forward or
    backward GEMM whose operands are wider (CPU autocast runs scaled-dot-product attention's math backend in fp32, a GPU
    in the autocast dtype)."""
    from te_core.capture.structure import _dtype_bits
    from te_core.gates import PRICES, gates, kind_of, per_mac

    import torch

    if gemm_dtype and phase in ("forward", "backward") and dtype is not None and \
            _dtype_bits(getattr(torch, dtype)) > _dtype_bits(getattr(torch, gemm_dtype)):
        dtype = gemm_dtype
    kind = kind_of(phase, "", op, dtype)[0]
    (name, _), = PRICES[kind]
    return per_mac(kind), float(gates(name))


def rows_of(graph, gemm_dtype: str | None = None, k: float = CONSOLE[1]) -> list[Row]:
    """Every GEMM of a step captured by `capture_step`/`capture_fn` as a `Row` at policy k: its operands' affected flags
    (each depends on θ′, the step's dependent slots), passed widths (`_passed_width`), rates (`_Rates`) and shortest
    whole-making chain (`_chain_ks`), and its kind:

    - ``fresh``: both operands affected and paid at their passed width (the bound counts it; an operand widened by a
      cast, such as ``x.float()`` of a bf16 x, keeps its argument's rate and is paid at the narrower width it passed);
    - ``theta``: one operand affected; the other is θ or computed from θ and token data, so a block reads it for nothing;
    - ``expansion``: both affected, one paid below its passed width (a gather's rows, a frozen widening product, a gain);
    - ``attention``: two token-indexed dims among m, k and n (module docstring);
    - ``self``: both operands one array (P-share);
    - ``unaffected``: neither operand affected.
    """
    from te_core.capture.structure import _dtype_name, _gemms, _macs, _pk

    st = _Step(graph)
    rates, chain_k = _Rates(st, k), _chain_ks(st)
    by_name = {n.name: n for n in st.nodes}
    rows = []
    for m in graph.matmuls:
        n = by_name[m.node]
        g = next(x for x in _gemms(n) if x.part == m.part)
        A, B = g.operands[0], g.operands[1]
        aff = (st.reads_affected(n, A), st.reads_affected(n, B))
        td = m.token_dims
        if not any(aff):
            kind = "unaffected"
        elif sum(x in td for x in ("m", "k", "n")) >= 2:
            kind = "attention"
        elif _root(A, scale=True) is _root(B, scale=True):
            kind = "self"
        elif not all(aff):
            kind = "theta"
        elif rates.paid(n, A) and rates.paid(n, B):
            kind = "fresh"
        else:
            kind = "expansion"
        gpm, cell = _gemm_price(m.phase, _pk(n.target), _dtype_name(A), gemm_dtype)
        d = g.dims
        whole = min((v for v in (chain_k.get(A, 0.0), chain_k.get(B, 0.0)) if v > 0), default=math.inf)
        rows.append(Row(n.name, g.part, m.phase, str(n.target), dict(d), _macs(d), d["batch"] * d["m"] * d["n"], d["k"],
                        gpm, cell, aff, (_passed_width(st, n, A), _passed_width(st, n, B)),
                        (rates.rate.get(A, 0.0), rates.rate.get(B, 0.0)), kind, m.step, whole))
    return rows


def step_gates(graph, gemm_dtype: str | None = None) -> float:
    """The step's |C| in gates, each counted op at te_core.gates' price for its dtype as te_core.capture.structure prices
    it: GEMMs (`_gemm_price`), elementwise multiplications and dense linear algebra (`_other_macs`), and the forward's
    other ops that combine an activation with an updated parameter (a gather of an updated table by token data at the
    circuit's gather, any other at its work)."""
    from te_core.capture.structure import _GATHERS, _arg_nodes, _arg_nodes_of, _dtype_bits, _dtype_name, _gemms, \
        _macs, _other_macs, _pk, _work
    from te_core.flops import _is_view, gather_macs
    from te_core.gates import kind_of, per_mac

    import torch

    st = _Step(graph)
    total = 0.0
    for n in st.nodes:
        if n.op != "call_function" or not isinstance(n.target, torch._ops.OpOverload):
            continue
        name, ph = _pk(n.target), st.phases.get(n, "forward")
        if name in ("_record_function_enter_new", "_record_function_exit"):
            continue
        gs = _gemms(n)
        if gs:
            for g in gs:
                total += _macs(g.dims) * _gemm_price(ph, name, _dtype_name(g.operands[0]), gemm_dtype)[0]
            continue
        k, kind = _other_macs(n)
        if kind:
            total += k * per_mac(kind_of(ph, "", name, _dtype_name(n))[0])
        if ph != "forward" or _is_view(n.target):
            continue
        ops = _arg_nodes(n)
        src = set(_arg_nodes_of(n.args[:1])) if name in _GATHERS else set()
        acts = any(st.reads_tokens(n, a) and not st.updated_param(a) for a in ops if a not in src)
        if src and any(st.updated_param(a) for a in src) and acts:
            gm_ = gather_macs(name, n.args[0].meta["val"].shape, n.args[1] if len(n.args) > 1 else None,
                              n.meta["val"].numel())
            total += gm_ * per_mac("gather", _dtype_bits(n.args[0].meta["val"].dtype))
        elif not src and not kind and acts and any(st.updated_param(a) for a in ops):
            total += _work(n) * per_mac(kind_of(ph, "", name, _dtype_name(n))[0])
    return total


def summarize(rows: list[Row], C: float) -> dict:
    """MACs and gates by kind and phase, each kind's share of |C|."""
    out: dict = {}
    for r in rows:
        e = out.setdefault(r.kind, {"macs": 0, "gates": 0.0, "by_phase": {}})
        e["macs"] += r.macs
        e["gates"] += r.macs * r.gates_per_mac
        e["by_phase"][r.phase] = e["by_phase"].get(r.phase, 0) + r.macs
    for e in out.values():
        e["share"] = e["gates"] / C if C else 0.0
    return dict(sorted(out.items(), key=lambda kv: KINDS.index(kv[0])))


def table(rows: list[Row]) -> list[dict]:
    """The rows grouped by (phase, op, dims, kind, affected, widths, rates, k_whole), with counts and MACs: one line per
    distinct product."""
    groups: dict = {}
    for r in rows:
        whole = r.k_whole if math.isfinite(r.k_whole) else None
        key = (r.phase, r.op, json.dumps(r.dims, sort_keys=True), r.kind, r.affected, r.widths,
               tuple(round(x, 3) for x in r.rates), whole)
        e = groups.setdefault(key, {"phase": r.phase, "op": r.op, "dims": r.dims, "kind": r.kind,
                                    "affected": list(r.affected), "widths": list(r.widths),
                                    "rates": [round(x, 3) for x in r.rates], "k_whole": whole, "count": 0, "macs": 0})
        e["count"] += 1
        e["macs"] += r.macs
    return sorted(groups.values(), key=lambda e: (KINDS.index(e["kind"]), e["phase"], -e["macs"]))


# -- the s-shape steps --------------------------------------------------------------------------------------------------

def newton_schulz(g, steps: int = 5):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        s = x @ x.T
        x = a * x + (b * s + c * s @ s) @ x
    return (x.T if tall else x).to(g.dtype)


def _muon(params, lr: float = 0.02, momentum: float = 0.95):
    """The ladder's matrix optimizer: Nesterov momentum, five Newton–Schulz steps in bf16, scaled by √max(1, rows/cols)."""
    import torch

    class Muon(torch.optim.Optimizer):
        def __init__(self, ps):
            super().__init__(ps, {"lr": lr, "momentum": momentum})

        @torch.no_grad()
        def step(self):
            for grp in self.param_groups:
                for p in grp["params"]:
                    if p.grad is None:
                        continue
                    st = self.state.setdefault(p, {"buf": torch.zeros_like(p)})
                    buf = st["buf"].mul_(grp["momentum"]).add_(p.grad)
                    u = newton_schulz(p.grad.add(buf, alpha=grp["momentum"]))
                    rows, cols = p.shape
                    p.add_(u, alpha=-grp["lr"] * max(1.0, rows / cols) ** 0.5)

    return Muon(params)


def _lora(model, r: int):
    """Every block matrix of `model` as ``base(x) + 2·b(a(x))`` with the base frozen, rank r, and everything else frozen."""
    import torch
    import torch.nn.functional as F
    from torch import nn

    class Adapter(nn.Module):
        def __init__(self, base) -> None:
            super().__init__()
            self.base = base
            d_out, d_in = base.weight.shape
            self.a = nn.Parameter(torch.randn(r, d_in) / d_in ** 0.5)
            self.b = nn.Parameter(torch.zeros(d_out, r))

        def forward(self, x):
            return self.base(x) + F.linear(F.linear(x, self.a), self.b) * 2.0

    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        for name in ("wq", "wk", "wv", "wo", "w_gate", "w_up", "w_down"):
            setattr(blk, name, Adapter(getattr(blk, name)))
    return model


def _recipe(name: str, size: str, vocab: int):
    """(model, optimizers) of one step kind: `dense` (the ladder's: Muon on the block matrices, AdamW on the embedding and
    norms), `frozen_embed` (the same with the tied embedding at θ), `lora` (rank-16 adapters on every block matrix, all
    else frozen, AdamW) and `partial` (the top two blocks and the final norm, AdamW, as island shallow)."""
    import torch
    from model import LM, SIZES

    torch.manual_seed(0)
    m = LM(SIZES[size], vocab=vocab)
    ps = dict(m.named_parameters())
    mats = [p for n, p in ps.items() if p.ndim == 2 and n != "embed"]
    norms = [p for p in ps.values() if p.ndim == 1]
    if name in ("dense", "frozen_embed"):
        ps["embed"].requires_grad_(name == "dense")
        adam = norms + ([ps["embed"]] if name == "dense" else [])
        return m, [_muon(mats), torch.optim.AdamW(adam, lr=0.006, betas=(0.9, 0.95), eps=1e-10)]
    if name == "lora":
        m = _lora(m, 16)
        return m, [torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)]
    if name == "partial":
        top = {f"blocks.{i}." for i in range(len(m.blocks) - 2, len(m.blocks))}
        tuned = [p for n, p in ps.items() if n == "norm" or any(n.startswith(t) for t in top)]
        for p in ps.values():
            p.requires_grad_(any(p is q for q in tuned))
        return m, [torch.optim.AdamW(tuned, lr=3e-4, betas=(0.9, 0.95))]
    raise ValueError(f"unknown step {name!r}; one of {', '.join(STEPS)}")


STEPS = ("dense", "lora", "partial", "frozen_embed")


def capture(name: str, size: str = "s", vocab: int = 32000, batch: int = 256, ctx: int = 1024, micro_steps: int = 4):
    """One training step of `name`'s recipe (`_recipe`) on covert_witness's `lf` model, captured: `micro_steps`
    micro-batches of the batch under bf16 autocast, each loss scaled and backpropagated, the gradient clipped to norm 1,
    then the optimizers' steps."""
    import torch
    import torch.nn.functional as F

    from te_core.capture.trace import capture_fn

    model, opts = _recipe(name, size, vocab)

    def step(m, ids, optimizers):
        for o in optimizers:
            o.zero_grad(set_to_none=True)
        for x in ids.chunk(micro_steps):
            with torch.autocast("cpu", dtype=torch.bfloat16):
                logits = m(x[:, :-1])
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1)) / micro_steps
            loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1.0)
        for o in optimizers:
            o.step()

    ids = torch.zeros(batch, ctx + 1, dtype=torch.long)
    return capture_fn(step, model, ids, optimizers=opts, name=name)


def measure(graph, policies: dict, gemm_dtype: str | None = "bfloat16") -> dict:
    """A captured step's |C| and, at each policy {name: (c, k)}, its rows' kinds and products (`rows_of` at k) and the
    bound."""
    C = step_gates(graph, gemm_dtype)
    out = {"name": graph.name, "tokens": graph.tokens, "C_gates": C, "policies": {}}
    for p, (c, k) in policies.items():
        rows = rows_of(graph, gemm_dtype, k)
        fresh = [r.product() for r in rows if r.kind == "fresh"]
        out["policies"][p] = {"c": c, "k": k, "kinds": summarize(rows, C), "products": table(rows),
                              "bound": bound(fresh, c, k, C)}
    return out


def vectors() -> dict:
    """Test vectors: hand-written product sets at the console's and the panels' policies, with the cap and the term."""
    bf16 = {"b_a": 16, "b_b": 16, "b_c": ACC_BITS, "gates_per_mac": 23682 / 16, "cell_gates": 23682.0}
    square = {"macs": 5e14, "outputs": 5e14 / 512, "k_dim": 512, "k_whole": 512} | bf16
    cases = [
        ("one product, square, console", CONSOLE, 1e18, [square]),
        ("one product, square, panel", (2.545e14, PANEL_K), 1e18, [square]),
        ("short chains, K = 16, none whole by itself", CONSOLE, 1e17,
         [{"macs": 5e13, "outputs": 5e13 / 16, "k_dim": 16, "k_whole": None} | bf16]),
        ("short and long chains: the linear form keeps the long", CONSOLE, 1e18,
         [{"macs": 5e13, "outputs": 5e13 / 16, "k_dim": 16, "k_whole": None} | bf16, square]),
        ("operands made whole from K = 128 chains", CONSOLE, 1e18, [square | {"k_whole": 128}]),
        ("operands made whole from K = 64 chains: no subset has a proved bound", CONSOLE, 1e18,
         [square | {"k_whole": 64}]),
        ("a rank-1 adapter's product beside a square one: the best subset drops it", CONSOLE, 1e18,
         [square, {"macs": 5e11, "outputs": 5e11, "k_dim": 1, "k_whole": 1} | bf16]),
        ("mixed widths", CONSOLE, 1e18, [{"macs": 4e14, "outputs": 4e14 / 512, "k_dim": 512, "k_whole": 512} | bf16,
                                         {"macs": 2e14, "outputs": 2e14 / 1408, "k_dim": 1408, "k_whole": 512, "b_a": 8,
                                          "b_b": 16, "b_c": ACC_BITS, "gates_per_mac": 9128 / 32, "cell_gates": 9128.0}]),
        ("f32 operands", CONSOLE, 1e18, [{"macs": 5e13, "outputs": 5e13 / 512, "k_dim": 512, "k_whole": 512, "b_a": 32,
                                          "b_b": 32, "b_c": ACC_BITS, "gates_per_mac": 12637.0, "cell_gates": 12637.0}]),
        ("nothing counted", CONSOLE, 1e18, []),
    ]
    out = []
    for name, (c, k), C, ps in cases:
        b = bound(ps, c, k, C)
        cap = {"alpha": [alpha(k, p["b_a"], p["b_b"], p["b_c"]) for p in ps], "beta": [beta(k, p["b_a"], p["b_b"]) for p in ps],
               "mu": [mu(k, p["b_a"], p["b_b"], p["b_c"]) for p in ps], "gamma0": b["gamma0"], "Q": b["Q"], "L": b["L"],
               "feed": b["feed"], "whole_chain": b["whole_chain"], "lam_proved": b["proved"]["lam"],
               "lam_numerical": b["numerical"]["lam"]}
        term = {f"{lam}_{x}": b[lam][x] for lam in ("proved", "numerical")
                for x in ("blocks_root", "blocks_linear", "blocks", "rho", "rho_main", "blocks_no_cell", "rho_no_cell")} | \
            {"share": b["share"], "proved_subset": b["proved"]["subset"],
             "proved_subset_no_cell": b["proved"]["subset_no_cell"]}
        out.append({"name": name, "inputs": {"c": c, "k": k, "C_gates": C, "products": ps}, "cap": cap, "term": term})
    return {"about": "te_core.io_bound's bound on hand-written product sets: `inputs` to `bound`, `cap` its terms on "
                     "every product, `term` the blocks and ρ, the proved ones on their best subsets "
                     "(`proved_subset`, `proved_subset_no_cell`; null where no subset has a proved Λ) "
                     "(python -m te_core.io_bound vectors)", "vectors": out}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vectors", help="write the test vectors")
    v.add_argument("--out", default=str(VECTORS))
    s = sub.add_parser("steps", help="capture the s-shape steps and bound each at the console's and the panels' policies")
    s.add_argument("--only", default=",".join(STEPS))
    s.add_argument("--size", default="s")
    s.add_argument("--vocab", type=int, default=32000)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--micro-steps", type=int, default=4)
    s.add_argument("--fwd-gates", type=float, default=None,
                   help="the start's forward gates a token, for the panels' c (default: the dense step's forward GEMMs)")
    s.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "vectors":
        Path(a.out).write_text(json.dumps(vectors(), indent=1) + "\n")
        print(a.out)
        return 0
    results = []
    fwd = a.fwd_gates
    for name in a.only.split(","):
        g = capture(name, a.size, a.vocab, a.batch, micro_steps=a.micro_steps)
        if fwd is None:
            rows = rows_of(g, "bfloat16")
            fwd = sum(r.macs * r.gates_per_mac for r in rows if r.phase == "forward") / g.tokens
        res = measure(g, {"console": CONSOLE, "panel": (PANEL_TOKENS * fwd, PANEL_K)})
        results.append(res)
        print(json.dumps({"step": name, "C_gates": res["C_gates"], "policies": {
            p: {"shares": {x: round(e["share"], 5) for x, e in r["kinds"].items()},
                "rho": {lam: round(r["bound"][lam]["rho"]) for lam in ("proved", "numerical")}}
            for p, r in res["policies"].items()}}), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"fwd_gates_per_token": fwd, "steps": results}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
