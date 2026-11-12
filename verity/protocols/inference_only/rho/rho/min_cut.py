"""The min-cut bound's arithmetic: the one place that computes Lean's `HE` and `HEShared`, the bound's two statements
(`HE` and `HE_SHARED` below are their full names).

Standard library only, and it imports nothing of this package, so another benchmark loads it by path. Each function is
one side of an inequality those statements conclude, or that inequality solved for one of its terms, and says which.

The statement. A circuit's charged work is `N` MACs with `O` output wires (Lean's `Work.N` and `Work.O`: an output's
wires, so its values times their width). Under (H) at `q` (`HARDNESS`: a pair of a matmul's two operands' values costs
`q = β_w β_x` cut gates) and (E) at `b` (`EXPORTS`: an output a set of gates finishes costs `b` cut gates to send on),
every partition into `m` blocks that each read at most `k` gates outside `θ` has

    N ≤ m (k + F α) + F (k/2) √(m O / q),    α = ½ √(k³ / (q b)),

and for every `c`, `(c / |C|) (N / (k + F α) − F (k/2) √(O N / q) / (k + F α)^{3/2}) ≤ m c / |C|`. `F` is 1 in `HE`
(two arrays `W` and `X`, no two matmuls sharing both a `w`- and an `x`-value) and `√(2 (1 − 1/ω))` in `HEShared`
(arrays no `ω + 1` of which are pairwise multiplied: `2/√3` where the work holds a layer's forward, input-gradient and
weight-gradient products, 1 with at most two of those kinds).

Units: `k`, `O`, `b` and the `β` in bits (gates of the circuit: a wire is a gate), `N` in MACs, `c` and `|C|` in gates.

What no function here checks, and a caller states of its work: the MACs are pairwise disjoint and each reads its two
words; `val` is one-to-one on a matmul's words (the casts of one value are one word); the sharing its `F` needs; every
block reads at most `k`; and (H) and (E), conjectures one set of gates refutes. A work's weights are outside `θ`: a
registered weight has no path from a fresh input, so (H) holds for it only at `β_w = 0`, where the bound says
nothing.

`beta(k)` is (H)'s `β` for an activation at `k`, the rows `premise.training-cut-hardness.beta` of
`data/premises.toml`, and `bound` the bound at a policy `(c, k, n)` on a circuit's own `N`, `O` and `|C|`.
"""

import math
import tomllib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

PREMISES = Path(__file__).resolve().parent.parent / "data" / "premises.toml"
_ROWS = tomllib.loads(PREMISES.read_text())["premise"]
#: (H) and (E), the bound's two conjectures: the Lean names their rows give (`premise.training-cut-hardness.lean`,
#: `premise.training-cut-exports.lean`)
HARDNESS: str = _ROWS["training-cut-hardness"]["lean"]
EXPORTS: str = _ROWS["training-cut-exports"]["lean"]
#: the two statements' Lean names, in their premises' area: its `Guarantees.HE` and `Guarantees.HEShared`
AREA = HARDNESS.removesuffix(".Assumptions.Hardness")
HE, HE_SHARED = f"{AREA}.Guarantees.HE", f"{AREA}.Guarantees.HEShared"
#: (H)'s `β` for an activation by block size, rows `(k, β)` in bits, `k` rising (`premise.training-cut-hardness.beta`)
BETA: tuple[tuple[float, float], ...] = tuple(
    (float(k), float(b)) for k, b in _ROWS["training-cut-hardness"]["beta"]["value"])

#: a bf16 word's wires. A word's own wires are a cut, so no value costs more than its width and no output more than
#: its own: (H) and (E) at a word's width are the most they can say.
BF16 = 16.0


def alpha(k: float, q: float, b: float) -> float:
    """Lean's `alphaHE`: `α = ½ √(k³ / (q b))`."""
    return math.sqrt(k ** 3 / (q * b)) / 2


def share(omega: int) -> float:
    """Lean's `shareF`: `F = √(2 (1 − 1/ω))`, 1 at `ω = 2` and `2/√3` at `ω = 3`."""
    return math.sqrt(2 * (1 - 1 / omega))


def macs_at_most(m: float, k: float, O: float, q: float, b: float, F: float = 1.0) -> float:
    """The right side of the count, `HE`'s and `HEShared`'s first conjunct: `m` blocks finish at most
    `m (k + F α) + F (k/2) √(m O / q)` of a work's MACs."""
    return m * (k + F * alpha(k, q, b)) + F * (k / 2) * math.sqrt(m * O / q)


def blocks_at_least(N: float, O: float, k: float, q: float, b: float, F: float = 1.0) -> float:
    """The least `m` the count allows a work of `N` MACs and `O` output wires: `N ≤ macs_at_most(m, …)` solved for
    `m`. With `A = k + F α` and `D = F (k/2) √(O / q)` the count is `N ≤ A m + D √m`, whose right side rises with `m`,
    so `√m ≥ (√(D² + 4 A N) − D) / (2 A)`. A real number: a partition has at least its ceiling."""
    a, d = k + F * alpha(k, q, b), F * (k / 2) * math.sqrt(O / q)
    return ((math.sqrt(d * d + 4 * a * N) - d) / (2 * a)) ** 2


def slowdown_at_least(N: float, O: float, size: float, c: float, k: float, q: float, b: float,
                      F: float = 1.0) -> float:
    """The left side of the slowdown, `HE`'s and `HEShared`'s second conjunct:
    `(c / |C|) (N / (k + F α) − F (k/2) √(O N / q) / (k + F α)^{3/2}) ≤ m c / |C|`, at `|C| = size`. It is the count
    with `√m ≤ √(N / (k + F α))` put in its last term, so it is never above `blocks_at_least(…) c / size`, and below 0
    where the outputs' term outweighs the first."""
    a = k + F * alpha(k, q, b)
    return c / size * (N / a - F * (k / 2) * math.sqrt(O * N / q) / a ** 1.5)


def macs_a_block(k: float, o: float, q: float, b: float, F: float = 1.0) -> float:
    """The most MACs a block of a work with `o = O / N` output wires a MAC: the count at `O = o N`, divided by `m`,
    is `N/m ≤ (k + F α) + B √(N/m)` with `B = F (k/2) √(o / q)`, so `N/m ≤ ((B + √(B² + 4 (k + F α))) / 2)²`. It is
    `N / blocks_at_least(N, o N, …)` for every `N`."""
    y = F * (k / 2) * math.sqrt(o / q)
    return ((y + math.sqrt(y * y + 4 * (k + F * alpha(k, q, b)))) / 2) ** 2


def beta(k: float) -> float | None:
    """(H)'s `β` for an activation at a block of `k` bits (`BETA`): the first row's up to its `k`, between two rows
    linear in `log k`, and `None` past the last row's `k`, where there is none and the bound says nothing."""
    if k <= BETA[0][0]:
        return BETA[0][1]
    for (k0, b0), (k1, b1) in pairwise(BETA):
        if k <= k1:
            return b1 if k == k1 else b0 + (b1 - b0) * math.log(k / k0) / math.log(k1 / k0)
    return None


@dataclass(frozen=True)
class Bound:
    """The min-cut bound at a policy on one circuit's work, with what it rests on."""

    #: `ρ = m c / |C| ≥` this, at least 1 (blocks of at most `c` gates cover `|C|`); 1 where the bound says nothing
    slowdown: float
    #: the least `m` the count allows (`blocks_at_least`), a real number; 0 where the bound says nothing
    blocks: float
    #: the most of the work's MACs one block finishes at the work's own `O / N` (`macs_a_block`); `inf` where the
    #: bound says nothing
    macs_a_block: float
    #: the statement: `HE` at `F = 1`, `HEShared` otherwise
    statement: str
    #: the conjectures it takes, (H) and (E)
    premises: tuple[str, ...]
    k: float
    beta_w: float
    #: (H)'s `β` for an activation; `None`: none at this `k`, so no bound
    beta_x: float | None
    b: float
    F: float
    #: `α` (`alpha`); `nan` where there is no bound
    alpha: float

    @property
    def rests_on(self) -> str:
        """What the figure rests on, as a sentence."""
        if self.beta_x is None:
            return (f"no bound at k = {self.k:.4g} bits: (H)'s β for an activation ({HARDNESS}) is stated up to "
                    f"k = {BETA[-1][0]:.4g}")
        return (f"{self.statement} at k = {self.k:.4g} bits and F = {self.F:.4g}, under (H) ({HARDNESS}) at "
                f"β_w = {self.beta_w:.4g} and β_x = {self.beta_x:.4g} bits a value and (E) ({EXPORTS}) at "
                f"b = {self.b:.4g} bits an output, conjectures one set of gates refutes; the work's structure is its "
                "caller's to state")


def bound(c: float, k: float, n: float | None = None, *, N: float, O: float, size: float, beta_w: float = BF16,
          beta_x: float | None = None, b: float = BF16, F: float = 1.0) -> Bound:
    """The bound at the policy `(c, k, n)` (blocks of at most `c` gates, each reading at most `k` fresh input bits and
    giving at most `n` output bits) on a circuit of `size` gates whose charged work is `N` MACs with `O` output wires:
    the least number of blocks the count allows, and the slowdown `m c / |C|` they give. The statement reads no `n`,
    so a policy that states none gives the same bound: a work's outputs enter through `O`. `N` and `O` are the
    caller's, the MACs of the work it charges and their output wires: nothing here converts a count, by a kind's share
    of the products or by a word's width.

    * `beta_w`, bits a weight value costs under (H). Default `BF16`, a bf16 weight's width: the most (H) can hold at.
      A weight with a shorter code costs its code's bits.
    * `beta_x`, bits an activation value costs under (H). Default `beta(k)`, and no bound where it has none.
    * `b`, bits an output costs under (E). Default `BF16`, a bf16 output's width: the most (E) can hold at.
    * `F`. Default 1, `HE`: products of at most two of the three kinds (forward, input-gradient, weight-gradient).
      `share(3)` where the work holds all three (`HEShared`).
    """
    statement = HE if F == 1 else HE_SHARED
    beta_x = beta(k) if beta_x is None else beta_x
    if beta_x is None or N <= 0:
        return Bound(1.0, 0.0, math.inf, statement, (HARDNESS, EXPORTS), k, beta_w, beta_x, b, F, math.nan)
    q = beta_w * beta_x
    m = blocks_at_least(N, O, k, q, b, F)
    return Bound(max(1.0, m * c / size), m, N / m, statement, (HARDNESS, EXPORTS), k, beta_w, beta_x, b, F,
                 alpha(k, q, b))
