"""The min-cut bound's arithmetic: the one place that computes Lean's `HE` and `HEShared`, the bound's two statements,
and `HECharged`, the count with some blocks charged whole (`HE`, `HE_SHARED` and `HE_CHARGED` below are their full
names).

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
nothing. So is any table a word of the work is read from: words that are a function of `θ` and a few fresh gates (a
registered embedding's row, looked up by a token) carry many values behind those gates, (H) fails at any `β` above
the gates over the values, and a caller whose work reads such words leaves the matmuls that read them out of `N`. A
`β` is at most the gates of the word it prices, 16 for a bf16 word, a word's own wires being a cut. `O` is wires, an
output's values times their width, never a count of values: the count is in pairs of a block and an output, which
the wires bound and the distinct outputs do not.

Blocks charged whole. `HECharged` is the count over a set `G` of a partition's blocks, the rest charged whole: with `T`
at least what all the blocks read outside `θ` between them and `S` at least the sum over `G` of the squares of what
each of its blocks reads, under (H) at the most a block of `G` reads,

    N ≤ H + T + ½ √(S O / q) + ½ √(S T / (q b)),

`H` the MACs the blocks outside `G` hold whole. A MAC split between blocks is inside `T`, whichever blocks hold its
gates. At `G` every block, `T = m k` and `S = m k²` it is `HE`'s count. `O` is the whole work's output wires, the
charged blocks' MACs' among them. MACs are disjoint, so where each has at least `γ` gates a block of at most `c`
gates holds at most `c / γ` whole, and `γ N ≤ x c + γ (T + …)` at `x` blocks outside `G`: `HECharged`'s second bound.
It asks (H) once, of every block of `G`. Asked of each block at what that block reads, at its own `q`, the count has
`Q` in the place of `S / q`, at least the sum over `G` of each block's square over its own `q`: `HE_ROWS`, which no
guarantee states.

`beta(k)` is (H)'s `β` for an activation at `k`, the rows `premise.training-cut-hardness.beta` of
`data/premises.toml`, `bound` the bound at a policy `(c, k, n)` on a circuit's own `N`, `O` and `|C|`, and
`bound_charged` the bound over a monitoring period's blocks when some of the words they read are untied, the blocks
that read more than a threshold of them charged whole. `capped` is either one beside the share of the work no block
is credited for, which no count here covers.
"""

import math
import tomllib
from dataclasses import dataclass
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
#: the count with blocks charged whole, in the same area
HE_CHARGED = f"{AREA}.Guarantees.HECharged"
#: `HECharged`'s count with (H) asked of each block of `G` at what that block reads, a `β_x` a block: no guarantee
#: states it
HE_ROWS = "draft:count-hardness-a-block"
#: what `bound_charged` takes beside (H) and (E), neither of them a Lean statement: that a monitoring period's accepted
#: execution phases hold at most its `untied` words between them, fed by at most its `untied_macs` MACs, at one δ (the
#: SPOT's bound, a node's: over the accepted execution phases of one node, however long the period, and over
#: several nodes their sum); and that an untied word is at most `word` fresh bits, in the one block that reads it
UNTIED_WORDS, WORD_BITS = "draft:period-untied-words", "draft:untied-word-bits"
#: (H)'s `β` for an activation by block size, rows `(k, β)` in bits, `k` rising (`premise.training-cut-hardness.beta`)
BETA: tuple[tuple[float, float], ...] = tuple(
    (float(k), float(b)) for k, b in _ROWS["training-cut-hardness"]["beta"]["value"])

#: a bf16 word's wires. A word's own wires are a cut, so no value costs more than its width and no output more than
#: its own: (H) and (E) at a word's width are the most they can say.
BF16 = 16.0
#: an untied output word's bits: the fresh input one is to the block that reads it, at most
WORD = 32.0


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


def macs_on(T: float, S: float, O: float, q: float, b: float) -> float:
    """Lean's `workHEOn`, the right side of `HECharged`'s count without the charged blocks' MACs:
    `T + ½ √(S O / q) + ½ √(S T / (q b))`. It is `macs_at_most(m, k, O, q, b)` at `T = m k` and `S = m k²`."""
    return T + math.sqrt(S * O / q) / 2 + math.sqrt(S * T / (q * b)) / 2


def macs_a_block(k: float, o: float, q: float, b: float, F: float = 1.0) -> float:
    """The most MACs a block of a work with `o = O / N` output wires a MAC: the count at `O = o N`, divided by `m`,
    is `N/m ≤ (k + F α) + B √(N/m)` with `B = F (k/2) √(o / q)`, so `N/m ≤ ((B + √(B² + 4 (k + F α))) / 2)²`. It is
    `N / blocks_at_least(N, o N, …)` for every `N`."""
    y = F * (k / 2) * math.sqrt(o / q)
    return ((y + math.sqrt(y * y + 4 * (k + F * alpha(k, q, b)))) / 2) ** 2


def beta(k: float) -> float | None:
    """(H)'s `β` for an activation at a block of `k` bits (`BETA`): the `β` of the first row whose `k` is at least
    this one, so between two rows the next row's, and `None` past the last row's `k`, where there is none and the
    bound says nothing. A row is the least ratio found over the sets with a cut of at most its `k`, and that least
    only falls as `k` rises: between two rows it is at least the next row's and at most the one before's, and the
    rows say nothing more. The next row's is the largest `β` no set behind the rows refutes anywhere between them."""
    return next((b for row_k, b in BETA if k <= row_k), None)


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


@dataclass(frozen=True)
class Charged:
    """The min-cut bound over a monitoring period's blocks with untied words, the blocks that read more than `w` of
    them charged whole, with what it rests on."""

    #: the period's blocks' gates over the gates of their MACs as a plain circuit, at least this and at least 1; 1
    #: where the bound says nothing
    slowdown: float
    blocks: float
    untied: float
    #: the MACs that feed the untied words, at most, in all: charged whole, whatever they compute
    untied_macs: float
    #: the threshold, fixed before the period: a block that reads more than `w` untied words is charged whole. `inf`:
    #: no threshold, and no block is charged
    w: float
    #: the most untied words one block reads, where its caller states one; else `untied`, all of them
    most: float
    word: float
    #: the blocks charged whole, at most: `⌊untied / (w + 1)⌋`, no more than `blocks`, and none where `⌊most⌋ ≤ w`
    charged: float
    #: the MACs the charged blocks hold whole, at most: each its whole work, `c / mac_gates`
    charged_macs: float
    #: the period's MACs, at most: those its blocks hold, the charged blocks' among them, and `untied_macs`; `inf`
    #: where the bound says nothing
    macs: float
    #: those MACs' gates as a plain circuit, at most, and no more than the blocks' own: the pooled total over the
    #: period, in the gates `c` counts
    gates: float
    #: the statement: `HECharged`, which asks (H) once, or `HE_ROWS`, the count with a `β_x` a block
    statement: str
    #: what it takes that no Lean statement proves: (H), (E), the period's bound on the untied words and an untied
    #: word's bits
    premises: tuple[str, ...]
    k: float
    #: the budget of a block that isn't charged, the most it reads: `k + word · min(w, most)`. It is where (H) is
    #: asked, and with a `β_x` a block the most it is asked at
    budget: float
    #: `T`, what all the blocks read outside `θ` between them, at most: `blocks · k + word · untied`
    total: float
    #: `Q`, the sum over the blocks that aren't charged of the square of what each reads over the `q = β_w β_x` it
    #: is asked (H) at, at most, however many of `charged` are
    squares_by_q: float
    mac_gates: float
    beta_w: float
    #: (H)'s `β` for an activation at `budget`, the least any block is asked it at; `None`: none there, so no bound
    beta_x: float | None
    b: float

    @property
    def reading(self) -> str:
        """Which of the readings the figure is, by where (H) is asked and whether a block may be charged: what a
        caller heads its column with."""
        threshold = "no threshold" if self.w == math.inf else f"with the threshold w = {self.w:.6g}"
        if self.statement == HE_ROWS:
            return f"a β_x a block, {threshold}"
        return "one β_x at the top budget, no block charged" if self.w == math.inf else f"one β_x, {threshold}"

    @property
    def share(self) -> float:
        """`gates` as a share of the period's blocks' gates, at most: one in `slowdown`."""
        return 1 / self.slowdown

    @property
    def charged_of_blocks(self) -> float:
        """The charged blocks as a share of the period's blocks, at most."""
        return self.charged / self.blocks if self.blocks > 0 else 0.0

    @property
    def charged_of_macs(self) -> float:
        """The MACs the charged blocks hold whole as a share of `macs`, the most the period's blocks hold."""
        return self.charged_macs / self.macs if 0 < self.macs < math.inf else 0.0

    @property
    def rests_on(self) -> str:
        """What the figure rests on, as a sentence."""
        threshold = ("no threshold, so no block charged" if self.w == math.inf else
                     f"no block charged, none reading more than the threshold w = {self.w:.6g} untied words"
                     if self.charged == 0 else
                     f"at most {self.charged:.6g} of {self.blocks:.6g} blocks charged whole at the threshold "
                     f"w = {self.w:.6g}, fixed before the period, each holding at most its gates over a MAC's "
                     f"{self.mac_gates:.4g}")
        if self.beta_x is None:
            return (f"no bound with {threshold}: a block that isn't charged reads up to {self.budget:.4g} bits, and "
                    f"(H)'s β for an activation ({HARDNESS}) is stated up to k = {BETA[-1][0]:.4g}")
        if self.statement == HE_ROWS:
            count = (f"the count with (H) asked of each block at what it reads, a β_x a block ({HE_ROWS}), which no "
                     f"guarantee states ({HE_CHARGED} asks (H) once)")
            hardness = (f"at what each block reads, up to k = {self.budget:.4g} bits, β_w = {self.beta_w:.4g} and "
                        f"β_x = {self.beta_x:.4g} bits a value there")
        else:
            count = f"{self.statement}, whose standing the lock gives"
            hardness = (f"once, at k = {self.budget:.4g} bits, the most a block that isn't charged reads, "
                        f"β_w = {self.beta_w:.4g} and β_x = {self.beta_x:.4g} bits a value")
        return (f"{count}, at T = {self.total:.4g} bits and Q = {self.squares_by_q:.4g}, with {threshold}; under "
                f"(H) ({HARDNESS}) {hardness}, and (E) ({EXPORTS}) at "
                f"b = {self.b:.4g} bits an output, conjectures one set of gates refutes, both at the circuit with "
                f"the untied words as inputs, whichever words they are; and two premises no Lean statement gives: "
                f"the period's accepted execution phases hold at most {self.untied:.6g} untied words between them, "
                f"at most {self.most:.6g} of them in one block, fed by at most {self.untied_macs:.6g} MACs, which "
                f"are charged whole ({UNTIED_WORDS}), and an untied word is at most {self.word:.4g} fresh bits in "
                f"the one block that reads it ({WORD_BITS}); the work's structure, the fewest gates of one of its "
                "MACs among it, is its caller's to state; the work no block is credited for is in no term here "
                "(`capped`)")


def bound_charged(c: float, k: float, n: float | None = None, *, blocks: float, untied: float, untied_macs: float,
                  w: float = math.inf, most: float = math.inf, word: float = WORD, N: float, O: float, size: float,
                  mac_gates: float | None = None, beta_w: float = BF16, beta_x: float | None = None,
                  b: float = BF16, rows: bool = False) -> Charged:
    """The bound over a monitoring period at the policy `(c, k, n)` when some of the words its blocks read are untied:
    `blocks` blocks of at most `c` gates in the period, each reading at most `k` fresh input bits and the untied words
    it reads, at most `untied` of them in all, a word at most `word` bits and read by one block. `w` is the threshold,
    a public parameter fixed before the period: a block that reads more than `w` untied words is charged whole, and
    stays in the count at its whole work. By default there is none: no block is charged, and the bound takes no
    parameter beyond the policy's. Where the untied words are as many as the blocks, `w = 0` charges every block and
    the bound is 1, so a figure there states its threshold. `most` is the most untied words one block reads, where
    the caller can state one (a block reads no more untied words than the tied words it reads); by default every
    untied word may be in one block. `N`, `O` and `size` are `bound`'s, the work's MACs, their output wires and its
    gates as a plain circuit; only `O / N` and `size / N` are read, so they may be one block's or the period's.

    The untied words are the period's, over every node in it. SPOT's bound is a node's: the accepted
    execution phases of one node hold at most so many untied words, however long the period, so `untied` and
    `untied_macs` over several nodes are the nodes' sums, and `most` is one node's. A node's bound is a share of one
    execution phase's ticket weight, so it grows with the blocks the node serves in a phase: a figure for nodes
    faster than the measured one scales the measured node's bound by their rate, never only its block count.

    `untied_macs` is the MACs that feed the untied words, at most, in all. The count is taken at the circuit with
    those words as inputs, where no MAC of the work has its output among them, so it covers none of the MACs that
    feed them: they are added whole, as if each were the work's.

    Three readings, each named by its premise (`Charged.reading`), which a caller prints side by side under those
    names, never one as another:

    * one `β_x`, with the threshold `w`: (H) is asked once, of every block that isn't charged, at `k + word · w`.
      `w` is fixed before the period and is a parameter of the figure, never left out of it; a block over it is
      charged at its whole work, which takes `c` and `mac_gates`, the circuit's with every word tied. It is
      `HECharged` at the best `G`;
    * one `β_x` at the top budget, no block charged: no threshold, and nothing is added to the policy's parameters, but
      (H) is asked of every block where the block that reads the most untied words reads. It is `HECharged` at `G`
      every block;
    * a `β_x` a block, no threshold (`rows`): no block is charged, nothing is added, and each block is asked (H) at
      the table's row for what it reads itself, so (H) is taken at every row up to the most a block reads. No
      guarantee states this count (`HE_ROWS`): it is not `HECharged`, which asks (H) once.

    All take the bound on the period's untied words as a premise, and a figure for more nodes than were measured is
    scaled from one node's, which its caller says.

    The count is over the period's blocks, `G` the blocks that read at most `w` untied words:

    * at most `⌊untied / (w + 1)⌋` blocks are charged, and none where `⌊most⌋ ≤ w`, each holding at most
      `c / mac_gates` MACs whole, MACs being disjoint;
    * all the blocks together, the charged ones among them, read `T ≤ blocks · k + word · untied`;
    * a block of `G` that reads `w_u` untied words, a whole number, reads at most `r_u = k + word · w_u`, with
      `w_u ≤ v = min(w, most)` and the `w_u` summing to at most `untied`. `Q` is at least the sum over `G` of
      `r_u² / q_u`, `q_u = β_w β_x` where the block is asked (H). Asked once, `q_u = q` at `k + word · v` and
      `Q = S / q`, with `S ≤ |G| k² + word · E_G (2 k + word · v)` at `E_G = min(untied, |G| v)`. With `rows`,
      `q_u = q(r_u)`, the table's at `r_u`, and `Q` is at most `|G| k² / q(k)` and, for the untied words, the
      lesser of `untied` times the most a word adds and `|G|` times the most a block adds,
      `(k + word · v)² / q(k + word · v) − k² / q(k)`, and never more than asked once. The most a word adds is the
      steepest chord of `r² / q(r)` from `r = k`: between two of `beta`'s rows the chord falls and then rises, so it
      is steepest at the last reads of a row, at the first past it, or at `k + word · v`;
    * `O` is the output wires of every MAC the period's blocks hold, `O / N` a MAC, the charged blocks' among them.

    Which blocks are charged is not the caller's to say, so each term is taken at its most: `H` at every block that
    can be charged, and `Q` at `|G| = blocks`, none charged. The count holds of any partition and any set of its
    blocks; that the partition a period's audit reads, fixed before its coins, has so few blocks over `w` is that
    audit's to give, a join the guarantee that reads this bound owes and nothing here states. The MACs `M'` of the
    count then satisfy
    `M' ≤ H + macs_on(T, Q, (O / N) M', 1, b)`, which is solved for `M'`; the period's MACs are `M = untied_macs + M'`,
    their gates as a plain circuit `M · size / N` (`Charged.gates`, the pooled total, no more than the blocks' own),
    and the slowdown is the blocks' gates over those, `blocks · c / (M · size / N)`, at least 1. With no untied word
    it is `bound`'s slowdown, whatever `w`; with every block charged it is 1; at `w = 0` every block that isn't
    charged reads `k`.

    * `mac_gates`, the fewest gates of one of the work's MACs, the caller's to state of its work and read only
      where a block is charged. Default `size / N`, the plain circuit's gates a MAC: right where that circuit is its
      MACs alone, and a caller whose circuit holds other gates states its own.
    * `beta_w`, `beta_x`, `b`: `bound`'s, with `beta_x` by default `beta` at the most a block that isn't charged
      reads, for every block. With `rows` it is `beta` at what each block reads: a block that reads more is asked
      (H) at a lower `β_x`, and no other block is; `rows` takes no `beta_x`.

    The statement reads two arrays (`HE`'s `F = 1`), and no `n`.
    """
    if rows and beta_x is not None:
        raise ValueError("bound_charged: `rows` asks (H) of each block at `beta`'s row for what it reads, so it takes "
                         "no `beta_x`: leave `beta_x` out, or `rows`")
    mac_gates = size / N if mac_gates is None and N > 0 else mac_gates
    most = min(most, untied)
    over = min(w, most)  # the most untied words a block that isn't charged reads
    budget = k + word * over
    beta_x = beta(budget) if beta_x is None else beta_x  # the least any block is asked (H) at
    # words are whole: a block that reads at most `most` of them reads at most `⌊most⌋`
    under = most <= w or (most < math.inf and math.floor(most) <= w)
    charged = 0 if under else min(blocks, math.floor(untied / (w + 1)))
    total = blocks * k + word * untied
    statement, premises = HE_ROWS if rows else HE_CHARGED, (HARDNESS, EXPORTS, UNTIED_WORDS, WORD_BITS)
    if beta_x is None or N <= 0 or blocks <= 0:
        return Charged(1.0, blocks, untied, untied_macs, w, most, word, charged, math.nan, math.inf, blocks * c,
                       statement, premises, k, budget, total, math.inf, mac_gates or math.nan, beta_w, beta_x, b)
    q = beta_w * beta_x
    by_q = (blocks * k ** 2 + word * min(untied, blocks * over) * (2 * k + word * over)) / q  # asked once: `S / q`
    if rows and over > 0:
        # `(reads, q)` where the chord of `reads² / q(reads)` from `k` may be steepest: `budget`, and at each of
        # `beta`'s rows below it the row's last reads and the first a whole number of words gives past it
        tied, ends = k ** 2 / (beta_w * beta(k)), [(budget, q)]
        for row_k, at in BETA:
            if k <= row_k < budget:
                past = k + word * (math.floor((row_k - k) / word) + 1)
                ends += [(row_k, beta_w * at)] * (row_k > k) + [(past, beta_w * beta(past))] * (past <= budget)
        a_word = max((reads ** 2 / at - tied) * word / (reads - k) for reads, at in ends)
        by_q = min(by_q, blocks * tied + min(untied * a_word, blocks * (budget ** 2 / q - tied)))
    held = charged * c / mac_gates
    # M' = a + d √M', with a = H + T + ½ √(Q T / b) and d = ½ √(Q (O / N))
    a, d = held + macs_on(total, by_q, 0.0, 1.0, b), math.sqrt(by_q * (O / N)) / 2
    macs = untied_macs + ((d + math.sqrt(d * d + 4 * a)) / 2) ** 2
    return Charged(max(1.0, blocks * c * N / (macs * size)), blocks, untied, untied_macs, w, most, word, charged,
                   held, macs, min(blocks * c, macs * size / N), statement, premises, k, budget, total, by_q, mac_gates,
                   beta_w, beta_x, b)


def best_w(c: float, k: float, n: float | None = None, *, blocks: float, untied: float, untied_macs: float,
           most: float = math.inf, word: float = WORD, **work) -> float:
    """The threshold at which `bound_charged` is largest for this many blocks, the least such: a choice made before
    the period, from its block count and the bounds on its untied words. It searches `0`, `⌊min(untied, most)⌋` (no
    block charged, as with no threshold), a grid of ratio 1.02 between them, the last `w` before each of `beta`'s
    rows, and the least `w` at which at most `j` blocks are charged for each `j` below 64."""
    top = math.floor(min(untied, most))
    grid = {0, top} | {round(1.02 ** i) for i in range(int(math.log(max(top, 1), 1.02)) + 1)}
    grid |= {math.floor((row_k - k) / word) for row_k, _ in BETA if row_k > k}
    grid |= {math.floor(untied / j) for j in range(1, 65)}
    return float(max((w for w in sorted(grid) if 0 <= w <= top),
                     key=lambda w: bound_charged(c, k, n, blocks=blocks, untied=untied, untied_macs=untied_macs,
                                                 w=w, most=most, word=word, **work).slowdown))


def capped(slowdown: float, uncredited: float) -> float:
    """A slowdown beside the work no block is credited for: where a share `uncredited` of a block's work is checked by
    nothing, a developer gets that share whole and the rest at one in `slowdown`, so the slowdown is at most
    `1 / (uncredited + 1 / slowdown)`, and no more than `1 / uncredited` whatever `slowdown` is. It is the dark share
    of `theorem.multiple_bound`, which no count of `bound` or `bound_charged` covers: a figure of theirs is quoted
    beside this one, at a stated share, never alone."""
    if slowdown == math.inf:
        return max(1.0, 1 / uncredited) if uncredited > 0 else math.inf
    return max(1.0, slowdown / (1 + uncredited * slowdown))
