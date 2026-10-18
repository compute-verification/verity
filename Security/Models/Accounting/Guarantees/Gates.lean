import Mathlib.Data.Int.ModEq
import Mathlib.Data.Nat.Prime.Defs
import Definitions.Accounting.Gates
import Models.Accounting.Assumptions.EncodeChain

/-!
# The circuit model's window, and its lower bounds: the fooling bound, the carry bound and the encode budget (trusted)

The statements over the circuit model (`Definitions/Accounting/Gates`) that the stacking property's encode bound rests
on, from the most general to the one 6a's carrier lemma takes as its `q`:

* `ProvenWindow`: the property's interval analysis is sound. A circuit whose window is proven (`Proven`) is in its
  window on every input on which its inputs and budgeted gates hold their declared intervals (`DeclaredHold`): the
  window is a fact of the operand bounds, not a run-time hypothesis.
* `ClampOrWrap`, its corollary: in the window, an s8-family step's clamp and wrap give the same values, so the census
  keys the s8·s8 (`.satfinite`) and u8·s8 (wrapped) steps to one class.
* `Fooling`: a 2-adic fooling bound. Free gates (affine maps and s8-family steps in their window) see only the low `j`
  bits of their operands when their outputs' low `j` bits are asked for. So if two inputs agree mod `2^j` in every
  registered and coin input and every charged gate takes one value on both, every gate does mod `2^j`, and a set of
  inputs that pairwise differ mod `2^j` in some output needs that many joint values of the charged gates:
  `|F| ≤ volume C`.
* `SquaringCarry`: the carry bound (workstream G's §(7), made precise). A circuit that computes the `n` canonical bytes
  of `a² mod p`, in its window, on the members `G` of the sparse fooling family (`sparse`), has `|G| ≤ volume C`. On
  the whole family (`2^s` members) its charged gates have at least `s` bits: `s = 63` at p1024 (`256^127 ≤ p`). It is a
  bound in bits, not in gates: a budgeted gate of `b` declared bits carries `b` of them. "Computes" is "on every member
  of `G`": on the fraction `ε` of the family, `G` has `ε · 2^s` members.
* `Renormalization`: the step the encode bound needs and no counting argument gives, stated and unproved (the red
  team's Lemma R): returning a square's limbs to byte format takes `L − c − d(ε)` budgeted units, `d(ε) =
  d₀ + log₂(1/ε)/8`, at every `ε` above a floor `2^−t` (claimed at p1024 with `d₀ = 4`, `t = 64`; false with no
  floor). `EncodeChainHardness` is its composition over the chain.
* `RhoPairs`: the parity-free target's symmetry. `ρ(p − x) = p − ρ(x)`, and `ρ(x)`'s parity, the bit the target
  drops, is the Legendre symbol of `x`.
* `EncodeBudget`: the connection. For a declared circuit `W` that satisfies the stacking property `P` and is in the
  restricted class, the words of fresh encodings `W`'s values hold on more than `ε` of the draws number at most
  `(budget)/E + (run-input bits)/(n − 1)`, under `EncodeChainHardness` at `E`: the budget is the caps times the
  audit's certified reads, summed over the windows, and the run inputs are `R_in` a window.
* `EncodeCost`: under `EncodeChainHardness` at `E`, the encode cost (`encodeCost`) of `q` words is at least `q · E`
  (`⊤` when no circuit computes them): the cost 6a's `encodeBitsCost` reads.
-/

namespace Accounting.Guarantees.Gates

open Accounting.Gates Accounting.Assumptions

/-- **The window, proven, holds.** A circuit whose window is proven (`Proven C`: property/v1's interval analysis) is in
its window on every input `x` on which its inputs and budgeted gates hold their declared intervals. -/
def ProvenWindow : Prop :=
  ∀ {ι : Type} (C : Dag ι) (x : ι → ℤ), Proven C → DeclaredHold C x → InWindow C x

/-- **Clamp or wrap, in the window.** For a circuit `C` whose window is proven, on an input on which its inputs and
budgeted gates hold their declared intervals: a circuit `C'` gate for gate like `C` but for its s8-family steps' clamp
or wrap (`Gate.Reclamped`) has `C`'s values. -/
def ClampOrWrap : Prop :=
  ∀ {ι : Type} (C C' : Dag ι) (x : ι → ℤ), Proven C → DeclaredHold C x → C'.length = C.length →
    (∀ i (h : i < C.length) (h' : i < C'.length), C[i].Reclamped C'[i]) → values C' x = values C x

/-- **The 2-adic fooling bound.** For a circuit `C`, a set `F` of inputs on each of which `C` is in its window, that
agree pairwise mod `2^j` in every registered and coin input, and that pairwise differ mod `2^j` in some output gate of
`out`: `|F| ≤ volume C`, the joint values the run inputs' and budgeted gates' declared intervals allow. -/
def Fooling : Prop :=
  ∀ {ι κ : Type} (C : Dag ι) (F : Finset κ) (x : κ → ι → ℤ) (j : ℕ) (out : List ℕ),
    (∀ s ∈ F, InWindow C (x s)) →
    (∀ s ∈ F, ∀ s' ∈ F, ∀ i, FreeInput C i → x s i ≡ x s' i [ZMOD 2 ^ j]) →
    (∀ s ∈ F, ∀ s' ∈ F, s ≠ s' → ∃ o ∈ out, ¬ val C (x s) o ≡ val C (x s') o [ZMOD 2 ^ j]) →
    F.card ≤ volume C

/-- **The carry bound.** For `s ≤ 255` with `256^(2s+1) ≤ p` and `2s ≤ n`: a circuit `C` whose registered and coin
inputs read the bytes of `a` (`slot i` the byte input `i` reads), and that on every `a = sparse S` with
`S ∈ G ⊆ 2^{0..s−1}` is in its window and outputs (`out l`, `l < n`) the bytes of `a² mod p`, has `|G| ≤ volume C`. -/
def SquaringCarry : Prop :=
  ∀ {ι : Type} (C : Dag ι) (p n s : ℕ) (G : Finset (Finset ℕ)) (x : Finset ℕ → ι → ℤ) (slot : ι → ℕ)
    (out : ℕ → ℕ),
    s ≤ 255 → 256 ^ (2 * s + 1) ≤ p → 2 * s ≤ n → G ⊆ (Finset.range s).powerset →
    (∀ S ∈ G, ∀ i, FreeInput C i → x S i = byte (sparse S) (slot i)) →
    (∀ S ∈ G, InWindow C (x S) ∧ ∀ l < n, val C (x S) (out l) = byte (sparse S ^ 2 % p) l) →
    G.card ≤ volume C

/-- **Re-normalization** (the red team's Lemma R, note:20261017T1859Z-report-property-redteam R3.3 and R4.1;
unproved): a square's limbs come back to byte format only through budgeted gates, about one a limb. Take a circuit
whose window is proven, with no run input, whose registered inputs are fixed and whose coin inputs are bytes of `a` or
fixed (so units count what reads `a`), that on more than `ε ≥ 2^−t` of the `a ∈ [0, p)` holds its declared intervals
(so is in its window) and outputs `L + c` values (`out l`, `L = n/8`), each within a byte format (`fmt l`), whose
base-256 sum is `a²` mod `p`. Then `ε · 2^(8L) ≤ 2^(8·(budgeted C + d₀ + c))`, that is `budgeted C ≥ L − c − d(ε)`
with `d(ε) = d₀ + log₂(1/ε)/8`. The outputs are any byte-format representation (signed digits, extra limbs,
unreduced), since the chain's next squaring reads any; for one squaring no cap on operand bits is needed. The s8
family forms the column sums of `a²` for free; what it can't do is carry, and `SquaringCarry` proves only bits of that
(63 at p1024), not units. The claimed instance is p1024 with `d₀ = 4` and `t = 64`.

**`d₀` is not absolute, so the floor.** A circuit with no budgeted gate (`a`'s bytes as coin inputs declared
`[0, D_l]`, `a²`'s column sums as s8-family steps, folded mod `p`) is exact on every `a` whose column sums don't carry,
and such families grow with `L` faster than any fixed `d₀` covers: `2^33.30` members at `p = 256^27 − 377` (`d₀ ≥
4.16`), `2^101.85` at p1024 (`d₀ ≥ 12.73`; the red team's, R4.1), and, sampled, at least `2^159.4` at p1024 (64
digits from a geometric law; `d₀ ≥ 19.9`), which refutes the slope `1/7` too (`d = 4 + log₂(1/ε)/7` allows at most
`2^156`). Every one sits at `ε ≤ 2^−182`, and none can reach the floor at p1024: a column holds `a_l²`, so a
non-carrying `a` has digits below 16 and `ε ≤ 16^L/p ≈ 2^−512`. Above the floor every family found is within `d₀ = 4`:
the exhaustive toy fields (implied `d₀ ≤ 2.18`), the non-carrying sets of the byte fields `L ≤ 10`, counted exactly
(`≤ 2.48`), and at p1024 the red team's carry-guessing frontier (modeled, 10 to 15 bits a dropped carry: `d = 1 … 5`
at `ε = 2^−9.8 … 2^−69.5`, against the formula's `5.2 … 12.7`). `t = 64` keeps the claim inside the range that
frontier reaches; the bound reads `ε = 2^−40` (the headline) and `2^−20` (the property's carrier threshold). -/
def Renormalization (F : Field) (d₀ c t εnum εden : ℕ) : Prop :=
  ∀ {ι : Type} (C : Dag ι) (x : ℕ → ι → ℤ) (out : ℕ → ℕ) (fmt : ℕ → Int8Fmt) (good : Finset ℕ),
    εden ≤ 2 ^ t * εnum →
    Proven C → runBits C = 0 → good ⊆ Finset.range F.p → MoreThan (Finset.range F.p) good εnum εden →
    (∀ i, (∃ g ∈ C, g.op = .input .registered i) → ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ i, (∃ g ∈ C, g.op = .input .coin i) →
      (∃ l, ∀ a ∈ good, x a i = byte a l) ∨ ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ a ∈ good, DeclaredHold C (x a) ∧
      (∀ l < F.bytes + c, (fmt l).lo ≤ val C (x a) (out l) ∧ val C (x a) (out l) ≤ (fmt l).hi) ∧
      ∑ l ∈ Finset.range (F.bytes + c), 256 ^ l * val C (x a) (out l) ≡ (a : ℤ) ^ 2 [ZMOD F.p]) →
    εnum * 2 ^ (8 * F.bytes) ≤ εden * 2 ^ (8 * (budgeted C + d₀ + c))

/-- **`ρ` on pairs.** At a prime `p ≡ 3 mod 4` and `0 < x < p`: `ρ(p − x) = p − ρ(x)`, and `ρ(x)` is odd exactly when
`x` is not a square mod `p`. So `ρ` maps the pair `{x, p − x}` onto the pair `{s, p − s}` of `±x`'s roots, taking the
member the Legendre symbol of `x` names: the ideal model of a round is a random function on such pairs with that sign
rule, not a random permutation, and the bit `lessParity` drops is the Legendre symbol of the last round's input. -/
def RhoPairs : Prop :=
  ∀ p x : ℕ, p.Prime → p % 4 = 3 → 0 < x → x < p →
    rho p (p - x) = p - rho p x ∧ (rho p x % 2 = 1 ↔ ¬ ∃ y, y ^ 2 % p = x)

/-- **The encode budget: what the stacking property bounds.** For a declared circuit `W` in the restricted class
(`Restricted B`) whose census satisfies the stacking property `P` (`Property caps A W`), whose registered and coin
inputs are the codec's (`CodecInputs`), and whose values hold, byte for byte less parity, the encodings of the `q`
distinct words `words` on more than `ε` of the fresh draws, its inputs and budgeted gates there in their declared
intervals (its window is then `ProvenWindow`'s, never a hypothesis): under `EncodeChainHardness F k B E ε`,

  `q · E · (n − 1) ≤ (c · Σ_w reads_w) · (n − 1) + 8 · R_in · windows · E`,

with `c` the caps' budgeted units per certified byte read (property/v1's units: gathers by fresh index bits):
`q ≤ c · reads / E + 8 · R_in · windows / (n − 1)`. -/
def EncodeBudget : Prop :=
  ∀ (F : Field) (k B E εnum εden : ℕ) {ι : Type} (W : Dag ι) (caps : Caps) (A : Audit) (m : ℕ → ℕ)
    (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    EncodeChainHardness F k B E εnum εden →
    Property caps A W → Restricted B W → CodecInputs F W (draws F k) m x → words.Nodup →
    (∀ i ∈ words, i < F.p) → ComputesWords F k W x m words out εnum εden →
    words.length * E * (F.n - 1) ≤
      (caps.total * ∑ v ∈ Finset.range A.windows, A.reads v) * (F.n - 1) + 8 * caps.runBytes * A.windows * E

/-- **The encode cost, bounded.** Under `EncodeChainHardness F k B E ε`, with `n ≥ 2`, `q` distinct words cost at
least `q · E` budgeted units: `q · E ≤ encodeCost F k B ε q`, which is `⊤` when no restricted circuit with no run input
computes them on more than `ε` of the draws. -/
def EncodeCost : Prop :=
  ∀ (F : Field) (k B E εnum εden q : ℕ),
    EncodeChainHardness F k B E εnum εden → 2 ≤ F.n → ((q * E : ℕ) : ℕ∞) ≤ encodeCost F k B εnum εden q

end Accounting.Guarantees.Gates
