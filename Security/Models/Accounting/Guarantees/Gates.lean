import Mathlib.Data.Int.ModEq
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
* `Renormalization`: the step the encode bound needs and no counting argument gives, stated and unproved: returning a
  square's limbs to the window takes `n/8 − d` budgeted gates. `EncodeChainHardness` is its composition over the chain.
* `EncodeBudget`: the connection. For a declared circuit `W` that satisfies the stacking property `P` and is in the
  restricted class, the words of fresh encodings `W`'s values hold on more than `ε` of the draws number at most
  `(budget)/E + (run-input bits)/(n − 1)`, under `EncodeChainHardness` at `E`: the budget is the caps times the
  audit's certified reads, summed over the windows, and the run inputs are `R_in` a window.
* `EncodeCost`: under `EncodeChainHardness` at `E`, the encode cost (`encodeCost`) of `q` words is at least `q · E`:
  the carrier lemma's `cost`, defined.
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

/-- **Re-normalization** (the 19:15Z ruling's lemma, unproved): a product's limbs come back to the window only through
budgeted gates, about one a limb. A restricted circuit with no run input whose registered and coin inputs are bytes of
`a` or fixed, and that on more than `ε` of the `a ∈ [0, p)` holds its declared intervals (so is in its window) and
outputs (`out l`, `l < n/8`) the canonical bytes of `a² mod p`, has at least `n/8 − d` budgeted gates. The s8 family
forms the column sums of `a²` for free; what it can't do is carry, and `SquaringCarry` proves only bits of that (63 at
p1024), not gates. -/
def Renormalization (F : Field) (f w d εnum εden : ℕ) : Prop :=
  ∀ {ι : Type} (C : Dag ι) (x : ℕ → ι → ℤ) (out : ℕ → ℕ) (good : Finset ℕ),
    Restricted f w C → runBits C = 0 → good ⊆ Finset.range F.p → MoreThan (Finset.range F.p) good εnum εden →
    (∀ i, FreeInput C i → (∃ l, ∀ a ∈ good, x a i = byte a l) ∨ ∀ a ∈ good, ∀ a' ∈ good, x a i = x a' i) →
    (∀ a ∈ good, DeclaredHold C (x a) ∧ ∀ l < F.bytes, val C (x a) (out l) = byte (a ^ 2 % F.p) l) →
    F.bytes - d ≤ budgeted C

/-- **The encode budget: what the stacking property bounds.** For a declared circuit `W` in the restricted class
(`Restricted f w`) whose census satisfies the stacking property `P` (`Property caps A W`), whose registered and coin
inputs are the codec's (`CodecInputs`), and whose values hold, byte for byte less parity, the encodings of the `q`
distinct words `words` on more than `ε` of the fresh draws, its inputs and budgeted gates there in their declared
intervals (its window is then `ProvenWindow`'s, never a hypothesis): under `EncodeChainHardness F k f w E ε`,

  `q · E · (n − 1) ≤ (c · Σ_w reads_w) · (n − 1) + 8 · R_in · windows · E`,

with `c` the caps' budgeted units per certified byte read: `q ≤ c · reads / E + 8 · R_in · windows / (n − 1)`. -/
def EncodeBudget : Prop :=
  ∀ (F : Field) (k f w E εnum εden : ℕ) {ι : Type} (W : Dag ι) (caps : Caps) (A : Audit) (m : ℕ → ℕ)
    (x : Draw k → ι → ℤ) (words : List ℕ) (out : ℕ → ℕ → ℕ),
    EncodeChainHardness F k f w E εnum εden →
    Property caps A W → Restricted f w W → CodecInputs F W (draws F k) m x → words.Nodup →
    (∀ i ∈ words, i < F.p) → ComputesWords F k W x m words out εnum εden →
    words.length * E * (F.n - 1) ≤
      (caps.total * ∑ v ∈ Finset.range A.windows, A.reads v) * (F.n - 1) + 8 * caps.runBytes * A.windows * E

/-- **The encode cost, bounded.** Under `EncodeChainHardness F k f w E ε`, with `n ≥ 2`, every `q` distinct words that
some restricted circuit with no run input computes on more than `ε` of the draws cost at least `q · E` budgeted
gates: `q · E ≤ encodeCost F k f w ε q`. -/
def EncodeCost : Prop :=
  ∀ (F : Field) (k f w E εnum εden q : ℕ),
    EncodeChainHardness F k f w E εnum εden → 2 ≤ F.n → (encodeCosts F k f w εnum εden q).Nonempty →
    q * E ≤ encodeCost F k f w εnum εden q

end Accounting.Guarantees.Gates
