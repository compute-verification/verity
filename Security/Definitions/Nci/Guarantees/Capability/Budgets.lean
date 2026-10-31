import Definitions.Nci.Assumptions.Capability.Information
import Definitions.Core.Guarantees.Formalism

/-!
# Local bounds: budgets, from information

A block's view is short: `k` fresh bits, and an index among `B` blocks. Each statement here combines that with an
information premise into a **budget** on what every block holds, the form the structural caps consume
(`Nci.Capability.HeldPriceAtMost`):

* `HeldBudget`: computational, through Lemma 1 (`Verity.Guarantees.BlockDescription`): `k + ⌈log₂ B⌉ + a_blk + Λ`.
* `StatHeldBudget`: statistical, by counting views: `k + Λ`, with no decoder and no time budget.
* `RecoveredBudget`: from a recoverable hard source, a budget on a block's eligible outputs.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability
open scoped ENNReal

/-- **The held budget** (Lemma 1 with description hardness). In a `(c, k)`-partition where, for each selection of
`X`'s items held per block, some evaluator `e` of at most `aBlk` bits extracts the values of their cells from the proof
blocks within `t`, description hardness at deficit `Λ` bounds what every proof block holds:
`Σ_{i ∈ U} price_i ≤ k + ⌈log₂ B⌉ + aBlk + Λ`, whether the block loads the values, decompresses them or computes them.
The evaluator is chosen per selection: one evaluator for every selection would have to output the empty selection's
empty string on every program, so no block could hold an item with a nonempty cell. -/
def HeldBudget : Prop :=
  ∀ (R : Committed) (c k : ℕ), R.d.𝓘.IsCK R.d c k →
    ∀ {ι : Type} (X : Items R.d.D ι) (ρ : Type) (M : Decoder ρ) (r : ρ) (t aBlk : ℕ) (Λ : ℝ),
      (∀ (U : Fin R.d.n → Finset ι) (σ : ι → Finset (Fin R.d.D.N)),
        (∀ u, Selects X (U u) σ ∧ ∀ i ∈ U u, σ i ⊆ holds R.d.D (R.d.𝓘.gates u)) →
        ∃ e : List Bool, e.length ≤ aBlk ∧ Evaluates M r t R e fun u T => selectionBits T (U u) σ) →
      Assumptions.DescriptionHard R X M r t Λ →
      HeldPriceAtMost R X ((k + Nat.clog 2 R.d.n + aBlk : ℕ) + Λ)

/-- **The statistical held budget** (min-entropy hardness). In a `(c, k)`-partition, under a distribution of honest runs,
a proof block's held values are a function of its at most `k` unregistered input bits, so some value of any held
selection has probability at least `2^-k`; min-entropy hardness at deficit `Λ` then bounds what every proof block holds:
`Σ_{i ∈ U} price_i ≤ k + Λ`. No decoder, no time budget and no block index enter. -/
def StatHeldBudget : Prop :=
  ∀ (R : Committed) (c k : ℕ), R.d.𝓘.IsCK R.d c k →
    ∀ {ι : Type} (X : Items R.d.D ι) (μ : PMF R.d.D.Transcript) (Λ : ℝ),
      (∀ a ∈ μ.support, R.Honest a) → Assumptions.MinEntropyHard R X μ Λ →
      HeldPriceAtMost R X ((k : ℝ) + Λ)

/-- **The recovered budget** (Lemma 1 with a recoverable hard source). In a `(c, k)`-partition with an evaluator `e` of at
most `aBlk` bits, source hardness at deficiency `δ` and recovery at `βv` bits per eligible output with `amb` bits of
ambiguity give, for every proof block, `βv · J u ≤ k + ⌈log₂ B⌉ + aBlk + δ + amb`. When the ambiguity grows with the
outputs, `amb ≤ ε · J u` with `ε < βv`, this is `(βv − ε) · J u ≤ k' + δ`: ambiguity lowers the information per output. -/
def RecoveredBudget : Prop :=
  ∀ (R : Committed) (c k : ℕ), R.d.𝓘.IsCK R.d c k →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (τ aBlk : ℕ) (e : List Bool) (z : R.d.D.Transcript → List Bool)
      (ℓz δ βv amb : ℝ) (J : Fin R.d.n → ℕ), e.length ≤ aBlk →
      Assumptions.SourceHard R M r τ z ℓz δ → Assumptions.Recovers R M r τ e z ℓz βv amb J →
      ∀ u, βv * J u ≤ (k + Nat.clog 2 R.d.n + aBlk : ℕ) + δ + amb

end Nci.Guarantees
