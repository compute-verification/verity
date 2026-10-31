import Definitions.Nci.Capability
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Basic.ENNReal.Real
import Mathlib.Basic.Real.ENatENNReal

/-!
# Local bounds: information premises

What a block can't cheaply hold. Each premise prices a selection of an item family (`Nci.Capability.Items`); with a
reconstruction bound it gives each block a budget (`Nci.Guarantees.HeldBudget`, `StatHeldBudget`, `RecoveredBudget`).
The two measures make different scientific claims:

* `DescriptionHard` (computational): the values are **expensive to recompute**. Time-bounded description length: a
  program that rebuilds them by replaying training counts only if it finishes within the decoder's budget. It holds even
  when training is a deterministic function of registered information, as long as recomputing it is slow.
* `MinEntropyHard` (statistical): the values are **unpredictable**. No value of the selection is likely under the run's
  randomness (data order, initialization, sampling). It needs no time budget and is tested by retraining with fresh
  randomness, but it fails when training is a deterministic function of registered information.

`SourceHard` with `Recovers` derives a bound on a block's outputs from a hard source they reveal, rather than assuming
one. Each premise's meaning is its item family's: updated slots (update hardness), operand scalars (activation hardness),
or both in one budget (`Items.sum`, joint hardness).
-/

namespace Nci.Assumptions

open Verity Finset Nci.Capability
open scoped ENNReal

/-- NEW PREMISE. **Description hardness** ([DESCRIPTION-HARD]) of the item family `X` with deficit `Λ`, within `t` steps
of the decoder `M` given the registered information `r`: in every honest run of the registered circuit, every selection
of `X`'s items, one eligible cell each, has local description length at least its price minus `Λ`:
`LDL_{t,r}(x_U) ≥ Σ_{i ∈ U} price_i − Λ`.

It covers every selection a block can make, not a typical tensor or a random subset; it charges an item once however
often a block reuses it; and it is time-bounded. Its meaning is its item family's:

* **updated slots** (update hardness): fails, or holds only at lower prices, where a block can replay the training
  prefix within `t`, where the update has a small effective representation (low rank, a short seed), or where the chosen
  slots are predictable from the registration;
* **operand occurrences** (activation hardness): covers operands a block produces itself from what it holds, so a block
  that imports updated weights and cheaply generates many operands refutes it; a code rate for one tensor is not such a
  bound, and the exact-decoding experiments refute several uniform rates;
* **both together** (`Items.sum`, joint hardness): stronger than the two separately, since the same bits can describe
  both families; it is what an additive budget rests on.

Each decoder found gives an upper bound and can refute a price; failing to find a shorter one is evidence, not proof. -/
def DescriptionHard (R : Committed) {ι : Type} (X : Items R.d.D ι) {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ)
    (Λ : ℝ) : Prop :=
  ∀ a, R.Honest a → ∀ (U : Finset ι) (σ : ι → Finset (Fin R.d.D.N)), Selects X U σ →
    ENNReal.ofReal (∑ i ∈ U, X.price i - Λ) ≤ (LDL M t r (selectionBits (R.d.D.eval a) U σ) : ℝ≥0∞)

/-- NEW PREMISE. **Min-entropy hardness** ([MIN-ENTROPY-HARD]) of the item family `X` with deficit `Λ`, under the
distribution `μ` of the registered circuit's inputs (the run's randomness: data order, initialization, sampling): every
value of every selection has probability at most `2^(Λ − Σ_{i ∈ U} price_i)`.

It says the values are unpredictable, not that they are slow to compute: a block holding a selection must have been
handed about its price in fresh bits, whatever it computes. Test it by retraining from the same registration with fresh
randomness: a weight whose trained value varies across runs by `s` quantization steps carries about `log₂ s` bits. It
fails when training is a deterministic function of registered information (public data in a registered order, registered
seeds), however slow that function is; `DescriptionHard` covers that case. It concerns exact values; a method that
learns with approximate ones is a different method, which coverage must account for. -/
def MinEntropyHard (R : Committed) {ι : Type} (X : Items R.d.D ι) (μ : PMF R.d.D.Transcript) (Λ : ℝ) : Prop :=
  ∀ (U : Finset ι) (σ : ι → Finset (Fin R.d.D.N)) (v : List Bool), Selects X U σ →
    μ.toOuterMeasure {a | selectionBits (R.d.D.eval a) U σ = v} ≤
      ENNReal.ofReal ((2 : ℝ) ^ (Λ - ∑ i ∈ U, X.price i))

/-- NEW PREMISE. **Source hardness** ([SOURCE-HARD]) of the source `z` (a string the run determines, such as the update's
canonical encoding) of length `ℓz`, with deficiency `δ`, within `τ` steps: in every honest run,
`LDL_{τ,r}(z) ≥ ℓz − δ`. The deficiency is paid in every block's bound, so a `δ` far above a block's budget leaves the
recovery route empty; a compressed representation of a structured update does not by itself give a source with small
deficiency. -/
def SourceHard (R : Committed) {ρ : Type} (M : Decoder ρ) (r : ρ) (τ : ℕ) (z : R.d.D.Transcript → List Bool)
    (ℓz δ : ℝ) : Prop :=
  ∀ a, R.Honest a → ENNReal.ofReal (ℓz - δ) ≤ (LDL M τ r (z (R.d.D.eval a)) : ℝ≥0∞)

/-- NEW PREMISE. **The source is recoverable from each block** ([SOURCE-RECOVERABLE]) at `βv` bits per eligible output,
up to `amb` bits of ambiguity: the decoder, on proof block `u`'s Lemma 1 program (`blockProgram`, whose evaluator `e`
also performs the recovery) followed by a complement of at most `ℓz − βv · J u + amb` bits, outputs the source within `τ`
steps. `J u` counts block `u`'s eligible outputs; `amb` is the index that picks the source among at most `2^amb`
candidates the outputs and complement leave.

It is a capability, like Lemma 1's `Evaluates`, and the structural heart of the recovery route: correlated outputs need
not shorten the complement one coordinate each (for an exact linear map the rank counts, not the number of scalars), a
bound on the candidates is not a decoder within `τ`, and inputs the recovery needs must be among what `SourceHard`'s
decoder is given. -/
def Recovers (R : Committed) {ρ : Type} (M : Decoder ρ) (r : ρ) (τ : ℕ) (e : List Bool)
    (z : R.d.D.Transcript → List Bool) (ℓz βv amb : ℝ) (J : Fin R.B → ℕ) : Prop :=
  ∀ a, R.Honest a → ∀ u, ∃ s : List Bool, (s.length : ℝ) ≤ ℓz - βv * J u + amb ∧
    M.run r (blockProgram R e (R.d.D.eval a) u ++ s) τ = some (z (R.d.D.eval a))

end Nci.Assumptions
