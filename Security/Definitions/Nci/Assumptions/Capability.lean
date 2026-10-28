import Definitions.Nci.Capability
import Mathlib.Basic.ENNReal.Real
import Mathlib.Basic.Real.ENatENNReal

/-!
# Local bounds: the premises

Why splitting declared computation into small inference blocks should slow the acquisition of a new capability. A block
can import information or reconstruct it; an argument needs three things, each with its own premises:

* **information**: which jointly selected values are expensive to reconstruct. One schema, `DescriptionHard`, over an
  item family: updated parameter slots (update hardness), operand occurrences (activation hardness), or both in one
  budget (`Nci.Capability.Items.sum`, joint hardness). `SourceHard` with `Recovers` is a way to prove an information
  bound on a block's outputs, not a fourth kind of hardness.
* **structure**: how much counted computation the held values support. Definitions checkable on a named circuit
  (`Nci.Capability.Incidence`, `Nci.Capability.Upstream`, `Nci.Capability.OverlapWeights`), no premise.
* **coverage**: why every sufficiently cheap successful method contains enough of that computation:
  `CoversEfficientMethods`, stated over tasks and runs in `Definitions.Accounting.Assumptions.Coverage`.

The rates and deficits belong to the decoder budget, the registration, the arithmetic and the computation evaluated;
none is a universal bits-per-value constant. The smallest sufficient premise comes first: update hardness with a
dependency certificate; an activation price only where it survives joint reconstruction attacks; a joint price only
where the additive cost is justified; recovery to derive rates rather than assume them. Protocol soundness and the device
model are outside these: they establish where the block restrictions apply, not that learning is hard.
-/

namespace Nci.Assumptions

open Verity Finset Nci.Capability
open scoped ENNReal

/-- NEW PREMISE. **Description hardness** ([DESCRIPTION-HARD]) of the item family `X` with deficit `Λ`, within `t` steps
of the decoder `M` given the registered information `r`: in every honest run of the registered circuit, every selection
of `X`'s items, one eligible cell each, has local description length at least its price minus `Λ`:
`LDL_{t,r}(x_U) ≥ Σ_{i ∈ U} price_i − Λ`.

It covers every selection a block can make, not a typical tensor or a random subset; it charges an item once however
often a block reuses it; and it is time-bounded, so a program that rebuilds the values by replaying training counts only
if it finishes within `t`. Its meaning is its item family's:

* **updated slots** (update hardness): fails, or holds only at lower prices, where a block can replay the training
  prefix within `t`, where the update has a small effective representation (low rank, a short seed), or where the chosen
  slots are predictable from the registration;
* **operand occurrences** (activation hardness): covers operands a block produces itself from what it holds, so a block
  that imports updated weights and cheaply generates many operands refutes it; a code rate for one tensor is not such a
  bound, and the exact-decoding experiments refute several uniform rates;
* **both together** (`Items.sum`, joint hardness): stronger than the two separately, since the same bits can describe
  both families; it is what an additive budget rests on.

A deficit that cancels every useful price leaves it empty. Each decoder found gives an upper bound and can refute a
price; failing to find a shorter one is evidence, not proof. -/
def DescriptionHard (R : Registration) {ι : Type} (X : Items R.d.D ι) {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ)
    (Λ : ℝ) : Prop :=
  ∀ a, R.Honest a → ∀ (U : Finset ι) (σ : ι → Finset (Fin R.d.D.N)), Selects X U σ →
    ENNReal.ofReal (∑ i ∈ U, X.price i - Λ) ≤
      (LDL M t r (R.d.D.bits (R.d.D.eval a) ((U.biUnion σ).sort (· ≤ ·))) : ℝ≥0∞)

/-- NEW PREMISE. **Source hardness** ([SOURCE-HARD]) of the source `z` (a string the run determines, such as the update's
canonical encoding) of length `ℓz`, with deficiency `δ`, within `τ` steps: in every honest run,
`LDL_{τ,r}(z) ≥ ℓz − δ`. The deficiency is paid in every block's bound, so a `δ` far above a block's budget leaves the
recovery route empty; a compressed representation of a structured update does not by itself give a source with small
deficiency. -/
def SourceHard (R : Registration) {ρ : Type} (M : Decoder ρ) (r : ρ) (τ : ℕ) (z : R.d.D.Transcript → List Bool)
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
def Recovers (R : Registration) {ρ : Type} (M : Decoder ρ) (r : ρ) (τ : ℕ) (e : List Bool)
    (z : R.d.D.Transcript → List Bool) (ℓz βv amb : ℝ) (J : Fin R.B → ℕ) : Prop :=
  ∀ a, R.Honest a → ∀ u, ∃ s : List Bool, (s.length : ℝ) ≤ ℓz - βv * J u + amb ∧
    M.run r (blockProgram R e (R.d.D.eval a) u ++ s) τ = some (z (R.d.D.eval a))

end Nci.Assumptions
