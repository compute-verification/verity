import Definitions.Nci.Formalism

/-!
# The training formalism v2, §4: the assumptions A1 and A2

* **A1, cut cost** (`CutCost`): every inference block `I` reads beyond `θ` at least the update bits its work reads and,
  for every token, the token's computed state where its work on it begins: `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`.
  `e_T(I)` is where `I`'s work on token `T` begins, `a_T(e)` the bits of `T`'s computed state there, the earlier tokens'
  keys and values that `I`'s work reads and doesn't compute among them, and `u_I` the update bits `I`'s work reads,
  counted once. Their values are set only by checked witness blocks, sets of the run's own gates whose gates and reads a
  checker counts, and A1 says no block beats the best witness: it is refuted by an inference block that reads less. A
  theorem reads upper bounds from it: a shorter update encoding lowers `u`, an activation code lowers `a`. A block that
  computes a token from its inputs enters it there, its state the token's data and the earlier tokens' keys and values
  it reads, and A1 prices it as any other block: its update bits and that state.
* **A2, efficiency floor** (`EfficiencyFloor`): no run of the methods considered whose update stays within `D` bits
  reaches `ℓ*` for less than `p · H(ℓ*)`. It is empirical. It is stated over a class of runs, not over every circuit: a
  circuit that outputs a run's `θ'` as constants reaches the same loss at about `|θ'|` gates, so over every circuit the
  floor fails at any `p · H(ℓ*)` above that (`Definitions.Nci.Formalism.Runs`).
-/

namespace Nci.Formalism.Assumptions

open Finset
open Verity (Declaration)
open Nci.Formalism
open scoped NNReal ENNReal

/-- `training-cut-cost` (conjectured; its values are the best checked witness blocks', and it is refuted by an
inference block that reads less). **A1, cut cost**: every inference block `I` of `d` reads beyond `θ` at least its
update bits `u_I` and, for every token `T`, the bits `a_T(e_T(I))` of `T`'s computed state where `I`'s work on it
begins: `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. -/
def CutCost (d : Declaration) {L : Type} [Fintype L] (c k : ℕ) {E : Type} (e : Finset (Fin d.D.N) → L → E)
    (a : L → E → ℝ≥0) (u : Finset (Fin d.D.N) → ℝ≥0) : Prop :=
  ∀ I, InferenceBlock d c k I → u I + ∑ T, a T (e I T) ≤ (d.unregistered I).card

/-- `training-efficiency-floor` (empirical; falsified by a measured run). **A2, efficiency floor**: no run of `methods`
with a valid annotation whose update stays within `D` bits (its `Δ`, decoded by `M` within `c` steps) reaches `ℓ*` for
less than `p · H(ℓ*)`, `H` the honest cost over `methods`. -/
def EfficiencyFloor (methods : Set Run) (M : Verity.Decoder (List Bool)) (c : ℕ) (ℓ : List Bool → ℝ) (ℓstar : ℝ)
    (D : ℕ∞) (p : ℝ) : Prop :=
  ∀ r ∈ methods, r.A.Valid → r.Within M c D → r.Reaches ℓ ℓstar →
    ENNReal.ofReal p * honestCost methods ℓ ℓstar ≤ (r.d.D.computed.card : ℝ≥0∞)

end Nci.Formalism.Assumptions
