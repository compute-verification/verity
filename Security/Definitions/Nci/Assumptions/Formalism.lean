import Definitions.Nci.Formalism

/-!
# The training formalism v1, §4: the assumptions A1 and A2

* **A1, cut cost** (`CutCost`): every set of at most `c` gates reads at least `u(max_T x_T(I)) + a · m(I)` computed
  values. It is refuted by exhibiting a block that violates it, so `u` and `a` are reported as the best values known from
  witnesses, upper bounds: a shorter update encoding lowers `u`, an activation code lowers `a`.
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

/-- `training-cut-cost` (conjectured; refuted by a block that violates it). **A1, cut cost**: every set `I` of at most
`c` gates of `d` reads at least `u(max_T x_T(I)) + a · m(I)` computed values, `|in(I) \ in(C)|`. -/
def CutCost (d : Declaration) {L : Type} [Fintype L] (A : Annotation d L) (c : ℕ) (u : ℝ≥0 → ℝ) (a : ℝ) : Prop :=
  ∀ I ⊆ d.D.computed, I.card ≤ c → u (A.maxShare I) + a * A.midEntries I ≤ (computedReads d I).card

/-- `training-efficiency-floor` (empirical; falsified by a measured run). **A2, efficiency floor**: no run of `methods`
with a valid annotation whose update stays within `D` bits (its `Δ`, decoded by `M` within `c` steps) reaches `ℓ*` for
less than `p · H(ℓ*)`, `H` the honest cost over `methods`. -/
def EfficiencyFloor (methods : Set Run) (M : Verity.Decoder (List Bool)) (c : ℕ) (ℓ : List Bool → ℝ) (ℓstar : ℝ)
    (D : ℕ∞) (p : ℝ) : Prop :=
  ∀ r ∈ methods, r.A.Valid → r.Within M c D → r.Reaches ℓ ℓstar →
    ENNReal.ofReal p * honestCost methods ℓ ℓstar ≤ (r.d.D.computed.card : ℝ≥0∞)

end Nci.Formalism.Assumptions
