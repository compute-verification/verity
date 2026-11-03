import Definitions.Nci.Formalism

/-!
# The training formalism v2, §4: the assumptions A1, A1′ and A2

* **A1, cut cost** (`CutCost`): every inference block `I` reads at least `u(max_T x_T(I)) + a · m(I)` computed values,
  `|in(I) \ in(C)|`. It is refuted by exhibiting an inference block that violates it, and a theorem reads upper bounds
  from it: a shorter update encoding lowers `u`, an activation code lowers `a`.
* **A1′, update cost from the inputs** (`InputCost`): an inference block that computes a token from its inputs and does
  a share `x` of its affected work reads at least `u_in(x)` values outside `θ`. Computing the token from its inputs, the
  block holds every gate of the token that feeds one of its own through the token, so the work it does is the first
  share `x` in dependency order. The bits it reads are counted as all of `in(I) \ θ`, which holds the update bits: which
  values encode `θ'` is no property of the circuit. With an evenly spread update `u_in = u`; with frozen layers above
  changed ones, `u_in` jumps to about the changed layers' full update almost at once.
* **A2, efficiency floor** (`EfficiencyFloor`): no run of the methods considered whose update stays within `D` bits
  reaches `ℓ*` for less than `p · H(ℓ*)`. It is empirical. It is stated over a class of runs, not over every circuit: a
  circuit that outputs a run's `θ'` as constants reaches the same loss at about `|θ'|` gates, so over every circuit the
  floor fails at any `p · H(ℓ*)` above that (`Definitions.Nci.Formalism.Runs`).

A1 does not charge an inference block for reading other tokens' affected values, as attention's keys and values are.
-/

namespace Nci.Formalism.Assumptions

open Finset
open Verity (Declaration)
open Nci.Formalism
open scoped NNReal ENNReal

/-- `training-cut-cost` (conjectured; refuted by an inference block that violates it). **A1, cut cost**: every
inference block `I` of `d` reads at least `u(max_T x_T(I)) + a · m(I)` computed values, `|in(I) \ in(C)|`. -/
def CutCost (d : Declaration) {L : Type} [Fintype L] (A : Annotation d L) (c k : ℕ) (u : ℝ≥0 → ℝ≥0) (a : ℝ≥0) :
    Prop :=
  ∀ I, InferenceBlock d c k I → u (A.maxShare I) + a * A.cuts I ≤ (computedReads d I).card

/-- `training-input-cost` (conjectured; refuted by an inference block that violates it). **A1′, update cost from the
inputs**: every inference block `I` of `d` that computes a token `T` from its inputs, doing the share `x_T(I)` of its
affected work, reads at least `u_in(x_T(I))` values outside `θ`, `|in(I) \ θ|`. -/
def InputCost (d : Declaration) {L : Type} (A : Annotation d L) (c k : ℕ) (u_in : ℝ≥0 → ℝ≥0) : Prop :=
  ∀ I, InferenceBlock d c k I → ∀ T, A.FromInputs I T → u_in (A.share T I) ≤ (d.unregistered I).card

/-- `training-efficiency-floor` (empirical; falsified by a measured run). **A2, efficiency floor**: no run of `methods`
with a valid annotation whose update stays within `D` bits (its `Δ`, decoded by `M` within `c` steps) reaches `ℓ*` for
less than `p · H(ℓ*)`, `H` the honest cost over `methods`. -/
def EfficiencyFloor (methods : Set Run) (M : Verity.Decoder (List Bool)) (c : ℕ) (ℓ : List Bool → ℝ) (ℓstar : ℝ)
    (D : ℕ∞) (p : ℝ) : Prop :=
  ∀ r ∈ methods, r.A.Valid → r.Within M c D → r.Reaches ℓ ℓstar →
    ENNReal.ofReal p * honestCost methods ℓ ℓstar ≤ (r.d.D.computed.card : ℝ≥0∞)

end Nci.Formalism.Assumptions
