import Definitions.Nci.Formalism

/-!
# The training formalism v2, §4: the assumptions A1 and A2

* **A1, cut cost** (`CutCost`): no inference block reads fewer bits than a finite family `𝒲` of witness blocks of the
  run's circuit gives it. Every inference block `I` reads beyond `θ` at least its update bits and, for every token it
  works on, the token's state where its work on it begins: `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. `e_T(I)` is where
  `I`'s work on token `T` begins, `a_T(x)` the least state bits any witness block entering `T` at `x` reads, the earlier
  tokens' keys and values that its work reads and doesn't compute among them, and `u_I` the update bits, counted once,
  the most over `I`'s tokens of the least any witness block entering there reads (`Nci.Formalism.Annotation.stateAt`,
  `Nci.Formalism.Annotation.blockUpdate`). A witness's reads are `in(W) \ θ`, counted as the policy counts a block's.
  So the values are tied to blocks of the circuit, not free: a checker produces `𝒲`, and A1 says no block beats the
  best witness, refuted by an inference block that reads less. A better witness lowers a value: one reading a shorter
  update encoding lowers `u`, one reading an activation code lowers `a`. A block that computes a token from its inputs
  enters it there, its state the token's data and the earlier tokens' keys and values it reads, and A1 prices it as any
  other block: its update bits and that state.
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

/-- `training-cut-cost` (conjectured; its values are the reads of the witness blocks `𝒲`, and it is refuted by an
inference block that reads less). **A1, cut cost**: every inference block `I` of `d` reads beyond `θ` at least its
update bits `u_I` and, for every token `T` it works on, the state bits `a_T(e_T(I))` where its work on `T` begins, each
the least that a witness block of `𝒲` entering there reads: `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. -/
def CutCost {d : Declaration} {L : Type} [Fintype L] (A : Annotation d L) (c k : ℕ) {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) : Prop :=
  ∀ I, InferenceBlock d c k I →
    A.blockUpdate e 𝒲 I + ∑ T ∈ A.worksOn I, A.stateAt e 𝒲 T (e I T) ≤ (d.unregistered I).card

/-- `training-efficiency-floor` (empirical; falsified by a measured run). **A2, efficiency floor**: no run of `methods`
with a valid annotation whose update stays within `D` bits (its `Δ`, decoded by `M` within `c` steps) reaches `ℓ*` for
less than `p · H(ℓ*)`, `H` the honest cost over `methods`. -/
def EfficiencyFloor (methods : Set Run) (M : Verity.Decoder (List Bool)) (c : ℕ) (ℓ : List Bool → ℝ) (ℓstar : ℝ)
    (D : ℕ∞) (p : ℝ) : Prop :=
  ∀ r ∈ methods, r.A.Valid → r.Within M c D → r.Reaches ℓ ℓstar →
    ENNReal.ofReal p * honestCost methods ℓ ℓstar ≤ (r.d.D.computed.card : ℝ≥0∞)

end Nci.Formalism.Assumptions
