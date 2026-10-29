import Definitions.Pouw.Dimension.Pointwise

/-!
# adaptive-dw: what a prover that reads the coins must write (trusted definitions)

**Draft model, pending review.** In the chunk model (`Pouw.Dimension.Pointwise`) a run may depend on the coins, so an
adaptive prover is one run a coin draw. `Holds m₀ P Y` says the run `P` from `m₀` holds every chunk value of `Y` in
some slot at some point, as the hash reads it. `AdaptiveDW Ω m₀ Y B` says every family of runs, the one at draw `g`
holding `Y g`, writes at least `B` summed over the draws `Ω`: a bound on the expected writes, times `|Ω|`.
-/

namespace Pouw.Dimension.Pointwise

variable {A V : Type} {w : ℕ} [DecidableEq A]

/-- The run `P` from `m₀` holds every chunk value of `Y` in some slot at some point. -/
def Holds (m₀ : A → Fin w → V) (P : List (Step A V w)) (Y : Finset (Fin w → V)) : Prop :=
  ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y

/-- **adaptive-dw at `B`**: every family of runs from `m₀`, one a coin draw `g`, the run at `g` holding `Y g`, writes
at least `B` summed over the draws. -/
def AdaptiveDW (Ω : Type) [Fintype Ω] (m₀ : A → Fin w → V) (Y : Ω → Finset (Fin w → V)) (B : ℕ) : Prop :=
  ∀ P : Ω → List (Step A V w), (∀ g, Holds m₀ (P g) (Y g)) → B ≤ ∑ g, writes (P g)

end Pouw.Dimension.Pointwise
