module

public import Definitions.Core.Prob
public import Definitions.Core.Hash
public import Mathlib.Data.Fintype.Card

/-! A check moved inside zero-knowledge proofs: an accepted run gives values today's check accepts, except with a
stated probability (`Gives`). -/

@[expose] public section

namespace Verity.InProofs

open Verity.Game
open scoped ENNReal

/-- Every field is free here: an instantiating guarantee builds `ports` from a counted extractor and `hashInputs` from
the run, so that neither is a value the prover picks. -/
structure Run (Ω : Type) (n : ℕ) (V : Fin n → Type) where
  accepts : Ω → Prop
  ports : (u : Fin n) → Ω → V u
  hashInputs : Ω → List Hash.Msg

open Classical in
noncomputable def Run.failing {Ω : Type} {n : ℕ} {V : Fin n → Type} (r : Run Ω n V) (ok : (u : Fin n) → V u → Prop)
    (ω : Ω) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ ok u (r.ports u ω)

def Gives {Ω : Type} {n : ℕ} {V : Fin n → Type} (μ : PMF Ω) (r : Run Ω n V) (ok : (u : Fin n) → V u → Prop)
    (K₀ : ℕ) (ε : ℝ≥0∞) : Prop :=
  prPMF μ (fun ω => r.accepts ω ∧ K₀ ≤ (r.failing ok ω).card) ≤ ε

end Verity.InProofs
