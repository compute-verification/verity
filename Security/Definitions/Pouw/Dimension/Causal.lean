import Definitions.Pouw.Dimension.Defs

/-!
# Theorem D with causal inputs (red-team P7-1): trusted statements

Unit `u`'s input `A_u` may depend on `F₁` and on the `E₁` of the units before it: each unit's `E₁` is derived from its
own commitment `com(A_u)`, as in sequential decoding on shared weights, so `A_u` may be computed from earlier units'
results. `Causal A` says that `A ω u` reads only `F₁` and the `E₁` of the units `u′ < u`; `A = g(F₁)` is the case that
reads no `E₁`. The program and its free values are still fixed before the salt, so the inputs stay in `V₀`.

The bound is Theorem D's, `16·|U|·m·n·(3k/16 − 1)`. The proof peels units from the last one: the four-point difference
at unit `u` kills the words of every earlier unit (they do not read `E₁` of unit `u`) and `V₀`, and the later units'
coefficients are already known to be 0.
-/

namespace Pouw.Dimension

open Finset Pouw.NCP Pouw.NCP.Relation

/-- **Causal inputs**: unit `u`'s `A` depends only on `F₁` and on the `E₁` of the units before `u`. -/
def Causal {N m n k : ℕ} (A : Noise N m n k → ℕ → Mat) : Prop :=
  ∀ (ω ω' : Noise N m n k) (u : ℕ), ω.2 = ω'.2 → (∀ u' : Fin N, (u' : ℕ) < u → ω.1 u' = ω'.1 u') →
    A ω u = A ω' u

/-- The checked word `p` of unit `u` for inputs `A` read at the noise itself, as a function of the noise. -/
def wordC (k m n N : ℕ) (R : Mat) (A : Noise N m n k → ℕ → Mat) (B : ℕ → Mat) (u p : ℕ) :
    Noise N m n k → ℚ :=
  fun ω => (checked k m n N R (A ω) B ω u p : ℚ)

end Pouw.Dimension
