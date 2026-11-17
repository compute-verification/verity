module

public import Definitions.Pouw.Assumptions.Dimension.H100W
public import Definitions.Pouw.Fp8Atom.H1T

/-! H-1T's gap corollary over several units in `MH100w` (draft semantics), under `DistinctLiveH1T`. -/

@[expose] public section

namespace Pouw.Fp8Atom.H1T

open Pouw.Dimension.H100

/-- The joint salt of units `Us`: unit `u`'s salt is coordinate `u`. -/
abbrev MSalt {N : ℕ} (Us : Fin N → Unit) : Type := (u : Fin N) → (Us u).Salt

/-- A credited word of units `Us`: a unit, and a row, column and position of it. -/
abbrev Idx {N : ℕ} (Us : Fin N → Unit) : Type := Σ u : Fin N, Fin (Us u).m × Fin (Us u).n × Pos (Us u).T

/-- The credited word `x` as a function of the joint salt: unit `x.1`'s word on its own coordinate. -/
def mword {N : ℕ} (Us : Fin N → Unit) (x : Idx Us) : MSalt Us → ℚ := fun G => (Us x.1).word x.2 (G x.1)

/-- The free values on the joint salt: the constants and every unit's raw XOF words. -/
def mfree {N : ℕ} (Us : Fin N → Unit) : Set (MSalt Us → ℚ) :=
  {f | (∃ q : ℚ, f = fun _ => q) ∨
    ∃ (u : Fin N) (i : Fin (Us u).m) (τ : Fin (Us u).T) (ℓ : Fin 18), f = fun G => ((G u i τ ℓ : ℕ) : ℚ)}

/-- **Several units**, under `DistinctLiveH1T`: per-unit Target A lifts to any number of units. -/
def H1TUnitsDistinct : Prop :=
  ∀ (N : ℕ) (Us : Fin N → Unit), (∀ u, (Us u).Legal) → Function.Injective (mword Us) ∧ ∀ x, mword Us x ∉ mfree Us

/-- **The gap for H-1T** (under `DistinctLiveH1T`; `MH100w` is draft semantics): under `H32`, an `MH100w` program on free
inputs whose registers include every credited word of legal units costs at least 32 per credited word. -/
def H1TGap : Prop :=
  ∀ pr : Prices100, H32 pr → ∀ (N : ℕ) (Us : Fin N → Unit), (∀ u, (Us u).Legal) →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (MSalt Us → ℚ)) (P : List (Op100 (MSalt Us))),
      (∀ f ∈ L, f ∈ mfree Us) → (∀ o ∈ P, o.WFw pr (mfree Us)) →
      (∀ x : Idx Us, ∃ r ∈ regs100 fpSem P L, ∀ G, r G = mword Us x G) →
      32 * ((∑ u, (Us u).m * (Us u).n * Fintype.card (Pos (Us u).T) : ℕ) : ℚ) ≤ (cost100 pr P : ℚ)

end Pouw.Fp8Atom.H1T
