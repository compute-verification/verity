module

public import Definitions.Core.Draw
public import Definitions.Flock.Game
public import Mathlib.Data.Fintype.Fin
public import Mathlib.Data.Fintype.Powerset
public import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! Sampling designs and their escape functions (`escape`, `miss`, `incl`), and the uniform `k`-subset (`subset`). -/

@[expose] public section

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

/-- A sampling design over `n` units: core's sampling design over `Fin n`. -/
abbrev SamplingDesign (n : ℕ) : Type 1 := Verity.SamplingDesign (Fin n)

namespace SamplingDesign

variable {n : ℕ}

/-- The uniform `k`-subset of `n` units. -/
def subset (n k : ℕ) (hk : k ≤ n) : SamplingDesign n where
  Ω := {S : Finset (Fin n) // S.card = k}
  nonempty := by
    obtain ⟨S, -, hS⟩ := Finset.exists_subset_card_eq (s := (Finset.univ : Finset (Fin n)))
      (by simpa using hk)
    exact ⟨⟨S, hS⟩⟩
  draw := Subtype.val

/-- Each unit sampled independently with probability `num / den`. -/
abbrev bernoulli (n num den : ℕ) [NeZero den] : SamplingDesign n where
  Ω := Fin n → Fin den
  draw ω := Finset.univ.filter fun u => (ω u).val < num

/-- Each unit `u` sampled independently with probability `num u / den`. -/
abbrev bernoulliOf (num : Fin n → ℕ) (den : ℕ) [NeZero den] : SamplingDesign n where
  Ω := Fin n → Fin den
  draw ω := Finset.univ.filter fun u => (ω u).val < num u

/-- One Bernoulli coordinate's average of `a` when sampled and `1` when not. -/
noncomputable def bernFactor (num den : ℕ) (a : ℝ≥0∞) : ℝ≥0∞ :=
  ((num : ℝ≥0∞) * a + ((den - num : ℕ) : ℝ≥0∞)) / den

def drop (n K s : ℕ) : ℕ := (n - s).choose K - (n - (s + 1)).choose K

end SamplingDesign

end FlockSoundness.Audit
