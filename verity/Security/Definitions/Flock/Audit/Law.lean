import Definitions.Flock.Game.Prob
import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Fintype.Fin

/-! The definitions of `Proofs.Flock.Soundness.Audit.Law` that C-Flock's guarantees read, or that a
definition here uses. The lemmas about them stay there. -/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

/-- A draw law over `n` units: the verifier's uniform coins, and the unit set each draws. -/
structure Law (n : ℕ) where
  Ω : Type
  [fintype : Fintype Ω]
  [nonempty : Nonempty Ω]
  draw : Ω → Finset (Fin n)

attribute [instance] Law.fintype Law.nonempty

namespace Law

variable {n : ℕ}

/-- The probability that a draw misses every unit of `B`. -/
noncomputable def escape (L : Law n) (B : Finset (Fin n)) : ℝ≥0∞ :=
  prCoin fun ω => Disjoint (L.draw ω) B

/-- The count curve's sampling term: the largest escape over sets of at least `K` units. -/
noncomputable def miss (L : Law n) (K : ℕ) : ℝ≥0∞ :=
  ⨆ (B : Finset (Fin n)) (_ : K ≤ B.card), L.escape B

/-- The inclusion probability of a unit. -/
noncomputable def incl (L : Law n) (u : Fin n) : ℝ≥0∞ :=
  prCoin fun ω => u ∈ L.draw ω

/-- The uniform `k`-subset of `n` units. -/
def subset (n k : ℕ) (hk : k ≤ n) : Law n where
  Ω := {S : Finset (Fin n) // S.card = k}
  nonempty := by
    obtain ⟨S, -, hS⟩ := Finset.exists_subset_card_eq (s := (Finset.univ : Finset (Fin n)))
      (by simpa using hk)
    exact ⟨⟨S, hS⟩⟩
  draw := Subtype.val

/-- Each unit drawn independently with probability `num / den`. -/
abbrev bernoulli (n num den : ℕ) [NeZero den] : Law n where
  Ω := Fin n → Fin den
  draw ω := Finset.univ.filter fun u => (ω u).val < num

end Law

end FlockSoundness.Audit
