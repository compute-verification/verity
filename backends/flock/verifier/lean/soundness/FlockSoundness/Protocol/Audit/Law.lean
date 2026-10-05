import Mathlib.Data.Fintype.Powerset
import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Data.Fintype.Fin
import FlockSoundness.Protocol.Game.Prob

/-!
# Draw laws

A draw law is the verifier's coin space `Ω`, uniform, and the set of units each coin draws. What a
law gives a consumer is its escape function: the probability that a draw misses a set `B` of units.

* `escape L B`: the draw is disjoint from `B`;
* `miss L K`: the largest escape over sets of at least `K` units, the sampling term of the count
  curve `δ(K) = miss K + ε`;
* `incl L u`: the probability that `u` is drawn; a single unit escapes with `1 - incl L u`
  (`escape_singleton`), and a set escapes with at most that for each of its members
  (`escape_le_of_mem`).

`subset n k` is the uniform `k`-subset of `n` units, whose escape is hypergeometric:
`C(n - |B|, k) / C(n, k)` (`subset_escape`), so its count curve is `C(n - K, k) / C(n, k)`
(`subset_miss`).
-/

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

/-! ## The uniform `k`-subset -/

/-- The uniform `k`-subset of `n` units. -/
def subset (n k : ℕ) (hk : k ≤ n) : Law n where
  Ω := {S : Finset (Fin n) // S.card = k}
  nonempty := by
    obtain ⟨S, -, hS⟩ := Finset.exists_subset_card_eq (s := (Finset.univ : Finset (Fin n)))
      (by simpa using hk)
    exact ⟨⟨S, hS⟩⟩
  draw := Subtype.val

/-! ## Bernoulli draws -/

/-- Each unit drawn independently with probability `num / den`. -/
abbrev bernoulli (n num den : ℕ) [NeZero den] : Law n where
  Ω := Fin n → Fin den
  draw ω := Finset.univ.filter fun u => (ω u).val < num

/-! ## The uniform subset is optimal

Averaging `escape B` over a uniformly random `K`-set `B` gives `E[C(n - |S|, K)] / C(n, K)` for any
law (`avg_escape_subsets`). The map `s ↦ C(n - s, K)` is convex on `ℕ` (its drops fall,
`drop_anti`), so it lies above its supporting line at `k` (`choose_support`), and Jensen's
inequality at a mean of `k` gives `C(n - k, K)`, which is the uniform `k`-subset's value. -/

end Law

end FlockSoundness.Audit
