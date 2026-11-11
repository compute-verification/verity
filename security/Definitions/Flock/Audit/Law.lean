module

public import Definitions.Core.Draw
public import Definitions.Flock.Game
public import Mathlib.Data.Fintype.Fin
public import Mathlib.Data.Fintype.Powerset
public import Mathlib.Data.Nat.Choose.Basic
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-!
# Draw laws

A sampling design is the verifier's coin space `Ω`, uniform, and the set of units each coin samples, core's `Verity.Law`
(`Definitions.Core.Draw`) over `Fin n`. What a sampling design gives a consumer is its escape function: the probability that a sample
misses a set `B` of units.

* `escape L B`: the sample is disjoint from `B`;
* `miss L K`: the largest escape over sets of at least `K` units, the sampling term of the count
  curve `δ(K) = miss K + ε`, zero for `K ≥ 1` at a sampling design that samples every unit (`miss_eq_zero_of_draw_univ`);
* `incl L u`: the probability that `u` is sampled; a single unit escapes with `1 - incl L u`
  (`escape_singleton`), and a set escapes with at most that for each of its members
  (`escape_le_of_mem`).

`subset n k` is the uniform `k`-subset of `n` units, whose escape is hypergeometric:
`C(n - |B|, k) / C(n, k)` (`subset_escape`), so its count curve is `C(n - K, k) / C(n, k)`
(`subset_miss`).

The lemmas named here are `Proofs/Flock/Soundness/Audit/Law.lean`'s.
-/

@[expose] public section

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

/-- A sampling design over `n` units: core's sampling design over `Fin n`. -/
abbrev Law (n : ℕ) : Type 1 := Verity.Law (Fin n)

namespace Law

variable {n : ℕ}

/-! ## The uniform `k`-subset -/

/-- The uniform `k`-subset of `n` units. -/
def subset (n k : ℕ) (hk : k ≤ n) : Law n where
  Ω := {S : Finset (Fin n) // S.card = k}
  nonempty := by
    obtain ⟨S, -, hS⟩ := Finset.exists_subset_card_eq (s := (Finset.univ : Finset (Fin n)))
      (by simpa using hk)
    exact ⟨⟨S, hS⟩⟩
  draw := Subtype.val

/-! ## Bernoulli samples -/

/-- Each unit sampled independently with probability `num / den`. -/
abbrev bernoulli (n num den : ℕ) [NeZero den] : Law n where
  Ω := Fin n → Fin den
  draw ω := Finset.univ.filter fun u => (ω u).val < num

/-- Each unit `u` sampled independently with probability `num u / den`. -/
abbrev bernoulliOf (num : Fin n → ℕ) (den : ℕ) [NeZero den] : Law n where
  Ω := Fin n → Fin den
  draw ω := Finset.univ.filter fun u => (ω u).val < num u

/-- One Bernoulli coordinate's average of `a` when sampled and `1` when not. -/
noncomputable def bernFactor (num den : ℕ) (a : ℝ≥0∞) : ℝ≥0∞ :=
  ((num : ℝ≥0∞) * a + ((den - num : ℕ) : ℝ≥0∞)) / den

/-! ## The uniform subset is optimal

Averaging `escape B` over a uniformly random `K`-set `B` gives `E[C(n - |S|, K)] / C(n, K)` for any
sampling design (`avg_escape_subsets`). The map `s ↦ C(n - s, K)` is convex on `ℕ` (its drops fall,
`drop_anti`), so it lies above its supporting line at `k` (`choose_support`), and Jensen's
inequality at a mean of `k` gives `C(n - k, K)`, which is the uniform `k`-subset's value. -/

/-- The drop of `s ↦ C(n - s, K)` at `s`. -/
def drop (n K s : ℕ) : ℕ := (n - s).choose K - (n - (s + 1)).choose K

end Law

end FlockSoundness.Audit
