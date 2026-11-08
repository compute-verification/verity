module

public import Proofs.Accounting.ReadUnit.Gates
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-!
# The read's values: numbers, lengths and openings

What the read's parts and its template check read of the values its forms hold: a bit list's number (`num_append`,
`num_lt`, `num_take_of_lt`), the lengths of evaluated forms, words and bytes, and the opening the read's witness forms
hold (`openingOf`). The frame read climbs at the token's low bits, and the climb reads only a domain's id and positions,
not its count (`climb_count`).
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm

theorem num_append : ∀ l₁ l₂ : List Bool, num (l₁ ++ l₂) = num l₁ + 2 ^ l₁.length * num l₂
  | [], l₂ => by simp [num]
  | b :: bs, l₂ => by
    simp only [List.cons_append, num, num_append bs l₂, List.length_cons, pow_succ]
    ring

theorem num_lt : ∀ l : List Bool, num l < 2 ^ l.length
  | [] => by simp [num]
  | b :: bs => by
    have := num_lt bs
    simp only [num, List.length_cons, pow_succ]
    cases b <;> simp only [Bool.toNat_false, Bool.toNat_true] <;> omega

/-- A number below `2 ^ k` is its low `k` bits'. -/
theorem num_take_of_lt (l : List Bool) (k : ℕ) (h : num l < 2 ^ k) : num (l.take k) = num l := by
  by_cases hk : k ≤ l.length
  · have hsplit := num_append (l.take k) (l.drop k)
    rw [List.take_append_drop, List.length_take_of_le hk] at hsplit
    have hdrop : num (l.drop k) = 0 := by
      by_contra hne
      have hle : 2 ^ k ≤ 2 ^ k * num (l.drop k) := Nat.le_mul_of_pos_right _ (Nat.pos_of_ne_zero hne)
      generalize 2 ^ k * num (l.drop k) = t at hsplit hle
      omega
    rw [hdrop, mul_zero, add_zero] at hsplit
    exact hsplit.symm
  · rw [List.take_of_length_le (by omega)]

theorem length_evs (z : Var → Bool) (l : List Lin) : (evs z l).length = l.length := by
  simp [evs]

theorem length_readBytes (bs : List Bool) : (ReadUnit.bytesOf bs).length = bs.length / 64 * 8 := by
  simp [ReadUnit.bytesOf, List.length_flatMap, List.map_const', List.sum_replicate, smul_eq_mul]

theorem length_flatten_of_shaped {k : ℕ} {ws : List Word} (h : Shaped k ws) : ws.flatten.length = k * 64 := by
  obtain ⟨hk, hw⟩ := h
  subst hk
  induction ws with
  | nil => simp
  | cons w ws ih =>
    simp only [List.flatten_cons, List.length_append, List.length_cons, hw w List.mem_cons_self,
      ih fun v hv => hw v (List.mem_cons_of_mem _ hv)]
    ring

theorem length_wordsBytes (z : Var → Bool) {k : ℕ} {ws : List Word} (h : Shaped k ws) :
    (wordsBytes z ws).length = k * 8 := by
  rw [wordsBytes, length_readBytes, length_evs, length_flatten_of_shaped h, Nat.mul_div_cancel _ (by norm_num : 0 < 64)]

/-- The climb reads a domain's id, not its count. -/
theorem climb_count (H : Hash.Msg → Verity.Digest) (id : Verity.Digest) (c₁ c₂ : ℕ) (p : ℕ → ℕ)
    (sibs : List Verity.Digest) : ∀ (lv cur : ℕ) (x : Verity.Digest),
      Hash.Tree.climb H ⟨id, c₁, p⟩ lv cur x sibs = Hash.Tree.climb H ⟨id, c₂, p⟩ lv cur x sibs := by
  induction sibs with
  | nil => intro lv cur x; rfl
  | cons s ss ih => intro lv cur x; simp only [Hash.Tree.climb, Hash.Tree.nodeMsg, ih]

/-- The opening the read's witness forms hold under `z`: the row, the salt's bytes and the siblings' digests. -/
def openingOf (z : Var → Bool) (row : List Lin) (salt : List Word) (path : List (List Word)) : Hash.RowLeaf.Opening :=
  ⟨evs z row, wordsBytes z salt, path.map (wordsDigest z)⟩

end Accounting.ReadUnit.Gates
