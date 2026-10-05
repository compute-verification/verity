import Mathlib.Analysis.MeanInequalities
import FlockSoundness.Protocol.ZerocheckPhase

/-!
# Ligerito's stratified queries (PROTOCOL.md §13.5)

If a set `D` covers a fraction `δ` of a level's `2^d` positions, the `Q` stratified query
positions all miss it with probability at most `(1 − δ)^Q`, the same as `Q` independent uniform
positions (`prCoin_positions_miss_le`):
* the strata sum to `Q` (`strata_sum`), and each query is uniform in its stratum's block, since the
  low bits of `lo` are balanced (`Arith.Correct.lo_balanced`);
* the queries' coins are independent, so the miss probabilities multiply (`prCoin_forall_fin`);
* within one summand the blocks partition the positions, so AM–GM bounds the product of the
  blocks' miss probabilities by `(1 − δ)^{2^c}`.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-! ## The strata -/

theorem strata_go_sum (d : ℕ) : ∀ fuel rem, rem ≤ fuel →
    ((strata.go d fuel rem).map (2 ^ ·)).sum = rem
  | 0, rem, h => by simp [strata.go]; omega
  | fuel + 1, rem, h => by
      rw [strata.go]
      split_ifs with h0
      · simp [h0]
      · simp only [List.map_cons, List.sum_cons]
        have hc : 2 ^ min (Nat.log 2 rem) d ≤ rem :=
          (Nat.pow_le_pow_right (by norm_num) (min_le_left _ _)).trans (Nat.pow_log_le_self 2 h0)
        have hc1 : 1 ≤ 2 ^ min (Nat.log 2 rem) d := Nat.one_le_two_pow
        rw [strata_go_sum d fuel (rem - 2 ^ min (Nat.log 2 rem) d) (by omega)]
        omega

/-- The strata's sizes sum to `Q`. -/
theorem strata_sum (Q d : ℕ) : ((strata Q d).map (2 ^ ·)).sum = Q :=
  strata_go_sum d Q Q le_rfl

theorem strata_go_le (d : ℕ) : ∀ fuel rem, ∀ c ∈ strata.go d fuel rem, c ≤ d
  | 0, rem => by simp [strata.go]
  | fuel + 1, rem => by
      rw [strata.go]
      split_ifs with h0
      · simp
      · intro c hc
        simp only [List.mem_cons] at hc
        rcases hc with rfl | hc
        · exact min_le_right _ _
        · exact strata_go_le d fuel _ c hc

/-- Every stratum's depth is at most `d`. -/
theorem strata_le (Q d : ℕ) : ∀ c ∈ strata Q d, c ≤ d := strata_go_le d Q Q

/-! ## Slots and positions -/

/-- The query slots `(depth, stratum)`, in query order (§13.5). -/
def querySlots (Q d : ℕ) : List (ℕ × ℕ) :=
  (strata Q d).flatMap fun c => (List.range (2 ^ c)).map fun s => (c, s)

theorem querySlots_length (Q d : ℕ) : (querySlots Q d).length = Q := by
  rw [querySlots, List.length_flatMap]
  simpa using strata_sum Q d

/-- The position of the query in slot `(c, s)`, from its coin. -/
def slotPos (A : Arith F K) (d : ℕ) (cs : ℕ × ℕ) (x : F) : ℕ :=
  cs.2 * 2 ^ (d - cs.1) + A.lo x % 2 ^ (d - cs.1)

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K] in
theorem positions_eq (A : Arith F K) (Q d : ℕ) (w : Fin Q → F) :
    positions A Q d w = List.ofFn (fun j : Fin Q =>
      slotPos A d ((querySlots Q d)[j.val]'(by rw [querySlots_length]; exact j.isLt)) (w j)) := by
  unfold positions
  apply List.ext_getElem
  · simpa [querySlots, List.length_flatMap] using strata_sum Q d
  · intro i h1 h2
    simp only [List.getElem_map, List.getElem_zipIdx, List.getElem_ofFn, slotPos, zero_add]
    have hi : i < Q := by simpa using h2
    rw [dif_pos hi]
    rfl

/-! ## Independent coins -/

/-! ## AM–GM over the strata -/

end FlockSoundness
