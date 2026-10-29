/-
The rate law's count (`draw.rate_count`): a public rate `q = NUM/DEN` over `T` tables draws a uniform `k`-subset of a
registration's `n` units at `k = T·⌈NUM·n/(DEN·T)⌉`, `⌈q·n⌉` rounded up to a multiple of `T`. Every unit is drawn with
probability `k/n ≥ q` whatever `n` the prover registers (`count_ge`), each of the `T` tables proves `k/T` of them
(`count_dvd`), `k ≥ 1` once `q > 0` and `n ≥ 1` (`count_pos`), and `k ≤ n` whenever `T` divides `n` (`count_le`).
-/

namespace SampledProofs.Rate

/-- `T·⌈NUM·n/(DEN·T)⌉`. -/
def count (num den tables n : Nat) : Nat := tables * ((num * n + den * tables - 1) / (den * tables))

theorem count_dvd (num den tables n : Nat) : tables ∣ count num den tables n := Nat.dvd_mul_right _ _

/-- At least the rate: `NUM·n ≤ DEN·k`, so each unit's inclusion `k/n` is at least `NUM/DEN`. -/
theorem count_ge (num den tables n : Nat) (hd : 0 < den) (ht : 0 < tables) :
    num * n ≤ den * count num den tables n := by
  unfold count
  rw [← Nat.mul_assoc]
  have hb : 0 < den * tables := Nat.mul_pos hd ht
  generalize den * tables = b at hb ⊢
  generalize num * n = a
  have hm := Nat.mod_lt (a + b - 1) hb
  have hd := Nat.div_add_mod (a + b - 1) b
  omega

theorem count_pos (num den tables n : Nat) (hq : 0 < num) (hn : 0 < n) (hd : 0 < den) (ht : 0 < tables) :
    0 < count num den tables n := by
  unfold count
  have hb : 0 < den * tables := Nat.mul_pos hd ht
  have ha : 0 < num * n := Nat.mul_pos hq hn
  refine Nat.mul_pos ht (Nat.div_pos ?_ hb)
  omega

/-- No more than the population when `T` divides it and `q ≤ 1`, so such a registration is never refused. -/
theorem count_le (num den tables n : Nat) (hd : 0 < den) (ht : 0 < tables) (hq : num ≤ den) (hn : tables ∣ n) :
    count num den tables n ≤ n := by
  obtain ⟨m, rfl⟩ := hn
  unfold count
  refine Nat.mul_le_mul_left tables (Nat.lt_succ_iff.mp ((Nat.div_lt_iff_lt_mul (Nat.mul_pos hd ht)).mpr ?_))
  have hle : num * (tables * m) ≤ den * tables * m := by
    rw [Nat.mul_assoc]
    exact Nat.mul_le_mul_right _ hq
  have hb : 0 < den * tables := Nat.mul_pos hd ht
  rw [Nat.succ_mul]
  generalize den * tables = b at hb hle ⊢
  generalize num * (tables * m) = a at hle ⊢
  have hc := Nat.mul_comm b m
  omega

end SampledProofs.Rate
