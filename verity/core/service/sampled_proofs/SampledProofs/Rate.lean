/-
The rate law's count (`draw.rate_count`): a public rate `q = NUM/DEN` over `T` tables samples a uniform `k`-subset of a
registration's `n` units at `k = T·⌈NUM·n/(DEN·T)⌉`, `⌈q·n⌉` rounded up to a multiple of `T`. Every unit is sampled with
probability `k/n ≥ q` whatever `n` the prover registers (`count_ge`), each of the `T` tables proves `k/T` of them
(`count_dvd`), `k ≥ 1` once `q > 0` and `n ≥ 1` (`count_pos`), and `k ≤ n` whenever `T` divides `n` (`count_le`).

The stratified and work rate laws' floor (`draw.rate_floor`): a stratum of `n` units draws at least `⌈q·n⌉` of them, the
verifier's own floor `f` where that is more, and never more than `n` (`floor`). A work law sizes each stratum at its floor
or above (`Flock.Draw.workK`), so under any `K`, any works and any population work, every unit of every stratum is
sampled with probability `k_s/n_s ≥ q` whatever strata the prover registers (`workK_ge`), and `1 ≤ k_s ≤ n_s`
(`floor_pos`, `workK_le`).
-/

module

public import Flock.Draw

@[expose] public section

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

/-- A stratum's floor at the rate `NUM/DEN`: `⌈NUM·n/DEN⌉` (`count` over one table), raised to the verifier's floor `f`,
never above the stratum's `n` units. -/
def floor (num den n f : Nat) : Nat := min n (max f (count num den 1 n))

theorem floor_le (num den n f : Nat) : floor num den n f ≤ n := Nat.min_le_left _ _

/-- At least the rate: `NUM·n ≤ DEN·floor` once `NUM ≤ DEN`. -/
theorem floor_ge (num den n f : Nat) (hd : 0 < den) (hq : num ≤ den) : num * n ≤ den * floor num den n f := by
  have hc := count_ge num den 1 n hd Nat.one_pos
  unfold floor
  generalize count num den 1 n = c at hc ⊢
  rcases Nat.le_total n (max f c) with h | h
  · rw [Nat.min_eq_left h]; exact Nat.mul_le_mul_right n hq
  · rw [Nat.min_eq_right h]; exact Nat.le_trans hc (Nat.mul_le_mul_left den (Nat.le_max_right f c))

theorem floor_pos (num den n f : Nat) (hq : 0 < num) (hn : 0 < n) (hd : 0 < den) : 0 < floor num den n f := by
  have hc := count_pos num den 1 n hq hn hd Nat.one_pos
  unfold floor
  generalize count num den 1 n = c at hc ⊢
  rcases Nat.le_total n (max f c) with h | h
  · rw [Nat.min_eq_left h]; exact hn
  · rw [Nat.min_eq_right h]; exact Nat.lt_of_lt_of_le hc (Nat.le_max_right f c)

/-- **A stratum sized from a rate floor samples at least the rate**: under any `K`, work and population work, the work
law's draw over a stratum of `n` units whose floor is `floor NUM DEN n f` is `k` with `NUM·n ≤ DEN·k`, so each of its
units is sampled with probability `k/n ≥ NUM/DEN` however many units the prover registers in it or elsewhere. -/
theorem workK_ge (num den n f k work total : Nat) (hd : 0 < den) (hq : num ≤ den) :
    num * n ≤ den * Flock.Draw.workK k work n total (floor num den n f) := by
  have hf := floor_ge num den n f hd hq
  unfold Flock.Draw.workK
  generalize floor num den n f = fl at hf ⊢
  generalize (k * work * n + total - 1) / total = x
  rcases Nat.le_total n (max fl x) with h | h
  · rw [Nat.min_eq_left h]; exact Nat.mul_le_mul_right n hq
  · rw [Nat.min_eq_right h]; exact Nat.le_trans hf (Nat.mul_le_mul_left den (Nat.le_max_left fl x))

theorem workK_le (n f k work total num den : Nat) : Flock.Draw.workK k work n total (floor num den n f) ≤ n :=
  Nat.min_le_left _ _

end SampledProofs.Rate
