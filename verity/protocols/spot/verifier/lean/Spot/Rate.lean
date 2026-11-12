/-
The rate law's count (`draw.rate_count`): a public rate `q = NUM/DEN` over `T` tables samples a uniform `k`-subset of a
registration's `n` units at `k = T·⌈NUM·n/(DEN·T)⌉`, `⌈q·n⌉` rounded up to a multiple of `T`. Every unit is sampled with
probability `k/n ≥ q` whatever `n` the prover registers (`count_ge`), each of the `T` tables proves `k/T` of them
(`count_dvd`), `k ≥ 1` once `q > 0` and `n ≥ 1` (`count_pos`), and `k ≤ n` whenever `T` divides `n` (`count_le`).

The stratified rate law's `K` (`draw.strat_k`): the stratified law over strata of `n_s` of `N` units at the largest
`K` a stratum needs to draw at least `⌈q·n_s⌉` of them (`need`), at most `N` (`stratK`), so every unit of every stratum
is sampled with probability `k_s/n_s ≥ q` whatever strata the prover registers (`stratK_ge`).

The work rate law's floor (`draw.rate_floor`): a stratum of `n` units draws at least `⌈q·n⌉` of them, the verifier's own
floor `f` where that is more, and never more than `n` (`floor`). A work law sizes each stratum at its floor or above
(`Flock.Draw.workK`), so under any `K`, any works and any population work, every unit of every stratum is sampled with
probability `k_s/n_s ≥ q` whatever strata the prover registers (`workK_ge`), and `1 ≤ k_s ≤ n_s` (`floor_pos`,
`workK_le`).
-/

module

public import Flock.Draw

@[expose] public section

namespace Spot.Rate

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

/-- The least `K ≥ 1` at which the stratified law draws the rate's `c = ⌈NUM·s/DEN⌉` of a stratum of `s` of `n` units
(`Flock.Draw.stratumK K s n ≥ c`): `⌈(2cn − n)/(2s)⌉`, and 1 where `c ≤ 1`, since `stratumK` is never below 1. -/
def need (num den s n : Nat) : Nat :=
  if count num den 1 s ≤ 1 then 1 else (2 * (count num den 1 s * n) - n + 2 * s - 1) / (2 * s)

/-- The stratified rate law's `K` over strata of `sizes` units of `n`: the most any stratum needs (`need`), at most `n`,
at which each stratum draws all of its units. -/
def stratK (num den : Nat) (sizes : List Nat) (n : Nat) : Nat :=
  min n (sizes.foldl (fun k s => max k (need num den s n)) 1)

theorem stratK_le (num den : Nat) (sizes : List Nat) (n : Nat) : stratK num den sizes n ≤ n := Nat.min_le_left _ _

theorem foldl_max (f : Nat → Nat) : ∀ (l : List Nat) (init : Nat),
    init ≤ l.foldl (fun k s => max k (f s)) init ∧ ∀ s ∈ l, f s ≤ l.foldl (fun k s => max k (f s)) init
  | [], _ => ⟨Nat.le_refl _, fun _ h => by cases h⟩
  | x :: xs, init => by
    obtain ⟨h1, h2⟩ := foldl_max f xs (max init (f x))
    simp only [List.foldl_cons]
    refine ⟨Nat.le_trans (Nat.le_max_left _ _) h1, fun s hs => ?_⟩
    rcases List.mem_cons.mp hs with rfl | hs
    · exact Nat.le_trans (Nat.le_max_right _ _) h1
    · exact h2 s hs

theorem ceil_mul (a b : Nat) (hb : 0 < b) : a ≤ b * ((a + b - 1) / b) := by
  have hm := Nat.mod_lt (a + b - 1) hb
  have hd := Nat.div_add_mod (a + b - 1) b
  omega

/-- At `K` at least a stratum's `need`, the stratified law draws the rate's count of it. -/
theorem need_le (num den s n K : Nat) (h0 : 0 < s) (hn : 0 < n) (hK : need num den s n ≤ K) :
    count num den 1 s ≤ Flock.Draw.stratumK K s n := by
  unfold need at hK
  unfold Flock.Draw.stratumK
  generalize count num den 1 s = c at hK ⊢
  by_cases h : c ≤ 1
  · exact Nat.le_trans h (Nat.le_max_left _ _)
  · simp only [h, ↓reduceIte] at hK
    refine Nat.le_trans ?_ (Nat.le_max_right _ _)
    rw [Nat.le_div_iff_mul_le (by omega : 0 < 2 * n), Nat.mul_left_comm c 2 n, Nat.mul_assoc 2 K s]
    have hc := ceil_mul (2 * (c * n) - n) (2 * s) (by omega)
    have hm := Nat.mul_le_mul_left (2 * s) hK
    rw [Nat.mul_assoc 2 s K, Nat.mul_comm s K] at hm
    generalize 2 * s * ((2 * (c * n) - n + 2 * s - 1) / (2 * s)) = r at hc hm
    generalize c * n = p at hc ⊢
    generalize K * s = q at hm ⊢
    omega

/-- **The stratified rate law samples each stratum at least at the rate**: over strata of `sizes` units of `n`, the
stratified law at `stratK NUM DEN sizes n` draws `k` of a stratum of `s` with `NUM·s ≤ DEN·k`, so each of its units is
sampled with probability `k/s ≥ NUM/DEN` however many units the prover registers in it or elsewhere. -/
theorem stratK_ge (num den s n : Nat) (sizes : List Nat) (hs : s ∈ sizes) (h0 : 0 < s) (hsn : s ≤ n) (hd : 0 < den)
    (hq : num ≤ den) : num * s ≤ den * Flock.Draw.stratumK (stratK num den sizes n) s n := by
  refine Nat.le_trans (count_ge num den 1 s hd Nat.one_pos) (Nat.mul_le_mul_left den ?_)
  unfold stratK
  rcases Nat.le_total n (sizes.foldl (fun k s => max k (need num den s n)) 1) with h | h
  · rw [Nat.min_eq_left h]
    refine Nat.le_trans (count_le num den 1 s hd Nat.one_pos hq (Nat.one_dvd s)) ?_
    unfold Flock.Draw.stratumK
    refine Nat.le_trans ?_ (Nat.le_max_right _ _)
    rw [Nat.le_div_iff_mul_le (by omega : 0 < 2 * n), Nat.mul_comm s (2 * n)]
    exact Nat.le_add_right _ _
  · rw [Nat.min_eq_right h]
    exact need_le num den s n _ h0 (by omega) ((foldl_max (fun s => need num den s n) sizes 1).2 s hs)

end Spot.Rate
