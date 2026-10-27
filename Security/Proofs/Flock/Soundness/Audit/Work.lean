import Proofs.Flock.Soundness.Audit.Stratified
import Mathlib.Analysis.MeanInequalities
import Definitions.Flock.Guarantees.Audit

/-!
# The work law

The draw law's second version: the stratified law's strata, with each stratum's draw sized by the PoUW
work its units do instead of by their number. Stratum `s` has `n_s` units, each doing work `w_s` (zero for units that do no
PoUW work: dequantization, the quantizer, attention), and the population does `W = Σ_s w_s n_s`. Stratum `s` draws
`k_s = min(n_s, max(f_s, ⌈K w_s n_s / W⌉))` (`Law.workRule`, the executable's `Flock.Draw.workK`) above its floor `f_s ≥ 1`,
which the verifier's work table sets per stratum beside `w_s` (one when it names none).

The work law is `Law.stratified` at those sizes (`Law.work`), so every theorem of the stratified law and of the one-stage
audit holds of it as stated: `stratified_escape`, `stratified_escape_floor`, `audit_whole_stratum`,
`extraction_audit_whole_stratum`, `audit_profile`, `audit_count`, `audit_drawn` and the Flock instantiations. What is new,
for every floor `f ≥ 1`:

* `floor_le_workK`: **the floor**. A stratum draws at least `min(n_s, f_s)` units, so every non-empty stratum draws at least
  one and a wholly wrong stratum is caught whatever its work (`audit_work_whole_stratum`,
  `extraction_audit_work_whole_stratum`).
* `work_escape_floor_le`: **the per-stratum floor escape**. A set with `b_s` units in a non-empty stratum `s` escapes with
  probability at most `((n_s − b_s)/n_s)^(f_s)`, so a floor `f_s ≈ ln(1/δ)/ε_s` catches a fraction `ε_s` of the stratum
  wrong up to `δ` (`audit_work_floor`, `extraction_audit_work_floor`).
* `workK_ge`: **draws sized by work**. A stratum draws at least `K w_s n_s / W` units, or all of them.
* `sum_workK_le`: at most `K + Σ_s f_s` units are drawn (`K` plus one per stratum at `f = 1`).
* `work_escape_le`: **the work bound**. A set of units doing `W_B` of the work escapes with probability at most
  `((W − W_B)/W)^K`, so wrong units doing a fraction `ε` of the work escape with at most `(1 − ε)^K`.
* `audit_work`, `extraction_audit_work`: the audit accepts while its wrong units do at least `T` of the work with
  probability at most `((W − T)/W)^K + ε_ks + δ_link`.
* `record_sizing`, `audit_work_of_record`: at the sizing of record (`δ = 2^-40`, `ε = 0.1%`), `K = 27,713` suffices;
  `record_sizing_one_fewer`: `K = 27,712` does not, under this bound.

For the strata that do no work:

* `workK_rest`: such a stratum draws exactly `min(n_s, f_s)`.
* `work_escape_le_count`: **floors at the count law's sizes keep its guarantee**. When every stratum that does no work has
  `f_s` at least `countK σ C s` (`main`'s stratified law's `stratumK C n_s N` for a non-empty stratum and `C ≤ N`,
  `countK_eq`), a set escapes the work law at most as often as its units in those strata escape the stratified law at `C`;
  `audit_work_rest` and `extraction_audit_work_rest` carry any such integrity profile to the audit.
* `countRule_self`, `countK_self`, `countK_self_miss`: at `C = N` the count law draws every unit of every stratum
  (`stratumK N n_s N = n_s`), so its sampling term is zero: a recursive inner session's full draw.

Closure draws (a drawn unit proved with the units it reads) are not part of this law. `Law.widen` proves more units than a
law draws on the same coins, and `escape_widen_le` carries every escape bound to it.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m : ℕ}

/-- The work rule, the executable's `Flock.Draw.workK`: `⌈K · work · size / total⌉`, at least `floor` and at most `size`. -/
def workRule (K work size total floor : ℕ) : ℕ := min size (max floor ((K * work * size + total - 1) / total))

/-- The work of a set of units, each doing its stratum's `w (σ u)`. -/
def workOf (σ : Fin n → Fin m) (w : Fin m → ℕ) (B : Finset (Fin n)) : ℕ := ∑ u ∈ B, w (σ u)

/-- The population's work, `W`. -/
abbrev totalWork (σ : Fin n → Fin m) (w : Fin m → ℕ) : ℕ := workOf σ w univ

/-- Stratum `s`'s draw under the work law with floors `f`. -/
def workK (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule K (w s) (stratum σ s).card (totalWork σ w) (f s)

theorem workK_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    workK σ w f K s ≤ (stratum σ s).card :=
  min_le_left _ _

/-- **The work law**: independent uniform subsets inside the strata of `σ`, stratum `s` drawing `workK σ w f K s`. -/
abbrev work (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) : Law n :=
  stratified σ (workK σ w f K) (workK_le σ w f K)

theorem workOf_eq (σ : Fin n → Fin m) (w : Fin m → ℕ) (B : Finset (Fin n)) :
    workOf σ w B = ∑ s, w s * (B.filter (σ · = s)).card := by
  unfold workOf
  rw [← Finset.sum_fiberwise B σ fun u => w (σ u)]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [Finset.sum_congr rfl fun u hu => congrArg w (Finset.mem_filter.1 hu).2, Finset.sum_const, smul_eq_mul,
    mul_comm]

theorem totalWork_eq (σ : Fin n → Fin m) (w : Fin m → ℕ) :
    totalWork σ w = ∑ s, w s * (stratum σ s).card :=
  workOf_eq σ w univ

/-! ## The rule -/

/-- **The floor**: a stratum draws at least `min(n_s, f_s)` units, whatever its work. -/
theorem floor_le_workK (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    min (stratum σ s).card (f s) ≤ workK σ w f K s :=
  min_le_min le_rfl (le_max_left _ _)

/-- With `f_s ≥ 1`, every non-empty stratum draws at least one unit, whatever its work. -/
theorem one_le_workK {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} {s : Fin m} (hf : 1 ≤ f s)
    (h : (stratum σ s).Nonempty) : 1 ≤ workK σ w f K s :=
  (le_min (Finset.card_pos.2 h) hf).trans (floor_le_workK σ w f K s)

theorem le_mul_ceil_div (x W : ℕ) (hW : 0 < W) : x ≤ W * ((x + W - 1) / W) := by
  have h1 := Nat.div_add_mod (x + W - 1) W
  have h2 := Nat.mod_lt (x + W - 1) hW
  generalize W * ((x + W - 1) / W) = p at h1 ⊢
  generalize (x + W - 1) % W = r at h1 h2
  omega

/-- **Draws sized by work**: stratum `s` draws all its units, or at least `K w_s n_s / W` of them. -/
theorem workK_ge {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ} (s : Fin m) (hW : 0 < totalWork σ w) :
    workK σ w f K s = (stratum σ s).card ∨
      K * w s * (stratum σ s).card ≤ totalWork σ w * workK σ w f K s := by
  unfold workK workRule
  rcases le_total (stratum σ s).card
      (max (f s) ((K * w s * (stratum σ s).card + totalWork σ w - 1) / totalWork σ w)) with h | h
  · exact Or.inl (min_eq_left h)
  · refine Or.inr ?_
    rw [min_eq_right h]
    exact (le_mul_ceil_div _ _ hW).trans (Nat.mul_le_mul_left _ (le_max_right _ _))

theorem mul_workRule_le (K work size total floor : ℕ) (hf : 1 ≤ floor) :
    total * workRule K work size total floor ≤ K * work * size + total * floor := by
  unfold workRule
  have hq := Nat.mul_div_le (K * work * size + total - 1) total
  have hW : total ≤ total * floor := Nat.le_mul_of_pos_right total hf
  refine (Nat.mul_le_mul_left total (min_le_right _ _)).trans ?_
  rcases le_total floor ((K * work * size + total - 1) / total) with h | h
  · rw [max_eq_right h]
    have := Nat.mul_comm total ((K * work * size + total - 1) / total)
    omega
  · rw [max_eq_left h]
    omega

/-- **The draw's size**: at most `K + Σ_s f_s` units; at `f = 1`, `K` plus one per stratum. -/
theorem sum_workK_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < totalWork σ w) : ∑ s, workK σ w f K s ≤ K + ∑ s, f s := by
  refine Nat.le_of_mul_le_mul_left ?_ hW
  rw [Finset.mul_sum]
  calc ∑ s, totalWork σ w * workK σ w f K s ≤ ∑ s, (K * w s * (stratum σ s).card + totalWork σ w * f s) :=
        Finset.sum_le_sum fun s _ => mul_workRule_le _ _ _ _ _ (hf s)
    _ = K * ∑ s, w s * (stratum σ s).card + totalWork σ w * ∑ s, f s := by
        rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.mul_sum]
        simp only [mul_assoc]
    _ = totalWork σ w * (K + ∑ s, f s) := by rw [← totalWork_eq]; ring

/-! ## The work bound -/

/-- `C(n − m, k) · n^k ≤ C(n, k) · (n − m)^k`: a uniform `k`-subset of `n` units misses `m` given units at most as
often as `k` independent uniform units do. -/
theorem choose_sub_mul_pow_le' (n m : ℕ) : ∀ k, (n - m).choose k * n ^ k ≤ n.choose k * (n - m) ^ k
  | 0 => by simp
  | k + 1 => by
    have ih := choose_sub_mul_pow_le' n m k
    have ha := Nat.choose_succ_right_eq (n - m) k
    have hn := Nat.choose_succ_right_eq n k
    have key : (n - m - k) * n ≤ (n - k) * (n - m) := by
      rcases le_or_gt (n - m) k with h | h
      · rw [Nat.sub_eq_zero_of_le h, zero_mul]
        exact Nat.zero_le _
      · have hmn : m ≤ n := by omega
        have hkn : k ≤ n := by omega
        zify [hmn, hkn, h.le]
        nlinarith [mul_nonneg (Int.natCast_nonneg k) (Int.natCast_nonneg m)]
    generalize n - m = a at ih ha key ⊢
    generalize a - k = b at ha key
    generalize n - k = c at hn key
    refine Nat.le_of_mul_le_mul_right ?_ (Nat.succ_pos k)
    calc a.choose (k + 1) * n ^ (k + 1) * (k + 1) = a.choose (k + 1) * (k + 1) * n ^ k * n := by ring
      _ = (a.choose k * n ^ k) * (b * n) := by rw [ha]; ring
      _ ≤ (n.choose k * a ^ k) * (c * a) := Nat.mul_le_mul ih key
      _ = n.choose k * c * a ^ (k + 1) := by ring
      _ = n.choose (k + 1) * a ^ (k + 1) * (k + 1) := by rw [← hn]; ring

/-- The hypergeometric escape of `m` of `n` units under a uniform `k`-subset is at most `((n − m)/n)^k`. -/
theorem choose_div_le_pow {n m k : ℕ} (hk : k ≤ n) (hn : 0 < n) :
    ((n - m).choose k : ℝ≥0) / (n.choose k : ℝ≥0) ≤ (((n - m : ℕ) : ℝ≥0) / (n : ℝ≥0)) ^ k := by
  have hC : (0 : ℝ≥0) < n.choose k := by exact_mod_cast Nat.choose_pos hk
  have hN : (0 : ℝ≥0) < (n : ℝ≥0) ^ k := pow_pos (by exact_mod_cast hn) k
  rw [div_pow, div_le_div_iff₀ hC hN]
  exact_mod_cast (choose_sub_mul_pow_le' n m k).trans_eq (mul_comm _ _)

/-- **The work bound.** Under the work law, a set of units doing `W_B` of the population's work `W` escapes with
probability at most `((W − W_B)/W)^K`. So wrong units doing a fraction `ε` of the work escape with at most `(1 − ε)^K`. -/
theorem work_escape_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < totalWork σ w) (B : Finset (Fin n)) :
    (work σ w f K).escape B ≤ (((totalWork σ w - workOf σ w B : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  classical
  set W := totalWork σ w with hWdef
  set N : Fin m → ℕ := fun s => (stratum σ s).card with hNdef
  set M : Fin m → ℕ := fun s => (B.filter (σ · = s)).card with hMdef
  set k : Fin m → ℕ := workK σ w f K with hkdef
  have hMN : ∀ s, M s ≤ N s := fun s => card_le_card (filter_subset_filter _ (subset_univ B))
  have hkN : ∀ s, k s ≤ N s := workK_le σ w f K
  have hW0 : (W : ℝ≥0) ≠ 0 := by exact_mod_cast hW.ne'
  -- each stratum's share `a s` of the work, and the fraction `x s` of its units outside `B`
  set x : Fin m → ℝ≥0 := fun s => ((N s - M s : ℕ) : ℝ≥0) / (N s : ℝ≥0) with hxdef
  set a : Fin m → ℝ≥0 := fun s => ((w s * N s : ℕ) : ℝ≥0) / (W : ℝ≥0) with hadef
  have hsumN : ∑ s, w s * N s = W := (totalWork_eq σ w).symm
  have ha1 : ∑ s, a s = 1 := by
    simp only [hadef]
    rw [← Finset.sum_div, ← Nat.cast_sum, hsumN, div_self hW0]
  have hterm : ∀ s, a s * x s = ((w s * (N s - M s) : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    intro s
    simp only [hadef, hxdef]
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hM0 : M s = 0 := by have := hMN s; omega
      simp [h0, hM0]
    · have hN0 : (N s : ℝ≥0) ≠ 0 := by exact_mod_cast hpos.ne'
      rw [div_mul_div_comm, ← Nat.cast_mul, show w s * N s * (N s - M s) = w s * (N s - M s) * N s by ring,
        Nat.cast_mul, mul_div_mul_right _ _ hN0]
  have hax : ∑ s, a s * x s = ((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0) := by
    rw [Finset.sum_congr rfl fun s _ => hterm s, ← Finset.sum_div, ← Nat.cast_sum]
    congr 2
    have hB : workOf σ w B = ∑ s, w s * M s := workOf_eq σ w B
    have hsplit : ∑ s, w s * (N s - M s) + ∑ s, w s * M s = ∑ s, w s * N s := by
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun s _ => by rw [← Nat.mul_add, Nat.sub_add_cancel (hMN s)]
    omega
  -- per stratum, the hypergeometric escape is at most `x s ^ (K · a s)`
  have hfac : ∀ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ ((a s : ℝ) * K) := by
    intro s
    rcases Nat.eq_zero_or_pos (N s) with h0 | hpos
    · have hk0 : k s = 0 := by have := hkN s; omega
      have hM0 : M s = 0 := by have := hMN s; omega
      simp [hadef, h0, hk0, hM0]
    have hC0 : ((N s).choose (k s) : ℝ≥0) ≠ 0 := by exact_mod_cast (Nat.choose_pos (hkN s)).ne'
    rcases Nat.eq_zero_or_pos (M s) with hM0 | hMpos
    · have hx1 : x s = 1 := by
        simp only [hxdef, hM0, Nat.sub_zero]
        exact div_self (by exact_mod_cast hpos.ne')
      rw [hx1, NNReal.one_rpow, hM0, Nat.sub_zero, div_self hC0]
    have hk1 : 1 ≤ k s := one_le_workK (hf s) (Finset.card_pos.1 hpos)
    have hMNs := hMN s
    by_cases hzero : k s = N s ∨ M s = N s
    · have : (N s - M s).choose (k s) = 0 := Nat.choose_eq_zero_of_lt (by omega)
      rw [this, Nat.cast_zero, zero_div]
      exact zero_le
    push Not at hzero
    have hge : K * w s * N s ≤ W * k s := (workK_ge s hW).resolve_left hzero.1
    have hx0 : 0 < x s := div_pos (by exact_mod_cast (show 0 < N s - M s by omega)) (by exact_mod_cast hpos)
    have hx1 : x s ≤ 1 := div_le_one_of_le₀ (by exact_mod_cast Nat.sub_le _ _) zero_le
    have hexp : (a s : ℝ) * K ≤ (k s : ℝ) := by
      simp only [hadef, NNReal.coe_div, NNReal.coe_natCast]
      rw [div_mul_eq_mul_div, div_le_iff₀ (by exact_mod_cast hW)]
      exact_mod_cast (show w s * N s * K ≤ k s * W by linarith [hge, mul_comm (K * w s) (N s)])
    calc ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤ x s ^ (k s) :=
          choose_div_le_pow (hkN s) hpos
      _ = x s ^ ((k s : ℕ) : ℝ) := (NNReal.rpow_natCast _ _).symm
      _ ≤ x s ^ ((a s : ℝ) * K) := NNReal.rpow_le_rpow_of_exponent_ge hx0 hx1 hexp
  -- weighted AM-GM over the strata, with weights the work shares
  have hprod : ∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) ≤
      (((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K := by
    calc _ ≤ ∏ s, x s ^ ((a s : ℝ) * K) := Finset.prod_le_prod fun s _ => hfac s
      _ = (∏ s, x s ^ (a s : ℝ)) ^ K := by
          rw [← Finset.prod_pow]
          exact Finset.prod_congr rfl fun s _ => by rw [NNReal.rpow_mul, NNReal.rpow_natCast]
      _ ≤ (∑ s, a s * x s) ^ K :=
          pow_le_pow_left₀ zero_le (NNReal.geom_mean_le_arith_mean_weighted univ a x ha1) K
      _ = _ := by rw [hax]
  have hC : ∀ s, ((N s).choose (k s) : ℝ≥0) ≠ 0 := fun s => by exact_mod_cast (Nat.choose_pos (hkN s)).ne'
  rw [stratified_escape]
  calc ∏ s, (((stratum σ s).card - (B.filter (σ · = s)).card).choose (workK σ w f K s) : ℝ≥0∞) /
        ((stratum σ s).card.choose (workK σ w f K s) : ℝ≥0∞)
      = ((∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) := by
        rw [ENNReal.ofNNReal_finsetProd]
        refine Finset.prod_congr rfl fun s _ => ?_
        rw [ENNReal.coe_div (hC s)]
        simp [hNdef, hMdef, hkdef]
    _ ≤ (((((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K : ℝ≥0) : ℝ≥0∞) := ENNReal.coe_le_coe.2 hprod
    _ = _ := by rw [ENNReal.coe_pow, ENNReal.coe_div hW0, ENNReal.coe_natCast, ENNReal.coe_natCast]

/-- The work bound, monotone in the threshold: units doing at least `T` of the work escape with at most
`((W − T)/W)^K`. -/
theorem work_escape_le_of_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < totalWork σ w) {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ workOf σ w B) :
    (work σ w f K).escape B ≤ (((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  refine (work_escape_le σ w f K hf hW B).trans ?_
  gcongr

/-! ## Floors -/

/-- The hypergeometric escape of `m` of `n` units under a uniform `k`-subset, `C(n − m, k)/C(n, k)`, in `ℝ≥0∞`, is at most
`((n − m)/n)^k`. -/
theorem choose_div_le_pow' {n m k : ℕ} (hk : k ≤ n) (hn : 0 < n) :
    ((n - m).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) ≤ (((n - m : ℕ) : ℝ≥0∞) / (n : ℝ≥0∞)) ^ k := by
  have hC : ((n.choose k : ℕ) : ℝ≥0) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk).ne'
  have hN : ((n : ℕ) : ℝ≥0) ≠ 0 := by exact_mod_cast hn.ne'
  have h := ENNReal.coe_le_coe.2 (choose_div_le_pow (m := m) hk hn)
  rw [ENNReal.coe_div hC, ENNReal.coe_pow, ENNReal.coe_div hN] at h
  simpa using h

/-- A hypergeometric factor is at most one. -/
theorem choose_div_le_one (n m k : ℕ) : ((n - m).choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) ≤ 1 := by
  refine ENNReal.div_le_of_le_mul ?_
  rw [one_mul]
  exact_mod_cast Nat.choose_le_choose _ (Nat.sub_le _ _)

/-- **The per-stratum floor escape**: a set with `b_s` units in a non-empty stratum `s` escapes with probability at most
`((n_s − b_s)/n_s)^(f_s)`. -/
theorem work_escape_floor_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m)
    (hs : (stratum σ s).Nonempty) (B : Finset (Fin n)) :
    (work σ w f K).escape B ≤
      ((((stratum σ s).card - (B.filter (σ · = s)).card : ℕ) : ℝ≥0∞) / ((stratum σ s).card : ℝ≥0∞)) ^ (f s) := by
  classical
  rw [stratified_escape, ← Finset.mul_prod_erase univ _ (mem_univ s)]
  refine (mul_le_of_le_one_right' (Finset.prod_le_one fun t _ => choose_div_le_one _ _ _)).trans ?_
  have hpos : 0 < (stratum σ s).card := Finset.card_pos.2 hs
  have hk := workK_le σ w f K s
  have hM : (B.filter (σ · = s)).card ≤ (stratum σ s).card := card_le_card (filter_subset_filter _ (subset_univ B))
  have hx1 : (((stratum σ s).card - (B.filter (σ · = s)).card : ℕ) : ℝ≥0∞) / ((stratum σ s).card : ℝ≥0∞) ≤ 1 :=
    ENNReal.div_le_of_le_mul (by rw [one_mul]; exact_mod_cast Nat.sub_le _ _)
  by_cases hfk : f s ≤ workK σ w f K s
  · exact (choose_div_le_pow' hk hpos).trans (pow_le_pow_of_le_one zero_le hx1 hfk)
  · have hkN : workK σ w f K s = (stratum σ s).card := by
      have := floor_le_workK σ w f K s
      omega
    rcases Nat.eq_zero_or_pos (B.filter (σ · = s)).card with h0 | hb
    · rw [h0, Nat.sub_zero, ENNReal.div_self (by exact_mod_cast (Nat.choose_pos hk).ne') (ENNReal.natCast_ne_top _),
        ENNReal.div_self (by exact_mod_cast hpos.ne') (ENNReal.natCast_ne_top _), one_pow]
    · rw [hkN, Nat.choose_eq_zero_of_lt (by omega), Nat.cast_zero, ENNReal.zero_div]
      exact zero_le

/-- The per-stratum floor escape, monotone in the count: a set with at least `b` units in stratum `s` escapes with at most
`((n_s − b)/n_s)^(f_s)`. -/
theorem work_escape_floor_le_of_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K : ℕ) (s : Fin m)
    (hs : (stratum σ s).Nonempty) {B : Finset (Fin n)} {b : ℕ} (hb : b ≤ (B.filter (σ · = s)).card) :
    (work σ w f K).escape B ≤ ((((stratum σ s).card - b : ℕ) : ℝ≥0∞) / ((stratum σ s).card : ℝ≥0∞)) ^ (f s) := by
  refine (work_escape_floor_le σ w f K s hs B).trans ?_
  gcongr

/-- **A stratum that does no work draws `min(n_s, f_s)`.** -/
theorem workK_rest {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} (K : ℕ) {s : Fin m} (hw : w s = 0) :
    workK σ w f K s = min (stratum σ s).card (f s) := by
  have h0 : (K * w s * (stratum σ s).card + totalWork σ w - 1) / totalWork σ w = 0 := by
    rw [hw, mul_zero, zero_mul, zero_add]
    rcases Nat.eq_zero_or_pos (totalWork σ w) with h | h
    · rw [h]
    · exact Nat.div_eq_of_lt (by omega)
  unfold workK workRule
  rw [h0, max_eq_left (Nat.zero_le _)]

/-- The count law's rule, the executable's `Flock.Draw.stratumK`: `C · size / population` rounded half up, at least one. -/
def countRule (C size population : ℕ) : ℕ := max 1 ((2 * C * size + population) / (2 * population))

/-- The count law's sizes over `σ`'s strata at count budget `C`: `main`'s stratified law, `stratumK C n_s N`. -/
def countK (σ : Fin n → Fin m) (C : ℕ) (s : Fin m) : ℕ := min (stratum σ s).card (countRule C (stratum σ s).card n)

theorem countK_le (σ : Fin n → Fin m) (C : ℕ) (s : Fin m) : countK σ C s ≤ (stratum σ s).card :=
  min_le_left _ _

/-- For a non-empty stratum and `C ≤ N`, `countK` is the count rule itself: the size `main`'s stratified law draws. -/
theorem countK_eq {σ : Fin n → Fin m} {C : ℕ} {s : Fin m} (hC : C ≤ n) (hs : (stratum σ s).Nonempty) :
    countK σ C s = countRule C (stratum σ s).card n := by
  have hn1 : 1 ≤ (stratum σ s).card := Finset.card_pos.2 hs
  have hnN : (stratum σ s).card ≤ n := by simpa using Finset.card_le_univ (stratum σ s)
  refine min_eq_right (max_le hn1 (Nat.lt_succ_iff.1 ((Nat.div_lt_iff_lt_mul (by omega)).2 ?_)))
  nlinarith

/-- **At the population, the count rule is the stratum's size**: `stratumK N size N = size` for `1 ≤ size`. -/
theorem countRule_self {size N : ℕ} (h1 : 1 ≤ size) (hN : 0 < N) : countRule N size N = size := by
  unfold countRule
  rw [Nat.mul_add_div (by omega : 0 < 2 * N), Nat.div_eq_of_lt (by omega : N < 2 * N), Nat.add_zero,
    max_eq_right h1]

/-- **The count law at `C = N` draws every unit of every stratum**: `countK σ N s` is stratum `s`'s size. -/
theorem countK_self (σ : Fin n → Fin m) (s : Fin m) : countK σ n s = (stratum σ s).card := by
  unfold countK
  rcases Nat.eq_zero_or_pos (stratum σ s).card with h0 | hpos
  · rw [h0]; exact Nat.zero_min _
  · have hn : 0 < n := hpos.trans_le (by simpa using Finset.card_le_univ (stratum σ s))
    rw [countRule_self hpos hn, min_self]

/-- **The count law at `C = N` has no sampling term** (a recursive inner session's full draw, `rec_live.full_draw`): its
draw is every unit (`stratified_draw_full`), so `miss K = 0` for `K ≥ 1`. -/
theorem countK_self_miss (σ : Fin n → Fin m) {K : ℕ} (hK : 1 ≤ K) :
    (stratified σ (countK σ n) (countK_le σ n)).miss K = 0 :=
  stratified_full_miss σ _ _ (countK_self σ) hK

theorem choose_mul_choose_succ_le (a n j : ℕ) (ha : a ≤ n) :
    a.choose (j + 1) * n.choose j ≤ a.choose j * n.choose (j + 1) := by
  refine Nat.le_of_mul_le_mul_right ?_ (Nat.succ_pos j)
  have h1 := Nat.choose_succ_right_eq a j
  have h2 := Nat.choose_succ_right_eq n j
  have hj : a - j ≤ n - j := by omega
  generalize a - j = x at h1 hj
  generalize n - j = y at h2 hj
  calc a.choose (j + 1) * n.choose j * (j + 1) = a.choose (j + 1) * (j + 1) * n.choose j := by ring
    _ = a.choose j * x * n.choose j := by rw [h1]
    _ ≤ a.choose j * y * n.choose j := by gcongr
    _ = a.choose j * (n.choose j * y) := by ring
    _ = a.choose j * (n.choose (j + 1) * (j + 1)) := by rw [h2]
    _ = a.choose j * n.choose (j + 1) * (j + 1) := by ring

theorem choose_mul_le_of_le {a n k : ℕ} (ha : a ≤ n) : ∀ {k'}, k ≤ k' → k' ≤ n →
    a.choose k' * n.choose k ≤ a.choose k * n.choose k'
  | k', hkk, hk' => by
    induction k', hkk using Nat.le_induction with
    | base => exact le_rfl
    | succ j hkj ih =>
      have hjn : j ≤ n := by omega
      have ih := ih hjn
      have hpos : 0 < n.choose j := Nat.choose_pos hjn
      refine Nat.le_of_mul_le_mul_right ?_ hpos
      calc a.choose (j + 1) * n.choose k * n.choose j = (a.choose (j + 1) * n.choose j) * n.choose k := by ring
        _ ≤ (a.choose j * n.choose (j + 1)) * n.choose k := Nat.mul_le_mul_right _ (choose_mul_choose_succ_le a n j ha)
        _ = (a.choose j * n.choose k) * n.choose (j + 1) := by ring
        _ ≤ (a.choose k * n.choose j) * n.choose (j + 1) := Nat.mul_le_mul_right _ ih
        _ = a.choose k * n.choose (j + 1) * n.choose j := by ring

/-- **The hypergeometric escape falls as the draw grows**: `C(a, k')/C(n, k') ≤ C(a, k)/C(n, k)` for `a ≤ n` and
`k ≤ k' ≤ n`. -/
theorem choose_div_anti {a n k k' : ℕ} (ha : a ≤ n) (hkk : k ≤ k') (hk' : k' ≤ n) :
    (a.choose k' : ℝ≥0∞) / (n.choose k' : ℝ≥0∞) ≤ (a.choose k : ℝ≥0∞) / (n.choose k : ℝ≥0∞) := by
  have hk0 : (n.choose k : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos (hkk.trans hk')).ne'
  have hk'0 : (n.choose k' : ℝ≥0∞) ≠ 0 := by exact_mod_cast (Nat.choose_pos hk').ne'
  rw [ENNReal.div_le_iff hk'0 (ENNReal.natCast_ne_top _), ← ENNReal.mul_div_right_comm,
    ENNReal.le_div_iff_mul_le (Or.inl hk0) (Or.inl (ENNReal.natCast_ne_top _))]
  exact_mod_cast choose_mul_le_of_le ha hkk hk'

/-- **Floors at the count law's sizes keep its guarantee.** When every stratum that does no work has a floor of at least
the count law's size at `C`, a set escapes the work law at most as often as its units in those strata escape the
stratified law at `C`. -/
theorem work_escape_le_count (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (K C : ℕ)
    (hf : ∀ s, w s = 0 → countK σ C s ≤ f s) (B : Finset (Fin n)) :
    (work σ w f K).escape B ≤ (stratified σ (countK σ C) (countK_le σ C)).escape (B.filter fun u => w (σ u) = 0) := by
  classical
  rw [stratified_escape, stratified_escape]
  refine Finset.prod_le_prod fun s _ => ?_
  by_cases hw : w s = 0
  · have hm : ((B.filter fun u => w (σ u) = 0).filter (σ · = s)).card = (B.filter (σ · = s)).card := by
      congr 1
      ext u
      simp only [Finset.mem_filter]
      constructor
      · rintro ⟨⟨hu, _⟩, hs⟩
        exact ⟨hu, hs⟩
      · rintro ⟨hu, hs⟩
        exact ⟨⟨hu, by rw [hs]; exact hw⟩, hs⟩
    rw [hm]
    refine choose_div_anti (Nat.sub_le _ _) ?_ (workK_le σ w f K s)
    rw [workK_rest K hw]
    exact le_min (countK_le σ C s) (hf s hw)
  · have hm : ((B.filter fun u => w (σ u) = 0).filter (σ · = s)).card = 0 := by
      rw [Finset.card_eq_zero, Finset.filter_eq_empty_iff]
      intro u hu hs
      exact hw (hs ▸ (Finset.mem_filter.1 hu).2)
    rw [hm, Nat.sub_zero, ENNReal.div_self (by exact_mod_cast (Nat.choose_pos (countK_le σ C s)).ne')
      (ENNReal.natCast_ne_top _)]
    exact choose_div_le_one _ _ _

/-! ## Closure draws: the seam -/

/-- A law that proves more units than it draws: the same coins, each draw widened by `f`. Closure draws are `f S = S`
with the units `S`'s members read. -/
def widen (L : Law n) (f : Finset (Fin n) → Finset (Fin n)) : Law n where
  Ω := L.Ω
  draw ω := f (L.draw ω)

/-- **Widening only helps**: when `f` keeps every drawn unit, the widened law escapes at most as often, so every escape
bound of `L` (the work bound, the floor) holds of it. -/
theorem escape_widen_le (L : Law n) {f : Finset (Fin n) → Finset (Fin n)} (hf : ∀ S, S ⊆ f S) (B : Finset (Fin n)) :
    (L.widen f).escape B ≤ L.escape B :=
  prCoin_mono fun _ hd => hd.mono_left (hf _)

/-! ## The sizing of record -/

theorem sizing_iff (K : ℕ) :
    ((999 : ℝ≥0∞) / 1000) ^ K ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ ↔ 999 ^ K * 2 ^ 40 ≤ 1000 ^ K := by
  rw [div_eq_mul_inv, mul_pow, ← ENNReal.inv_pow, ← div_eq_mul_inv,
    ENNReal.div_le_iff (pow_ne_zero _ (by norm_num)) (ENNReal.pow_ne_top (by norm_num)),
    ← ENNReal.div_eq_inv_mul, ENNReal.le_div_iff_mul_le (Or.inl (by norm_num)) (Or.inl (by norm_num))]
  norm_cast

/-- **The sizing of record** (`δ = 2^-40`, `ε = 0.1%`): `(1 − 1/1000)^27713 ≤ 2^-40`. -/
theorem record_sizing : ((999 : ℝ≥0∞) / 1000) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ :=
  (sizing_iff 27713).2 (by decide +kernel)

/-- One draw fewer does not meet it under the work bound: `(1 − 1/1000)^27712 > 2^-40`. -/
theorem record_sizing_one_fewer : ¬ ((999 : ℝ≥0∞) / 1000) ^ 27712 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ :=
  fun h => absurd ((sizing_iff 27712).1 h) (by decide +kernel)

/-- Units doing at least 0.1% of the work (`W ≤ 1000 · W_B`) escape `K = 27,713` draws with at most `2^-40`. -/
theorem work_escape_of_record (σ : Fin n → Fin m) (w : Fin m → ℕ) (f : Fin m → ℕ) (hf : ∀ s, 1 ≤ f s)
    (hW : 0 < totalWork σ w) {B : Finset (Fin n)} (hB : totalWork σ w ≤ 1000 * workOf σ w B) :
    (work σ w f 27713).escape B ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ := by
  refine (work_escape_le σ w f 27713 hf hW B).trans (le_trans (pow_le_pow_left₀ zero_le ?_ 27713) record_sizing)
  have hWt : (totalWork σ w : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hW0 : (totalWork σ w : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW.ne'
  rw [ENNReal.div_le_iff hW0 hWt, ← ENNReal.mul_div_right_comm, ENNReal.le_div_iff_mul_le (Or.inl (by norm_num))
    (Or.inl (by norm_num))]
  exact_mod_cast (show (totalWork σ w - workOf σ w B) * 1000 ≤ 999 * totalWork σ w by omega)

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {f : Fin m → ℕ} {K : ℕ}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- **The work audit** (oracle layer): the audit accepts while its wrong units do at least `T` of the work with
probability at most `((W − T)/W)^K + ε_ks + δ_link`. -/
theorem audit_work (A : Analysis (Law.work σ w f K) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ Law.workOf σ w (A.wrong (A.committedOf σ')))
        (audit (Law.work σ w f K) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => T ≤ Law.workOf σ w B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.work_escape_le_of_le σ w f K hf hW hB

/-- **The work audit** (compiled layer, randomized extractor): at most
`((W − T)/W)^K + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work (A : ExtractionAnalysis (Law.work σ w f K) Reg session)
    {εks δlink : (R : Reg) → Cont (Law.work σ w f K) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ Law.workOf σ w (A.wrong (A.committedOf σ')))
        (audit (Law.work σ w f K) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ Law.workOf σ w (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.workOf σ w (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (Law.work_escape_le_of_le σ w f K hf hW hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **The work audit at the sizing of record**: with `K = 27,713`, the audit accepts while its wrong units do at least
0.1% of the work with probability at most `2^-40 + ε_ks + δ_link`. -/
theorem audit_work_of_record (A : Analysis (Law.work σ w f 27713) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hf : ∀ s, 1 ≤ f s) (hW : 0 < Law.totalWork σ w)
    (σ' : Strategy (audit (Law.work σ w f 27713) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.totalWork σ w ≤ 1000 * Law.workOf σ w (A.wrong (A.committedOf σ')))
        (audit (Law.work σ w f 27713) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => Law.totalWork σ w ≤ 1000 * Law.workOf σ w B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.work_escape_of_record σ w f hf hW hB

/-- **The integrity floor, audited** (oracle layer): a wholly wrong non-empty stratum, one that does no PoUW work
included, is caught: the audit accepts while all of it is wrong with probability at most `ε_ks + δ_link`. -/
theorem audit_work_whole_stratum (A : Analysis (Law.work σ w f K) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (s : Fin m) (hf : 1 ≤ f s)
    (hs : (Law.stratum σ s).Nonempty) (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.stratum σ s ⊆ A.wrong (A.committedOf σ'))
        (audit (Law.work σ w f K) Reg session) σ' ≤ εks + δlink :=
  audit_whole_stratum A hks hlink s (Law.one_le_workK hf hs) σ'

/-- **The integrity floor, audited** (compiled layer, randomized extractor): at most `ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work_whole_stratum (A : ExtractionAnalysis (Law.work σ w f K) Reg session)
    {εks δlink : (R : Reg) → Cont (Law.work σ w f K) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (s : Fin m) (hf : 1 ≤ f s)
    (hs : (Law.stratum σ s).Nonempty) (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.stratum σ s ⊆ A.wrong (A.committedOf σ'))
        (audit (Law.work σ w f K) Reg session) σ' ≤ εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') :=
  extraction_audit_whole_stratum A hks hlink s (Law.one_le_workK hf hs) σ'

/-- **The per-stratum floor, audited** (oracle layer): the audit accepts while at least `b` units of a non-empty stratum
`s` are wrong with probability at most `((n_s − b)/n_s)^(f_s) + ε_ks + δ_link`. -/
theorem audit_work_floor (A : Analysis (Law.work σ w f K) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (s : Fin m) (hs : (Law.stratum σ s).Nonempty) (b : ℕ)
    (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ b ≤ ((A.wrong (A.committedOf σ')).filter (σ · = s)).card)
        (audit (Law.work σ w f K) Reg session) σ' ≤
      ((((Law.stratum σ s).card - b : ℕ) : ℝ≥0∞) / ((Law.stratum σ s).card : ℝ≥0∞)) ^ (f s) + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => b ≤ (B.filter (σ · = s)).card) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.work_escape_floor_le_of_le σ w f K s hs hB

/-- **The per-stratum floor, audited** (compiled layer, randomized extractor): at most
`((n_s − b)/n_s)^(f_s) + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work_floor (A : ExtractionAnalysis (Law.work σ w f K) Reg session)
    {εks δlink : (R : Reg) → Cont (Law.work σ w f K) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (s : Fin m) (hs : (Law.stratum σ s).Nonempty) (b : ℕ)
    (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ b ≤ ((A.wrong (A.committedOf σ')).filter (σ · = s)).card)
        (audit (Law.work σ w f K) Reg session) σ' ≤
      ((((Law.stratum σ s).card - b : ℕ) : ℝ≥0∞) / ((Law.stratum σ s).card : ℝ≥0∞)) ^ (f s) +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => b ≤ ((A.wrong X).filter (σ · = s)).card) σ').trans ?_
  gcongr
  by_cases hB : b ≤ ((A.wrong (A.committedOf σ')).filter (σ · = s)).card
  · exact (prCoin_mono fun ω h => h.2).trans (Law.work_escape_floor_le_of_le σ w f K s hs hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **Floors at the count law's sizes keep its profile, audited** (oracle layer): when every stratum that does no work has
a floor of at least the count law's size at `C`, for any family `𝓑`, the audit accepts while the wrong units in those
strata form a set in `𝓑` with probability at most the stratified law's `sup_{B ∈ 𝓑} escape B` at `C`, plus
`ε_ks + δ_link`. -/
theorem audit_work_rest {C : ℕ} (hf : ∀ s, w s = 0 → Law.countK σ C s ≤ f s)
    (A : Analysis (Law.work σ w f K) Reg session) {εks δlink : ℝ≥0∞} (hks : A.KnowledgeSound εks)
    (hlink : A.LinkSound δlink) (𝓑 : Finset (Fin n) → Prop) (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ 𝓑 ((A.wrong (A.committedOf σ')).filter fun u => w (σ u) = 0))
        (audit (Law.work σ w f K) Reg session) σ' ≤
      (⨆ (B : Finset (Fin n)) (_ : 𝓑 B), (Law.stratified σ (Law.countK σ C) (Law.countK_le σ C)).escape B) +
        εks + δlink := by
  refine (audit_profile A hks hlink (fun B => 𝓑 (B.filter fun u => w (σ u) = 0)) σ').trans
    (add_le_add (add_le_add ?_ le_rfl) le_rfl)
  exact iSup₂_le fun B hB => (Law.work_escape_le_count σ w f K C hf B).trans
    (le_iSup₂_of_le (f := fun (B : Finset (Fin n)) (_ : 𝓑 B) =>
      (Law.stratified σ (Law.countK σ C) (Law.countK_le σ C)).escape B) _ hB le_rfl)

/-- **Floors at the count law's sizes keep its profile, audited** (compiled layer, randomized extractor): the same with
`ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work_rest {C : ℕ} (hf : ∀ s, w s = 0 → Law.countK σ C s ≤ f s)
    (A : ExtractionAnalysis (Law.work σ w f K) Reg session)
    {εks δlink : (R : Reg) → Cont (Law.work σ w f K) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (𝓑 : Finset (Fin n) → Prop)
    (σ' : Strategy (audit (Law.work σ w f K) Reg session)) :
    prob (fun o => o.1 = true ∧ 𝓑 ((A.wrong (A.committedOf σ')).filter fun u => w (σ u) = 0))
        (audit (Law.work σ w f K) Reg session) σ' ≤
      (⨆ (B : Finset (Fin n)) (_ : 𝓑 B), (Law.stratified σ (Law.countK σ C) (Law.countK_le σ C)).escape B) +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => 𝓑 ((A.wrong X).filter fun u => w (σ u) = 0)) σ').trans ?_
  gcongr
  by_cases hB : 𝓑 ((A.wrong (A.committedOf σ')).filter fun u => w (σ u) = 0)
  · exact (prCoin_mono fun ω h => h.2).trans ((Law.work_escape_le_count σ w f K C hf _).trans
      (le_iSup₂_of_le (f := fun (B : Finset (Fin n)) (_ : 𝓑 B) =>
        (Law.stratified σ (Law.countK σ C) (Law.countK_le σ C)).escape B) _ hB le_rfl))
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

end Audits

end FlockSoundness.Audit

theorem FlockSoundness.Audit.SecurityProofs.Law.RecordSizing : FlockSoundness.Audit.Guarantees.Law.RecordSizing := @FlockSoundness.Audit.Law.record_sizing

theorem FlockSoundness.Audit.SecurityProofs.Law.RecordSizingOneFewer : FlockSoundness.Audit.Guarantees.Law.RecordSizingOneFewer := @FlockSoundness.Audit.Law.record_sizing_one_fewer
