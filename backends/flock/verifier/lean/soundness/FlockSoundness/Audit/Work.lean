import FlockSoundness.Audit.Stratified
import Mathlib.Analysis.MeanInequalities

/-!
# The work law

The draw law's second version (Daniel, Sep 28): the stratified law's strata, with each stratum's draw sized by the PoUW
work its units do instead of by their number. Stratum `s` has `n_s` units, each doing work `w_s` (zero for units that do no
PoUW work: dequantization, the quantizer, attention), and the population does `W = Σ_s w_s n_s`. Stratum `s` draws
`k_s = min(n_s, max(1, ⌈K w_s n_s / W⌉))` (`Law.workRule`, the executable's `Flock.Draw.workK`).

The work law is `Law.stratified` at those sizes (`Law.work`), so every theorem of the stratified law and of the one-stage
audit holds of it as stated: `stratified_escape`, `stratified_escape_floor`, `audit_whole_stratum`,
`extraction_audit_whole_stratum`, `audit_profile`, `audit_count`, `audit_drawn` and the Flock instantiations. What is new:

* `one_le_workK`: **the integrity floor**. Every non-empty stratum draws at least one unit, so a wholly wrong stratum is
  caught whatever its work (`audit_work_whole_stratum`, `extraction_audit_work_whole_stratum`).
* `workK_ge`: **draws sized by work**. A stratum draws at least `K w_s n_s / W` units, or all of them.
* `sum_workK_le`: at most `K` units plus one per stratum are drawn.
* `work_escape_le`: **the work bound**. A set of units doing `W_B` of the work escapes with probability at most
  `((W − W_B)/W)^K`, so wrong units doing a fraction `ε` of the work escape with at most `(1 − ε)^K`.
* `audit_work`, `extraction_audit_work`: the audit accepts while its wrong units do at least `T` of the work with
  probability at most `((W − T)/W)^K + ε_ks + δ_link`.
* `record_sizing`, `audit_work_of_record`: at the sizing of record (`δ = 2^-40`, `ε = 0.1%`), `K = 27,713` suffices;
  `record_sizing_one_fewer`: `K = 27,712` does not, under this bound.

Closure draws (a drawn unit proved with the units it reads) are not part of this law. `Law.widen` proves more units than a
law draws on the same coins, and `escape_widen_le` carries every escape bound to it.
-/

namespace FlockSoundness.Audit

open Game Finset
open scoped ENNReal NNReal

namespace Law

variable {n m : ℕ}

/-- The work rule, the executable's `Flock.Draw.workK`: `⌈K · work · size / total⌉`, at least one and at most `size`. -/
def workRule (K work size total : ℕ) : ℕ := min size (max 1 ((K * work * size + total - 1) / total))

/-- The work of a set of units, each doing its stratum's `w (σ u)`. -/
def workOf (σ : Fin n → Fin m) (w : Fin m → ℕ) (B : Finset (Fin n)) : ℕ := ∑ u ∈ B, w (σ u)

/-- The population's work, `W`. -/
abbrev totalWork (σ : Fin n → Fin m) (w : Fin m → ℕ) : ℕ := workOf σ w univ

/-- Stratum `s`'s draw under the work law. -/
def workK (σ : Fin n → Fin m) (w : Fin m → ℕ) (K : ℕ) (s : Fin m) : ℕ :=
  workRule K (w s) (stratum σ s).card (totalWork σ w)

theorem workK_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (K : ℕ) (s : Fin m) :
    workK σ w K s ≤ (stratum σ s).card :=
  min_le_left _ _

/-- **The work law**: independent uniform subsets inside the strata of `σ`, stratum `s` drawing `workK σ w K s`. -/
abbrev work (σ : Fin n → Fin m) (w : Fin m → ℕ) (K : ℕ) : Law n :=
  stratified σ (workK σ w K) (workK_le σ w K)

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

/-- **The integrity floor**: every non-empty stratum draws at least one unit, whatever its work. -/
theorem one_le_workK {σ : Fin n → Fin m} {w : Fin m → ℕ} {K : ℕ} {s : Fin m} (h : (stratum σ s).Nonempty) :
    1 ≤ workK σ w K s :=
  le_min (Finset.card_pos.2 h) (le_max_left _ _)

theorem le_mul_ceil_div (x W : ℕ) (hW : 0 < W) : x ≤ W * ((x + W - 1) / W) := by
  have h1 := Nat.div_add_mod (x + W - 1) W
  have h2 := Nat.mod_lt (x + W - 1) hW
  generalize W * ((x + W - 1) / W) = p at h1 ⊢
  generalize (x + W - 1) % W = r at h1 h2
  omega

/-- **Draws sized by work**: stratum `s` draws all its units, or at least `K w_s n_s / W` of them. -/
theorem workK_ge {σ : Fin n → Fin m} {w : Fin m → ℕ} {K : ℕ} (s : Fin m) (hW : 0 < totalWork σ w) :
    workK σ w K s = (stratum σ s).card ∨
      K * w s * (stratum σ s).card ≤ totalWork σ w * workK σ w K s := by
  unfold workK workRule
  rcases le_total (stratum σ s).card (max 1 ((K * w s * (stratum σ s).card + totalWork σ w - 1) / totalWork σ w))
    with h | h
  · exact Or.inl (min_eq_left h)
  · refine Or.inr ?_
    rw [min_eq_right h]
    exact (le_mul_ceil_div _ _ hW).trans (Nat.mul_le_mul_left _ (le_max_right _ _))

theorem mul_workRule_le (K work size total : ℕ) : total * workRule K work size total ≤ K * work * size + total := by
  unfold workRule
  have hq := Nat.mul_div_le (K * work * size + total - 1) total
  refine (Nat.mul_le_mul_left total (min_le_right _ _)).trans ?_
  rcases le_total 1 ((K * work * size + total - 1) / total) with h | h
  · rw [max_eq_right h]
    have := Nat.mul_comm total ((K * work * size + total - 1) / total)
    omega
  · rw [max_eq_left h, mul_one]
    omega

/-- **The draw's size**: at most `K` units, plus one per stratum. -/
theorem sum_workK_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w) :
    ∑ s, workK σ w K s ≤ K + m := by
  refine Nat.le_of_mul_le_mul_left ?_ hW
  rw [Finset.mul_sum]
  calc ∑ s, totalWork σ w * workK σ w K s ≤ ∑ s, (K * w s * (stratum σ s).card + totalWork σ w) :=
        Finset.sum_le_sum fun s _ => mul_workRule_le _ _ _ _
    _ = K * ∑ s, w s * (stratum σ s).card + m * totalWork σ w := by
        rw [Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          smul_eq_mul]
        simp only [mul_assoc]
    _ = totalWork σ w * (K + m) := by rw [← totalWork_eq]; ring

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
theorem work_escape_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w) (B : Finset (Fin n)) :
    (work σ w K).escape B ≤ (((totalWork σ w - workOf σ w B : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  classical
  set W := totalWork σ w with hWdef
  set N : Fin m → ℕ := fun s => (stratum σ s).card with hNdef
  set M : Fin m → ℕ := fun s => (B.filter (σ · = s)).card with hMdef
  set k : Fin m → ℕ := workK σ w K with hkdef
  have hMN : ∀ s, M s ≤ N s := fun s => card_le_card (filter_subset_filter _ (subset_univ B))
  have hkN : ∀ s, k s ≤ N s := workK_le σ w K
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
    have hk1 : 1 ≤ k s := one_le_workK (Finset.card_pos.1 hpos)
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
  calc ∏ s, (((stratum σ s).card - (B.filter (σ · = s)).card).choose (workK σ w K s) : ℝ≥0∞) /
        ((stratum σ s).card.choose (workK σ w K s) : ℝ≥0∞)
      = ((∏ s, ((N s - M s).choose (k s) : ℝ≥0) / ((N s).choose (k s) : ℝ≥0) : ℝ≥0) : ℝ≥0∞) := by
        rw [ENNReal.ofNNReal_finsetProd]
        refine Finset.prod_congr rfl fun s _ => ?_
        rw [ENNReal.coe_div (hC s)]
        simp [hNdef, hMdef, hkdef]
    _ ≤ (((((W - workOf σ w B : ℕ) : ℝ≥0) / (W : ℝ≥0)) ^ K : ℝ≥0) : ℝ≥0∞) := ENNReal.coe_le_coe.2 hprod
    _ = _ := by rw [ENNReal.coe_pow, ENNReal.coe_div hW0, ENNReal.coe_natCast, ENNReal.coe_natCast]

/-- The work bound, monotone in the threshold: units doing at least `T` of the work escape with at most
`((W − T)/W)^K`. -/
theorem work_escape_le_of_le (σ : Fin n → Fin m) (w : Fin m → ℕ) (K : ℕ) (hW : 0 < totalWork σ w)
    {B : Finset (Fin n)} {T : ℕ} (hT : T ≤ workOf σ w B) :
    (work σ w K).escape B ≤ (((totalWork σ w - T : ℕ) : ℝ≥0∞) / (totalWork σ w : ℝ≥0∞)) ^ K := by
  refine (work_escape_le σ w K hW B).trans ?_
  gcongr

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

/-- **The sizing of record** (Daniel, Sep 29: `δ = 2^-40`, `ε = 0.1%`): `(1 − 1/1000)^27713 ≤ 2^-40`. -/
theorem record_sizing : ((999 : ℝ≥0∞) / 1000) ^ 27713 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ :=
  (sizing_iff 27713).2 (by decide +kernel)

/-- One draw fewer does not meet it under the work bound: `(1 − 1/1000)^27712 > 2^-40`. -/
theorem record_sizing_one_fewer : ¬ ((999 : ℝ≥0∞) / 1000) ^ 27712 ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ :=
  fun h => absurd ((sizing_iff 27712).1 h) (by decide +kernel)

/-- Units doing at least 0.1% of the work (`W ≤ 1000 · W_B`) escape `K = 27,713` draws with at most `2^-40`. -/
theorem work_escape_of_record (σ : Fin n → Fin m) (w : Fin m → ℕ) (hW : 0 < totalWork σ w) {B : Finset (Fin n)}
    (hB : totalWork σ w ≤ 1000 * workOf σ w B) :
    (work σ w 27713).escape B ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ := by
  refine (work_escape_le σ w 27713 hW B).trans (le_trans (pow_le_pow_left₀ zero_le ?_ 27713) record_sizing)
  have hWt : (totalWork σ w : ℝ≥0∞) ≠ ⊤ := ENNReal.natCast_ne_top _
  have hW0 : (totalWork σ w : ℝ≥0∞) ≠ 0 := by exact_mod_cast hW.ne'
  rw [ENNReal.div_le_iff hW0 hWt, ← ENNReal.mul_div_right_comm, ENNReal.le_div_iff_mul_le (Or.inl (by norm_num))
    (Or.inl (by norm_num))]
  exact_mod_cast (show (totalWork σ w - workOf σ w B) * 1000 ≤ 999 * totalWork σ w by omega)

end Law

section Audits

variable {n m : ℕ} {σ : Fin n → Fin m} {w : Fin m → ℕ} {K : ℕ}
  {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- **The work audit** (oracle layer): the audit accepts while its wrong units do at least `T` of the work with
probability at most `((W − T)/W)^K + ε_ks + δ_link`. -/
theorem audit_work (A : Analysis (Law.work σ w K) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit (Law.work σ w K) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ Law.workOf σ w (A.wrong (A.committedOf σ')))
        (audit (Law.work σ w K) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => T ≤ Law.workOf σ w B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.work_escape_le_of_le σ w K hW hB

/-- **The work audit** (compiled layer, randomized extractor): at most
`((W − T)/W)^K + ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work (A : ExtractionAnalysis (Law.work σ w K) Reg session)
    {εks δlink : (R : Reg) → Cont (Law.work σ w K) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w) (T : ℕ)
    (σ' : Strategy (audit (Law.work σ w K) Reg session)) :
    prob (fun o => o.1 = true ∧ T ≤ Law.workOf σ w (A.wrong (A.committedOf σ')))
        (audit (Law.work σ w K) Reg session) σ' ≤
      (((Law.totalWork σ w - T : ℕ) : ℝ≥0∞) / (Law.totalWork σ w : ℝ≥0∞)) ^ K +
        εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') := by
  refine (extraction_audit_le A hks hlink (fun X _ => T ≤ Law.workOf σ w (A.wrong X)) σ').trans ?_
  gcongr
  by_cases hB : T ≤ Law.workOf σ w (A.wrong (A.committedOf σ'))
  · exact (prCoin_mono fun ω h => h.2).trans (Law.work_escape_le_of_le σ w K hW hB)
  · exact (prCoin_mono (q := fun _ => False) fun ω h => hB h.1).trans (prCoin_false.le.trans zero_le)

/-- **The work audit at the sizing of record**: with `K = 27,713`, the audit accepts while its wrong units do at least
0.1% of the work with probability at most `2^-40 + ε_ks + δ_link`. -/
theorem audit_work_of_record (A : Analysis (Law.work σ w 27713) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (hW : 0 < Law.totalWork σ w)
    (σ' : Strategy (audit (Law.work σ w 27713) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.totalWork σ w ≤ 1000 * Law.workOf σ w (A.wrong (A.committedOf σ')))
        (audit (Law.work σ w 27713) Reg session) σ' ≤ ((2 : ℝ≥0∞) ^ 40)⁻¹ + εks + δlink := by
  refine (audit_profile A hks hlink (fun B => Law.totalWork σ w ≤ 1000 * Law.workOf σ w B) σ').trans ?_
  gcongr
  exact iSup₂_le fun B hB => Law.work_escape_of_record σ w hW hB

/-- **The integrity floor, audited** (oracle layer): a wholly wrong non-empty stratum, one that does no PoUW work
included, is caught: the audit accepts while all of it is wrong with probability at most `ε_ks + δ_link`. -/
theorem audit_work_whole_stratum (A : Analysis (Law.work σ w K) Reg session) {εks δlink : ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (s : Fin m) (hs : (Law.stratum σ s).Nonempty)
    (σ' : Strategy (audit (Law.work σ w K) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.stratum σ s ⊆ A.wrong (A.committedOf σ'))
        (audit (Law.work σ w K) Reg session) σ' ≤ εks + δlink :=
  audit_whole_stratum A hks hlink s (Law.one_le_workK hs) σ'

/-- **The integrity floor, audited** (compiled layer, randomized extractor): at most `ε_ks(σ) + δ_link(σ)`. -/
theorem extraction_audit_work_whole_stratum (A : ExtractionAnalysis (Law.work σ w K) Reg session)
    {εks δlink : (R : Reg) → Cont (Law.work σ w K) Reg session R → ℝ≥0∞}
    (hks : A.KnowledgeSound εks) (hlink : A.LinkSound δlink) (s : Fin m) (hs : (Law.stratum σ s).Nonempty)
    (σ' : Strategy (audit (Law.work σ w K) Reg session)) :
    prob (fun o => o.1 = true ∧ Law.stratum σ s ⊆ A.wrong (A.committedOf σ'))
        (audit (Law.work σ w K) Reg session) σ' ≤ εks (reg σ') (cont σ') + δlink (reg σ') (cont σ') :=
  extraction_audit_whole_stratum A hks hlink s (Law.one_le_workK hs) σ'

end Audits

end FlockSoundness.Audit
