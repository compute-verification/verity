import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Unequal.Bounds

/-!
# Optimal partitioning against free compute: the proofs

The proofs of the free-compute statements of `Partitioning.Guarantees`.
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs

open Finset Game
open scoped ENNReal

theorem incl_le_one {n : ℕ} (L : Law n) (u : Fin n) : L.incl u ≤ 1 := by
  classical
  unfold Verity.Law.incl prCoin
  refine (avg_mono fun ω => ?_).trans (avg_const (C := L.Ω) 1).le
  split_ifs <;> simp

/-- `C(n, j) (M − s)^j ≤ C(n − s, j) M^j` for `j ≤ K` and `s + K ≤ n`, where `M = n − K + 1`: each of the `j` factors
`(n − s − i)/(n − i)` of `C(n − s, j)/C(n, j)` is at least `1 − s/M`. -/
theorem choose_mul_pow_le (n K s : ℕ) (hs : s + K ≤ n) : ∀ j, j ≤ K →
    (n.choose j : ℝ) * (((n : ℝ) - K + 1) - s) ^ j ≤ ((n - s).choose j : ℝ) * ((n : ℝ) - K + 1) ^ j := by
  intro j
  induction j with
  | zero => intro _; simp
  | succ j ih =>
    intro hj
    have ih := ih (by omega)
    have e1 : (n.choose (j + 1) : ℝ) * ((j : ℝ) + 1) = n.choose j * ((n : ℝ) - j) := by
      have h := Nat.choose_succ_right_eq n j
      have h' : (n.choose (j + 1) : ℝ) * ((j + 1 : ℕ) : ℝ) = n.choose j * ((n - j : ℕ) : ℝ) := by exact_mod_cast h
      rw [Nat.cast_sub (by omega)] at h'
      push_cast at h'
      exact h'
    have e2 : ((n - s).choose (j + 1) : ℝ) * ((j : ℝ) + 1) = (n - s).choose j * ((n : ℝ) - s - j) := by
      have h := Nat.choose_succ_right_eq (n - s) j
      have h' : ((n - s).choose (j + 1) : ℝ) * ((j + 1 : ℕ) : ℝ) = (n - s).choose j * ((n - s - j : ℕ) : ℝ) := by
        exact_mod_cast h
      rw [Nat.cast_sub (by omega : j ≤ n - s), Nat.cast_sub (by omega : s ≤ n)] at h'
      push_cast at h'
      exact h'
    set M : ℝ := (n : ℝ) - K + 1 with hM
    have hsK : (s : ℝ) + K ≤ n := by exact_mod_cast hs
    have hjK : (j : ℝ) + 1 ≤ K := by exact_mod_cast hj
    have hMs : 0 ≤ M - s := by simp only [M]; linarith
    have hfac : ((n : ℝ) - j) * (M - s) ≤ ((n : ℝ) - s - j) * M := by
      have h : ((n : ℝ) - s - j) * M - ((n : ℝ) - j) * (M - s) = s * ((K : ℝ) - 1 - j) := by simp only [M]; ring
      have : 0 ≤ (s : ℝ) * ((K : ℝ) - 1 - j) := mul_nonneg (by positivity) (by linarith)
      linarith
    have hpos1 : 0 ≤ ((n : ℝ) - j) * (M - s) := mul_nonneg (by linarith) hMs
    have hpos2 : 0 ≤ ((n - s).choose j : ℝ) * M ^ j := mul_nonneg (by positivity) (pow_nonneg (by linarith) _)
    have key := mul_le_mul ih hfac hpos1 hpos2
    have hj1 : (0 : ℝ) < j + 1 := by positivity
    refine le_of_mul_le_mul_right ?_ hj1
    calc (n.choose (j + 1) : ℝ) * (M - s) ^ (j + 1) * (j + 1)
        = (n.choose (j + 1) * ((j : ℝ) + 1)) * (M - s) ^ (j + 1) := by ring
      _ = (n.choose j * (M - s) ^ j) * (((n : ℝ) - j) * (M - s)) := by rw [e1]; ring
      _ ≤ ((n - s).choose j * M ^ j) * (((n : ℝ) - s - j) * M) := key
      _ = ((n - s).choose (j + 1) * ((j : ℝ) + 1)) * M ^ (j + 1) := by rw [e2]; ring
      _ = ((n - s).choose (j + 1) : ℝ) * M ^ (j + 1) * (j + 1) := by ring

/-- The tangent line of `t ↦ max(0, t)^K` at `t₀ > 0`, written `t₀^K (1 + K (t/t₀ − 1))`: Bernoulli's inequality for
`t ≥ 0`, and a nonpositive line for `t < 0`. -/
theorem tangent_le (K : ℕ) (hK : 0 < K) {t₀ : ℝ} (ht₀ : 0 < t₀) (t : ℝ) :
    t₀ ^ K * (1 + K * (t / t₀ - 1)) ≤ max 0 t ^ K := by
  rcases le_or_gt 0 t with ht | ht
  · rw [max_eq_right ht]
    have hb := one_add_mul_le_pow (show (-2 : ℝ) ≤ t / t₀ - 1 by
      have : 0 ≤ t / t₀ := div_nonneg ht ht₀.le
      linarith) K
    rw [show 1 + (t / t₀ - 1) = t / t₀ by ring, div_pow] at hb
    calc t₀ ^ K * (1 + K * (t / t₀ - 1)) ≤ t₀ ^ K * (t ^ K / t₀ ^ K) := mul_le_mul_of_nonneg_left hb (by positivity)
      _ = t ^ K := by field_simp
  · rw [max_eq_left ht.le, zero_pow hK.ne']
    have hx : t / t₀ - 1 ≤ -1 := by
      have := div_neg_of_neg_of_pos ht ht₀
      linarith
    have hK1 : (1 : ℝ) ≤ K := by exact_mod_cast hK
    have := mul_le_mul_of_nonneg_left hx (Nat.cast_nonneg K : (0 : ℝ) ≤ K)
    exact mul_nonpos_of_nonneg_of_nonpos (by positivity) (by linarith)

theorem ConverseExact : Guarantees.ConverseExact := by
  classical
  intro n K L δ hK hKn hδ
  have hKn' : (K : ℝ) ≤ n := by exact_mod_cast hKn
  set M : ℝ := (n : ℝ) - K + 1 with hMdef
  have hM : 0 < M := by linarith
  have hc : (0 : ℝ) < n.choose K := by exact_mod_cast Nat.choose_pos hKn
  set S : L.Ω → ℕ := fun ω => (L.draw ω).card
  set c : ℝ := (Fintype.card L.Ω : ℝ)⁻¹
  have hcard : (0 : ℝ) < Fintype.card L.Ω := by exact_mod_cast Fintype.card_pos
  have hc0 : 0 ≤ c := by positivity
  have hcsum : ∑ _ω : L.Ω, c = 1 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact mul_inv_cancel₀ hcard.ne'
  have havg_toReal : ∀ g : L.Ω → ℝ≥0∞, (∀ ω, g ω ≠ ⊤) → (avg g).toReal = ∑ ω, c * (g ω).toReal := by
    intro g hg
    unfold avg
    rw [ENNReal.toReal_sum fun ω _ => ENNReal.mul_ne_top (ENNReal.inv_ne_top.2 (by simp)) (hg ω)]
    refine Finset.sum_congr rfl fun ω _ => ?_
    rw [ENNReal.toReal_mul, ENNReal.toReal_inv, ENNReal.toReal_natCast]
  have hmean : meanDraws L = ∑ ω, c * (S ω : ℝ) := by
    unfold meanDraws
    rw [← ENNReal.toReal_sum fun u _ => ne_top_of_le_ne_top ENNReal.one_ne_top (incl_le_one L u), ← L.avg_card,
      havg_toReal _ fun ω => ENNReal.natCast_ne_top _]
    simp [S]
  have havg : ∑ ω, c * (((n - S ω).choose K : ℝ) / (n.choose K : ℝ)) ≤ δ := by
    have h2 : avg (fun B : (Law.subset n K hKn).Ω => L.escape B.1) ≤ L.miss K :=
      (avg_mono fun B => L.escape_le_miss B.2.ge).trans (avg_const _).le
    rw [L.avg_escape_subsets hKn] at h2
    have h3 := ENNReal.toReal_mono (ne_top_of_le_ne_top ENNReal.one_ne_top (L.miss_le_one K)) h2
    refine le_trans (le_of_eq ?_) (h3.trans hδ)
    rw [havg_toReal]
    · refine Finset.sum_congr rfl fun ω _ => ?_
      rw [ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast]
    · intro ω
      exact ENNReal.div_ne_top (ENNReal.natCast_ne_top _) (by exact_mod_cast (Nat.choose_pos hKn).ne')
  have hgle : ∀ s : ℕ, max 0 (1 - (s : ℝ) / M) ^ K ≤ ((n - s).choose K : ℝ) / (n.choose K : ℝ) := by
    intro s
    rcases le_or_gt (s + K) n with hs | hs
    · have hle := choose_mul_pow_le n K s hs K le_rfl
      have hsK : (s : ℝ) + K ≤ n := by exact_mod_cast hs
      have hMs : 0 ≤ 1 - (s : ℝ) / M := by
        rw [sub_nonneg, div_le_one hM]
        linarith
      rw [max_eq_right hMs, show 1 - (s : ℝ) / M = (M - s) / M by field_simp, div_pow,
        div_le_div_iff₀ (by positivity) hc]
      linarith
    · have hsK : (n : ℝ) + 1 ≤ s + K := by exact_mod_cast (show n + 1 ≤ s + K by omega)
      have hMs : 1 - (s : ℝ) / M ≤ 0 := by
        rw [sub_nonpos, le_div_iff₀ hM]
        linarith
      rw [max_eq_left hMs, zero_pow hK.ne']
      positivity
  have hδ0 : 0 ≤ δ := ENNReal.toReal_nonneg.trans hδ
  have key : 1 - meanDraws L / M ≤ δ ^ ((K : ℝ)⁻¹) := by
    rcases le_or_gt (1 - meanDraws L / M) 0 with h | h
    · exact h.trans (Real.rpow_nonneg hδ0 _)
    · set t₀ := 1 - meanDraws L / M with ht₀
      have hsumt : ∑ ω, c * (1 - (S ω : ℝ) / M) = t₀ := by
        have e : ∀ ω, c * (1 - (S ω : ℝ) / M) = c - c * (S ω : ℝ) / M := fun ω => by ring
        rw [Finset.sum_congr rfl fun ω _ => e ω, Finset.sum_sub_distrib, hcsum, ← Finset.sum_div, ← hmean]
      have hlow : t₀ ^ K ≤ δ := by
        have e : ∀ ω, c * (t₀ ^ K * (1 + K * ((1 - (S ω : ℝ) / M) / t₀ - 1))) =
            t₀ ^ K * (1 - K) * c + t₀ ^ K * K / t₀ * (c * (1 - (S ω : ℝ) / M)) := fun ω => by
          field_simp
          ring
        calc t₀ ^ K = ∑ ω, c * (t₀ ^ K * (1 + K * ((1 - (S ω : ℝ) / M) / t₀ - 1))) := by
              rw [Finset.sum_congr rfl fun ω _ => e ω, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum,
                hcsum, hsumt]
              field_simp
              ring
          _ ≤ ∑ ω, c * (max 0 (1 - (S ω : ℝ) / M) ^ K) :=
              Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left (tangent_le K hK h _) hc0
          _ ≤ ∑ ω, c * (((n - S ω).choose K : ℝ) / (n.choose K : ℝ)) :=
              Finset.sum_le_sum fun ω _ => mul_le_mul_of_nonneg_left (hgle (S ω)) hc0
          _ ≤ δ := havg
      have := Real.rpow_le_rpow (by positivity) hlow (by positivity : (0 : ℝ) ≤ (K : ℝ)⁻¹)
      rwa [Real.pow_rpow_inv_natCast h.le hK.ne'] at this
  have h2 : 1 - δ ^ ((K : ℝ)⁻¹) ≤ meanDraws L / M := by linarith
  rw [le_div_iff₀ hM] at h2
  calc M * (1 - δ ^ ((K : ℝ)⁻¹)) = (1 - δ ^ ((K : ℝ)⁻¹)) * M := mul_comm _ _
    _ ≤ meanDraws L := h2

/-- `e^{−x} ≤ 1 − x + x²/2` for `x ≥ 0`: `e^x ≥ 1 + x + x²/2`, and `(1 + x + x²/2)(1 − x + x²/2) = 1 + x⁴/4`. -/
theorem exp_neg_le (x : ℝ) (hx : 0 ≤ x) : Real.exp (-x) ≤ 1 - x + x ^ 2 / 2 := by
  have h1 := Real.quadratic_le_exp_of_nonneg hx
  have hq : 0 < 1 + x + x ^ 2 / 2 := by positivity
  have hprod : (1 + x + x ^ 2 / 2) * (1 - x + x ^ 2 / 2) = 1 + (x ^ 2) ^ 2 / 4 := by ring
  have h4 : 0 ≤ (x ^ 2) ^ 2 / 4 := by positivity
  rw [Real.exp_neg]
  calc (Real.exp x)⁻¹ ≤ (1 + x + x ^ 2 / 2)⁻¹ := inv_anti₀ hq h1
    _ = (1 + x + x ^ 2 / 2)⁻¹ * 1 := (mul_one _).symm
    _ ≤ (1 + x + x ^ 2 / 2)⁻¹ * ((1 + x + x ^ 2 / 2) * (1 - x + x ^ 2 / 2)) := by
        gcongr
        linarith
    _ = 1 - x + x ^ 2 / 2 := by field_simp

theorem ConverseAsymptotic : Guarantees.ConverseAsymptotic := by
  intro n K L δ hK hKn hδ0 hδ1 hmiss
  refine le_trans ?_ (ConverseExact n K L δ hK hKn hmiss)
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  have hKn' : (K : ℝ) ≤ n := by exact_mod_cast hKn
  set Λ := Real.log δ⁻¹ with hΛdef
  have hΛ : 0 ≤ Λ := Real.log_nonneg ((one_le_inv₀ hδ0).2 hδ1)
  have hroot : δ ^ ((K : ℝ)⁻¹) = Real.exp (-(Λ / K)) := by
    rw [Real.rpow_def_of_pos hδ0]
    congr 1
    simp only [Λ, Real.log_inv]
    ring
  rw [hroot]
  have hx := exp_neg_le (Λ / K) (div_nonneg hΛ hKpos.le)
  have hM : 0 ≤ (n : ℝ) - K + 1 := by linarith
  calc ((n : ℝ) - K + 1) / K * Λ * (1 - Λ / (2 * K)) = ((n : ℝ) - K + 1) * (Λ / K - (Λ / K) ^ 2 / 2) := by
        field_simp
    _ ≤ ((n : ℝ) - K + 1) * (1 - Real.exp (-(Λ / K))) := mul_le_mul_of_nonneg_left (by linarith) hM

theorem Achievability : Guarantees.Achievability := Unequal.achievability

theorem GranularityOptimum : Guarantees.GranularityOptimum := Unequal.granularityOptimum ConverseAsymptotic

theorem GranularityChoice : Guarantees.GranularityChoice := Unequal.granularityChoice GranularityOptimum

theorem ServedGranularityOptimum : Guarantees.ServedGranularityOptimum := by
  intro n s C f o δ w β h hs hC hf ho hδ hδ1
  obtain ⟨hlo, hhi⟩ := GranularityOptimum n s C f o δ hs hC hf ho hδ hδ1
  refine ⟨fun L hL => by linarith [hlo L hL], fun ε hε => ?_⟩
  obtain ⟨L, hL, hc⟩ := hhi ε hε
  exact ⟨L, hL, by linarith⟩

/-- `A·s + B/s` over `s > 0` is least at `t` with `t² = B/A`: the difference is `A (s − t)²/s`. -/
theorem root_law_le {A t s : ℝ} (hA : 0 ≤ A) (ht : 0 < t) (hs : 0 < s) :
    A * t + A * (t * t) / t ≤ A * s + A * (t * t) / s := by
  have hs0 : s ≠ 0 := hs.ne'
  have ht0 : t ≠ 0 := ht.ne'
  have hd : A * s + A * (t * t) / s - (A * t + A * (t * t) / t) = A * ((s - t) ^ 2 / s) := by
    first | (field_simp; ring) | field_simp
  have : 0 ≤ A * ((s - t) ^ 2 / s) := mul_nonneg hA (div_nonneg (sq_nonneg _) hs.le)
  linarith

theorem ServedGranularityChoice : Guarantees.ServedGranularityChoice := by
  intro F C f o δ w β h hF hC ho hδ hδ1 hw hβ hh s hs
  unfold servedCost unitCost
  have hΛ : 0 < Real.log δ⁻¹ := by rw [Real.log_inv]; linarith [Real.log_neg hδ hδ1]
  have hB : 0 ≤ w * β * h := mul_nonneg (mul_nonneg hw hβ) hh
  have hC0 : C ≠ 0 := hC.ne'
  set Λ := Real.log δ⁻¹ with hΛdef
  have hoΛ : 0 < o * Λ := mul_pos ho hΛ
  have hoΛ0 : o * Λ ≠ 0 := hoΛ.ne'
  have ho0 : o ≠ 0 := ho.ne'
  have hΛ0 : Λ ≠ 0 := hΛ.ne'
  set t := Real.sqrt (w * β * h * C / (o * Λ)) with htdef
  have htt : t * t = w * β * h * C / (o * Λ) := Real.mul_self_sqrt (div_nonneg (mul_nonneg hB hC.le) hoΛ.le)
  have hFC : 0 ≤ F / C := div_nonneg hF.le hC.le
  have hA : 0 ≤ F / C * (o * Λ) := mul_nonneg hFC hoΛ.le
  have hcommit : ∀ x : ℝ, x ≠ 0 → w * (F / x * β * h) = F / C * (o * Λ) * (t * t) / x := by
    intro x hx
    rw [htt]
    first | (field_simp; ring) | field_simp
  rw [hcommit s hs.ne']
  rcases (Real.sqrt_nonneg _ : (0 : ℝ) ≤ t).eq_or_lt with h0 | h0
  · rw [← h0]
    have : 0 ≤ o * s * (F / C * Λ) := mul_nonneg (mul_nonneg ho.le hs.le) (mul_nonneg hFC hΛ.le)
    simp only [mul_zero, zero_mul, div_zero, zero_div, add_zero]
    nlinarith
  · rw [hcommit t h0.ne']
    have key := root_law_le hA h0 hs
    nlinarith [key]

theorem HeavyUnit : Guarantees.HeavyUnit := by
  intro n w C δ L hL u hu
  have h := hL {u} (by simpa using hu)
  rw [Law.escape_singleton, ENNReal.toReal_sub_of_le (incl_le_one L u) ENNReal.one_ne_top,
    ENNReal.toReal_one] at h
  linarith

theorem UnequalDraws : Guarantees.UnequalDraws := Unequal.unequalDraws ConverseAsymptotic

theorem UnequalCost : Guarantees.UnequalCost := fun _ w _ _ _ _ _ _ _ hρ ha haA hw hC hδ0 hδ1 hf ho hAΛ L hL =>
  Unequal.unequalCost_of_le ConverseAsymptotic w hρ ha haA hw hC hδ0 hδ1 hf ho hAΛ L hL

end FlockSoundness.Audit.Partitioning.SecurityProofs
