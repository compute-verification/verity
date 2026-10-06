import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.FreeCompute

/-!
# Exfiltration: a law that certifies the target

Independent draws with probability `q = num/N` (`Law.bernoulli`) give a set of `K + 1` or more units the escape
`(1 − q)^(K+1)` at most. When that is below `δ`, the harm bound `b·K` and the count `K` certify `b·K + locBits n K`
(`bern_certifies`). Taking `q` just above `Λ/(K+1)`, or `1` when that is at least `1`, makes `1 − q < e^{−Λ/(K+1)}`, so the
law certifies at `n·q ≤ n·Λ/(K+1) + ε` draws (`exists_certifying_law`). At `δ = 1` this still draws every unit with a small
positive probability, so only the empty set escapes with probability `1`.
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil

open Finset
open scoped ENNReal

theorem cost_const {n : ℕ} (L : Law n) (a : ℝ) : cost L (fun _ => a) = meanDraws L * a := by
  unfold cost meanDraws
  rw [Finset.sum_mul]

theorem meanDraws_nonneg {n : ℕ} (L : Law n) : 0 ≤ meanDraws L :=
  Finset.sum_nonneg fun _ _ => ENNReal.toReal_nonneg

theorem bern_escape_toReal {n num N : ℕ} [NeZero N] (h : num ≤ N) (B : Finset (Fin n)) :
    ((Law.bernoulli n num N).escape B).toReal = (1 - (num : ℝ) / N) ^ B.card := by
  have hN : (N : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne N
  rw [Law.bernoulli_escape h, ENNReal.toReal_pow]
  congr 1
  unfold Law.bernFactor
  rw [mul_zero, zero_add, ENNReal.toReal_div, ENNReal.toReal_natCast, ENNReal.toReal_natCast, Nat.cast_sub h]
  field_simp

theorem bern_incl_toReal {n num N : ℕ} [NeZero N] (h : num ≤ N) (u : Fin n) :
    ((Law.bernoulli n num N).incl u).toReal = (num : ℝ) / N := by
  have e := congrArg ENNReal.toReal (Law.escape_singleton (Law.bernoulli n num N) u)
  rw [bern_escape_toReal h, Finset.card_singleton, pow_one,
    ENNReal.toReal_sub_of_le (incl_le_one _ u) ENNReal.one_ne_top, ENNReal.toReal_one] at e
  linarith

theorem bern_meanDraws {n num N : ℕ} [NeZero N] (h : num ≤ N) :
    meanDraws (Law.bernoulli n num N) = n * ((num : ℝ) / N) := by
  unfold meanDraws
  simp only [bern_incl_toReal h, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **Independent draws certify `b·K + locBits n K`** once a set of `K + 1` escapes with less than `δ`. -/
theorem bern_certifies {n num N b K : ℕ} [NeZero N] (h : num ≤ N) {δ ℓ : ℝ}
    (hesc : (1 - (num : ℝ) / N) ^ (K + 1) < δ) (hℓ : (b : ℝ) * K + locBits n K ≤ ℓ) :
    CertifiesLeak (Law.bernoulli n num N) (fun _ => b) δ ℓ := by
  have hN : (0 : ℝ) < N := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne N)
  have ht0 : 0 ≤ 1 - (num : ℝ) / N := by
    rw [sub_nonneg, div_le_one hN]
    exact_mod_cast h
  have ht1 : 1 - (num : ℝ) / N ≤ 1 := by
    have : 0 ≤ (num : ℝ) / N := by positivity
    linarith
  have hδ : 0 < δ := lt_of_le_of_lt (pow_nonneg ht0 _) hesc
  have hbig : ∀ B : Finset (Fin n), K + 1 ≤ B.card →
      (Law.bernoulli n num N).escape B ≤ ENNReal.ofReal ((1 - (num : ℝ) / N) ^ (K + 1)) := by
    intro B hB
    rw [ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (Law.escape_le_one _ B))
      (pow_nonneg ht0 _), bern_escape_toReal h]
    exact pow_le_pow_of_le_one ht0 ht1 hB
  have hmiss_le : (Law.bernoulli n num N).miss (K + 1) ≤ ENNReal.ofReal ((1 - (num : ℝ) / N) ^ (K + 1)) :=
    iSup₂_le fun B hB => hbig B hB
  have hmiss : (Law.bernoulli n num N).miss (K + 1) < ENNReal.ofReal δ :=
    lt_of_le_of_lt hmiss_le ((ENNReal.ofReal_lt_ofReal_iff hδ).2 hesc)
  unfold CertifiesLeak
  refine ⟨b * K, K, ?_, hmiss, by push_cast; exact hℓ⟩
  intro B hB
  have hcard : B.card ≤ K := by
    by_contra hc
    exact absurd (hB.trans ((Law.bernoulli n num N).escape_le_miss (K := K + 1) (by omega))) (not_le.2 hmiss)
  have hcard' : (B.card : ℝ) ≤ K := by exact_mod_cast hcard
  simp only [Finset.sum_const, nsmul_eq_mul]
  push_cast
  nlinarith [Nat.cast_nonneg (α := ℝ) b]

/-- **A law certifying `ℓ` that draws at most `n·Λ/(K+1) + ε` units.** -/
theorem exists_certifying_law (n b K : ℕ) {δ ℓ ε : ℝ} (hδ0 : 0 < δ) (hδ1 : δ ≤ 1)
    (hℓ : (b : ℝ) * K + locBits n K ≤ ℓ) (hε : 0 < ε) :
    ∃ L : Law n, CertifiesLeak L (fun _ => b) δ ℓ ∧ meanDraws L ≤ n * Real.log δ⁻¹ / (K + 1) + ε := by
  have hΛ : 0 ≤ Real.log δ⁻¹ := Real.log_nonneg ((one_le_inv₀ hδ0).2 hδ1)
  have hK : (0 : ℝ) < K + 1 := by positivity
  obtain ⟨x, hx⟩ : ∃ x, x = Real.log δ⁻¹ / (K + 1) := ⟨_, rfl⟩
  have hx0 : 0 ≤ x := by rw [hx]; exact div_nonneg hΛ hK.le
  obtain ⟨N, hN⟩ := exists_nat_gt ((n : ℝ) / ε)
  have hNpos : (0 : ℝ) < N := lt_of_le_of_lt (div_nonneg (Nat.cast_nonneg n) hε.le) hN
  have : NeZero N := ⟨by exact_mod_cast hNpos.ne'⟩
  set num := min N (⌊x * N⌋₊ + 1) with hnumdef
  have hnum : num ≤ N := min_le_left _ _
  -- the escape factor is below `e^{-x}`
  have hfac : 1 - (num : ℝ) / N < Real.exp (-x) := by
    rcases min_cases N (⌊x * N⌋₊ + 1) with ⟨he, -⟩ | ⟨he, -⟩
    · rw [← hnumdef] at he
      rw [he, div_self hNpos.ne', sub_self]
      exact Real.exp_pos _
    · rw [← hnumdef] at he
      have h1 : x * N < num := by rw [he]; push_cast; exact Nat.lt_floor_add_one _
      have h2 : x < (num : ℝ) / N := by rw [lt_div_iff₀ hNpos]; exact h1
      have h3 := Real.add_one_le_exp (-x)
      linarith
  have hδeq : Real.exp (-x) ^ (K + 1) = δ := by
    have e : ((K + 1 : ℕ) : ℝ) * -x = Real.log δ := by
      rw [hx, Real.log_inv, Nat.cast_add, Nat.cast_one]
      field_simp
    rw [← Real.exp_nat_mul, e, Real.exp_log hδ0]
  have ht0 : 0 ≤ 1 - (num : ℝ) / N := by
    rw [sub_nonneg, div_le_one hNpos]
    exact_mod_cast hnum
  have hesc : (1 - (num : ℝ) / N) ^ (K + 1) < δ := by
    rw [← hδeq]
    exact pow_lt_pow_left₀ hfac ht0 (Nat.succ_ne_zero K)
  refine ⟨Law.bernoulli n num N, bern_certifies hnum hesc hℓ, ?_⟩
  rw [bern_meanDraws hnum]
  -- `num ≤ x·N + 1`, and `n/N ≤ ε`
  have hnum' : (num : ℝ) ≤ x * N + 1 := by
    have : (num : ℝ) ≤ (⌊x * N⌋₊ : ℝ) + 1 := by exact_mod_cast min_le_right N (⌊x * N⌋₊ + 1)
    linarith [Nat.floor_le (mul_nonneg hx0 hNpos.le)]
  have hq : (num : ℝ) / N ≤ x + 1 / N := by
    rw [div_le_iff₀ hNpos, add_mul, one_div_mul_cancel hNpos.ne']
    exact hnum'
  have hnN : (n : ℝ) / N ≤ ε := by
    rw [div_lt_iff₀ hε] at hN
    rw [div_le_iff₀ hNpos]
    linarith
  have hn0 : (0 : ℝ) ≤ n := Nat.cast_nonneg n
  calc (n : ℝ) * ((num : ℝ) / N) ≤ n * (x + 1 / N) := mul_le_mul_of_nonneg_left hq hn0
    _ = n * x + n / N := by ring
    _ ≤ n * x + ε := by linarith
    _ = n * Real.log δ⁻¹ / (K + 1) + ε := by rw [hx]; ring

end FlockSoundness.Audit.Partitioning.SecurityProofs.Exfil
