import Proofs.Flock.Soundness.Audit.Partitioning.Guarantees
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Unequal.Laws
import Proofs.Flock.Soundness.Audit.Partitioning.SecurityProofs.Unequal.Witnesses

/-!
# Achievability, the granularity optimum and unequal units: the proofs

`SecurityProofs.FreeCompute` proves `ConverseAsymptotic` and imports this file, so the lemmas here that need it take it
as a hypothesis (`hCA`), and `FreeCompute` passes its proof.

* `achievability`: independent draws with `q_u = min(1, Λ/C · w u)` rounded up to multiples of `1/den`. Rounding up
  only lowers each escape, and `1 − q_u ≤ e^{−Λ w_u / C}`, so a set doing work at least `C` escapes with at most
  `e^{−Λ} = δ`.
* `granularityOptimum`: the lower side is `ConverseAsymptotic` at `K = ⌈C/s⌉` (`granularity_arith`); the upper side is
  `achievability` at `ε/(n·c(s) + 1)`. `granularityChoice` composes the two.
* `unequalDraws`: unit `u` is in class `⌊log_ρ (w u / a)⌋`, so it does work in `[aρ^j, aρ^{j+1})`. On each class,
  `ConverseAsymptotic` at `K_j = ⌈C/(aρ^j)⌉` bounds its draws (`class_bound`), and the classes' bounds sum.
* `unequalCost_of_le`: the same per class, weighted by the class's cheapest unit, with `c(ρt) ≤ ρ·c(t)`. It assumes
  `a ≤ A`, which every `n ≥ 1` implies; without it the statement is false at `n = 0` (`unequalCost_needs_le`).
-/

namespace FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal

open Finset Game
open scoped ENNReal

theorem log_inv_nonneg {δ : ℝ} (hδ0 : 0 < δ) (hδ1 : δ ≤ 1) : 0 ≤ Real.log δ⁻¹ :=
  Real.log_nonneg ((one_le_inv₀ hδ0).2 hδ1)

theorem exp_neg_log_inv {δ : ℝ} (hδ0 : 0 < δ) : Real.exp (-Real.log δ⁻¹) = δ := by
  rw [Real.log_inv, neg_neg, Real.exp_log hδ0]

/-- `1 − min(1, x) ≤ e^{−x}`. -/
theorem one_sub_min_le_exp (x : ℝ) : 1 - min 1 x ≤ Real.exp (-x) := by
  rcases le_total 1 x with h | h
  · rw [min_eq_left h, sub_self]
    exact (Real.exp_pos _).le
  · rw [min_eq_right h]
    linarith [Real.add_one_le_exp (-x)]

/-- `AΛ ≤ C` keeps the second-order factor `1 − AΛ/(2C)` at least `1/2`. -/
theorem half_le_factor {A Λ C : ℝ} (hC : 0 < C) (hAΛ : A * Λ ≤ C) : 0 ≤ 1 - A * Λ / (2 * C) := by
  have : A * Λ / (2 * C) ≤ 1 / 2 := by
    rw [div_le_iff₀ (by positivity)]
    linarith
  linarith

theorem unitCost_nonneg {f o s : ℝ} (hf : 0 ≤ f) (ho : 0 ≤ o) (hs : 0 ≤ s) : 0 ≤ unitCost f o s := by
  unfold unitCost
  positivity

theorem unitCost_mono {f o x y : ℝ} (ho : 0 ≤ o) (h : x ≤ y) : unitCost f o x ≤ unitCost f o y := by
  unfold unitCost
  nlinarith

theorem cost_const {n : ℕ} (L : Law n) (c : ℝ) : cost L (fun _ => c) = meanDraws L * c := by
  unfold cost meanDraws
  rw [Finset.sum_mul]

/-- A law under which every set of at least `K` units escapes with at most `δ` misses `K` units with at most `δ`. -/
theorem miss_toReal_le_of {n : ℕ} {L : Law n} {K : ℕ} {δ : ℝ} (hδ : 0 ≤ δ)
    (h : ∀ B : Finset (Fin n), K ≤ B.card → (L.escape B).toReal ≤ δ) : (L.miss K).toReal ≤ δ := by
  have hle : L.miss K ≤ ENNReal.ofReal δ := iSup₂_le fun B hB =>
    (ENNReal.le_ofReal_iff_toReal_le (ne_top_of_le_ne_top ENNReal.one_ne_top (L.escape_le_one B)) hδ).2 (h B hB)
  calc (L.miss K).toReal ≤ (ENNReal.ofReal δ).toReal := ENNReal.toReal_mono ENNReal.ofReal_ne_top hle
    _ = δ := ENNReal.toReal_ofReal hδ

/-! ## Achievability -/

/-- **Achievability**, by independent draws with `q_u = min(1, Λ/C · w u)` rounded up to a multiple of
`1/(⌈1/ε⌉ + 1)`. -/
theorem achievability : Guarantees.Achievability := by
  intro n w C δ ε hw hC hδ0 hδ1 hε
  set Λ := Real.log δ⁻¹ with hΛdef
  have hΛ : 0 ≤ Λ := log_inv_nonneg hδ0 hδ1
  set q : Fin n → ℝ := fun u => min 1 (Λ / C * w u) with hqdef
  have hq0 : ∀ u, 0 ≤ q u := fun u => le_min zero_le_one (mul_nonneg (div_nonneg hΛ hC.le) (hw u))
  have hq1 : ∀ u, q u ≤ 1 := fun u => min_le_left _ _
  set den : ℕ := ⌈ε⁻¹⌉₊ + 1 with hden
  have : NeZero den := ⟨Nat.succ_ne_zero _⟩
  have hden0 : (0 : ℝ) < den := by rw [hden]; positivity
  have hdenε : (den : ℝ)⁻¹ ≤ ε := by
    rw [inv_le_comm₀ hden0 hε, hden]
    push_cast
    linarith [Nat.le_ceil ε⁻¹]
  set num : Fin n → ℕ := fun u => ⌈q u * den⌉₊ with hnumdef
  have hnum : ∀ u, num u ≤ den := fun u => Nat.ceil_le.2 (by nlinarith [hq1 u])
  have hp_ge : ∀ u, q u ≤ (num u : ℝ) / den := fun u => by
    rw [le_div_iff₀ hden0]
    exact Nat.le_ceil _
  have hp_le : ∀ u, (num u : ℝ) / den ≤ q u + ε := fun u => by
    have h1 : (num u : ℝ) < q u * den + 1 := Nat.ceil_lt_add_one (mul_nonneg (hq0 u) hden0.le)
    have h2 : (num u : ℝ) / den < q u + (den : ℝ)⁻¹ := by
      rw [div_lt_iff₀ hden0]
      calc (num u : ℝ) < q u * den + 1 := h1
        _ = (q u + (den : ℝ)⁻¹) * den := by field_simp
    linarith
  refine ⟨bern num den, fun B hB => ?_, fun u => ?_⟩
  · rw [bern_escape_toReal hnum]
    calc ∏ u ∈ B, (1 - (num u : ℝ) / den) ≤ ∏ u ∈ B, Real.exp (-(Λ / C * w u)) := by
          refine Finset.prod_le_prod₀ (fun u _ => ?_) (fun u _ => ?_)
          · rw [sub_nonneg, div_le_one hden0]
            exact_mod_cast hnum u
          · exact (sub_le_sub_left (hp_ge u) 1).trans (one_sub_min_le_exp _)
      _ = Real.exp (-(Λ / C * ∑ u ∈ B, w u)) := by
          rw [← Real.exp_sum, Finset.mul_sum, ← Finset.sum_neg_distrib]
      _ ≤ Real.exp (-Λ) := by
          refine Real.exp_le_exp.2 (neg_le_neg ?_)
          calc Λ = Λ / C * C := by field_simp
            _ ≤ Λ / C * ∑ u ∈ B, w u := mul_le_mul_of_nonneg_left hB (div_nonneg hΛ hC.le)
      _ = δ := exp_neg_log_inv hδ0
  · rw [bern_incl_toReal hnum]
    exact hp_le u

/-! ## The granularity optimum -/

/-- The arithmetic of `GranularityOptimum`'s lower side, at `x = C/s`: `(n − K + 1)/K ≥ (n/x)(1 − 1/x − x/n)` for
`x ≤ K ≤ x + 1`, and `1 − Λ/(2K) ≥ 1 − Λ/(2x)`. -/
theorem granularity_arith' {n x Λ : ℝ} {K : ℕ} (hx : 0 < x) (hΛ : 0 ≤ Λ) (hn : 0 < n) (hK1 : x ≤ K)
    (hK2 : (K : ℝ) ≤ x + 1) (hX : 0 < 1 - 1 / x - x / n - Λ / (2 * x)) :
    n / x * Λ * (1 - 1 / x - x / n - Λ / (2 * x)) ≤ (n - K + 1) / K * Λ * (1 - Λ / (2 * K)) := by
  have hK0 : (0 : ℝ) < K := lt_of_lt_of_le hx hK1
  set a := 1 - 1 / x - x / n with ha
  set b := Λ / (2 * x) with hb
  have hb0 : 0 ≤ b := div_nonneg hΛ (by positivity)
  have ha1 : a ≤ 1 := by
    have h1 : 0 ≤ 1 / x := by positivity
    have h2 : 0 ≤ x / n := by positivity
    linarith
  have hab : b < a := by linarith
  have hnx : x < n := by
    have : x / n < 1 := by
      have : 0 ≤ 1 / x := by positivity
      linarith
    rwa [div_lt_one hn] at this
  have h1 : n / x * a ≤ (n - K + 1) / K := by
    have e1 : (n - x) / (x + 1) ≤ (n - K + 1) / K :=
      calc (n - x) / (x + 1) ≤ (n - x) / K := div_le_div_of_nonneg_left (by linarith) hK0 hK2
        _ ≤ (n - K + 1) / K := div_le_div_of_nonneg_right (by linarith) hK0.le
    have e2 : (n - x) / (x + 1) - n / x * a = (n + x ^ 2) / (x ^ 2 * (x + 1)) := by
      have hx' := hx.ne'
      have hn' := hn.ne'
      have hx1 : x + 1 ≠ 0 := by positivity
      rw [ha]
      field_simp
      ring
    have e3 : 0 ≤ (n + x ^ 2) / (x ^ 2 * (x + 1)) := by positivity
    linarith
  have h2 : 1 - b ≤ 1 - Λ / (2 * K) := by
    have : Λ / (2 * K) ≤ b := div_le_div_of_nonneg_left hΛ (by positivity) (by linarith)
    linarith
  have hna : 0 ≤ n / x * a := mul_nonneg (by positivity) (by linarith)
  have h3 : n / x * a * (1 - b) ≤ (n - K + 1) / K * (1 - Λ / (2 * K)) :=
    mul_le_mul h1 h2 (by linarith) (hna.trans h1)
  have h4 : n / x * (a - b) ≤ n / x * a * (1 - b) := by
    rw [mul_assoc]
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    nlinarith
  calc n / x * Λ * (a - b) = Λ * (n / x * (a - b)) := by ring
    _ ≤ Λ * (n / x * a * (1 - b)) := mul_le_mul_of_nonneg_left h4 hΛ
    _ ≤ Λ * ((n - K + 1) / K * (1 - Λ / (2 * K))) := mul_le_mul_of_nonneg_left h3 hΛ
    _ = (n - K + 1) / K * Λ * (1 - Λ / (2 * K)) := by ring

/-- `granularity_arith'` in the statement's variables. -/
theorem granularity_arith {n s C Λ : ℝ} {K : ℕ} (hs : 0 < s) (hC : 0 < C) (hΛ : 0 ≤ Λ) (hn : 0 < n)
    (hK1 : C / s ≤ K) (hK2 : (K : ℝ) ≤ C / s + 1) (hX : 0 < 1 - s / C - C / (n * s) - s * Λ / (2 * C)) :
    n * s / C * Λ * (1 - s / C - C / (n * s) - s * Λ / (2 * C)) ≤ (n - K + 1) / K * Λ * (1 - Λ / (2 * K)) := by
  have hs' := hs.ne'
  have hC' := hC.ne'
  have hn' := hn.ne'
  have e1 : n * s / C = n / (C / s) := by field_simp
  have e2 : s / C = 1 / (C / s) := by field_simp
  have e3 : C / (n * s) = C / s / n := by field_simp
  have e4 : s * Λ / (2 * C) = Λ / (2 * (C / s)) := by field_simp
  rw [e2, e3, e4] at hX
  rw [e1, e2, e3, e4]
  exact granularity_arith' (div_pos hC hs) hΛ hn hK1 hK2 hX

/-- **The granularity optimum.** -/
theorem granularityOptimum (hCA : Guarantees.ConverseAsymptotic) : Guarantees.GranularityOptimum := by
  intro n s C f o δ hs hC hf ho hδ0 hδ1
  set c := unitCost f o s with hcdef
  have hc : 0 ≤ c := unitCost_nonneg hf ho hs.le
  set Λ := Real.log δ⁻¹ with hΛdef
  have hΛ : 0 ≤ Λ := log_inv_nonneg hδ0 hδ1
  refine ⟨fun L hL => ?_, fun ε hε => ?_⟩
  · rw [cost_const]
    suffices h : n * s / C * Λ * (1 - s / C - C / (n * s) - s * Λ / (2 * C)) ≤ meanDraws L by
      calc c * (n * s / C * Λ) * (1 - s / C - C / (n * s) - s * Λ / (2 * C))
          = c * (n * s / C * Λ * (1 - s / C - C / (n * s) - s * Λ / (2 * C))) := by ring
        _ ≤ c * meanDraws L := mul_le_mul_of_nonneg_left h hc
        _ = meanDraws L * c := mul_comm _ _
    have hpre : 0 ≤ (n : ℝ) * s / C * Λ := by positivity
    rcases le_or_gt (1 - s / C - C / (n * s) - s * Λ / (2 * C)) 0 with hX | hX
    · exact (mul_nonpos_of_nonneg_of_nonpos hpre hX).trans (meanDraws_nonneg L)
    rcases Nat.eq_zero_or_pos n with hn | hn
    · subst hn
      simpa using meanDraws_nonneg L
    have hn' : (0 : ℝ) < n := by exact_mod_cast hn
    have hCs : 0 < C / s := div_pos hC hs
    set K := ⌈C / s⌉₊ with hKdef
    have hK0 : 0 < K := Nat.ceil_pos.2 hCs
    have hK1 : C / s ≤ K := Nat.le_ceil _
    have hK2 : (K : ℝ) ≤ C / s + 1 := (Nat.ceil_lt_add_one hCs.le).le
    have hKn : K ≤ n := by
      refine Nat.ceil_le.2 ?_
      have h1 : C / (n * s) < 1 := by
        have : 0 ≤ s / C := by positivity
        have : 0 ≤ s * Λ / (2 * C) := by positivity
        linarith
      rw [div_lt_one (by positivity)] at h1
      rw [div_le_iff₀ hs]
      linarith
    have hmiss : (L.miss K).toReal ≤ δ := miss_toReal_le_of hδ0.le fun B hB => hL B (by
      rw [Finset.sum_const, nsmul_eq_mul]
      calc C = C / s * s := by field_simp
        _ ≤ K * s := mul_le_mul_of_nonneg_right hK1 hs.le
        _ ≤ B.card * s := mul_le_mul_of_nonneg_right (by exact_mod_cast hB) hs.le)
    exact (granularity_arith hs hC hΛ hn' hK1 hK2 hX).trans (hCA n K L δ hK0 hKn hδ0 hδ1 hmiss)
  · set ε' := ε / (n * c + 1) with hε'def
    have hden : 0 < (n : ℝ) * c + 1 := by positivity
    have hε' : 0 < ε' := div_pos hε hden
    obtain ⟨L, hL, hincl⟩ := achievability n (fun _ => s) C δ ε' (fun _ => hs.le) hC hδ0 hδ1 hε'
    refine ⟨L, hL, ?_⟩
    have hu : ∀ u, (L.incl u).toReal * c ≤ (Λ / C * s + ε') * c := fun u =>
      mul_le_mul_of_nonneg_right ((hincl u).trans (by linarith [min_le_right 1 (Λ / C * s)])) hc
    calc cost L (fun _ => c) = ∑ u, (L.incl u).toReal * c := rfl
      _ ≤ ∑ _u : Fin n, (Λ / C * s + ε') * c := Finset.sum_le_sum fun u _ => hu u
      _ = c * (n * s / C * Λ) + n * c * ε' := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
          ring
      _ ≤ c * (n * s / C * Λ) + ε := by
          have : (n : ℝ) * c * ε' ≤ ε := by
            rw [hε'def, ← mul_div_assoc, div_le_iff₀ hden]
            nlinarith
          linarith

/-- **The best granularity minimizes `c(s)`**, from both sides of `GranularityOptimum`. -/
theorem granularityChoice (hGO : Guarantees.GranularityOptimum) : Guarantees.GranularityChoice := by
  intro n n' s s' C f o δ hs hs' hns hC hf ho hδ0 hδ1 L' hL' ε hε
  set c := unitCost f o s with hcdef
  set c' := unitCost f o s' with hc'def
  have hc : 0 ≤ c := unitCost_nonneg hf ho hs.le
  have hc' : 0 ≤ c' := unitCost_nonneg hf ho hs'.le
  set Λ := Real.log δ⁻¹ with hΛdef
  set X' := 1 - s' / C - C / (n' * s') - s' * Λ / (2 * C) with hX'def
  have hlow := (hGO n' s' C f o δ hs' hC hf ho hδ0 hδ1).1 L' hL'
  have hL'0 : 0 ≤ cost L' (fun _ => c') := cost_nonneg L' fun _ => hc'
  rcases le_or_gt X' 0 with hX | hX
  · obtain ⟨L, hL, -⟩ := (hGO n s C f o δ hs hC hf ho hδ0 hδ1).2 1 one_pos
    refine ⟨L, hL, ?_⟩
    have h1 : c' * (X' * cost L (fun _ => c)) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos hc' (mul_nonpos_of_nonpos_of_nonneg hX (cost_nonneg L fun _ => hc))
    nlinarith
  · set ε₁ := ε / (c' * X' + 1) with hε₁def
    have hden : 0 < c' * X' + 1 := by positivity
    obtain ⟨L, hL, hcost⟩ := (hGO n s C f o δ hs hC hf ho hδ0 hδ1).2 ε₁ (div_pos hε hden)
    refine ⟨L, hL, ?_⟩
    have hsame : (n : ℝ) * s / C * Λ = n' * s' / C * Λ := by rw [hns]
    have hcX : 0 ≤ c' * X' := mul_nonneg hc' hX.le
    have hε₁ : c' * X' * ε₁ ≤ ε := by
      rw [hε₁def, ← mul_div_assoc, div_le_iff₀ hden]
      nlinarith
    calc c' * (X' * cost L (fun _ => c)) = c' * X' * cost L (fun _ => c) := by ring
      _ ≤ c' * X' * (c * (n * s / C * Λ) + ε₁) := mul_le_mul_of_nonneg_left hcost hcX
      _ = c * (c' * (n' * s' / C * Λ) * X') + c' * X' * ε₁ := by rw [hsame]; ring
      _ ≤ c * cost L' (fun _ => c') + ε := by
          have := mul_le_mul_of_nonneg_left hlow hc
          linarith

/-! ## Unequal units -/

/-- **A unit's size class.** A unit of work `x ≥ a` is in class `j = ⌊log_ρ (x/a)⌋`, so `aρ^j ≤ x < aρ^{j+1}`. -/
theorem class_spec {ρ a x : ℝ} (hρ : 1 < ρ) (ha : 0 < a) (hx : a ≤ x) :
    a * ρ ^ ⌊Real.logb ρ (x / a)⌋₊ ≤ x ∧ x < ρ * (a * ρ ^ ⌊Real.logb ρ (x / a)⌋₊) := by
  have hxa : 1 ≤ x / a := (one_le_div ha).2 hx
  have hxa0 : 0 < x / a := lt_of_lt_of_le one_pos hxa
  set y := Real.logb ρ (x / a) with hydef
  have hy : 0 ≤ y := Real.logb_nonneg hρ hxa
  have hρy : ρ ^ y = x / a := Real.rpow_logb (by linarith) hρ.ne' hxa0
  constructor
  · have h1 : ρ ^ (⌊y⌋₊ : ℝ) ≤ ρ ^ y := Real.rpow_le_rpow_of_exponent_le hρ.le (Nat.floor_le hy)
    rw [Real.rpow_natCast, hρy, le_div_iff₀ ha] at h1
    linarith
  · have h1 : ρ ^ y < ρ ^ ((⌊y⌋₊ + 1 : ℕ) : ℝ) :=
      Real.rpow_lt_rpow_of_exponent_lt hρ (by push_cast; exact Nat.lt_floor_add_one y)
    rw [Real.rpow_natCast, hρy, pow_succ, div_lt_iff₀ ha] at h1
    linarith

theorem class_le {ρ a x A : ℝ} (hρ : 1 < ρ) (ha : 0 < a) (hx : a ≤ x) (hxA : x ≤ A) :
    ⌊Real.logb ρ (x / a)⌋₊ ≤ ⌊Real.logb ρ (A / a)⌋₊ :=
  Nat.floor_mono (Real.logb_le_logb_of_le hρ (div_pos (ha.trans_le hx) ha) (div_le_div_of_nonneg_right hxA ha.le))

/-- `(m − K + 1)/K ≥ m·t/C·(1 − A/C) − 1` for `K ≤ C/t + 1`, when `t ≤ A` or `m = 0`. -/
theorem class_arith {m t A C : ℝ} {K : ℕ} (ht : 0 < t) (hC : 0 < C) (hm : 0 ≤ m) (hK0 : (0 : ℝ) < K)
    (hK2 : (K : ℝ) ≤ C / t + 1) (htA : 0 < m → t ≤ A) : m * t / C * (1 - A / C) - 1 ≤ (m - K + 1) / K := by
  have hCt : (m + 1) / (C / t + 1) = (m + 1) * t / (C + t) := by
    field_simp
  have h1 : (m + 1) * t / (C + t) - 1 ≤ (m - K + 1) / K := by
    have e : (m - K + 1) / K = (m + 1) / K - 1 := by
      field_simp
      ring
    rw [e, ← hCt]
    have : (m + 1) / (C / t + 1) ≤ (m + 1) / K := div_le_div_of_nonneg_left (by linarith) hK0 hK2
    linarith
  have h2 : m * t / C * (1 - A / C) ≤ (m + 1) * t / (C + t) := by
    rcases hm.eq_or_lt with h0 | h0
    · rw [← h0]
      simp only [zero_mul, zero_div, zero_add, one_mul]
      positivity
    have htA' := htA h0
    calc m * t / C * (1 - A / C) ≤ m * t / C * (1 - t / C) := by
          refine mul_le_mul_of_nonneg_left ?_ (by positivity)
          have : t / C ≤ A / C := div_le_div_of_nonneg_right htA' hC.le
          linarith
      _ = m * t * (C - t) / C ^ 2 := by field_simp
      _ ≤ (m + 1) * t / (C + t) := by
          rw [div_le_div_iff₀ (by positivity) (by positivity)]
          have : 0 ≤ m * t * t ^ 2 := by positivity
          have : 0 ≤ t * C ^ 2 := by positivity
          nlinarith
  linarith

/-- **One class's draws.** A set `S` of units each doing work at least `t`, with `t ≤ A` when `S` has a unit, is drawn
`Λ (1 − AΛ/(2C)) (|S|·t/C·(1 − A/C) − 1)` times in expectation, by `ConverseAsymptotic` on `L` restricted to `S` at
`K = ⌈C/t⌉`. -/
theorem class_bound (hCA : Guarantees.ConverseAsymptotic) {n : ℕ} (L : Law n) (w : Fin n → ℝ) {A C δ t : ℝ}
    (hC : 0 < C) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1) (hAΛ : A * Real.log δ⁻¹ ≤ C) (hL : MeetsTarget L w C δ)
    (S : Finset (Fin n)) (ht : 0 < t) (htS : ∀ u ∈ S, t ≤ w u) (htA : S.Nonempty → t ≤ A) :
    Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) * (S.card * t / C * (1 - A / C) - 1) ≤
      ∑ u ∈ S, (L.incl u).toReal := by
  set Λ := Real.log δ⁻¹ with hΛdef
  have hΛ : 0 ≤ Λ := log_inv_nonneg hδ0 hδ1
  have hβ : 0 ≤ 1 - A * Λ / (2 * C) := half_le_factor hC hAΛ
  have hCt : 0 < C / t := div_pos hC ht
  set K := ⌈C / t⌉₊ with hKdef
  have hK0 : 0 < K := Nat.ceil_pos.2 hCt
  have hK0' : (0 : ℝ) < K := by exact_mod_cast hK0
  have hK1 : C / t ≤ K := Nat.le_ceil _
  have hK2 : (K : ℝ) ≤ C / t + 1 := (Nat.ceil_lt_add_one hCt.le).le
  have hsum0 : 0 ≤ ∑ u ∈ S, (L.incl u).toReal := sum_nonneg fun _ _ => ENNReal.toReal_nonneg
  have harith : (S.card : ℝ) * t / C * (1 - A / C) - 1 ≤ ((S.card : ℝ) - K + 1) / K :=
    class_arith ht hC (Nat.cast_nonneg _) hK0' hK2 fun hm => htA (Finset.card_pos.1 (by exact_mod_cast hm))
  have hstep : Λ * (1 - A * Λ / (2 * C)) * (((S.card : ℝ) - K + 1) / K) ≤ ∑ u ∈ S, (L.incl u).toReal := by
    rcases le_or_gt K S.card with hKm | hKm
    · have htA' := htA (Finset.card_pos.1 (lt_of_lt_of_le hK0 hKm))
      have hmiss : ((restrict L S).miss K).toReal ≤ δ := miss_toReal_le_of hδ0.le fun B hB => by
        rw [restrict_escape]
        refine hL _ ?_
        rw [sum_map_restrict]
        calc C = C / t * t := by field_simp
          _ ≤ K * t := mul_le_mul_of_nonneg_right hK1 ht.le
          _ ≤ B.card * t := mul_le_mul_of_nonneg_right (by exact_mod_cast hB) ht.le
          _ = ∑ _i ∈ B, t := by rw [sum_const, nsmul_eq_mul]
          _ ≤ ∑ i ∈ B, w (S.orderEmbOfFin rfl i) := sum_le_sum fun i _ => htS _ (orderEmbOfFin_mem' S i)
      have h := hCA S.card K (restrict L S) δ hK0 hKm hδ0 hδ1 hmiss
      rw [meanDraws_restrict] at h
      refine le_trans ?_ h
      have hq : 0 ≤ ((S.card : ℝ) - K + 1) / K := by
        have : (K : ℝ) ≤ S.card := by exact_mod_cast hKm
        exact div_nonneg (by linarith) hK0'.le
      have hCAK : C ≤ A * K :=
        calc C = C / t * t := by field_simp
          _ ≤ K * t := mul_le_mul_of_nonneg_right hK1 ht.le
          _ ≤ K * A := mul_le_mul_of_nonneg_left htA' hK0'.le
          _ = A * K := mul_comm _ _
      have hb : 1 - A * Λ / (2 * C) ≤ 1 - Λ / (2 * K) := by
        have : Λ / (2 * K) ≤ A * Λ / (2 * C) := by
          rw [div_le_div_iff₀ (by positivity) (by positivity)]
          have := mul_le_mul_of_nonneg_left hCAK hΛ
          nlinarith
        linarith
      calc Λ * (1 - A * Λ / (2 * C)) * (((S.card : ℝ) - K + 1) / K)
          = ((S.card : ℝ) - K + 1) / K * Λ * (1 - A * Λ / (2 * C)) := by ring
        _ ≤ ((S.card : ℝ) - K + 1) / K * Λ * (1 - Λ / (2 * K)) := mul_le_mul_of_nonneg_left hb (mul_nonneg hq hΛ)
    · have hneg : ((S.card : ℝ) - K + 1) / K ≤ 0 := by
        have : (S.card : ℝ) + 1 ≤ K := by exact_mod_cast hKm
        exact div_nonpos_of_nonpos_of_nonneg (by linarith) hK0'.le
      exact (mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hΛ hβ) hneg).trans hsum0
  exact (mul_le_mul_of_nonneg_left harith (mul_nonneg hΛ hβ)).trans hstep

/-- The units in class `j`. -/
noncomputable abbrev classOf {n : ℕ} (w : Fin n → ℝ) (ρ a : ℝ) (j : ℕ) : Finset (Fin n) :=
  univ.filter fun u => ⌊Real.logb ρ (w u / a)⌋₊ = j

theorem sum_classes {n : ℕ} (w : Fin n → ℝ) {ρ a A : ℝ} (hρ : 1 < ρ) (ha : 0 < a) (hw : ∀ u, a ≤ w u ∧ w u ≤ A)
    (g : Fin n → ℝ) :
    ∑ j ∈ range (⌊Real.logb ρ (A / a)⌋₊ + 1), ∑ u ∈ classOf w ρ a j, g u = ∑ u, g u :=
  sum_fiberwise_of_maps_to (fun u _ => by
    rw [mem_range, Nat.lt_succ_iff]
    exact class_le hρ ha (hw u).1 (hw u).2) g

/-- **The converse for unequal units, in draws.** -/
theorem unequalDraws (hCA : Guarantees.ConverseAsymptotic) : Guarantees.UnequalDraws := by
  intro n w ρ a A C δ hρ ha hw hC hδ0 hδ1 hAΛ L hL
  have hρ0 : 0 < ρ := by linarith
  set Λ := Real.log δ⁻¹ with hΛdef
  have hΛ : 0 ≤ Λ := log_inv_nonneg hδ0 hδ1
  have hβ : 0 ≤ 1 - A * Λ / (2 * C) := half_le_factor hC hAΛ
  set J := ⌊Real.logb ρ (A / a)⌋₊ with hJdef
  have hF : 0 ≤ ∑ u, w u := sum_nonneg fun u _ => ha.le.trans (hw u).1
  have hJ0 : (0 : ℝ) ≤ J := Nat.cast_nonneg _
  rcases le_or_gt C A with hAC | hAC
  · have h1 : (∑ u, w u) / (ρ * C) * (1 - A / C) ≤ 0 := by
      refine mul_nonpos_of_nonneg_of_nonpos (div_nonneg hF (by positivity)) ?_
      rw [sub_nonpos, le_div_iff₀ hC]
      linarith
    have h2 : (∑ u, w u) / (ρ * C) * (1 - A / C) - ((J : ℝ) + 1) ≤ 0 := by linarith
    exact (mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hΛ hβ) h2).trans (meanDraws_nonneg L)
  have hAC' : 0 ≤ 1 - A / C := by
    rw [sub_nonneg, div_le_one hC]
    exact hAC.le
  have hbound : ∀ j ∈ range (J + 1), Λ * (1 - A * Λ / (2 * C)) *
      ((classOf w ρ a j).card * (a * ρ ^ j) / C * (1 - A / C) - 1) ≤ ∑ u ∈ classOf w ρ a j, (L.incl u).toReal :=
    fun j _ => class_bound hCA L w hC hδ0 hδ1 hAΛ hL _ (by positivity)
      (fun u hu => (mem_filter.1 hu).2 ▸ (class_spec hρ ha (hw u).1).1)
      (fun ⟨u, hu⟩ => (mem_filter.1 hu).2 ▸ (class_spec hρ ha (hw u).1).1.trans (hw u).2)
  have hwork : (∑ u, w u) / ρ ≤ ∑ j ∈ range (J + 1), ((classOf w ρ a j).card : ℝ) * (a * ρ ^ j) := by
    calc (∑ u, w u) / ρ = ∑ u, w u / ρ := Finset.sum_div _ _ _
      _ ≤ ∑ u, a * ρ ^ ⌊Real.logb ρ (w u / a)⌋₊ := sum_le_sum fun u _ => by
          rw [div_le_iff₀ (by linarith)]
          linarith [(class_spec hρ ha (hw u).1).2]
      _ = ∑ j ∈ range (J + 1), ∑ u ∈ classOf w ρ a j, a * ρ ^ ⌊Real.logb ρ (w u / a)⌋₊ :=
          (sum_classes w hρ ha hw _).symm
      _ = ∑ j ∈ range (J + 1), ((classOf w ρ a j).card : ℝ) * (a * ρ ^ j) := sum_congr rfl fun j _ => by
          rw [sum_congr rfl fun u hu => by rw [(mem_filter.1 hu).2], sum_const, nsmul_eq_mul]
  have hmean : meanDraws L = ∑ j ∈ range (J + 1), ∑ u ∈ classOf w ρ a j, (L.incl u).toReal :=
    (sum_classes w hρ ha hw _).symm
  rw [hmean]
  calc Λ * (1 - A * Λ / (2 * C)) * ((∑ u, w u) / (ρ * C) * (1 - A / C) - ((J : ℝ) + 1))
      ≤ Λ * (1 - A * Λ / (2 * C)) *
          ((∑ j ∈ range (J + 1), ((classOf w ρ a j).card : ℝ) * (a * ρ ^ j)) / C * (1 - A / C) - ((J : ℝ) + 1)) := by
        refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg hΛ hβ)
        rw [← div_div]
        gcongr
    _ = ∑ j ∈ range (J + 1), Λ * (1 - A * Λ / (2 * C)) *
          (((classOf w ρ a j).card : ℝ) * (a * ρ ^ j) / C * (1 - A / C) - 1) := by
        rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.sum_div, Finset.sum_const,
          card_range, nsmul_eq_mul, mul_one]
        push_cast
        ring
    _ ≤ ∑ j ∈ range (J + 1), ∑ u ∈ classOf w ρ a j, (L.incl u).toReal := sum_le_sum hbound

/-- `x · c(x) ≤ ρ² · t · c(t)` for `0 < t ≤ x ≤ ρt`: the work-weighted cost grows by at most `ρ²` across a class. -/
theorem mul_unitCost_le {f o ρ t x : ℝ} (hf : 0 ≤ f) (ho : 0 ≤ o) (hρ : 1 ≤ ρ) (ht : 0 < t) (htx : t ≤ x)
    (hx : x ≤ ρ * t) : x * unitCost f o x ≤ ρ ^ 2 * (t * unitCost f o t) := by
  unfold unitCost
  have hx0 : 0 ≤ x := ht.le.trans htx
  have h1 : x * f ≤ ρ ^ 2 * (t * f) := by
    have : ρ * t ≤ ρ ^ 2 * t := by nlinarith
    nlinarith
  have h2 : x * (o * x) ≤ ρ ^ 2 * (t * (o * t)) := by
    have : x * x ≤ (ρ * t) * (ρ * t) := mul_le_mul hx hx hx0 (by nlinarith)
    nlinarith
  nlinarith

/-- **The converse for unequal units, in cost.** Every `n ≥ 1` implies `a ≤ A`; at `n = 0`, without it, the statement
is false (`unequalCost_needs_le`). -/
theorem unequalCost_of_le (hCA : Guarantees.ConverseAsymptotic) {n : ℕ} (w : Fin n → ℝ) {ρ a A C δ f o : ℝ}
    (hρ : 1 < ρ) (ha : 0 < a) (haA : a ≤ A) (hw : ∀ u, a ≤ w u ∧ w u ≤ A) (hC : 0 < C) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1)
    (hf : 0 ≤ f) (ho : 0 ≤ o) (hAΛ : A * Real.log δ⁻¹ ≤ C) (L : Law n) (hL : MeetsTarget L w C δ) :
    Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) *
        ((∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) -
          (⌊Real.logb ρ (A / a)⌋₊ + 1) * unitCost f o A) ≤ cost L fun u => unitCost f o (w u) := by
  have hρ0 : 0 < ρ := by linarith
  set Λ := Real.log δ⁻¹ with hΛdef
  have hΛ : 0 ≤ Λ := log_inv_nonneg hδ0 hδ1
  have hβ : 0 ≤ 1 - A * Λ / (2 * C) := half_le_factor hC hAΛ
  set J := ⌊Real.logb ρ (A / a)⌋₊ with hJdef
  have hJ0 : (0 : ℝ) ≤ J := Nat.cast_nonneg _
  have hcA : 0 ≤ unitCost f o A := unitCost_nonneg hf ho (ha.le.trans haA)
  have hcw : ∀ u, 0 ≤ unitCost f o (w u) := fun u => unitCost_nonneg hf ho (ha.le.trans (hw u).1)
  have hF : 0 ≤ ∑ u, w u * unitCost f o (w u) := sum_nonneg fun u _ => mul_nonneg (ha.le.trans (hw u).1) (hcw u)
  have hcost0 : 0 ≤ cost L fun u => unitCost f o (w u) := cost_nonneg L hcw
  rcases le_or_gt C A with hAC | hAC
  · have h1 : (∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) ≤ 0 := by
      refine mul_nonpos_of_nonneg_of_nonpos (div_nonneg hF (by positivity)) ?_
      rw [sub_nonpos, le_div_iff₀ hC]
      linarith
    have h2 : (∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) - ((J : ℝ) + 1) * unitCost f o A ≤ 0 := by
      have : 0 ≤ ((J : ℝ) + 1) * unitCost f o A := mul_nonneg (by linarith) hcA
      linarith
    exact (mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hΛ hβ) h2).trans hcost0
  have hAC' : 0 ≤ 1 - A / C := by
    rw [sub_nonneg, div_le_one hC]
    exact hAC.le
  have hbound : ∀ j ∈ range (J + 1), Λ * (1 - A * Λ / (2 * C)) *
      (unitCost f o (a * ρ ^ j) * ((classOf w ρ a j).card * (a * ρ ^ j)) / C * (1 - A / C) - unitCost f o A) ≤
        ∑ u ∈ classOf w ρ a j, (L.incl u).toReal * unitCost f o (w u) := by
    intro j _
    set S := classOf w ρ a j
    set t := a * ρ ^ j with htdef
    have ht : 0 < t := by positivity
    have hct : 0 ≤ unitCost f o t := unitCost_nonneg hf ho ht.le
    have htS : ∀ u ∈ S, t ≤ w u := fun u hu => by
      rw [htdef, ← (mem_filter.1 hu).2]
      exact (class_spec hρ ha (hw u).1).1
    rcases S.eq_empty_or_nonempty with hS | hS
    · rw [hS, Finset.card_empty, Finset.sum_empty]
      simp only [Nat.cast_zero, zero_mul, mul_zero, zero_div, zero_sub]
      exact mul_nonpos_of_nonneg_of_nonpos (mul_nonneg hΛ hβ) (neg_nonpos.2 hcA)
    obtain ⟨u₀, hu₀⟩ := hS
    have htA : t ≤ A := (htS u₀ hu₀).trans (hw u₀).2
    have hcl := class_bound hCA L w hC hδ0 hδ1 hAΛ hL S ht htS fun _ => htA
    have hctA : unitCost f o t ≤ unitCost f o A := unitCost_mono ho htA
    calc Λ * (1 - A * Λ / (2 * C)) * (unitCost f o t * (S.card * t) / C * (1 - A / C) - unitCost f o A)
        ≤ Λ * (1 - A * Λ / (2 * C)) * (unitCost f o t * (S.card * t) / C * (1 - A / C) - unitCost f o t) :=
          mul_le_mul_of_nonneg_left (by linarith) (mul_nonneg hΛ hβ)
      _ = unitCost f o t * (Λ * (1 - A * Λ / (2 * C)) * (S.card * t / C * (1 - A / C) - 1)) := by ring
      _ ≤ unitCost f o t * ∑ u ∈ S, (L.incl u).toReal := mul_le_mul_of_nonneg_left hcl hct
      _ = ∑ u ∈ S, (L.incl u).toReal * unitCost f o t := by rw [Finset.mul_sum]; exact sum_congr rfl fun _ _ => mul_comm _ _
      _ ≤ ∑ u ∈ S, (L.incl u).toReal * unitCost f o (w u) := sum_le_sum fun u hu =>
          mul_le_mul_of_nonneg_left (unitCost_mono ho (htS u hu)) ENNReal.toReal_nonneg
  have hwork : (∑ u, w u * unitCost f o (w u)) / ρ ^ 2 ≤
      ∑ j ∈ range (J + 1), unitCost f o (a * ρ ^ j) * ((classOf w ρ a j).card * (a * ρ ^ j)) := by
    calc (∑ u, w u * unitCost f o (w u)) / ρ ^ 2 = ∑ u, w u * unitCost f o (w u) / ρ ^ 2 := Finset.sum_div _ _ _
      _ ≤ ∑ u, a * ρ ^ ⌊Real.logb ρ (w u / a)⌋₊ * unitCost f o (a * ρ ^ ⌊Real.logb ρ (w u / a)⌋₊) :=
          sum_le_sum fun u _ => by
            have hs := class_spec hρ ha (hw u).1
            rw [div_le_iff₀ (by positivity)]
            have := mul_unitCost_le hf ho hρ.le (by positivity) hs.1 hs.2.le
            linarith
      _ = ∑ j ∈ range (J + 1), ∑ u ∈ classOf w ρ a j,
            a * ρ ^ ⌊Real.logb ρ (w u / a)⌋₊ * unitCost f o (a * ρ ^ ⌊Real.logb ρ (w u / a)⌋₊) :=
          (sum_classes w hρ ha hw _).symm
      _ = ∑ j ∈ range (J + 1), unitCost f o (a * ρ ^ j) * ((classOf w ρ a j).card * (a * ρ ^ j)) :=
          sum_congr rfl fun j _ => by
            rw [sum_congr rfl fun u hu => by rw [(mem_filter.1 hu).2], sum_const, nsmul_eq_mul]
            ring
  have hcost : (cost L fun u => unitCost f o (w u)) =
      ∑ j ∈ range (J + 1), ∑ u ∈ classOf w ρ a j, (L.incl u).toReal * unitCost f o (w u) :=
    (sum_classes w hρ ha hw _).symm
  rw [hcost]
  calc Λ * (1 - A * Λ / (2 * C)) *
        ((∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) - ((J : ℝ) + 1) * unitCost f o A)
      ≤ Λ * (1 - A * Λ / (2 * C)) *
          ((∑ j ∈ range (J + 1), unitCost f o (a * ρ ^ j) * ((classOf w ρ a j).card * (a * ρ ^ j))) / C *
            (1 - A / C) - ((J : ℝ) + 1) * unitCost f o A) := by
        refine mul_le_mul_of_nonneg_left ?_ (mul_nonneg hΛ hβ)
        rw [← div_div]
        gcongr
    _ = ∑ j ∈ range (J + 1), Λ * (1 - A * Λ / (2 * C)) *
          (unitCost f o (a * ρ ^ j) * ((classOf w ρ a j).card * (a * ρ ^ j)) / C * (1 - A / C) - unitCost f o A) := by
        rw [← Finset.mul_sum, Finset.sum_sub_distrib, ← Finset.sum_mul, ← Finset.sum_div, Finset.sum_const,
          card_range, nsmul_eq_mul]
        push_cast
        ring
    _ ≤ _ := sum_le_sum hbound

/-- **`UnequalCost` needs `a ≤ A`**: the statement without it fails with no units. At `n = 0`,
`∀ u, a ≤ w u ∧ w u ≤ A` is vacuous, so `A` may be negative and `c(A) = f + o·A` with it. At `ρ = 2`, `a = 1`, `A = −1`,
`C = 1`, `δ = 1/2`, `f = 0`, `o = 1`, the bound is `log 2 · (1 + log 2 / 2) > 0`, while every law on no units costs
`0`. -/
theorem unequalCost_needs_le : ¬ ∀ (n : ℕ) (w : Fin n → ℝ) (ρ a A C δ f o : ℝ), 1 < ρ → 0 < a →
    (∀ u, a ≤ w u ∧ w u ≤ A) → 0 < C → 0 < δ → δ ≤ 1 → 0 ≤ f → 0 ≤ o → A * Real.log δ⁻¹ ≤ C →
      ∀ L : Law n, MeetsTarget L w C δ →
        Real.log δ⁻¹ * (1 - A * Real.log δ⁻¹ / (2 * C)) *
            ((∑ u, w u * unitCost f o (w u)) / (ρ ^ 2 * C) * (1 - A / C) -
              (⌊Real.logb ρ (A / a)⌋₊ + 1) * unitCost f o A) ≤ cost L fun u => unitCost f o (w u) := by
  intro h
  have hmeets : MeetsTarget (Law.subset 0 0 le_rfl) (Fin.elim0 : Fin 0 → ℝ) 1 (1 / 2) := fun B hB => by
    rw [Finset.eq_empty_of_isEmpty B, Finset.sum_empty] at hB
    norm_num at hB
  have hΛ : Real.log (1 / 2 : ℝ)⁻¹ = Real.log 2 := by norm_num
  have hlog2 : 0 < Real.log 2 := Real.log_pos one_lt_two
  have hAΛ : (-1 : ℝ) * Real.log (1 / 2 : ℝ)⁻¹ ≤ 1 := by rw [hΛ]; linarith
  have := h 0 Fin.elim0 2 1 (-1) 1 (1 / 2) 0 1 one_lt_two one_pos (fun u => u.elim0) one_pos (by norm_num)
    (by norm_num) le_rfl zero_le_one hAΛ _ hmeets
  have hJ : ⌊Real.logb 2 ((-1 : ℝ) / 1)⌋₊ = 0 := by
    simp [Real.logb, Real.log_neg_eq_log]
  rw [hΛ, hJ] at this
  simp only [cost, unitCost, Finset.univ_eq_empty, Finset.sum_empty, zero_div, zero_mul, Nat.cast_zero] at this
  nlinarith

end FlockSoundness.Audit.Partitioning.SecurityProofs.Unequal
