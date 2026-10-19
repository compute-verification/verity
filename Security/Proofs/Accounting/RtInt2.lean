import Definitions.Accounting.Compose

/-!
# RT2-INT's witness: Compose's live floor reads each stage's share, not the round count (scratch, not for `main`)

Checked against #2677's tree (`cursor/int-tie-ea24` @ f3569b633b, which merges Compose's b2899a6bf0); `main` has no
`Definitions.Accounting.Compose` yet.

`SplitRateAt` at `w_max`, at its folded and whole vertices at any share `w ≤ w_max`, gives `a·w ≤ Λ·ln(1/p)` and
`b·w ≤ Λ·ln(1/p0)` (RT-INT's `folded_floor`, `whole_floor` and `log_floor`, r20261019-055920-6d6f, restated here).
`CircuitWork`'s and `CircuitWorkPeriod`'s premises put every stage's share at most `w_max` and give `ρ·share ≤ 1`, so
under them, whatever `K_d`, every stage `j` has `a·stageShare j ≤ Λ·ln(1/p)` and `b·stageShare j ≤ Λ·ln(1/p0)`. The
logical term `Λ·ln(1/δ_s)` is at least `a·ln(1/δ_s)/ln(1/p)` times the largest stage's share, however many rounds the
window has: `Llama31Rounds`' 10,552 rounds reach `ε < 1%` only if no round holds more than `1/10,552` of the words.
-/

namespace RtInt2

open Accounting.Compose

theorem whole_floor {K : ℕ} {ρ p p0 a b wmax Λ ν ν0 w : ℝ}
    (h : SplitRateAt K ρ p p0 a b wmax Λ ν ν0) (hw : 0 ≤ w) (hww : w ≤ wmax) (hp0 : 0 ≤ p0) (hp1 : p0 ≤ 1) :
    Real.exp (b * w / Λ) * (p0 + (1 - p0) * (1 - ρ * w) ^ K) ≤ 1 := by
  unfold SplitRateAt at h
  obtain ⟨-, -, hA⟩ := h
  have h1 := hA 0 w 0 w le_rfl le_rfl hw le_rfl (by linarith)
  have h2 := hA 0 w 0 0 le_rfl le_rfl le_rfl hw (by linarith)
  simp only [mul_zero, add_zero, zero_add, sub_self, sub_zero, zero_sub, one_pow, mul_one] at h1 h2
  nlinarith [mul_le_mul_of_nonneg_left h1 hp0, mul_le_mul_of_nonneg_left h2 (sub_nonneg.2 hp1)]

theorem folded_floor {K : ℕ} {ρ p p0 a b wmax Λ ν ν0 w : ℝ}
    (h : SplitRateAt K ρ p p0 a b wmax Λ ν ν0) (hw : 0 ≤ w) (hww : w ≤ wmax) (hp : 0 ≤ p) (hp1 : p ≤ 1) :
    Real.exp (a * w / Λ) * (p + (1 - p) * (1 - ρ * w) ^ K) ≤ 1 := by
  unfold SplitRateAt at h
  obtain ⟨-, -, hA⟩ := h
  have h1 := hA w 0 w 0 hw le_rfl le_rfl le_rfl (by linarith)
  have h2 := hA w 0 0 0 le_rfl hw le_rfl le_rfl (by linarith)
  simp only [mul_zero, add_zero, sub_self, sub_zero, zero_sub, one_pow, mul_one] at h1 h2
  nlinarith [mul_le_mul_of_nonneg_left h1 hp, mul_le_mul_of_nonneg_left h2 (sub_nonneg.2 hp1)]

theorem log_floor {q π w Λ c : ℝ} (hΛ : 0 < Λ) (hπ : 0 < π) (hπ1 : π ≤ 1) (hq : 0 ≤ q)
    (h : Real.exp (c * w / Λ) * (π + (1 - π) * q) ≤ 1) : c * w ≤ Λ * Real.log (1 / π) := by
  have hE := Real.exp_pos (c * w / Λ)
  have h1 : Real.exp (c * w / Λ) * π ≤ 1 := by nlinarith [mul_nonneg (sub_nonneg.2 hπ1) hq]
  have h2 : Real.exp (c * w / Λ) ≤ 1 / π := (le_div_iff₀ hπ).2 h1
  have h3 : c * w / Λ ≤ Real.log (1 / π) := by
    have := Real.log_le_log hE h2
    rwa [Real.log_exp] at this
  have h4 := mul_le_mul_of_nonneg_left h3 hΛ.le
  have e : Λ * (c * w / Λ) = c * w := by field_simp
  linarith

theorem tileWeight_nonneg {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) :
    0 ≤ tileWeight Sc Cd Np t := by
  unfold tileWeight
  positivity

theorem tileWeight_le_shareCone {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) :
    tileWeight Sc Cd Np t ≤ shareCone Sc Cd Np t := by
  unfold shareCone
  split_ifs with h
  · simp [tileWeight, h]
  · unfold tileCone
    have hr : Cd.Reaches t.val t.val := Relation.ReflTransGen.refl
    refine le_trans ?_ (Finset.single_le_sum (fun t' _ => ?_) (Finset.mem_univ t))
    all_goals first
      | simp only [if_pos hr, le_refl]
      | (split_ifs <;> first | exact tileWeight_nonneg Sc Cd Np _ | exact le_rfl)

theorem shareCone_nonneg {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) :
    0 ≤ shareCone Sc Cd Np t :=
  le_trans (tileWeight_nonneg Sc Cd Np t) (tileWeight_le_shareCone Sc Cd Np t)

theorem stageShare_nonneg {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (j : ℕ) :
    0 ≤ stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j := by
  unfold stageShare
  refine Finset.sum_nonneg fun t _ => ?_
  split_ifs
  · exact shareCone_nonneg Sc Cd Np t
  · exact le_rfl

theorem stageShare_le_sum {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (j : ℕ) :
    stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ ∑ t : Fin Np, shareCone Sc Cd Np t := by
  unfold stageShare
  refine Finset.sum_le_sum fun t _ => ?_
  split_ifs
  · exact le_rfl
  · exact shareCone_nonneg Sc Cd Np t

theorem rho_mul_sum_le {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) {ρ : ℝ}
    (hρ : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) :
    ρ * ∑ t : Fin Np, shareCone Sc Cd Np t ≤ 1 := by
  rcases Nat.eq_zero_or_pos Np with h0 | h0
  · subst h0
    simp
  · have hNp : (0 : ℝ) < Np := by exact_mod_cast h0
    have hs : ∑ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ ∑ _t : Fin Np, (1 : ℝ) :=
      Finset.sum_le_sum fun t _ => hρ t
    rw [← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_one] at hs
    have e : (Np : ℝ) * (ρ * ∑ t : Fin Np, shareCone Sc Cd Np t) = ρ * Np * ∑ t : Fin Np, shareCone Sc Cd Np t := by
      ring
    have h2 : (Np : ℝ) * (ρ * ∑ t : Fin Np, shareCone Sc Cd Np t) ≤ (Np : ℝ) * 1 := by
      rw [e, mul_one]
      exact hs
    exact le_of_mul_le_mul_left h2 hNp

theorem rho_stageShare_le {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) {ρ : ℝ} (hρ0 : 0 ≤ ρ)
    (hρ : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1) (j : ℕ) :
    ρ * stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ 1 :=
  le_trans (mul_le_mul_of_nonneg_left (stageShare_le_sum Sc Cd Np j) hρ0) (rho_mul_sum_le Sc Cd Np hρ)

/-- **The folded floor at every stage**: under `CircuitWork`'s premises on `ρ`, the stage share, `0 < Λ` and
`SplitRate` at folded luck `p ∈ (0, 1]`, each stage `j` has `a·stageShare j ≤ Λ·ln(1/p)`, whatever `K_d`. -/
theorem stage_floor_folded {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ}
    {ρ p p0 a b wmax Λ : ℝ} (hρ0 : 0 < ρ) (hρ : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1)
    (hshare : ∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax)
    (hΛ : 0 < Λ) (hp : 0 < p) (hp1 : p ≤ 1) (hrate : SplitRate Kd ρ p p0 a b wmax Λ) (j : ℕ) :
    a * stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ Λ * Real.log (1 / p) := by
  obtain ⟨ν, ν0, h⟩ := hrate
  have hw0 := stageShare_nonneg Sc Cd Np j
  have hfl := folded_floor h hw0 (hshare j) hp.le hp1
  have hq : 0 ≤ (1 - ρ * stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j) ^ Kd :=
    pow_nonneg (by linarith [rho_stageShare_le Sc Cd Np hρ0.le hρ j]) Kd
  exact log_floor hΛ hp hp1 hq hfl

/-- **The whole floor at every stage**: the same at whole luck `p0 ∈ (0, 1]`: `b·stageShare j ≤ Λ·ln(1/p0)`. -/
theorem stage_floor_whole {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ}
    {ρ p p0 a b wmax Λ : ℝ} (hρ0 : 0 < ρ) (hρ : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1)
    (hshare : ∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax)
    (hΛ : 0 < Λ) (hp0 : 0 < p0) (hp1 : p0 ≤ 1) (hrate : SplitRate Kd ρ p p0 a b wmax Λ) (j : ℕ) :
    b * stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ Λ * Real.log (1 / p0) := by
  obtain ⟨ν, ν0, h⟩ := hrate
  have hw0 := stageShare_nonneg Sc Cd Np j
  have hfl := whole_floor h hw0 (hshare j) hp0.le hp1
  have hq : 0 ≤ (1 - ρ * stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j) ^ Kd :=
    pow_nonneg (by linarith [rho_stageShare_le Sc Cd Np hρ0.le hρ j]) Kd
  exact log_floor hΛ hp0 hp1 hq hfl

/-- **At the live line's folded luck `p = 7/8`, `a = 0.218` and `δ_s = 2^-40`**: a stage holding share `s` of the
window's detected weight puts the logical term `Λ·ln(1/δ_s)` at `s·0.218·40·ln 2/ln(8/7)` or more (about `45.26·s`),
whatever the round count. At `perCall`'s largest call (gate_up, at least 0.01686/18 of an 18-step window's words) that
is at least 4.24%. -/
theorem stage_floor_live {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) {Np Kd : ℕ}
    {ρ p0 b wmax Λ : ℝ} (hρ0 : 0 < ρ) (hρ : ∀ t : Fin Np, ρ * Np * shareCone Sc Cd Np t ≤ 1)
    (hshare : ∀ j, stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j ≤ wmax)
    (hΛ : 0 < Λ) (hrate : SplitRate Kd ρ (7 / 8) p0 0.218 b wmax Λ) (j : ℕ) :
    0.218 * stageShare (tileStage Cd Np) (shareCone Sc Cd Np) j * Real.log (1 / (1 / 2 : ℝ) ^ 40) ≤
      Λ * Real.log (1 / (1 / 2 : ℝ) ^ 40) * Real.log (1 / (7 / 8 : ℝ)) := by
  have h := stage_floor_folded Sc Cd hρ0 hρ hshare hΛ (by norm_num) (by norm_num) hrate j
  have hl : 0 ≤ Real.log (1 / (1 / 2 : ℝ) ^ 40) := Real.log_nonneg (by norm_num)
  nlinarith [mul_le_mul_of_nonneg_right h hl]

end RtInt2

#print axioms RtInt2.stage_floor_folded
#print axioms RtInt2.stage_floor_whole
#print axioms RtInt2.stage_floor_live
