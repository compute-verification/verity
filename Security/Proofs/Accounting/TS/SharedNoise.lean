import Definitions.Accounting.Guarantees.TS.SharedNoise
import Proofs.Accounting.TS.Grind

/-!
# One noise seed a call: proofs

`CondReceiptLuckMarginal` averages each round's bound over the rest (`prob_equiv`, then `Stages.prob_prod`).

`GridLuck` integrates the columns out from the last (`SharedNoise.sum_prod_seq`): given the rows and the columns before
`j`, column `j`'s `r` units are bounded by Hölder at equal exponents (`SharedNoise.sum_prod_le_prod_rpow`, AM-GM after
scaling each factor by its norm) by `∏_i φ_{ij}^{1/r}`, where `φ_{ij}` is the unit's moment at `rθ` (`e^{rθw}` dead,
`LuckMoment`'s bound otherwise). That reads row `i`'s coins alone, so the sum over the rows factors
(`Fintype.prod_sum`), and AM-GM over a row's `n` units (`Grind.prod_le_mean_pow`) leaves each unit's `E[φ^{n/r}]`, the condition.

`GridTemplateCondition` reduces every ratio `e = n/r` to `e = 64`: below it by convexity of `exp` in `e` (the left side
is `1` at `e = 0`), above it by concavity of `y ↦ y^{64/e}` (the alive term's base is a power mean). At `e = 64` it is
`SharedNoise.grid_point`, from `Grind.alive_le` squared and `e^{1/4} ≤ 1.2842`. `GridTemplateMoment` is `GridLuck` at
`θ = 1/(4·max(64·r, n))`, with `r·θ = 1/(4·max(64, n/r))`.
-/

namespace Accounting.SecurityProofs.TS

open Accounting.TS Pouw.RelativeSecurity

namespace SharedNoise

/-- Hölder at equal exponents, from AM-GM after scaling each factor by its norm: if each `f i` has `r`th moment at
most `φ i > 0`, their product has mean at most `∏ φ i^{1/r}`. -/
theorem sum_prod_le_prod_rpow {X : Type} [Fintype X] {r : ℕ} (hr : 0 < r) (f : Fin r → X → ℝ)
    (hf : ∀ i x, 0 ≤ f i x) (φ : Fin r → ℝ) (hφ : ∀ i, 0 < φ i)
    (h : ∀ i, ∑ x, f i x ^ r ≤ φ i * Fintype.card X) :
    ∑ x, ∏ i, f i x ≤ (∏ i, φ i ^ ((1 : ℝ) / r)) * Fintype.card X := by
  have hr' : (r : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
  set a : Fin r → ℝ := fun i => φ i ^ ((1 : ℝ) / r) with ha_def
  have ha : ∀ i, 0 < a i := fun i => Real.rpow_pos_of_pos (hφ i) _
  have har : ∀ i, a i ^ r = φ i := fun i => by
    simp only [ha_def]
    rw [← Real.rpow_natCast, ← Real.rpow_mul (hφ i).le, one_div_mul_cancel hr', Real.rpow_one]
  have hpt : ∀ x, ∏ i, f i x = (∏ i, a i) * ∏ i, (f i x / a i) := fun x => by
    rw [← Finset.prod_mul_distrib]
    refine Finset.prod_congr rfl fun i _ => ?_
    field_simp [(ha i).ne']
  have hmean : ∀ x, ∏ i, (f i x / a i) ≤ (∑ i, (f i x / a i) ^ r) / r := fun x =>
    Grind.prod_le_mean_pow hr _ fun i => div_nonneg (hf i x) (ha i).le
  have hA : 0 ≤ ∏ i, a i := Finset.prod_nonneg fun i _ => (ha i).le
  have hi : ∀ i, ∑ x, (f i x / a i) ^ r ≤ Fintype.card X := fun i => by
    have : ∑ x, (f i x / a i) ^ r = (∑ x, f i x ^ r) / φ i := by
      rw [Finset.sum_div]; refine Finset.sum_congr rfl fun x _ => ?_; rw [div_pow, har]
    rw [this, div_le_iff₀ (hφ i)]; linarith [h i]
  calc ∑ x, ∏ i, f i x = (∏ i, a i) * ∑ x, ∏ i, (f i x / a i) := by
        rw [Finset.mul_sum]; exact Finset.sum_congr rfl fun x _ => hpt x
    _ ≤ (∏ i, a i) * ∑ x, (∑ i, (f i x / a i) ^ r) / r := by
        gcongr with x; exact hmean x
    _ = (∏ i, a i) * ((∑ i, ∑ x, (f i x / a i) ^ r) / r) := by
        rw [← Finset.sum_div, Finset.sum_comm]
    _ ≤ (∏ i, a i) * ((∑ _i : Fin r, (Fintype.card X : ℝ)) / r) := by
        gcongr with i; exact hi i
    _ = (∏ i, a i) * Fintype.card X := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; field_simp

theorem castSucc_of_le {n : ℕ} {k : Fin (n + 1)} {j : Fin n} (hk : k ≤ j.castSucc) :
    ∃ k' : Fin n, k = k'.castSucc ∧ k' ≤ j := by
  have hne : k ≠ Fin.last n := fun h => by
    subst h; exact absurd hk (not_le.mpr (Fin.castSucc_lt_last j))
  refine ⟨k.castPred hne, (Fin.castSucc_castPred k hne).symm, ?_⟩
  rw [← Fin.castSucc_le_castSucc_iff, Fin.castSucc_castPred]; exact hk

/-- **Integrating the columns out from the last.** Factors `A j` that read the columns up to `j` only, each with mean
at most `c j` over its own column whatever the others, have a product of mean at most `∏ c j`. -/
theorem sum_prod_seq {Cl : Type} [Fintype Cl] [Nonempty Cl] : ∀ (n : ℕ) (A : Fin n → (Fin n → Cl) → ℝ)
    (c : Fin n → ℝ), (∀ j γ, 0 ≤ A j γ) → (∀ j γ γ', (∀ k, k ≤ j → γ k = γ' k) → A j γ = A j γ') →
    (∀ j γ, ∑ x, A j (Function.update γ j x) ≤ c j * Fintype.card Cl) →
    ∑ γ : Fin n → Cl, ∏ j, A j γ ≤ (∏ j, c j) * (Fintype.card Cl : ℝ) ^ n
  | 0, A, c, _, _, _ => by simp
  | n + 1, A, c, h0, hread, hc => by
    obtain ⟨x₀⟩ := ‹Nonempty Cl›
    set A' : Fin n → (Fin n → Cl) → ℝ := fun j γ => A j.castSucc (Fin.snoc γ x₀) with hA'_def
    have hA' : ∀ j γ x, A j.castSucc (Fin.snoc γ x) = A' j γ := by
      intro j γ x
      apply hread
      intro k hk
      obtain ⟨k', rfl, -⟩ := castSucc_of_le hk
      simp only [Fin.snoc_castSucc]
    have hread' : ∀ j γ γ', (∀ k, k ≤ j → γ k = γ' k) → A' j γ = A' j γ' := by
      intro j γ γ' h
      apply hread
      intro k hk
      obtain ⟨k', rfl, hk'⟩ := castSucc_of_le hk
      simp only [Fin.snoc_castSucc]
      exact h k' hk'
    have hc' : ∀ j γ, ∑ x, A' j (Function.update γ j x) ≤ c j.castSucc * Fintype.card Cl := by
      intro j γ
      calc ∑ x, A' j (Function.update γ j x) = ∑ x, A j.castSucc (Function.update (Fin.snoc γ x₀) j.castSucc x) := by
            refine Finset.sum_congr rfl fun x _ => ?_
            simp only [hA'_def, Fin.snoc_update]
        _ ≤ _ := hc _ _
    have IH := sum_prod_seq n A' (fun j => c j.castSucc) (fun j γ => h0 _ _) hread' hc'
    have hcard : (0 : ℝ) < Fintype.card Cl := by exact_mod_cast Fintype.card_pos
    have hcl : 0 ≤ c (Fin.last n) := by
      have h1 := hc (Fin.last n) (fun _ => x₀)
      have h2 : 0 ≤ ∑ x, A (Fin.last n) (Function.update (fun _ => x₀) (Fin.last n) x) :=
        Finset.sum_nonneg fun x _ => h0 _ _
      by_contra hneg
      push Not at hneg
      nlinarith
    have hP : ∀ γ', 0 ≤ ∏ j : Fin n, A' j γ' := fun γ' => Finset.prod_nonneg fun j _ => h0 _ _
    calc ∑ γ : Fin (n + 1) → Cl, ∏ j, A j γ
        = ∑ γ' : Fin n → Cl, ∑ x : Cl, (∏ j : Fin n, A' j γ') * A (Fin.last n) (Fin.snoc γ' x) := by
          rw [← (Fin.snocEquiv fun _ => Cl).sum_comp, Fintype.sum_prod_type, Finset.sum_comm]
          refine Finset.sum_congr rfl fun γ' _ => Finset.sum_congr rfl fun x _ => ?_
          have e : (Fin.snocEquiv fun _ => Cl) (x, γ') = Fin.snoc γ' x := rfl
          rw [e, Fin.prod_univ_castSucc]
          simp only [hA']
      _ = ∑ γ' : Fin n → Cl, (∏ j : Fin n, A' j γ') *
            ∑ x : Cl, A (Fin.last n) (Function.update (Fin.snoc γ' x₀) (Fin.last n) x) := by
          refine Finset.sum_congr rfl fun γ' _ => ?_
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [Fin.update_snoc_last]
      _ ≤ ∑ γ' : Fin n → Cl, (∏ j : Fin n, A' j γ') * (c (Fin.last n) * Fintype.card Cl) := by
          gcongr with γ'
          · exact hP γ'
          · exact hc _ _
      _ = (∑ γ' : Fin n → Cl, ∏ j : Fin n, A' j γ') * (c (Fin.last n) * Fintype.card Cl) := by
          rw [Finset.sum_mul]
      _ ≤ ((∏ j : Fin n, c j.castSucc) * (Fintype.card Cl : ℝ) ^ n) * (c (Fin.last n) * Fintype.card Cl) := by
          gcongr
      _ = (∏ j, c j) * (Fintype.card Cl : ℝ) ^ (n + 1) := by
          rw [Fin.prod_univ_castSucc, pow_succ]; ring

open Classical in
theorem gridLuck {Rw Cl : Type} [Fintype Rw] [Fintype Cl] {r n : ℕ} (w : Fin r → Fin n → ℝ)
    (dead : Fin r → Fin n → Rw → Prop) (lucky : Fin r → Fin n → (Fin r → Rw) → (Fin n → Cl) → Prop)
    (p q : Fin r → Fin n → ℝ) (θ κ : ℝ)
    (hn : 0 < n) (hr : 0 < r) (hθ : 0 ≤ θ) (hκ : 0 ≤ κ) (hw : ∀ i j, 0 ≤ w i j) (hp : ∀ i j, 0 ≤ p i j ∧ p i j ≤ 1)
    (hread : ∀ i j ρ γ γ', (∀ k, k ≤ j → γ k = γ' k) → (lucky i j ρ γ ↔ lucky i j ρ γ'))
    (halive : ∀ i j ρ γ, ¬dead i j (ρ i) → prob (fun c => lucky i j ρ (Function.update γ j c)) ≤ p i j)
    (hdead : ∀ i j, prob (dead i j) ≤ q i j)
    (hcond : ∀ i j, q i j * Real.exp (n * θ * w i j) +
        (1 - q i j) * (p i j * Real.exp (r * θ * w i j) + (1 - p i j) * Real.exp (-(κ * (r * θ) * w i j))) ^
          ((n : ℝ) / r) ≤ 1) :
    ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (θ * ∑ i, ∑ j, w i j * ((1 + κ) * (if lucky i j ω.1 ω.2 then 1 else 0) - κ)) ≤
      Fintype.card ((Fin r → Rw) × (Fin n → Cl)) := by
  rcases isEmpty_or_nonempty Cl with hC | hC
  · have : IsEmpty (Fin n → Cl) := ⟨fun γ => hC.false (γ ⟨0, hn⟩)⟩
    simp
  set gain : Fin r → Fin n → (Fin r → Rw) → (Fin n → Cl) → ℝ := fun i j ρ γ =>
    w i j * ((1 + κ) * (if lucky i j ρ γ then 1 else 0) - κ) with hgain
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have hrθ : 0 ≤ (r : ℝ) * θ := mul_nonneg hrpos.le hθ
  set f : Fin r → Fin n → ℝ := fun i j =>
    p i j * Real.exp (r * θ * w i j) + (1 - p i j) * Real.exp (-(κ * (r * θ) * w i j)) with hf
  set φ : Fin r → Fin n → Rw → ℝ := fun i j y => if dead i j y then Real.exp (r * θ * w i j) else f i j with hφ
  have hle : ∀ i j, Real.exp (-(κ * (r * θ) * w i j)) ≤ Real.exp (r * θ * w i j) := fun i j =>
    Real.exp_le_exp.mpr (by nlinarith [mul_nonneg (mul_nonneg hκ hrθ) (hw i j), mul_nonneg hrθ (hw i j)])
  have hf0 : ∀ i j, 0 < f i j := fun i j => by
    have h1 := (hp i j).1
    have h2 := hle i j
    have h3 := Real.exp_pos (-(κ * (r * θ) * w i j))
    simp only [hf]; nlinarith
  have hfe : ∀ i j, f i j ≤ Real.exp (r * θ * w i j) := fun i j => by
    simp only [hf]; nlinarith [hle i j, (hp i j).2]
  have hφ0 : ∀ i j y, 0 < φ i j y := fun i j y => by
    simp only [hφ]; split_ifs
    · exact Real.exp_pos _
    · exact hf0 i j
  -- one unit at rate rθ, given the rows and the other columns
  have hunit : ∀ i j ρ γ, ∑ x : Cl, Real.exp (r * θ * gain i j ρ (Function.update γ j x)) ≤
      φ i j (ρ i) * Fintype.card Cl := by
    intro i j ρ γ
    by_cases hd : dead i j (ρ i)
    · simp only [hφ, hd, ↓reduceIte]
      calc ∑ x : Cl, Real.exp (r * θ * gain i j ρ (Function.update γ j x))
          ≤ ∑ _x : Cl, Real.exp (r * θ * w i j) := by
            refine Finset.sum_le_sum fun x _ => Real.exp_le_exp.mpr ?_
            refine mul_le_mul_of_nonneg_left ?_ hrθ
            have : (1 + κ) * (if lucky i j ρ (Function.update γ j x) then (1 : ℝ) else 0) - κ ≤ 1 := by
              split_ifs <;> linarith
            simp only [hgain]; nlinarith [hw i j]
        _ = Real.exp (r * θ * w i j) * Fintype.card Cl := by
            rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]
    · simp only [hφ, hd, ↓reduceIte]
      exact LuckMoment (fun x => lucky i j ρ (Function.update γ j x)) (p i j) (w i j) (r * θ) κ (hw i j) hrθ hκ
        (halive i j ρ γ hd)
  -- a column's r units: Hölder over them, given the rows and the columns before it
  set A : (Fin r → Rw) → Fin n → (Fin n → Cl) → ℝ := fun ρ j γ => ∏ i, Real.exp (θ * gain i j ρ γ) with hA
  have hcol : ∀ ρ j γ, ∑ x, A ρ j (Function.update γ j x) ≤
      (∏ i, φ i j (ρ i) ^ ((1 : ℝ) / r)) * Fintype.card Cl := by
    intro ρ j γ
    refine sum_prod_le_prod_rpow hr (fun i x => Real.exp (θ * gain i j ρ (Function.update γ j x)))
      (fun i x => (Real.exp_pos _).le) (fun i => φ i j (ρ i)) (fun i => hφ0 i j (ρ i)) fun i => ?_
    calc ∑ x, Real.exp (θ * gain i j ρ (Function.update γ j x)) ^ r
        = ∑ x, Real.exp (r * θ * gain i j ρ (Function.update γ j x)) := by
          refine Finset.sum_congr rfl fun x _ => ?_
          rw [← Real.exp_nat_mul]; ring_nf
      _ ≤ _ := hunit i j ρ γ
  have hAread : ∀ ρ j γ γ', (∀ k, k ≤ j → γ k = γ' k) → A ρ j γ = A ρ j γ' := by
    intro ρ j γ γ' h
    simp only [hA, hgain]
    refine Finset.prod_congr rfl fun i _ => ?_
    simp only [if_congr (hread i j ρ γ γ' h) rfl rfl]
  have hseq : ∀ ρ, ∑ γ : Fin n → Cl, ∏ j, A ρ j γ ≤
      (∏ j, ∏ i, φ i j (ρ i) ^ ((1 : ℝ) / r)) * (Fintype.card Cl : ℝ) ^ n := fun ρ =>
    sum_prod_seq n (A ρ) _ (fun j γ => Finset.prod_nonneg fun i _ => (Real.exp_pos _).le) (hAread ρ) (hcol ρ)
  -- the rows: each unit's dead mass at exponent n/r
  have hpow : ∀ i j, ∑ y : Rw, φ i j y ^ ((n : ℝ) / r) ≤ Fintype.card Rw := by
    intro i j
    have hc := Grind.card_filter_le (dead i j) (hdead i j)
    set E := Real.exp (r * θ * w i j) ^ ((n : ℝ) / r) with hE
    have hEn : E = Real.exp (n * θ * w i j) := by
      rw [hE, ← Real.exp_mul]; congr 1; field_simp
    have hsplit : ∀ y, φ i j y ^ ((n : ℝ) / r) =
        f i j ^ ((n : ℝ) / r) + (E - f i j ^ ((n : ℝ) / r)) * (if dead i j y then 1 else 0) := by
      intro y; simp only [hφ]; split_ifs <;> ring
    have hd : 0 ≤ E - f i j ^ ((n : ℝ) / r) :=
      sub_nonneg.mpr (Real.rpow_le_rpow (hf0 i j).le (hfe i j) (by positivity))
    simp_rw [hsplit]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_boole, Finset.sum_const, Finset.card_univ,
      nsmul_eq_mul]
    have hcond' := hcond i j
    rw [← hEn] at hcond'
    calc (Fintype.card Rw : ℝ) * f i j ^ ((n : ℝ) / r) +
          (E - f i j ^ ((n : ℝ) / r)) * ((Finset.univ.filter (dead i j)).card : ℝ)
        ≤ (Fintype.card Rw : ℝ) * f i j ^ ((n : ℝ) / r) +
          (E - f i j ^ ((n : ℝ) / r)) * (q i j * Fintype.card Rw) := by gcongr
      _ = (q i j * E + (1 - q i j) * f i j ^ ((n : ℝ) / r)) * Fintype.card Rw := by ring
      _ ≤ 1 * Fintype.card Rw := by gcongr
      _ = Fintype.card Rw := one_mul _
  have hrow : ∀ i, ∑ y : Rw, ∏ j, φ i j y ^ ((1 : ℝ) / r) ≤ Fintype.card Rw := by
    intro i
    calc ∑ y : Rw, ∏ j, φ i j y ^ ((1 : ℝ) / r)
        ≤ ∑ y : Rw, (∑ j, (φ i j y ^ ((1 : ℝ) / r)) ^ n) / n :=
          Finset.sum_le_sum fun y _ => Grind.prod_le_mean_pow hn _ fun j => (Real.rpow_pos_of_pos (hφ0 i j y) _).le
      _ = (∑ j, ∑ y : Rw, φ i j y ^ ((n : ℝ) / r)) / n := by
          rw [← Finset.sum_div, Finset.sum_comm]
          congr 1
          refine Finset.sum_congr rfl fun j _ => Finset.sum_congr rfl fun y _ => ?_
          rw [← Real.rpow_natCast, ← Real.rpow_mul (hφ0 i j y).le]
          congr 1; ring
      _ ≤ (∑ _j : Fin n, (Fintype.card Rw : ℝ)) / n := by
          gcongr with j; exact hpow i j
      _ = Fintype.card Rw := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; field_simp
  -- assemble
  have hsplitω : ∀ ω : (Fin r → Rw) × (Fin n → Cl),
      Real.exp (θ * ∑ i, ∑ j, w i j * ((1 + κ) * (if lucky i j ω.1 ω.2 then 1 else 0) - κ)) =
        ∏ j, A ω.1 j ω.2 := by
    intro ω
    simp only [hA, hgain]
    rw [Finset.mul_sum, Real.exp_sum, Finset.prod_comm]
    refine Finset.prod_congr rfl fun i _ => ?_
    rw [Finset.mul_sum, Real.exp_sum]
  calc ∑ ω : (Fin r → Rw) × (Fin n → Cl),
        Real.exp (θ * ∑ i, ∑ j, w i j * ((1 + κ) * (if lucky i j ω.1 ω.2 then 1 else 0) - κ))
      = ∑ ρ : Fin r → Rw, ∑ γ : Fin n → Cl, ∏ j, A ρ j γ := by
        rw [Fintype.sum_prod_type]
        exact Finset.sum_congr rfl fun ρ _ => Finset.sum_congr rfl fun γ _ => hsplitω (ρ, γ)
    _ ≤ ∑ ρ : Fin r → Rw, (∏ j, ∏ i, φ i j (ρ i) ^ ((1 : ℝ) / r)) * (Fintype.card Cl : ℝ) ^ n :=
        Finset.sum_le_sum fun ρ _ => hseq ρ
    _ = (∏ i, ∑ y : Rw, ∏ j, φ i j y ^ ((1 : ℝ) / r)) * (Fintype.card Cl : ℝ) ^ n := by
        rw [← Finset.sum_mul, Fintype.prod_sum]
        congr 1
        exact Finset.sum_congr rfl fun ρ _ => Finset.prod_comm
    _ ≤ (∏ _i : Fin r, (Fintype.card Rw : ℝ)) * (Fintype.card Cl : ℝ) ^ n := by
        refine mul_le_mul_of_nonneg_right (Finset.prod_le_prod₀ (fun i _ => Finset.sum_nonneg fun y _ =>
          Finset.prod_nonneg fun j _ => (Real.rpow_pos_of_pos (hφ0 i j y) _).le) fun i _ => hrow i) (by positivity)
    _ = Fintype.card ((Fin r → Rw) × (Fin n → Cl)) := by
        simp [Fintype.card_prod]

/-- `e^{-a} ≤ 1/(1 + a + a²/2)` for `a ≥ 0`. -/
theorem exp_neg_le_quad (a : ℝ) (ha : 0 ≤ a) : Real.exp (-a) ≤ 1 / (1 + a + a ^ 2 / 2) := by
  rw [Real.exp_neg, ← one_div]
  exact one_div_le_one_div_of_le (by positivity) (Real.quadratic_le_exp_of_nonneg ha)

theorem exp_quarter_le : Real.exp (1 / 4) ≤ 12842 / 10000 := by
  have := Real.exp_bound' (x := 1 / 4) (by norm_num) (by norm_num) (n := 4) (by norm_num)
  have hs : ∑ m ∈ Finset.range 4, (1 / 4 : ℝ) ^ m / (m.factorial : ℝ) = 1 + 1 / 4 + 1 / 32 + 1 / 384 := by
    simp only [Finset.sum_range_succ, Finset.sum_range_zero, Nat.factorial_succ, Nat.factorial_zero]
    norm_num
  rw [hs] at this
  have h4 : ((4 : ℕ).factorial : ℝ) = 24 := by
    rw [show (4 : ℕ).factorial = 24 from rfl]; norm_num
  rw [h4] at this
  norm_num at this
  linarith

theorem grid_point_num (w : ℕ) (hw : w ≤ 32) :
    (6421 / 10000 : ℝ) ^ w + (1 - (1 / 2 : ℝ) ^ w) * Real.exp (-(3 / 8) * w + 17 / 2048 * (w : ℝ) ^ 2) ≤ 1 := by
  have hx32 : (w : ℝ) ≤ 32 := by exact_mod_cast hw
  have hq0 : 0 ≤ (1 / 2 : ℝ) ^ w := by positivity
  rcases Nat.lt_or_ge w 3 with h | h
  · interval_cases w
    · norm_num
    · have he := exp_neg_le_quad (751 / 2048) (by norm_num)
      have hx : -(3 / 8) * ((1 : ℕ) : ℝ) + 17 / 2048 * ((1 : ℕ) : ℝ) ^ 2 = -(751 / 2048) := by norm_num
      rw [hx]
      norm_num at he ⊢
      linarith
    · have he := exp_neg_le_quad (1468 / 2048) (by norm_num)
      have hx : -(3 / 8) * ((2 : ℕ) : ℝ) + 17 / 2048 * ((2 : ℕ) : ℝ) ^ 2 = -(1468 / 2048) := by norm_num
      rw [hx]
      norm_num at he ⊢
      linarith
  · have h3 : (3 : ℝ) ≤ w := by exact_mod_cast h
    have hE : -(3 / 8) * (w : ℝ) + 17 / 2048 * (w : ℝ) ^ 2 ≤ -1 := by
      nlinarith [mul_nonneg (sub_nonneg.2 h3) (sub_nonneg.2 hx32)]
    have he : Real.exp (-(3 / 8) * (w : ℝ) + 17 / 2048 * (w : ℝ) ^ 2) ≤ 2 / 5 :=
      calc _ ≤ Real.exp (-1) := Real.exp_le_exp.2 hE
        _ ≤ 1 / (1 + 1 + 1 ^ 2 / 2) := exp_neg_le_quad _ (by norm_num)
        _ = 2 / 5 := by norm_num
    have hd : (6421 / 10000 : ℝ) ^ w ≤ (6421 / 10000) ^ 3 := pow_le_pow_of_le_one (by norm_num) (by norm_num) h
    nlinarith [mul_nonneg hq0 (Real.exp_pos (-(3 / 8) * (w : ℝ) + 17 / 2048 * (w : ℝ) ^ 2)).le]


/-- The grid template's condition at `n/r = 64`, the point every ratio reduces to:
`2^{-w}·e^{w/4} + (1 − 2^{-w})·(½e^{w/256} + ½e^{-w/64})^{64} ≤ 1`. -/
theorem grid_point (w : ℕ) (hw : w ≤ 32) :
    (1 / 2 : ℝ) ^ w * Real.exp (w / 4) +
      (1 - (1 / 2 : ℝ) ^ w) * (1 / 2 * Real.exp (1 / 256 * w) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * w))) ^ 64
        ≤ 1 := by
  have hx0 : (0 : ℝ) ≤ w := Nat.cast_nonneg w
  have hx32 : (w : ℝ) ≤ 32 := by exact_mod_cast hw
  have hq1 : (1 / 2 : ℝ) ^ w ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  have hq0 : 0 ≤ (1 / 2 : ℝ) ^ w := by positivity
  set B := 1 / 2 * Real.exp (1 / 256 * (w : ℝ)) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * (w : ℝ))) with hB
  have hB0 : 0 ≤ B := by positivity
  have halive : B ^ 64 ≤ Real.exp (-(3 / 8) * w + 17 / 2048 * (w : ℝ) ^ 2) := by
    have h := Grind.alive_le w hx0 hx32
    calc B ^ 64 = (B ^ 32) ^ 2 := by rw [← pow_mul]
      _ ≤ Real.exp (-(3 / 16) * w + 17 / 4096 * (w : ℝ) ^ 2) ^ 2 := pow_le_pow_left₀ (by positivity) h 2
      _ = _ := by rw [← Real.exp_nat_mul]; congr 1; push_cast; ring
  have hdead : (1 / 2 : ℝ) ^ w * Real.exp (w / 4) ≤ (6421 / 10000) ^ w := by
    have he : Real.exp ((w : ℝ) / 4) = Real.exp (1 / 4) ^ w := by
      rw [← Real.exp_nat_mul]; ring_nf
    rw [he, ← mul_pow]
    exact pow_le_pow_left₀ (by positivity) (by linarith [exp_quarter_le]) w
  have key : (1 / 2 : ℝ) ^ w * Real.exp (w / 4) + (1 - (1 / 2 : ℝ) ^ w) * B ^ 64 ≤
      (6421 / 10000) ^ w + (1 - (1 / 2 : ℝ) ^ w) * Real.exp (-(3 / 8) * w + 17 / 2048 * (w : ℝ) ^ 2) :=
    add_le_add hdead (mul_le_mul_of_nonneg_left halive (sub_nonneg.2 hq1))
  exact key.trans (grid_point_num w hw)

theorem condReceiptLuck_marginal {Ω Z O : Type} [Fintype Ω] [Fintype Z] [Fintype O] (e : Ω ≃ Z × O) (pr : ℝ) :
    ∀ {n : ℕ} (G : ReceiptGame Ω n), G.CondReceiptLuck e pr → G.ReceiptLuck pr
  | _, .done, _ => trivial
  | _, .round d _ right same next, h => by
    obtain ⟨h1, h2⟩ := h
    refine ⟨?_, fun ω => condReceiptLuck_marginal e pr (next ω) (h2 ω)⟩
    rw [prob_equiv e (fun ω => same ω ∧ ¬right ω ∧ 0 < d ω), prob_equiv e (fun ω => ¬right ω ∧ 0 < d ω),
      Stages.prob_prod, Stages.prob_prod, ← expect_mul_left]
    exact expect_mono fun z => h1 z


end SharedNoise

open SharedNoise

theorem CondReceiptLuckMarginal : Accounting.Guarantees.TS.CondReceiptLuckMarginal := by
  intro Ω Z O _ _ _ e pr n G h
  exact condReceiptLuck_marginal e pr G h

theorem GridLuck : Accounting.Guarantees.TS.GridLuck := by
  intro Rw Cl _ _ r n w dead lucky p q θ κ
  exact SharedNoise.gridLuck w dead lucky p q θ κ

theorem GridTemplateCondition : Accounting.Guarantees.TS.GridTemplateCondition := by
  intro e he w hw
  have hV := SharedNoise.grid_point w hw
  have hq1 : (1 / 2 : ℝ) ^ w ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  have hq0 : 0 ≤ (1 / 2 : ℝ) ^ w := by positivity
  set B := 1 / 2 * Real.exp (1 / 256 * (w : ℝ)) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * (w : ℝ))) with hB
  have hB0 : 0 < B := by positivity
  rcases le_total e 64 with h | h
  · rw [max_eq_left h]
    have hx : (1 : ℝ) / (4 * 64) = 1 / 256 := by norm_num
    rw [hx, ← hB]
    set t := e / 64 with ht
    have ht0 : 0 ≤ t := by positivity
    have ht1 : t ≤ 1 := by rw [ht, div_le_one (by norm_num)]; exact h
    have h1 : Real.exp (e * (1 / 256) * w) ≤ (1 - t) + t * Real.exp (w / 4) := by
      have := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ ((w : ℝ) / 4)) (sub_nonneg.2 ht1) ht0 (by ring)
      simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at this
      calc Real.exp (e * (1 / 256) * w) = Real.exp (t * (w / 4)) := by congr 1; rw [ht]; ring
        _ ≤ _ := this
    have h2 : B ^ e ≤ (1 - t) + t * B ^ 64 := by
      have := convexOn_exp.2 (Set.mem_univ 0) (Set.mem_univ (64 * Real.log B)) (sub_nonneg.2 ht1) ht0 (by ring)
      simp only [smul_eq_mul, mul_zero, zero_add, Real.exp_zero, mul_one] at this
      have h64 : Real.exp (64 * Real.log B) = B ^ 64 := by
        rw [show (64 : ℝ) = ((64 : ℕ) : ℝ) by norm_num, Real.exp_nat_mul, Real.exp_log hB0]
      rw [h64] at this
      calc B ^ e = Real.exp (t * (64 * Real.log B)) := by
            rw [Real.rpow_def_of_pos hB0]; congr 1; rw [ht]; ring
        _ ≤ _ := this
    calc (1 / 2 : ℝ) ^ w * Real.exp (e * (1 / 256) * w) + (1 - (1 / 2 : ℝ) ^ w) * B ^ e
        ≤ (1 / 2 : ℝ) ^ w * ((1 - t) + t * Real.exp (w / 4)) + (1 - (1 / 2 : ℝ) ^ w) * ((1 - t) + t * B ^ 64) :=
          add_le_add (mul_le_mul_of_nonneg_left h1 hq0) (mul_le_mul_of_nonneg_left h2 (sub_nonneg.2 hq1))
      _ = (1 - t) + t * ((1 / 2 : ℝ) ^ w * Real.exp (w / 4) + (1 - (1 / 2 : ℝ) ^ w) * B ^ 64) := by ring
      _ ≤ (1 - t) + t * 1 := by gcongr
      _ = 1 := by ring
  · rw [max_eq_right h]
    have he0 : e ≠ 0 := he.ne'
    set l := 64 / e with hl
    have hl0 : 0 ≤ l := by positivity
    have hl1 : l ≤ 1 := by rw [hl, div_le_one he]; exact h
    have hexp1 : e * (1 / (4 * e)) * w = (w : ℝ) / 4 := by field_simp
    have hu : Real.exp (1 / (4 * e) * w) = Real.exp (1 / 256 * w) ^ l := by
      rw [← Real.exp_mul]; congr 1; rw [hl]; field_simp; ring
    have hv : Real.exp (-(4 * (1 / (4 * e)) * w)) = Real.exp (-(4 * (1 / 256) * w)) ^ l := by
      rw [← Real.exp_mul]; congr 1; rw [hl]; field_simp; ring
    have hconc : 1 / 2 * Real.exp (1 / 256 * w) ^ l + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * w)) ^ l ≤ B ^ l := by
      have := (Real.concaveOn_rpow hl0 hl1).2 (Set.mem_Ici.2 (Real.exp_pos (1 / 256 * (w : ℝ))).le)
        (Set.mem_Ici.2 (Real.exp_pos (-(4 * (1 / 256) * (w : ℝ)))).le) (by norm_num : (0 : ℝ) ≤ 1 / 2)
        (by norm_num : (0 : ℝ) ≤ 1 - 1 / 2) (by norm_num)
      simpa only [smul_eq_mul, hB] using this
    have hpow : (B ^ l) ^ e = B ^ 64 := by
      rw [← Real.rpow_mul hB0.le, hl, div_mul_cancel₀ _ he0]
      exact_mod_cast Real.rpow_natCast B 64
    rw [hexp1, hu, hv]
    calc (1 / 2 : ℝ) ^ w * Real.exp (w / 4) + (1 - (1 / 2 : ℝ) ^ w) *
          (1 / 2 * Real.exp (1 / 256 * w) ^ l + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * w)) ^ l) ^ e
        ≤ (1 / 2 : ℝ) ^ w * Real.exp (w / 4) + (1 - (1 / 2 : ℝ) ^ w) * (B ^ l) ^ e := by
          gcongr
      _ = _ := by rw [hpow]
      _ ≤ 1 := hV


theorem GridTemplateMoment : Accounting.Guarantees.TS.GridTemplateMoment := by
  intro Rw Cl _ _ r n w dead lucky hr hn hw hread hl hd
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have hnpos : (0 : ℝ) < n := by exact_mod_cast hn
  have he0 : 0 < (n : ℝ) / r := div_pos hnpos hrpos
  have hmax : max (64 * (r : ℝ)) n = r * max 64 ((n : ℝ) / r) := by
    rw [mul_max_of_nonneg _ _ hrpos.le, mul_div_cancel₀ _ hrpos.ne', mul_comm]
  have hrθ : (r : ℝ) * (1 / (4 * max (64 * (r : ℝ)) n)) = 1 / (4 * max 64 ((n : ℝ) / r)) := by
    rw [hmax]; field_simp
  have hnθ : (n : ℝ) * (1 / (4 * max (64 * (r : ℝ)) n)) = (n : ℝ) / r * (1 / (4 * max 64 ((n : ℝ) / r))) := by
    rw [← hrθ]; field_simp
  exact SharedNoise.gridLuck (fun i j => (w i j : ℝ)) dead lucky (fun _ _ => 1 / 2) (fun i j => (1 / 2 : ℝ) ^ w i j)
    (1 / (4 * max (64 * (r : ℝ)) n)) 4 hn hr (by positivity) (by norm_num) (fun _ _ => Nat.cast_nonneg _)
    (fun _ _ => ⟨by norm_num, by norm_num⟩) hread hl hd fun i j => by
      rw [hnθ, hrθ]
      exact GridTemplateCondition _ he0 (w i j) (hw i j)

end Accounting.SecurityProofs.TS
