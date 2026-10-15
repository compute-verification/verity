import Proofs.Flock.Claims.Finder

/-!
# Bits from KS, and soundness from KS

`B`'s two runs make at most `2t` SHA-512 calls when each run makes at most `t` (both parties'), so the strict birthday
bound gives `Pr[B] ≤ (2t)²/2^513`, and KS gives
`Pr[acc ∧ ¬R] ≤ κ + t·√γ·2^{−255.5} ≤ t · 2 · max(κ, √γ·2^{−255.5}) ≤ t · 2^{−λ}` at
`λ = min(log₂(1/κ), 255.5 − ½ log₂ γ) − 1` (`Bits_of_KS`). Flock's `Bits` follows from Flock's `KS`
(`Flock.Bits_of_KS`). Soundness is a corollary of KS under `R(x, ⊥) := False` (`sound_of_KS`).
-/

namespace Claims

/-- **KS and the birthday bound give the bits.** If `p ≤ κ + √(γ · q²/2^513)` with `q = 2t`, then
`p ≤ t · 2 · max(κ, √γ · 2^{−255.5})`. -/
theorem bits_of_ks {p κ γ t : ℝ} (hκ : 0 ≤ κ) (hγ : 0 ≤ γ) (ht : 1 ≤ t)
    (h : p ≤ κ + Real.sqrt (γ * ((2 * t) ^ 2 / 2 ^ (513 : ℝ)))) :
    p ≤ t * (2 * max κ (Real.sqrt γ * 2 ^ (-(255.5 : ℝ)))) := by
  have h513 : Real.sqrt ((2 : ℝ) ^ (513 : ℝ)) = 2 ^ (256.5 : ℝ) := by
    rw [Real.sqrt_eq_iff_mul_self_eq (by positivity) (by positivity), ← Real.rpow_add (by norm_num)]
    norm_num
  have hs : Real.sqrt (γ * ((2 * t) ^ 2 / 2 ^ (513 : ℝ))) = t * (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) := by
    rw [Real.sqrt_mul hγ, Real.sqrt_div (by positivity), Real.sqrt_sq (by linarith), h513]
    have : (2 : ℝ) ^ (-(255.5 : ℝ)) = 2 / 2 ^ (256.5 : ℝ) := by
      rw [Real.rpow_neg (by norm_num), show (256.5 : ℝ) = 255.5 + 1 by norm_num,
        Real.rpow_add (by norm_num), Real.rpow_one]
      field_simp
    rw [this]; ring
  rw [hs] at h
  have h1 : κ ≤ t * κ := le_mul_of_one_le_left hκ ht
  have hm1 : κ ≤ max κ (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) := le_max_left _ _
  have hm2 : Real.sqrt γ * 2 ^ (-(255.5 : ℝ)) ≤ max κ (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) := le_max_right _ _
  have ht0 : 0 ≤ t := by linarith
  nlinarith [mul_le_mul_of_nonneg_left hm1 ht0, mul_le_mul_of_nonneg_left hm2 ht0]

/-- `2 · max(κ, √γ·2^{−255.5}) = 2^{−λ}` at `λ = lamOf κ γ`, for `κ, γ > 0`. -/
theorem two_max_eq {κ γ : ℝ} (hκ : 0 < κ) (hγ : 0 < γ) :
    2 * max κ (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) = 2 ^ (-lamOf κ γ) := by
  unfold lamOf
  set a := Real.logb 2 (1 / κ) with ha
  set b := 255.5 - Real.logb 2 γ / 2 with hb
  have hk : κ = 2 ^ (-a) := by
    rw [ha, one_div, Real.logb_inv, neg_neg, Real.rpow_logb (by norm_num) (by norm_num) hκ]
  have hg : Real.sqrt γ * 2 ^ (-(255.5 : ℝ)) = 2 ^ (-b) := by
    have : (2 : ℝ) ^ (-b) = 2 ^ (Real.logb 2 γ * (1 / 2)) * 2 ^ (-(255.5 : ℝ)) := by
      rw [← Real.rpow_add (by norm_num)]; congr 1; rw [hb]; ring
    rw [this, Real.rpow_mul (by norm_num), Real.rpow_logb (by norm_num) (by norm_num) hγ, Real.sqrt_eq_rpow]
  have hmax : max ((2 : ℝ) ^ (-a)) (2 ^ (-b)) = 2 ^ (-(min a b)) := by
    rcases le_total a b with h | h
    · rw [min_eq_left h, max_eq_left (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith))]
    · rw [min_eq_right h, max_eq_right (Real.rpow_le_rpow_of_exponent_le (by norm_num) (by linarith))]
  rw [hg]
  nth_rewrite 1 [hk]
  rw [hmax, show -(min a b - 1) = 1 + -(min a b) by ring, Real.rpow_add (by norm_num), Real.rpow_one]

/-- `2 · max(κ, √γ·2^{−255.5}) ≤ 2^{−λ}` at `λ = lamOf κ γ`, for `κ > 0` and `γ ≥ 0` (`γ = 0` reads `log₂ 0` as `0`). -/
theorem two_max_le {κ γ : ℝ} (hκ : 0 < κ) (hγ : 0 ≤ γ) :
    2 * max κ (Real.sqrt γ * 2 ^ (-(255.5 : ℝ))) ≤ 2 ^ (-lamOf κ γ) := by
  rcases hγ.lt_or_eq with hγ | rfl
  · exact (two_max_eq hκ hγ).le
  · unfold lamOf
    have hk : κ = 2 ^ (-Real.logb 2 (1 / κ)) := by
      rw [one_div, Real.logb_inv, neg_neg, Real.rpow_logb (by norm_num) (by norm_num) hκ]
    simp only [Real.sqrt_zero, zero_mul, Real.logb_zero, zero_div, sub_zero]
    rw [max_eq_left hκ.le, show -(min (Real.logb 2 (1 / κ)) 255.5 - 1) = 1 + -(min (Real.logb 2 (1 / κ)) 255.5)
      by ring, Real.rpow_add (by norm_num), Real.rpow_one]
    refine mul_le_mul_of_nonneg_left ?_ (by norm_num)
    nth_rewrite 1 [hk]
    exact Real.rpow_le_rpow_of_exponent_le (by norm_num) (neg_le_neg (min_le_left _ _))

section KS

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Bits from KS**: KS with `κ > 0` and `γ ≥ 0`, and the strict birthday bound for `B`, give `λ = lamOf κ γ` bits. -/
theorem Bits_of_KS {H : Msg → Msg} {V : Verifier Coin} {R : Rel W} {E : Extractor Coin W V} {κ γ : Msg → ℝ}
    (h : KS H V R E κ γ) (hκ : ∀ x, 0 < κ x) (hγ : ∀ x, 0 ≤ γ x) :
    Bits H V R E fun x => lamOf (κ x) (γ x) := by
  intro x Ω _ _ P t ht hrun hSB
  have hks := h x Ω P
  have ht1 : (1 : ℝ) ≤ t := by exact_mod_cast ht
  have hB : PrB H V P x ≤ (2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ) := by
    unfold PrB
    split
    · rename_i hk
      refine hSB hk (2 * t) fun s => ?_
      exact_mod_cast B_cost_le H V P x hk t hrun s
    · positivity
  have hks' : PrBad H V R E x P ≤ κ x + Real.sqrt (γ x * ((2 * (t : ℝ)) ^ 2 / 2 ^ (513 : ℝ))) :=
    hks.trans (by gcongr; exact hγ x)
  refine (bits_of_ks (hκ x).le (hγ x) ht1 hks').trans ?_
  exact mul_le_mul_of_nonneg_left (two_max_le (hκ x) (hγ x)) (by linarith)

/-- **Soundness is a corollary of KS** (`R(x, ⊥) := False`): at `x ∉ L_R`, `Pr[acc] ≤ κ(x) + √(γ(x)·Pr[B])`. -/
theorem sound_of_KS {H : Msg → Msg} {V : Verifier Coin} {R : Rel W} {E : Extractor Coin W V} {κ γ : Msg → ℝ}
    (h : KS H V R E κ γ) (x : Msg) (hx : ¬ R.L x) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) :
    Pr (fun s : Ω × (Fin (V.k x) → Coin) => (run H P V x s.1 s.2).out = .acc) ≤
      κ x + Real.sqrt (γ x * PrB H V P x) := by
  have hR : ∀ o, ¬ R.ext x o := by
    intro o; cases o with
    | none => exact id
    | some w => exact fun hw => hx ⟨w, hw⟩
  have e := h x Ω P
  unfold PrBad at e
  rw [Pr_congr (q := fun s : Ω × (Fin (V.k x) → Coin) × E.C x => (run H P V x s.1 s.2.1).out = .acc)
    (fun s => ⟨fun h => h.1, fun h' => ⟨h', hR _⟩⟩)] at e
  rw [Pr_drop (fun (ω : Ω) (r : Fin (V.k x) → Coin) => (run H P V x ω r).out = .acc)] at e
  exact e

end KS

namespace Flock

theorem κ_pos {Coin Col T : Type} (D : Deploy Coin Col T) (x : Msg) (hε : 0 ≤ D.εtable x) : 0 < κ D x := by
  unfold κ
  have h1 : (0 : ℝ) < 2 ^ (-(D.waits x : ℝ)) := by positivity
  have h2 : (0 : ℝ) ≤ 2 * D.k x / D.cap x := by positivity
  have h3 : 0 < Real.exp (-(D.k x : ℝ) / 8) := Real.exp_pos _
  linarith

theorem γ_nonneg {Coin Col T : Type} (D : Deploy Coin Col T) (x : Msg) : 0 ≤ γ D x := by
  unfold γ; positivity

/-- **Flock: bits from KS**: Flock's KS, with the table error nonnegative, gives Flock's bits. -/
theorem Bits_of_KS {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    {H : Msg → Msg} {D : Deploy Coin Col T} (h : KS H D) (hε : ∀ x, 0 ≤ D.εtable x) : Bits H D :=
  fun decode hdec => Claims.Bits_of_KS (h decode hdec) (fun x => κ_pos D x (hε x)) (γ_nonneg D)

end Flock

end Claims
