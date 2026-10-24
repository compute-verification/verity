import Definitions.Accounting.Guarantees.TS.Grind
import Proofs.Accounting.TS.Steer
import Mathlib.Analysis.MeanInequalitiesPow

/-!
# The grinding premium under hashed seeds: proofs

`ExchangeTail` is Markov at `e^{θπ}`. `GrindTail` is `Steer.GrindUnion`'s split: query `q`'s answer split off the coins
(`Steer.splitAt`), its event bounded over that answer with the rest fixed, then a union over the `T` queries.
`GrindPremium` is the two, plus `Break` by a union; `GrindPremiumAt` sets `π` so that `T·M·e^{-θπ} = δ`. `GrindMass`
sums the same split's per-query bound instead of taking a union.

`LuckMoment` splits the exponential at the luck indicator (`Grind.exp_gain`) and counts the lucky coins. `PrivateMoment`
is Fubini over the private coins (`Grind.sum_exp_pi`). `TileLuck` takes both AM-GM steps:

* pointwise in a column's coins, `∏_i e^{θ g_i} ≤ (1/r)·Σ_i e^{rθ g_i}` (`Grind.prod_exp_le_mean`), so a column of `r`
  units is at most the mean of its units each at rate `rθ`, which `LuckMoment` bounds by `f` when the unit is alive and
  `e^{rθw}` when dead;
* pointwise in the shared coins, `∏_k H_k ≤ (1/n)·Σ_k H_k^n` (`Grind.prod_le_mean_pow`), and `H_k^n` is at most the
  mean of its units' `n`th powers (the power mean), each of which is `f^n` off its dead event and `e^{n·rθw}` on it.
-/

namespace Accounting.SecurityProofs.TS

open Pouw.RelativeSecurity

namespace Grind

theorem card_filter_le {Ω : Type} [Fintype Ω] (P : Ω → Prop) [DecidablePred P] {p : ℝ} (h : prob P ≤ p) :
    ((Finset.univ.filter P).card : ℝ) ≤ p * Fintype.card Ω := by
  unfold prob at h
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · have : IsEmpty Ω := Fintype.card_eq_zero_iff.mp h0
    simp
  · have hc : (0 : ℝ) < Fintype.card Ω := by exact_mod_cast h0
    rw [div_le_iff₀ hc] at h
    convert h using 3
    ext; simp

theorem update_split {Ω₀ X : Type} {T : ℕ} (q : Fin T) (z : Ω₀ × ({j // j ≠ q} → X)) (x x₀ : X) :
    ((Steer.splitAt (Ω₀ := Ω₀) q).symm (z, x)).2 = Function.update ((Steer.splitAt (Ω₀ := Ω₀) q).symm (z, x₀)).2 q x := by
  funext j
  by_cases h : j = q
  · subst h; simp [Steer.splitAt]
  · simp [Steer.splitAt, h]

theorem exp_gain (w θ κ : ℝ) (b : Prop) [Decidable b] :
    Real.exp (θ * (w * ((1 + κ) * (if b then 1 else 0) - κ))) =
      Real.exp (-(κ * θ * w)) + (Real.exp (θ * w) - Real.exp (-(κ * θ * w))) * (if b then 1 else 0) := by
  split_ifs
  · ring_nf
  · ring_nf

theorem sum_exp_pi {R ι : Type} {E : ι → Type} [Fintype R] [Fintype ι] [DecidableEq ι] [∀ i, Fintype (E i)]
    (g : ∀ i, R → E i → ℝ) (θ : ℝ) :
    ∑ ω : R × (∀ i, E i), Real.exp (θ * ∑ i, g i ω.1 (ω.2 i)) = ∑ r, ∏ i, ∑ x, Real.exp (θ * g i r x) := by
  rw [Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [Fintype.prod_sum]
  refine Finset.sum_congr rfl fun e _ => ?_
  rw [Finset.mul_sum, Real.exp_sum]

theorem prod_le_mean_pow {n : ℕ} (hn : 0 < n) (h : Fin n → ℝ) (h0 : ∀ k, 0 ≤ h k) :
    ∏ k, h k ≤ (∑ k, h k ^ n) / n := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
  have hw : ∑ _k : Fin n, (n : ℝ)⁻¹ = 1 := by
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_inv_cancel₀ hn']
  have := Real.geom_mean_le_arith_mean_weighted (Finset.univ : Finset (Fin n)) (fun _ => (n : ℝ)⁻¹)
    (fun k => h k ^ n) (fun _ _ => inv_nonneg.mpr (Nat.cast_nonneg n)) hw fun k _ => pow_nonneg (h0 k) n
  simp only [Real.pow_rpow_inv_natCast (h0 _) hn.ne'] at this
  rw [div_eq_inv_mul, Finset.mul_sum]
  exact this

theorem prod_exp_le_mean {r : ℕ} (hr : 0 < r) (θ : ℝ) (g : Fin r → ℝ) :
    Real.exp (θ * ∑ i, g i) ≤ (∑ i, Real.exp (r * θ * g i)) / r := by
  rw [Finset.mul_sum, Real.exp_sum]
  calc ∏ i, Real.exp (θ * g i) ≤ (∑ i, Real.exp (θ * g i) ^ r) / r :=
        prod_le_mean_pow hr _ fun i => (Real.exp_pos _).le
    _ = (∑ i, Real.exp (r * θ * g i)) / r := by
        congr 1; refine Finset.sum_congr rfl fun i _ => ?_
        rw [← Real.exp_nat_mul]; ring_nf

theorem exp_neg_le (a : ℝ) (ha : 0 ≤ a) : Real.exp (-a) ≤ 1 / (1 + a) := by
  rw [Real.exp_neg, ← one_div]
  exact one_div_le_one_div_of_le (by linarith) (by linarith [Real.add_one_le_exp a])

/-- A dead word's term: `2^{-w}·e^{w/8} ≤ (4/7)^w`, from `e^{1/8} < 8/7`. -/
theorem dead_le (w : ℕ) : (1 / 2 : ℝ) ^ w * Real.exp (32 * (1 / 256 * w)) ≤ (4 / 7) ^ w := by
  have h8 : Real.exp (1 / 8) ≤ 8 / 7 := by
    have := Real.exp_bound_div_one_sub_of_interval' (x := 1 / 8) (by norm_num) (by norm_num)
    norm_num at this ⊢
    linarith
  have he : Real.exp (32 * (1 / 256 * w)) = Real.exp (1 / 8) ^ w := by
    rw [← Real.exp_nat_mul]
    ring_nf
  rw [he, ← mul_pow]
  exact pow_le_pow_left₀ (by positivity) (by linarith) w

/-- An alive word's term over its 32 columns, from `e^x ≤ 1 + x + x²` on `|x| ≤ 1` and `1 + x ≤ e^x`. -/
theorem alive_le (x : ℝ) (h0 : 0 ≤ x) (h32 : x ≤ 32) :
    (1 / 2 * Real.exp (1 / 256 * x) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * x))) ^ 32 ≤
      Real.exp (-(3 / 16) * x + 17 / 4096 * x ^ 2) := by
  set s := 1 / 256 * x with hs
  have hs0 : 0 ≤ s := by positivity
  have hs1 : s ≤ 1 / 8 := by rw [hs]; linarith
  have e1 : Real.exp s ≤ 1 + s + s ^ 2 := by
    have := Real.abs_exp_sub_one_sub_id_le (x := s) (by rw [abs_of_nonneg hs0]; linarith)
    linarith [le_abs_self (Real.exp s - 1 - s)]
  have e2 : Real.exp (-(4 * s)) ≤ 1 - 4 * s + 16 * s ^ 2 := by
    have := Real.abs_exp_sub_one_sub_id_le (x := -(4 * s)) (by rw [abs_neg, abs_of_nonneg (by linarith)]; linarith)
    nlinarith [le_abs_self (Real.exp (-(4 * s)) - 1 - -(4 * s))]
  have h4 : -(4 * (1 / 256) * x) = -(4 * s) := by rw [hs]; ring
  rw [h4]
  have hg0 : 0 ≤ 1 / 2 * Real.exp s + (1 - 1 / 2) * Real.exp (-(4 * s)) := by positivity
  have hg : 1 / 2 * Real.exp s + (1 - 1 / 2) * Real.exp (-(4 * s)) ≤ Real.exp (-(3 / 2) * s + 17 / 2 * s ^ 2) := by
    have := Real.add_one_le_exp (-(3 / 2) * s + 17 / 2 * s ^ 2)
    nlinarith
  calc _ ≤ Real.exp (-(3 / 2) * s + 17 / 2 * s ^ 2) ^ 32 := pow_le_pow_left₀ hg0 hg 32
    _ = _ := by
      rw [← Real.exp_nat_mul]
      congr 1
      rw [hs]
      push_cast
      ring

end Grind

open Grind

theorem ExchangeTail : Accounting.Guarantees.TS.ExchangeTail := by
  classical
  intro B _ f θ M π hθ hM h
  have hmass : expect (fun b => Real.exp (θ * f b)) ≤ M := by
    unfold expect
    rcases Nat.eq_zero_or_pos (Fintype.card B) with h0 | h0
    · simp [h0, hM]
    · rw [div_le_iff₀ (by exact_mod_cast h0)]; exact h
  have hm := markov (fun b => (Real.exp_pos (θ * f b)).le) (Real.exp (θ * π))
  have hsub : prob (fun b => π ≤ f b) ≤ prob (fun b => Real.exp (θ * π) ≤ Real.exp (θ * f b)) :=
    prob_mono fun b hb => Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hb hθ)
  have hpos := Real.exp_pos (θ * π)
  rw [Real.exp_neg, ← div_eq_mul_inv, le_div_iff₀ hpos]
  calc prob (fun b => π ≤ f b) * Real.exp (θ * π)
      ≤ prob (fun b => Real.exp (θ * π) ≤ Real.exp (θ * f b)) * Real.exp (θ * π) :=
        mul_le_mul_of_nonneg_right hsub hpos.le
    _ = Real.exp (θ * π) * prob (fun b => Real.exp (θ * π) ≤ Real.exp (θ * f b)) := mul_comm _ _
    _ ≤ M := hm.trans hmass

theorem GrindTail : Accounting.Guarantees.TS.GrindTail := by
  classical
  intro Ω₀ B _ _ T Bad τ hτ hq
  have hq' : ∀ q, prob (fun ω : Ω₀ × (Fin T → B) => Bad q ω.1 ω.2) ≤ τ := by
    intro q
    rcases isEmpty_or_nonempty B with hB | hne
    · have : IsEmpty (Ω₀ × (Fin T → B)) := ⟨fun ω => hB.false (ω.2 q)⟩
      have h0 : prob (fun ω : Ω₀ × (Fin T → B) => Bad q ω.1 ω.2) = 0 := by unfold prob; simp
      rw [h0]; exact hτ
    obtain ⟨b₀⟩ := hne
    set Φ := Steer.splitAt (Ω₀ := Ω₀) (X := B) q
    rw [prob_equiv Φ (fun ω : Ω₀ × (Fin T → B) => Bad q ω.1 ω.2), Stages.prob_prod]
    refine expect_le_of_le hτ fun z => ?_
    calc prob (fun b => Bad q (Φ.symm (z, b)).1 (Φ.symm (z, b)).2)
        = prob (fun b => Bad q z.1 (Function.update (Φ.symm (z, b₀)).2 q b)) := by
          congr 1; funext b; rw [update_split q z b b₀]; rfl
      _ ≤ τ := hq q z.1 _
  calc prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, Bad q ω.1 ω.2)
      ≤ prob (fun ω => ∃ q ∈ Finset.univ, Bad q ω.1 ω.2) :=
        prob_mono fun ω ⟨q, h⟩ => ⟨q, Finset.mem_univ _, h⟩
    _ ≤ ∑ q ∈ (Finset.univ : Finset (Fin T)), prob (fun ω : Ω₀ × (Fin T → B) => Bad q ω.1 ω.2) :=
        prob_exists_le _ _
    _ ≤ ∑ _q ∈ (Finset.univ : Finset (Fin T)), τ := Finset.sum_le_sum fun q _ => hq' q
    _ = T * τ := by rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

theorem GrindPremium : Accounting.Guarantees.TS.GrindPremium := by
  classical
  intro Ω₀ B _ _ T g R Break θ M π δb hθ hM hg hB hR
  have htail := GrindTail (fun q ω₀ a => π ≤ g q ω₀ a) (M * Real.exp (-(θ * π)))
    (mul_nonneg hM (Real.exp_pos _).le)
    fun q ω₀ a => ExchangeTail (fun b => g q ω₀ (Function.update a q b)) θ M π hθ hM (hg q ω₀ a)
  calc prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, π ≤ R q ω.1 ω.2)
      ≤ prob (fun ω => Break ω ∨ ∃ q, π ≤ g q ω.1 ω.2) := by
        refine prob_mono fun ω ⟨q, h⟩ => ?_
        by_cases hb : Break ω
        · exact Or.inl hb
        · exact Or.inr ⟨q, hR ω q hb ▸ h⟩
    _ ≤ prob Break + prob (fun ω : Ω₀ × (Fin T → B) => ∃ q, π ≤ g q ω.1 ω.2) := prob_or_le _ _
    _ ≤ δb + T * (M * Real.exp (-(θ * π))) := add_le_add hB htail
    _ = T * M * Real.exp (-(θ * π)) + δb := by ring

theorem GrindMass : Accounting.Guarantees.TS.GrindMass := by
  classical
  intro Ω₀ B _ _ T g θ M hg
  have hq : ∀ q, ∑ ω : Ω₀ × (Fin T → B), Real.exp (θ * g q ω.1 ω.2) ≤ M * Fintype.card (Ω₀ × (Fin T → B)) := by
    intro q
    rcases isEmpty_or_nonempty B with hB | hne
    · have : IsEmpty (Ω₀ × (Fin T → B)) := ⟨fun ω => hB.false (ω.2 q)⟩
      simp
    obtain ⟨b₀⟩ := hne
    set Φ := Steer.splitAt (Ω₀ := Ω₀) (X := B) q
    calc ∑ ω : Ω₀ × (Fin T → B), Real.exp (θ * g q ω.1 ω.2)
        = ∑ z : Ω₀ × ({j // j ≠ q} → B), ∑ b : B, Real.exp (θ * g q (Φ.symm (z, b)).1 (Φ.symm (z, b)).2) := by
          rw [← Φ.symm.sum_comp, Fintype.sum_prod_type]
      _ ≤ ∑ _z : Ω₀ × ({j // j ≠ q} → B), M * Fintype.card B := by
          refine Finset.sum_le_sum fun z _ => ?_
          calc ∑ b, Real.exp (θ * g q (Φ.symm (z, b)).1 (Φ.symm (z, b)).2)
              = ∑ b, Real.exp (θ * g q z.1 (Function.update (Φ.symm (z, b₀)).2 q b)) := by
                refine Finset.sum_congr rfl fun b _ => ?_
                rw [update_split q z b b₀]; rfl
            _ ≤ M * Fintype.card B := hg q z.1 _
      _ = M * Fintype.card (Ω₀ × (Fin T → B)) := by
          rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Fintype.card_congr Φ]
          simp only [Fintype.card_prod, Nat.cast_mul]; ring
  rw [Finset.sum_comm]
  calc ∑ q, ∑ ω : Ω₀ × (Fin T → B), Real.exp (θ * g q ω.1 ω.2)
      ≤ ∑ _q : Fin T, M * Fintype.card (Ω₀ × (Fin T → B)) := Finset.sum_le_sum fun q _ => hq q
    _ = T * M * Fintype.card (Ω₀ × (Fin T → B)) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; ring

theorem LuckMoment : Accounting.Guarantees.TS.LuckMoment := by
  classical
  intro Ω _ lucky p w θ κ hw hθ hκ hp
  have hle : Real.exp (-(κ * θ * w)) ≤ Real.exp (θ * w) :=
    Real.exp_le_exp.mpr (by nlinarith [mul_nonneg (mul_nonneg hκ hθ) hw, mul_nonneg hθ hw])
  have hc := card_filter_le lucky hp
  simp_rw [exp_gain]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_boole, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  have hd : 0 ≤ Real.exp (θ * w) - Real.exp (-(κ * θ * w)) := sub_nonneg.mpr hle
  calc (Fintype.card Ω : ℝ) * Real.exp (-(κ * θ * w)) +
        (Real.exp (θ * w) - Real.exp (-(κ * θ * w))) * ((Finset.univ.filter lucky).card : ℝ)
      ≤ (Fintype.card Ω : ℝ) * Real.exp (-(κ * θ * w)) +
        (Real.exp (θ * w) - Real.exp (-(κ * θ * w))) * (p * Fintype.card Ω) := by
        gcongr
    _ = (p * Real.exp (θ * w) + (1 - p) * Real.exp (-(κ * θ * w))) * Fintype.card Ω := by ring

theorem PrivateMoment : Accounting.Guarantees.TS.PrivateMoment := by
  intro R ι E _ _ _ _ g θ M hM
  rw [sum_exp_pi, Fintype.card_prod, Fintype.card_pi, Nat.cast_mul, Nat.cast_prod]
  calc ∑ r, ∏ i, ∑ x, Real.exp (θ * g i r x)
      ≤ ∑ _r : R, ∏ i, (M i * Fintype.card (E i)) :=
        Finset.sum_le_sum fun r _ => Finset.prod_le_prod₀ (fun i _ => Finset.sum_nonneg fun x _ =>
          (Real.exp_pos _).le) fun i _ => hM i r
    _ = (∏ i, M i) * ((Fintype.card R : ℝ) * ∏ i, (Fintype.card (E i) : ℝ)) := by
        rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, Finset.prod_mul_distrib]; ring

theorem TileLuck : Accounting.Guarantees.TS.TileLuck := by
  classical
  intro Ω₀ E _ _ n r w dead lucky p q θ κ M hn hr hθ hκ hw hp halive hdead hcond
  set gain : Fin n → Fin r → Ω₀ → E → ℝ := fun k i ω₀ x =>
    w k i * ((1 + κ) * (if lucky k i ω₀ x then 1 else 0) - κ) with hgain
  have hrθ : 0 ≤ (r : ℝ) * θ := mul_nonneg (Nat.cast_nonneg r) hθ
  set f : Fin n → Fin r → ℝ := fun k i =>
    p k i * Real.exp (r * θ * w k i) + (1 - p k i) * Real.exp (-(κ * (r * θ) * w k i)) with hf
  set h : Fin n → Fin r → Ω₀ → ℝ := fun k i ω₀ => if dead k i ω₀ then Real.exp (r * θ * w k i) else f k i with hh
  have hle : ∀ k i, Real.exp (-(κ * (r * θ) * w k i)) ≤ Real.exp (r * θ * w k i) := fun k i =>
    Real.exp_le_exp.mpr (by nlinarith [mul_nonneg (mul_nonneg hκ hrθ) (hw k i), mul_nonneg hrθ (hw k i)])
  have hf0 : ∀ k i, 0 ≤ f k i := fun k i => by
    have := (hp k i).1; have := (hp k i).2
    simp only [hf]; positivity
  have hfe : ∀ k i, f k i ≤ Real.exp (r * θ * w k i) := fun k i => by
    simp only [hf]; nlinarith [hle k i, (hp k i).2]
  have hh0 : ∀ k i ω₀, 0 ≤ h k i ω₀ := fun k i ω₀ => by
    simp only [hh]; split_ifs
    · exact (Real.exp_pos _).le
    · exact hf0 k i
  -- one unit at rθ, given the shared coins
  have hunit : ∀ k i ω₀, ∑ x : E, Real.exp (r * θ * gain k i ω₀ x) ≤ h k i ω₀ * Fintype.card E := by
    intro k i ω₀
    by_cases hd : dead k i ω₀
    · simp only [hh, hd, ↓reduceIte]
      calc ∑ x : E, Real.exp (r * θ * gain k i ω₀ x) ≤ ∑ _x : E, Real.exp (r * θ * w k i) := by
            refine Finset.sum_le_sum fun x _ => Real.exp_le_exp.mpr ?_
            refine mul_le_mul_of_nonneg_left ?_ hrθ
            have : (1 + κ) * (if lucky k i ω₀ x then (1 : ℝ) else 0) - κ ≤ 1 := by split_ifs <;> linarith
            simp only [hgain]; nlinarith [hw k i]
        _ = Real.exp (r * θ * w k i) * Fintype.card E := by
            rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_comm]
    · simp only [hh, hd, ↓reduceIte]
      have := LuckMoment (lucky k i ω₀) (p k i) (w k i) (r * θ) κ (hw k i) hrθ hκ (halive k i ω₀ hd)
      calc _ ≤ _ := this
        _ = f k i * Fintype.card E := by simp only [hf]
  -- a column's r units share its coins: AM-GM over them, pointwise in the coins
  set H : Fin n → Ω₀ → ℝ := fun k ω₀ => (∑ i, h k i ω₀) / r with hH
  have hrpos : (0 : ℝ) < r := by exact_mod_cast hr
  have hcol : ∀ k ω₀, ∑ x : E, Real.exp (θ * ∑ i, gain k i ω₀ x) ≤ H k ω₀ * Fintype.card E := by
    intro k ω₀
    calc ∑ x : E, Real.exp (θ * ∑ i, gain k i ω₀ x)
        ≤ ∑ x : E, (∑ i, Real.exp (r * θ * gain k i ω₀ x)) / r :=
          Finset.sum_le_sum fun x _ => prod_exp_le_mean hr θ _
      _ = (∑ i, ∑ x : E, Real.exp (r * θ * gain k i ω₀ x)) / r := by
          rw [← Finset.sum_div, Finset.sum_comm]
      _ ≤ (∑ i, h k i ω₀ * Fintype.card E) / r := by
          gcongr with i; exact hunit k i ω₀
      _ = H k ω₀ * Fintype.card E := by simp only [hH]; rw [← Finset.sum_mul]; ring
  have hH0 : ∀ k ω₀, 0 ≤ H k ω₀ := fun k ω₀ => by
    simp only [hH]; exact div_nonneg (Finset.sum_nonneg fun i _ => hh0 k i ω₀) hrpos.le
  -- the shared coins: each unit's dead mass
  have hpow : ∀ k i, ∑ ω₀, h k i ω₀ ^ n ≤ M * Fintype.card Ω₀ := by
    intro k i
    have hc := card_filter_le (dead k i) (hdead k i)
    have hsplit : ∀ ω₀, h k i ω₀ ^ n =
        f k i ^ n + (Real.exp (r * θ * w k i) ^ n - f k i ^ n) * (if dead k i ω₀ then 1 else 0) := by
      intro ω₀; simp only [hh]; split_ifs <;> ring
    have hd : 0 ≤ Real.exp (r * θ * w k i) ^ n - f k i ^ n :=
      sub_nonneg.mpr (pow_le_pow_left₀ (hf0 k i) (hfe k i) n)
    simp_rw [hsplit]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_boole, Finset.sum_const, Finset.card_univ,
      nsmul_eq_mul]
    have hcond' := hcond k i
    rw [Real.exp_nat_mul] at hcond'
    calc (Fintype.card Ω₀ : ℝ) * f k i ^ n +
          (Real.exp (r * θ * w k i) ^ n - f k i ^ n) * ((Finset.univ.filter (dead k i)).card : ℝ)
        ≤ (Fintype.card Ω₀ : ℝ) * f k i ^ n + (Real.exp (r * θ * w k i) ^ n - f k i ^ n) * (q k i * Fintype.card Ω₀) := by
          gcongr
      _ = (q k i * Real.exp (r * θ * w k i) ^ n + (1 - q k i) * f k i ^ n) * Fintype.card Ω₀ := by ring
      _ ≤ M * Fintype.card Ω₀ := by gcongr
  have hHpow : ∀ k, ∑ ω₀, H k ω₀ ^ n ≤ M * Fintype.card Ω₀ := by
    intro k
    have hmean : ∀ ω₀, H k ω₀ ^ n ≤ (∑ i, h k i ω₀ ^ n) / r := by
      intro ω₀
      have hw1 : ∑ _i : Fin r, (r : ℝ)⁻¹ = 1 := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_inv_cancel₀ hrpos.ne']
      have := Real.pow_arith_mean_le_arith_mean_pow (Finset.univ : Finset (Fin r)) (fun _ => (r : ℝ)⁻¹)
        (fun i => h k i ω₀) (fun _ _ => inv_nonneg.mpr hrpos.le) hw1 (fun i _ => hh0 k i ω₀) n
      simp only [hH]
      rw [div_eq_inv_mul, Finset.mul_sum, div_eq_inv_mul, Finset.mul_sum]
      exact this
    calc ∑ ω₀, H k ω₀ ^ n ≤ ∑ ω₀, (∑ i, h k i ω₀ ^ n) / r := Finset.sum_le_sum fun ω₀ _ => hmean ω₀
      _ = (∑ i, ∑ ω₀, h k i ω₀ ^ n) / r := by rw [← Finset.sum_div, Finset.sum_comm]
      _ ≤ (∑ _i : Fin r, M * (Fintype.card Ω₀ : ℝ)) / r := by gcongr with i; exact hpow k i
      _ = M * Fintype.card Ω₀ := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]; field_simp
  rw [sum_exp_pi (fun k ω₀ x => ∑ i, gain k i ω₀ x) θ]
  calc ∑ ω₀, ∏ k, ∑ x : E, Real.exp (θ * ∑ i, gain k i ω₀ x)
      ≤ ∑ ω₀, ∏ k, (H k ω₀ * Fintype.card E) :=
        Finset.sum_le_sum fun ω₀ _ => Finset.prod_le_prod₀ (fun k _ => Finset.sum_nonneg fun x _ =>
          (Real.exp_pos _).le) fun k _ => hcol k ω₀
    _ = (∑ ω₀, ∏ k, H k ω₀) * (Fintype.card E : ℝ) ^ n := by
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl fun ω₀ _ => ?_
        rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_univ, Fintype.card_fin]
    _ ≤ (∑ ω₀, (∑ k, H k ω₀ ^ n) / n) * (Fintype.card E : ℝ) ^ n := by
        gcongr with ω₀
        exact prod_le_mean_pow hn (fun k => H k ω₀) fun k => hH0 k ω₀
    _ = (∑ k, ∑ ω₀, H k ω₀ ^ n) / n * (Fintype.card E : ℝ) ^ n := by
        rw [← Finset.sum_div, Finset.sum_comm]
    _ ≤ (∑ _k : Fin n, M * (Fintype.card Ω₀ : ℝ)) / n * (Fintype.card E : ℝ) ^ n := by
        gcongr with k
        exact hHpow k
    _ = M * Fintype.card (Ω₀ × (Fin n → E)) := by
        rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Fintype.card_prod,
          Fintype.card_fun, Fintype.card_fin, Nat.cast_mul, Nat.cast_pow]
        have : (n : ℝ) ≠ 0 := by exact_mod_cast hn.ne'
        field_simp

theorem GrindPremiumAt : Accounting.Guarantees.TS.GrindPremiumAt := by
  intro Ω₀ B _ _ T g R Break θ M δ δb hθ hM hδ hg hB hR
  rcases Nat.eq_zero_or_pos T with hT | hT
  · subst hT
    have h0 : prob (fun ω : Ω₀ × (Fin 0 → B) => ∃ q, Real.log ((0 : ℕ) * M / δ) / θ ≤ R q ω.1 ω.2) = 0 := by
      unfold prob; simp
    rw [h0]; exact add_nonneg hδ.le ((prob_nonneg _).trans hB)
  have hTM : 0 < (T : ℝ) * M / δ := div_pos (mul_pos (by exact_mod_cast hT) hM) hδ
  have := GrindPremium g R Break θ M (Real.log (T * M / δ) / θ) δb hθ.le hM.le hg hB hR
  rw [mul_div_cancel₀ _ hθ.ne', Real.exp_neg, Real.exp_log hTM] at this
  calc _ ≤ _ := this
    _ = δ + δb := by field_simp

theorem TemplateCondition : Accounting.Guarantees.TS.TemplateCondition := by
  intro w hw
  have hx0 : (0 : ℝ) ≤ w := Nat.cast_nonneg w
  have hx32 : (w : ℝ) ≤ 32 := by exact_mod_cast hw
  have hq1 : (1 / 2 : ℝ) ^ w ≤ 1 := pow_le_one₀ (by norm_num) (by norm_num)
  have hq0 : 0 ≤ (1 / 2 : ℝ) ^ w := by positivity
  have key : (1 / 2 : ℝ) ^ w * Real.exp (32 * (1 / 256 * w)) +
      (1 - (1 / 2 : ℝ) ^ w) * (1 / 2 * Real.exp (1 / 256 * w) + (1 - 1 / 2) * Real.exp (-(4 * (1 / 256) * w))) ^ 32
        ≤ (4 / 7) ^ w + (1 - (1 / 2 : ℝ) ^ w) * Real.exp (-(3 / 16) * w + 17 / 4096 * (w : ℝ) ^ 2) :=
    add_le_add (dead_le w) (mul_le_mul_of_nonneg_left (alive_le w hx0 hx32) (by linarith))
  refine key.trans ?_
  rcases Nat.lt_or_ge w 3 with h | h
  · interval_cases w
    · norm_num
    · have he := exp_neg_le (751 / 4096) (by norm_num)
      have hx : -(3 / 16) * ((1 : ℕ) : ℝ) + 17 / 4096 * ((1 : ℕ) : ℝ) ^ 2 = -(751 / 4096) := by norm_num
      rw [hx]
      norm_num at he ⊢
      linarith
    · have he := exp_neg_le (1468 / 4096) (by norm_num)
      have hx : -(3 / 16) * ((2 : ℕ) : ℝ) + 17 / 4096 * ((2 : ℕ) : ℝ) ^ 2 = -(1468 / 4096) := by norm_num
      rw [hx]
      norm_num at he ⊢
      linarith
  · have h3 : (3 : ℝ) ≤ w := by exact_mod_cast h
    have hE : -(3 / 16) * (w : ℝ) + 17 / 4096 * (w : ℝ) ^ 2 ≤ -(1 / 2) := by
      nlinarith [mul_nonneg (sub_nonneg.2 h3) (sub_nonneg.2 hx32)]
    have he : Real.exp (-(3 / 16) * (w : ℝ) + 17 / 4096 * (w : ℝ) ^ 2) ≤ 2 / 3 :=
      calc _ ≤ Real.exp (-(1 / 2)) := Real.exp_le_exp.2 hE
        _ ≤ 1 / (1 + 1 / 2) := exp_neg_le _ (by norm_num)
        _ = 2 / 3 := by norm_num
    have hd : (4 / 7 : ℝ) ^ w ≤ (4 / 7) ^ 3 := pow_le_pow_of_le_one (by norm_num) (by norm_num) h
    nlinarith [mul_nonneg hq0 (Real.exp_pos (-(3 / 16) * (w : ℝ) + 17 / 4096 * (w : ℝ) ^ 2)).le]

theorem TemplateMoment : Accounting.Guarantees.TS.TemplateMoment := by
  intro Ω₀ E _ _ r w dead lucky hr hw hl hd
  have hr' : (0 : ℝ) < r := by exact_mod_cast hr
  have hθ : (r : ℝ) * (1 / (256 * r)) = 1 / 256 := by field_simp
  have h := TileLuck (n := 32) (r := r) (fun k i => (w k i : ℝ)) dead lucky (fun _ _ => 1 / 2)
    (fun k i => (1 / 2) ^ w k i) (1 / (256 * r)) 4 1 (by norm_num) hr (by positivity) (by norm_num)
    (fun _ _ => Nat.cast_nonneg _) (fun _ _ => ⟨by norm_num, by norm_num⟩) hl hd (fun k i => by
      rw [hθ]
      have := TemplateCondition (w k i) (hw k i)
      push_cast
      exact this)
  simpa only [one_mul] using h

end Accounting.SecurityProofs.TS
