import Proofs.Accounting.Compose.ServiceBudget
import Proofs.Accounting.Compose.ServiceCovers
import Proofs.Accounting.Compose.CircuitWork

/-!
# `CircuitWorkSeeded`: the circuit's work under seeded coins

* **GRIND's premium per round** (`grind_le`): one oracle answer redrawn is an involution of (answers, answer)
  (`resample_resample`), so the window's mean of `e^{θ·g q}` is the mean of its per-query moment, at most `1`
  (`sum_resample`); Markov at `ln(T_q/δ_g)/θ` and the union over the `T_q` queries give `δ_g`. Off it and off
  `SeedLuckAt`'s `Break`, at the query `sq a r` names, the round's gain is at most `roundPremium` (`gain_le_premium`;
  the round's weight `roundCone` where `θ_r = 0`), and summed over the rounds (every tile's round is one, under
  `Declared.Wf`) the mode's `L − κ·U` is at most `premiumOf` (`gainSum_le`).
* **The live draw** (`live_le`): given the oracle's answers, a uniform `K_d`-subset misses the tiles whose checked values
  aren't honest with probability at most `(1 − ρ·U)^{K_d} ≤ e^{−ρ·K_d·U}`, so a pass with `ρ·K_d·U ≥ ln(1/δ_s)` costs
  `δ_s`.
* **The work**: off `DrawnRightOff`'s event on `Declared.atDraw`'s layer, `Of` gives the share game's pass; off the
  rest, `S = (L − κ·U) + (1 + κ)·U` per mode (`share_split`) puts `a·S_F + b·S_0` below `rOfSeeded`'s first three terms,
  and `UnitFixed` with `ServiceCovers` on `Declared.atIR`'s layer gives the words as in `CircuitWork`
  (`WorkProof.work_of_bounds`).
-/

namespace Accounting.SecurityProofs.Compose.Seeded

open Accounting.TS Accounting.Compose Accounting.Assumptions.TS Accounting.Assumptions.Compose Pouw.RelativeSecurity
open Accounting.SecurityProofs.TS.Stages (prob_prod expect_empty)

/-! ## One oracle answer redrawn -/

section Redraw

variable {C : Type}

theorem coinAt_setAt : ∀ {n : ℕ} (x : Paths C n) (r : Fin n) (c : C), coinAt (setAt x r c) r = c
  | 0, _, r, _ => r.elim0
  | _ + 1, _, ⟨0, _⟩, _ => rfl
  | _ + 1, x, ⟨r + 1, h⟩, c => coinAt_setAt x.2 ⟨r, Nat.lt_of_succ_lt_succ h⟩ c

theorem setAt_setAt : ∀ {n : ℕ} (x : Paths C n) (r : Fin n) (c c' : C), setAt (setAt x r c) r c' = setAt x r c'
  | 0, _, r, _, _ => r.elim0
  | _ + 1, _, ⟨0, _⟩, _, _ => rfl
  | _ + 1, x, ⟨r + 1, h⟩, c, c' => congrArg (Prod.mk x.1) (setAt_setAt x.2 ⟨r, Nat.lt_of_succ_lt_succ h⟩ c c')

theorem setAt_coinAt : ∀ {n : ℕ} (x : Paths C n) (r : Fin n), setAt x r (coinAt x r) = x
  | 0, _, r => r.elim0
  | _ + 1, _, ⟨0, _⟩ => rfl
  | _ + 1, x, ⟨r + 1, h⟩ => congrArg (Prod.mk x.1) (setAt_coinAt x.2 ⟨r, Nat.lt_of_succ_lt_succ h⟩)

end Redraw

variable {B : Type} {M Tq : ℕ}

theorem resample_at (a : Paths (Fin Tq → B) M) (r : Fin M) (q : Fin Tq) (b : B) : coinAt (resample a r q b) r q = b := by
  unfold resample
  rw [coinAt_setAt, Function.update_self]

theorem resample_resample (a : Paths (Fin Tq → B) M) (r : Fin M) (q : Fin Tq) (b : B) :
    resample (resample a r q b) r q (coinAt a r q) = a := by
  unfold resample
  rw [coinAt_setAt, setAt_setAt, Function.update_idem, Function.update_eq_self, setAt_coinAt]

/-- **Summing over one answer redrawn** counts every window `|B|` times. -/
theorem sum_resample [Fintype B] (r : Fin M) (q : Fin Tq) (f : Paths (Fin Tq → B) M → ℝ) :
    ∑ a, ∑ b : B, f (resample a r q b) = Fintype.card B * ∑ a, f a := by
  have hσ : Function.Involutive fun p : Paths (Fin Tq → B) M × B => (resample p.1 r q p.2, coinAt p.1 r q) :=
    fun p => by simp only [resample_resample, resample_at]
  rw [← Fintype.sum_prod_type' (f := fun a b => f (resample a r q b)),
    show (∑ p : Paths (Fin Tq → B) M × B, f (resample p.1 r q p.2)) = ∑ p : Paths (Fin Tq → B) M × B, f p.1 from
      Equiv.sum_comp (hσ.toPerm _) (fun p => f p.1),
    Fintype.sum_prod_type]
  simp [Finset.mul_sum]

/-- **One query's tail**: a per-query moment at most `1` over its own answer puts `g ≥ t` below `e^{−θt}`. -/
theorem tail_le [Fintype B] (r : Fin M) (q : Fin Tq) (g : Paths (Fin Tq → B) M → ℝ) (θ t : ℝ) (hθ : 0 ≤ θ)
    (hm : ∀ a, ∑ b : B, Real.exp (θ * g (resample a r q b)) ≤ Fintype.card B) :
    prob (fun a => t ≤ g a) ≤ Real.exp (-(θ * t)) := by
  have hE : expect (fun a => Real.exp (θ * g a)) ≤ 1 := by
    rcases isEmpty_or_nonempty (Paths (Fin Tq → B) M) with he | hne
    · rw [expect_empty]
      exact zero_le_one
    have a₀ : Paths (Fin Tq → B) M := Classical.choice hne
    have hB : (0 : ℝ) < Fintype.card B := Nat.cast_pos.mpr (Fintype.card_pos_iff.mpr ⟨coinAt a₀ r q⟩)
    have hle : ∑ a, ∑ b : B, Real.exp (θ * g (resample a r q b)) ≤
        ∑ _a : Paths (Fin Tq → B) M, (Fintype.card B : ℝ) := Finset.sum_le_sum fun a _ => hm a
    rw [sum_resample r q (fun a => Real.exp (θ * g a)), Finset.sum_const, Finset.card_univ, nsmul_eq_mul,
      mul_comm (Fintype.card (Paths (Fin Tq → B) M) : ℝ)] at hle
    unfold expect
    rw [div_le_one (Nat.cast_pos.mpr Fintype.card_pos)]
    exact le_of_mul_le_mul_left hle hB
  have hmk : Real.exp (θ * t) * prob (fun a => Real.exp (θ * t) ≤ Real.exp (θ * g a)) ≤
      expect (fun a => Real.exp (θ * g a)) := markov (fun a => (Real.exp_pos _).le) _
  calc prob (fun a => t ≤ g a) ≤ prob (fun a => Real.exp (θ * t) ≤ Real.exp (θ * g a)) :=
        prob_mono fun a h => Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left h hθ)
    _ = Real.exp (-(θ * t)) * (Real.exp (θ * t) * prob (fun a => Real.exp (θ * t) ≤ Real.exp (θ * g a))) := by
        rw [← mul_assoc, ← Real.exp_add, neg_add_cancel, Real.exp_zero, one_mul]
    _ ≤ Real.exp (-(θ * t)) * 1 := mul_le_mul_of_nonneg_left (hmk.trans hE) (Real.exp_pos _).le
    _ = Real.exp (-(θ * t)) := mul_one _

/-- **GRIND's premium**: over the `T_q` queries, some query's gain reaches `ln(T_q/δ_g)/θ` with probability at most
`δ_g`. -/
theorem grind_le [Fintype B] (r : Fin M) (g : Fin Tq → Paths (Fin Tq → B) M → ℝ) (θ δg : ℝ) (hθ : 0 < θ)
    (hδ : 0 < δg) (hm : ∀ q a, ∑ b : B, Real.exp (θ * g q (resample a r q b)) ≤ Fintype.card B) :
    prob (fun a => ∃ q, Real.log (Tq / δg) / θ ≤ g q a) ≤ δg := by
  have hsum : ∑ _q : Fin Tq, Real.exp (-(θ * (Real.log (Tq / δg) / θ))) ≤ δg := by
    rcases Nat.eq_zero_or_pos Tq with h0 | hT
    · subst h0
      simp [hδ.le]
    · have hc : θ * (Real.log (Tq / δg) / θ) = Real.log (Tq / δg) := by field_simp
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, hc, Real.exp_neg,
        Real.exp_log (div_pos (Nat.cast_pos.mpr hT) hδ), inv_div]
      exact le_of_eq (by field_simp)
  calc prob (fun a => ∃ q, Real.log (Tq / δg) / θ ≤ g q a)
      ≤ ∑ q ∈ (Finset.univ : Finset (Fin Tq)), prob (fun a => Real.log (Tq / δg) / θ ≤ g q a) :=
        (prob_mono fun a ⟨q, hq⟩ => ⟨q, Finset.mem_univ q, hq⟩).trans (prob_exists_le _ _)
    _ ≤ ∑ _q : Fin Tq, Real.exp (-(θ * (Real.log (Tq / δg) / θ))) :=
        Finset.sum_le_sum fun q _ => tail_le r q (g q) θ _ hθ.le (hm q)
    _ ≤ δg := hsum

/-- A round's bad event: `Break`, or (at `θ > 0`) some query's gain at GRIND's threshold. -/
theorem bad_le [Fintype B] (r : Fin M) (g : Fin Tq → Paths (Fin Tq → B) M → ℝ) (Br : Paths (Fin Tq → B) M → Prop)
    (θ δb δg : ℝ) (hδg : 0 < δg) (hm : ∀ q a, ∑ b : B, Real.exp (θ * g q (resample a r q b)) ≤ Fintype.card B)
    (hB : prob Br ≤ δb) :
    prob (fun a => Br a ∨ (0 < θ ∧ ∃ q, Real.log (Tq / δg) / θ ≤ g q a)) ≤ δb + δg := by
  refine (prob_or_le _ _).trans (add_le_add hB ?_)
  by_cases hθ : 0 < θ
  · exact (prob_mono fun a h => h.2).trans (grind_le r g θ δg hθ hδg hm)
  · refine (prob_mono (q := fun _ => False) fun a h => absurd h.1 hθ).trans ?_
    rw [Accounting.SecurityProofs.TS.Rounds.prob_false]
    exact hδg.le

/-! ## A round's gain and the premium -/

section Gain

variable {ι K A : Type} (Sc : Scheme K) (Cd : Declared ι) {Np : ℕ}

theorem tileRound_lt (hw : Cd.Wf) (hr : 0 < Cd.rounds) (t : Fin Np) : tileRound Cd Np t < Cd.rounds := by
  unfold tileRound Declared.roundOf
  rcases hl : unitGates Cd t.val with _ | ⟨i, l⟩
  · exact hr
  · have hi : i ∈ unitGates Cd t.val := by rw [hl]; exact List.mem_cons_self
    exact hw.2.1 _ (hw.1 i (List.mem_range.mp (List.mem_filter.mp hi).1)).1

/-- A round's gain is at most its tiles' weight. -/
theorem roundGain_le_cone (R : JointRun A (Fin Np)) (s : Mode) (κ : ℝ) (hκ : 0 ≤ κ) (r : ℕ) (a : A) :
    roundGain (tileRound Cd Np) r R (shareCone Sc Cd Np) s κ a ≤ roundCone Sc Cd Np r := by
  classical
  unfold roundGain roundCone
  refine Finset.sum_le_sum fun t _ => ?_
  have hw := WorkProof.shareCone_nonneg Sc Cd Np t
  by_cases h1 : tileRound Cd Np t = r
  · have h1' : Cd.roundOf t.val = r := h1
    rw [ite_eq_left h1']
    split_ifs <;> nlinarith [mul_nonneg hκ hw]
  · have h1' : ¬ Cd.roundOf t.val = r := h1
    rw [ite_eq_right h1', ite_eq_right fun h => h1 h.1]

/-- **A round's gain is at most its premium** off its bad event, at the query its proofs' seed answers. -/
theorem gain_le_premium (Tq : ℕ) (R : JointRun A (Fin Np)) (s : Mode) (κ δg : ℝ) (θ : ℕ → ℝ) (hκ : 0 ≤ κ)
    (r : ℕ) (a : A) {Q : Type} (g : Q → ℝ) (Br : Prop) (q : Q)
    (hnb : ¬ (Br ∨ (0 < θ r ∧ ∃ q, Real.log (Tq / δg) / θ r ≤ g q)))
    (hgain : ¬ Br → roundGain (tileRound Cd Np) r R (shareCone Sc Cd Np) s κ a ≤ g q) :
    roundGain (tileRound Cd Np) r R (shareCone Sc Cd Np) s κ a ≤ roundPremium Sc Cd Np Tq δg θ r := by
  have hBr : ¬ Br := fun h => hnb (Or.inl h)
  unfold roundPremium
  split_ifs with hθ
  · exact le_min ((hgain hBr).trans (not_le.mp fun h => hnb (Or.inr ⟨hθ, q, h⟩)).le)
      (roundGain_le_cone Sc Cd R s κ hκ r a)
  · exact roundGain_le_cone Sc Cd R s κ hκ r a

theorem premiumOf_eq (Tq : ℕ) (δg : ℝ) (θ : ℕ → ℝ) :
    premiumOf Sc Cd Np Tq δg θ = ∑ r : Fin Cd.rounds, roundPremium Sc Cd Np Tq δg θ r.val := by
  unfold premiumOf
  rw [← Finset.sum_range (fun r => roundPremium Sc Cd Np Tq δg θ r)]
  rfl

open Classical in
theorem sum_roundGain {V : Type} [Fintype V] (rd : V → ℕ) (n : ℕ) (hrd : ∀ v, rd v < n) (R : JointRun A V)
    (w : V → ℝ) (s : Mode) (κ : ℝ) (a : A) :
    ∑ r : Fin n, roundGain rd r.val R w s κ a =
      ∑ v, if R.mode a v = s then w v * ((1 + κ) * (if R.same a v then 1 else 0) - κ) else 0 := by
  unfold roundGain
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [Fintype.sum_eq_single ⟨rd v, hrd v⟩ fun r hr => ite_eq_right fun h => hr (Fin.ext h.1.symm)]
  simp

open Classical in
/-- **A mode's `L − κ·U` is at most its premium** when each round's gain is at most the round's. -/
theorem gainSum_le (hw : Cd.Wf) (hr : 0 < Cd.rounds) (Tq : ℕ) (R : JointRun A (Fin Np)) (s : Mode) (κ δg : ℝ)
    (θ : ℕ → ℝ) (a : A)
    (h : ∀ r : Fin Cd.rounds,
      roundGain (tileRound Cd Np) r.val R (shareCone Sc Cd Np) s κ a ≤ roundPremium Sc Cd Np Tq δg θ r.val) :
    (∑ v, if R.mode a v = s then shareCone Sc Cd Np v * ((1 + κ) * (if R.same a v then 1 else 0) - κ) else 0) ≤
      premiumOf Sc Cd Np Tq δg θ := by
  rw [← sum_roundGain (tileRound Cd Np) Cd.rounds (tileRound_lt Cd hw hr), premiumOf_eq]
  exact Finset.sum_le_sum fun r _ => h r

end Gain

open Classical in
/-- **A mode's share** is its `L − κ·U` plus `(1 + κ)` times its unlucky share. -/
theorem share_split {A V : Type} [Fintype V] (R : JointRun A V) (w : V → ℝ) (s : Mode) (κ : ℝ) (a : A) :
    R.share w a s = (∑ v, if R.mode a v = s then w v * ((1 + κ) * (if R.same a v then 1 else 0) - κ) else 0) +
      (1 + κ) * ∑ v, if R.mode a v = s ∧ ¬ R.same a v then w v else 0 := by
  unfold JointRun.share
  rw [Finset.mul_sum, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun v _ => ?_
  by_cases hm : R.mode a v = s <;> by_cases hs : R.same a v <;> simp [hm, hs]
  ring

open Classical in
theorem unlucky_le {A V : Type} [Fintype V] (R : JointRun A V) (w : V → ℝ) (hw : ∀ v, 0 ≤ w v) (a : A) :
    (∑ v, if R.mode a v = .folded ∧ ¬ R.same a v then w v else 0) +
      (∑ v, if R.mode a v = .whole ∧ ¬ R.same a v then w v else 0) ≤ ∑ v, if ¬ R.same a v then w v else 0 := by
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_le_sum fun v _ => ?_
  by_cases hs : R.same a v <;> cases hm : R.mode a v <;> simp [hs, hw v]

/-! ## The live draw -/

open Classical in
/-- **The live draw's tail**: a pass with `ρ·K_d·U ≥ ln(1/δ_s)` has probability at most `δ_s`. -/
theorem live_le {A Ξ : Type} [Fintype A] [Fintype Ξ] {Np : ℕ} (R : JointRun A (Fin Np)) (w : Fin Np → ℝ) (Kd : ℕ)
    (ρ δs : ℝ) (hρN : ∀ t, ρ * Np * w t ≤ 1) (hδs : 0 < δs) :
    prob (fun ω : A × (Equiv.Perm (Fin Np) × Ξ) =>
      (∀ v ∈ Pouw.Window.subsetDraw Kd ω.2.1, R.same ω.1 v) ∧
        Real.log (1 / δs) ≤ ρ * Kd * ∑ v, if ¬ R.same ω.1 v then w v else 0) ≤ δs := by
  rw [prob_prod]
  refine expect_le_of_le hδs.le fun a => ?_
  by_cases hU : Real.log (1 / δs) ≤ ρ * Kd * ∑ v, if ¬ R.same a v then w v else 0
  · set Bd : Finset (Fin Np) := Finset.univ.filter fun v => ¬ R.same a v
    have hU' : ρ * ∑ v, (if ¬ R.same a v then w v else 0) ≤ (Bd.card : ℝ) / Np := by
      have := Budget.weight_le_card hρN Bd
      rwa [Finset.sum_filter] at this
    have hB1 : (Bd.card : ℝ) / Np ≤ 1 :=
      div_le_one_of_le₀ (by exact_mod_cast (Finset.card_le_univ Bd).trans_eq (Fintype.card_fin Np))
        (Nat.cast_nonneg _)
    refine prob_le_of_slices _ hδs.le fun _ => ?_
    calc prob (fun σ : Equiv.Perm (Fin Np) => (∀ v ∈ Pouw.Window.subsetDraw Kd σ, R.same a v) ∧
          Real.log (1 / δs) ≤ ρ * Kd * ∑ v, if ¬ R.same a v then w v else 0)
        ≤ prob (fun σ : Equiv.Perm (Fin Np) => ∀ v ∈ Pouw.Window.subsetDraw Kd σ, v ∉ Bd) :=
          prob_mono fun σ h v hv hB => (Finset.mem_filter.mp hB).2 (h.1 v hv)
      _ ≤ (1 - (Bd.card : ℝ) / Np) ^ Kd := Budget.miss_le Kd Np Bd
      _ ≤ (1 - ρ * ∑ v, if ¬ R.same a v then w v else 0) ^ Kd :=
          pow_le_pow_left₀ (sub_nonneg.mpr hB1) (by linarith) Kd
      _ ≤ Real.exp (-(ρ * ∑ v, if ¬ R.same a v then w v else 0)) ^ Kd :=
          pow_le_pow_left₀ (by linarith) (Real.one_sub_le_exp_neg _) Kd
      _ = Real.exp (-(ρ * Kd * ∑ v, if ¬ R.same a v then w v else 0)) := by
          rw [← Real.exp_nat_mul]
          ring_nf
      _ ≤ Real.exp (-Real.log (1 / δs)) := Real.exp_le_exp.mpr (by linarith)
      _ = δs := by rw [Real.exp_neg, Real.exp_log (one_div_pos.mpr hδs), one_div, inv_inv]
  · refine (prob_mono (q := fun _ => False) fun _ h => absurd h.2 hU).trans ?_
    rw [Accounting.SecurityProofs.TS.Rounds.prob_false]
    exact hδs.le

/-- An event of the window's one round's coin has its probability on the coin. -/
theorem prob_lift {Ω : Type} [Fintype Ω] (P : Ω → Prop) {κ : ℝ} (hP : prob P ≤ κ) :
    prob (fun x : Paths Ω 1 => P (coinAt x 0)) ≤ κ :=
  prob_le_of_slices _ ((prob_nonneg P).trans hP) fun _ => hP

theorem prob_lift_fst {A Y : Type} [Fintype A] [Fintype Y] (P : A → Prop) {κ : ℝ} (hP : prob P ≤ κ) :
    prob (fun x : Paths (A × Y) 1 => P (coinAt x 0).1) ≤ κ :=
  prob_lift (fun ω : A × Y => P ω.1) (prob_le_of_slices _ ((prob_nonneg P).trans hP) fun _ => hP)

/-! ## The work -/

open Classical in
theorem circuitWorkSeeded : Accounting.Guarantees.Compose.CircuitWorkSeeded := by
  intro ι K T' L U B Ξ V _ _ _ _ _ _ _ Tq Sc Cd Np Kd hN hr hK R Pl pr folded sq εid T J ρ a b κ0 δs δg δb δbind δks
    hS hδg hKd hρ hρN ha hb hκ hG hOf hMoved hJS hPins hSE hSLF hSLW hUF
  obtain ⟨hwf, -, -, hδs, -⟩ := hS
  have hw0 := WorkProof.shareCone_nonneg Sc Cd Np
  obtain ⟨E, hE, hEoff⟩ := Accounting.SecurityProofs.TS.DrawnRightOff (Fin Tq → B) (Equiv.Perm (Fin Np) × Ξ) 1
    _ Kd (Cd.atDraw.layer hr hK pr) δbind δks hMoved hJS hPins
  have hE' : prob E ≤ δbind + δks := by simpa using hE
  obtain ⟨Off, hOffP, hOffS⟩ := hSE
  choose gF BrF hF using hSLF
  choose gW BrW hW using hSLW
  let BadF : Fin Cd.rounds → Paths (Fin Tq → B) Cd.rounds → Prop := fun r a' =>
    BrF r a' ∨ (0 < seedTheta Sc Cd r.val ∧ ∃ q, Real.log (Tq / δg) / seedTheta Sc Cd r.val ≤ gF r q a')
  let BadW : Fin Cd.rounds → Paths (Fin Tq → B) Cd.rounds → Prop := fun r a' =>
    BrW r a' ∨ (0 < wholeTheta Sc Cd r.val ∧ ∃ q, Real.log (Tq / δg) / wholeTheta Sc Cd r.val ≤ gW r q a')
  let PL : Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ) → Prop := fun ω =>
    (∀ v ∈ Pouw.Window.subsetDraw Kd ω.2.1, R.same ω.1 v) ∧
      Real.log (1 / δs) ≤ ρ * Kd * ∑ v, if ¬ R.same ω.1 v then shareCone Sc Cd Np v else 0
  let Live : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 → Prop := fun x => PL (coinAt x 0)
  have hOff' : prob (fun x : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 =>
      Off (coinAt x 0).1) ≤ εid T J := prob_lift_fst Off hOffP
  have hBad : prob (fun x : Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 =>
      ∃ r ∈ (Finset.univ : Finset (Fin Cd.rounds)), BadF r (coinAt x 0).1 ∨ BadW r (coinAt x 0).1) ≤
      Cd.rounds * (2 * (δb + δg)) := by
    refine (prob_exists_le _ _).trans ?_
    calc ∑ r ∈ (Finset.univ : Finset (Fin Cd.rounds)), prob (fun x :
          Paths (Paths (Fin Tq → B) Cd.rounds × (Equiv.Perm (Fin Np) × Ξ)) 1 =>
            BadF r (coinAt x 0).1 ∨ BadW r (coinAt x 0).1)
        ≤ ∑ _r ∈ (Finset.univ : Finset (Fin Cd.rounds)), 2 * (δb + δg) := by
          refine Finset.sum_le_sum fun r _ => (prob_or_le _ _).trans ?_
          have h1 := prob_lift_fst (Y := Equiv.Perm (Fin Np) × Ξ) (BadF r)
            (bad_le r (gF r) (BrF r) _ δb δg hδg (hF r).1 (hF r).2.1)
          have h2 := prob_lift_fst (Y := Equiv.Perm (Fin Np) × Ξ) (BadW r)
            (bad_le r (gW r) (BrW r) _ δb δg hδg (hW r).1 (hW r).2.1)
          linarith
      _ = Cd.rounds * (2 * (δb + δg)) := by simp
  have hLive : prob Live ≤ δs := prob_lift PL (live_le R (shareCone Sc Cd Np) Kd ρ δs hρN hδs)
  have hCov : (Cd.atIR.layer hr hK pr).Covers Pl (jointMode R) (fun _ t => (tileWords Sc Cd Np t : ℝ))
      (wordsOf Sc Cd : ℝ) :=
    CoversProof.serviceCovers Sc Cd.atIR hr hK R Pl pr hwf (fun _ _ => Or.inl (Nat.zero_le _)) hG
  have hFix := hUF (WorkProof.writerReads Cd.atIR hr hK pr)
  have hCount := WorkProof.wordsOf_le_sum Sc Cd hwf hN
  have hsub : ∀ x, (Pl.accepts x ∧
      ¬ WorkAt (wordsOf Sc Cd : ℝ) a b (rOfSeeded Sc Cd Np Kd Tq a b κ0 ρ δs δg) (Pl.run x).fixed (folded x)) →
      ((E x ∨ Off (coinAt x 0).1) ∨
        ∃ r ∈ (Finset.univ : Finset (Fin Cd.rounds)), BadF r (coinAt x 0).1 ∨ BadW r (coinAt x 0).1) ∨ Live x := by
    rintro x ⟨hacc, hnw⟩
    by_cases hEx : E x
    · exact Or.inl (Or.inl (Or.inl hEx))
    by_cases hOx : Off (coinAt x 0).1
    · exact Or.inl (Or.inl (Or.inr hOx))
    by_cases hBx : ∃ r ∈ (Finset.univ : Finset (Fin Cd.rounds)), BadF r (coinAt x 0).1 ∨ BadW r (coinAt x 0).1
    · exact Or.inl (Or.inr hBx)
    have hpass : ∀ v ∈ Pouw.Window.subsetDraw Kd (coinAt x 0).2.1, R.same (coinAt x 0).1 v := by
      have h1 := hOf.2 x fun w i => hEoff x hEx w i (hOf.1 x hacc w i)
      rw [hG] at h1
      exact h1.1
    refine Or.inr ⟨hpass, not_lt.mp fun hUlt => hnw ?_⟩
    set a' := (coinAt x 0).1 with ha'
    have hq : ∀ r : Fin Cd.rounds, ∃ q, sq a' r = some q := fun r => Option.isSome_iff_exists.mp (hOffS a' hOx r)
    have hGF := gainSum_le Sc Cd hwf hr Tq R .folded 4 δg (seedTheta Sc Cd) a' fun r => by
      obtain ⟨q, hq'⟩ := hq r
      exact gain_le_premium Sc Cd Tq R .folded 4 δg (seedTheta Sc Cd) (by norm_num) r.val a' (fun q => gF r q a')
        (BrF r a') q (fun h => hBx ⟨r, Finset.mem_univ r, Or.inl h⟩) fun hb => (hF r).2.2 a' q hb hq'
    have hGW := gainSum_le Sc Cd hwf hr Tq R .whole κ0 δg (wholeTheta Sc Cd) a' fun r => by
      obtain ⟨q, hq'⟩ := hq r
      exact gain_le_premium Sc Cd Tq R .whole κ0 δg (wholeTheta Sc Cd) hκ r.val a' (fun q => gW r q a')
        (BrW r a') q (fun h => hBx ⟨r, Finset.mem_univ r, Or.inr h⟩) fun hb => (hW r).2.2 a' q hb hq'
    have hSF := share_split R (shareCone Sc Cd Np) .folded 4 a'
    have hSW := share_split R (shareCone Sc Cd Np) .whole κ0 a'
    have hUs := unlucky_le R (shareCone Sc Cd Np) hw0 a'
    have hUF0 : 0 ≤ ∑ v, if R.mode a' v = .folded ∧ ¬ R.same a' v then shareCone Sc Cd Np v else 0 :=
      Finset.sum_nonneg fun v _ => by split_ifs <;> simp [hw0 v]
    have hUW0 : 0 ≤ ∑ v, if R.mode a' v = .whole ∧ ¬ R.same a' v then shareCone Sc Cd Np v else 0 :=
      Finset.sum_nonneg fun v _ => by split_ifs <;> simp [hw0 v]
    set mx := max (a * (1 + 4)) (b * (1 + κ0)) with hmx
    have hmx1 : a * (1 + 4) ≤ mx := le_max_left _ _
    have hmx2 : b * (1 + κ0) ≤ mx := le_max_right _ _
    have hmx0 : 0 < mx := lt_of_lt_of_le (by linarith) hmx1
    have hρK : 0 < ρ * Kd := mul_pos hρ (Nat.cast_pos.mpr hKd)
    have hUq : (∑ v, if ¬ R.same a' v then shareCone Sc Cd Np v else 0) < Real.log (1 / δs) / (ρ * Kd) := by
      rw [lt_div_iff₀ hρK]
      linarith
    have h1 := mul_le_mul_of_nonneg_left hGF ha.le
    have h2 := mul_le_mul_of_nonneg_left hGW hb
    have h3 := mul_le_mul_of_nonneg_right hmx1 hUF0
    have h4 := mul_le_mul_of_nonneg_right hmx2 hUW0
    have h5 := mul_le_mul_of_nonneg_left hUs hmx0.le
    have h6 := mul_lt_mul_of_pos_left hUq hmx0
    have h7 : mx * (Real.log (1 / δs) / (ρ * Kd)) = mx * Real.log (1 / δs) / (ρ * Kd) := (mul_div_assoc _ _ _).symm
    have hlt' : a * R.share (shareCone Sc Cd Np) a' .folded + b * R.share (shareCone Sc Cd Np) a' .whole + 0 <
        mx * Real.log (1 / δs) / (ρ * Kd) + a * premiumOf Sc Cd Np Tq δg (seedTheta Sc Cd) +
          b * premiumOf Sc Cd Np Tq δg (wholeTheta Sc Cd) := by
      rw [hSF, hSW]
      set GF := ∑ v, if R.mode a' v = .folded then
        shareCone Sc Cd Np v * ((1 + 4) * (if R.same a' v then 1 else 0) - 4) else 0
      set GW := ∑ v, if R.mode a' v = .whole then
        shareCone Sc Cd Np v * ((1 + κ0) * (if R.same a' v then 1 else 0) - κ0) else 0
      set UF := ∑ v, if R.mode a' v = .folded ∧ ¬ R.same a' v then shareCone Sc Cd Np v else 0
      set UW := ∑ v, if R.mode a' v = .whole ∧ ¬ R.same a' v then shareCone Sc Cd Np v else 0
      set Ut := ∑ v, if ¬ R.same a' v then shareCone Sc Cd Np v else 0
      clear_value GF GW UF UW Ut
      linarith
    have hD : Pl.game.deficiency x =
        1 * (∑ u, if R.mode a' u = .folded then tileCone Sc Cd Np u else 0) +
          1 * (∑ u, if R.mode a' u = .whole then tileCone Sc Cd Np u else 0) + 0 := by
      rw [hG]
      rfl
    have hcv : (∑ j : Fin Np, if (Cd.atIR.layer hr hK pr).Good (jointMode R) x 0 j then (0 : ℝ)
        else (tileWords Sc Cd Np j : ℝ)) ≤ Pl.game.deficiency x * (wordsOf Sc Cd : ℝ) := by
      have h := hCov x hacc
      simp only [Fin.sum_univ_one] at h
      exact h
    have hfx := hFix x hacc
    have hfix1 : (∑ j : Fin Np, if (Cd.atIR.layer hr hK pr).Good (jointMode R) x 0 j
        then (tileWords Sc Cd Np j : ℝ) else 0) ≤ (Pl.run x).fixed := by
      have h := hfx.1
      simp only [Fin.sum_univ_one, Nat.cast_one, div_one] at h
      exact h
    have hfold1 : (∑ j : Fin Np, if R.mode a' j = .whole then (0 : ℝ)
        else (tileWords Sc Cd Np j : ℝ)) ≤ folded x := by
      have h := hfx.2
      simp only [Fin.sum_univ_one, Nat.cast_one, div_one] at h
      exact h
    exact WorkProof.work_of_bounds Sc Cd (R.mode a') (fun j => (Cd.atIR.layer hr hK pr).Good (jointMode R) x 0 j)
      ha.le hb hCount hD hcv hfix1 hfold1 hlt'
  refine (prob_mono hsub).trans ((prob_or_le _ _).trans ?_)
  refine (add_le_add ((prob_or_le _ _).trans
    (add_le_add ((prob_or_le _ _).trans (add_le_add hE' hOff')) hBad)) hLive).trans (le_of_eq ?_)
  ring

end Accounting.SecurityProofs.Compose.Seeded
