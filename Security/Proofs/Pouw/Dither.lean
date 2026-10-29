import Definitions.Pouw.Guarantees.Dither
import Proofs.Pouw.RelativeSecurity
import Proofs.Pouw.ShaRealize

/-!
# The no-fold Dither protocol's work-soundness: the headline's proof

* `FreedOfFloor`: the guarantee `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`, weak duality of the period's linear
  program.
* `WorkSound`: the guarantee `Definitions.Pouw.Guarantees.Dither.WorkSound`.
  - `ExtractAdv` splits each round's work into a words program and a commitment program on advice fixed before the
    round's coins, the commitment program's output the prover's root. `OutsourceWithinEgress` replaces both by
    programs that exchange nothing, at `β` a bit they exchanged.
  - Their two moments (SAK(matmul)'s and SHA-512's speed's, each outside its degenerate dithers) combine by the
    convexity of `exp` at the round's honest shares (`combined_moment_le`), in units of priced cost at one rate `η`.
  - A round's factor reads only the coins up to that round, and is frozen at 1 on a round whose oracle and advice
    are bad for `ShaSpeed` (`roundF`), so the rounds' product is a supermartingale over the fresh coins
    (`expect_prod_rounds`), and Markov bounds the luck by `δ` (`prob_rounds_le`).
  - The union bound adds the service's `κ_S`, the extraction's `κ_x(A)` on runs within the cap (a run whose sampling
    blocks' timed work fits makes at most `Q` oracle calls, `calls_le_of_fits`), the bad oracles' `κ_sha(A)` and the
    degenerate dithers'
    `k (κ_dmm + κ_dsha)` (`prob_round_le`). Off those events the rounds meet the work floor (`floor_of_rounds`), and
    `FreedOfFloor` turns it into freed compute.
* `WorkSoundNoFold`: `WorkSound` on the no-fold round with `ExtractAdv` from
  `Proofs.Pouw.ShaRealize.ExtractOfRealize` at the budget `(k + 1)(⌈Q⌉ + V)` and the schedule
  `τ_w = (w + 1)(⌈Q⌉ + V)` (`sum_segments`).
* `FleetSound`: the union bound over the units, each unit's event sliced at the other units' coins.
-/

namespace Pouw.SecurityProofs.Dither

open Pouw.RelativeSecurity Pouw.Dither Pouw.Assumptions Finset

/-! ## Freed compute -/

/-- `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`. -/
theorem FreedOfFloor : Pouw.Guarantees.Dither.FreedOfFloor := by
  intro G Pd W ε X μ z z' hμ hneeds hval hW hε hfit' hslack hfit hfloor
  refine ⟨W, z', hW, hfit', ?_⟩
  have h1 : ∑ j, Pd.val j * z j ≤ ∑ r, μ r * (Pd.cap r - X r) := by
    calc ∑ j, Pd.val j * z j ≤ ∑ j, (∑ r, μ r * Pd.needs j r) * z j :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_right (hval j) (hfit.1 j)
      _ = ∑ r, μ r * ∑ j, z j * Pd.needs j r := by
          simp_rw [Finset.sum_mul, Finset.mul_sum]
          rw [Finset.sum_comm]
          exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun j _ => by ring
      _ ≤ ∑ r, μ r * (Pd.cap r - X r) :=
          Finset.sum_le_sum fun r _ => mul_le_mul_of_nonneg_left (by linarith [hfit.2 r]) (hμ r)
  have h3 : ∑ r, μ r * (Pd.cap r - X r) =
      ∑ r, μ r * (Pd.cap r - W * Pd.hon r) + (∑ r, μ r * (W * Pd.hon r) - ∑ r, μ r * X r) := by
    rw [← Finset.sum_sub_distrib, ← Finset.sum_add_distrib]
    exact Finset.sum_congr rfl fun r _ => by ring
  unfold Period.value
  nlinarith [h1, h3, hfloor, hslack]

/-! ## Expectations over fresh coins -/

theorem expect_equiv {α β : Type*} [Fintype α] [Fintype β] (e : α ≃ β) (f : β → ℝ) :
    expect (fun a => f (e a)) = expect f := by
  unfold Pouw.RelativeSecurity.expect
  rw [Fintype.card_congr e]
  congr 1
  exact Fintype.sum_equiv e _ _ fun _ => rfl

theorem expect_affine3 {Ω : Type*} [Fintype Ω] (a b c : ℝ) (hc : 0 ≤ c) (f g : Ω → ℝ) :
    expect (fun ω => a * f ω + b * g ω + c) ≤ a * expect f + b * expect g + c := by
  unfold Pouw.RelativeSecurity.expect
  rcases Nat.eq_zero_or_pos (Fintype.card Ω) with h0 | h0
  · simp [h0, hc]
  · have hN : (Fintype.card Ω : ℝ) ≠ 0 := by exact_mod_cast h0.ne'
    rw [Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.mul_sum, Finset.sum_const,
      Finset.card_univ, nsmul_eq_mul]
    apply le_of_eq
    rw [add_div, add_div, mul_div_assoc a, mul_div_assoc b, mul_div_cancel_left₀ c hN]

/-- An event whose every slice at the first coordinate has probability at most `κ ≥ 0` has probability at most
`κ`. -/
theorem prob_le_of_slices_fst {α β : Type*} [Fintype α] [Fintype β] (E : α × β → Prop) {κ : ℝ} (hκ : 0 ≤ κ)
    (h : ∀ a, prob (fun b => E (a, b)) ≤ κ) : prob E ≤ κ := by
  classical
  rw [prob_eq_expect, expect_prod]
  exact expect_le_of_le hκ fun a => by rw [← prob_eq_expect (fun b => E (a, b))]; exact h a

/-- Markov's inequality at `e^t` for a nonnegative function of mean at most 1. -/
theorem prob_le_of_expect_le_one {Ω : Type*} [Fintype Ω] {f : Ω → ℝ} (hf : ∀ ω, 0 ≤ f ω) (hE : expect f ≤ 1)
    (t : ℝ) {E : Ω → Prop} (hsub : ∀ ω, E ω → Real.exp t ≤ f ω) : prob E ≤ Real.exp (-t) := by
  have hpos := Real.exp_pos t
  have h2 : Real.exp t * prob E ≤ 1 :=
    (mul_le_mul_of_nonneg_left (prob_mono hsub) hpos.le).trans ((markov hf (Real.exp t)).trans hE)
  rw [Real.exp_neg, ← one_div, le_div_iff₀ hpos, mul_comm]
  exact h2

/-- **The supermartingale over fresh coins**, as a finite product (`RoundGame.expect_weight_le` on `Fin n → Ω`): if
round `i`'s factor `Y i` is nonnegative, reads only the coins of rounds up to `i`, and averages at most 1 over round
`i`'s coin whatever the other coins, then the product of the factors averages at most 1. -/
theorem expect_prod_adapted {Ω : Type} [Fintype Ω] : ∀ (n : ℕ) (Y : Fin n → (Fin n → Ω) → ℝ),
    (∀ i x, 0 ≤ Y i x) → (∀ i x x', (∀ j, j ≤ i → x j = x' j) → Y i x = Y i x') →
    (∀ i x, expect (fun ω => Y i (Function.update x i ω)) ≤ 1) →
    expect (fun x => ∏ i, Y i x) ≤ 1
  | 0, Y, _, _, _ => expect_le_of_le zero_le_one fun x => by simp
  | n + 1, Y, h0, hY, h1 => by
    rcases isEmpty_or_nonempty Ω with hΩ | ⟨⟨ω₀⟩⟩
    · exact expect_le_of_le zero_le_one fun x => (hΩ.false (x 0)).elim
    let c : Fin n → Ω := fun _ => ω₀
    have hY0 : ∀ ω (x' : Fin n → Ω), Y 0 (Fin.cons ω x' : Fin (n + 1) → Ω) = Y 0 (Fin.cons ω c) :=
      fun ω x' => hY 0 _ _ fun j hj => by
        obtain rfl := Fin.le_zero_iff.mp hj
        simp only [Fin.cons_zero]
    have hstep : ∀ ω, expect (fun x' : Fin n → Ω => ∏ i, Y i (Fin.cons ω x' : Fin (n + 1) → Ω)) ≤
        Y 0 (Fin.cons ω c : Fin (n + 1) → Ω) := by
      intro ω
      have hIH := expect_prod_adapted n (fun i x' => Y i.succ (Fin.cons ω x' : Fin (n + 1) → Ω))
        (fun i x' => h0 _ _)
        (fun i x x' hx => hY _ _ _ fun j hj => by
          induction j using Fin.cases with
          | zero => simp only [Fin.cons_zero]
          | succ j => simp only [Fin.cons_succ]; exact hx j (Fin.succ_le_succ_iff.mp hj))
        (fun i x => by
          have := h1 i.succ (Fin.cons ω x : Fin (n + 1) → Ω)
          simpa only [Fin.cons_update] using this)
      calc expect (fun x' : Fin n → Ω => ∏ i, Y i (Fin.cons ω x' : Fin (n + 1) → Ω))
          = expect (fun x' : Fin n → Ω =>
              Y 0 (Fin.cons ω c : Fin (n + 1) → Ω) * ∏ i : Fin n, Y i.succ (Fin.cons ω x' : Fin (n + 1) → Ω)) := by
            congr 1
            funext x'
            rw [Fin.prod_univ_succ, hY0 ω x']
        _ = Y 0 (Fin.cons ω c : Fin (n + 1) → Ω) *
              expect (fun x' : Fin n → Ω => ∏ i : Fin n, Y i.succ (Fin.cons ω x' : Fin (n + 1) → Ω)) :=
            expect_mul_left _ _
        _ ≤ Y 0 (Fin.cons ω c : Fin (n + 1) → Ω) * 1 := mul_le_mul_of_nonneg_left hIH (h0 _ _)
        _ = Y 0 (Fin.cons ω c : Fin (n + 1) → Ω) := mul_one _
    calc expect (fun x => ∏ i, Y i x)
        = expect (fun p : Ω × (Fin n → Ω) => ∏ i, Y i (Fin.cons p.1 p.2 : Fin (n + 1) → Ω)) :=
          (expect_equiv (Fin.consEquiv fun _ => Ω) (fun x => ∏ i, Y i x)).symm
      _ = expect (fun ω => expect fun x' : Fin n → Ω => ∏ i, Y i (Fin.cons ω x' : Fin (n + 1) → Ω)) :=
          expect_prod _
      _ ≤ expect (fun ω => Y 0 (Fin.cons ω c : Fin (n + 1) → Ω)) := expect_mono hstep
      _ ≤ 1 := by
          have := h1 0 (Fin.cons ω₀ c : Fin (n + 1) → Ω)
          simpa only [Fin.update_cons_zero] using this

/-- Convexity of `exp` at three points, the third `0`. -/
theorem exp_convex3 {a b c u v : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a + b + c = 1) :
    Real.exp (a * u + b * v) ≤ a * Real.exp u + b * Real.exp v + c := by
  have := convexOn_exp.map_sum_le (t := Finset.univ) (w := ![a, b, c]) (p := ![u, v, 0])
    (fun i _ => by fin_cases i <;> simp [ha, hb, hc]) (by simp [Fin.sum_univ_three, h])
    (fun i _ => Set.mem_univ _)
  simpa [Fin.sum_univ_three] using this


/-- An event on the first coordinate of a pair has at most its own probability. -/
theorem prob_fst_le {α β : Type*} [Fintype α] [Fintype β] (D : α → Prop) :
    prob (fun ω : α × β => D ω.1) ≤ prob D :=
  prob_le_of_slices (fun ω : α × β => D ω.1) (prob_nonneg D) fun _ => le_rfl

theorem prob_false {Ω : Type*} [Fintype Ω] : prob (fun _ : Ω => False) = 0 := by
  unfold prob
  simp

/-- The union bound over five events. -/
theorem prob_or5_le {Ω : Type*} [Fintype Ω] (p1 p2 p3 p4 p5 : Ω → Prop) :
    prob (fun ω => (((p1 ω ∨ p2 ω) ∨ p3 ω) ∨ p4 ω) ∨ p5 ω) ≤ prob p1 + prob p2 + prob p3 + prob p4 + prob p5 :=
  (prob_or_le _ _).trans (add_le_add ((prob_or_le _ _).trans (add_le_add ((prob_or_le _ _).trans
    (add_le_add (prob_or_le _ _) le_rfl)) le_rfl)) le_rfl)

/-- **Rounds on adapted advice**: if round `w`'s factor is a nonnegative function of advice `adv x w` that reads only
the coins before round `w` and of round `w`'s own coin, averaging at most 1 over that coin at every advice, the
factors' product averages at most 1. -/
theorem expect_prod_rounds {Ω Adv : Type} [Fintype Ω] {k : ℕ} (adv : (Fin k → Ω) → Fin k → Adv)
    (hadv : ∀ x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv x w = adv x' w) (F : Fin k → Adv → Ω → ℝ)
    (h0 : ∀ w s ω, 0 ≤ F w s ω) (h1 : ∀ w s, expect (F w s) ≤ 1) :
    expect (fun x => ∏ w, F w (adv x w) (x w)) ≤ 1 :=
  expect_prod_adapted k (fun w x => F w (adv x w) (x w)) (fun w x => h0 _ _ _)
    (fun i x x' h => by rw [hadv x x' i fun v hv => h v hv.le, h i le_rfl])
    (fun i x => by
      have e : (fun ω => F i (adv (Function.update x i ω) i) (Function.update x i ω i)) = F i (adv x i) := by
        funext ω
        rw [hadv (Function.update x i ω) x i fun v hv => Function.update_of_ne hv.ne ω x, Function.update_self]
      rw [e]
      exact h1 i _)

/-- Markov's inequality at `e^t` for the rounds' product. -/
theorem prob_rounds_le {Ω Adv : Type} [Fintype Ω] {k : ℕ} (adv : (Fin k → Ω) → Fin k → Adv)
    (hadv : ∀ x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv x w = adv x' w) (F : Fin k → Adv → Ω → ℝ)
    (h0 : ∀ w s ω, 0 ≤ F w s ω) (h1 : ∀ w s, expect (F w s) ≤ 1) (t : ℝ) {E : (Fin k → Ω) → Prop}
    (hsub : ∀ x, E x → Real.exp t ≤ ∏ w, F w (adv x w) (x w)) : prob E ≤ Real.exp (-t) :=
  prob_le_of_expect_le_one (fun _ => Finset.prod_nonneg fun _ _ => h0 _ _ _) (expect_prod_rounds adv hadv F h0 h1) t
    hsub

/-- **One round's event on adapted advice**: an event of round `w`'s advice and coin of probability at most `κ` over
the coin at every advice has probability at most `κ` over every round's coins. -/
theorem prob_round_le {Ω Adv : Type} [Fintype Ω] {k : ℕ} (adv : (Fin k → Ω) → Fin k → Adv)
    (hadv : ∀ x x' w, (∀ v : Fin k, v < w → x v = x' v) → adv x w = adv x' w) (w : Fin k) (E : Adv → Ω → Prop)
    {κ : ℝ} (hκ : 0 ≤ κ) (hE : ∀ s, prob (E s) ≤ κ) : prob (fun x => E (adv x w) (x w)) ≤ κ := by
  classical
  refine le_of_forall_pos_le_add fun t ht => ?_
  have hκt : 0 < κ + t := by linarith
  have hinv : 0 ≤ (κ + t)⁻¹ := inv_nonneg.mpr hκt.le
  have h := expect_prod_rounds adv hadv
    (fun v s ω => if v = w then (κ + t)⁻¹ * (if E s ω then 1 else 0) else 1)
    (fun v s ω => by
      try dsimp only
      split_ifs
      · exact mul_nonneg hinv zero_le_one
      · exact mul_nonneg hinv le_rfl
      · exact zero_le_one)
    (fun v s => by
      try dsimp only
      by_cases hv : v = w
      · simp only [if_pos hv]
        rw [expect_mul_left, ← prob_eq_expect (E s)]
        calc (κ + t)⁻¹ * prob (E s) ≤ (κ + t)⁻¹ * (κ + t) :=
              mul_le_mul_of_nonneg_left (by linarith [hE s]) hinv
          _ = 1 := inv_mul_cancel₀ hκt.ne'
      · simp only [if_neg hv]
        exact expect_le_of_le zero_le_one fun _ => le_rfl)
  have h2 : expect (fun x : Fin k → Ω => (κ + t)⁻¹ * (if E (adv x w) (x w) then (1 : ℝ) else 0)) ≤ 1 := by
    refine le_trans (le_of_eq ?_) h
    congr 1
    funext x
    rw [Finset.prod_eq_single w (fun v _ hv => by simp only [if_neg hv]) (fun hw => absurd (Finset.mem_univ w) hw)]
    simp
  rw [expect_mul_left, ← prob_eq_expect (fun x : Fin k → Ω => E (adv x w) (x w))] at h2
  calc prob (fun x => E (adv x w) (x w)) = (κ + t) * ((κ + t)⁻¹ * prob (fun x => E (adv x w) (x w))) := by
        rw [← mul_assoc, mul_inv_cancel₀ hκt.ne', one_mul]
    _ ≤ (κ + t) * 1 := mul_le_mul_of_nonneg_left h2 hκt.le
    _ = κ + t := mul_one _

/-! ## One round -/

open Classical in
/-- **One round's combined factor** at the advice `s` and the round's coin: `e^{η D}` where the dither is outside both
premises' degenerate sets, the check passes on `Mw`'s words and their checked cells open under `Mh`'s root, `0`
elsewhere, with `D` the two programs' shortfalls against the public bests, in units of priced cost. -/
noncomputable def combY (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pm ps η : ℝ) (o : P.Orc) (a : P.Inst)
    {St : Type} (s : St) (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (Mh : G.Mc.Alg (P.Orc × (St × P.Draw) × P.Tr) P.Root)
    (Dm Ds : P.Draw → Prop) (ω : P.Draw × P.Chk) : ℝ :=
  if ¬ Dm ω.1 ∧ ¬ Ds ω.1 ∧ P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) ∧
      P.opens o (G.Mc.eval Mh (o, (s, ω.1), G.Mc.eval Mw (s, ω.1))) ω.2 (G.Mc.eval Mw (s, ω.1)) then
    Real.exp (η * (roundShortfall pm (G.cost μ Mw (s, ω.1)) +
      roundShortfall ps (G.cost μ Mh (o, (s, ω.1), G.Mc.eval Mw (s, ω.1))))) else 0

open Classical in
/-- **One round's combined moment**: SAK(matmul)'s moment on the words at rate `η H/H_mm` and SHA-512's speed's on
their commitment at rate `η H/H_sha`, at the weights `H_mm/H` and `H_sha/H`, with the rest of the honest round `H_o/H`
at shortfall `0`, bound the combined factor's average by 1 (the convexity of `exp`). -/
theorem combined_moment_le (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (pm ps η Hmm Hsha Ho : ℝ)
    (hHmm : 0 < Hmm) (hHsha : 0 < Hsha) (hHo : 0 ≤ Ho) (o : P.Orc) (a : P.Inst) {St : Type} (s : St)
    (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (Mh : G.Mc.Alg (P.Orc × (St × P.Draw) × P.Tr) P.Root) (Dm Ds : P.Draw → Prop)
    (hmm : wordsMoment G P μ pm (η * (Hmm + Hsha + Ho) / Hmm) a s Mw Dm ≤ 1)
    (hsh : commitMoment G P μ ps (η * (Hmm + Hsha + Ho) / Hsha) o a s Mw Mh Ds ≤ 1) :
    expect (combY G P μ pm ps η o a s Mw Mh Dm Ds) ≤ 1 := by
  have hH : 0 < Hmm + Hsha + Ho := by linarith
  have hHmm' := hHmm.ne'
  have hHsha' := hHsha.ne'
  have hH' := hH.ne'
  have ha : 0 ≤ Hmm / (Hmm + Hsha + Ho) := div_nonneg hHmm.le hH.le
  have hb : 0 ≤ Hsha / (Hmm + Hsha + Ho) := div_nonneg hHsha.le hH.le
  have hc : 0 ≤ Ho / (Hmm + Hsha + Ho) := div_nonneg hHo hH.le
  have hsum : Hmm / (Hmm + Hsha + Ho) + Hsha / (Hmm + Hsha + Ho) + Ho / (Hmm + Hsha + Ho) = 1 := by
    rw [← add_div, ← add_div, div_self hH']
  have hra : Hmm / (Hmm + Hsha + Ho) * (η * (Hmm + Hsha + Ho) / Hmm) = η := by
    field_simp
  have hrb : Hsha / (Hmm + Hsha + Ho) * (η * (Hmm + Hsha + Ho) / Hsha) = η := by
    field_simp
  have hie : ∀ (p : Prop) [Decidable p] (r : ℝ), 0 ≤ (if p then Real.exp r else 0) := fun p _ r => by
    split_ifs
    · exact (Real.exp_pos r).le
    · exact le_rfl
  calc expect (combY G P μ pm ps η o a s Mw Mh Dm Ds)
      ≤ expect (fun ω : P.Draw × P.Chk =>
        Hmm / (Hmm + Hsha + Ho) * (if ¬ Dm ω.1 ∧ P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) then
          Real.exp (η * (Hmm + Hsha + Ho) / Hmm * roundShortfall pm (G.cost μ Mw (s, ω.1))) else 0) +
        Hsha / (Hmm + Hsha + Ho) * (if ¬ Ds ω.1 ∧ P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) ∧
            P.opens o (G.Mc.eval Mh (o, (s, ω.1), G.Mc.eval Mw (s, ω.1))) ω.2 (G.Mc.eval Mw (s, ω.1)) then
          Real.exp (η * (Hmm + Hsha + Ho) / Hsha *
            roundShortfall ps (G.cost μ Mh (o, (s, ω.1), G.Mc.eval Mw (s, ω.1)))) else 0) +
        Ho / (Hmm + Hsha + Ho)) := by
        refine expect_mono fun ω => ?_
        unfold combY
        by_cases hall : ¬ Dm ω.1 ∧ ¬ Ds ω.1 ∧ P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) ∧
            P.opens o (G.Mc.eval Mh (o, (s, ω.1), G.Mc.eval Mw (s, ω.1))) ω.2 (G.Mc.eval Mw (s, ω.1))
        · rw [if_pos hall, if_pos (And.intro hall.1 hall.2.2.1), if_pos (And.intro hall.2.1 hall.2.2)]
          set X := roundShortfall pm (G.cost μ Mw (s, ω.1))
          set Y := roundShortfall ps (G.cost μ Mh (o, (s, ω.1), G.Mc.eval Mw (s, ω.1)))
          have e : η * (X + Y) = Hmm / (Hmm + Hsha + Ho) * (η * (Hmm + Hsha + Ho) / Hmm * X) +
              Hsha / (Hmm + Hsha + Ho) * (η * (Hmm + Hsha + Ho) / Hsha * Y) := by
            linear_combination (-X) * hra + (-Y) * hrb
          rw [e]
          exact exp_convex3 ha hb hc hsum
        · rw [if_neg hall]
          exact add_nonneg (add_nonneg (mul_nonneg ha (hie _ _)) (mul_nonneg hb (hie _ _))) hc
    _ ≤ Hmm / (Hmm + Hsha + Ho) * wordsMoment G P μ pm (η * (Hmm + Hsha + Ho) / Hmm) a s Mw Dm +
          Hsha / (Hmm + Hsha + Ho) * commitMoment G P μ ps (η * (Hmm + Hsha + Ho) / Hsha) o a s Mw Mh Ds +
          Ho / (Hmm + Hsha + Ho) :=
        expect_affine3 _ _ _ hc _ _
    _ ≤ Hmm / (Hmm + Hsha + Ho) * 1 + Hsha / (Hmm + Hsha + Ho) * 1 + Ho / (Hmm + Hsha + Ho) :=
        add_le_add (add_le_add (mul_le_mul_of_nonneg_left hmm ha) (mul_le_mul_of_nonneg_left hsh hb)) le_rfl
    _ = 1 := by rw [mul_one, mul_one, hsum]

open Classical in
/-- **Round `w`'s factor of the supermartingale** at the oracle `o`, its advice `s` and its coin: the combined factor
where the round's oracle and advice are good for `ShaSpeed` (`Good`), and `1` where they aren't (the stopped
process: a bad round is counted in `κ_sha`, not in the luck). -/
noncomputable def roundF {G : Accel} {P : RoundProto} {k : ℕ} {Adv : Type} (reg : Fin k → P.Inst) (μ : G.Res → ℝ)
    (pm ps : P.Inst → ℝ) (η : ℝ) (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (Mh : Fin k → G.Mc.Alg (P.Orc × (Adv × P.Draw) × P.Tr) P.Root) (Good : P.Orc → Fin k → Adv → Prop)
    (Dm : Fin k → Adv → P.Draw → Prop) (Ds : P.Orc → Fin k → Adv → P.Draw → Prop) (o : P.Orc) (w : Fin k) (s : Adv)
    (ω : P.Draw × P.Chk) : ℝ :=
  if Good o w s then combY G P μ (pm (reg w)) (ps (reg w)) η o (reg w) s (Mw w) (Mh w) (Dm w s) (Ds o w s) ω else 1

theorem roundF_nonneg {G : Accel} {P : RoundProto} {k : ℕ} {Adv : Type} (reg : Fin k → P.Inst) (μ : G.Res → ℝ)
    (pm ps : P.Inst → ℝ) (η : ℝ) (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (Mh : Fin k → G.Mc.Alg (P.Orc × (Adv × P.Draw) × P.Tr) P.Root) (Good : P.Orc → Fin k → Adv → Prop)
    (Dm : Fin k → Adv → P.Draw → Prop) (Ds : P.Orc → Fin k → Adv → P.Draw → Prop) (o : P.Orc) (w : Fin k) (s : Adv)
    (ω : P.Draw × P.Chk) : 0 ≤ roundF reg μ pm ps η Mw Mh Good Dm Ds o w s ω := by
  unfold roundF combY
  split_ifs <;> first | exact (Real.exp_pos _).le | exact le_rfl | exact zero_le_one

theorem expect_roundF_le {G : Accel} {P : RoundProto} {k : ℕ} {Adv : Type} (reg : Fin k → P.Inst) (μ : G.Res → ℝ)
    (pm ps Hmm Hsha Ho : P.Inst → ℝ) (η : ℝ) (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (Mh : Fin k → G.Mc.Alg (P.Orc × (Adv × P.Draw) × P.Tr) P.Root) (Good : P.Orc → Fin k → Adv → Prop)
    (Dm : Fin k → Adv → P.Draw → Prop) (Ds : P.Orc → Fin k → Adv → P.Draw → Prop) (o : P.Orc) (w : Fin k) (s : Adv)
    (hHmm : 0 < Hmm (reg w)) (hHsha : 0 < Hsha (reg w)) (hHo : 0 ≤ Ho (reg w))
    (hmm : wordsMoment G P μ (pm (reg w)) (η * (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) / Hmm (reg w)) (reg w) s
      (Mw w) (Dm w s) ≤ 1)
    (hsh : Good o w s → commitMoment G P μ (ps (reg w)) (η * (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) / Hsha (reg w))
      o (reg w) s (Mw w) (Mh w) (Ds o w s) ≤ 1) :
    expect (roundF reg μ pm ps η Mw Mh Good Dm Ds o w s) ≤ 1 := by
  by_cases hg : Good o w s
  · have e : roundF reg μ pm ps η Mw Mh Good Dm Ds o w s =
        combY G P μ (pm (reg w)) (ps (reg w)) η o (reg w) s (Mw w) (Mh w) (Dm w s) (Ds o w s) :=
      funext fun ω => by unfold roundF; rw [if_pos hg]
    rw [e]
    exact combined_moment_le G P μ _ _ η _ _ _ hHmm hHsha hHo o (reg w) s (Mw w) (Mh w) _ _ hmm (hsh hg)
  · have e : roundF reg μ pm ps η Mw Mh Good Dm Ds o w s = fun _ => 1 :=
      funext fun ω => by unfold roundF; rw [if_neg hg]
    rw [e]
    exact expect_le_of_le zero_le_one fun _ => le_rfl

/-! ## The floor -/

/-- **The floor over the period**: where the rounds' combined shortfall stays below `ln(1/δ)/η`, each round's
shortfall is at least its priced shortfall against the public bests, the exchange-free programs cost at most the
extracted ones plus `β` a bit, and those cost at most the credited use `S` plus the walk and log `Ξ` and exchanged at
most `K` bits, the credited use is at least `(1 − ε_A) C`. -/
theorem floor_of_rounds {k : ℕ} (H pm ps cw ch cw' ch' ew eh D : Fin k → ℝ) {η δ β K S Ξ : ℝ} (hβ : 0 ≤ β)
    (hS : 0 ≤ S) (hD : ∑ w, D w < Real.log (1 / δ) / η) (hDw : ∀ w, pm w - cw' w + (ps w - ch' w) ≤ D w)
    (hcw : ∀ w, cw' w ≤ cw w + β * ew w) (hch : ∀ w, ch' w ≤ ch w + β * eh w)
    (hcost : ∑ w, (cw w + ch w) ≤ S + Ξ) (hext : ∑ w, (ew w + eh w) ≤ K) :
    (1 - (epsWorkSound H pm ps η δ β K + Ξ / ∑ w, H w)) * ∑ w, H w ≤ S := by
  have h1 : ∑ w, (pm w - cw' w + (ps w - ch' w)) ≤ ∑ w, D w := Finset.sum_le_sum fun w _ => hDw w
  have h2 : ∑ w, (cw' w + ch' w) ≤ ∑ w, (cw w + ch w) + β * ∑ w, (ew w + eh w) := by
    rw [Finset.mul_sum, ← Finset.sum_add_distrib]
    exact Finset.sum_le_sum fun w _ => by linarith [hcw w, hch w]
  have h3 : β * ∑ w, (ew w + eh w) ≤ β * K := mul_le_mul_of_nonneg_left hext hβ
  have e1 : ∑ w, (pm w - cw' w + (ps w - ch' w)) = ∑ w, (pm w + ps w) - ∑ w, (cw' w + ch' w) := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun w _ => by ring
  have e2 : ∑ w, (H w - pm w - ps w) = ∑ w, H w - ∑ w, (pm w + ps w) := by
    rw [← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun w _ => by ring
  unfold epsWorkSound
  rw [e2]
  by_cases hC : ∑ w, H w = 0
  · rw [hC, mul_zero]
    exact hS
  · rw [sub_mul, add_mul, one_mul, div_mul_cancel₀ _ hC, div_mul_cancel₀ _ hC]
    linarith

/-! ## The cap on oracle calls -/

/-- **A run whose timed use fits makes at most `Q` oracle calls**: by (G3) the calls in the
sampling blocks' work use at least `calls · callUse_r` of a resource `r` a call uses, which fits within its capacity
`cap_r ≤ Q callUse_r`. -/
theorem calls_le_of_fits {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k) (Pd : Period G) {Q : ℝ}
    (hG3 : G.G3) (hneeds : ∀ j r, 0 ≤ Pd.needs j r) (hcap : ∃ r, 0 < G.callUse r ∧ Pd.cap r ≤ Q * G.callUse r)
    (o : P.Orc) (x : Fin k → P.Draw × P.Chk) {z : Pd.J → ℝ} (hfit : Pd.Fits (A.timedUse o x) z) :
    A.calls o x ≤ Q := by
  obtain ⟨r, hr, hc⟩ := hcap
  have h1 : A.calls o x * G.callUse r ≤ A.timedUse o x r := by
    unfold Prover.calls Prover.timedUse
    rw [Finset.sum_mul]
    exact Finset.sum_le_sum fun b _ => (hG3 (A.work b) (A.workIn o x b)).2 r
  have h2 : A.timedUse o x r ≤ Pd.cap r :=
    le_trans (le_add_of_nonneg_right (Finset.sum_nonneg fun j _ => mul_nonneg (hfit.1 j) (hneeds j r))) (hfit.2 r)
  exact le_of_mul_le_mul_right ((h1.trans h2).trans hc) hr

/-! ## The headline -/

/-- `Definitions.Pouw.Guarantees.Dither.WorkSound`. -/
theorem WorkSound : Pouw.Guarantees.Dither.WorkSound := by
  intro G P k Reg reg Obs verdict dev Pd μ Hmm Hsha Ho pubmm pubsha η δ β K κS Q κdmm κdsha Badv ξ κx κsha
    hreg hdual hpos hH hη hδ hδ1 hβ hK hξ hκd hcap hG1 hG2 hG3 hP1 hP2 hsak hsha hout hserv himp hextr A
  beta_reduce
  classical
  obtain ⟨Adv, adv, Mw, Mh, henc, hadv, hroot, huse, hext, hx⟩ := hextr hG1 hG2 hG3 hP1 hP2 A
  obtain ⟨hμ, hneeds, hval, z', hfit', hslack⟩ := hdual
  have hreg' : ∀ w, 0 < Hmm (reg w) ∧ 0 < Hsha (reg w) ∧ 0 ≤ Ho (reg w) ∧ pubmm (reg w) ≤ Hmm (reg w) ∧
      pubsha (reg w) ≤ Hsha (reg w) := fun w => hpos _ (hreg w)
  -- The exchange-free programs.
  choose Mw' hMw' using fun w => hout hG1 hG2 hP1 hP2 (Mw w)
  choose Mh' hMh' using fun w => hout hG1 hG2 hP1 hP2 (Mh w)
  have hevw : ∀ w i, G.Mc.eval (Mw' w) i = G.Mc.eval (Mw w) i := fun w i => (hMw' w i).1
  have hroot' : ∀ o x w, G.Mc.eval (Mh' w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw' w) (adv o x w, (x w).1)) =
      A.root o x w := by
    intro o x w
    rw [(hMh' w _).1, hevw]
    exact hroot o x w
  -- SAK(matmul)'s degenerate dithers, per round and advice.
  choose Dm hDm using fun (w : Fin k) (s : Adv) =>
    hsak hG1 hG2 hP1 hP2 (reg w) (hreg w) (Badv A) Adv henc s (Mw' w) fun i => (hMw' w i).2.1
  -- SHA-512's speed: the good oracles and advice, and their degenerate dithers.
  obtain ⟨Good, hGood⟩ : ∃ Good : P.Orc → Fin k → Adv → Prop, ∀ o w s, Good o w s ↔
      ∃ D : P.Draw → Prop, prob D ≤ κdsha (Badv A) ∧ commitMoment G P μ (pubsha (reg w))
        (η * (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) / Hsha (reg w)) o (reg w) s (Mw' w) (Mh' w) D ≤ 1 :=
    ⟨_, fun _ _ _ => Iff.rfl⟩
  have hDs0 : ∀ o w s, ∃ D : P.Draw → Prop, Good o w s → prob D ≤ κdsha (Badv A) ∧ commitMoment G P μ
      (pubsha (reg w)) (η * (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) / Hsha (reg w)) o (reg w) s (Mw' w) (Mh' w)
        D ≤ 1 := by
    intro o w s
    by_cases hg : Good o w s
    · obtain ⟨D, hD⟩ := (hGood o w s).mp hg
      exact ⟨D, fun _ => hD⟩
    · exact ⟨fun _ => False, fun h => absurd h hg⟩
  choose Ds hDs using hDs0
  -- The events' predicates.
  obtain ⟨okw, hokw⟩ : ∃ okw : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Prop, ∀ o x w, okw o x w ↔
      (P.ok (reg w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1)) ∧
        P.opens o (A.root o x w) (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1))) :=
    ⟨_, fun _ _ _ => Iff.rfl⟩
  obtain ⟨deg, hdeg⟩ : ∃ deg : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Prop, ∀ o x w, deg o x w ↔
      (Dm w (adv o x w) (x w).1 ∨ Ds o w (adv o x w) (x w).1) :=
    ⟨_, fun _ _ _ => Iff.rfl⟩
  obtain ⟨Dval, hDval⟩ : ∃ Dval : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → ℝ, ∀ o x w, Dval o x w =
      roundShortfall (pubmm (reg w)) (G.cost μ (Mw' w) (adv o x w, (x w).1)) +
        roundShortfall (pubsha (reg w))
          (G.cost μ (Mh' w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw' w) (adv o x w, (x w).1))) :=
    ⟨_, fun _ _ _ => rfl⟩
  have hF0 := roundF_nonneg reg μ pubmm pubsha η Mw' Mh' Good Dm Ds
  have hF1 : ∀ o w s, expect (roundF reg μ pubmm pubsha η Mw' Mh' Good Dm Ds o w s) ≤ 1 := fun o w s =>
    expect_roundF_le reg μ pubmm pubsha Hmm Hsha Ho η Mw' Mh' Good Dm Ds o w s (hreg' w).1 (hreg' w).2.1
      (hreg' w).2.2.1 (hDm w s).2 fun hg => (hDs o w s hg).2
  have hfac : ∀ o x w, Good o w (adv o x w) → ¬ deg o x w → okw o x w →
      roundF reg μ pubmm pubsha η Mw' Mh' Good Dm Ds o w (adv o x w) (x w) = Real.exp (η * Dval o x w) := by
    intro o x w hg hd hk
    rw [hdeg] at hd
    rw [hokw] at hk
    rw [hDval]
    unfold roundF
    rw [if_pos hg]
    unfold combY
    rw [if_pos]
    refine ⟨fun h => hd (Or.inl h), fun h => hd (Or.inr h), ?_, ?_⟩
    · rw [hevw]
      exact hk.1
    · rw [hroot', hevw]
      exact hk.2
  -- The error is nonnegative.
  have hC0 : 0 ≤ ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) :=
    Finset.sum_nonneg fun w _ => by linarith [hreg' w]
  have hΞ0 : 0 ≤ ∑ r, μ r * ξ A r := Finset.sum_nonneg fun r _ => mul_nonneg (hμ r) (hξ A r)
  have hlog : 0 ≤ Real.log (1 / δ) := Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
  have hε : 0 ≤ epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
      (fun w => pubsha (reg w)) η δ β K + (∑ r, μ r * ξ A r) / ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) := by
    unfold epsWorkSound
    refine add_nonneg (div_nonneg (add_nonneg (add_nonneg (Finset.sum_nonneg fun w _ => ?_)
      (div_nonneg hlog hη.le)) (mul_nonneg hβ hK)) hC0) (div_nonneg hΞ0 hC0)
    show 0 ≤ Hmm (reg w) + Hsha (reg w) + Ho (reg w) - pubmm (reg w) - pubsha (reg w)
    linarith [hreg' w]
  -- The floor, off the luck.
  have hfloor : ∀ o x, A.Passes verdict dev o x → A.calls o x ≤ Q → ∑ w, Dval o x w < Real.log (1 / δ) / η →
      (1 - (epsWorkSound (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
        (fun w => pubsha (reg w)) η δ β K + (∑ r, μ r * ξ A r) / ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)))) *
        ∑ r, μ r * (1 * Pd.hon r) ≤ ∑ r, μ r * A.timedUse o x r := by
    intro o x hv hq hD
    have hC : ∑ r, μ r * (1 * Pd.hon r) = ∑ w, (Hmm (reg w) + Hsha (reg w) + Ho (reg w)) := by
      rw [hH]
      simp only [one_mul]
    rw [hC]
    have hS0 : 0 ≤ ∑ r, μ r * A.timedUse o x r := Finset.sum_nonneg fun r _ => mul_nonneg (hμ r)
      (Finset.sum_nonneg fun w _ => ((hG2.2 (A.work w) (A.workIn o x w)).2 r).1)
    have hcost : ∑ w, (G.cost μ (Mw w) (adv o x w, (x w).1) +
          G.cost μ (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1))) ≤
        ∑ r, μ r * A.timedUse o x r + ∑ r, μ r * ξ A r := by
      have h1 : ∑ w, (G.cost μ (Mw w) (adv o x w, (x w).1) +
            G.cost μ (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1))) =
          ∑ r, μ r * ∑ w, (G.use (Mw w) (adv o x w, (x w).1) r +
            G.use (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)) r) := by
        unfold Accel.cost
        simp only [Finset.mul_sum, mul_add, Finset.sum_add_distrib]
        congr 1 <;> exact Finset.sum_comm
      rw [h1, ← Finset.sum_add_distrib]
      exact Finset.sum_le_sum fun r _ => by
        rw [← mul_add]
        exact mul_le_mul_of_nonneg_left (huse o x hq r) (hμ r)
    have hextk : ∑ w, (G.ext (Mw w) (adv o x w, (x w).1) +
        G.ext (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1))) ≤ K :=
      calc _ ≤ ∑ w, G.ext (A.work w) (A.workIn o x w) := Finset.sum_le_sum fun w _ => hext o x w
        _ ≤ A.ext o x := Finset.sum_le_sum fun w _ =>
            le_add_of_nonneg_right (hG2.2 (A.answer w) (A.answerIn o x w)).1
        _ ≤ K := himp A o x hv
    exact floor_of_rounds (fun w => Hmm (reg w) + Hsha (reg w) + Ho (reg w)) (fun w => pubmm (reg w))
      (fun w => pubsha (reg w))
      (fun w => G.cost μ (Mw w) (adv o x w, (x w).1))
      (fun w => G.cost μ (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)))
      (fun w => G.cost μ (Mw' w) (adv o x w, (x w).1))
      (fun w => G.cost μ (Mh' w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw' w) (adv o x w, (x w).1)))
      (fun w => G.ext (Mw w) (adv o x w, (x w).1))
      (fun w => G.ext (Mh w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw w) (adv o x w, (x w).1)))
      (fun w => Dval o x w) hβ hS0 hD
      (fun w => by
        show _ ≤ Dval o x w
        rw [hDval]
        unfold roundShortfall
        exact add_le_add (le_max_right _ _) (le_max_right _ _))
      (fun w => (hMw' w _).2.2)
      (fun w => by
        show G.cost μ (Mh' w) (o, (adv o x w, (x w).1), G.Mc.eval (Mw' w) (adv o x w, (x w).1)) ≤ _
        rw [hevw]
        exact (hMh' w _).2.2)
      hcost hextk
  -- The luck: Markov over the stopped supermartingale.
  have E5 : prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
      (∀ w, Good ox.1 w (adv ox.1 ox.2 w) ∧ ¬ deg ox.1 ox.2 w ∧ okw ox.1 ox.2 w) ∧
        Real.log (1 / δ) / η ≤ ∑ w, Dval ox.1 ox.2 w) ≤ δ := by
    refine prob_le_of_slices_fst _ hδ.le fun o => ?_
    have hδeq : Real.exp (-(η * (Real.log (1 / δ) / η))) = δ := by
      rw [mul_comm η, div_mul_cancel₀ _ hη.ne', one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
    refine (prob_rounds_le (adv o) (hadv o) (roundF reg μ pubmm pubsha η Mw' Mh' Good Dm Ds o) (hF0 o) (hF1 o)
      (η * (Real.log (1 / δ) / η)) fun x hx => ?_).trans hδeq.le
    obtain ⟨hall, hD⟩ := hx
    rw [Finset.prod_congr rfl fun w _ => hfac o x w (hall w).1 (hall w).2.1 (hall w).2.2, ← Real.exp_sum,
      ← Finset.mul_sum]
    exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hD hη.le)
  -- The degenerate dithers.
  have hκ2 : 0 ≤ κdmm (Badv A) + κdsha (Badv A) := add_nonneg (hκd _).1 (hκd _).2
  have E4 : prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
      ∃ w, Good ox.1 w (adv ox.1 ox.2 w) ∧ deg ox.1 ox.2 w) ≤ k * (κdmm (Badv A) + κdsha (Badv A)) := by
    refine le_trans (prob_mono (q := fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
      ∃ w ∈ (Finset.univ : Finset (Fin k)), Good ox.1 w (adv ox.1 ox.2 w) ∧
        (Dm w (adv ox.1 ox.2 w) (ox.2 w).1 ∨ Ds ox.1 w (adv ox.1 ox.2 w) (ox.2 w).1))
      fun ox h => ?_) ?_
    · obtain ⟨w, hg, hd⟩ := h
      exact ⟨w, Finset.mem_univ w, hg, (hdeg _ _ _).mp hd⟩
    refine le_trans (prob_exists_le _ _) ?_
    rw [show (k : ℝ) * (κdmm (Badv A) + κdsha (Badv A)) = ∑ _w : Fin k, (κdmm (Badv A) + κdsha (Badv A)) by
      rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]]
    refine Finset.sum_le_sum fun w _ => prob_le_of_slices_fst _ hκ2 fun o => ?_
    refine prob_round_le (adv o) (hadv o) w (fun s ω => Good o w s ∧ (Dm w s ω.1 ∨ Ds o w s ω.1)) hκ2 fun s => ?_
    by_cases hg : Good o w s
    · refine le_trans (prob_mono fun ω h => h.2) (le_trans (prob_or_le _ _) (add_le_add ?_ ?_))
      · exact le_trans (prob_fst_le (Dm w s)) (hDm w s).1
      · exact le_trans (prob_fst_le (Ds o w s)) (hDs o w s hg).1
    · refine le_trans (prob_mono (q := fun _ => False) fun ω h => hg h.1) ?_
      rw [prob_false]
      exact hκ2
  -- The bad oracles and advice.
  have E3 : prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => ∃ w, ¬ Good ox.1 w (adv ox.1 ox.2 w)) ≤ κsha A := by
    refine le_trans (prob_mono fun ox h => ?_) (hsha hG1 hG2 hG3 hP1 hP2 A (Badv A) Adv henc adv hadv Mw' Mh'
      (fun w i => (hMw' w i).2.1) (fun w i => (hMh' w i).2.1) hroot')
    obtain ⟨w, hw⟩ := h
    exact ⟨w, fun hD => hw ((hGood _ _ _).mpr hD)⟩
  -- The extraction.
  have E2 : prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
      A.Accepted reg ox.1 ox.2 ∧ A.calls ox.1 ox.2 ≤ Q ∧ ∃ w, ¬ okw ox.1 ox.2 w) ≤ κx A := by
    refine le_trans (prob_mono fun ox h => ?_) hx
    obtain ⟨hacc, hq, w, hw⟩ := h
    exact ⟨hacc, hq, w, fun hk => hw ((hokw _ _ _).mpr hk)⟩
  -- The union bound.
  refine le_trans (prob_mono (q := fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
      ((((A.Passes verdict dev ox.1 ox.2 ∧ ¬ A.Accepted reg ox.1 ox.2) ∨
          (A.Accepted reg ox.1 ox.2 ∧ A.calls ox.1 ox.2 ≤ Q ∧ ∃ w, ¬ okw ox.1 ox.2 w)) ∨
        (∃ w, ¬ Good ox.1 w (adv ox.1 ox.2 w))) ∨
        (∃ w, Good ox.1 w (adv ox.1 ox.2 w) ∧ deg ox.1 ox.2 w)) ∨
        ((∀ w, Good ox.1 w (adv ox.1 ox.2 w) ∧ ¬ deg ox.1 ox.2 w ∧ okw ox.1 ox.2 w) ∧
          Real.log (1 / δ) / η ≤ ∑ w, Dval ox.1 ox.2 w)) fun ox hox => ?_) ?_
  · obtain ⟨o, x⟩ := ox
    obtain ⟨hv, z, hfit, hnot⟩ := hox
    by_cases hacc : A.Accepted reg o x
    · by_cases hok : ∀ w, okw o x w
      · by_cases hg : ∀ w, Good o w (adv o x w)
        · by_cases hdg : ∃ w, deg o x w
          · obtain ⟨w, hw⟩ := hdg
            exact Or.inl (Or.inr ⟨w, hg w, hw⟩)
          · by_cases hD : Real.log (1 / δ) / η ≤ ∑ w, Dval o x w
            · exact Or.inr ⟨fun w => ⟨hg w, fun h => hdg ⟨w, h⟩, hok w⟩, hD⟩
            · exact absurd (FreedOfFloor G Pd 1 _ (A.timedUse o x) μ z z' hμ hneeds hval zero_le_one hε hfit'
                hslack hfit (hfloor o x hv (calls_le_of_fits A Pd hG3 hneeds hcap o x hfit) (not_le.mp hD))) hnot
        · exact Or.inl (Or.inl (Or.inr (not_forall.mp hg)))
      · exact Or.inl (Or.inl (Or.inl (Or.inr
          ⟨hacc, calls_le_of_fits A Pd hG3 hneeds hcap o x hfit, not_forall.mp hok⟩)))
    · exact Or.inl (Or.inl (Or.inl (Or.inl ⟨hv, hacc⟩)))
  · refine (prob_or5_le _ _ _ _ _).trans ?_
    have := add_le_add (add_le_add (add_le_add (add_le_add (hserv A) E2) E3) E4) E5
    linarith

/-- The segmented schedule's sum: `Σ_{i < n} (2 (i + 1) M + 1) = n ((n + 1) M + 1)`. -/
theorem sum_segments (M n : ℕ) : ∑ i : Fin n, (2 * (((i : ℕ) + 1) * M) + 1) = n * ((n + 1) * M + 1) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Fin.sum_univ_castSucc]
    simp only [Fin.val_castSucc, Fin.val_last]
    rw [ih]
    ring

/-- `Definitions.Pouw.Guarantees.Dither.WorkSoundNoFold`: `WorkSound` with `ExtractAdv` from `ExtractOfRealize` at
the budget `q = (k + 1)(⌈Q⌉ + V)` and the schedule `τ_w = (w + 1)(⌈Q⌉ + V)`, whose extraction term
`q² + Σ_w q (2 τ_w + 1)` is `q² + k q (q + 1)` (`sum_segments`). -/
theorem WorkSoundNoFold : Pouw.Guarantees.Dither.WorkSoundNoFold := by
  intro G nf k Reg reg Obs verdict dev Pd μ Hmm Hsha Ho pubmm pubsha η δ β K κS Q κsha κdmm κdsha Badv V X D _ _ _ _ _
    E eo rD pl dirs ξw ξl hreg hdual hpos hH hη hδ hδ1 hβ hK hξw hξl hκd hcap hopens hG1 hG2 hG3 horc hP1 hP2
    hsak hsha hout hserv himp hR A
  have hx := Pouw.SecurityProofs.ShaRealize.ExtractOfRealize G nf k reg Q X D E eo rD pl dirs ξw ξl
    (fun _ => (k + 1) * (⌈Q⌉₊ + V)) (fun _ w => ((w : ℕ) + 1) * (⌈Q⌉₊ + V)) (fun _ => Badv) hopens horc hR
  have hk : ∑ w : Fin k, (k + 1) * (⌈Q⌉₊ + V) * (2 * (((w : ℕ) + 1) * (⌈Q⌉₊ + V)) + 1) =
      k * ((k + 1) * (⌈Q⌉₊ + V) * ((k + 1) * (⌈Q⌉₊ + V) + 1)) := by
    rw [← Finset.mul_sum, sum_segments]
    ring
  refine (WorkSound G (noFoldRound nf) k Reg reg Obs verdict dev Pd μ Hmm Hsha Ho pubmm pubsha η δ β K κS Q κdmm κdsha
    (fun _ => Badv) (fun _ r => k * ξw r + ((k + 1) * (⌈Q⌉₊ + V) : ℕ) * ξl r)
    (fun _ => (((k + 1) * (⌈Q⌉₊ + V) * ((k + 1) * (⌈Q⌉₊ + V)) +
        ∑ w : Fin k, (k + 1) * (⌈Q⌉₊ + V) * (2 * (((w : ℕ) + 1) * (⌈Q⌉₊ + V)) + 1) : ℕ) : ℝ) /
      Fintype.card D) (fun _ => κsha) hreg hdual hpos hH hη hδ hδ1 hβ hK
    (fun _ r => add_nonneg (mul_nonneg (Nat.cast_nonneg _) (hξw r)) (mul_nonneg (Nat.cast_nonneg _) (hξl r))) hκd
    hcap hG1 hG2 hG3 hP1 hP2 hsak hsha hout hserv himp hx A).trans (le_of_eq ?_)
  simp only [hk]

/-- `Definitions.Pouw.Guarantees.Dither.FleetSound`: the union bound over the units, each unit's event sliced at the
other units' coins. -/
theorem FleetSound : Pouw.Guarantees.Dither.FleetSound := by
  intro G P k N Obs verdict dev Pd ε κ C κmax hκ hunit A hA
  classical
  refine le_trans (prob_mono (q := fun ox : P.Orc × (Fin N → Fin k → P.Draw × P.Chk) =>
      ∃ u ∈ (Finset.univ : Finset (Fin N)), (A u fun v => ox.2 v.1).Passes (verdict u) (dev u) ox.1 (ox.2 u) ∧
        ∃ z, (Pd u).Fits ((A u fun v => ox.2 v.1).timedUse ox.1 (ox.2 u)) z ∧
          ¬ (Pd u).FreedWithin 1 z (ε u (A u fun v => ox.2 v.1)) (C u))
      fun ox ⟨u, hu⟩ => ⟨u, Finset.mem_univ u, hu⟩) ((prob_exists_le _ _).trans (Finset.sum_le_sum fun u _ => ?_))
  let e : P.Orc × (Fin N → Fin k → P.Draw × P.Chk) ≃
      (P.Orc × (Fin k → P.Draw × P.Chk)) × ({v : Fin N // v ≠ u} → Fin k → P.Draw × P.Chk) :=
    ((Equiv.refl P.Orc).prodCongr (Equiv.piSplitAt u fun _ => Fin k → P.Draw × P.Chk)).trans
      (Equiv.prodAssoc _ _ _).symm
  let F : (P.Orc × (Fin k → P.Draw × P.Chk)) × ({v : Fin N // v ≠ u} → Fin k → P.Draw × P.Chk) → Prop :=
    fun b => (A u b.2).Passes (verdict u) (dev u) b.1.1 b.1.2 ∧
      ∃ z, (Pd u).Fits ((A u b.2).timedUse b.1.1 b.1.2) z ∧ ¬ (Pd u).FreedWithin 1 z (ε u (A u b.2)) (C u)
  calc prob _ = prob (fun ox => F (e ox)) := rfl
    _ = prob F := by rw [prob_equiv e]; simp only [Equiv.apply_symm_apply]
    _ ≤ κmax u := prob_le_of_slices F (hκ u) fun y => (hunit u (A u y)).trans (hA u y)

end Pouw.SecurityProofs.Dither
