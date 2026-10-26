import Definitions.Pouw.Guarantees.Dither
import Proofs.Pouw.RelativeSecurity

/-!
# The no-fold Dither protocol's work-soundness: the headline's proof

* `FreedOfFloor`: the guarantee `Definitions.Pouw.Guarantees.Dither.FreedOfFloor`, weak duality of the period's linear
  program.
* `WorkSound`: the guarantee `Definitions.Pouw.Guarantees.Dither.WorkSound`. `ExtractAdv` splits each round's work into a
  words program on advice fixed before the round's coins and a commitment program on those words. Their two moments (SAK(matmul)'s and SHA-512's speed's)
  combine by the convexity of `exp` at the honest round's shares (`combined_moment_le`). A round's factor reads only the
  coins up to that round, so the rounds' product is a supermartingale over the fresh coins (`expect_prod_adapted`, the
  argument of `RoundGame.expect_weight_le` on `Fin k → Draw × Chk`), and Markov bounds the luck by `δ`. The union bound
  adds the service's `κ_S` and the extraction's `κ_x`; off those three events the rounds meet the work floor
  (`floor_of_luck`), and `FreedOfFloor` turns the floor into freed compute.
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

/-! ## One round -/

/-- Convexity of `exp` at three points, the third `0`. -/
theorem exp_convex3 {a b c u v : ℝ} (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h : a + b + c = 1) :
    Real.exp (a * u + b * v) ≤ a * Real.exp u + b * Real.exp v + c := by
  have := convexOn_exp.map_sum_le (t := Finset.univ) (w := ![a, b, c]) (p := ![u, v, 0])
    (fun i _ => by fin_cases i <;> simp [ha, hb, hc]) (by simp [Fin.sum_univ_three, h])
    (fun i _ => Set.mem_univ _)
  simpa [Fin.sum_univ_three] using this

open Classical in
/-- **One round's combined moment**: SAK(matmul)'s moment on the words and SHA-512's speed's on their commitment, at
the weights `H_mm/H` and `H_sha/H`, with the rest of the honest round `H_o/H` at shortfall `0`, bound the combined
shortfall's moment by 1 (the convexity of `exp`). -/
theorem combined_moment_le (G : Accel) (P : RoundProto) (μ : G.Res → ℝ) (cmm csha η β Hmm Hsha Ho : ℝ)
    (hHmm : 0 < Hmm) (hHsha : 0 < Hsha) (hHo : 0 ≤ Ho) (o : P.Orc) (a : P.Inst) {St : Type} (s : St)
    (Mw : G.Mc.Alg (St × P.Draw) P.Tr) (Mh : G.Mc.Alg (P.Orc × P.Tr) P.Root)
    (hmm : wordsMoment G P μ cmm η β Hmm a s Mw ≤ 1) (hsh : commitMoment G P μ csha η β Hsha o a s Mw Mh ≤ 1) :
    expect (fun ω : P.Draw × P.Chk => if P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) then
      Real.exp (η * (Hmm / (Hmm + Hsha + Ho) *
          roundShortfall cmm β Hmm (G.cost μ Mw (s, ω.1)) (G.ext Mw (s, ω.1)) +
        Hsha / (Hmm + Hsha + Ho) * roundShortfall csha β Hsha (G.cost μ Mh (o, G.Mc.eval Mw (s, ω.1)))
          (G.ext Mh (o, G.Mc.eval Mw (s, ω.1))))) else 0) ≤ 1 := by
  have hH : 0 < Hmm + Hsha + Ho := by linarith
  have ha : 0 ≤ Hmm / (Hmm + Hsha + Ho) := div_nonneg hHmm.le hH.le
  have hb : 0 ≤ Hsha / (Hmm + Hsha + Ho) := div_nonneg hHsha.le hH.le
  have hc : 0 ≤ Ho / (Hmm + Hsha + Ho) := div_nonneg hHo hH.le
  have hsum : Hmm / (Hmm + Hsha + Ho) + Hsha / (Hmm + Hsha + Ho) + Ho / (Hmm + Hsha + Ho) = 1 := by
    rw [← add_div, ← add_div, div_self hH.ne']
  calc _ ≤ expect (fun ω : P.Draw × P.Chk =>
        Hmm / (Hmm + Hsha + Ho) * (if P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) then
          Real.exp (η * roundShortfall cmm β Hmm (G.cost μ Mw (s, ω.1)) (G.ext Mw (s, ω.1))) else 0) +
        Hsha / (Hmm + Hsha + Ho) * (if P.ok a ω.1 ω.2 (G.Mc.eval Mw (s, ω.1)) then
          Real.exp (η * roundShortfall csha β Hsha (G.cost μ Mh (o, G.Mc.eval Mw (s, ω.1)))
            (G.ext Mh (o, G.Mc.eval Mw (s, ω.1)))) else 0) + Ho / (Hmm + Hsha + Ho)) := by
        refine expect_mono fun ω => ?_
        split_ifs with hok
        · rw [mul_add, mul_left_comm η (Hmm / (Hmm + Hsha + Ho)), mul_left_comm η (Hsha / (Hmm + Hsha + Ho))]
          exact exp_convex3 ha hb hc hsum
        · simp only [mul_zero, zero_add]
          exact hc
    _ ≤ Hmm / (Hmm + Hsha + Ho) * wordsMoment G P μ cmm η β Hmm a s Mw +
          Hsha / (Hmm + Hsha + Ho) * commitMoment G P μ csha η β Hsha o a s Mw Mh + Ho / (Hmm + Hsha + Ho) :=
        expect_affine3 _ _ _ hc _ _
    _ ≤ Hmm / (Hmm + Hsha + Ho) * 1 + Hsha / (Hmm + Hsha + Ho) * 1 + Ho / (Hmm + Hsha + Ho) :=
        add_le_add (add_le_add (mul_le_mul_of_nonneg_left hmm ha) (mul_le_mul_of_nonneg_left hsh hb)) le_rfl
    _ = 1 := by rw [mul_one, mul_one, hsum]

/-- The floor of one round: its combined shortfall is at least its priced shortfall against the public bests. -/
theorem roundD_ge {Hmm Hsha H pm ps cw ch ew eh β : ℝ} (hHmm : 0 < Hmm) (hHsha : 0 < Hsha) (hH : 0 < H) :
    ((pm + ps) - (cw + ch) - β * (ew + eh)) / H ≤
      Hmm / H * roundShortfall (pm / Hmm) β Hmm cw ew + Hsha / H * roundShortfall (ps / Hsha) β Hsha ch eh := by
  unfold roundShortfall
  have h1 : Hmm / H * (pm / Hmm - (cw + β * ew) / Hmm) ≤ Hmm / H * max 0 (pm / Hmm - (cw + β * ew) / Hmm) :=
    mul_le_mul_of_nonneg_left (le_max_right _ _) (div_nonneg hHmm.le hH.le)
  have h2 : Hsha / H * (ps / Hsha - (ch + β * eh) / Hsha) ≤ Hsha / H * max 0 (ps / Hsha - (ch + β * eh) / Hsha) :=
    mul_le_mul_of_nonneg_left (le_max_right _ _) (div_nonneg hHsha.le hH.le)
  have e1 : Hmm / H * (pm / Hmm - (cw + β * ew) / Hmm) = (pm - (cw + β * ew)) / H := by
    rw [← sub_div, div_mul_div_comm, mul_comm Hmm, mul_div_mul_right _ _ hHmm.ne']
  have e2 : Hsha / H * (ps / Hsha - (ch + β * eh) / Hsha) = (ps - (ch + β * eh)) / H := by
    rw [← sub_div, div_mul_div_comm, mul_comm Hsha, mul_div_mul_right _ _ hHsha.ne']
  have e3 : ((pm + ps) - (cw + ch) - β * (ew + eh)) / H = (pm - (cw + β * ew)) / H + (ps - (ch + β * eh)) / H := by
    ring
  linarith

/-- The floor over the period: off the luck event, the rounds' priced use is at least `(1 − ε) k H`. -/
theorem floor_of_luck {k : ℕ} {Hmm Hsha Ho pm ps η δ β K S Dsum Cw E : ℝ} (hHmm : 0 < Hmm) (hHsha : 0 < Hsha)
    (hHo : 0 ≤ Ho) (hη : 0 < η) (hβ : 0 ≤ β) (hS0 : 0 ≤ S) (hD : Dsum < Real.log (1 / δ) / η)
    (hDlow : ((k : ℝ) * (pm + ps) - Cw - β * E) / (Hmm + Hsha + Ho) ≤ Dsum) (hCw : Cw ≤ S) (hE : E ≤ K) :
    (1 - epsWorkSound (pm / Hmm) (ps / Hsha) Hmm Hsha Ho η δ β K k) * ((k : ℝ) * (Hmm + Hsha + Ho)) ≤ S := by
  have hH : 0 < Hmm + Hsha + Ho := by linarith
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simpa using hS0
  · have hk' : (0 : ℝ) < k := by exact_mod_cast hk
    have e : (1 - epsWorkSound (pm / Hmm) (ps / Hsha) Hmm Hsha Ho η δ β K k) * ((k : ℝ) * (Hmm + Hsha + Ho)) =
        (k : ℝ) * (pm + ps) - (Hmm + Hsha + Ho) * (Real.log (1 / δ) / η) - β * K := by
      unfold epsWorkSound
      generalize Real.log (1 / δ) = L
      have := hHmm.ne'
      have := hHsha.ne'
      have := hη.ne'
      have := hk'.ne'
      have := hH.ne'
      field_simp
      ring
    have h1 : (k : ℝ) * (pm + ps) - Cw - β * E < (Hmm + Hsha + Ho) * (Real.log (1 / δ) / η) := by
      rw [mul_comm (Hmm + Hsha + Ho)]
      exact (div_lt_iff₀ hH).mp (hDlow.trans_lt hD)
    have h2 : β * E ≤ β * K := mul_le_mul_of_nonneg_left hE hβ
    rw [e]
    linarith

/-! ## The rounds of a prover -/

section Rounds

variable {G : Accel} {P : RoundProto} {k : ℕ} (A : Prover G P k)

/-- Round `w`'s state reads only the coins of the rounds before it. -/
theorem state_congr (o : P.Orc) (x x' : Fin k → P.Draw × P.Chk) :
    ∀ w : ℕ, (∀ j : Fin k, j.val < w → x j = x' j) → A.state o x w = A.state o x' w
  | 0, _ => rfl
  | w + 1, h => by
    have h' := state_congr o x x' w fun j hj => h j (Nat.lt_succ_of_lt hj)
    by_cases hw : w < k
    · simp only [Prover.state, dite_eq_left hw, h', h ⟨w, hw⟩ (Nat.lt_succ_self w)]
    · simp only [Prover.state, dite_eq_right hw, h']

/-- Round `w`'s instance reads only the coins of the rounds before it. -/
theorem instAt_congr (o : P.Orc) (x x' : Fin k → P.Draw × P.Chk) (w : Fin k)
    (h : ∀ j : Fin k, j < w → x j = x' j) : A.instAt o x w = A.instAt o x' w := by
  have hs := state_congr A o x x' w fun j hj => h j (Fin.lt_def.mpr hj)
  simp only [Prover.instAt, hs]

/-- Round `w`'s instance at its own coin `ω`, the other coins fixed. -/
theorem instAt_update (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) (ω : P.Draw × P.Chk) :
    A.instAt o (Function.update x w ω) w = A.instAt o x w := by
  have hs := state_congr A o (Function.update x w ω) x w fun j hj =>
    Function.update_of_ne (Fin.ne_of_val_ne (Nat.ne_of_lt hj)) _ _
  simp only [Prover.instAt, hs]

variable {Adv : Type} (adv : P.Orc → (Fin k → P.Draw × P.Chk) → Fin k → Adv)

/-- Round `w`'s combined shortfall at the oracle and the coins: the extracted words program's (on the round's advice
and dither) and commitment program's shortfalls against the public bests, weighted by their shares `H_mm/H` and
`H_sha/H` of the honest round. -/
noncomputable def roundD (μ : G.Res → ℝ) (cmm csha β Hmm Hsha H : ℝ) (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (Mh : Fin k → G.Mc.Alg (P.Orc × P.Tr) P.Root) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) : ℝ :=
  Hmm / H * roundShortfall cmm β Hmm (G.cost μ (Mw w) (adv o x w, (x w).1)) (G.ext (Mw w) (adv o x w, (x w).1)) +
    Hsha / H * roundShortfall csha β Hsha (G.cost μ (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)))
      (G.ext (Mh w) (o, G.Mc.eval (Mw w) (adv o x w, (x w).1)))

/-- Round `w`'s check on the extracted words. -/
def roundOk (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) :
    Prop :=
  P.ok (A.instAt o x w) (x w).1 (x w).2 (G.Mc.eval (Mw w) (adv o x w, (x w).1))

open Classical in
/-- Round `w`'s factor of the supermartingale: `1[check passes] · e^{η D_w}`. -/
noncomputable def roundY (μ : G.Res → ℝ) (cmm csha η β Hmm Hsha H : ℝ)
    (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr) (Mh : Fin k → G.Mc.Alg (P.Orc × P.Tr) P.Root) (o : P.Orc)
    (x : Fin k → P.Draw × P.Chk) (w : Fin k) : ℝ :=
  if roundOk A adv Mw o x w then Real.exp (η * roundD adv μ cmm csha β Hmm Hsha H Mw Mh o x w) else 0

theorem roundY_nonneg (μ : G.Res → ℝ) (cmm csha η β Hmm Hsha H : ℝ) (Mw : Fin k → G.Mc.Alg (Adv × P.Draw) P.Tr)
    (Mh : Fin k → G.Mc.Alg (P.Orc × P.Tr) P.Root) (o : P.Orc) (x : Fin k → P.Draw × P.Chk) (w : Fin k) :
    0 ≤ roundY A adv μ cmm csha η β Hmm Hsha H Mw Mh o x w := by
  unfold roundY
  split_ifs
  · exact (Real.exp_pos _).le
  · exact le_rfl

end Rounds

/-! ## The headline -/

/-- `Definitions.Pouw.Guarantees.Dither.WorkSound`. -/
theorem WorkSound : Pouw.Guarantees.Dither.WorkSound := by
  intro G P k verdict Pd μ pubmm pubsha Hmm Hsha Ho η δ β K κS κx hdual hHmm hHsha hHo hH hpm hps hη hδ hδ1 hβ hK
    hG1 hG2 hP1 hP2 hsak hsha hserv himp hextr A
  classical
  obtain ⟨Adv, adv, Mw, Mh, hadv, hcomp, huse, hext, hx⟩ := hextr hG1 hG2 hP1 hP2 A
  obtain ⟨hμ, hneeds, hval, z', hfit', hslack⟩ := hdual
  have hHpos : 0 < Hmm + Hsha + Ho := by linarith
  have hlog : 0 ≤ Real.log (1 / δ) := Real.log_nonneg (by rw [le_div_iff₀ hδ]; linarith)
  have e1 : Hmm * (1 - pubmm / Hmm) = Hmm - pubmm := by
    rw [mul_sub, mul_one, mul_comm Hmm (pubmm / Hmm), div_mul_cancel₀ pubmm hHmm.ne']
  have e2 : Hsha * (1 - pubsha / Hsha) = Hsha - pubsha := by
    rw [mul_sub, mul_one, mul_comm Hsha (pubsha / Hsha), div_mul_cancel₀ pubsha hHsha.ne']
  have hε0 : 0 ≤ epsWorkSound (pubmm / Hmm) (pubsha / Hsha) Hmm Hsha Ho η δ β K k := by
    unfold epsWorkSound
    rw [e1, e2]
    exact add_nonneg (add_nonneg (div_nonneg (by linarith) hHpos.le)
      (div_nonneg hlog (mul_nonneg hη.le (Nat.cast_nonneg k))))
      (div_nonneg (mul_nonneg hβ hK) (mul_nonneg (Nat.cast_nonneg k) hHpos.le))
  -- The luck: the rounds' checks all pass on the extracted words, and their combined shortfall reaches `ln(1/δ)/η`.
  have E3 : prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) => (∀ w, roundOk A adv Mw ox.1 ox.2 w) ∧
      Real.log (1 / δ) / η ≤
        ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh ox.1 ox.2 w) ≤ δ := by
    refine prob_le_of_slices_fst _ hδ.le fun o => ?_
    have hE : expect (fun x : Fin k → P.Draw × P.Chk =>
        ∏ w, roundY A adv μ (pubmm / Hmm) (pubsha / Hsha) η β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh o x w) ≤ 1 := by
      refine expect_prod_adapted k _ (fun w x => roundY_nonneg A adv _ _ _ _ _ _ _ _ _ _ o x w) ?_ ?_
      · intro i x x' h
        have h1 := instAt_congr A o x x' i fun j hj => h j hj.le
        have h2 : adv o x i = adv o x' i := hadv o x x' i fun v hv => h v hv.le
        have h3 := h i le_rfl
        unfold roundY roundOk roundD
        rw [h1, h2, h3]
      · intro i x
        refine Eq.trans_le ?_ (combined_moment_le G P μ (pubmm / Hmm) (pubsha / Hsha) η β Hmm Hsha Ho hHmm hHsha hHo
          o (A.instAt o x i) (adv o x i) (Mw i) (Mh i)
          (hsak hG1 hG2 hP1 hP2 _ _ _ _) (hsha hG1 hG2 hP1 hP2 _ _ _ _ _ _ (hcomp i o)))
        congr 1
        funext ω
        have h1 := instAt_update A o x i ω
        have h2 : adv o (Function.update x i ω) i = adv o x i :=
          hadv o _ _ i fun v hv => Function.update_of_ne hv.ne ω x
        unfold roundY roundOk roundD
        rw [h1, h2, Function.update_self]
    have hδeq : Real.exp (-(η * (Real.log (1 / δ) / η))) = δ := by
      rw [mul_comm η, div_mul_cancel₀ _ hη.ne', one_div, Real.log_inv, neg_neg, Real.exp_log hδ]
    refine (prob_le_of_expect_le_one
      (fun x => Finset.prod_nonneg fun w _ => roundY_nonneg A adv _ _ _ _ _ _ _ _ _ _ o x w) hE
      (η * (Real.log (1 / δ) / η)) fun x hx => ?_).trans hδeq.le
    have hok : ∀ w, roundOk A adv Mw o x w := hx.1
    have hD : Real.log (1 / δ) / η ≤
        ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh o x w := hx.2
    have hprod : ∏ w, roundY A adv μ (pubmm / Hmm) (pubsha / Hsha) η β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh o x w =
        Real.exp (η * ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh o x w) := by
      rw [Finset.mul_sum, Real.exp_sum]
      exact Finset.prod_congr rfl fun w _ => by unfold roundY; rw [ite_eq_left (hok w)]
    rw [hprod]
    exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left hD hη.le)
  -- Off the three bad events, the rounds meet the work floor.
  have key : ∀ o x, verdict A o x → (∀ w, roundOk A adv Mw o x w) →
      ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh o x w <
        Real.log (1 / δ) / η →
      (1 - epsWorkSound (pubmm / Hmm) (pubsha / Hsha) Hmm Hsha Ho η δ β K k) * ∑ r, μ r * ((k : ℝ) * Pd.hon r) ≤
        ∑ r, μ r * A.use o x r := by
    intro o x hv hok hD
    have hC : ∑ r, μ r * ((k : ℝ) * Pd.hon r) = (k : ℝ) * (Hmm + Hsha + Ho) := by
      rw [hH, Finset.mul_sum]
      exact Finset.sum_congr rfl fun r _ => by ring
    rw [hC]
    have hcost : ∀ w, G.cost μ (Mw w) ((adv o x w, (x w).1)) + G.cost μ (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1))) ≤
        G.cost μ (A.work w) (A.workIn o x w) := by
      intro w
      unfold Accel.cost
      rw [← Finset.sum_add_distrib]
      exact Finset.sum_le_sum fun r _ => by
        rw [← mul_add]
        exact mul_le_mul_of_nonneg_left (huse o x w r) (hμ r)
    have hwork : ∑ w, G.cost μ (A.work w) (A.workIn o x w) ≤ ∑ r, μ r * A.use o x r := by
      calc ∑ w, G.cost μ (A.work w) (A.workIn o x w)
          ≤ ∑ w, ∑ r, μ r * (G.use (A.inst w) (o, A.state o x w) r + G.use (A.work w) (A.workIn o x w) r +
              G.use (A.answer w) (A.answerIn o x w) r) :=
            Finset.sum_le_sum fun w _ => Finset.sum_le_sum fun r _ =>
              mul_le_mul_of_nonneg_left (by
                linarith [((hG2.2 (A.inst w) (o, A.state o x w)).2 r).1,
                  ((hG2.2 (A.answer w) (A.answerIn o x w)).2 r).1]) (hμ r)
        _ = ∑ r, μ r * A.use o x r := by
            simp only [Prover.use, Finset.mul_sum]
            exact Finset.sum_comm
    have hextk : ∑ w, (G.ext (Mw w) ((adv o x w, (x w).1)) + G.ext (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1)))) ≤
        K := by
      calc ∑ w, (G.ext (Mw w) ((adv o x w, (x w).1)) + G.ext (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1))))
          ≤ ∑ w, (G.ext (A.inst w) (o, A.state o x w) + G.ext (A.work w) (A.workIn o x w) +
              G.ext (A.answer w) (A.answerIn o x w)) :=
            Finset.sum_le_sum fun w _ => by
              linarith [hext o x w, (hG2.2 (A.inst w) (o, A.state o x w)).1,
                (hG2.2 (A.answer w) (A.answerIn o x w)).1]
        _ = A.ext o x := rfl
        _ ≤ K := himp A o x hv
    have hS0 : 0 ≤ ∑ r, μ r * A.use o x r := Finset.sum_nonneg fun r _ => mul_nonneg (hμ r)
      (Finset.sum_nonneg fun w _ => by
        linarith [((hG2.2 (A.inst w) (o, A.state o x w)).2 r).1, ((hG2.2 (A.work w) (A.workIn o x w)).2 r).1,
          ((hG2.2 (A.answer w) (A.answerIn o x w)).2 r).1])
    have hDlow : ((k : ℝ) * (pubmm + pubsha) -
          ∑ w, (G.cost μ (Mw w) ((adv o x w, (x w).1)) + G.cost μ (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1)))) -
          β * ∑ w, (G.ext (Mw w) ((adv o x w, (x w).1)) + G.ext (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1))))) /
          (Hmm + Hsha + Ho) ≤
        ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh o x w := by
      rw [show ((k : ℝ) * (pubmm + pubsha) -
          ∑ w, (G.cost μ (Mw w) ((adv o x w, (x w).1)) + G.cost μ (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1)))) -
          β * ∑ w, (G.ext (Mw w) ((adv o x w, (x w).1)) + G.ext (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1))))) =
          ∑ w : Fin k, ((pubmm + pubsha) -
            (G.cost μ (Mw w) ((adv o x w, (x w).1)) + G.cost μ (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1)))) -
            β * (G.ext (Mw w) ((adv o x w, (x w).1)) + G.ext (Mh w) (o, G.Mc.eval (Mw w) ((adv o x w, (x w).1))))) by
        simp only [Finset.sum_sub_distrib, ← Finset.mul_sum, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
          nsmul_eq_mul], Finset.sum_div]
      exact Finset.sum_le_sum fun w _ => roundD_ge hHmm hHsha hHpos
    exact floor_of_luck hHmm hHsha hHo hη hβ hS0 hD hDlow ((Finset.sum_le_sum fun w _ => hcost w).trans hwork)
      hextk
  refine (prob_mono (q := fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
      ((verdict A ox.1 ox.2 ∧ ¬ A.Accepted ox.1 ox.2) ∨ (A.Accepted ox.1 ox.2 ∧ ∃ w, ¬ roundOk A adv Mw ox.1 ox.2 w)) ∨
        ((∀ w, roundOk A adv Mw ox.1 ox.2 w) ∧ Real.log (1 / δ) / η ≤
          ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh ox.1 ox.2 w))
      fun ox hox => ?_).trans ?_
  · obtain ⟨hv, z, hfit, hnot⟩ := hox
    by_cases hacc : A.Accepted ox.1 ox.2
    · by_cases hok : ∀ w, roundOk A adv Mw ox.1 ox.2 w
      · by_cases hD : Real.log (1 / δ) / η ≤
            ∑ w, roundD adv μ (pubmm / Hmm) (pubsha / Hsha) β Hmm Hsha (Hmm + Hsha + Ho) Mw Mh ox.1 ox.2 w
        · exact Or.inr ⟨hok, hD⟩
        · exact absurd (FreedOfFloor G Pd k _ (A.use ox.1 ox.2) μ z z' hμ hneeds hval (Nat.cast_nonneg k) hε0 hfit'
            hslack hfit (key ox.1 ox.2 hv hok (not_le.mp hD))) hnot
      · simp only [not_forall] at hok
        exact Or.inl (Or.inr ⟨hacc, hok⟩)
    · exact Or.inl (Or.inl ⟨hv, hacc⟩)
  · have hx' : prob (fun ox : P.Orc × (Fin k → P.Draw × P.Chk) =>
        A.Accepted ox.1 ox.2 ∧ ∃ w, ¬ roundOk A adv Mw ox.1 ox.2 w) ≤ κx := hx
    refine ((prob_or_le _ _).trans (add_le_add (prob_or_le _ _) E3)).trans ?_
    linarith [hserv A, hx']

end Pouw.SecurityProofs.Dither
