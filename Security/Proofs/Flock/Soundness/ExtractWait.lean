import Proofs.Flock.Soundness.ExtractAcc
import Proofs.Flock.Soundness.Game.Seq

/-!
# The expected-time extractor as an algorithm: waits of at most `R` runs

The extractor's law `Extract.expectAcc` is `k` independent accepted runs. An algorithm gets them by
waiting: it runs the session until a run accepts, at most `R` times, and does that `k` times. This
file is that algorithm's law and cost, the form a collision finder runs.

* `expectN_find`: over `R` runs, the first accepted one is found with probability
  `ε·geom ε R = 1 − (1 − ε)^R` (`mul_geom`), distributed as an accepted run;
* `expectN_firsts`: `k` waits that all find one give `k` independent accepted runs, weighted by
  `geom ε R ^ k`, so their law is `(1 − (1 − ε)^R)^k` times `expectAcc`;
* `expectN_waitCost`: a wait plays `geom ε R` runs in expectation, so after an accepted run the `k`
  waits play `k·geom ε R` runs, and `ε·k·geom ε R = k·(1 − (1 − ε)^R) ≤ k`, whatever `ε`.
-/

namespace FlockSoundness

namespace Extract

open Game

variable {α : Type} (g : Game α) (s : Strategy g) (acc : α → Prop)

/-- `Σ_{i<R} (1 − ε)^i`, the expected length of a wait of at most `R` runs. -/
noncomputable def geom (ε : ℝ) (R : ℕ) : ℝ := ∑ i ∈ Finset.range R, (1 - ε) ^ i

theorem geom_succ (ε : ℝ) (R : ℕ) : geom ε (R + 1) = 1 + (1 - ε) * geom ε R := by
  unfold geom
  rw [Finset.sum_range_succ', pow_zero, add_comm, Finset.mul_sum]
  congr 1
  exact Finset.sum_congr rfl fun i _ => by ring

theorem mul_geom (ε : ℝ) (R : ℕ) : ε * geom ε R = 1 - (1 - ε) ^ R := by
  induction R with
  | zero => simp [geom]
  | succ R ih =>
      rw [geom_succ, mul_add, mul_one, ← mul_assoc, mul_comm ε (1 - ε), mul_assoc, ih]
      ring

theorem geom_nonneg {ε : ℝ} (h1 : ε ≤ 1) (R : ℕ) : 0 ≤ geom ε R :=
  Finset.sum_nonneg fun i _ => pow_nonneg (by linarith) i

/-- **One wait**: over `R` runs, the first accepted one. -/
theorem expectN_find [DecidablePred acc] (F : Option α → ℝ) : ∀ R : ℕ,
    expectN g s R (fun bs => F (bs.find? fun b => decide (acc b))) =
      geom (expect (ind acc) g s) R * expect (fun b => ind acc b * F (some b)) g s +
        (1 - expect (ind acc) g s) ^ R * F none
  | 0 => by simp [expectN, geom]
  | R + 1 => by
      have ih := expectN_find F R
      set ε := expect (ind acc) g s with hε
      set Y := geom ε R * expect (fun b => ind acc b * F (some b)) g s + (1 - ε) ^ R * F none
      have h : ∀ a, expectN g s R (fun as => F ((a :: as).find? fun b => decide (acc b))) =
          ind acc a * F (some a) + Y * (1 - ind acc a) := by
        intro a
        by_cases ha : acc a
        · simp only [List.find?_cons, ha, decide_true, expectN_const]
          simp [ind, ha]
        · simp only [List.find?_cons, ha, decide_false]
          rw [ih]
          simp [ind, ha, Y]
      simp only [expectN]
      rw [show (fun a => expectN g s R fun as => F ((a :: as).find? fun b => decide (acc b))) =
        fun a => ind acc a * F (some a) + Y * (1 - ind acc a) from funext h, expect_add,
        expect_const_mul, expect_sub, expect_const, geom_succ]
      simp only [Y]
      ring

open Classical in
/-- **The accepted runs of `k` waits**, if each found one. -/
noncomputable def firsts : List (List α) → Option (List α)
  | [] => some []
  | b :: bl => (b.find? fun x => decide (acc x)).bind fun x => (firsts bl).map (x :: ·)

open Classical in
/-- **`k` waits of at most `R` runs**: when all find an accepted run, the runs they find are `k`
independent accepted runs, weighted by `geom ε R ^ k`. -/
theorem expectN_firsts (R : ℕ) : ∀ (k : ℕ) (Φ : List α → ℝ),
    expectN (runsG g R) (runsStrat g s R) k (fun bl => (firsts acc bl).elim 0 Φ) =
      geom (expect (ind acc) g s) R ^ k *
        expectN g s k (fun bs => (bs.map (ind acc)).prod * Φ bs)
  | 0, Φ => by simp [expectN, firsts]
  | k + 1, Φ => by
      set ε := expect (ind acc) g s with hε
      set Z : α → ℝ := fun x => geom ε R ^ k *
        expectN g s k (fun bs => (bs.map (ind acc)).prod * Φ (x :: bs))
      have hpt : ∀ b : List α, expectN (runsG g R) (runsStrat g s R) k
          (fun bl => (firsts acc (b :: bl)).elim 0 Φ) =
            (b.find? fun x => decide (acc x)).elim 0 Z := by
        intro b
        cases hb : b.find? fun x => decide (acc x) with
        | none =>
            simp only [firsts, hb, Option.bind_none, Option.elim_none]
            exact expectN_const _ _ 0 k
        | some x =>
            simp only [firsts, hb, Option.bind_some, Option.elim_some, Z]
            rw [← expectN_firsts R k (fun bs => Φ (x :: bs))]
            refine congrArg (expectN (runsG g R) (runsStrat g s R) k) (funext fun bl => ?_)
            cases firsts acc bl <;> rfl
      simp only [expectN]
      rw [show (fun b => expectN (runsG g R) (runsStrat g s R) k
          fun bl => (firsts acc (b :: bl)).elim 0 Φ) =
        fun b => (b.find? fun x => decide (acc x)).elim 0 Z from funext hpt, expect_runsG]
      rw [expectN_find g s acc (fun o => o.elim 0 Z) R]
      simp only [Option.elim_none, mul_zero, add_zero, Option.elim_some, Z]
      rw [show (fun b => ind acc b * (geom ε R ^ k *
          expectN g s k fun bs => (bs.map (ind acc)).prod * Φ (b :: bs))) =
        fun b => geom ε R ^ k * expectN g s k (fun bs => ((b :: bs).map (ind acc)).prod *
          Φ (b :: bs)) from funext fun b => by
          have e : (fun bs : List α => ((b :: bs).map (ind acc)).prod * Φ (b :: bs)) =
              fun bs => ind acc b * ((bs.map (ind acc)).prod * Φ (b :: bs)) :=
            funext fun bs => by simp only [List.map_cons, List.prod_cons]; ring
          rw [e, expectN_const_mul]
          ring,
        expect_const_mul, pow_succ]
      ring

open Classical in
/-- **The runs a wait plays**: up to and including the first accepted one, or all of them. -/
noncomputable def waitCost : List α → ℝ
  | [] => 0
  | b :: bs => 1 + if acc b then 0 else waitCost bs

theorem waitCost_nonneg : ∀ bs : List α, 0 ≤ waitCost acc bs
  | [] => le_rfl
  | b :: bs => by
      classical
      unfold waitCost
      split_ifs
      · norm_num
      · linarith [waitCost_nonneg bs]

theorem expectN_waitCost : ∀ R : ℕ, expectN g s R (waitCost acc) = geom (expect (ind acc) g s) R
  | 0 => by simp [expectN, waitCost, geom]
  | R + 1 => by
      classical
      have ih := expectN_waitCost R
      have h : ∀ a, expectN g s R (fun as => waitCost acc (a :: as)) =
          1 + geom (expect (ind acc) g s) R * (1 - ind acc a) := by
        intro a
        by_cases ha : acc a
        · simp only [waitCost, ha, ↓reduceIte, add_zero, expectN_const]
          simp [ind, ha]
        · simp only [waitCost, ha, ↓reduceIte]
          rw [expectN_add, expectN_const, ih]
          simp [ind, ha]
      simp only [expectN]
      rw [show (fun a => expectN g s R fun as => waitCost acc (a :: as)) =
        fun a => 1 + geom (expect (ind acc) g s) R * (1 - ind acc a) from funext h, expect_add,
        expect_const, expect_const_mul, expect_sub, expect_const, geom_succ]
      ring

theorem expectAcc_sum {ι : Type} (t : Finset ι) (k : ℕ) (Ψ : ι → List α → ℝ) :
    expectAcc g s acc k (fun bs => ∑ i ∈ t, Ψ i bs) = ∑ i ∈ t, expectAcc g s acc k (Ψ i) := by
  unfold expectAcc
  rw [show (fun bs : List α => (bs.map (ind acc)).prod * ∑ i ∈ t, Ψ i bs) =
      fun bs => ∑ i ∈ t, (bs.map (ind acc)).prod * Ψ i bs from funext fun bs => Finset.mul_sum _ _ _,
    expectN_sum, div_eq_mul_inv, Finset.sum_mul]
  simp only [← div_eq_mul_inv]

theorem expectAcc_zero (k : ℕ) : expectAcc g s acc k (fun _ => 0) = 0 := by
  unfold expectAcc
  simp only [mul_zero, expectN_const, zero_div]

/-- **The waits against the ideal law**: when each of the `k` waits gives up after `R` runs, at
most `k/(eR)` of `ε · expectAcc Ψ` is lost, for `0 ≤ Ψ ≤ 1`. -/
theorem expectAcc_le_waits (k R : ℕ) (hR : 1 ≤ R) (Ψ : List α → ℝ) (h0 : ∀ bs, 0 ≤ Ψ bs)
    (h1 : ∀ bs, Ψ bs ≤ 1) :
    expect (ind acc) g s * expectAcc g s acc k Ψ ≤
      expect (ind acc) g s * geom (expect (ind acc) g s) R ^ k *
          expectN g s k (fun bs => (bs.map (ind acc)).prod * Ψ bs) +
        k / (Real.exp 1 * R) := by
  set ε := expect (ind acc) g s with hε
  have hε0 : 0 ≤ ε := expect_nonneg (ind_nonneg acc) g s
  have hε1 : ε ≤ 1 := expect_ind_le_one acc g s
  have hk : 0 ≤ (k : ℝ) / (Real.exp 1 * R) := by positivity
  rcases hε0.eq_or_lt with hz | hpos
  · rw [← hz, zero_mul, zero_mul, zero_mul, zero_add]
    exact hk
  set A := expectAcc g s acc k Ψ with hA
  have hA0 : 0 ≤ A := expectAcc_nonneg g s acc k h0
  have hA1 : A ≤ 1 := by
    have := expectAcc_mono g s acc k h1
    rwa [expectAcc_const g s acc k 1 hpos] at this
  have hN : expectN g s k (fun bs => (bs.map (ind acc)).prod * Ψ bs) = ε ^ k * A := by
    rw [hA]
    unfold expectAcc
    rw [← hε]
    field_simp
  set a := ε * geom ε R with ha
  have ha1 : a = 1 - (1 - ε) ^ R := mul_geom ε R
  have ha0 : 0 ≤ a := by
    rw [ha1]; have := pow_le_one₀ (by linarith : 0 ≤ 1 - ε) (by linarith : 1 - ε ≤ 1) (n := R)
    linarith
  have ha1' : a ≤ 1 := by rw [ha1]; have := pow_nonneg (by linarith : 0 ≤ 1 - ε) R; linarith
  -- Bernoulli: 1 − a^k ≤ k(1 − a)
  have hber : 1 - a ^ k ≤ k * (1 - a) := by
    have := one_add_mul_le_pow (by linarith : (-2 : ℝ) ≤ a - 1) k
    simp only [add_sub_cancel] at this
    linarith
  have hmiss : ε * (1 - ε) ^ R ≤ 1 / (Real.exp 1 * R) := mul_one_sub_pow_le ε hε0 hε1 R hR
  have hgeom : ε * geom ε R ^ k * (ε ^ k * A) = ε * a ^ k * A := by
    rw [ha, mul_pow]; ring
  rw [hN, hgeom]
  have h1a : 1 - a = (1 - ε) ^ R := by rw [ha1]; ring
  have key : ε * A * (1 - a ^ k) ≤ k / (Real.exp 1 * R) := by
    calc ε * A * (1 - a ^ k) ≤ ε * 1 * (k * (1 - a)) := by
          apply mul_le_mul (mul_le_mul_of_nonneg_left hA1 hε0) hber
          · have := pow_le_one₀ ha0 ha1' (n := k); linarith
          · positivity
      _ = k * (ε * (1 - ε) ^ R) := by rw [h1a]; ring
      _ ≤ k * (1 / (Real.exp 1 * R)) := mul_le_mul_of_nonneg_left hmiss (Nat.cast_nonneg k)
      _ = k / (Real.exp 1 * R) := by ring
  nlinarith

end Extract

end FlockSoundness
