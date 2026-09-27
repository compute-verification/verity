import FlockSoundness.ExtractAcc
import FlockSoundness.Game.Seq

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

end Extract

end FlockSoundness
