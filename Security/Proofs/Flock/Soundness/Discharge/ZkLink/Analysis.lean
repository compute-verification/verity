import Proofs.Flock.Soundness.Discharge.ZkLink.Compiled

/-!
# The compiled `--zk` audit's analysis, with the expected-time extractor

`analysisBEL` (`Discharge/Exec/FlockBatchedAccL.lean`) for the compiled `--zk` session (`batchedSessionZC`): each drawn
unit's table is extracted from `k` reruns of everything after `Commit` in which the session accepts (`extractTableZ`),
and the analysis carries
$$\varepsilon\le\frac{a+\varepsilon\Pr[\text{link}]}{1-r}$$
as its `cover`, from the joint inequality `ε·Pr[fails] ≤ a + r·ε` (`JointZC`) at a knowledge bound `a`, with `ε` the
session's acceptance probability and `r = N₀/(ek) < 1`:

* `ks`: `a/(1 − r)` for the unit's table;
* `link`: `ε·Pr[the extracted oracle has the link event]/(1 − r)`, with `LinkEvZK` on the oracle extracted from
  accepted reruns.

`JointZC` at a closed-form `a` from the caps (the compiled `ε_ks`) is open: it is the `--zk` counterpart of
`joint_le_accBL`. The link term does not depend on `a`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal NNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

section Session

variable (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))

omit [Fintype K] in
theorem accProbZC_nonneg (τ : Strategy (sessionZC A Hs ts)) : 0 ≤ accProbZC A Hs ts τ :=
  expect_nonneg (ind_nonneg _) _ _

omit [Fintype K] in
theorem extractProbAccZ_nonneg (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts)) (G : Oracle F → Prop) :
    0 ≤ extractProbAccZ A Hs ts j k τ G :=
  Extract.expectAcc_nonneg _ _ _ _ (ind_nonneg _)

omit [Fintype K] in
open Classical in
/-- Over accepted reruns, table `j`'s extraction fails or has any event its success implies. -/
theorem one_le_fail_add_accZ (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionZC A Hs ts)) (G : Oracle F → Prop)
    (hG : ∀ T, (ts.get j).Committed A T → G T) (hε : 0 < accProbZC A Hs ts τ) :
    1 ≤ failProbAccZ A Hs ts j k τ + extractProbAccZ A Hs ts j k τ G := by
  unfold failProbAccZ extractProbAccZ
  rw [← Extract.expectAcc_add]
  calc (1 : ℝ) = Extract.expectAcc (sessAfterZC (gamesZC A Hs ts τ.1)) τ.2 (acceptedZC ts) k (fun _ => 1) :=
        (Extract.expectAcc_const _ _ _ _ 1 hε).symm
    _ ≤ _ := Extract.expectAcc_mono _ _ _ _ fun bs => by
        have h0 := ind_nonneg (fun bs => ¬ (ts.get j).Committed A (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs
        have h1 := ind_nonneg (fun bs => G (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs
        by_cases hc : (ts.get j).Committed A (extractTableZ Hs (ts.get j) j (τ.1 j) bs)
        · have hg : ind (fun bs => G (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs = 1 := by
            simp only [ind, hG _ hc, ↓reduceIte]
          show (1 : ℝ) ≤ ind (fun bs => ¬ (ts.get j).Committed A (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs +
            ind (fun bs => G (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs
          linarith
        · have hf : ind (fun bs => ¬ (ts.get j).Committed A (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs = 1 := by
            simp only [ind, hc, not_false_eq_true, ↓reduceIte]
          show (1 : ℝ) ≤ ind (fun bs => ¬ (ts.get j).Committed A (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs +
            ind (fun bs => G (extractTableZ Hs (ts.get j) j (τ.1 j) bs)) bs
          linarith

end Session

section Audit

variable (A : Arith F K) (Hs : Hashes F D Sl) {C : Circuit} {n : ℕ} (P : Partition C n) {Reg : Type}
  (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length)

omit [Fintype K] in
theorem prob_batchedSessionZC (S : Finset (Fin n)) (R : Reg) (τ : Strategy (batchedSessionZC A Hs planZ S R)) :
    prob (· = true) (batchedSessionZC A Hs planZ S R) τ =
      prob (acceptedZC (planZ S R)) (sessionZC A Hs (planZ S R)) (zcStrat A Hs planZ τ) := by
  classical
  refine (prob_map _ _ _ _).trans ?_
  congr 1
  funext outs
  exact propext decide_eq_true_iff

omit [Fintype K] in
theorem prob_batchedSessionZC_eq (S : Finset (Fin n)) (R : Reg) (τ : Strategy (batchedSessionZC A Hs planZ S R)) :
    prob (· = true) (batchedSessionZC A Hs planZ S R) τ =
      ENNReal.ofReal (accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ)) := by
  rw [prob_batchedSessionZC, ← ENNReal.ofReal_toReal
    (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)), prob_toReal]
  rfl

/-- **The compiled joint inequality at the knowledge bound `a`** (`joint_le_accBL`'s form): for every plan's table `j`
and every strategy in the compiled session after the draw, `a ≥ 0` and
`Pr[the session accepts]·Pr[table j's extraction from k accepted reruns fails] ≤ a + N₀/(ek)·Pr[the session accepts]`. -/
def JointZC (k : ℕ) (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) →
    Fin (planZ S R).length → ℝ) : Prop :=
  ∀ S R (τ : Strategy (sessionZC A Hs (planZ S R))) (j : Fin (planZ S R).length), 0 ≤ a S R τ j ∧
    accProbZC A Hs (planZ S R) τ * failProbAccZ A Hs (planZ S R) j k τ ≤
      a S R τ j + rateZ (planZ S R) j k * accProbZC A Hs (planZ S R) τ

variable {L : Law n}

/-- **The compiled link term**: `ε·Pr[the oracle extracted from k accepted reruns has the link event]/(1 − N₀/(ek))`
at drawn unit `u`. -/
noncomputable def linkTermZC (decode : DecoderZK (C := C) planZ tab) (k : ℕ) (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSessionZC A Hs planZ S R)) (u : Fin n) (Xt : Fin C.N → Bool) : ℝ≥0∞ :=
  ENNReal.ofReal (accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ) *
    extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ) (LinkEvZK A P planZ tab decode S R u Xt) /
    (1 - rateZ (planZ S R) (tab S R u) k))

/-- **The compiled `--zk` audit's analysis, with the expected-time extractor**, at a knowledge bound `a` with its joint
inequality (`hj`). `Xc` is the committed transcript (the link theorem defines it), `decode` reads each drawn unit from
its table's padded witness, and `k` is the number of accepted reruns, with `N₀/(ek) < 1` for every table of every plan. -/
noncomputable def analysisZC (decode : DecoderZK (C := C) planZ tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) (k : ℕ)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (hj : JointZC A Hs planZ k a) :
    ExtractionAnalysis L Reg (batchedSessionZC A Hs planZ) where
  Tr := Fin C.N → Bool
  wrong := P.wrong
  committed := Xc
  ks S R τ u := ENNReal.ofReal (a S R (zcStrat A Hs planZ τ) (tab S R u) / (1 - rateZ (planZ S R) (tab S R u) k))
  link := linkTermZC A Hs P planZ tab decode k
  cover S R τ u Xt hS hu := by
    have hG : ∀ T, (tabOfZ planZ tab S R u).Committed A T → LinkEvZK A P planZ tab decode S R u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        push Not at hno
        exact P.mem_wrong.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R u hS M hsat))⟩
    set ε := accProbZC A Hs (planZ S R) (zcStrat A Hs planZ τ)
    set r := rateZ (planZ S R) (tab S R u) k
    set a' := a S R (zcStrat A Hs planZ τ) (tab S R u)
    set Lp := extractProbAccZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ)
      (LinkEvZK A P planZ tab decode S R u Xt)
    have hr1 : 0 < 1 - r := by linarith [hr S R (tab S R u)]
    have ha : 0 ≤ a' := (hj S R _ (tab S R u)).1
    have hL : 0 ≤ Lp := extractProbAccZ_nonneg _ _ _ _ _ _ _
    have hε0 : 0 ≤ ε := accProbZC_nonneg _ _ _ _
    have key : ε ≤ a' / (1 - r) + ε * Lp / (1 - r) := by
      rcases hε0.eq_or_lt with h0 | hpos
      · rw [← h0]
        positivity
      · have h1 := one_le_fail_add_accZ A Hs (planZ S R) (tab S R u) k (zcStrat A Hs planZ τ) _ hG hpos
        have h2 := (hj S R (zcStrat A Hs planZ τ) (tab S R u)).2
        rw [← add_div, le_div_iff₀ hr1]
        nlinarith
    show prob (· = true) (batchedSessionZC A Hs planZ S R) τ ≤
      ENNReal.ofReal (a' / (1 - r)) + ENNReal.ofReal (ε * Lp / (1 - r))
    rw [prob_batchedSessionZC_eq, ← ENNReal.ofReal_add (div_nonneg ha hr1.le)
      (div_nonneg (mul_nonneg hε0 hL) hr1.le)]
    exact ENNReal.ofReal_le_ofReal key

/-- **The compiled knowledge term**, per prover: after each draw, the largest over the plan's tables of
`a/(1 − N₀/(ek))`, averaged over the draw. -/
noncomputable def ksAvgZC (k : ℕ)
    (a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ)
    (R : Reg) (τ : Cont L Reg (batchedSessionZC A Hs planZ) R) : ℝ≥0∞ :=
  avg fun ω => ⨆ j : Fin (planZ (L.draw ω) R).length,
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k))

omit [Fintype K] in
/-- **The compiled analysis's knowledge term is its own bound**, averaged over the draw. -/
theorem zc_knowledgeSound (decode : DecoderZK (C := C) planZ tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ}
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode)
    {a : (S : Finset (Fin n)) → (R : Reg) → Strategy (sessionZC A Hs (planZ S R)) → Fin (planZ S R).length → ℝ}
    (hj : JointZC A Hs planZ k a) :
    (analysisZC A Hs P planZ tab decode Xc k hr hlow a hj).KnowledgeSound (ksAvgZC A Hs planZ k a) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact le_iSup (fun j : Fin (planZ (L.draw ω) R).length =>
    ENNReal.ofReal (a (L.draw ω) R (zcStrat A Hs planZ (τ ω)) j / (1 - rateZ (planZ (L.draw ω) R) j k)))
    (tab (L.draw ω) R u)

end Audit

end FlockSoundness.Discharge.ZkLink
