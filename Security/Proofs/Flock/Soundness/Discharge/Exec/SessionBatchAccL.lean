import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchL
import Proofs.Flock.Soundness.Discharge.Exec.SessionAccL
import Proofs.Flock.Soundness.SessionBatchAcc

/-!
# Several salted tables in one batched session, from accepted reruns

`SessionBatchAcc.lean` over a leaf scheme `Lf`: the extractor of table `j` fills it from `k` reruns after `Commit` in
which the salted session accepts (`extractProbAccBL`, `failProbAccBL`), and
`joint_le_accBL`: `Pr[the session accepts] · failProbAccBL ≤ ksBoundAccBL + rateB · Pr[accepts]`, with `ksBoundAccBL`
table `j`'s `ε_c⁻ + √(k·adv₀)` (`session_knowledge_sound_accL`); `one_le_fail_add_accBL` is the audit's `cover`.
-/

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D Sl : Type} {mPts : ℕ}

open Classical in
/-- **An event of table `j`'s extraction, from accepted reruns**: the probability, over `k` reruns
of everything after `Commit` in which the session accepts, that the table the extractor fills from
table `j`'s openings satisfies `G`. -/
noncomputable def extractProbAccBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop) : ℝ :=
  Extract.expectAcc (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 (acceptedBL ts) k
    (ind fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs))

/-- **Table `j`'s extraction from accepted reruns fails**: no codeword near the extracted table
packs a satisfying witness. -/
noncomputable def failProbAccBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) : ℝ :=
  extractProbAccBL A H E Lf ts j k τ fun T => ¬ Committed A (ts.get j).S (ts.get j).sch T

/-- **Table `j`'s knowledge bound from accepted reruns**, without its relative term:
`ε_c⁻ + √(k·adv₀L)`, every term the table's own, computed in the session. -/
noncomputable def ksBoundAccBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) : ℝ :=
  epsSL A H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest
      (ts.get j).ext (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm
      (ts.get j).ptLocal mPts ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1))
      (τ.1 j) (stratBL A H E Lf ts j τ) +
    Real.sqrt (k * adv₀SL A H E Lf (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesBL A H E Lf ts τ.1).take j) ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j)
      (stratBL A H E Lf ts j τ))

omit [Fintype K] in
theorem extractProbAccB_nonnegL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop) :
    0 ≤ extractProbAccBL A H E Lf ts j k τ G :=
  Extract.expectAcc_nonneg _ _ _ _ (ind_nonneg _)

theorem ksBoundAccB_nonnegL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) : 0 ≤ ksBoundAccBL A H E Lf ts j k τ :=
  add_nonneg (epsS_nonnegL _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) (Real.sqrt_nonneg _)

open Classical in
/-- **Knowledge soundness of table `j` in a batched session, from accepted reruns**, in the product
form: `Pr[the session accepts] · Pr[table j's extraction fails] ≤ ε_c⁻ + √(k·adv₀L) + N₀/(ek) ·
Pr[the session accepts]`, over `k ≥ 1` reruns in which the session accepts. -/
theorem joint_le_accBL (A : Arith F K) (hA : A.Correct)
    [Inhabited D] [Inhabited Sl] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (k : ℕ) (hk : 1 ≤ k) (τ : Strategy (sessionBL A H E Lf ts)) :
    expect (ind (acceptedBL ts)) (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 *
        failProbAccBL A H E Lf ts j k τ ≤
      ksBoundAccBL A H E Lf ts j k τ +
        rateB ts j k * expect (ind (acceptedBL ts)) (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 := by
  have hAcc : ∀ outs : List (OutCL F D Sl), acceptedBL ts outs →
      (outs.getD ((gamesBL A H E Lf ts τ.1).take j).length dfltOutL).accepted = true := by
    intro outs h
    rw [take_gamesB_lengthL]
    exact h j
  have h := session_knowledge_sound_accL A hA H E Lf (ts.get j).S (ts.get j).m (ts.get j).k0
    (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext (ts.get j).hsch (ts.get j).hm
    (ts.get j).ptLocal mPts (ts.get j).hlay ((gamesBL A H E Lf ts τ.1).take j)
    ((gamesBL A H E Lf ts τ.1).drop (j + 1)) (τ.1 j) (acceptedBL ts) hAcc (stratBL A H E Lf ts j τ) k hk
  rw [take_gamesB_lengthL] at h
  unfold stratBL at h
  rw [expect_cast' (sessAfterB_gamesBL A H E Lf ts τ.1 j) τ.2,
    expectAcc_cast (sessAfterB_gamesBL A H E Lf ts τ.1 j) τ.2] at h
  calc expect (ind (acceptedBL ts)) (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 *
        failProbAccBL A H E Lf ts j k τ
      ≤ _ := h
    _ = _ := by unfold ksBoundAccBL rateB stratBL; ring

omit [Fintype K] in
open Classical in
/-- **The audit's `cover`, per table, from accepted reruns.** If commitment implies `G` and the
session accepts with positive probability, every extracted table is uncommitted or satisfies `G`,
so table `j`'s failure and `G` have probabilities adding up to at least `1`. -/
theorem one_le_fail_add_accBL [Inhabited D] [Inhabited Sl] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (Lf : Merkle.Leaf (List F) Sl D H) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionBL A H E Lf ts)) (G : Oracle F → Prop)
    (hG : ∀ T, Committed A (ts.get j).S (ts.get j).sch T → G T)
    (hε : 0 < expect (ind (acceptedBL ts)) (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2) :
    1 ≤ failProbAccBL A H E Lf ts j k τ + extractProbAccBL A H E Lf ts j k τ G := by
  unfold failProbAccBL extractProbAccBL
  rw [← Extract.expectAcc_add]
  calc (1 : ℝ) = Extract.expectAcc (sessAfterBL (gamesBL A H E Lf ts τ.1)) τ.2 (acceptedBL ts) k
        (fun _ => 1) := (Extract.expectAcc_const _ _ _ _ 1 hε).symm
    _ ≤ _ := Extract.expectAcc_mono _ _ _ _ fun bs => by
        have h0 := ind_nonneg (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
          (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs
        have h1 := ind_nonneg (fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs
        by_cases hc : Committed A (ts.get j).S (ts.get j).sch
            (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)
        · have hg : ind (fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs = 1 := by
            simp only [ind, hG _ hc, ↓reduceIte]
          show (1 : ℝ) ≤ ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
            (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs +
              ind (fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs
          linarith
        · have hf : ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
              (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs = 1 := by
            simp only [ind, hc, not_false_eq_true, ↓reduceIte]
          show (1 : ℝ) ≤ ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
            (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs +
              ind (fun bs => G (extractTableSL H E Lf (ts.get j).sch j (τ.1 j) bs)) bs
          linarith

end FlockSoundness.Discharge.Exec
