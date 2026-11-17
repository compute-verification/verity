module

public import Proofs.Flock.Soundness.ExtractAcc
public import Proofs.Flock.Soundness.SessionAcc
public import Proofs.Flock.Soundness.SessionBatch
import Proofs.Flock.Soundness.Game

/-! `SessionBatch.lean` for the expected-time extractor, from reruns in which the session accepts. -/

@[expose] public section

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type} {mPts : ℕ}

theorem expect_cast' {α : Type} {g₁ g₂ : Game α} (h : g₁ = g₂) (s : Strategy g₁) (f : α → ℝ) :
    expect f g₂ (cast (congrArg Strategy h) s) = expect f g₁ s := by
  subst h; rfl

theorem expectAcc_cast {α : Type} {g₁ g₂ : Game α} (h : g₁ = g₂) (s : Strategy g₁)
    (acc : α → Prop) (k : ℕ) (Φ : List α → ℝ) :
    Extract.expectAcc g₂ (cast (congrArg Strategy h) s) acc k Φ =
      Extract.expectAcc g₁ s acc k Φ := by
  subst h; rfl

open Classical in
/-- Over reruns in which the session accepts, not table `j`: conditioning on table `j`'s own acceptance would make the
relative term relative to that, which can be far larger. -/
noncomputable def extractProbAccB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop) : ℝ :=
  Extract.expectAcc (sessAfterB (gamesB A H E ts τ.1)) τ.2 (acceptedB ts) k
    (ind fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs))

noncomputable def failProbAccB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionB A H E ts)) : ℝ :=
  extractProbAccB A H E ts j k τ fun T => ¬ Committed A (ts.get j).S (ts.get j).sch T

noncomputable def ksBoundAccB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionB A H E ts)) : ℝ :=
  epsS A H E (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest
      (ts.get j).ext (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm
      (ts.get j).ptLocal mPts ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1))
      (τ.1 j) (stratB A H E ts j τ) +
    Real.sqrt (k * adv₀S A H E (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j)
      (stratB A H E ts j τ))

/-- **The rate of table `j`'s relative term**, `N₀/(ek)`. -/
noncomputable def rateB (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) : ℝ :=
  2 ^ (ts.get j).l₀.logLen / (Real.exp 1 * k)

omit [Fintype K] in
theorem extractProbAccB_nonneg [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop) :
    0 ≤ extractProbAccB A H E ts j k τ G :=
  Extract.expectAcc_nonneg _ _ _ _ (ind_nonneg _)

theorem ksBoundAccB_nonneg [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionB A H E ts)) : 0 ≤ ksBoundAccB A H E ts j k τ :=
  add_nonneg (epsS_nonneg _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) (Real.sqrt_nonneg _)

open Classical in
theorem joint_le_accB (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (k : ℕ) (hk : 1 ≤ k) (τ : Strategy (sessionB A H E ts)) :
    expect (ind (acceptedB ts)) (sessAfterB (gamesB A H E ts τ.1)) τ.2 *
        failProbAccB A H E ts j k τ ≤
      ksBoundAccB A H E ts j k τ +
        rateB ts j k * expect (ind (acceptedB ts)) (sessAfterB (gamesB A H E ts τ.1)) τ.2 := by
  have hAcc : ∀ outs : List (OutC F D), acceptedB ts outs →
      (outs.getD ((gamesB A H E ts τ.1).take j).length dfltOut).accepted = true := by
    intro outs h
    rw [take_gamesB_length]
    exact h j
  have h := session_knowledge_sound_acc A hA H E (ts.get j).S (ts.get j).m (ts.get j).k0
    (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext (ts.get j).hsch (ts.get j).hm
    (ts.get j).ptLocal mPts (ts.get j).hlay ((gamesB A H E ts τ.1).take j)
    ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j) (acceptedB ts) hAcc (stratB A H E ts j τ) k hk
  rw [take_gamesB_length] at h
  unfold stratB at h
  rw [expect_cast' (sessAfterB_gamesB A H E ts τ.1 j) τ.2,
    expectAcc_cast (sessAfterB_gamesB A H E ts τ.1 j) τ.2] at h
  calc expect (ind (acceptedB ts)) (sessAfterB (gamesB A H E ts τ.1)) τ.2 *
        failProbAccB A H E ts j k τ
      ≤ _ := h
    _ = _ := by unfold ksBoundAccB rateB stratB; ring

omit [Fintype K] in
open Classical in
theorem one_le_fail_add_accB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop)
    (hG : ∀ T, Committed A (ts.get j).S (ts.get j).sch T → G T)
    (hε : 0 < expect (ind (acceptedB ts)) (sessAfterB (gamesB A H E ts τ.1)) τ.2) :
    1 ≤ failProbAccB A H E ts j k τ + extractProbAccB A H E ts j k τ G := by
  unfold failProbAccB extractProbAccB
  rw [← Extract.expectAcc_add]
  calc (1 : ℝ) = Extract.expectAcc (sessAfterB (gamesB A H E ts τ.1)) τ.2 (acceptedB ts) k
        (fun _ => 1) := (Extract.expectAcc_const _ _ _ _ 1 hε).symm
    _ ≤ _ := Extract.expectAcc_mono _ _ _ _ fun bs => by
        have h0 := ind_nonneg (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
          (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs
        have h1 := ind_nonneg (fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs
        by_cases hc : Committed A (ts.get j).S (ts.get j).sch
            (extractTableS H E (ts.get j).sch j (τ.1 j) bs)
        · have hg : ind (fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs = 1 := by
            simp only [ind, hG _ hc, ↓reduceIte]
          show (1 : ℝ) ≤ ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
            (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs +
              ind (fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs
          linarith
        · have hf : ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
              (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs = 1 := by
            simp only [ind, hc, not_false_eq_true, ↓reduceIte]
          show (1 : ℝ) ≤ ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
            (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs +
              ind (fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs
          linarith

end FlockSoundness
