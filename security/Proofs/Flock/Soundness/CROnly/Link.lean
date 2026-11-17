module

public import Proofs.Flock.Soundness.CROnly.Defs
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.CROnly.Truncate
import Proofs.Flock.Soundness.Game

/-! The link finders under `cr/sha-512`, at the per-run cost `strictRunCost t' q` in place of `t'`. -/

@[expose] public section

namespace FlockSoundness.CROnly

open Game Model Accounting Extract LinkFinder Audit Audit.Partition
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

section Cost

variable {τ Val : Type} [Fintype Val] [DecidableEq Val] (rd : τ → Finset Val)

omit [Fintype Val] [DecidableEq Val] in
theorem loopCost_mul (a : ℝ) (c : τ → ℝ) :
    ∀ l : List τ, loopCost rd (fun e => a * c e) l = a * loopCost rd c l
  | [] => by simp [loopCost]
  | e :: es => by
      simp only [loopCost]
      rw [loopCost_mul a c es]
      split_ifs <;> ring

omit [Fintype Val] [DecidableEq Val] in
theorem loopCost_nonneg {c : τ → ℝ} (hc : ∀ e, 0 ≤ c e) : ∀ l : List τ, 0 ≤ loopCost rd c l
  | [] => by simp [loopCost]
  | e :: es => by
      simp only [loopCost]
      have := loopCost_nonneg hc es
      split_ifs
      · linarith [hc e]
      · linarith [hc e]

omit [Fintype Val] [DecidableEq Val] in
theorem finderCost_mul (a : ℝ) (c : τ → ℝ) (x : τ × List τ) :
    finderCost rd (fun e => a * c e) x = a * finderCost rd c x := by
  unfold finderCost
  rw [loopCost_mul]
  split_ifs <;> ring

end Cost

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D : Type} [Inhabited D]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)
  {L : SamplingDesign n} {decode : DecoderB (C := C) (Reg := Reg) plan tab} {Dh : Type}
  {Hc : List UInt8 → Dh} (vb : P.ValueBinding plan tab decode Hc)
  (R : Reg) (τ : Cont L Reg (batchedSession H E plan) R)

theorem cT_mul (a t' : ℝ) (e : TrialOut L D) : cT plan R (a * t') e = a * cT plan R t' e := by
  unfold cT
  ring

theorem finderCost_cT_ge (c : vb.Pos) {t' : ℝ} (ht : 0 ≤ t') (x : TrialOut L D × List (TrialOut L D)) :
    t' ≤ finderCost (P.rdT H E plan tab vb R τ c) (cT plan R t') x := by
  classical
  have h1 : t' ≤ cT plan R t' x.1 := by
    unfold cT
    have : 0 ≤ if accω plan R x.1.1 x.1.2.1 then (x.1.2.2.map (waitCost (accω plan R x.1.1))).sum else 0 := by
      split_ifs
      · exact List.sum_nonneg (by
          simp only [List.mem_map]
          rintro _ ⟨b, -, rfl⟩
          exact waitCost_nonneg _ b)
      · exact le_rfl
    nlinarith
  have h2 : 0 ≤ if (P.rdT H E plan tab vb R τ c x.1).card = 1 then
      loopCost (P.rdT H E plan tab vb R τ c) (cT plan R t') x.2 else 0 := by
    split_ifs
    · exact loopCost_nonneg _ (cT_nonneg plan R ht) _
    · exact le_rfl
  unfold finderCost
  linarith

theorem linkCR_of_strict {k Rw M : ℕ} {t' q : ℝ} (ht : 0 < t') (hq : 0 < q)
    (hS : LinkCRStrict P H E plan tab vb R τ k Rw M t' q) :
    P.LinkCR H E plan tab vb R τ k Rw M (strictRunCost t' q) := by
  intro c hne
  unfold Assumptions.SHA512CRExpected
  set rd := P.rdT H E plan tab vb R τ c
  set g := finderG (P.trialG H E plan tab vb R τ c k Rw) M
  set s := finderStrat _ (P.trialStrat H E plan tab vb R τ c k Rw) M
  have hc0 : ∀ x, 0 ≤ finderCost rd (cT plan R t') x := fun x =>
    ht.le.trans (finderCost_cT_ge P H E plan tab vb R τ c ht.le x)
  have h := expected_of_strict Hc g s (P.outC H E plan tab vb R τ c) (finderCost rd (cT plan R t')) hc0 hq (hS c hne)
  set a := strictRunCost t' q / t' with ha
  have hcost : finderCost rd (cT plan R (strictRunCost t' q)) = fun x => a * finderCost rd (cT plan R t') x := by
    have hcT : (cT plan R (strictRunCost t' q) : TrialOut L D → ℝ) = fun e => a * cT plan R t' e := by
      funext e
      rw [← cT_mul plan R, ha, div_mul_cancel₀ _ ht.ne']
    funext x
    rw [hcT, finderCost_mul]
  rw [hcost, expect_const_mul]
  set T := expect (finderCost rd (cT plan R t')) g s
  have hT : t' ≤ T := by
    have := expect_mono (finderCost_cT_ge P H E plan tab vb R τ c ht.le) g s
    rwa [expect_const] at this
  have h256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  have h513 : (0 : ℝ) < 2 ^ (513 : ℝ) := by positivity
  have e1 : a * T / 2 ^ (256 : ℝ) = T / q + q ^ 2 / 2 ^ (513 : ℝ) * (T / t') := by
    rw [ha]
    unfold strictRunCost
    field_simp
  have e2 : 1 ≤ T / t' := (one_le_div ht).2 hT
  have e3 : q ^ 2 / 2 ^ (513 : ℝ) ≤ q ^ 2 / 2 ^ (513 : ℝ) * (T / t') :=
    le_mul_of_one_le_right (by positivity) e2
  rw [e1]
  linarith

theorem strictRunCost_pos {t' q : ℝ} (ht : 0 < t') (hq : 0 < q) : 0 < strictRunCost t' q := by
  unfold strictRunCost
  positivity

theorem strictRunCost_at_cube {t' q : ℝ} (k : ℕ) (hq : 0 < q) (hcube : q ^ 3 = t' * 2 ^ (512 : ℝ)) :
    2 * (strictRunCost t' q * (1 + k)) / 2 ^ (256 : ℝ) = 3 * (1 + k) * (q ^ 2 / 2 ^ (512 : ℝ)) := by
  have h := trunc_at_cube hq hcube
  have h256 : (0 : ℝ) < 2 ^ (256 : ℝ) := by positivity
  unfold strictRunCost
  rw [h]
  field_simp

end FlockSoundness.CROnly
