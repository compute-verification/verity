module

public import Proofs.Flock.Soundness.Discharge.Exec.SessionBatchL
public import Proofs.Flock.Soundness.Audit.FlockBatched
import Mathlib.Combinatorics.Matroid.Init
import Proofs.Flock.Soundness.Audit.Circuit
import Proofs.Flock.Soundness.Game

/-! The Flock audit at the compiled layer with several tables per session (`batchedSessionL`), its knowledge term discharged with no named assumption and `ε_bind` named (`BindSound`). -/

@[expose] public section

namespace FlockSoundness.Discharge.Exec

open Game Model Accounting Audit Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128


variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} {D Sl : Type} [Inhabited D] [Inhabited Sl]
  (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H) {mPts : ℕ}
  (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
  (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length)

/-- The strategy in the protocol's session inside a verdict strategy. -/
noncomputable def batchStratL {S : Finset (Fin n)} {R : Reg} (τ : Strategy (batchedSessionL H E Lf plan S R)) :
    Strategy (sessionBL execArith H E Lf (plan S R)) :=
  Strategy.ofMap _ _ τ

theorem prob_batchedSessionL (S : Finset (Fin n)) (R : Reg)
    (τ : Strategy (batchedSessionL H E Lf plan S R)) :
    prob (· = true) (batchedSessionL H E Lf plan S R) τ =
      prob (acceptedBL (plan S R)) (sessionBL execArith H E Lf (plan S R)) (batchStratL H E Lf plan τ) := by
  classical
  refine (prob_map _ _ _ _).trans ?_
  congr 1
  funext outs
  exact propext decide_eq_true_iff

variable {L : SamplingDesign n}

/-- **The batched audit's analysis.** `X` is the committed transcript (the link theorem defines it),
`decode` reads each sampled proof block from its table's witness, `Kr` is the extractor's rerun count. -/
noncomputable def analysisBL (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) (Kr : ℕ)
    (hlow : P.LoweringSoundB plan tab decode) : ExtractionAnalysis L Reg (batchedSessionL H E Lf plan) where
  Tr := Fin C.N → Bool
  incorrect := P.incorrect
  committed := Xc
  ks S R τ u := prob (acceptedBL (plan S R)) (sessionBL execArith H E Lf (plan S R)) (batchStratL H E Lf plan τ) *
    ENNReal.ofReal (failProbBL execArith H E Lf (plan S R) (tab S R u) Kr (batchStratL H E Lf plan τ))
  link S R τ u Xt := prob (acceptedBL (plan S R)) (sessionBL execArith H E Lf (plan S R)) (batchStratL H E Lf plan τ) *
    ENNReal.ofReal (extractProbBL execArith H E Lf (plan S R) (tab S R u) Kr (batchStratL H E Lf plan τ)
      (P.LinkEvB plan tab decode S R u Xt))
  cover S R τ u Xt hS hu := by
    have hG : ∀ T, Committed execArith (Partition.tabOf plan tab S R u).S (Partition.tabOf plan tab S R u).sch T →
        P.LinkEvB plan tab decode S R u Xt T := fun T hc =>
      ⟨hc, fun M _ hsat => by
        by_contra hno
        push Not at hno
        exact P.mem_incorrect.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) (hlow S R u hS M hsat))⟩
    rw [prob_batchedSessionL, ← mul_add]
    exact le_mul_of_one_le_right zero_le
      (one_le_fail_add_BL' execArith H E Lf (plan S R) (tab S R u) Kr (batchStratL H E Lf plan τ) _ hG)

/-- **The batched audit's knowledge term**, per prover: after each sample, the largest over the
plan's tables of the table's bound in the session, averaged over the sample. -/
noncomputable def ksAvgBL (Kr : ℕ) (R : Reg) (τ : Cont L Reg (batchedSessionL H E Lf plan) R) : ℝ≥0∞ :=
  avg fun ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksBoundBL execArith H E Lf (plan (L.draw ω) R) j Kr (batchStratL H E Lf plan (τ ω)))

theorem flock_batched_knowledgeSoundL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) {Kr : ℕ}
    (hK : 1 ≤ Kr) (hlow : P.LoweringSoundB plan tab decode) :
    (analysisBL P H E Lf plan tab decode Xc Kr hlow).KnowledgeSound (ksAvgBL H E Lf plan Kr) := by
  intro R τ tgt
  refine avg_mono fun ω => atTgt_le (fun u _ => ?_) (tgt ω)
  exact (joint_le_BL' execArith execArith_correct H E Lf (plan (L.draw ω) R) (tab (L.draw ω) R u)
    Kr hK (batchStratL H E Lf plan (τ ω))).trans (le_iSup (fun j : Fin (plan (L.draw ω) R).length =>
      ENNReal.ofReal (ksBoundBL execArith H E Lf (plan (L.draw ω) R) j Kr (batchStratL H E Lf plan (τ ω))))
      (tab (L.draw ω) R u))

theorem flock_batched_countL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) {Kr : ℕ}
    (hK : 1 ≤ Kr) (hlow : P.LoweringSoundB plan tab decode)
    {εbind : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → ℝ≥0∞}
    (hlink : (analysisBL P H E Lf plan tab decode Xc Kr hlow).BindSound εbind) (K : ℕ)
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan))) :
    prob (fun o => o.1 = true ∧ K ≤ (P.incorrect (Xc (reg σ) (cont σ))).card)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      L.miss K + ksAvgBL H E Lf plan Kr (reg σ) (cont σ) + εbind (reg σ) (cont σ) :=
  extraction_audit_count (analysisBL P H E Lf plan tab decode Xc Kr hlow)
    (flock_batched_knowledgeSoundL P H E Lf plan tab decode Xc hK hlow) hlink K σ

theorem flock_batched_sampledL
    (decode : DecoderB (C := C) (Reg := Reg) plan tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → (Fin C.N → Bool)) {Kr : ℕ}
    (hK : 1 ≤ Kr) (hlow : P.LoweringSoundB plan tab decode)
    {εbind : (R : Reg) → Cont L Reg (batchedSessionL H E Lf plan) R → ℝ≥0∞}
    (hlink : (analysisBL P H E Lf plan tab decode Xc Kr hlow).BindSound εbind)
    (σ : Strategy (audit L Reg (batchedSessionL H E Lf plan))) :
    prob (fun o => o.1 = true ∧ ¬ Disjoint (P.incorrect (Xc (reg σ) (cont σ))) o.2)
        (audit L Reg (batchedSessionL H E Lf plan)) σ ≤
      ksAvgBL H E Lf plan Kr (reg σ) (cont σ) + εbind (reg σ) (cont σ) :=
  extraction_audit_sampled (analysisBL P H E Lf plan tab decode Xc Kr hlow)
    (flock_batched_knowledgeSoundL P H E Lf plan tab decode Xc hK hlow) hlink σ

end FlockSoundness.Discharge.Exec
