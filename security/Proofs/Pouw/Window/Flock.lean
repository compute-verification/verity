module

public import Definitions.Pouw.Guarantees.Window
public import Proofs.Flock.Soundness.Audit.Extraction
public import Proofs.Flock.Soundness.Discharge.ZkLink.SessionZ

/-! The window's calls from Flock's analysis: a Flock `ExtractionAnalysis` of a call's session is the call's
`CallAnalysis`, and at `--zk` sessions Flock's knowledge bound holds with `κ = 0`. -/

@[expose] public section

namespace Pouw.Window

open scoped ENNReal

section Call

variable {Reg Tr : Type} {C : ℕ} {n : Fin C → ℕ}

/-- `proj` reads a window transcript as Flock's transcript of call `c`. The window's `ks`, `link` and `cover` are
Flock's, which read no sampling design `LF`. -/
def CallAnalysis.ofFlock (U : UnitStmts Tr n) (c : Fin C)
    {fsess : Finset (Fin (n c)) → Reg → Game Bool} {LF : FlockSoundness.Audit.SamplingDesign (n c)}
    (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (proj : Tr → AF.Tr)
    (hwrong : ∀ Xw u, (⟨c, u⟩ : WUnit n) ∈ U.incorrect Xw → u ∈ AF.incorrect (proj Xw)) :
    CallAnalysis U c fsess where
  ks := AF.ks
  link S R τ u Xw := AF.link S R τ u (proj Xw)
  cover S R τ u Xw hS hX := AF.cover S R τ u (proj Xw) hS (hwrong Xw u hX)

/-- At `--zk` sessions (`analysisZC`) it holds with `κ = 0` and `g = ⨆_j ofReal (ksBoundAccZ_j / (1 − rateZ_j))` at
`zcStrat τ` (`flockKnowledgeSound_analysisZC`). -/
def FlockKnowledgeSound {m : ℕ} {fsess : Finset (Fin m) → Reg → Game Bool}
    {LF : FlockSoundness.Audit.SamplingDesign m} (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (κ : ℝ≥0∞)
    (g : (S : Finset (Fin m)) → (R : Reg) → Strategy (fsess S R) → ℝ≥0∞) : Prop :=
  ∀ S R τ u, u ∈ S → AF.ks S R τ u ≤ κ + g S R τ

theorem CallAnalysis.ofFlock_knowledgeSound (U : UnitStmts Tr n) (c : Fin C)
    {fsess : Finset (Fin (n c)) → Reg → Game Bool} {LF : FlockSoundness.Audit.SamplingDesign (n c)}
    (AF : FlockSoundness.Audit.ExtractionAnalysis LF Reg fsess) (proj : Tr → AF.Tr)
    (hwrong : ∀ Xw u, (⟨c, u⟩ : WUnit n) ∈ U.incorrect Xw → u ∈ AF.incorrect (proj Xw)) {κ : ℝ≥0∞}
    {g : (S : Finset (Fin (n c))) → (R : Reg) → Strategy (fsess S R) → ℝ≥0∞}
    (h : FlockKnowledgeSound AF κ g) : (CallAnalysis.ofFlock U c AF proj hwrong).KnowledgeSound κ g :=
  h

end Call

end Pouw.Window

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting Audit ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

/-- Holds for `k ≥ 1` accepted reruns, with the state term at `zcStrat τ`: a sampled proof block's `ks` is that term at
its own table, conditional on everything before its session and never over the sample's coin or another session's. -/
theorem _root_.Pouw.Window.flockKnowledgeSound_analysisZC (A : Arith F K) (hA : A.Correct) (Hs : Hashes F D Sl)
    {C : Verity.Boolean.Circuit} {n : ℕ} (P : Partition C n) {Reg : Type} (planZ : Finset (Fin n) → Reg → List (TabZK F mPts))
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planZ S R).length) {L : SamplingDesign n}
    (decode : DecoderZK (C := C) planZ tab)
    (Xc : (R : Reg) → Cont L Reg (batchedSessionZC A Hs planZ) R → (Fin C.N → Bool)) {k : ℕ} (hk : 1 ≤ k)
    (hr : ∀ S R j, rateZ (planZ S R) j k < 1) (hlow : LoweringSoundZK A P planZ tab decode) :
    Pouw.Window.FlockKnowledgeSound
      (analysisZC A Hs P planZ tab decode Xc k hr hlow (fun S R τ j => ksBoundAccZ A Hs (planZ S R) j k τ)
        (jointZC_ksBound A hA Hs planZ hk)) 0
      (fun S R τ => ⨆ j : Fin (planZ S R).length,
        ENNReal.ofReal (ksBoundAccZ A Hs (planZ S R) j k (zcStrat A Hs planZ τ) / (1 - rateZ (planZ S R) j k))) := by
  intro S R τ u _
  rw [zero_add]
  exact le_iSup (fun j : Fin (planZ S R).length =>
    ENNReal.ofReal (ksBoundAccZ A Hs (planZ S R) j k (zcStrat A Hs planZ τ) / (1 - rateZ (planZ S R) j k)))
    (tab S R u)

end FlockSoundness.Discharge.ZkLink
