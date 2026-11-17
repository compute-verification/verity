module

public import Proofs.Flock.Soundness.Discharge.Composed.Defs
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsLive
public import Proofs.Flock.Soundness.Discharge.ZkHidden.DefsJ
import Proofs.Flock.Soundness.Discharge.ZkLink.Inh
public import Proofs.Flock.Soundness.Discharge.ZkReg.Bind
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
public import Proofs.Flock.Soundness.Discharge.ZkReg.Defs
import Proofs.Flock.Soundness.Game
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

/-! # What V*'s bridge obligation reads of `Recursive.ZkReg.DrawnRefines`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR DecodeZL

open Composed

open scoped ENNReal

section Hidden

open ZkHidden (planZHJ SampleSetupZHJ TagsOkZHJ SampleOkZKHJ opnAtZHJ runsZCJH_live)

open FlockSoundness.Discharge.Tables (LiveOutJ)

variable {mPts n : ℕ} {Reg : Type}

/-- **The simulation of a hidden-output `--zk` session of `J` tables**: `simZJ` at the tables `planZHJ`, with the decoder
`runsZCJH_live` gives. -/
noncomputable abbrev simZHJ {I : Inputs} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} {d₀ : Lean.Json}
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (ht : I.tags.typed = false) {c : Flock.Circuit}
    (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c) (hscope : Layout.scopeOkH c = true)
    (hU : Built c.unit)
    (wr : (S : Finset (Fin n)) → (R : Reg) → LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ) {L : SamplingDesign n} (hTags : TagsOkZHJ I)
    (hRec : RecordsLiveZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr)
    (hDraw : ∀ ω, SampleOkZKHJ I dj mPts (L.draw ω))
    (τ : Strategy (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr))) :
    Strategy (audit L Reg (batchedSessionZC execArith hs512 (planZHJ dj y₀))) :=
  simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
    (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
      (runsZCJH_live I ht hc (Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr L hTags hRec hDraw)) τ

end Hidden

end FlockSoundness.Discharge.ZkReg
