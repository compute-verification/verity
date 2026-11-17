module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
public import Proofs.Flock.Soundness.Discharge.Recursion.Flock
public import Proofs.Flock.Soundness.Discharge.Recursion.Off
public import Proofs.Flock.Claims.Steps.VStar
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
public import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
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
import Mathlib.Tactic.ReduceModChar

/-! # What V*'s bridge obligation reads of `Recursive.Raw`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

open ZkHidden (planZHJ planZHJ_length SampleSetupZHJ TagsOkZHJ SampleOkZKHJ layShHJ)

open FlockSoundness.Discharge.Tables (LiveOutJ liveSession liveRoundsJ)

open scoped ENNReal

set_option genInjectivity false in
/-- **V*'s `--zk --registered` session at one history, as the prover supplies it**: the verifier's inputs `I`, its own
ports `own`, the sample file `dj` and the record writer `wr` of the live session of `J` tables, with `ZkOuterD`'s
`hDraw`, `hTags` and `hJ` (the verifier's own table count is `J`). The registrant's record of the registered reads
`rd`, and `hown`, are `ZkOuterD`'s at each typing of the session (`ht`, a setup `y₀` at table 0, the parse `hc` and the
load `hpub`), which they read. -/
structure ZkOuterR (mPts nv J : ℕ) (Reg' : Type) (c : Flock.Circuit) (hU : Built c.unit)
    (hscope : Layout.scopeOk c = true) (Lv : SamplingDesign nv) where
  I : Inputs
  own : Array Flock.Registered.Port
  dj : Finset (Fin nv) → Lean.Json
  wr : Finset (Fin nv) → Reg' → LiveOutJ J mPts → Exec.Transcript
  hDraw : FlockSoundness.Assumptions.Zk.DrawCustodyZKHJ I dj mPts Lv
  hTags : TagsOkZHJ I
  hJ : nTab I = J
  rd : ∀ (ht : I.tags.typed = false) (S₀ : Finset (Fin nv)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJ (inputsJ I 0) mPts S₀ d₀) (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
    (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok (pubZ I)),
    FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg') dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU (pub := pubZ I))
      (rsShHJS dj y₀ ht hc (Layout.scopeOkH_of_scopeOk hscope) hU hpub)
  hown : ∀ ht S₀ d₀ y₀ hc hpub ω R' o, LiveAcceptsZKR I own (R', Lv.draw ω, wr (Lv.draw ω) R' o) →
    OwnReadsHJ I own c (pubZ I) (wr (Lv.draw ω) R' o) ((rd ht S₀ d₀ y₀ hc hpub).restrict (rowIn (Lv.draw ω))).isRead
      (rd ht S₀ d₀ y₀ hc hpub).port (rd ht S₀ d₀ y₀ hc hpub).readAt

namespace ZkOuterR

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : SamplingDesign nv} (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) (dg : ByteArray → ByteArray) (N : ℕ)

/-- The strategy a name stands for in V*'s session. -/
def ofBase (τ₀ : Strategy (baseGame Lv Reg' dg J mPts N)) : Strategy (x.game dg N) :=
  ⟨τ₀.1, fun ω => Strategy.map (fun t => (τ₀.1, Lv.draw ω, t)) _ (Strategy.map (x.wr (Lv.draw ω) τ₀.1) _ (τ₀.2 ω))⟩

end ZkOuterR

end FlockSoundness.Discharge.Recursion
