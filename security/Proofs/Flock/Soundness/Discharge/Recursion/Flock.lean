module

public import Proofs.Flock.Soundness.Assumptions.CustodyZk
public import Proofs.Flock.Soundness.Discharge.Composed.Sound
import Proofs.Flock.Soundness.Discharge.Exec.SamplingDesignSample
import Proofs.Flock.Soundness.Discharge.Hidden.Facts
public import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
public import Proofs.Flock.Soundness.Discharge.ZkReg.CheckedHJ
public import Proofs.Flock.Soundness.Discharge.ZkReg.DefsHJ
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

/-! # What V*'s bridge obligation reads of `Recursive.Flock`, without its recursion imports. -/

@[expose] public section

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg

open ZkHidden (planZHJ tabZHJ planZHJ_length SampleSetupZHJ TagsOkZHJ SampleOkZKHJ layShHJ)

open FlockSoundness.Discharge.Tables (LiveOutJ liveSession)

open scoped ENNReal

/-- **The sampling design of a public verifier's units**: the full sample, the live subset sampling design of all `nv` units over `[0, nv)` on a
byte source (`live`, `--draw subset:<units>`, as V* is staged), and the model's uniform `nv`-subset it escapes no more
often than (`model`, which misses no unit). -/
structure VSamplingDesign (nv : ℕ) : Type 1 where
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256

attribute [instance] VSamplingDesign.fintypeΩ VSamplingDesign.nonemptyΩ

/-- **A public verifier**: its circuit `c` over `nv` units, proved in sessions of `J` tables, and its sampling design. -/
structure VStmt : Type 1 where
  nv : ℕ
  mPts : ℕ
  J : ℕ
  c : Flock.Circuit
  hU : Built c.unit
  hscope : Layout.scopeOk c = true
  samplingDesign : VSamplingDesign nv

end FlockSoundness.Discharge.Recursion
