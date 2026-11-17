module

public import Proofs.Flock.Recursive.Raw
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

/-! The registrant's openings: the recursive audit's premise on the provers it covers. -/

@[expose] public section

namespace Flock.Assumptions

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed

/-- Premise (`registrant-holds-openings`) on the provers the guarantee covers: a prover that registers a hash it
can't open is outside the guarantee. -/
def RegistrantCovers {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : SamplingDesign nv} (x : ZkOuterR mPts nv J Reg' c hU hscope Lv) : Prop :=
  ∀ ht S₀ d₀ y₀ hc hpub (u : Fin nv) (q : Fin c.ports.size), (∃ pt ∈ x.own, pt.name = c.ports[q].name) →
    (x.rd ht S₀ d₀ y₀ hc hpub).isRead (.inl (u, q))

end Flock.Assumptions
