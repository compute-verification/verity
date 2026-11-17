module

public import Proofs.Properties.InferenceOnly.BodiesOk
public import Proofs.Properties.InferenceOnly.BodiesOk.ReadsDecoded
public import Proofs.Properties.InferenceOnly.Fresh
public import Proofs.Properties.InferenceOnly.OperandsResolve
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
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

/-! `DescAgrees` from `Resolve.BodiesOk`, `Walk.OperandsResolve` and `Fresh.ReadsExact`. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs.Fresh

theorem descAgrees_of_readsExact (hR : ReadsExact) : Properties.DescAgrees :=
  Fresh.descAgrees_of_open (OperandsResolve.operandsResolve_of Resolve.bodiesOk) Resolve.bodiesOk hR

theorem readsExact : ReadsExact :=
  BodiesOk.readsExact_of_set BodiesOk.rootSet

end InferenceOnlyCheck.Properties.SecurityProofs.Fresh

namespace InferenceOnlyCheck.Properties.SecurityProofs

/-- **The check at the size of the description agrees with the check gate by gate.** -/
theorem DescAgrees : Properties.DescAgrees :=
  Fresh.descAgrees_of_readsExact Fresh.readsExact

end InferenceOnlyCheck.Properties.SecurityProofs
