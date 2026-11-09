module

public import Proofs.Properties.InferenceOnly.BodiesOk
public import Proofs.Properties.InferenceOnly.BodiesOk.ReadsDecoded
public import Proofs.Properties.InferenceOnly.Novel
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

/-!
# The description-sized check agrees with the check gate by gate

`descAgrees_of_open` proves `DescAgrees` from three statements: `Resolve.bodiesOk` (`Proofs.Properties.InferenceOnly.BodiesOk`)
proves `Resolve.BodiesOk`, `operandsResolve_of` (`Proofs.Properties.InferenceOnly.OperandsResolve`) proves `Walk.OperandsResolve`
from it, and `readsExact` proves `Novel.ReadsExact` (`Desc.analyze`'s reads are the unrolled operands outside each root
node) from `BodiesOk.rootSet` (`readsExact_of_set`, `Proofs.Properties.InferenceOnly.BodiesOk.ReadsDecoded`).
`descAgrees_of_readsExact` proves `DescAgrees` from `ReadsExact`, and `SecurityProofs.DescAgrees` is the guarantee.
-/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs.Novel

/-- **`DescAgrees`, from `ReadsExact`.** -/
theorem descAgrees_of_readsExact (hR : ReadsExact) : Properties.DescAgrees :=
  Novel.descAgrees_of_open (OperandsResolve.operandsResolve_of Resolve.bodiesOk) Resolve.bodiesOk hR

/-- **`analyze`'s reads are the unrolled operands outside each root node.** -/
theorem readsExact : ReadsExact :=
  BodiesOk.readsExact_of_set BodiesOk.rootSet

end InferenceOnlyCheck.Properties.SecurityProofs.Novel

namespace InferenceOnlyCheck.Properties.SecurityProofs

/-- **The check at the size of the description agrees with the check gate by gate.** -/
theorem DescAgrees : Properties.DescAgrees :=
  Novel.descAgrees_of_readsExact Novel.readsExact

end InferenceOnlyCheck.Properties.SecurityProofs
