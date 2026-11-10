module

public import Properties.Nci.Desc.Verdict
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
import Proofs.Nci.Verdict
import Proofs.Properties.Nci.Agrees
import Proofs.Properties.Nci.DescVerdict

/-!
# A verdict of compliance of the description-sized check carries the thread bound

Each thread bound of `Properties.Nci.Desc.Verdict`, from `DescAgrees` (`SecurityProofs.DescAgrees`,
`Proofs.Properties.Nci.Agrees`) and the bound of a verdict of `NciCheck.evaluate` (`Throttles` and its forms,
`Proofs.Nci.Verdict`), through `Proofs.Properties.Nci.DescVerdict`.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs

/-- **A verdict of compliance of the description-sized check throttles every threaded reference circuit the declaration
contains, if its weights are incompressible and its θ registered** (`WeightsIncompressible`,
`ThetaRegistered`). -/
theorem DescThrottles : Properties.DescThrottles :=
  descThrottles_of DescAgrees Throttles

/-- **A verdict of compliance of the description-sized check puts counted multiplications off the books in dark
gates**, for every threaded reference circuit the declaration contains partly, if its weights are incompressible and its
θ registered (`WeightsIncompressible`, `ThetaRegistered`). -/
theorem DescThrottlesDark : Properties.DescThrottlesDark :=
  descThrottlesDark_of DescAgrees ThrottlesDark

/-- **`DescThrottles` per parameter slot** (`WeightsIncompressibleSlots`, `ThetaRegistered`). -/
theorem DescThrottlesSlots : Properties.DescThrottlesSlots :=
  descThrottlesSlots_of DescAgrees ThrottlesSlots

/-- **`DescThrottlesDark` per parameter slot** (`WeightsIncompressibleSlots`, `ThetaRegistered`). -/
theorem DescThrottlesDarkSlots : Properties.DescThrottlesDarkSlots :=
  descThrottlesDarkSlots_of DescAgrees ThrottlesDarkSlots

end NciCheck.Properties.SecurityProofs
