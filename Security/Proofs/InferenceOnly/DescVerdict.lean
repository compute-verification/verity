module

public import Properties.InferenceOnly.Desc.Verdict
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
import Proofs.InferenceOnly.Verdict
import Proofs.Properties.InferenceOnly.Agrees
import Proofs.Properties.InferenceOnly.DescVerdict

/-!
# A verdict of compliance of the description-sized check carries the thread bound

Each thread bound of `Properties.InferenceOnly.Desc.Verdict`, from `DescAgrees` (`SecurityProofs.DescAgrees`,
`Proofs.Properties.InferenceOnly.Agrees`) and the bound of a verdict of `InferenceOnlyCheck.evaluate` (`Throttles` and its forms,
`Proofs.InferenceOnly.Verdict`), through `Proofs.Properties.InferenceOnly.DescVerdict`.
-/

@[expose] public section

namespace InferenceOnlyCheck.Properties.SecurityProofs

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

end InferenceOnlyCheck.Properties.SecurityProofs
