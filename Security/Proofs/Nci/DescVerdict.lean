import Proofs.Nci.Verdict
import Proofs.Properties.Nci.DescVerdict

/-!
# A verdict of compliance of the description-sized check carries the thread bound

Each thread bound of `Properties.NciDesc.Verdict`, from `DescAgrees` (`SecurityProofs.DescAgrees`,
`Proofs.Properties.Nci.Agrees`) and the bound of a verdict of `NciCheck.evaluate` (`Throttles` and its forms,
`Proofs.Nci.Verdict`), through `Proofs.Properties.Nci.DescVerdict`.
-/

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
