import Proofs.Nci.Verdict
import Proofs.Properties.Nci.DescVerdict

/-!
# A verdict of compliance of the description-sized check carries the thread bound, from `DescAgrees`

Each thread bound of `Properties.NciDesc.Verdict`, from `DescAgrees` and the bound of a verdict of `NciCheck.evaluate`
(`Throttles` and its forms, `Proofs.Nci.Verdict`), through `Proofs.Properties.Nci.DescVerdict`. They take `DescAgrees` as
a hypothesis while its proof rests on one open statement, `Novel.ReadsExact` (`descAgrees_of_readsExact`,
`Proofs.Properties.Nci.Agrees`).
-/

namespace NciCheck.Properties.SecurityProofs

theorem descThrottles_of_descAgrees (h : Properties.DescAgrees) : Properties.DescThrottles :=
  descThrottles_of h Throttles

theorem descThrottlesDark_of_descAgrees (h : Properties.DescAgrees) : Properties.DescThrottlesDark :=
  descThrottlesDark_of h ThrottlesDark

theorem descThrottlesSlots_of_descAgrees (h : Properties.DescAgrees) : Properties.DescThrottlesSlots :=
  descThrottlesSlots_of h ThrottlesSlots

theorem descThrottlesDarkSlots_of_descAgrees (h : Properties.DescAgrees) : Properties.DescThrottlesDarkSlots :=
  descThrottlesDarkSlots_of h ThrottlesDarkSlots

end NciCheck.Properties.SecurityProofs
