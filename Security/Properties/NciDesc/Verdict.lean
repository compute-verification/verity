import Properties.NciDesc
import Properties.Nci.Threads

/-!
# What a verdict of the description-sized check carries

`NciCheck.Desc.evaluate` is the policy check `nci-check` runs (`NciCheck.Desc.check` writes its refusals, and the check
of record, `NciCheck.Baseline.evaluate`, returns its verdict once θ's provenance is checked). `DescAgrees`
(`Properties.NciDesc`) makes its verdict the one `NciCheck.evaluate` gives on every expansion of the instance gate by gate
(`Expands`), of which there is one. The statements here restate the verdict's properties (`Properties.Nci`,
`Properties.Nci.Threads`) of `Desc.evaluate`: on every instance with a verdict, the instance has an expansion, and on each
the verdict has the property (`OnExpansions`).

* `DescReports`: the verdict is the definition's (`Reported`).
* `DescWitnesses`: a verdict of compliance witnesses the policy, on the registered compiler's inference blocks `d.𝓘`
  when the owners are they (`Witnessed`).
* `DescThrottles`, `DescThrottlesDark`, `DescThrottlesSlots`, `DescThrottlesDarkSlots`: a verdict of compliance carries
  the thread bounds (`Throttled`, `ThrottledDark` and their `Slots` forms), under their named assumptions
  (`WeightsIncompressible` or `WeightsIncompressibleSlots`, and `ThetaRegistered`), which the check does not see.

Each is proved from `DescAgrees` (`SecurityProofs.DescAgrees`, `Proofs.Properties.Nci.Agrees`): `DescReports` and
`DescWitnesses` in `Proofs.Properties.Nci.DescVerdict`, and the four bounds in `Proofs.Nci.DescVerdict` (package
`security_proofs`), whose proofs need the thread bounds'.
-/

namespace NciCheck.Properties

open Lean (Json)

/-- `Q` holds of the verdict `out` on every expansion of `j` gate by gate (`Expands`), of which there is one. -/
def OnExpansions (Q : Json → Json → Prop) (j out : Json) : Prop :=
  (∃ j', Expands j j') ∧ ∀ j', Expands j j' → Q j' out

/-- **The description-sized check's verdict is the definition's.** On every instance with a verdict of
`NciCheck.Desc.evaluate`, the verdict is `Reported` on every expansion. -/
def DescReports : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions Reported j out

/-- **A verdict of compliance of the description-sized check witnesses the policy.** On every instance with a verdict of
`NciCheck.Desc.evaluate`, the verdict is `Witnessed` on every expansion: the registered circuit complies, and when the
owners are its inference blocks `d.𝓘`, the registered compiler's, each meets both bounds. -/
def DescWitnesses : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions Witnessed j out

/-- **A verdict of compliance of the description-sized check throttles every threaded reference circuit the declaration
contains, if its weights are incompressible and its θ registered** (`WeightsIncompressible`,
`ThetaRegistered`). On every instance with a verdict of `NciCheck.Desc.evaluate`, the verdict is `Throttled` on every
expansion. -/
def DescThrottles : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions Throttled j out

/-- **A verdict of compliance of the description-sized check puts counted multiplications off the books in dark
gates**, for every threaded reference circuit the declaration contains partly, if its weights are incompressible and its
θ registered (`WeightsIncompressible`, `ThetaRegistered`). On every instance with a verdict of
`NciCheck.Desc.evaluate`, the verdict is `ThrottledDark` on every expansion. -/
def DescThrottlesDark : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions ThrottledDark j out

/-- **`DescThrottles` per parameter slot** (`WeightsIncompressibleSlots`, `ThetaRegistered`): on every instance with
a verdict of `NciCheck.Desc.evaluate`, the verdict is `ThrottledSlots` on every expansion. -/
def DescThrottlesSlots : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions ThrottledSlots j out

/-- **`DescThrottlesDark` per parameter slot** (`WeightsIncompressibleSlots`, `ThetaRegistered`): on every instance
with a verdict of `NciCheck.Desc.evaluate`, the verdict is `ThrottledDarkSlots` on every expansion. -/
def DescThrottlesDarkSlots : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions ThrottledDarkSlots j out

end NciCheck.Properties
