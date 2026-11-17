module

public import Properties.InferenceOnly.Desc
public import Properties.InferenceOnly.Threads

/-! What a verdict of the description-sized check carries. -/

@[expose] public section

namespace InferenceOnlyCheck.Properties

open Lean (Json)

/-- `Q` holds of the verdict `out` on every expansion of `j` gate by gate (`Expands`), of which there is one. -/
def OnExpansions (Q : Json → Json → Prop) (j out : Json) : Prop :=
  (∃ j', Expands j j') ∧ ∀ j', Expands j j' → Q j' out

/-- **The description-sized check's verdict is the definition's.** On every instance with a verdict of
`InferenceOnlyCheck.Desc.evaluate`, the verdict is `Reported` on every expansion. -/
def DescReports : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions Reported j out

/-- **A verdict of compliance of the description-sized check witnesses the policy.** On every instance with a verdict of
`InferenceOnlyCheck.Desc.evaluate`, the verdict is `Witnessed` on every expansion: the registered circuit complies, and when the
owners are its inference blocks `d.𝓘`, the registered compiler's, each meets both bounds. -/
def DescWitnesses : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions Witnessed j out

/-- **A verdict of compliance of the description-sized check throttles every threaded reference training circuit the
declaration contains, if its weights are incompressible and its θ registered** (`WeightsIncompressible`,
`ThetaRegistered`). On every instance with a verdict of `InferenceOnlyCheck.Desc.evaluate`, the verdict is `Throttled` on every
expansion. -/
def DescThrottles : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions Throttled j out

/-- **A verdict of compliance of the description-sized check puts counted multiplications off the books in dark
gates**, for every threaded reference training circuit the declaration contains partly, if its weights are
incompressible and its θ registered (`WeightsIncompressible`, `ThetaRegistered`). On every instance with a verdict of
`InferenceOnlyCheck.Desc.evaluate`, the verdict is `ThrottledDark` on every expansion. -/
def DescThrottlesDark : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions ThrottledDark j out

/-- **`DescThrottles` per parameter slot** (`WeightsIncompressibleSlots`, `ThetaRegistered`): on every instance with
a verdict of `InferenceOnlyCheck.Desc.evaluate`, the verdict is `ThrottledSlots` on every expansion. -/
def DescThrottlesSlots : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions ThrottledSlots j out

/-- **`DescThrottlesDark` per parameter slot** (`WeightsIncompressibleSlots`, `ThetaRegistered`): on every instance
with a verdict of `InferenceOnlyCheck.Desc.evaluate`, the verdict is `ThrottledDarkSlots` on every expansion. -/
def DescThrottlesDarkSlots : Prop :=
  ∀ (j out : Json), Desc.evaluate j = .ok out → OnExpansions ThrottledDarkSlots j out

end InferenceOnlyCheck.Properties
