module

public import Properties.InferenceOnly.Desc

/-!
# What a verdict of the description-sized check carries

`InferenceOnlyCheck.Desc.evaluate` is the policy check `inference-only-check` runs (`InferenceOnlyCheck.Desc.check` writes its refusals, and the check
of record, `InferenceOnlyCheck.Theta.evaluate`, returns its verdict once θ's provenance is checked). `DescAgrees`
(`Properties.InferenceOnly.Desc`) makes its verdict the one `InferenceOnlyCheck.evaluate` gives on every expansion of the instance gate by gate
(`Expands`), of which there is one. The statements here restate the verdict's properties (`Properties.InferenceOnly`) of
`Desc.evaluate`: on every instance with a verdict, the instance has an expansion, and on each
the verdict has the property (`OnExpansions`).

* `DescReports`: the verdict is the definition's (`Reported`).
* `DescWitnesses`: a verdict of compliance witnesses the policy, on the registered compiler's inference blocks `d.𝓘`
  when the owners are they (`Witnessed`).

Each is proved from `DescAgrees` (`SecurityProofs.DescAgrees`, `Proofs.Properties.InferenceOnly.Agrees`), in
`Proofs.Properties.InferenceOnly.DescVerdict`.
-/

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

end InferenceOnlyCheck.Properties
