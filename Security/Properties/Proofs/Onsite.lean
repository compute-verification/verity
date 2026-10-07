import Properties.Onsite
import Properties.Proofs.Composition

/-!
# Proofs of onsite Verity's properties (`Properties.Onsite`)

`Components` by definition, and the rest as the composition's (`Properties.Proofs.Composition`) at onsite's id and
components.
-/

namespace Properties.Onsite

theorem components : Components := ⟨rfl, rfl⟩

theorem rejectDominates : RejectDominates := Composition.rejectDominates _ _

theorem misfitRejects : MisfitRejects := Composition.misfitRejects _ _

theorem acceptsIff : AcceptsIff := Composition.acceptsIff _ _

theorem undecidedIff : UndecidedIff := Composition.undecidedIff _ _

theorem publishesEach : PublishesEach := Composition.publishesEach _ _

theorem nothingPublicUnlessAccepted : NothingPublicUnlessAccepted :=
  Composition.nothingPublicUnlessAccepted Core.Onsite.protocol

end Properties.Onsite
