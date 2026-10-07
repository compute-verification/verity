import Properties.Composition
import CoreRemote

/-!
# Remote Verity: what its Lean code (`CoreRemote`) does

Remote Verity (`verity/experimental/core/protocols/remote/`, `verity/remote/v0`) is the composition of its two
components (`Components`), so `Properties.Composition`'s rule holds of it at its id and components: any component's
rejection rejects it, it accepts exactly when every component accepts, and nothing is public unless it accepts. Each
component is a stub until its port lands, and the properties other than `Components` hold whatever the components
become. `Properties.Proofs.Remote` proves them all, with no assumption.
-/

namespace Properties.Remote

open Core Core.Composition Properties.Composition

/-- Remote Verity is the composition `verity/remote/v0` of these components, in the order of their ids: proof of useful
work and sampled proofs' one-stage audit (integrity), each a stub until its port lands. -/
def Components : Prop :=
  Core.Remote.protocol = product "verity/remote/v0" Core.Remote.components ∧
    Core.Remote.components.map (·.id) = ["verity/stub/pouw/v0", "verity/stub/sampled-proofs/v0"]

/-- Any component's rejection rejects remote Verity, whatever the others answer. -/
def RejectDominates : Prop := RejectDominatesAt "verity/remote/v0" Core.Remote.components

/-- Cases that don't fit remote's components reject it. -/
def MisfitRejects : Prop := MisfitRejectsAt "verity/remote/v0" Core.Remote.components

/-- Remote Verity accepts exactly when its cases fit, every component accepts, and the commitment binds their digests. -/
def AcceptsIff : Prop := AcceptsIffAt "verity/remote/v0" Core.Remote.components

/-- Remote Verity is undecided exactly when no component rejects and some is undecided (its cases fitting, the
commitment binding their digests). -/
def UndecidedIff : Prop := UndecidedIffAt "verity/remote/v0" Core.Remote.components

/-- On acceptance remote Verity makes public each component's public outputs and keeps each one's profile, keyed by id. -/
def PublishesEach : Prop := PublishesEachAt "verity/remote/v0" Core.Remote.components

/-- Nothing of remote Verity's answer is public unless it accepts. -/
def NothingPublicUnlessAccepted : Prop :=
  ∀ params commitment evidence : ByteArray,
    (Core.Remote.protocol.answer params commitment evidence).verdict ≠ .accept →
      (Core.Remote.protocol.answer params commitment evidence).published = none ∧
        (Core.Remote.protocol.answer params commitment evidence).profile = none

end Properties.Remote
