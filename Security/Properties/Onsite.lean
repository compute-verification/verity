import Properties.Composition
import CoreOnsite

/-!
# Onsite Verity: what its Lean code (`CoreOnsite`) does

Onsite Verity (`verity/experimental/core/protocols/onsite/`, `verity/onsite/v0`) is the composition of its four
components (`Components`), so `Properties.Composition`'s rule holds of it at its id and components: any component's
rejection rejects it, it accepts exactly when every component accepts, and nothing is public unless it accepts. Each
component is a stub until its port lands, and the properties other than `Components` hold whatever the components
become. `Properties.Proofs.Onsite` proves them all, with no assumption.
-/

namespace Properties.Onsite

open Core Core.Composition Properties.Composition

/-- Onsite Verity is the composition `verity/onsite/v0` of these components, in the order of their ids: the network
certifier (communication), proof of useful space, proof of useful work, and sampled proofs' one-stage audit (integrity),
each a stub until its port lands. -/
def Components : Prop :=
  Core.Onsite.protocol = product "verity/onsite/v0" Core.Onsite.components ∧
    Core.Onsite.components.map (·.id) =
      ["verity/stub/certifier/v0", "verity/stub/pous/v0", "verity/stub/pouw/v0", "verity/stub/sampled-proofs/v0"]

/-- Any component's rejection rejects onsite Verity, whatever the others answer. -/
def RejectDominates : Prop := RejectDominatesAt "verity/onsite/v0" Core.Onsite.components

/-- Cases that don't fit onsite's components reject it. -/
def MisfitRejects : Prop := MisfitRejectsAt "verity/onsite/v0" Core.Onsite.components

/-- Onsite Verity accepts exactly when its cases fit, every component accepts, and the commitment binds their digests. -/
def AcceptsIff : Prop := AcceptsIffAt "verity/onsite/v0" Core.Onsite.components

/-- Onsite Verity is undecided exactly when no component rejects and some is undecided (its cases fitting, the
commitment binding their digests). -/
def UndecidedIff : Prop := UndecidedIffAt "verity/onsite/v0" Core.Onsite.components

/-- On acceptance onsite Verity makes public each component's public outputs and keeps each one's profile, keyed by id. -/
def PublishesEach : Prop := PublishesEachAt "verity/onsite/v0" Core.Onsite.components

/-- Nothing of onsite Verity's answer is public unless it accepts. -/
def NothingPublicUnlessAccepted : Prop :=
  ∀ params commitment evidence : ByteArray,
    (Core.Onsite.protocol.answer params commitment evidence).verdict ≠ .accept →
      (Core.Onsite.protocol.answer params commitment evidence).published = none ∧
        (Core.Onsite.protocol.answer params commitment evidence).profile = none

end Properties.Onsite
