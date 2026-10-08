/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space and communication. Its components are integrity by sampled proofs' one-stage audit, work by proof of
useful work, space by proof of useful space, and communication by the network certifier; each is a stub
(`Core.Composition.Stub`) until its port lands. `Core.Composition.product` composes them: every component accepts or
onsite doesn't, and any component's rejection rejects onsite whatever the others answer.
-/
import CoreComposition.Cases
import CoreComposition.Stubs

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List Protocol := [Stub.certifier, Stub.pous, Stub.pouw, Stub.sampledProofs]

def protocol : Protocol := product "verity/onsite/v0" components

open Cases in
private def every (f : String → Stubbed) : List Stubbed := components.map (f ·.id)

open Cases in
private def answer (xs : List Stubbed) (ps : List Stubbed := xs) : Answer := Cases.answer protocol xs ps

#guard ordered components
#guard components.map (·.id) ==
  ["verity/stub/certifier/v0", "verity/stub/pous/v0", "verity/stub/pouw/v0", "verity/stub/sampled-proofs/v0"]
#guard (answer (every Cases.accepts)).verdict == .accept
#guard (answer (every Cases.accepts)).published.map (·.compress) == some ("{\"verity/stub/certifier/v0\":{\"passed\":2}," ++
  "\"verity/stub/pous/v0\":{\"passed\":2},\"verity/stub/pouw/v0\":{\"passed\":2}," ++
  "\"verity/stub/sampled-proofs/v0\":{\"passed\":2}}")
-- integrity undecided and space rejected: onsite rejects, and makes nothing public
#guard (answer [Cases.accepts Stub.certifier.id, Cases.rejects Stub.pous.id, Cases.accepts Stub.pouw.id,
  Cases.undecided Stub.sampledProofs.id]).verdict == .reject
#guard (answer [Cases.accepts Stub.certifier.id, Cases.rejects Stub.pous.id, Cases.accepts Stub.pouw.id,
  Cases.undecided Stub.sampledProofs.id]).published.isNone
#guard (answer [Cases.undecided Stub.certifier.id, Cases.accepts Stub.pous.id, Cases.accepts Stub.pouw.id,
  Cases.accepts Stub.sampledProofs.id]).verdict == .undecided
-- remote's components alone are not onsite's
#guard (answer [Cases.accepts Stub.pouw.id, Cases.accepts Stub.sampledProofs.id]).reason ==
  some "params: verity/stub/certifier/v0 is missing"

end Core.Onsite
