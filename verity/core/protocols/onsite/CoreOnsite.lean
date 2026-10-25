/-
Onsite Verity (`verity/onsite/v0`): remote Verity's accounting with the verifier's hardware at the site, so it also
accounts for space and communication. Its components are communication by the network certifier over a period
(`NetworkCheck.Period.protocol`, `verity/certifier/v0`), space by proof of useful space, work by proof of useful work,
and integrity by sampled proofs' one-stage audit; the last three are stubs (`Core.Composition.Stub`) until their ports
land. `Core.Composition.product` composes them: every component accepts or onsite doesn't, and any component's rejection
rejects onsite whatever the others answer.
-/
import CoreComposition.Cases
import CoreComposition.Stubs
import NetworkCheck.Period

namespace Core.Onsite

open Composition

/-- Onsite's components, in the order of their ids. -/
def components : List Protocol := [NetworkCheck.Period.protocol, Stub.pous, Stub.pouw, Stub.sampledProofs]

def protocol : Protocol := product "verity/onsite/v0" components

open Cases in
private def every (f : String → Stubbed) : List Stubbed := components.map (f ·.id)

open Cases in
private def answer (xs : List Stubbed) (ps : List Stubbed := xs) : Answer := Cases.answer protocol xs ps

#guard ordered components
#guard components.map (·.id) ==
  ["verity/certifier/v0", "verity/stub/pous/v0", "verity/stub/pouw/v0", "verity/stub/sampled-proofs/v0"]
-- a stub's case is not a period's: the certifier refuses its params, and onsite rejects whatever the stubs answer
#guard (answer (every Cases.accepts)).verdict == .reject
#guard (answer (every Cases.accepts)).published.isNone
#guard (answer [Cases.accepts NetworkCheck.Period.protocolId, Cases.accepts Stub.pous.id, Cases.undecided Stub.pouw.id,
  Cases.accepts Stub.sampledProofs.id]).verdict == .reject
-- remote's components alone are not onsite's
#guard (answer [Cases.accepts Stub.pouw.id, Cases.accepts Stub.sampledProofs.id]).reason ==
  some "params: verity/certifier/v0 is missing"

end Core.Onsite
