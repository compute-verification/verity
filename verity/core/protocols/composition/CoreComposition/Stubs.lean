/-
The components a composition stands on until their ports land: each is `Core.Example.protocol` (k units drawn, all
passed) under an id of its own, which says it is a stand-in. A port replaces its stub in `Core.Onsite.components` and
`Core.Remote.components` with its own `Core.Protocol` value, under its own id.
-/
import CoreProtocol.Example

namespace Core.Composition.Stub

/-- `Core.Example.protocol` under the id `id`. -/
def stub (id : String) : Protocol := { Core.Example.protocol with id }

/-- Sampled proofs' one-stage audit: what ran (integrity). -/
def sampledProofs : Protocol := stub "verity/stub/sampled-proofs/v0"
/-- Proof of useful work: the tries made between the salt and the shares. -/
def pouw : Protocol := stub "verity/stub/pouw/v0"
/-- Proof of useful space: the blocks held, if answered in time. -/
def pous : Protocol := stub "verity/stub/pous/v0"
/-- The network certifier: what left the site. -/
def certifier : Protocol := stub "verity/stub/certifier/v0"

end Core.Composition.Stub
