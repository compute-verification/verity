/-
The components a composition stands on until their ports land: each is `Core.Example.protocol` (k units drawn, all
passed) under an id of its own, which says it is a stand-in. A port replaces its stub in `Core.Onsite.components` and
`Core.Remote.components` with its own `Core.Protocol` value, under its own id, and its stub goes. Work, integrity and
communication have been ported (`PouwCheck.protocol`, `SampledProofs.OneStage.protocol`, `NetworkCheck.Check.protocol`).
-/
import CoreProtocol.Example

namespace Core.Composition.Stub

/-- `Core.Example.protocol` under the id `id`. -/
def stub (id : String) : Protocol := { Core.Example.protocol with id }

/-- Proof of useful space: the blocks held, if answered in time. -/
def pous : Protocol := stub "verity/stub/pous/v0"
/-- Proof of useful bandwidth: the reads of the store held, over PoUS's store or a node's registered one. -/
def bandwidth : Protocol := stub "verity/stub/bandwidth/v0"

end Core.Composition.Stub
