/-
The subprotocols a composition stands on until their ports land: each is `Core.Example.protocol` (k units drawn, all
passed) under an id of its own, which says it is a stand-in. A port replaces its stub in `Core.Onsite.components` and
`Core.Remote.components` with its own `Core.Protocol` value, under its own id, and its stub goes. Every subprotocol of
onsite and remote has been ported; `stub` stays for the composition's own guards.
-/
import CoreProtocol.Example

namespace Core.Composition.Stub

/-- `Core.Example.protocol` under the id `id`. -/
def stub (id : String) : Protocol := { Core.Example.protocol with id }

end Core.Composition.Stub
