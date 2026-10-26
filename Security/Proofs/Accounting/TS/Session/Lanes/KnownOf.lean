import Proofs.Accounting.TS.Session.Lanes.Decode
import PouwWindow.KnownRows

/-!
# The known content a leaf's sessions are handed

`knownOf P salts k`: statement `k`'s known rows at the public inputs `P` (`PouwWindow.Lanes.knownRows`, held to the
Python that writes them by its difftest): lane `l`'s `admit` row (`P.admit`), its `pub` row (`P.salt`, `P.position`,
`P.roots` and the index `l`, each least significant bit first at its field's width) and its `o0` row 1, and the
combine's `o0`, each under its salt `salts`. `Accounting.Guarantees.TS.LeafLanesSound` hands each session this content,
so its known-content check (`Flock.Registered.checkKnownAll`) binds the path's public inputs.
-/

namespace Accounting.TS.Lanes

/-- **Statement `k`'s known content at the public inputs `P`**: `PouwWindow.Lanes.knownRows` at `P`'s fields, lane `l`'s
(`some l`) or the combine's (`none`), each row under its salt `salts`. -/
def knownOf (P : LanePub) (salts : String → ByteArray) (k : Option (Fin 32)) : Known :=
  PouwWindow.Lanes.knownRows (List.ofFn P.admit) P.salt P.position P.roots salts (k.map Fin.val)

end Accounting.TS.Lanes
