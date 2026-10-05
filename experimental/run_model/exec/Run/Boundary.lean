/-
The developer's boundary: its four parties as machines, each over the same agreement.
-/
import Run.Boundary.Proxy
import Run.Boundary.ZkLayer
import Run.Boundary.Recheck
import Run.Boundary.SendGate

namespace Run.Boundary

def machine (A : Agreement) : (p : Party) → p.side = .boundary → Machine
  | .proxy, _ => Proxy.machine A
  | .zkLayer, _ => ZkLayer.machine A
  | .recheck, _ => Recheck.machine A
  | .sendGate, _ => SendGate.machine A
  | .coinServer, h | .record, h | .verifier, h | .profile, h => absurd h (by decide)

end Run.Boundary
