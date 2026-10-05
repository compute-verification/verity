/-
The auditor's side: its four parties as machines, each over the same agreement.
-/
import Run.Auditor.CoinServer
import Run.Auditor.Record
import Run.Auditor.Verifier
import Run.Auditor.Profile

namespace Run.Auditor

def machine (A : Agreement) : (p : Party) → p.side = .auditor → Machine
  | .coinServer, _ => CoinServer.machine A
  | .record, _ => Record.machine
  | .verifier, _ => Verifier.machine A
  | .profile, _ => Profile.machine A
  | .proxy, h | .zkLayer, h | .recheck, h | .sendGate, h => absurd h (by decide)

end Run.Auditor
