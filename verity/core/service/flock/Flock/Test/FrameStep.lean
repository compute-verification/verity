/-
`Frame.step`, the frame's answer to one request and its state after it, and `Frame.run`, its fold: a round squeezing two
coins passes against itself and owes them; a second round, coins for no round, coins of another count and an earlier
rep's coins after the first round are refused; and `run` ends at the first answer that ends the rep.
-/
import Flock.Firewall.Frame

namespace Flock.Test.FrameStep

open Flock.Firewall.Frame
open Flock.Transcript (header)

def c : Cfg := default

/-- A layout whose algebraic phase squeezes `K_SKIP + 0 + 1 + 7 + 2 + 0 + 1 = 17` coins (`claimPoints`). -/
def c7 : Cfg := { (default : Cfg) with lay := ⟨7, 0, 0, 0⟩ }

def zeros (k : Nat) : ByteArray := ⟨Array.replicate (16 * k) 0⟩

def sq : ByteArray := header 4 2 2

def kind : Reply → String
  | .round _ v => s!"round{v}"
  | .ok => "ok"
  | .err _ => "err"
  | .done b => s!"done {b}"
  | .refuse _ => "refuse"
  | .stop _ => "stop"

def after (qs : List Req) : State := (run c {} qs).1

#guard kind (step c {} (.round 2 sq (.sched sq))).1 == "round pass"
#guard (step c {} (.round 2 sq (.sched sq))).2.m.owed == some 2
#guard (step c {} (.round 2 sq (.sched sq))).2.begun
#guard kind (step c {} (.round 2 sq .fail)).1 == "round"
#guard (step c {} (.round 2 sq .fail)).2.m.stopped
#guard kind (step c {} (.round 2 sq .own)).1 == "err"
#guard (step c {} (.round 2 sq .own)).2.m.stopped
#guard kind (step c (after [.round 2 sq (.sched sq)]) (.round 2 sq (.sched sq))).1 == "refuse"
#guard kind (step c {} (.coins (.ok #[0, 0]))).1 == "refuse"
#guard kind (step c (after [.round 2 sq (.sched sq)]) (.coins (.ok #[0]))).1 == "refuse"
#guard kind (step c (after [.round 2 sq (.sched sq)]) (.coins (.ok #[0, 0]))).1 == "ok"
#guard (after [.round 2 sq (.sched sq), .coins (.ok #[0, 0])]).m.owed == none
#guard kind (step c (after [.round 2 sq (.sched sq)]) (.coins (.error "no coins"))).1 == "refuse"
#guard kind (step c7 {} (.prior 17 (zeros 17))).1 == "ok"
#guard (step c7 {} (.prior 17 (zeros 17))).2.m.prior.size > 0
#guard kind (step c7 {} (.prior 16 (zeros 16))).1 == "refuse"
#guard kind (step c7 (after [.round 2 sq (.sched sq)]) (.prior 17 (zeros 17))).1 == "refuse"
#guard kind (step c {} .finish).1 == "refuse"
#guard (run c {} [.coins (.ok #[]), .round 2 sq (.sched sq)]).2
#guard !(run c {} [.round 2 sq (.sched sq), .coins (.ok #[0, 0])]).2

end Flock.Test.FrameStep
