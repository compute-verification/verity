/-
The audit record: an append-only log of what the coin server received and issued, in its order. It takes entries from
the coin server alone, and when the outer session finishes it closes and hands the whole log to the verifier. It holds
no secret and makes no choice; it is a party so that what the verifier checks is exactly what the coin server saw.
-/
import Run.Machine

namespace Run.Record

structure State where
  log : Array Entry := #[]
  closed : Bool := false
  deriving Inhabited

def step (s : State) (src : Node) (b : Body) (_ : ByteArray) : Except String (State × List (Node × Body)) := do
  if src != .party .coinServer then throw "record: entries come from the coin server only"
  let .log e := b | throw "record: not an entry"
  let log := s.log.push e
  match e with
  | .finished _ _ => return ({ log, closed := true }, [(.party .verifier, .closed log)])
  | _ => return ({ s with log }, [])

def core : Core where
  State := State
  init := {}
  fresh _ _ _ := 0
  step := step
  halted s := s.closed

def machine : Machine := core.seq (.party .record)

end Run.Record
