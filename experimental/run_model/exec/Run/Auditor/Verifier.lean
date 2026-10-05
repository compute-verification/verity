/-
The verifier: the executable Lean verifier on the closed record. It reads the log back (each commitment followed by its
answer, the registration first, `finished` last) and runs the outer check (`Run.Statement.verify`): V*'s public words
from the proxy's hashes, its own coins and draw and the comb it computes from the public inner circuit, and
`Flock.Zk.verify` with V*'s setup pinned by digest. It never sees a prover message of the inner phase.
-/
import Run.Machine
import Run.Statement

namespace Run.Verifier

/-- The closed log read back, refusing any log the coin server could not have written. -/
def parse (log : Array Entry) : Except String Parsed := do
  let mut p : Parsed := {}
  let mut pending : Option (Slot × Array Digest) := none
  let mut pendingOuter : Option (Nat × ByteArray) := none
  for e in log do
    if p.finished then throw "verifier: an entry after `finished`"
    match e with
    | .committed .registration hs =>
      if !p.reg.isEmpty || pending.isSome then throw "verifier: a second registration"
      p := { p with reg := hs }
      pending := some (.registration, hs)
    | .drew d =>
      if pending.map (·.1) != some .registration then throw "verifier: a draw not answering the registration"
      p := { p with draw := some d }
      pending := none
    | .committed slot hs =>
      if pending.isSome || p.draw.isNone then throw "verifier: a commitment before the last one's answer"
      pending := some (slot, hs)
    | .outer k msg =>
      if pending.isSome || pendingOuter.isSome then throw "verifier: an outer message before the last one's answer"
      pendingOuter := some (k, msg)
    | .issued slot c =>
      match slot, pending, pendingOuter with
      | .inner _ _, some (s', hs), _ =>
        if s' != slot then throw "verifier: coins for another slot"
        p := { p with inner := p.inner.push (slot, hs, c) }
        pending := none
      | .outer k, _, some (k', msg) =>
        if k != k' then throw "verifier: an opening for another outer round"
        p := { p with outer := p.outer.push (k, msg, c) }
        pendingOuter := none
      | _, _, _ => throw "verifier: coins answering no commitment"
    | .finished record proofs =>
      if pending.isSome || pendingOuter.isSome then throw "verifier: `finished` before the last answer"
      p := { p with record, proofs, finished := true }
  if !p.finished then throw "verifier: the log is not closed"
  return p

def check (A : Agreement) (log : Array Entry) : Except String Parsed := do
  let p ← parse log
  if p.draw.isNone then throw "verifier: the record keeps no draw"
  if p.inner.size != A.batches * A.rounds then throw "verifier: not every inner round was answered"
  if p.outer.size != A.outerRounds + 1 then throw "verifier: not every outer round was answered"
  Statement.verify A p
  return p

structure State where
  done : Bool := false
  deriving Inhabited

def step (A : Agreement) (_ : State) (src : Node) (b : Body) (_ : ByteArray) :
    Except String (State × List (Node × Body)) := do
  if src != .party .record then throw "verifier: it reads the closed record only"
  let .closed log := b | throw "verifier: not a closed record"
  let out := match check A log with
    | .ok _ => Body.verdict true "" log
    | .error why => Body.verdict false why log
  return ({ done := true }, [(.party .profile, out)])

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh _ _ _ := 0
  step := step A
  halted s := s.done

def machine (A : Agreement) : Machine := (core A).seq (.party .verifier)

end Run.Verifier
