/-
The send gate: the boundary's only exit. A frame reaches the network only from here, and only in the step that checked
it:

* a proxy commitment: one per slot, the slot's count of 64-byte hashes;
* an outer message from the outer prover: the next round's, and canonical (every byte the one the outer prover's code
  determines from the transcript so far, V*'s witness and the ZK layer's masks and salts); the level-0 message is held
  until the recheck returns the same commitment, recomputed;
* the outer session's `finish`: the outer check the verifier will run (`Run.Statement.verify`), run here first on the
  boundary's own copy of the run's public data.

Everything else it takes (the draw, coins, witness, masks) it keeps, to check with.
-/
import Run.Boundary.Proxy
import Run.Statement

namespace Run.SendGate

structure State where
  released : List Slot := []
  commits : List (Slot × Array Digest) := []
  /-- the proxy's hashes it released and the coins that answered them, as the record will keep them -/
  pub : Parsed := {}
  witness : Option (Array ByteArray × Array Salt) := none
  masks : Option (ByteArray × Array Salt) := none
  /-- the outer messages released, in round order -/
  outer : Array ByteArray := #[]
  /-- the level-0 message, held for the recheck -/
  held : Option (Nat × ByteArray) := none
  finished : Bool := false
  deriving Inhabited

/-- The outer round whose message carries the level-0 commitment. -/
def level0Round : Nat := 1

/-- The level-0 commitment the round's message carries. -/
def level0Root (msg : ByteArray) : Digest := sorry

/-- Round `k`'s message is canonical: the one the outer prover's code computes from the transcript so far, V*'s
witness and the masks and salts, byte for byte. -/
def canonical (A : Agreement) (s : State) (k : Nat) (msg : ByteArray) : Bool := sorry

def toServer (b : Body) : Node × Body := (.party .coinServer, b)

def step (A : Agreement) (s : State) (src : Node) (b : Body) (_ : ByteArray) :
    Except String (State × List (Node × Body)) := do
  match src, b with
  | .party .proxy, .commit slot hs =>
    if s.released.contains slot then throw "send gate: the slot was committed before"
    if hs.size != A.commitCount slot || !hs.all (·.size == 64) then throw "send gate: not the slot's hashes"
    let pub := match slot with
      | .registration => { s.pub with reg := hs }
      | _ => s.pub
    return ({ s with released := slot :: s.released, commits := (slot, hs) :: s.commits, pub },
      [toServer (.commit slot hs)])
  | .party .proxy, .draw d => return ({ s with pub := { s.pub with draw := some d } }, [])
  | .party .proxy, .coins slot@(.inner _ _) c =>
    let some (_, hs) := s.commits.find? (·.1 == slot) | throw "send gate: coins for a slot it did not release"
    return ({ s with pub := { s.pub with inner := s.pub.inner.push (slot, hs, c) } }, [])
  | .party .proxy, .coins (.outer k) c =>
    match s.outer[k]? with
    | some msg => return ({ s with pub := { s.pub with outer := s.pub.outer.push (k, msg, c) } }, [])
    | none => throw "send gate: coins for an outer round it did not release"
  | .party .proxy, .witness rows salts => return ({ s with witness := some (rows, salts) }, [])
  | .party .zkLayer, .masks m salts => return ({ s with masks := some (m, salts) }, [])
  | .worker .outerProver, .outer k msg =>
    if s.held.isSome then throw "send gate: the level-0 message waits for the recheck"
    if k != s.outer.size || k > A.outerRounds then throw "send gate: not the next outer round"
    if !canonical A s k msg then throw "send gate: not the canonical message"
    if k == level0Round then
      return ({ s with held := some (k, msg) }, [(.party .recheck, .level0 (level0Root msg))])
    return ({ s with outer := s.outer.push msg }, [toServer (.outer k msg)])
  | .party .recheck, .level0 root =>
    let some (k, msg) := s.held | throw "send gate: nothing held"
    if level0Root msg != root then throw "send gate: the recheck's commitment is not the held message's"
    return ({ s with held := none, outer := s.outer.push msg }, [toServer (.outer k msg)])
  | .worker .outerProver, .finish record proofs =>
    if s.outer.size != A.outerRounds + 1 then throw "send gate: the outer session is not over"
    let p := { s.pub with record, proofs, finished := true }
    Statement.verify A p
    return ({ s with finished := true, pub := p }, [toServer (.finish record proofs)])
  | _, _ => throw "send gate: not a message it takes"

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh _ _ _ := 0
  step := step A
  halted s := s.finished

def machine (A : Agreement) : Machine := (core A).seq (.party .sendGate)

end Run.SendGate
