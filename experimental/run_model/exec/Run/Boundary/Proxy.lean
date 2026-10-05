/-
The proxy: every message leaving the GPUs goes through it, and what it lets out is a salted hash. It takes the
registration roots and each inner round's message from the GPU rack, commits each root and each 64-word row under its
own fresh 192-byte salt (hm96-sha512/v1), sends the commitments to the send gate, and keeps the rows and salts: they are
V*'s witness, handed inward to the ZK layer, the recheck and the outer prover once the inner phase is over. Coins and the
draw from the coin server go to the GPU rack and to the send gate.

A salt is never used twice: a step whose fresh salts repeat one another or an earlier one is refused.
-/
import Run.Machine
import Run.Agreement

namespace Run.Proxy

inductive Phase where
  | registration
  | inner
  | outer
  deriving DecidableEq, Repr, Inhabited

structure State where
  phase : Phase := .registration
  /-- every committed value (a root, or a row) and its salt, in commitment order: V*'s witness -/
  rows : Array ByteArray := #[]
  salts : Array Salt := #[]
  /-- the slot committed and not yet answered: the proxy forwards one at a time -/
  pending : Option Slot := none
  answered : List Slot := []
  deriving Inhabited

def saltBytes : Nat := Flock.Hm96.sha512.saltBytes
/-- A row: 64 u16 words, one SHA-512 block, the last zero-padded. -/
def rowBytes : Nat := 128

def rowsOf (msg : ByteArray) : Array ByteArray :=
  (Array.range ((msg.size + rowBytes - 1) / rowBytes)).map fun i =>
    Flock.Hm96.pad (msg.extract (i * rowBytes) (i * rowBytes + rowBytes)) rowBytes

def saltsOf (r : ByteArray) (n : Nat) : Array Salt :=
  (Array.range n).map fun i => r.extract (i * saltBytes) (i * saltBytes + saltBytes)

/-- A value's commitment: the hm96 leaf of its SHA-512 digest under its salt. -/
def hide (v salt : ByteArray) : Digest := Flock.Hm96.Default512.leaf (Flock.Sha512.hash v) salt

def nodupBytes (xs : Array ByteArray) : Bool :=
  (List.range xs.size).all fun i => (List.range i).all fun j => xs[i]! != xs[j]!

def fresh (A : Agreement) (s : State) : Node → Body → Nat
  | .worker .innerProver, .roots rs => if s.phase == .registration then saltBytes * rs.size else 0
  | .worker .innerProver, .innerMsg _ r _ => if s.phase == .inner then saltBytes * A.innerRows r else 0
  | _, _ => 0

/-- Commit `values` for `slot` under the step's fresh salts. -/
def commitAll (s : State) (slot : Slot) (values : Array ByteArray) (r : ByteArray) :
    Except String (State × List (Node × Body)) := do
  let fresh := saltsOf r values.size
  if !nodupBytes (s.salts ++ fresh) then throw "proxy: a salt repeats; refused"
  let hs := (values.zip fresh).map fun (v, y) => hide v y
  return ({ s with rows := s.rows ++ values, salts := s.salts ++ fresh, pending := some slot },
    [(.party .sendGate, .commit slot hs)])

def step (A : Agreement) (s : State) (src : Node) (b : Body) (r : ByteArray) :
    Except String (State × List (Node × Body)) := do
  match src, b with
  | .worker .innerProver, .roots rs =>
    if s.phase != .registration || s.pending.isSome then throw "proxy: not the registration"
    if rs.size != A.roots then throw "proxy: not the agreement's number of roots"
    commitAll s .registration rs r
  | .worker .innerProver, .innerMsg bt rd msg =>
    if s.phase != .inner || s.pending.isSome then throw "proxy: a round is waiting for its coins"
    let rows := rowsOf msg
    if rows.size != A.innerRows rd then throw "proxy: not the round's rows"
    commitAll s (.inner bt rd) rows r
  | .party .coinServer, .draw d =>
    if s.pending != some .registration then throw "proxy: a draw answering nothing"
    return ({ s with phase := .inner, pending := none, answered := .registration :: s.answered },
      [(.worker .innerProver, .draw d), (.party .sendGate, .draw d)])
  | .party .coinServer, .coins slot@(.inner _ _) c =>
    if s.pending != some slot then throw "proxy: coins answering nothing"
    let answered := slot :: s.answered
    let fwd : List (Node × Body) := [(.worker .innerProver, .coins slot c), (.party .sendGate, .coins slot c)]
    if A.innerDone answered then
      let w := Body.witness s.rows s.salts
      return ({ s with phase := .outer, pending := none, answered },
        fwd ++ [(.party .zkLayer, w), (.party .recheck, w), (.party .sendGate, w), (.worker .outerProver, w)])
    return ({ s with pending := none, answered }, fwd)
  | .party .coinServer, .coins slot@(.outer _) c =>
    if s.phase != .outer then throw "proxy: no outer session"
    return (s, [(.worker .outerProver, .coins slot c), (.party .sendGate, .coins slot c)])
  | _, _ => throw "proxy: not a message it takes"

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh := fresh A
  step := step A
  halted _ := false

def machine (A : Agreement) : Machine := (core A).seq (.party .proxy)

end Run.Proxy
