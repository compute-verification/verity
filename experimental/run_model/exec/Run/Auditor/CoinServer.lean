/-
The coin server: the auditor's only party on the network. It takes commitments from the boundary's send gate and
answers each with coins drawn after it, logs both to the record, and never answers a slot twice.

* registration: the proxy's salted hashes of the registration roots, answered with the unit draw (`Flock.Draw.drawOn`
  on the step's fresh bytes);
* each inner round: the proxy's salted hashes of the round's message, answered with the round's live coins;
* the outer session: its `Hello` (R7, `Flock.CoinTree.checkHello`), answered with the coin tree's root under a fresh
  key, every coin of the session drawn then and committed; each later round's message, answered with that round's
  opening of the tree;
* `finish`: logged, and the record closes.
-/
import Run.Machine
import Run.Agreement
import Flock.CoinTree

namespace Run.CoinServer

open Lean (Json)

inductive Phase where
  | registration
  | inner
  | outer
  | done
  deriving DecidableEq, Repr, Inhabited

/-- The session's coin tree, drawn at `Hello`: its key, every coin and salt under it, and its root. -/
structure Tree where
  key : ByteArray
  leaves : ByteArray
  root : Digest
  deriving Inhabited

structure State where
  phase : Phase := .registration
  /-- the slots whose commitment arrived, each at most once -/
  committed : List Slot := []
  /-- the slots answered, each once, each after its commitment -/
  issued : List Slot := []
  tree : Option Tree := none
  /-- the last outer round answered -/
  outerAt : Nat := 0
  halted : Bool := false
  deriving Inhabited

/-- The bytes `Flock.Draw.drawOn` reads for the law: one stream for `subset` and `bernoulli`, one segment per stratum. -/
def drawNeed (law : Flock.Draw.Law) (population : Nat) : Nat :=
  match law with
  | .subset k => 2 * k * Flock.Draw.width population + 64
  | .bernoulli _ den => 2 * population * Flock.Draw.width den + 64
  | .stratified _ strata => strata.foldl (fun acc s => acc + s.need) 0
  | .work _ strata => strata.foldl (fun acc s => acc + s.toStratum.need) 0

/-- The key's bytes and the leaves' (every round's coins with their salts) a coin tree of `spec` takes. -/
def treeNeed (spec : Flock.CoinTree.Spec) : Nat := 256 + spec.streams.size * 2 ^ spec.roundsLog * (spec.block + 192)

/-- The tree's root over `leaves` under `key` and the prover's `nonce` (`coin-tree/hm96-sha512/v2` §3). The server's
side of the tree: the executable verifier holds only the record's check. -/
def treeRoot (spec : Flock.CoinTree.Spec) (nonce : String) (key leaves : ByteArray) : Digest := sorry

/-- Round `k`'s opening on the tree: its coins, their salts and the paths to the root. -/
def treeOpen (spec : Flock.CoinTree.Spec) (t : Tree) (k : Nat) : ByteArray := sorry

/-- The key with bit 2047 cleared, as `checkRecord` requires. -/
def clearTop (key : ByteArray) : ByteArray :=
  if key.size == 256 then key.set! 255 (key.get! 255 &&& 0x7f) else key

def wellFormed (A : Agreement) (slot : Slot) (hs : Array Digest) : Bool :=
  hs.size == A.commitCount slot && hs.all (·.size == 64)

def toProxy (b : Body) : Node × Body := (.party .proxy, b)
def toRecord (e : Entry) : Node × Body := (.party .record, .log e)

def fresh (A : Agreement) (s : State) : Node → Body → Nat
  | .party .sendGate, .commit .registration _ => if s.phase == .registration then drawNeed A.law A.population else 0
  | .party .sendGate, .commit (.inner _ _) _ => if s.phase == .inner then A.coinBytes else 0
  | .party .sendGate, .outer 0 _ => match s.phase, s.tree, A.outer.coinTree with
    | .outer, none, some spec => treeNeed spec
    | _, _, _ => 0
  | _, _ => 0

def step (A : Agreement) (s : State) (src : Node) (b : Body) (r : ByteArray) :
    Except String (State × List (Node × Body)) := do
  if src != .party .sendGate then throw "coin server: frames come from the boundary's send gate only"
  match b with
  | .commit .registration hs =>
    if s.phase != .registration then throw "coin server: registration is over"
    if !wellFormed A .registration hs then throw "coin server: not the registration's hashes"
    match Flock.Draw.drawOn A.law A.population r with
    | none => return ({ s with halted := true }, [])
    | some d =>
      return ({ s with phase := .inner, committed := [.registration], issued := [.registration] },
        [toRecord (.committed .registration hs), toRecord (.drew d), toProxy (.draw d)])
  | .commit (.inner bt rd) hs =>
    let slot := Slot.inner bt rd
    if s.phase != .inner then throw "coin server: not in the inner phase"
    if s.committed.contains slot then throw "coin server: the slot's commitment arrived before"
    if !A.innerNext s.issued bt rd then throw "coin server: not the batch's next round"
    if !wellFormed A slot hs then throw "coin server: not the round's hashes"
    let issued := slot :: s.issued
    let phase := if A.innerDone issued then .outer else .inner
    return ({ s with phase, committed := slot :: s.committed, issued },
      [toRecord (.committed slot hs), toRecord (.issued slot r), toProxy (.coins slot r)])
  | .outer 0 hello =>
    if s.phase != .outer || s.tree.isSome then throw "coin server: not the outer session's Hello"
    let some spec := A.outer.coinTree | throw "coin server: the outer session names no coin tree"
    let own ← match Json.parse A.outer.hello with
      | .ok j => pure j
      | .error e => throw s!"coin server: the agreement's Hello: {e}"
    let nonce ← Flock.CoinTree.checkHello own (String.fromUTF8! hello)
    let key := clearTop (r.extract 0 256)
    let leaves := r.extract 256 r.size
    let t : Tree := { key, leaves, root := treeRoot spec nonce key leaves }
    return ({ s with tree := some t, committed := .outer 0 :: s.committed, issued := .outer 0 :: s.issued },
      [toRecord (.outer 0 hello), toRecord (.issued (.outer 0) (t.root ++ key)), toProxy (.coins (.outer 0) (t.root ++ key))])
  | .outer k msg =>
    let some t := s.tree | throw "coin server: no outer session"
    if k != s.outerAt + 1 || k > A.outerRounds then throw "coin server: not the next outer round"
    let some spec := A.outer.coinTree | throw "coin server: the outer session names no coin tree"
    let opening := treeOpen spec t k
    return ({ s with outerAt := k, committed := .outer k :: s.committed, issued := .outer k :: s.issued },
      [toRecord (.outer k msg), toRecord (.issued (.outer k) opening), toProxy (.coins (.outer k) opening)])
  | .finish record proofs =>
    if s.phase != .outer || s.outerAt != A.outerRounds then throw "coin server: the outer session is not over"
    return ({ s with phase := .done }, [toRecord (.finished record proofs)])
  | _ => throw "coin server: not a message it takes"

def core (A : Agreement) : Core where
  State := State
  init := {}
  fresh := fresh A
  step := step A
  halted s := s.halted || s.phase == .done

def machine (A : Agreement) : Machine := (core A).seq (.party .coinServer)

end Run.CoinServer
