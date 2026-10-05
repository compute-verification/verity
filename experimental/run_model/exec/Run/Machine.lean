/-
A party as a pure state machine, and the one wrapper that sequences it.

A party's own logic is a `Core`: a step from its state, a message (sender and body) and the fresh random bytes the step
reads, to its next state and the messages it sends, or a refusal. `Core.seq` puts it on the wire: frames addressed to
it, from each sender in order, each exactly once, numbered on the way out. Every party is a `Core.seq`, so the rule
that a party refuses an out-of-order or duplicate message is one theorem about `Core.seq` (`Rules.seq_inOrder`), not
one per party.

A refusal leaves the state as it was and sends nothing: a bug or an attack on the untrusted side can stop a run, never
move a party to a state its logic didn't choose.
-/
import Run.Wire

namespace Run

structure Core where
  State : Type
  init : State
  /-- the fresh random bytes the step on this message reads: zero for a deterministic step -/
  fresh : State → Node → Body → Nat
  step : State → Node → Body → ByteArray → Except String (State × List (Node × Body))
  /-- the party takes no more messages: it issued its verdict, or a check failed -/
  halted : State → Bool

structure Machine where
  State : Type
  init : State
  fresh : State → Frame → Nat
  step : State → Frame → ByteArray → Except String (State × List Frame)

/-- A count per node. -/
abbrev Counts := Node → Nat

def Counts.bump (c : Counts) (n : Node) : Counts := fun m => if m = n then c m + 1 else c m

/-- Number a step's messages, `seq` the count of earlier frames from `self` to the same receiver. -/
def number (self : Node) : Counts → List (Node × Body) → Counts × List Frame
  | sent, [] => (sent, [])
  | sent, (dst, body) :: rest =>
    let (sent', fs) := number self (sent.bump dst) rest
    (sent', { src := self, dst, seq := sent dst, body } :: fs)

structure Seq (S : Type) where
  core : S
  /-- the next `seq` expected from each sender -/
  recv : Counts
  /-- the next `seq` to each receiver -/
  sent : Counts

/-- `C` at node `self` on the wire. -/
def Core.seq (C : Core) (self : Node) : Machine where
  State := Seq C.State
  init := ⟨C.init, fun _ => 0, fun _ => 0⟩
  fresh s f := C.fresh s.core f.src f.body
  step s f r := do
    if f.dst != self then throw "not addressed to this party"
    if C.halted s.core then throw "halted"
    if f.seq < s.recv f.src then throw "duplicate"
    if f.seq > s.recv f.src then throw "out of order"
    if r.size != C.fresh s.core f.src f.body then throw "fresh bytes: not the step's count"
    let (c', outs) ← C.step s.core f.src f.body r
    let (sent', frames) := number self s.sent outs
    return (⟨c', s.recv.bump f.src, sent'⟩, frames)

/-- One machine's run on its inputs, each with the fresh bytes it was handed: the final state, and each accepted frame
with what the step sent. A refused frame leaves no trace. -/
def Machine.run (M : Machine) : M.State → List (Frame × ByteArray) → M.State × List (Frame × ByteArray × List Frame)
  | s, [] => (s, [])
  | s, (f, r) :: rest => match M.step s f r with
    | .ok (s', outs) => let (t, tr) := M.run s' rest; (t, (f, r, outs) :: tr)
    | .error _ => M.run s rest

end Run
