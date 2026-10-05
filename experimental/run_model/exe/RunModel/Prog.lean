import RunModel.Frame

/-!
# A party's program over the whole run

`Prog α` is what one party runs, written as data: it asks its environment for bytes (`ask`), draws coins from the
operating system (`coin`), and ends with an output (`ret`). The environment is everything that isn't this program: the
other party, every worker (Python, Rust, CUDA), the network, the store and the launcher. It answers each request with
bytes of its choosing, so the model quantifies over it as an adversary.

`run` interprets a program in `IO`: a request becomes a frame to the named worker and the reply is read back as a frame;
a coin is `IO.getRandomBytes`. `replay` is the same program against a recorded transcript, a pure function: re-checking
a run offline is this program again, with the environment's replies and the coins read from the record.

The program's control flow is the protocol's order: a coin is drawn only after the replies before it are held, and
nothing outside the program can ask it for another coin.
-/

namespace RunModel

/-- A request to the environment: the channel (a worker or a peer), canonical bytes, and the longest reply it takes. -/
structure Req where
  chan : String
  body : ByteArray
  max : Nat

inductive Prog (α : Type) : Type where
  | ret : α → Prog α
  | ask : Req → (ByteArray → Prog α) → Prog α
  | coin : Nat → (ByteArray → Prog α) → Prog α

namespace Prog

def bind {α β : Type} : Prog α → (α → Prog β) → Prog β
  | .ret a, f => f a
  | .ask r k, f => .ask r fun b => bind (k b) f
  | .coin n k, f => .coin n fun b => bind (k b) f

instance : Monad Prog where
  pure := .ret
  bind := bind

def req (chan : String) (body : ByteArray) (max : Nat) : Prog ByteArray := .ask ⟨chan, body, max⟩ .ret

def coins (n : Nat) : Prog ByteArray := .coin n .ret

/-- One step of a run as the interpreter records it. -/
inductive Event where
  | asked (chan : String) (body reply : ByteArray)
  | drew (bytes : ByteArray)

/-- What the interpreter needs from the host: an answer to each request, and `n` bytes of OS randomness. -/
structure Io where
  ask : Req → IO ByteArray
  random : Nat → IO ByteArray

def run {α : Type} (io : Io) : Prog α → IO α
  | .ret a => pure a
  | .ask r k => do run io (k (← io.ask r))
  | .coin n k => do run io (k (← io.random n))

/-- The program against a transcript: each request must be the one recorded, each reply within its bound, each draw of
the recorded length. -/
def replay {α : Type} : Prog α → List Event → Except String α
  | .ret a, [] => .ok a
  | .ret _, _ :: _ => .error "replay: the transcript continues past the program's end"
  | .ask r k, .asked c b reply :: es =>
    if c == r.chan && b == r.body && reply.size ≤ r.max then replay (k reply) es
    else .error s!"replay: the recorded request on {c} is not the program's"
  | .coin n k, .drew b :: es => if b.size == n then replay (k b) es else .error "replay: a draw of the wrong length"
  | _, _ => .error "replay: the transcript is not this program's run"

end Prog

/-- A worker the program talks to over its stdin and stdout. -/
structure Worker where
  name : String
  stdin : IO.FS.Handle
  stdout : IO.FS.Handle

/-- Spawn `cmd args` as the worker `name`, with piped stdin and stdout. -/
def Worker.spawn (name cmd : String) (args : Array String) : IO Worker := do
  let child ← IO.Process.spawn { cmd, args, stdin := .piped, stdout := .piped, stderr := .inherit }
  return ⟨name, child.stdin, child.stdout⟩

/-- The interpreter over workers and the OS, recording every event in `log`. -/
def Prog.Io.ofWorkers (ws : List Worker) (log : IO.Ref (Array Prog.Event)) : Prog.Io where
  ask r := do
    let some w := ws.find? (·.name == r.chan) | throw (IO.userError s!"no worker serves {r.chan}")
    Frame.write w.stdin r.body
    let reply ← Frame.read w.stdout r.max
    log.modify (·.push (.asked r.chan r.body reply))
    return reply
  random n := do
    let b ← IO.getRandomBytes n.toUSize
    log.modify (·.push (.drew b))
    return b

end RunModel
