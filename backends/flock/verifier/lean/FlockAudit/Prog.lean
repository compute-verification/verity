/-!
# The auditor's program as data

`Prog α` is what the auditor runs, written as data: it asks its environment for bytes (`ask`), draws coins (`coin`), and
ends with an output (`ret`). The environment is everything that isn't this program: every worker (Python, Rust, CUDA), the
prover behind them, the auditor's content store and the launcher. It answers each request with bytes of its choosing, so
the analysis quantifies over it as an adversary.

`Talk α` is a program with no coin node: a part of the run that can draw nothing, whatever its replies say.

`runM` interprets a program in any monad, given an answer to each request and `n` coin bytes; `flock-audit` runs it in
`IO`, a request a frame to the channel's worker and a coin `IO.getRandomBytes`. `replay` is the program against a recorded
transcript, a pure function. `simulate` is `runM` against an environment that answers as a function of the run so far,
and `replay_simulate` says a run's own transcript replays to the run's output.
-/

namespace FlockAudit

/-- A request to the environment: the channel, canonical bytes, and the longest reply it takes. -/
structure Req where
  chan : String
  body : ByteArray
  max : Nat

inductive Prog (α : Type) : Type where
  | ret : α → Prog α
  | ask : Req → (ByteArray → Prog α) → Prog α
  | coin : Nat → (ByteArray → Prog α) → Prog α

/-- A program with requests and an output and no coin node. -/
inductive Talk (α : Type) : Type where
  | ret : α → Talk α
  | ask : Req → (ByteArray → Talk α) → Talk α

namespace Talk

def bind {α β : Type} : Talk α → (α → Talk β) → Talk β
  | .ret a, f => f a
  | .ask r k, f => .ask r fun b => bind (k b) f

instance : Monad Talk where
  pure := .ret
  bind := bind

def req (chan : String) (body : ByteArray) (max : Nat) : Talk ByteArray := .ask ⟨chan, body, max⟩ .ret

def lift {α : Type} : Talk α → Prog α
  | .ret a => .ret a
  | .ask r k => .ask r fun b => lift (k b)

end Talk

namespace Prog

def bind {α β : Type} : Prog α → (α → Prog β) → Prog β
  | .ret a, f => f a
  | .ask r k, f => .ask r fun b => bind (k b) f
  | .coin n k, f => .coin n fun b => bind (k b) f

instance : Monad Prog where
  pure := .ret
  bind := bind

def coins (n : Nat) : Prog ByteArray := .coin n .ret

/-- No coin node anywhere in `p`, whatever the replies. -/
inductive NoCoin {α : Type} : Prog α → Prop
  | ret (a : α) : NoCoin (.ret a)
  | ask (r : Req) (k : ByteArray → Prog α) : (∀ b, NoCoin (k b)) → NoCoin (.ask r k)

/-- One step of a run as the interpreter records it. -/
inductive Event where
  | asked (chan : String) (body reply : ByteArray)
  | drew (bytes : ByteArray)

/-- The interpreter, in any monad: each request answered by `ask`, each coin node's `n` bytes from `random`. -/
def runM {m : Type → Type} [Monad m] {α : Type} (ask : Req → m ByteArray) (random : Nat → m ByteArray) : Prog α → m α
  | .ret a => pure a
  | .ask r k => do runM ask random (k (← ask r))
  | .coin n k => do runM ask random (k (← random n))

/-- The program against a transcript: each request must be the one recorded, each reply within its bound, each draw of
the recorded length, and the transcript must end where the program does. -/
def replay {α : Type} : Prog α → List Event → Except String α
  | .ret a, [] => .ok a
  | .ret _, _ :: _ => .error "replay: the transcript continues past the program's end"
  | .ask r k, .asked c b reply :: es =>
    if c = r.chan ∧ b = r.body ∧ reply.size ≤ r.max then replay (k reply) es
    else .error s!"replay: the recorded request on {c} is not the program's"
  | .coin n k, .drew b :: es => if b.size = n then replay (k b) es else .error "replay: a draw of the wrong length"
  | .ask _ _, _ => .error "replay: the program asks where the transcript does not"
  | .coin _ _, _ => .error "replay: the program draws where the transcript does not"

/-- `runM` against an environment that answers each request (`env`) and each coin node (`draw`) as a function of the run
so far, holding each reply to its bound and each draw to its length as `flock-audit` does: the output and the run's
events, after the events `h` before it. -/
def simulate {α : Type} (env : List Event → Req → ByteArray) (draw : List Event → Nat → ByteArray) :
    Prog α → List Event → Except String (α × List Event)
  | .ret a, h => .ok (a, h)
  | .ask r k, h =>
    let b := env h r
    if b.size ≤ r.max then simulate env draw (k b) (h ++ [.asked r.chan r.body b])
    else .error s!"{r.chan}: a reply over the request's bound"
  | .coin n k, h =>
    let b := draw h n
    if b.size = n then simulate env draw (k b) (h ++ [.drew b]) else .error "a draw of the wrong length"

end Prog

theorem Prog.bind_assoc {α β γ : Type} (p : Prog α) (f : α → Prog β) (g : β → Prog γ) :
    (p.bind f).bind g = p.bind fun a => (f a).bind g := by
  induction p with
  | ret a => rfl
  | ask r k ih => exact congrArg (Prog.ask r) (funext ih)
  | coin n k ih => exact congrArg (Prog.coin n) (funext ih)

theorem Talk.lift_bind {α β : Type} (t : Talk α) (f : α → Talk β) :
    (t.bind f).lift = t.lift.bind fun a => (f a).lift := by
  induction t with
  | ret a => rfl
  | ask r k ih => exact congrArg (Prog.ask r) (funext ih)

open Prog in
/-- A `Talk` lifted to a program has no coin node. -/
theorem Talk.noCoin_lift {α : Type} (t : Talk α) : NoCoin t.lift := by
  induction t with
  | ret a => exact .ret a
  | ask r k ih => exact .ask r _ ih

open Prog in
/-- **A run's own transcript replays to the run's output.** Whatever the environment answers, if the program's run against
it ends with output `a`, the events it appended after `h` replay to `a`. -/
theorem replay_simulate {α : Type} (env : List Event → Req → ByteArray) (draw : List Event → Nat → ByteArray)
    (p : Prog α) (h : List Event) {a : α} {es : List Event} (hs : simulate env draw p h = .ok (a, es)) :
    ∃ suf, es = h ++ suf ∧ replay p suf = .ok a := by
  induction p generalizing h with
  | ret x =>
    simp only [simulate, Except.ok.injEq, Prod.mk.injEq] at hs
    obtain ⟨rfl, rfl⟩ := hs
    exact ⟨[], by simp, rfl⟩
  | ask r k ih =>
    simp only [simulate] at hs
    split at hs
    · next hb =>
      obtain ⟨suf, hes, hr⟩ := ih _ _ hs
      refine ⟨.asked r.chan r.body (env h r) :: suf, by simp [hes], ?_⟩
      simp [replay, hb, hr]
    · exact absurd hs (by simp)
  | coin n k ih =>
    simp only [simulate] at hs
    split at hs
    · next hb =>
      obtain ⟨suf, hes, hr⟩ := ih _ _ hs
      refine ⟨.drew (draw h n) :: suf, by simp [hes], ?_⟩
      simp [replay, hb, hr]
    · exact absurd hs (by simp)

end FlockAudit
