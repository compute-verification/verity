import World
import MemoryChallenger.Step

/-!
# The driver

`step` run over a `Core.World Out`: the only IO of the memory challenger. It needs only an order: it reads the clock for
a challenge before it sends the challenge (`tick`, then `send`), and for an answer after its bytes arrive (`readUntil`,
then `received`).

With no round open the next event is a `tick`. With one open it is the link's next bytes before the round's deadline
(`readUntil (send_ns + limit_ns + 1)`), the end of the link (an empty read or an error), or, once the clock reads past the
deadline with nothing arrived, a `tick`, which ends the session `LATE`. The link is read only with a round open, so bytes
sent before a challenge are read after it, as the start of its answer: a peer learns nothing by sending early that it
could not by guessing once the challenge arrives.

The loop is bounded: every event but a deadline's `tick` either opens a round, adds at least one byte to its answer, or
ends the session, so `1 + k · (block_bytes + 2)` events end any session whose `World` keeps `readUntil`'s contract (the
clock reads at least the deadline once it returns `none`). A `World` that breaks it runs the loop out, and the session
ends as at the end of the link: `LATE`.
-/

namespace MemoryChallenger

/-- What the driver does for an action other than `reserve`. -/
private def emit (w : Core.World Out) : Action → IO Unit
  | .send b => w.link.write b
  | .publish o => w.publish o
  | .reserve => pure ()

/-- Perform `step`'s actions: a `reserve` asks `w` for a leaf and performs what the reservation's step does, which is at
most a `publish` (`step` on `reserved` takes no `reserve`, so `emit` never sees one). -/
def perform (cfg : Config) (w : Core.World Out) (s : State) (acts : List Action) : IO State :=
  acts.foldlM (init := s) fun s a => match a with
    | .reserve => do
      let (s, more) := step cfg s (.reserved (← w.reserve))
      more.forM (emit w)
      pure s
    | a => do emit w a; pure s

/-- The next event of a session in state `s`. -/
def nextEvent (cfg : Config) (w : Core.World Out) (s : State) : IO Event := do
  match s.current with
  | none => pure (.tick (← w.clock))
  | some o =>
    match ← (w.link.readUntil (o.sendNs + cfg.limitNs + 1)).toBaseIO with
    | .ok none => pure (.tick (← w.clock))
    | .ok (some b) => if b.isEmpty then pure .closed else pure (.received (← w.clock) b)
    | .error _ => pure .closed

/-- At most `fuel` more events, then the end of the link. -/
def loop (cfg : Config) (w : Core.World Out) : Nat → State → IO State
  | fuel, s =>
    if s.phase == .done || s.phase == .signing then pure s else
    match fuel with
    | 0 => let (s, acts) := step cfg s .closed; perform cfg w s acts
    | fuel + 1 => do
      let (s, acts) := step cfg s (← nextEvent cfg w s)
      loop cfg w fuel (← perform cfg w s acts)

/-- Run one session over `w`: its last state. A refused reservation is in `State.refusal`. -/
def drive (cfg : Config) (w : Core.World Out) : IO State := do
  let s ← loop cfg w (1 + cfg.coins.k * (cfg.blockBytes + 2)) {}
  w.link.close
  pure s

end MemoryChallenger
