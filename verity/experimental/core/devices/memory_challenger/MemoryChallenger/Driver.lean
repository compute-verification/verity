import World
import MemoryChallenger.Step

/-!
# The driver

`step` run over a `Core.World Out`: the only IO of the memory challenger. It needs only an order: it reads the clock for
a challenge before it sends the challenge (`tick`, then `send`), and for an answer after its bytes arrive (`read`, then
`received`).

`World`'s link has no read with a deadline, so while a round is open the driver waits on a read and on `sleepUntil` of
the round's deadline at once (`IO.waitAny`), each in a dedicated thread; a read that finishes first is the next event,
and a sleep that does is a `tick` past the deadline. The read stays outstanding across rounds: bytes it returns with no
round open are dropped by `step`, after any that had already arrived before a challenge.
-/

namespace MemoryChallenger

/-- The next link event: its bytes and the clock read after them, or the end of the link (an empty read or an error). -/
private def linkEvent (w : Core.World Out) (r : Except IO.Error ByteArray) : IO Event := do
  match r with
  | .ok b => if b.isEmpty then pure .closed else pure (.received (← w.clock) b)
  | .error _ => pure .closed

private def read (w : Core.World Out) : BaseIO (Task (Except IO.Error ByteArray)) := IO.asTask w.link.read .dedicated

/-- Run one session over `w`: its last state. A refused reservation is in `State.refusal`. -/
partial def drive (cfg : Config) (w : Core.World Out) : IO State := do
  let rec perform (s : State) : List Action → IO State
    | [] => pure s
    | .send b :: rest => do w.link.write b; perform s rest
    | .publish o :: rest => do w.publish o; perform s rest
    | .reserve :: rest => do
      let (s, acts) := step cfg s (.reserved (← w.reserve))
      let s ← perform s acts
      perform s rest
  -- `sleeper` waits for the deadline of round `j`, the round open when it started.
  let rec loop (s : State) (reader : Task (Except IO.Error ByteArray)) (sleeper : Option (Nat × Task (Except IO.Error Unit))) :
      IO State := do
    if s.phase == .done || s.phase == .signing then return s
    if ← IO.hasFinished reader then
      let (s, acts) := step cfg s (← linkEvent w reader.get)
      return ← loop (← perform s acts) (← read w) sleeper
    match s.current with
    | none =>
      let (s, acts) := step cfg s (.tick (← w.clock))
      loop (← perform s acts) reader none
    | some o =>
      let sleeper ← match sleeper with
        | some (j, t) => if j == o.j then pure t else IO.asTask (w.sleepUntil (o.sendNs + cfg.limitNs + 1)) .dedicated
        | none => IO.asTask (w.sleepUntil (o.sendNs + cfg.limitNs + 1)) .dedicated
      match ← IO.waitAny [reader.map (some ·), sleeper.map fun _ => none] with
      | some r =>
        let (s, acts) := step cfg s (← linkEvent w r)
        loop (← perform s acts) (← read w) (some (o.j, sleeper))
      | none =>
        let (s, acts) := step cfg s (.tick (← w.clock))
        loop (← perform s acts) reader none
  let s ← loop {} (← read w) none
  w.link.close
  pure s

end MemoryChallenger
