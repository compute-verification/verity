import MemoryChallenger.Driver
import MemoryChallenger.Difftest

/-!
# The driver over an in-memory world

`drive` run on a `Core.World` whose clock, link and leaf state are a script, which `memory-challenger-drive` serves
behind the difftest's line protocol (`Difftest.serve`'s, with an `IO` function: `serve` takes pure ones). A request is
`{"fn": "drive", "input": {"config", "start", "rounds", "leaf", "stuck"}}`: the session's `Config` as `step`'s difftest reads it,
the clock's first reading, for each round the chunks the peer sends once it has read that round's challenge
(`{"at": ns, "bytes": hex}`, the clock reading when the chunk arrives), and the leaf state's answer (a leaf, or
`{"refused": reason}`). The response is the bytes the challenger sent, what it published, and whether it ended
unsigned.

The peer answers a round only after its challenge, and a round's last chunk completes its block (or the round's chunks
end short of it, and then the peer closes the link). The clock moves when a chunk arrives (to its `at`) and when a wait
on a deadline `t` passes it (to `t`): `readUntil t` returns the next chunk if it arrives before `t`, and otherwise moves
the clock to `t` and returns `none`, so a chunk at or past the deadline is never read. Nothing here blocks but `read`,
which waits for the next challenge.

A `stuck` world breaks `readUntil`'s contract: it returns `none` at the deadline without moving the clock, so the
driver's deadline `tick` never ends the round and only the driver's bound does (`MemoryChallenger.Driver`).
-/

namespace MemoryChallenger.MemoryWorld

open Lean (Json)

/-- The peer's side of the link: the challenges read so far, the chunks of the current round delivered, the clock. -/
structure Peer where
  revealed : Nat := 0
  delivered : Nat := 0
  now : Nat
  closed : Bool := false

/-- A world whose peer plays `rounds` and whose leaf state answers `leaf`, the bytes sent and outputs published. -/
def scripted (blockBytes start : Nat) (rounds : Array (Array (Nat × ByteArray))) (leaf : Except String Nat)
    (stuck : Bool := false) :
    IO (Core.World Out × IO.Ref (Array ByteArray) × IO.Ref (Array Out)) := do
  let peer ← IO.mkRef ({ now := start } : Peer)
  let wake ← IO.mkRef (← IO.Promise.new : IO.Promise Unit)
  let sent ← IO.mkRef (#[] : Array ByteArray)
  let published ← IO.mkRef (#[] : Array Out)
  -- What the link holds: `some (some b)` the next chunk (or the end of the link, empty), `some none` nothing before
  -- `deadline` (the clock moved to it), and `none` nothing until the next challenge (only with no deadline).
  let take (deadline : Option Nat) : IO (Option (Option ByteArray)) := peer.modifyGet fun p =>
    let idle : Option (Option ByteArray) × Peer := match deadline with
      | some t => (some none, if stuck then p else { p with now := max p.now t })
      | none => (none, p)
    let gone : Option (Option ByteArray) := some (some ByteArray.empty)
    if p.closed then (gone, p)
    else if p.revealed = 0 then idle
    else match rounds[p.revealed - 1]? with
      | none => (gone, p)
      | some cs =>
        if h : p.delivered < cs.size then
          if deadline.any (· ≤ cs[p.delivered].1) then idle
          else (some (some cs[p.delivered].2), { p with delivered := p.delivered + 1, now := cs[p.delivered].1 })
        else if cs.foldl (· + ·.2.size) 0 < blockBytes then (gone, p) else idle
  -- A read waits once, for the next write or the close, and may see one stale wake first: three tries always suffice.
  let rec read (tries : Nat) : IO ByteArray := do
    let w ← wake.get
    let next := (← take none).join
    match next, tries with
    | some b, _ => pure b
    | none, 0 => pure .empty
    | none, tries + 1 => do let _ ← IO.wait w.result?; read tries
  let world : Core.World Out := {
    coins := Core.World.coinsOS
    clock := do pure (← peer.get).now
    sleepUntil := fun t => peer.modify fun p => { p with now := max p.now t }
    link := {
      read := read 3
      readUntil := fun t => do pure (← take (some t)).join
      write := fun b => do
        sent.modify (·.push b)
        peer.modify fun p => { p with revealed := p.revealed + 1, delivered := 0 }
        let w ← wake.get
        wake.set (← IO.Promise.new)
        w.resolve ()
      backlog := pure 0
      close := do
        peer.modify ({ · with closed := true })
        (← wake.get).resolve () }
    connect := pure (.error "no upstream")
    publish := fun o => published.modify (·.push o)
    reserve := pure leaf }
  pure (world, sent, published)

private def hex (j : Json) (k : String) : Except String ByteArray := do Flock.unhex (← Flock.J.str j k "input")

def driveFn (j : Json) : IO (Except String Json) := do
  let parsed : Except String _ := do
    let cfg ← MemoryChallenger.Difftest.configOf (← Flock.J.get j "config" "input")
    let rounds ← (← Flock.J.arr j "rounds" "input").mapM fun r => do
      (← Flock.J.asArr r "rounds").mapM fun c => do pure ((← Flock.J.nat c "at" "chunk"), ← hex c "bytes")
    let leaf ← match (← Flock.J.get j "leaf" "input").getObjValAs? String "refused" with
      | .ok e => pure (.error e)
      | .error _ => pure (.ok (← Flock.J.nat j "leaf" "input"))
    pure (cfg, ← Flock.J.nat j "start" "input", rounds, leaf, ← Flock.J.bool j "stuck" "input")
  match parsed with
  | .error e => pure (.error e)
  | .ok (cfg, start, rounds, leaf, stuck) =>
    let (world, sent, published) ← scripted cfg.blockBytes start rounds leaf stuck
    let s ← drive cfg world
    pure (.ok (Json.mkObj [("sent", .arr ((← sent.get).map fun b => .str (Flock.hexOf b))),
      ("published", .arr ((← published.get).map MemoryChallenger.Difftest.outJ)), ("refused", .bool s.refusal.isSome)]))

end MemoryChallenger.MemoryWorld
