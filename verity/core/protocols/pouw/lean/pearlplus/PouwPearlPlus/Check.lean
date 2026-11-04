import PouwPearlPlus.Load
import PouwPearlPlus.Tile
import PouwPearlPlus.Tie
import PouwPearlPlus.Credit

/-!
# Pearl+'s run verdict (`verify_run.verify_passes`)

`check input opener` is the verdict on a served run's retained passes, in core's answer form: accept, or reject with
`verify_passes`'s reason (and its refusal (code, why) where it has one).

* **Input** (`Input`): the epoch's coin commitment (an in-process challenger's or a live session's, `EpochCoins`), the
  retained passes' manifests (their records, the voluntary rows among them, and what each published), the number of
  draws, the verifier's coin: a source, or its record of the coin it drew on this run's commitment (`CoinRecord`), which
  pins the source, the window record W the passes are tied to where there is one, with the passes' rounds and the
  dropped rounds (the coin record's), and whether the verdict is to be of record (`ofRecord`).
* **The prover's openings** (`opener pass tile`): each opened tile's rows, leaf and y rows with their paths. The verifier
  asks only for the tiles its draw names.
* **The verdict**: every pass loads (`load`), the passes are W's (`tieOf`, refused as its `ValueError`), the run's
  commitment is the coin record's where there is one (`after-coin`), the record's challenger draw is coin ``pouw/draw`` 0
  opened after it (`drawReason`, refused as `draw`), a verdict of record takes only a challenger draw on live coins
  (`ofRecordReason`, refused as `of-record`), the draw over every pass's tickets names the drawn tiles, each pass adds one tile per excluded row, and the run accepts
  iff every one of them passes `checkTile`. A failing tile's reason is `pass i tile t: why`, joined by `; `.
* **The work** an accepted run publishes: its credit at Pearl's assumption at δ = 1/100 (`creditAtAssumption`).

Out of scope, refused by name in `load`: sm_120's schemes.
-/

namespace PouwCheck.PearlPlus

open Lean PouwCommit Words

structure Input where
  coins : EpochCoins
  manifests : List Json
  draws : Nat
  source : Option ByteArray := none
  held : Option CoinRecord := none
  tie : Option Json := none
  rounds : Option (List Nat) := none
  dropped : List Nat := []
  ofRecord : Bool := false

/-- The tiles a verdict opens, once its passes load and its coin is fixed. -/
structure Plan where
  passes : List Pass
  root : ByteArray
  source : ByteArray
  todo : List (Nat × Nat)
  drawn : Nat
  tie : Option Tied := none

structure TileVerdict where
  pass : Nat
  tile : Nat
  reason : Option String

structure RunAnswer where
  accept : Bool
  reason : Option String := none
  refused : Option (String × String) := none
  plan : Option Plan := none
  verdicts : List TileVerdict := []
  work : Option Credit := none
  tie : Option Tied := none

/-- `verify_passes` after the tie: the record, the source, the draw's holds and the tiles to open. -/
def afterTie (i : Input) (passes : List Pass) (tie : Option Tied) : Except Refusal Plan := do
  let root := runCommitment passes
  let source ← match i.held with
    | some h =>
      if h.commitment != root || (i.source.isSome && i.source != some h.source) then
        throw { refused := some AFTER_COIN,
                reason := s!"{AFTER_COIN.1}: {AFTER_COIN.2} (its coin was drawn on run {(hex h.commitment).take 16}, " ++
                  s!"these passes are run {(hex root).take 16})" }
      pure h.source
    | none => match i.source with
      | some s => pure s
      | none => throw { reason := "the verifier's coin: a source, or its record of the coin" }
  if let some h := i.held then
    if let some why := drawReason h root then
      throw { refused := some (DRAW_REFUSED, why), reason := s!"{DRAW_REFUSED}: {why}" }
  if i.ofRecord then
    if let some why := ofRecordReason i.coins i.source i.held then
      throw { refused := some (NOT_OF_RECORD, why), reason := s!"{NOT_OF_RECORD}: {why}" }
  let key ← match drawKey source root with
    | .ok k => pure k
    | .error e => throw { reason := s!"ValueError: {e}" }
  let drawn ← match draw passes key i.draws with
    | .ok d => pure d
    | .error e => throw { reason := s!"ValueError: {e}" }
  pure { passes, root, source, todo := todo passes drawn, drawn := drawn.length, tie }

/-- `verify_passes` up to the replay: the passes, the run's commitment, the source and the tiles to open. -/
def plan (i : Input) : Except Refusal Plan := do
  if i.manifests.isEmpty then throw { reason := "the run holds no pass with calls" }
  throwRefusal (passesDistinct i.manifests) fun _ => do
  let passes ← i.manifests.mapM (load i.coins)
  let tie ← match i.tie with
    | none => pure none
    | some w => match tieOf w (i.manifests.zip passes) i.rounds i.dropped with
      | .ok t => pure (some t)
      | .error e => throw { reason := s!"ValueError: {e}" }
  tryCatch (afterTie i passes tie) fun r => throw { r with tie := r.tie <|> tie }

/-- The verdict of every tile of `pl.todo`, in order, replayed in parallel: the fixed operand's noise once per pass and
weight, then each tile on it. -/
def replay (pl : Plan) (opener : Nat → Nat → Except String TileOpening) : List TileVerdict := Id.run do
  let mut noise : List ((Nat × Nat) × Task (Except String FixedNoise)) := []
  let mut tasks : Array (Task TileVerdict) := #[]
  for (i, tid) in pl.todo do
    let p := pl.passes[i]!
    let some (pos, _, _) := p.tile? tid
      | tasks := tasks.push (.pure ⟨i, tid, some "tile does not match the verifier's own enumeration"⟩)
    let u := p.records[pos]!.matmul
    let t ← match noise.lookup (i, u.weight) with
      | some t => pure t
      | none =>
        let rootB := (p.weightRoots.lookup u.weight).getD ByteArray.empty
        let t := Task.spawn fun _ => fixedNoise p.salt rootB u.k
        noise := ((i, u.weight), t) :: noise
        pure t
    tasks := tasks.push <| t.map fun wn =>
      match opener i tid with
      | .error e => ⟨i, tid, some s!"the prover's files don't open it: {e}"⟩
      | .ok o => ⟨i, tid, checkTile p wn tid o⟩
  return tasks.toList.map Task.get

/-- `verify_passes`: the run's verdict. -/
def check (i : Input) (opener : Nat → Nat → Except String TileOpening) : RunAnswer :=
  match plan i with
  | .error r => { accept := false, reason := some r.reason, refused := r.refused, tie := r.tie }
  | .ok pl =>
    let vs := replay pl opener
    let bad := vs.filterMap fun v => v.reason.map fun r => s!"pass {v.pass} tile {v.tile}: {r}"
    if bad.isEmpty then
      { accept := true, plan := some pl, verdicts := vs, work := some (creditAtAssumption pl.passes i.draws DELTA),
        tie := pl.tie }
    else { accept := false, reason := some ("; ".intercalate bad), plan := some pl, verdicts := vs, tie := pl.tie }

end PouwCheck.PearlPlus
