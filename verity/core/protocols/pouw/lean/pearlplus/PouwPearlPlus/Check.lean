import PouwPearlPlus.Load
import PouwPearlPlus.Tile
import PouwPearlPlus.Credit

/-!
# Pearl+'s run verdict (`verify_run.verify_passes`)

`check input opener` is the verdict on a served run's retained passes, in core's answer form: accept, or reject with
`verify_passes`'s reason (and its refusal (code, why) where it has one).

* **Input** (`Input`): the epoch's coin commitment, the retained passes' manifests (their records, the voluntary rows
  among them, and what each published), the number of draws, and the verifier's coin: a source, or its record of the
  coin it drew on this run's commitment (`CoinRecord`), which pins the source.
* **The prover's openings** (`opener pass tile`): each opened tile's rows, leaf and y rows with their paths. The verifier
  asks only for the tiles its draw names.
* **The verdict**: every pass loads (`load`), the run's commitment is the coin record's where there is one (`after-coin`),
  the draw over every pass's tickets names the drawn tiles, each pass adds one tile per excluded row, and the run accepts
  iff every one of them passes `checkTile`, or is over its inert-work bound with every other check passed under
  `over_cap` `uncredited` (the default; `reject` rejects such a run). A rejected tile's reason is `pass i tile t: why`,
  joined by `; `.
* **The work** an accepted run publishes: its credit at Pearl's assumption at δ = 1/100 (`creditAtAssumption`), at X_O,
  the number of draws on tiles over their bound, which are credited nothing.

Out of scope, refused by name in `load`: sm_120's schemes, live coins, the window record's tie (`tie`, `rounds`, `dropped`).
-/

namespace PouwCheck.PearlPlus

open Lean PouwCommit Words

/-- The rule on an opened tile over its inert-work bound (`verify_run`'s ``over_cap``, `served_passes.py verify
--over-cap`): `uncredited` credits it nothing and goes on (CMP-311 option A, the default); `reject` rejects the run. -/
inductive OverCap where
  | uncredited
  | reject
  deriving BEq, DecidableEq, Repr, Inhabited

def OverCap.name : OverCap → String
  | .uncredited => "uncredited"
  | .reject => "reject"

def OverCap.ofName? : String → Option OverCap
  | "uncredited" => some .uncredited
  | "reject" => some .reject
  | _ => none

structure Input where
  coins : CoinCommitment
  manifests : List Json
  draws : Nat
  source : Option ByteArray := none
  held : Option CoinRecord := none
  overCap : OverCap := .uncredited

/-- The tiles a verdict opens, once its passes load and its coin is fixed: `picks` the draw's, in its order. -/
structure Plan where
  passes : List Pass
  root : ByteArray
  source : ByteArray
  todo : List (Nat × Nat)
  drawn : Nat
  picks : List (Nat × Nat)

structure TileVerdict where
  pass : Nat
  tile : Nat
  outcome : TileOutcome

/-- The tile's reason: why it fails, or why it is over its inert-work bound. -/
def TileVerdict.reason (v : TileVerdict) : Option String :=
  match v.outcome with
  | .pass => none
  | .overCap why | .fail why => some why

/-- Whether the tile is accepted under `m`: it passes, or it is over its bound and `m` credits it nothing. -/
def TileVerdict.ok (m : OverCap) (v : TileVerdict) : Bool :=
  match v.outcome with
  | .pass => true
  | .overCap _ => m == .uncredited
  | .fail _ => false

/-- X_O: how many of the plan's draws landed on a tile over its inert-work bound. -/
def overCapDraws (pl : Plan) (vs : List TileVerdict) : Nat :=
  pl.picks.countP fun (i, tid) => (vs.find? fun v => v.pass == i && v.tile == tid).any fun v =>
    match v.outcome with
    | .overCap _ => true
    | _ => false

structure RunAnswer where
  accept : Bool
  reason : Option String := none
  refused : Option (String × String) := none
  plan : Option Plan := none
  verdicts : List TileVerdict := []
  work : Option Credit := none
  overCap : OverCap := .uncredited

/-- `verify_passes` up to the replay: the passes, the run's commitment, the source and the tiles to open. -/
def plan (i : Input) : Except Refusal Plan := do
  if i.manifests.isEmpty then throw { reason := "the run holds no pass with calls" }
  throwRefusal (passesDistinct i.manifests) fun _ => do
  let passes ← i.manifests.mapM (load i.coins)
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
  let key ← match drawKey source root with
    | .ok k => pure k
    | .error e => throw { reason := s!"ValueError: {e}" }
  let drawn ← match draw passes key i.draws with
    | .ok d => pure d
    | .error e => throw { reason := s!"ValueError: {e}" }
  pure ⟨passes, root, source, todo passes drawn, drawn.length, drawn⟩

/-- The verdict of every tile of `pl.todo`, in order, replayed in parallel: the fixed operand's noise once per pass and
weight, then each tile on it. -/
def replay (pl : Plan) (opener : Nat → Nat → Except String TileOpening) : List TileVerdict := Id.run do
  let mut noise : List ((Nat × Nat) × Task (Except String FixedNoise)) := []
  let mut tasks : Array (Task TileVerdict) := #[]
  for (i, tid) in pl.todo do
    let p := pl.passes[i]!
    let some (pos, _, _) := p.tile? tid
      | tasks := tasks.push (.pure ⟨i, tid, .fail "tile does not match the verifier's own enumeration"⟩)
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
      | .error e => ⟨i, tid, .fail s!"the prover's files don't open it: {e}"⟩
      | .ok o => ⟨i, tid, checkTile p wn tid o⟩
  return tasks.toList.map Task.get

/-- `verify_passes`: the run's verdict. -/
def check (i : Input) (opener : Nat → Nat → Except String TileOpening) : RunAnswer :=
  match plan i with
  | .error r => { accept := false, reason := some r.reason, refused := r.refused, overCap := i.overCap }
  | .ok pl =>
    let vs := replay pl opener
    let bad := vs.filterMap fun v =>
      if v.ok i.overCap then none else v.reason.map fun r => s!"pass {v.pass} tile {v.tile}: {r}"
    if bad.isEmpty then
      { accept := true, plan := some pl, verdicts := vs, overCap := i.overCap,
        work := some (creditAtAssumption pl.passes i.draws (overCapDraws pl vs) DELTA) }
    else { accept := false, reason := some ("; ".intercalate bad), plan := some pl, verdicts := vs, overCap := i.overCap }

end PouwCheck.PearlPlus
