module

public import PouwPearlPlus.Load
public import PouwPearlPlus.Tile
public import PouwPearlPlus.Tie
public import PouwPearlPlus.Credit

/-!
# Pearl+'s run verdict (`verify_run.verify_passes`)

`check input opener` is the verdict on a served run's retained passes, in core's answer form: accept, or reject with
`verify_passes`'s reason (and its refusal (code, why) where it has one).

* **Input** (`Input`): the epoch's coin commitment (an in-process challenger's or a live session's, `EpochCoins`), the
  retained passes' manifests (their records, the voluntary rows among them, and what each published), the number of
  draws, the verifier's coin: a source, or its record of the coin it drew on this run's commitment (`CoinRecord`), which
  pins the source, the window record W the passes are tied to where there is one, with the passes' rounds and the
  dropped rounds (the coin record's) and the window's class where the passes are to be its fixed load (`load`), and
  whether the verdict is to be of record (`ofRecord`).
* **The prover's openings** (`opener pass tile`): each opened tile's rows, leaf and y rows with their paths. The verifier
  asks only for the tiles its draw names.
* **The verdict**: every pass loads (`load`), all of one salt and, unless the input is a test instance's
  (`testInstance`), each at the served instance, `k_min` `K_MIN` and `TILE × TILE` tiles; the passes are W's and, given
  the window's class, its fixed load (`tieAt`: `tieOf` and `heldToLoad`, refused as Python's `ValueError`), the run's
  commitment is the coin record's where there is one (`after-coin`), the record's challenger draw is coin ``pouw/draw`` 0
  opened after it (`drawReason`, refused as `draw`), a verdict of record takes only a challenger draw on live coins
  (`ofRecordReason`, refused as `of-record`), the draw over every pass's tickets names the drawn tiles, each pass adds one
  tile per excluded row, and the run accepts iff every one of them passes `checkTile`, or is over its inert-work bound
  with every other check passed under `over_cap` `uncredited` (the default, and the run of record's, whose passes hold
  no voluntary row; `reject` rejects such a run). A rejected
  tile's reason is `pass i tile t: why`, joined by `; `.
* **The work** an accepted run publishes: its credit at Pearl's assumption at δ = 1/100 (`creditAtAssumption`), at X_O,
  the number of draws on tiles over their bound, which are credited nothing. A test instance's credits nothing
  (`credited` 0), as `--recorded`'s salt claims nothing.

Out of scope, refused by name in `load`: sm_120's schemes.
-/

@[expose] public section

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

/-- `pearl_c.TILE`: a served pass's tiles are `TILE × TILE`. -/
def TILE : Nat := 64

/-- `pearl_c.K_MIN`, Pearl's domain: a served pass's `k_min`. -/
def K_MIN : Nat := 1024

structure Input where
  coins : EpochCoins
  manifests : List Json
  draws : Nat
  source : Option ByteArray := none
  held : Option CoinRecord := none
  overCap : OverCap := .uncredited
  /-- The passes are a test instance's (`verify_passes`' ``test_instance``): their `k_min` and tile may be other than
  `K_MIN` and `TILE × TILE`, and the verdict credits nothing. -/
  testInstance : Bool := false
  tie : Option Json := none
  /-- The window's class (`verify_passes`' ``window_class``), read only with `tie`: passes that are not its fixed load are
  refused (`Tie.tieAt`). -/
  load : Option LoadClass := none
  rounds : Option (List Nat) := none
  dropped : List Nat := []
  ofRecord : Bool := false

/-- The tiles a verdict opens, once its passes load and its coin is fixed: `picks` the draw's, in its order. -/
structure Plan where
  passes : List Pass
  root : ByteArray
  source : ByteArray
  todo : List (Nat × Nat)
  drawn : Nat
  picks : List (Nat × Nat)
  tie : Option Tied := none

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
  pure { passes, root, source, todo := todo passes drawn, drawn := drawn.length, picks := drawn, tie }

/-- `verify_passes` up to the replay: the passes, the run's commitment, the source and the tiles to open. -/
def plan (i : Input) : Except Refusal Plan := do
  if i.manifests.isEmpty then throw { reason := "the run holds no pass with calls" }
  throwRefusal (passesDistinct i.manifests) fun _ => do
  let passes ← i.manifests.mapM (load i.coins)
  if passes.any (·.salt != passes[0]!.salt) then throw { reason := "ValueError: the passes are not all of one salt" }
  if !i.testInstance then
    match (passes.zip (List.range passes.length)).find? fun (p, _) =>
        p.scheme.kMin != K_MIN || p.scheme.tileM != TILE || p.scheme.tileN != TILE with
    | some (p, j) =>
      throw { reason := s!"ValueError: pass {j}'s instance is k_min {p.scheme.kMin} at {p.scheme.tileM} x " ++
        s!"{p.scheme.tileN} tiles, not the served {K_MIN} at {TILE} x {TILE}: " ++
        "a test instance is read only as one, crediting nothing" }
    | none => pure ()
  let tie ← match i.tie with
    | none => pure none
    | some w => match tieAt i.load w (i.manifests.zip passes) i.rounds i.dropped with
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

/-- The credit a verdict publishes: `c`, or, on a test instance's passes, `c` crediting nothing. -/
def testCredit (testInstance : Bool) (c : Credit) : Credit :=
  if testInstance then { c with credited := 0 } else c

/-- `verify_passes`: the run's verdict. -/
def check (i : Input) (opener : Nat → Nat → Except String TileOpening) : RunAnswer :=
  match plan i with
  | .error r => { accept := false, reason := some r.reason, refused := r.refused, overCap := i.overCap, tie := r.tie }
  | .ok pl =>
    let vs := replay pl opener
    let bad := vs.filterMap fun v =>
      if v.ok i.overCap then none else v.reason.map fun r => s!"pass {v.pass} tile {v.tile}: {r}"
    if bad.isEmpty then
      { accept := true, plan := some pl, verdicts := vs, overCap := i.overCap, tie := pl.tie,
        work := some (testCredit i.testInstance (creditAtAssumption pl.passes i.draws (overCapDraws pl vs) DELTA)) }
    else { accept := false, reason := some ("; ".intercalate bad), plan := some pl, verdicts := vs, overCap := i.overCap,
           tie := pl.tie }

end PouwCheck.PearlPlus
