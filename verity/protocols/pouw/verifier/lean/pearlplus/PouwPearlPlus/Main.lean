module

public import PouwPearlPlus.Open

/-!
# `pouw-pearlplus`: the run verdict from a served run's retained passes

    pouw-pearlplus --coins COINS.json --draws T (--source HEX | --record RECORD.json | both)
      [--over-cap uncredited|reject] [--test-instance] [--tie W.json [--class SLOTS,PROMPT,OUTPUTS]] [--of-record]
      MANIFEST...

prints the verdict as one JSON object: `verdict` (`accept` or `reject`), `reason`, `refused` (`[code, why]` or null),
`commitment` and `source` (hex, once the passes load and the coin is fixed), `over_cap` (the rule on a tile over its
inert-work bound, `uncredited` unless `--over-cap` says otherwise), `verdicts` (each opened tile's pass, id, ok and
reason, in the verdict's order; a tile over its bound has its reason, and is ok under `uncredited`), `work`, and `tie`
(`{record, tables}` once the passes are held to W, with `weights` `WEIGHTS_UNTIED` for a version-1 W and `load`
`[slots, prompt, outputs]` where `--class` held them to that class's fixed load, or null).
`--test-instance` reads passes whose tile is other than 64 × 64 (a test instance's), and their work credits nothing.

`COINS.json` is the epoch's coin commitment, an in-process challenger's or a live session's (`EpochCoins.ofJson`). The
prover's side is `Open.opening` over the pass's files. A record that exists pins the source (`after-coin`) and names its
draw, and its ``rounds`` and ``dropped`` place the passes for the tie; one that doesn't is written with the run's
commitment, the given source and the node's draw. `--tie W.json` holds the passes to the window record W (`Tie.tieOf`),
`--class SLOTS,PROMPT,OUTPUTS`, read only beside it, refuses passes that are not that window class's fixed load
(`Tie.tieAt`), and `--of-record` takes only a challenger draw on live coins (`Coins.ofRecordReason`).
-/

@[expose] public section

open Lean PouwCommit PouwCheck.PearlPlus

structure Args where
  coins : Option String := none
  draws : Nat := 3
  source : Option String := none
  record : Option String := none
  overCap : OverCap := .uncredited
  testInstance : Bool := false
  tie : Option String := none
  load : Option LoadClass := none
  ofRecord : Bool := false
  manifests : List String := []

/-- `--class`'s `SLOTS,PROMPT,OUTPUTS`, each a positive integer. -/
def loadClass? (v : String) : Option LoadClass :=
  match (v.splitOn ",").map String.toNat? with
  | [some s, some p, some o] => if s == 0 || p == 0 || o == 0 then none else some ⟨s, p, o⟩
  | _ => none

def parse : List String → Args → Except String Args
  | [], a => pure a
  | "--coins" :: v :: rest, a => parse rest { a with coins := some v }
  | "--draws" :: v :: rest, a => match v.toNat? with
    | some n => parse rest { a with draws := n }
    | none => throw s!"--draws {v}: an integer"
  | "--source" :: v :: rest, a => parse rest { a with source := some v }
  | "--record" :: v :: rest, a => parse rest { a with record := some v }
  | "--over-cap" :: v :: rest, a => match OverCap.ofName? v with
    | some m => parse rest { a with overCap := m }
    | none => throw s!"--over-cap {v}: uncredited or reject"
  | "--test-instance" :: rest, a => parse rest { a with testInstance := true }
  | "--tie" :: v :: rest, a => parse rest { a with tie := some v }
  | "--class" :: v :: rest, a => match loadClass? v with
    | some c => parse rest { a with load := some c }
    | none => throw s!"--class {v}: SLOTS,PROMPT,OUTPUTS, each a positive integer"
  | "--of-record" :: rest, a => parse rest { a with ofRecord := true }
  | m :: rest, a => if m.startsWith "--" then throw s!"unknown option {m}" else parse rest { a with manifests := a.manifests ++ [m] }

def readJson (path : String) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"{path}: {e}")

def main (argv : List String) : IO UInt32 := do
  let a ← IO.ofExcept (parse argv {})
  let some coinsPath := a.coins | throw (IO.userError "--coins COINS.json is required")
  if a.load.isSome && a.tie.isNone then
    throw (IO.userError "--class holds the passes --tie W ties to its class's fixed load: it is read only beside --tie")
  let coins ← match EpochCoins.ofJson (← readJson coinsPath) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"{coinsPath}: {e}")
  let mans ← a.manifests.mapM readJson
  let source ← match a.source with
    | some s => match unhex s with
      | some b => pure (some b)
      | none => throw (IO.userError "--source: hex")
    | none => pure none
  let mut rounds : Option (List Nat) := none
  let mut dropped : List Nat := []
  let mut held : Option CoinRecord := none
  if let some r := a.record then
    if ← System.FilePath.pathExists r then
      let j ← readJson r
      let (some c, some s) := ((str? j "commitment").bind unhex, (str? j "source").bind unhex)
        | throw (IO.userError s!"{r}: a coin record has a commitment and a source")
      held := some { commitment := c, source := s, schema := (str? j "schema").getD "", draw := (get? j "draw").getD .null }
      let rows (k : String) : IO (Option (List Nat)) := match get? j k with
        | none => pure none
        | some (.arr xs) => match xs.toList.mapM natOf? with
          | some ns => pure (some ns)
          | none => throw (IO.userError s!"{r}: a coin record's {k} are integers")
        | some _ => throw (IO.userError s!"{r}: a coin record's {k} are a list")
      rounds ← rows "rounds"
      dropped := (← rows "dropped").getD []
  let tie ← a.tie.mapM readJson
  let input : Input := { coins, manifests := mans, draws := a.draws, source, held, overCap := a.overCap,
                         testInstance := a.testInstance, tie, load := a.load, rounds, dropped, ofRecord := a.ofRecord }
  let mut cache : Std.HashMap String ByteArray := {}
  let mut checked : Std.HashMap (String × String) (Except String ByteArray) := {}
  if let .ok pl := plan input then
    if let (some r, none) := (a.record, held) then
      IO.FS.writeFile r ((Json.mkObj [("schema", jstr COIN_SCHEMA), ("commitment", jstr (hex pl.root)),
        ("source", jstr (hex pl.source)), ("draw", nodeDraw)]).pretty ++ "\n")
    for (i, tid) in pl.todo do
      let dir := (System.FilePath.mk a.manifests[i]!).parent.getD "."
      for spec in filesOf mans[i]! pl.passes[i]! tid do
        let some f := str? spec "path" | continue
        let full := (dir / f).toString
        if !cache.contains full then
          try cache := cache.insert full (← IO.FS.readBinFile full) catch _ => pure ()
        if !checked.contains (full, spec.compress) then
          checked := checked.insert (full, spec.compress) (match cache.get? full with
            | some b => PouwCheck.PearlPlus.held spec b
            | none => throw s!"FileNotFoundError: {f}")
  let files := checked
  let pl? := (plan input).toOption
  let opener (i tid : Nat) : Except String TileOpening :=
    match pl? with
    | none => throw "no plan"
    | some pl =>
      let dir := (System.FilePath.mk a.manifests[i]!).parent.getD "."
      opening mans[i]! pl.passes[i]! (fun spec =>
        let f := (str? spec "path").getD ""
        (files.get? ((dir / f).toString, spec.compress)).getD (throw s!"FileNotFoundError: {f}")) tid
  let ans := check input opener
  IO.println (output ans).compress
  return if ans.accept then 0 else 1
