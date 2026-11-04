import PouwPearlPlus.Open

/-!
# `pouw-pearlplus`: the run verdict from a served run's retained passes

    pouw-pearlplus --coins COINS.json --draws T (--source HEX | --record RECORD.json | both) MANIFEST...

prints the verdict as one JSON object: `verdict` (`accept` or `reject`), `reason`, `refused` (`[code, why]` or null),
`commitment` and `source` (hex, once the passes load and the coin is fixed), and `verdicts` (each opened tile's pass, id,
ok and reason, in the verdict's order).

The prover's side is `Open.opening` over the pass's files. A record that exists pins the source (`after-coin`); one that
doesn't is written with the run's commitment and the given source.
-/

open Lean PouwCommit PouwCheck.PearlPlus

structure Args where
  coins : Option String := none
  draws : Nat := 3
  source : Option String := none
  record : Option String := none
  manifests : List String := []

partial def parse : List String → Args → Except String Args
  | [], a => pure a
  | "--coins" :: v :: rest, a => parse rest { a with coins := some v }
  | "--draws" :: v :: rest, a => match v.toNat? with
    | some n => parse rest { a with draws := n }
    | none => throw s!"--draws {v}: an integer"
  | "--source" :: v :: rest, a => parse rest { a with source := some v }
  | "--record" :: v :: rest, a => parse rest { a with record := some v }
  | m :: rest, a => if m.startsWith "--" then throw s!"unknown option {m}" else parse rest { a with manifests := a.manifests ++ [m] }

def readJson (path : String) : IO Json := do
  match Json.parse (← IO.FS.readFile path) with
  | .ok j => pure j
  | .error e => throw (IO.userError s!"{path}: {e}")

def main (argv : List String) : IO UInt32 := do
  let a ← IO.ofExcept (parse argv {})
  let some coinsPath := a.coins | throw (IO.userError "--coins COINS.json is required")
  let coins ← match CoinCommitment.ofJson (← readJson coinsPath) with
    | .ok c => pure c
    | .error e => throw (IO.userError s!"{coinsPath}: {e}")
  let mans ← a.manifests.mapM readJson
  let source ← match a.source with
    | some s => match unhex s with
      | some b => pure (some b)
      | none => throw (IO.userError "--source: hex")
    | none => pure none
  let held ← match a.record with
    | some r => do
      if ← System.FilePath.pathExists r then
        let j ← readJson r
        match (str? j "commitment").bind unhex, (str? j "source").bind unhex with
        | some c, some s => pure (some ({ commitment := c, source := s } : CoinRecord))
        | _, _ => throw (IO.userError s!"{r}: a coin record has a commitment and a source")
      else pure none
    | none => pure none
  let input : Input := { coins, manifests := mans, draws := a.draws, source, held }
  let mut cache : Std.HashMap String ByteArray := {}
  let mut checked : Std.HashMap (String × String) (Except String ByteArray) := {}
  if let .ok pl := plan input then
    if let (some r, none) := (a.record, held) then
      IO.FS.writeFile r ((Json.mkObj [("schema", jstr COIN_SCHEMA), ("commitment", jstr (hex pl.root)),
        ("source", jstr (hex pl.source))]).pretty ++ "\n")
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
