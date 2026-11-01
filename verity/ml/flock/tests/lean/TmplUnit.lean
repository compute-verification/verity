import Flock

/-!
The prover's side of a templated statement: its unit's rows at the positions the verifier folds them at. It
unrolls the unit (`Tmpl.posMap`), which no verifier of record may do (`tests/test_verifier_cost.py`), so it is its own
executable, built beside `flock-verify` by `pod/90-tmpl-unit.sh` (`lean_exe flock-tmpl-unit`, root `TmplUnit`).

    flock-tmpl-unit TMPL --layout MODE [--out F]

A `flock-ir-tmpl/v2` template read at layout `MODE` (`Tmpl.modeOf`) as a templated statement's META reads it
(`TmplUnit.read`). One `TMPL` line: the logical and committed sizes and the slot's log. With `--out`, the committed net's
`flock-ir-unit/v2` rows, then an `ORDER` line, `Tmpl.posMap`: the prover's unit, committed where the verifier folds it
(`flock-circuit --tmpl-unit F`).
-/

open Flock

def argOf (args : List String) (k : String) : Option String :=
  match args.dropWhile (· != k) with
  | _ :: v :: _ => some v
  | _ => none

def main (args : List String) : IO UInt32 := do
  let (some path, some m) := (args.head?, argOf args "--layout")
    | IO.eprintln "flock-tmpl-unit TMPL --layout MODE [--out F]"; return 2
  let some mode := Tmpl.modeOf m | IO.eprintln s!"flock-tmpl-unit: {m} is not a layout mode"; return 2
  let lines := ((← IO.FS.readFile path).splitOn "\n").toArray.filter (· ≠ "")
  let line ← match Tmpl.parse lines >>= fun t => TmplUnit.read t mode with
    | .error e => pure (Lean.Json.mkObj [("ok", Lean.Json.bool false), ("error", Lean.Json.str e)])
    | .ok tu =>
      let pos := Tmpl.posMap tu.t tu.lays
      if let some out := argOf args "--out" then
        let rows := (List.range tu.net.useful).map (Tmpl.rowLine tu.net)
        IO.FS.writeFile out ("\n".intercalate (Tmpl.headerLine tu.net :: rows ++ [s!"ORDER {(Lean.toJson pos).compress}"]) ++ "\n")
      pure (Lean.Json.mkObj [("ok", Lean.Json.bool true), ("name", Lean.Json.str tu.net.name), ("logical", Lean.toJson pos.size),
        ("useful", Lean.toJson tu.net.useful), ("unit_log", Lean.toJson tu.net.unitLog)])
  IO.println s!"TMPL {line.compress}"
  return (if (line.getObjValAs? Bool "ok").toOption.getD false then 0 else 1)
