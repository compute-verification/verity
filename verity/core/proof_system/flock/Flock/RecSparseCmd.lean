import Flock.HmRow
import Flock.RecSparse

/-!
`flock-rec-sparse`'s command: the verifier's sparse-phase unit (`Flock.RecSparse`), which `test_lean_rec_sparse` holds to
`rec_sparse.lowering`.

    flock-rec-sparse net NAME                the unit named `rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>` as text
    flock-rec-sparse check CIRCUIT NAME ...  `RecSparse.check` on the file's unit and on it renamed to each NAME

`net` prints `RecSparse.unitLaid` as `flock-ir-unit/v2` text (`ir_lower.netlist`'s form). `check` parses the file as
`verity/flock-circuit` and prints one `{"unit", "ok"}` line for each name, with `"error"` when refused.
-/

namespace Flock.RecSparseCmd

open Lean

def report (unit : String) (res : V Unit) : IO Unit :=
  let err := match res with
    | .ok () => []
    | .error e => [("error", Json.str e)]
  IO.println (Json.mkObj ([("unit", Json.str unit), ("ok", Json.bool res.toBool)] ++ err)).compress

def cmd (args : List String) : IO UInt32 := do
  match args with
  | ["net", name] =>
    match RecSparse.params name with
    | some sh =>
      if !sh.valid then
        IO.eprintln s!"{name}: not a valid rec-sparse shape"
        return 2
      IO.print (RecSparse.netlist name (RecSparse.unitLaid sh))
      return 0
    | none =>
      IO.eprintln s!"{name} is not rec-sparse/n<NU>-e<ELL>-s<S_LOG>-t<T_LOG>"
      return 2
  | "check" :: path :: names =>
    match HmRow.parse Tags.circuit (← IO.FS.readBinFile path) [] with
    | .error e =>
      report "" (.error e)
      return 1
    | .ok c =>
      let mut rc : UInt32 := 0
      for nm in c.unit.name :: names do
        let res := RecSparse.check { c with nets := c.nets.modify c.unitNet fun (k, n) => (k, { n with name := nm }) }
        report nm res
        if !res.toBool then rc := 1
      return rc
  | _ =>
    IO.eprintln "usage: flock-rec-sparse (net NAME | check CIRCUIT NAME ...)"
    return 2

end Flock.RecSparseCmd
