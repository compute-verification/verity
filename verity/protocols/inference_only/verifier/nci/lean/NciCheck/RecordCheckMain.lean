module

public import NciCheck.RecordCount

/-! `nci-record-check CASE`: the record form's two verdicts on one case, a line of the record difftest's cases
(`{instance, trusted, count}`; `scripts/served_record_case.py` writes one from a served run's files). It prints
`Record.checkDetail`, the rate rule's whole verdict, on a line `[S s] check: JSON`, then `RecordCount.checkDetail`,
the count's, on a line `[S s] count: JSON`, each with the seconds since it started. It runs the form once for each,
where the difftest runs it once for every result it compares. Both verdicts are for the private record and are
never published. It checks the record against the Program and the registration log, as the form does, and nothing
the form does not: not the values the record commits, nor SPOT's openings of its rows. -/

@[expose] public section

def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: nci-record-check CASE"; return 2
  let t0 ← IO.monoMsNow
  let said (what : String) : IO Unit := do
    IO.println s!"[{((← IO.monoMsNow) - t0) / 1000} s] {what}"
    (← IO.getStdout).flush
  let .ok c := Lean.Json.parse (← IO.FS.readFile path)
    | IO.eprintln s!"nci-record-check: {path} is not JSON"; return 2
  let inst := c.getObjValD "instance"
  let tr := c.getObjValD "trusted"
  let limits := c.getObjValD "count"
  said "read the case"
  said s!"check: {(NciCheck.Record.checkDetail inst tr).compress}"
  said s!"count: {(NciCheck.RecordCount.checkDetail inst tr limits).compress}"
  return 0
