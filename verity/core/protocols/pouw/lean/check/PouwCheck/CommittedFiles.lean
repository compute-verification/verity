import PouwCheck.Verified

/-!
# A period's committed circuit files (`credit.committed_files`)

The circuit files a period's open committed, which `credit.period_credit` verifies each window's statements and live
statements on (`credit._lean_committed_files`, through `pouw-check-difftest`'s `credit-committed-files`), with
`credit.committed_files` the Python reference the difftest (`tests/test_live_verified_difftest.py`) holds to it.
-/

namespace PouwCheck.CommittedFiles

open Lean (Json)
open PouwCheck.Credit
open PouwCheck.Verified (CIRCUIT_FILES)

/-- `credit._TEMPLATE`: a committed circuit file's template, `k<K>` (K a positive integer, no leading zero) or `live`. -/
def template? (x : String) : Bool :=
  x == LIVE || match x.toList with
    | 'k' :: d :: ds => d != '0' && (d :: ds).all Char.isDigit
    | _ => false

def noFiles (name : String) : String :=
  s!"period {name}'s open commits no circuit files ({CIRCUIT_FILES}, each K's template k<K>, and the live " ++
  s!"Program's template {LIVE}, to the SHA-512 of the circuit file the auditor's lowering built for it), so no " ++
  "window's statements can be verified: commit them in the period-open, beside its circuit and K"

/-- `credit.committed_files`: the circuit files period `name`'s open committed (`CIRCUIT_FILES`, `{k<K>: SHA-512,
live: SHA-512}`: each template's circuit file, as the auditor's lowering built it) in the auditor's ledger `ledger`,
one `period` has trusted; else the refusal (`circuit`). -/
def committedFiles (ledger : Json) (name : String) : R Json := do
  let .arr lines := ledger | throw ("period", "the period is not named")
  let body := ((lines.filterMap SampledProofs.Challenger.ledgerEntry?).find? fun e =>
    e.kind == "period-open" && get e.body "period" == Json.str name).map (·.body)
  let files := (body.map (get · CIRCUIT_FILES)).getD .null
  unless isObj files && !(fields files).isEmpty
      && (fields files).all (fun (x, v) => template? x && (digest? v).isSome) do
    throw ("circuit", noFiles name)
  pure files

/-- `credit-committed-files`: on `{ledger, name}`, `{"files": …}` (`committedFiles`) or `{"refused": [code, why]}`. -/
def filesOf (j : Json) : R Json := do
  let some name := str? (get j "name") | throw ("error", "credit-committed-files: the period's name is not a string")
  pure (Json.mkObj [("files", ← committedFiles (get j "ledger") name)])

end PouwCheck.CommittedFiles
