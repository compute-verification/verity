/-
`flock-verify rec-compose`: a recursive statement's verdict from its V* parts' verdicts. Each part is one
`flock-verify verify --zk` process at one V* statement (`--rec-part J`), and each names in its `VERDICT` lines the
SHA-512 of the shared rows file it checked its row ports against (`rec_rows_sha512`, `--rec-rows`). The shared rows are
one set, which the prover stages once, after the inner rounds and before any V* session: its running sums' `rec-acc`
registration (`acc0`), the tops' `rec-top-L<l>` and the message rows' `rec-c<k>`. No part sees another, so only the
composition can hold every part to one set. It accepts exactly when there is one log per V* statement, `rec_part` `0`
to `n − 1` each once, every log has a `VERDICT`, every `VERDICT` accepts, and each names the digest of the one staged
rows file.
-/
import Lean.Data.Json

namespace Flock
namespace RecCompose

open Lean (Json)

/-- A log's `VERDICT` lines, each a JSON object, in order. -/
def verdicts (log : String) : Except String (Array Json) :=
  (log.splitOn "\n").foldlM (init := #[]) fun out line =>
    if line.startsWith "VERDICT " then
      match Json.parse (line.drop 8).toString with
      | .ok j => pure (out.push j)
      | .error e => throw s!"a VERDICT line that is not JSON: {e}"
    else pure out

/-- **One part's log**: it has a `VERDICT`, every `VERDICT` accepts at one `rec_part`, and each names `digest` as its
`rec_rows_sha512`. Its part. -/
def checkPart (digest : String) (name log : String) : Except String Nat := do
  let vs ← (verdicts log).mapError (s!"{name}: " ++ ·)
  let some v0 := vs[0]? | throw s!"{name}: no VERDICT"
  let some j := (v0.getObjValAs? Nat "rec_part").toOption
    | throw s!"{name}: its VERDICT names no rec_part: not one of V*'s parts (--rec-part J)"
  for v in vs do
    if (v.getObjValAs? Bool "accepted").toOption != some true then
      throw s!"{name}: a session is not accepted ({(v.getObjValAs? String "why").toOption.getD ""})"
    if (v.getObjValAs? Nat "rec_part").toOption != some j then
      throw s!"{name}: its VERDICTs name more than one rec_part"
    match (v.getObjValAs? String "rec_rows_sha512").toOption with
    | none => throw s!"{name}: its VERDICT names no rec_rows_sha512: it was not checked against the shared rows (--rec-rows)"
    | some h =>
      if h != digest then
        throw s!"{name}: checked against the rows file {h}, not the one staged rows file {digest}"
  pure j

/-- **The composition**: `parts`' logs (name, text), one per V* statement of `n`, each accepted at the one staged rows
file of SHA-512 `digest` (`checkPart`). -/
def check (n : Nat) (digest : String) (parts : List (String × String)) : Except String Unit := do
  if n == 0 then throw "no V* statement"
  let js ← parts.mapM fun (name, log) => checkPart digest name log
  for j in js do
    if j ≥ n then throw s!"rec_part {j}: V* has {n} statements"
    if js.count j != 1 then throw s!"rec_part {j}: {js.count j} logs, not one"
  if js.length != n then throw s!"{js.length} parts, not V*'s {n} statements"

end RecCompose
end Flock
