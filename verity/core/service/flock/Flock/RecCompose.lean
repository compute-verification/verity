/-
`flock-verify rec-compose`: a recursive statement's verdict from its V* parts' verdicts. Each part is one
`flock-verify verify --zk` process at one V* statement (`--rec-part J`), and each names in its `VERDICT` lines the
SHA-512 of the shared rows file it checked its row ports against (`rec_rows_sha512`, `--rec-rows`), of the verifier's
own inner shape file it read (`rec_inner_sha512`, `--rec-inner`), and of the inner session's record and the coin
server's transcript of it that its rows and coins were held to (`rec_inner_session_sha512`, `--rec-inner-session`'s
`session.json`; `rec_inner_transcript_sha512`, `--rec-inner-transcript`). The shared rows are one set, which the prover
stages once, after the inner rounds and before any V* session: its running sums' `rec-acc` registration (`acc0`), the
tops' `rec-top-L<l>` and the message rows' `rec-c<k>`. The inner shape is one too: the compiled statement holds one,
which gives V*'s statements; and so is the inner session, which the prover staged every V* statement from. No part sees
another, so only the composition can hold every part to one set, one shape and one inner session. It accepts exactly
when there is one log per V* statement, `rec_part` `0` to `n − 1` each once, every log has a `VERDICT`, every `VERDICT`
accepts, and each names the digest of the one staged rows file, of the composition's own inner shape file, and of the
first part's inner session and transcript.
-/
import Lean.Data.Json
import Flock.JsonParse

namespace Flock
namespace RecCompose

open Lean (Json)

/-- A log's `VERDICT` lines, each a JSON object, in order. -/
def verdicts (log : String) : Except String (Array Json) :=
  (log.splitOn "\n").foldlM (init := #[]) fun out line =>
    if line.startsWith "VERDICT " then
      match JsonParse.parse (line.drop 8).toString with
      | .ok j => pure (out.push j)
      | .error e => throw s!"a VERDICT line that is not JSON: {e}"
    else pure out

/-- A digest every `VERDICT` of every part names: at `key` (`verify` names it when given `flag`) the SHA-512 of the
`what` it read, which must be `digest`, `whose`. -/
structure Pin where
  key : String
  flag : String
  what : String
  whose : String
  digest : String

/-- The digest `v`, a `VERDICT` of the part `name`, names at `p`'s key. -/
def named (name : String) (v : Json) (p : Pin) : Except String String :=
  match (v.getObjValAs? String p.key).toOption with
  | some h => pure h
  | none => throw s!"{name}: its VERDICT names no {p.key}: it read no {p.what} ({p.flag})"

/-- The digests of the files the composition reads itself: the one staged rows file `digest` and its own inner shape file
`innerDigest`. -/
def filePins (digest innerDigest : String) : List Pin :=
  [{ key := "rec_rows_sha512", flag := "--rec-rows", what := "rows file", whose := "the one staged", digest },
   { key := "rec_inner_sha512", flag := "--rec-inner", what := "inner shape file", whose := "the composition's",
     digest := innerDigest }]

/-- The inner session's record and transcript, which no file the composition reads fixes: the first part's first
`VERDICT` names them, and every part must name the same. None without a part; the first part refused without a
`VERDICT`. -/
def sessionPins (parts : List (String × String)) : Except String (List Pin) := do
  let some (name, log) := parts.head? | pure []
  let some v := (← (verdicts log).mapError (s!"{name}: " ++ ·))[0]? | throw s!"{name}: no VERDICT"
  let pins : List Pin :=
    [{ key := "rec_inner_session_sha512", flag := "--rec-inner-session", what := "inner session record",
       whose := s!"{name}'s", digest := "" },
     { key := "rec_inner_transcript_sha512", flag := "--rec-inner-transcript", what := "inner transcript",
       whose := s!"{name}'s", digest := "" }]
  pins.mapM fun p => do return { p with digest := ← named name v p }

/-- **One part's log**: it has a `VERDICT`, every `VERDICT` accepts at one `rec_part`, and each names every pin's
digest at its key. Its part. -/
def checkPart (pins : List Pin) (name log : String) : Except String Nat := do
  let vs ← (verdicts log).mapError (s!"{name}: " ++ ·)
  let some v0 := vs[0]? | throw s!"{name}: no VERDICT"
  let some j := (v0.getObjValAs? Nat "rec_part").toOption
    | throw s!"{name}: its VERDICT names no rec_part: not one of V*'s parts (--rec-part J)"
  for v in vs do
    if (v.getObjValAs? Bool "accepted").toOption != some true then
      throw s!"{name}: a session is not accepted ({(v.getObjValAs? String "why").toOption.getD ""})"
    if (v.getObjValAs? Nat "rec_part").toOption != some j then
      throw s!"{name}: its VERDICTs name more than one rec_part"
    for p in pins do
      let h ← named name v p
      if h != p.digest then throw s!"{name}: read the {p.what} {h}, not {p.whose} {p.what} {p.digest}"
  pure j

/-- **The composition**: `parts`' logs (name, text), one per V* statement of `n`, each accepted at the one staged rows
file of SHA-512 `digest`, the one inner shape file of SHA-512 `innerDigest` and one inner session and transcript
(`checkPart`). The digests every part named. -/
def check (n : Nat) (digest innerDigest : String) (parts : List (String × String)) : Except String (List Pin) := do
  if n == 0 then throw "no V* statement"
  let pins := filePins digest innerDigest ++ (← sessionPins parts)
  let js ← parts.mapM fun (name, log) => checkPart pins name log
  for j in js do
    if j ≥ n then throw s!"rec_part {j}: V* has {n} statements"
    if js.count j != 1 then throw s!"rec_part {j}: {js.count j} logs, not one"
  if js.length != n then throw s!"{js.length} parts, not V*'s {n} statements"
  pure pins

end RecCompose
end Flock
