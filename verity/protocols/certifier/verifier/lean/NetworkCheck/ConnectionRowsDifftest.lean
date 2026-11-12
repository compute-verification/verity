module

public import NetworkCheck.ConnectionRows
public import NetworkCheck.Difftest

/-!
# The difftest: the connection rows' head against `connection_rows.py`, on generated instances

`scripts/connection_rows_vectors.py` writes instances and the Python's results on each.
`NetworkCheck/ConnectionRowsDifftestMain.lean` reads both files and fails unless `evaluate`, which computes the same
results with `NetworkCheck.ConnectionRows`' definitions, gives every one of them. The Lean audit runs it (`runs` in
`lean-audit.json`): compiled evaluation, a test and not a proof.

An instance is a schedule (`Period.scheduleOf`'s keys), its lengths rows' form and content classes, the digest Python
gives it (`parameters`: 64 zero bytes where `WardenParams` refuses it), a session, a link and an execution phase, the
network certifier's LMS key and parameters, and the queries. `evaluate` gives, with `null` where Python raises
`ValueError`:
* `tables`, `connection_rows.tables_of` as `[name, rows, row bytes]` each;
* `rooms`, each `[B, P]`'s `[content_rows.request_room, content_rows.request_bytes]`; `ids`, each O's
  `content_rows.response_bytes`;
* `domains`, each `[table, rows, row bytes]`'s `connection_rows.domain(...).domain_id`;
* `heads`, each given head's encoding and digest;
* `registrations`, each registration's head as `connection_rows.check_registration` returns it under the key in the
  session, by its encoding and digest.
-/

@[expose] public section

namespace NetworkCheck.ConnectionRowsDifftest

open Lean (Json toJson)
open NetworkCheck.Difftest (hexJson bytesOf arrOf)
open NetworkCheck.ConnectionRows

def classOf (j : Json) : Except String Class := do
  match ← arrOf j with
  | #[p, o] => pure { promptIds := ← p.getNat?, ids := ← o.getNat? }
  | #[p, o, r] => pure { promptIds := ← p.getNat?, ids := ← o.getNat?, rows := some (← r.getNat?) }
  | _ => throw "a class is [P, O] or [P, O, rows]"

/-- The instance's schedule with lengths rows, or what refuses it. -/
def tabledOf (j : Json) : Except String Tabled := do
  let schedule ← Period.scheduleOf (← j.getObjVal? "schedule")
  let lengths ← (← j.getObjVal? "lengths").getNat?
  let content ← match ← j.getObjVal? "content" with
    | .null => pure none
    | c => some <$> (← arrOf c).mapM classOf
  let t : Tabled := { schedule, lengths, content }
  t.check
  pure t

def entryOf (j : Json) : Except String Entry := do
  pure { name := ← j.getObjValAs? String "name", rows := ← j.getObjValAs? Nat "rows",
         rowBytes := ← j.getObjValAs? Nat "row_bytes", domain := ← bytesOf (← j.getObjVal? "domain"),
         root := ← bytesOf (← j.getObjVal? "root") }

def headOf (j : Json) : Except String Head := do
  pure { link := ← j.getObjValAs? String "link", window := ← j.getObjValAs? Nat "window",
         parameters := ← bytesOf (← j.getObjVal? "parameters"),
         tables := ← (← arrOf (← j.getObjVal? "tables")).mapM entryOf }

/-- A head by its encoding and digest, or `null` for one that doesn't encode. -/
def headJson (h : Head) : Json :=
  match h.encode, h.digest with
  | .ok e, .ok d => Json.mkObj [("encoding", hexJson e), ("digest", hexJson d)]
  | _, _ => .null

def natOr (x : Except String Nat) : Json :=
  match x with
  | .ok n => toJson n
  | .error _ => .null

def evaluate (j : Json) : Except String Json := do
  let sd ← bytesOf (← j.getObjVal? "parameters")
  let session ← bytesOf (← j.getObjVal? "session")
  let link ← j.getObjValAs? String "link"
  let w ← j.getObjValAs? Nat "window"
  let key ← bytesOf (← j.getObjVal? "key")
  let l ← j.getObjVal? "lms"
  let lms ← Lms.Params.make (← l.getObjValAs? Nat "N") (← l.getObjValAs? Nat "W") (← l.getObjValAs? Nat "H") false
  let tabled := (tabledOf j).toOption
  let tables := match tabled.bind (·.tables.toOption) with
    | some ts => Json.arr (ts.map fun x => Json.arr #[.str x.name, toJson x.rows, toJson x.rowBytes])
    | none => .null
  let rooms ← (← arrOf (← j.getObjVal? "rooms")).mapM fun q => do
    let #[b, p] ← arrOf q | throw s!"not [B, P]: {q.compress}"
    let (B, P) := (← b.getNat?, ← p.getNat?)
    pure (match requestRoom B P, requestBytes B P with
      | .ok room, .ok size => Json.arr #[toJson room, toJson size]
      | _, _ => .null)
  let ids ← (← arrOf (← j.getObjVal? "ids")).mapM fun o => do pure (natOr (responseBytes (← o.getNat?)))
  let domains ← (← arrOf (← j.getObjVal? "domains")).mapM fun q => do
    let #[table, rows, size] ← arrOf q | throw s!"not [table, rows, row bytes]: {q.compress}"
    pure (match domainId sd session link w (← table.getStr?) (← rows.getNat?) (← size.getNat?) with
      | .ok d => hexJson d
      | .error _ => .null)
  let heads ← (← arrOf (← j.getObjVal? "heads")).mapM fun h => headJson <$> headOf h
  let registrations := (← arrOf (← j.getObjVal? "registrations")).map fun doc =>
    match tabled with
    | none => Json.null
    | some t =>
      let p : Period.Params :=
        { audits := #[], cells := #[], first := 0, jitter := 0, link, lms, key, session, program := "",
          form := .rows t.schedule sd ⟨0, 0, 0, 0, 0⟩, windows := 1 }
      match registered p t sd w doc with
      | .ok h => headJson h
      | .error _ => .null
  pure (Json.mkObj [("tables", tables), ("rooms", .arr rooms), ("ids", .arr ids), ("domains", .arr domains),
    ("heads", .arr heads), ("registrations", .arr registrations)])

/-- The results' fields, in the order they are compared. -/
def fields : List String := ["tables", "rooms", "ids", "domains", "heads", "registrations"]

/-- `lake env lean --run NetworkCheck/ConnectionRowsDifftestMain.lean INSTANCES PYTHON N`. -/
def main (args : List String) : IO UInt32 :=
  NetworkCheck.Difftest.mainOf evaluate fields "NetworkCheck/ConnectionRowsDifftestMain.lean" args

end NetworkCheck.ConnectionRowsDifftest
