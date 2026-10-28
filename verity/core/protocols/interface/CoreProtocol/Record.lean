/-
A record entry: one message of a run's transcript (a registration, a commitment, the challenger's coins or sample, a
proof, a verdict), numbered, chained to the entry before it and signed by the party that sent it.

    entry   = {body, kind, prev, schema, seq, session, sig}
    message = SHA-512([role] ‖ I ‖ [d])                d: the schema's tagged SHA-512 of the entry without its sig

each `[x]` a u64 big-endian length then `x`, and `I` the signer's 16-byte LMS identifier. `body` is the message, an
object of the schema's form; `kind` its kind under the schema; `schema` the protocol's entry schema and version, which
names the tag `d` is under; `session` the session it belongs to, `null` for an entry of none. Entry `i` of a record has
`seq = i` and `prev` the SHA-512 of entry `i - 1`'s line (`ZERO` for entry 0), the line being the protocol's canonical
text of the entry.

`Record.ofJson` reads the form: the seven keys, each of its type. Everything a protocol accepts beyond the form (its
schemas, kinds and bodies, which key signs, its sessions) is its own check; the auditor's ledger
(`SampledProofs.Challenger.ledgerEntry?`) reads its entries through this. `verity/core/protocols/record.py` mirrors it.
-/
import Lean.Data.Json
import Flock.Bytes
import Flock.Hash

namespace Core

open Lean (Json)
open Flock (V)

/-- A record entry's keys, sorted. -/
def Record.FIELDS : Array String := #["body", "kind", "prev", "schema", "seq", "session", "sig"]

/-- The `prev` of a record's first entry: 128 zeros. -/
def Record.ZERO : String := String.ofList (List.replicate 128 '0')

def Record.FORM : String :=
  "a record entry is {body, kind, prev, schema, seq, session, sig}: an object, a string, a SHA-512 in lowercase hex, a string, a natural number, a string or null, and lowercase hex"

structure Record where
  body : Json
  kind : String
  prev : String
  schema : String
  seq : Nat
  session : Option String
  sig : ByteArray

/-- `record.entry_of`: `j` as a record entry, its seven keys each of its type; `FORM` otherwise. -/
def Record.ofJson (j : Json) : V Record := do
  let .obj kvs := j | throw Record.FORM
  unless (kvs.toArray.map (·.1)).qsort (· < ·) == Record.FIELDS do throw Record.FORM
  let get (k : String) : Json := (j.getObjVal? k).toOption.getD Json.null
  let .obj _ := get "body" | throw Record.FORM
  let .str kind := get "kind" | throw Record.FORM
  let .str prev := get "prev" | throw Record.FORM
  unless prev.length == 128 && (Flock.unhex prev).toOption.isSome do throw Record.FORM
  let .str schema := get "schema" | throw Record.FORM
  let .num seq := get "seq" | throw Record.FORM
  unless seq.exponent == 0 && 0 ≤ seq.mantissa do throw Record.FORM
  let session ← match get "session" with
    | .str s => pure (some s)
    | .null => pure none
    | _ => throw Record.FORM
  let .str sig := get "sig" | throw Record.FORM
  let .ok sig := Flock.unhex sig | throw Record.FORM
  return ⟨get "body", kind, prev, schema, seq.mantissa.toNat, session, sig⟩

def Record.toJson (r : Record) : Json :=
  Json.mkObj [("body", r.body), ("kind", Json.str r.kind), ("prev", Json.str r.prev), ("schema", Json.str r.schema),
    ("seq", Lean.toJson r.seq), ("session", (r.session.map Json.str).getD Json.null),
    ("sig", Json.str (Flock.hexOf r.sig))]

/-- `record.unsigned`: an entry's JSON without its `sig`, what its signature's digest is over. -/
def Record.unsigned (entry : Json) : Json :=
  match entry with
  | .obj kvs => Json.mkObj (kvs.toArray.toList.filter (·.1 != "sig"))
  | _ => Json.mkObj []

/-- `record.message`: what an entry's `sig` signs under the protocol's `role`, the signer's identifier `I` and `d`, the
digest of the entry without its sig under the tag its schema names. -/
def Record.message (role : String) (I d : ByteArray) : ByteArray :=
  let field (b : ByteArray) : ByteArray := Flock.be64 b.size.toUInt64 ++ b
  Flock.Sha512.hash (field role.toUTF8 ++ I ++ field d)

/-- `record.prev_of`: the `prev` of the entry after the one whose line is `line`. -/
def Record.prevOf (line : String) : String := Flock.hexOf (Flock.Sha512.hash line.toUTF8)

end Core
