/-
S3/F1 (PROTOCOL.md §5.3) reads `config.require_link` by a match (`matches Json.bool true`), which decides what `j ==
Json.bool true` decides on `true`, `false`, `null`, the string `"true"`, the numbers 1 and 0, a list holding `true` and
an object, so `Record.decode` keeps `==`'s verdicts. A record whose `config.require_link` is `true` passes S3/F1 and is
refused only later (here at S6, which finds no `root_f`); every other value is refused at S3/F1, and an absent one as
missing.
-/
import Flock.Record

namespace Flock.Test.RequireLink

open Flock Lean

def vals : List Json :=
  [.bool true, .bool false, .null, .str "true", .num (JsonNumber.fromNat 1), .num (JsonNumber.fromNat 0),
    .arr #[.bool true], Json.mkObj [("require_link", .bool true)]]

#guard vals.all fun j => (j matches .bool true) == (j == Json.bool true)
#guard (vals.map (· matches .bool true)) == [true, false, false, false, false, false, false, false]

def spec : SessionSpec := { (default : SessionSpec) with hello := "h" }

/-- A record through S4 at `spec`, with `config.require_link` set to `req` (absent at `none`). -/
def record (req : Option Json) : String :=
  let cfg := match req with
    | some v => Json.mkObj [("require_link", v)]
    | none => Json.mkObj []
  (Json.mkObj [("kind", .str "flock-live-session/v1"), ("hello", .str "h"), ("config", cfg),
    ("link_mode", .str "exchange"), ("link", Json.mkObj [("mode", .str "exchange"), ("sigma", .str "")])]).compress

def why (req : Option Json) : Option String :=
  match Record.decode spec (record req) with
  | .ok _ => none
  | .error e => some e

def s3 : String := "S3/F1: require_link is not true"

#guard why (some (.bool true)) == some "record: root_f missing"
#guard (vals.drop 1).all fun j => why (some j) == some s3
#guard why none == some "record: require_link missing"

end Flock.Test.RequireLink
