/-
`canon` prints only strings and keys of ASCII without DEL, the domain on which its bytes are Python's
`codec.canonical_json`'s, and refuses any other, wherever it sits; `canonLine`, the form a difftest's or a command's line
is printed in, prints any string.
-/
import Flock.Canon

namespace Flock.Test.Canon

open Lean (Json)

def refused (j : Json) : Bool :=
  match canon j with
  | .error e => e == notAscii
  | .ok _ => false

#guard refused (.str "dé") && refused (.str "\x7f") && refused (.str "a\u0080")
#guard refused (Json.mkObj [("dé", .null)]) && refused (Json.mkObj [("k", .str "dé")])
#guard refused (.arr #[.num 1, .arr #[.str "dé"]]) && refused (Json.mkObj [("a", Json.mkObj [("b\x7f", .bool true)])])
#guard (canon (Json.mkObj [("k", .str "d~\x01\"\\"), ("a", .arr #[.null])])).toOption ==
  some "{\"a\":[null],\"k\":\"d~\\u0001\\\"\\\\\"}"
#guard (canonLine (Json.mkObj [("dé", .str "\x7f")])).toOption == some "{\"dé\":\"\x7f\"}"
#guard (canonLine (.str "Δ ≥ 1")).toOption == some "\"Δ ≥ 1\""

end Flock.Test.Canon
