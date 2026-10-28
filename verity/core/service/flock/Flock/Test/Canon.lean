/-
`canon` escapes every character outside printable ASCII, so its text has no byte at or above `0x7f`: these are the bytes
Python's `codec.canonical_json` writes for the same values (`json.dumps(..., ensure_ascii=True)` with DEL escaped), and
each reads back to its value.
-/
import Flock.Canon

namespace Flock.Test.Canon

open Lean (Json)

def VECTORS : List (Json × String) := [
  (.str "dé", "\"d\\u00e9\""),
  (.str "\x7f", "\"\\u007f\""),
  (.str "a\u0080", "\"a\\u0080\""),
  (.str "Δ ≥ 1", "\"\\u0394 \\u2265 1\""),
  (.str "😀", "\"\\ud83d\\ude00\""),
  (.str (String.singleton (Char.ofNat 0x10ffff)), "\"\\udbff\\udfff\""),
  (.str "\uffff", "\"\\uffff\""),
  (.str "\ue000", "\"\\ue000\""),
  (.str "\x00\x01\x08\t\n\x0b\x0c\r\x1f \"\\/~", "\"\\u0000\\u0001\\b\\t\\n\\u000b\\f\\r\\u001f \\\"\\\\/~\""),
  (Json.mkObj [("dé", .str "\x7f"), ("a", .arr #[.str "😀"])], "{\"a\":[\"\\ud83d\\ude00\"],\"d\\u00e9\":\"\\u007f\"}"),
  (Json.mkObj [("k", .str "d~\x01\"\\"), ("a", .arr #[.null])], "{\"a\":[null],\"k\":\"d~\\u0001\\\"\\\\\"}")]

#guard VECTORS.all fun (j, t) => (canon j).toOption == some t
#guard VECTORS.all fun (_, t) => t.toUTF8.data.all (· < 0x7f)
#guard VECTORS.all fun (j, t) => match Json.parse t with
  | .ok j' => (canon j').toOption == (canon j).toOption
  | .error _ => false

/-- An identity manifest's bytes (`identity.canonical_json_bytes`, `ensure_ascii=False`): strings in UTF-8, DEL raw. -/
def UTF8_VECTORS : List (Json × String) := [
  (.str "dé", "\"dé\""),
  (.str "\x7f", "\"\x7f\""),
  (.str "😀\x01", "\"😀\\u0001\""),
  (Json.mkObj [("dé", .str "Δ"), ("a", .num 1)], "{\"a\":1,\"dé\":\"Δ\"}")]

#guard UTF8_VECTORS.all fun (j, t) => (canonUtf8 j).toOption == some t

end Flock.Test.Canon
