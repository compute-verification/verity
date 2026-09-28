/-
JSON as Python's `json.dumps(v, sort_keys=True)` writes it, the form of every line of the firewall's record and of the
values its error messages quote (Rust's `release::py_json`): `", "` and `": "` between items, keys sorted, and every
character outside `' '`..`'~'` escaped (`\"`, `\\`, `\n`, `\r`, `\t`, `\b`, `\f`, else `\uXXXX` in lowercase hex, a
character above U+FFFF as its UTF-16 surrogate pair).
-/
import Lean.Data.Json

namespace Core.Firewall

open Lean (Json)

private def hex4 (n : Nat) : String :=
  let d := fun (k : Nat) => let x := (n >>> (4 * k)) % 16; if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)
  String.ofList [d 3, d 2, d 1, d 0]

private def escape (c : Char) : String :=
  match c with
  | '"' => "\\\""
  | '\\' => "\\\\"
  | '\n' => "\\n"
  | '\r' => "\\r"
  | '\t' => "\\t"
  | c =>
    let n := c.toNat
    if n == 8 then "\\b"
    else if n == 12 then "\\f"
    else if 0x20 ≤ n && n ≤ 0x7e then c.toString
    else if n < 0x10000 then "\\u" ++ hex4 n
    else "\\u" ++ hex4 (0xD800 + (n - 0x10000) / 0x400) ++ "\\u" ++ hex4 (0xDC00 + (n - 0x10000) % 0x400)

def pyStr (s : String) : String := "\"" ++ String.join (s.toList.map escape) ++ "\""

def pyStrs (xs : Array String) : String := "[" ++ ", ".intercalate (xs.toList.map pyStr) ++ "]"

/-- Any JSON value. Numbers are written as Lean writes them, which is Python's for integers, the only numbers the record
holds. -/
partial def pyJson : Json → String
  | Json.null => "null"
  | Json.bool b => if b then "true" else "false"
  | Json.num n => if n.exponent == 0 then toString n.mantissa else toString n
  | Json.str s => pyStr s
  | Json.arr a => "[" ++ ", ".intercalate (a.toList.map pyJson) ++ "]"
  | Json.obj kvs =>
    let items := kvs.toArray.qsort (fun a b => a.1 < b.1)
    "{" ++ ", ".intercalate (items.toList.map fun (k, v) => pyStr k ++ ": " ++ pyJson v) ++ "}"

-- `release::tests::py_json_matches_python`'s value
#guard pyJson (Json.mkObj [("b", Json.arr #[Json.num 1, Json.str "x"]),
    ("a", Json.mkObj [("z", Json.null), ("y", Json.bool true)]),
    ("c", Json.str ("é" ++ (Char.ofNat 0x1F600).toString ++ "\"\\\n" ++ (Char.ofNat 1).toString ++ "~" ++
      (Char.ofNat 0x7f).toString))]) ==
  "{\"a\": {\"y\": true, \"z\": null}, \"b\": [1, \"x\"], \"c\": \"\\u00e9\\ud83d\\ude00\\\"\\\\\\n\\u0001~\\u007f\"}"
#guard pyStrs #["L0", "L0"] == "[\"L0\", \"L0\"]"
#guard pyStrs #[] == "[]"
#guard pyJson (Json.num (-3)) == "-3"

end Core.Firewall
