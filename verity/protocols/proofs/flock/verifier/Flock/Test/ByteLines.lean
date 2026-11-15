/-
`HmRow.parseWith` runs its twin on an ASCII file (`parseWith_eq_bytes`), which reads the file's lines and rows through
`ByteLines.View`. These guards hold the view's compiled code to what it stands for: its lines to `String.splitOn "\n"`'s
without a last empty one, a line's numbers to `Net.lineNums` (a token past 19 digits, a token that is not a number, a line
of whitespace), and a `NET` block to `Net.parse` of its lines, whole, truncated, with a `LEAVES` or `CUT` line after it,
or with a row that is not numbers. A file that is empty, ends without a `'\n'` or holds empty lines has a view; one that
is not ASCII has none.
-/
module

public import Flock.ByteLines
public import Flock.Test.Tmpl
public import Flock.With.Setup
meta import Flock.ByteLines
meta import Flock.Test.Tmpl
meta import Flock.With.Setup

@[expose] public section

namespace Flock.Test.ByteLines

open Flock ByteLines

/-- The lines `HmRow.parseWith` splits off `b`. -/
def ref (b : ByteArray) : Option (Array String) :=
  (String.fromUTF8? b).map fun text =>
    let l := (text.splitOn "\n").toArray
    if l.back? == some "" then l.pop else l

def files : List String :=
  ["", "\n", "\n\n", "a", "a\n", "a\nb", "a\nb\n", "x\n\ny\n\n", "1 2 3\n 4  5\t6 \n", "  \n7\n",
   "12345678901234567890123 1\n1 2 x\n0\n"]

#guard files.all fun f => (View.of? f.toUTF8).map (·.lines) == ref f.toUTF8
#guard files.all fun f => match View.of? f.toUTF8 with
  | some v => (List.range (v.count + 2)).all fun k => v.nums k == Net.lineNums (v.str k)
  | none => false
#guard (View.of? "é\n".toUTF8).isNone
#guard (View.of? (ByteArray.mk #[49, 10, 200])).isNone

/-- A file of a line `X`, the `NET` block `lines`, and `tail`. -/
def file (lines : Array String) (tail : List String) : ByteArray :=
  (String.intercalate "\n" (["X"] ++ lines.toList ++ tail) ++ "\n").toUTF8

/-- `View.net` over lines `[a, e)` of `b` gives what `Net.parse` gives on them. -/
def netAgrees (b : ByteArray) (a e : Nat) : Bool :=
  match View.of? b with
  | some v =>
    match v.net a e, Net.parse (v.lines.extract a e) with
    | .ok n, .ok m => Flock.Test.Tmpl.same n m
    | .error x, .error y => x == y
    | _, _ => false
  | none => false

def flat : Array String := Flock.Test.Tmpl.flatText

#guard (match View.of? (file flat []) with | some v => (v.net 1 (1 + flat.size)).toOption.isSome | none => false)
#guard netAgrees (file flat []) 1 (1 + flat.size)
#guard netAgrees (file flat [Flock.Test.Tmpl.LEAVES]) 1 (2 + flat.size)
#guard netAgrees (file flat ["CUT 1"]) 1 (2 + flat.size)
#guard netAgrees (file flat []) 1 300
#guard netAgrees (file flat []) 0 (1 + flat.size)
#guard netAgrees (file (flat.set! 5 "1 2 x") []) 1 (1 + flat.size)

/-- A file of `tags` whose header and META pass `HmRow.headWith` and whose META `HmRow.parseRest` refuses after the
sections: `sec`, then the section `CIRCUIT unit` holding `flat`. -/
def stmt (tags : Tags) (sec : String) : ByteArray :=
  let hex := String.ofList (List.replicate 128 'a')
  (tags.fileFormat ++ " u\nMETA {\"format\":\"" ++ tags.fileFormat ++ "\",\"statement\":\"" ++ tags.statement ++
    "\",\"program_sha512\":\"" ++ hex ++ "\",\"partition\":{\"rule\":\"" ++ HmRow.PARTITION_V1 ++ "\",\"digest\":\"" ++
    hex ++ "\"}}\n" ++ sec ++ String.intercalate "\n" (s!"CIRCUIT unit {flat.size}" :: flat.toList) ++ "\n").toUTF8

/-- What `HmRow.parseWith`'s compiled code (`parseWithB`) and `HmRow.parse` give on `b`: one error, or both a circuit. -/
def sameParse (tags : Tags) (b : ByteArray) : Bool :=
  match HmRow.parseWith (M := V) (Hashes.ofSha512 Sha512.hash) tags b, HmRow.parse tags b with
  | .error x, .error y => x == y
  | .ok _, .ok _ => true
  | _, _ => false

def sec : String := "TEMPLATE 2\na b\nc d\n"

#guard [Tags.circuitTmpl, Tags.circuit].all fun t => sameParse t (stmt t sec) && sameParse t (stmt t "")
#guard match HmRow.parse Tags.circuitTmpl (stmt Tags.circuitTmpl sec) with
  | .error e => !e.startsWith "circuit: expected" && !e.startsWith "unit:"
  | .ok _ => true

end Flock.Test.ByteLines
