/-
`Tmpl.unroll` against `Net.parse`: a templated unit (two copies of a pair, each two copies of a two-input AND and their
sum, bound to input bits that step by one, and two output bits) unrolls to the net `Net.parse` reads from its flat text,
written out here by hand, and `Tmpl.flatLines` prints that text byte for byte, sorted and reduced over GF(2) as
`flock-ir-unit/v2` writes it, with the carried `LEAVES` line, and so does the same unit with its bindings as two-level
segments. The forms `Tmpl.parse` and `Tmpl.unroll` refuse: overlapping
copies, a callee that is not an earlier definition, segments that don't cover the callee's inputs or bind past the
caller's references, a column written twice, a root with inputs, a copy reading a later column, a Definition of `2^64`
columns.
-/
import Flock.Tmpl

namespace Flock.Test.Tmpl

open Flock

def same (n m : Net) : Bool :=
  n.name == m.name && n.useful == m.useful && n.constPos == m.constPos && n.inWords == m.inWords &&
    n.unitLog == m.unitLog && n.inGroups.size == m.inGroups.size && n.outGroups.size == m.outGroups.size &&
    ((n.inGroups ++ n.outGroups).toList.zip (m.inGroups ++ m.outGroups).toList).all
      (fun (g, h) => g.col == h.col && g.words == h.words && g.bits == h.bits) &&
    [(n.a, m.a), (n.b, m.b)].all fun (x, y) =>
      x.rows == y.rows && x.width == y.width && x.data.data == y.data.data && x.start == y.start && x.len == y.len

def header : String := "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 2 1 1 2"

/-- Definition 0, an AND of its inputs: own column 0 is `[in 1]·[in 0]`. -/
def and2 : List String := ["DEF 1 2 1 0", "0 1 2 1 1"]

/-- Definition 1, a pair: two ANDs of its inputs at own columns 0 and 1, and own column 2 their sum times the constant. -/
def pair : List String := ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 2 3 0 1 0"]

/-- The root: input bits 0 to 2, two pairs at 128 and 131 bound to bits `(j, j + 1)`, the outputs at 256 and 257, and at
300 the row `[0, 1, 0]·[const]`, which reduces to `[1]·[399]`. -/
def root (call : String) (extra : List String) : List String :=
  [s!"DEF 400 0 {6 + extra.length} 1", "0 1 0 1 0", "1 1 1 1 1", "2 1 2 1 2", "256 1 130 1 399", "257 1 133 1 399",
    "300 3 0 1 0 1 400"] ++ extra ++ [call]

def CALL : String := "CALL 1 2 128 3 1 2 0 1 1 0"

def tmplLines (call : String := CALL) (extra : List String := []) : Array String :=
  (header :: and2 ++ pair ++ root call extra).toArray

def flatText : Array String := Id.run do
  let rows : List (Nat × String) :=
    [(0, "1 0 1 0"), (1, "1 1 1 1"), (2, "1 2 1 2"), (128, "1 1 1 0"), (129, "1 1 1 0"), (130, "2 128 129 1 399"),
     (131, "1 2 1 1"), (132, "1 2 1 1"), (133, "2 131 132 1 399"), (256, "1 130 1 399"), (257, "1 133 1 399"),
     (300, "1 1 1 399"), (399, "1 399 1 399")]
  let mut out := #["flock-ir-unit/v2 t 400 399 1 0 1 1 3 1 2 1 1 2"]
  for i in [0:400] do out := out.push ((rows.lookup i).getD "0 0")
  return out

def unrolled (lines : Array String) : Except String Net := do
  _root_.Flock.Tmpl.unroll (← _root_.Flock.Tmpl.parse lines)

def printed (lines : Array String) : Except String (Array String) :=
  _root_.Flock.Tmpl.parse lines >>= _root_.Flock.Tmpl.flatLines

def agrees : Bool :=
  match unrolled tmplLines, Net.parse flatText with
  | .ok n, .ok m => same n m
  | _, _ => false

def bytesAgree (lines flat : Array String) : Bool :=
  match printed lines with
  | .ok ls => ls == flat
  | .error _ => false

def LEAVES : String := "LEAVES {\"x\":[1]}"

def refuses (lines : Array String) : Bool := (unrolled lines).toOption.isNone

#guard (Net.parse flatText).toOption.isSome
#guard agrees
#guard bytesAgree tmplLines flatText
#guard bytesAgree (tmplLines.push LEAVES) (flatText.push LEAVES)
#guard (Net.parse (flatText.push LEAVES)).toOption.isSome
#guard _root_.Flock.Tmpl.canon #[5, 3, 5, 3, 3, 1] == #[1, 3]
-- copies at stride 2 overlap a callee of 3 columns
#guard refuses (tmplLines "CALL 1 2 128 2 1 2 0 1 1 0")
-- a callee that is not an earlier definition
#guard refuses (tmplLines "CALL 2 2 128 3 1 2 0 1 1 0")
-- segments binding one input of two
#guard refuses (tmplLines "CALL 1 2 128 3 1 1 0 1 1 0")
-- copy 1 binds references 400 and 401, past the root's (its constant is 400)
#guard refuses (tmplLines "CALL 1 2 128 3 1 2 399 1 1 0")
-- the copies overrun the root's columns
#guard refuses (tmplLines "CALL 1 2 395 3 1 2 0 1 1 0")
-- the copies' rows meet the root's row at 130
#guard refuses (tmplLines (extra := ["130 1 0 1 1"]))
-- copy 1 over the root's output rows
#guard refuses (tmplLines "CALL 1 2 254 3 1 2 0 1 1 0")
-- a copy reading columns 300 and 301, after its own
#guard refuses (tmplLines "CALL 1 1 128 3 1 2 300 0 1 0")
-- a row reading past its definition's references
#guard refuses (header :: ["DEF 1 2 1 0", "0 1 4 1 1"] ++ pair ++ root CALL []).toArray
-- a root with inputs
#guard refuses #[header, "DEF 1 2 1 0", "0 1 2 1 1", "DEF 3 2 0 0", "DEF 4 2 0 0"]
-- the order, checked on the description: the template above is ordered
#guard ((_root_.Flock.Tmpl.parse tmplLines).toOption.map _root_.Flock.Tmpl.Tmpl.ordered) == some true
-- a pair's sum row reading its own later column
#guard refuses (header :: and2 ++ ["DEF 4 2 1 1", "2 2 1 3 1 6", "CALL 0 2 0 1 1 2 4 0 1 0"] ++ root CALL []).toArray
-- a segment straddling the pair's own columns and its inputs (references 2 and 3)
#guard refuses (header :: and2 ++ ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 2 2 0 1 0"] ++ root CALL []).toArray
-- copy 1 bound to bits 3 and 4, input padding past the group's 3 ports
#guard refuses (tmplLines "CALL 1 2 128 3 1 2 2 1 1 0")
-- a copy straddling the input rows' end
#guard refuses (tmplLines "CALL 1 2 126 3 1 2 0 1 1 0")
-- two-level segments: each binding above as one input repeated twice, a step of 1 per run, unrolls to the same text
#guard bytesAgree (header :: and2 ++ ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 1 3 0 2 1"] ++
  root "CALL 1 2 128 3 1 1 0 1 2 1" []).toArray flatText
-- a run stride of 0 binds one reference to both inputs
#guard !refuses (header :: and2 ++ ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 1 3 0 2 0"] ++ root CALL []).toArray
-- two runs of one input binding one input of two
#guard refuses (tmplLines "CALL 1 2 128 3 1 1 0 1 1 1")
-- run 1 straddling the pair's own columns and its inputs (references 2 and 3)
#guard refuses (header :: and2 ++ ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 1 2 0 2 1"] ++ root CALL []).toArray
-- run 1 binding reference 400 and copy 1's 401, past the root's
#guard refuses (tmplLines "CALL 1 2 128 3 1 1 0 1 2 400")
-- run 1 reading input padding (bits 3 and 4) past the group's 3 ports
#guard refuses (tmplLines "CALL 1 2 128 3 1 1 0 1 2 3")
-- a root of `2^64` columns, past what `locate` halves
#guard (_root_.Flock.Tmpl.parse #["flock-ir-tmpl/v2 t 0 0 0", "DEF 18446744073709551616 0 0 0"]).toOption.isNone
#guard (_root_.Flock.Tmpl.parse #["flock-ir-tmpl/v2 t 0 0 0", "DEF 18446744073709551615 0 0 0"]).toOption.isSome

end Flock.Test.Tmpl
