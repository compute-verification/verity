/-
`Tmpl.inputsT` against `Tmpl.unroll`: `Flock.Test.Tmpl`'s unit, its input rows written by copies of a self-row Definition
(`SELF`, directly or through a block of three) instead of the root's own rows, passes and unrolls to the same text, with
input padding written by copies of an empty Definition. The forms it refuses, each one `Net.ofRows` refuses on the
unrolled rows too: a copy in the input rows of a Definition writing a row that is not a self row, and input port bits
left without a self row (copies too few, or not side by side). And `Tmpl.apart` against `Tmpl.rows`: the units above
pass, and the collisions `Tmpl.rows` refuses fail it (a root row meeting a copy's, a row at the constant, two runs of
copies meeting). And `Tmpl.groupsOk`, the port groups' and size's checks of `Tmpl.unroll` (`unroll_ok`): every unit above
that passes all five description checks unrolls, and the groups `Net.ofRows` refuses fail it (an output group past the
unit's slot, D11, or not aligned).
-/
import Flock.Test.Tmpl
import Flock.TmplUnroll

namespace Flock.Test.TmplRows

open Flock Flock.Test.Tmpl

/-- `Flock.Test.Tmpl.header` with the root at Definition `root`. -/
def headerAt (root : Nat) : String := s!"flock-ir-tmpl/v2 t {root} 1 0 1 1 3 1 2 1 1 2"

/-- Definition 0, a self row: own column 0 is `[0]·[0]`. -/
def SELF : List String := ["DEF 1 0 1 0", "0 1 0 1 0"]

/-- Definitions 1 and 2, `Flock.Test.Tmpl`'s AND and pair, the pair calling Definition 1. -/
def andPair : List String := ["DEF 1 2 1 0", "0 1 2 1 1", "DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 1 2 0 1 1 2 3 0 1 0"]

/-- The root after `extra` Definitions: `Flock.Test.Tmpl.root`'s rows past the input rows, its input rows written by
`inputs`, and the pairs at 128 and 131. -/
def rootAt (inputs : List String) : List String :=
  [s!"DEF 400 0 3 {inputs.length + 1}", "256 1 130 1 399", "257 1 133 1 399", "300 3 0 1 0 1 400"] ++ inputs ++
    ["CALL 2 2 128 3 1 2 0 1 1 0"]

def lines (self : List String) (extra : List String) (inputs : List String) : Array String :=
  (headerAt (3 + (extra.filter (·.startsWith "DEF ")).length) :: self ++ andPair ++ extra ++ rootAt inputs).toArray

def inputsT (ls : Array String) : Option Bool := (_root_.Flock.Tmpl.parse ls).toOption.map _root_.Flock.Tmpl.Tmpl.inputsT

-- the root's own self rows (`Flock.Test.Tmpl`)
#guard inputsT tmplLines == some true
-- three copies of `SELF` side by side
#guard inputsT (lines SELF [] ["CALL 0 3 0 1 0"]) == some true
#guard bytesAgree (lines SELF [] ["CALL 0 3 0 1 0"]) flatText
-- a block of three copies of `SELF`, copied once
#guard inputsT (lines SELF ["DEF 3 0 0 1", "CALL 0 3 0 1 0"] ["CALL 3 1 0 3 0"]) == some true
#guard bytesAgree (lines SELF ["DEF 3 0 0 1", "CALL 0 3 0 1 0"] ["CALL 3 1 0 3 0"]) flatText
-- input padding past the port bits written by copies of an empty Definition
#guard inputsT (lines SELF ["DEF 1 0 0 0"] ["CALL 0 3 0 1 0", "CALL 3 125 3 1 0"]) == some true
#guard bytesAgree (lines SELF ["DEF 1 0 0 0"] ["CALL 0 3 0 1 0", "CALL 3 125 3 1 0"]) flatText
-- `SELF` writing `[0]·[const]`, not a self row
#guard inputsT (lines ["DEF 1 0 1 0", "0 1 0 1 1"] [] ["CALL 0 3 0 1 0"]) == some false
#guard refuses (lines ["DEF 1 0 1 0", "0 1 0 1 1"] [] ["CALL 0 3 0 1 0"])
-- two copies: port bit 2 has no self row
#guard inputsT (lines SELF [] ["CALL 0 2 0 1 0"]) == some false
#guard refuses (lines SELF [] ["CALL 0 2 0 1 0"])
-- copies at 0 and 2, not side by side: port bit 1 has no self row
#guard inputsT (lines SELF [] ["CALL 0 2 0 2 0"]) == some false
#guard refuses (lines SELF [] ["CALL 0 2 0 2 0"])
-- the root's row 1 reading column 2, not a self row
#guard inputsT (header :: and2 ++ pair ++ ["DEF 400 0 6 1", "0 1 0 1 0", "1 1 2 1 1", "2 1 2 1 2", "256 1 130 1 399",
  "257 1 133 1 399", "300 3 0 1 0 1 400", CALL]).toArray == some false

def apart (ls : Array String) : Option Bool := (_root_.Flock.Tmpl.parse ls).toOption.map _root_.Flock.Tmpl.Tmpl.apart

#guard apart tmplLines == some true
#guard apart (lines SELF [] ["CALL 0 3 0 1 0"]) == some true
#guard apart (lines SELF ["DEF 3 0 0 1", "CALL 0 3 0 1 0"] ["CALL 3 1 0 3 0"]) == some true
-- the root's row at 130 meets the first pair's sum row
#guard apart (tmplLines (extra := ["130 1 0 1 1"])) == some false
#guard refuses (tmplLines (extra := ["130 1 0 1 1"]))
-- a root row at 399, the constant's column
#guard apart (tmplLines (extra := ["399 1 0 1 0"])) == some false
#guard refuses (tmplLines (extra := ["399 1 0 1 0"]))
-- two rows at column 0
#guard apart (tmplLines (extra := ["0 1 1 1 1"])) == some false
#guard refuses (tmplLines (extra := ["0 1 1 1 1"]))
-- runs of `SELF` at 0 to 2 and 2 to 3, meeting at 2
#guard apart (lines SELF [] ["CALL 0 3 0 1 0", "CALL 0 2 2 1 0"]) == some false
#guard refuses (lines SELF [] ["CALL 0 3 0 1 0", "CALL 0 2 2 1 0"])

def described (ls : Array String) : Bool :=
  match _root_.Flock.Tmpl.parse ls with
  | .ok t => t.wellFormed && t.apart && t.ordered && t.inputsT && t.groupsOk
  | .error _ => false

def groupsOk (ls : Array String) : Option Bool :=
  (_root_.Flock.Tmpl.parse ls).toOption.map _root_.Flock.Tmpl.Tmpl.groupsOk

#guard groupsOk tmplLines == some true
#guard [tmplLines, lines SELF [] ["CALL 0 3 0 1 0"], lines SELF ["DEF 3 0 0 1", "CALL 0 3 0 1 0"] ["CALL 3 1 0 3 0"],
  lines SELF ["DEF 1 0 0 0"] ["CALL 0 3 0 1 0", "CALL 3 125 3 1 0"]].all fun ls => described ls && !refuses ls
-- the output group at word 4 of a 4-word slot (D11)
#guard groupsOk (tmplLines.set! 0 "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 4 1 1 2") == some false
#guard refuses (tmplLines.set! 0 "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 4 1 1 2")
-- a two-word output group at word 1, not aligned
#guard groupsOk (tmplLines.set! 0 "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 1 2 1 2") == some false
#guard refuses (tmplLines.set! 0 "flock-ir-tmpl/v2 t 2 1 0 1 1 3 1 1 2 1 2")

end Flock.Test.TmplRows
