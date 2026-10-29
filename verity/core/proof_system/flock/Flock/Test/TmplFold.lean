/-
`Tmpl.fold` against the lincheck's unit fold: on `Flock.Test.Tmpl`'s unit and variants (one copy, copies in lanes that
reach 64 and past, a run of 3 in 4 lanes, a lane binding its caller's input, the constant bound), laid out by
`Tmpl.layoutM` under each `Mode`, the fold from the description equals `foldRef`, the sum over the unrolled rows at their
positions, and that equals `Stmt.project (Stmt.baseOf α (Stmt.tensor lw ρ) A B none sl) (eqTable r)` on `A` and `B` of the
net committed at the layout's positions (`Tmpl.unrollAt`), the matrices a flat unit's slot would hold, and so does
`foldSplit`, on these and on units whose pair or quad the call tree reaches on two paths, where it folds tables.
-/
import Flock.TmplSplit
import Flock.NetAt
import Flock.Statement
import Flock.Test.TmplLayout

namespace Flock.Test.TmplFold

open Flock Flock.Test.Tmpl

def f (i : Nat) : F128 := ⟨(0x9e3779b97f4a7c15 * (i.toUInt64 + 1)) ^^^ 0x5bd1e995, 0xc2b2ae3d27d4eb4f * (i.toUInt64 + 7)⟩

def lw : Array F128 := (Array.range 64).map fun i => f (100 + i)

/-- The fold four ways, `fold`, `foldRef`, `project ∘ baseOf` and `foldSplit` (with its table count), on the template
`lines` laid out under `mode`. -/
def three (lines : Array String) (mode : _root_.Flock.Tmpl.Mode := .lenient) :
    Except String (Array F128 × Array F128 × Array F128 × Array F128 × Nat) := do
  let t ← _root_.Flock.Tmpl.parse lines
  let lays ← _root_.Flock.Tmpl.layoutM t mode
  let size := (lays.getD t.root default).size
  let sl := max 7 (log2ceil size)
  let rho := (Array.range (sl - 6)).map fun i => f (200 + i)
  let r := (Array.range (sl - 6)).map fun i => f (300 + i)
  let alpha := f 7
  let (p, _) ← _root_.Flock.Tmpl.fold t lays alpha lw rho r
  let q ← _root_.Flock.Tmpl.foldRef t lays alpha lw rho r
  let n ← _root_.Flock.Tmpl.unrollAt t lays
  if n.useful != size || n.unitLog != sl then throw "the committed net is not the layout's size"
  let base := Stmt.baseOf alpha (Stmt.tensor lw rho) n.a n.b none sl
  let (s, _, tables, _) ← _root_.Flock.Tmpl.foldSplit t lays alpha lw rho r
  return (p, q, Stmt.project base (eqTable r), s, tables)

def modes : List _root_.Flock.Tmpl.Mode := [.lenient, .strict, .hybrid 0, .hybrid 1, .hybrid 2]

/-- The four agree under every mode. -/
def agree (lines : Array String) : Bool :=
  modes.all fun mode => match three lines mode with
    | .ok (p, q, w, s, _) => p.size == 64 && p == q && q == w && w == s && p.any (!·.isZero)
    | .error _ => false

/-- `foldSplit`'s table count under `mode`. -/
def tables (lines : Array String) (mode : _root_.Flock.Tmpl.Mode) : Nat :=
  match three lines mode with
  | .ok (_, _, _, _, n) => n
  | .error _ => 0

/-- Definition 2, a quad: two pairs in two lanes at 0, its own sum at 6. -/
def quad : List String := ["DEF 7 2 1 1", "6 2 2 5 1 9", "CALL 1 2 0 3 1 1 7 0 2 1"]

/-- A unit of 1300 columns rooted at Definition 3: 128 input bits, its output group at word 9 (column 1152, reading the
first quad's sum), a row at 1100, and `call`. -/
def quads (call : String) (q : String := String.intercalate "\n" quad) (more : List String := []) : Array String :=
  ("flock-ir-tmpl/v2 t 3 1 0 1 1 128 1 9 1 1 2" :: and2 ++ pair ++ q.splitOn "\n" ++
    [s!"DEF 1300 0 130 {1 + more.length}"] ++ (List.range 128).map (fun i => s!"{i} 1 {i} 1 {i}") ++
    ["1152 1 134 1 1299", "1100 3 0 1 0 1 1300", call] ++ more).toArray

/-- `n` quads at 128, copy `j` bound to bits `(j·step, j·step + 1)`. -/
def quadRun (n step : Nat) : String := s!"CALL 2 {n} 128 7 1 2 0 {step} 1 0"

#guard agree tmplLines
#guard agree (tmplLines "CALL 1 1 128 3 1 2 0 1 1 0")
#guard agree (tmplLines "CALL 1 2 250 3 1 2 0 1 1 0")
-- quads: lanes of 2 under lanes of n, below 64 lanes (low parts explicit), reaching 64 and past (summed)
#guard agree (quads (quadRun 1 1))
#guard agree (quads (quadRun 2 1))
#guard agree (quads (quadRun 3 1))
#guard agree (quads (quadRun 32 1))
#guard agree (quads (quadRun 33 1))
#guard agree (quads (quadRun 100 1))
#guard agree (quads (quadRun 100 0))
-- the quad's pairs bound to one of its inputs twice, and the pair's ANDs to the constant
#guard agree (quads (quadRun 100 1) "DEF 7 2 1 1\n6 2 2 5 1 9\nCALL 1 2 0 3 1 1 7 0 2 0")
#guard agree (header :: and2 ++ ["DEF 3 2 1 1", "2 2 1 0 1 5", "CALL 0 2 0 1 1 1 5 0 2 0"] ++ root CALL []).toArray

/-- The root's two pairs at 128 and a third at 200, bound to bits 1 and 2: the pair and its ANDs on two paths. -/
def twoPaths : Array String :=
  (header :: and2 ++ pair ++ ((root CALL []).map fun l => if l == "DEF 400 0 6 1" then "DEF 400 0 6 2" else l) ++
    ["CALL 1 1 200 3 1 2 1 0 1 0"]).toArray

#guard agree twoPaths
-- quads in a second run at 830, below 64 lanes and past it
#guard agree (quads (quadRun 2 1) (more := ["CALL 2 3 830 7 1 2 0 1 1 0"]))
#guard agree (quads (quadRun 32 1) (more := ["CALL 2 33 830 7 1 2 0 2 1 0"]))
#guard agree (quads (quadRun 100 1) (more := ["CALL 2 3 830 7 1 2 5 1 1 0"]))
#guard agree (quads (quadRun 100 0) (more := ["CALL 2 1 830 7 1 2 0 1 1 0"]))
-- the shared quads (and their pairs) are folded as tables, under every mode; `twoPaths`'s pair is too small to be
#guard modes.all fun m => tables (quads (quadRun 100 1) (more := ["CALL 2 3 830 7 1 2 5 1 1 0"])) m == 3
#guard modes.all fun m => tables (quads (quadRun 32 1) (more := ["CALL 2 33 830 7 1 2 0 2 1 0"])) m == 6
#guard modes.all fun m => tables twoPaths m == 0

end Flock.Test.TmplFold
