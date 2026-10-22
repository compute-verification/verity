/-
`Tmpl.fold` against the lincheck's unit fold: on `Flock.Test.Tmpl`'s unit and variants (one copy, copies in lanes that
reach 64 and past, a run of 3 in 4 lanes, a lane binding its caller's input, the constant bound), laid out by
`Tmpl.layout`, the fold from the description equals `foldRef`, the sum over the unrolled rows at their positions, and that
equals `Stmt.project (Stmt.baseOf α (Stmt.tensor lw ρ) A B none sl) (eqTable r)` on `A` and `B` built at the positions
(`posMap`) by `Sparse.ofRows`, the matrices a flat unit's slot would hold.
-/
import Flock.TmplFold
import Flock.Statement
import Flock.Test.TmplLayout

namespace Flock.Test.TmplFold

open Flock Flock.Test.Tmpl

def f (i : Nat) : F128 := ⟨(0x9e3779b97f4a7c15 * (i.toUInt64 + 1)) ^^^ 0x5bd1e995, 0xc2b2ae3d27d4eb4f * (i.toUInt64 + 7)⟩

def lw : Array F128 := (Array.range 64).map fun i => f (100 + i)

/-- The fold three ways, `fold`, `foldRef` and `project ∘ baseOf`, on the template `lines` laid out lenient. -/
def three (lines : Array String) : Except String (Array F128 × Array F128 × Array F128) := do
  let t ← _root_.Flock.Tmpl.parse lines
  let lays ← _root_.Flock.Tmpl.layout t false
  let size := (lays.getD t.root default).size
  let sl := max 7 (log2ceil size)
  let rho := (Array.range (sl - 6)).map fun i => f (200 + i)
  let r := (Array.range (sl - 6)).map fun i => f (300 + i)
  let alpha := f 7
  let (p, _) ← _root_.Flock.Tmpl.fold t lays alpha lw rho r
  let q ← _root_.Flock.Tmpl.foldRef t lays alpha lw rho r
  let acc ← t.rows
  let π := _root_.Flock.Tmpl.posMap t lays
  let at_ := fun (rows : Array (Array Nat)) => Id.run do
    let mut out := Array.replicate size #[]
    for x in [0:rows.size] do out := out.set! (π.getD x 0) ((rows.getD x #[]).map (π.getD · 0))
    return out
  let a ← Sparse.ofRows (at_ acc.ra) (2 ^ sl)
  let b ← Sparse.ofRows (at_ acc.rb) (2 ^ sl)
  let base := Stmt.baseOf alpha (Stmt.tensor lw rho) a b none sl
  return (p, q, Stmt.project base (eqTable r))

def agree (lines : Array String) : Bool :=
  match three lines with
  | .ok (p, q, w) => p.size == 64 && p == q && q == w && p.any (!·.isZero)
  | .error _ => false

/-- Definition 2, a quad: two pairs in two lanes at 0, its own sum at 6. -/
def quad : List String := ["DEF 7 2 1 1", "6 2 2 5 1 9", "CALL 1 2 0 3 1 1 7 0 2 1"]

/-- A unit of 1300 columns rooted at Definition 3: 128 input bits, its output group at word 9 (column 1152, reading the
first quad's sum), a row at 1100, and `call`. -/
def quads (call : String) (q : String := String.intercalate "\n" quad) : Array String :=
  ("flock-ir-tmpl/v2 t 3 1 0 1 1 128 1 9 1 1 2" :: and2 ++ pair ++ q.splitOn "\n" ++
    ["DEF 1300 0 130 1"] ++ (List.range 128).map (fun i => s!"{i} 1 {i} 1 {i}") ++
    ["1152 1 134 1 1299", "1100 3 0 1 0 1 1300", call]).toArray

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

end Flock.Test.TmplFold
