/-
`Net.ofRowsAt` on `Flock.Test.Tmpl`'s unit: at the identity map it is `Tmpl.unroll`'s net; at `Tmpl.posMap` under every
layout mode its rows are the logical rows moved; and it refuses a map that sends two columns to one position, moves an
input row, breaks an output group, or places a column past the constant.
-/
import Flock.TmplAt
import Flock.Test.Tmpl

namespace Flock.Test.NetAt

open Flock Flock.Test.Tmpl

def rowsOf (m : Sparse) : Array (Array Nat) := (Array.range m.rows).map m.row

def logical : Except String (Tmpl.Tmpl × Tmpl.Acc) := do
  let t ← Tmpl.parse tmplLines
  return (t, ← t.rows)

def at_ (pos : Array Nat) : Except String Net := do
  let (t, acc) ← logical
  Net.ofRowsAt t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb pos

def refuses (pos : Array Nat → Array Nat) : Bool :=
  match logical with
  | .ok (t, _) => (at_ (pos (Array.range (t.defs.getD t.root default).cols))).toOption.isNone
  | .error _ => false

-- at the identity, `ofRowsAt` is `unroll`'s net
#guard match logical, (do let (t, _) ← logical; Tmpl.unroll t) with
  | .ok (t, _), .ok n => match at_ (Array.range (t.defs.getD t.root default).cols) with
    | .ok m => same n m
    | .error _ => false
  | _, _ => false

/-- At each layout's map, committed row `pos[x]` is logical row `x` read through `pos`, and every other row is empty. -/
def moved (mode : Tmpl.Mode) : Bool :=
  match (do
    let (t, acc) ← logical
    let lays ← Tmpl.layoutM t mode
    return (t, acc, lays, Tmpl.posMap t lays, ← Tmpl.unrollAt t lays)) with
  | .ok (_, acc, lays, π, n) =>
    let ra := rowsOf n.a
    let rb := rowsOf n.b
    n.useful == (lays.getD 2 default).size &&
      (List.range π.size).all (fun x => ra.getD (π.getD x 0) #[] == (acc.ra.getD x #[]).map (π.getD · 0) &&
        rb.getD (π.getD x 0) #[] == (acc.rb.getD x #[]).map (π.getD · 0)) &&
      (List.range n.useful).all fun p => π.contains p || ((ra.getD p #[]).isEmpty && (rb.getD p #[]).isEmpty)
  | .error _ => false

#guard [.lenient, .strict, .hybrid 0, .hybrid 2].all moved
-- the pair's two copies in two lanes move columns 128 to 133 (`Flock.Test.TmplLayout`)
#guard match logical with
  | .ok (t, _) => (Tmpl.layoutM t .lenient).toOption.any fun l => Tmpl.posMap t l != Array.range 400
  | .error _ => false

#guard refuses fun p => p.set! 129 128
#guard refuses fun p => (p.set! 1 200).set! 200 1
#guard refuses fun p => (p.set! 256 260).set! 260 256
#guard refuses fun p => p.set! 300 500
#guard refuses fun p => p.pop
-- a computed row may move: the first pair's sum (column 130, read by the output at 256) and free column 200 swapped
#guard !refuses fun p => (p.set! 130 200).set! 200 130

-- the layout modes by name
#guard Tmpl.modeOf "iv" == some .lenient && Tmpl.modeOf "iv/strict" == some .strict &&
  Tmpl.modeOf "iv/hybrid" == some (.hybrid 0) && Tmpl.modeOf "iv/hybrid/4" == some (.hybrid 4)
#guard Tmpl.modeOf "iv/hybrid/x" == none && Tmpl.modeOf "hybrid/4" == none

-- `TmplUnit.read`'s net (`Tmpl.unitNet`) is `unrollAt`'s at `layoutM`'s layouts, every field and row, and it refuses a
-- template that is not well-formed
#guard [Tmpl.Mode.lenient, .strict, .hybrid 4].all fun mode => match logical with
  | .ok (t, _) => match TmplUnit.read t mode, (do Tmpl.unrollAt t (← Tmpl.layoutM t mode)) with
    | .ok u, .ok n => decide (u.net = n)
    | _, _ => false
  | .error _ => false
#guard match logical with
  | .ok (t, _) => (TmplUnit.read { t with root := t.defs.size } .lenient).toOption.isNone
  | .error _ => false

end Flock.Test.NetAt
