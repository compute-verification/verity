/-
The lincheck fold of a templated unit, split (PRF-133): `Tmpl.fold`'s sum, with each Definition that the call tree
reaches on more than one path folded once per class into a table and each placement combining the table, so the fold
reads each such Definition's description once per lane scale, not once per call path. No verifier path reads it yet.

A class `(β, as)` of Definition `d` (`Tmpl.fold`'s: copies at `q = a + 2^β · (h + p)`, `as` its low parts while explicit)
whose origin `h` is a multiple of `2^al` splits each local position `P = o + p` at bit `al`: the high part `⌊P / 2^al⌋`
is a *chunk*, the low part `P mod 2^al`. Once `β + al ≥ 6`, `q mod 64` and the coordinates of `⌊q / 64⌋` below
`β + al − 6` read only the low part, so every entry's weight is a low factor, independent of `h`, times
`eq(ρ[β + al − 6 ..], ⌊h / 2^al⌋ + chunk)` for its row and the same of `r` for its column. A walk keeps the low factors in
sparse tables keyed by chunks (`Tab`): own entries by row chunk, column chunk and slot (`P`, explicit) or by the two
chunks (`S`, summed); the constant's by row chunk (`PC`, `C`); an open input's by input, low part and row chunk (`U`).

A Definition reached on more than one path, placed at a multiple of its `palign` (`Lay.palign`, `2^al`), is walked once
per class at `o = 0` (`tabs`); its caller *re-keys* the table into its own chunks: the child's chunk `c` at child-unit
offset `oc` is the caller's chunk `⌊(oc + c · 2^al) / 2^al'⌋`, times `eq` of the coordinates between, `al'` the caller's
granularity in the child's units. Any other call is walked directly, in its caller's chunks. The root is one chunk
(`al = log2ceil` of its size), so its table is the fold. Lane runs, the switch from explicit low parts to summed ones and
the bindings are `Tmpl.fold`'s, on tables.

Cost: per class of a Definition reached on more than one path, its own entries and its calls once; per placement of it,
its table's entries (at most `4^T` chunk pairs under layout `hybrid T`, times 64 slots while explicit) and its bound
inputs' table entries times its copies.
-/
import Flock.TmplFold
import Std.Data.HashMap

namespace Flock.Tmpl

open Std

/-- `eq(v[lo .. lo + n], ·)` in two halves of `n / 2` and `n − n / 2` bits. -/
abbrev Half := Nat × Array F128 × Array F128

def eqHalf (v : Array F128) (lo n : Nat) : Half :=
  let h := n / 2
  (h, (Array.range (2 ^ h)).map (eqAt v lo h), (Array.range (2 ^ (n - h))).map (eqAt v (lo + h) (n - h)))

def Half.at (T : Half) (u : Nat) : F128 := T.2.1.getD (u % 2 ^ T.1) 0 * T.2.2.getD (u / 2 ^ T.1) 0

def bump {κ : Type} [BEq κ] [Hashable κ] (m : HashMap κ F128) (k : κ) (v : F128) : HashMap κ F128 :=
  m.insert k (m.getD k 0 + v)

/-- A class's fold in chunks: explicit own entries `P[(row chunk, column chunk, slot)]` and the constant's
`PC[(row chunk, slot)]` (its slot weight `g(qc)` in); summed own entries `S[(row chunk, column chunk)]` and the
constant's `C[row chunk]`; open inputs `U[(input, low part, row chunk)]`. -/
structure Tab where
  P : HashMap (Nat × Nat × Nat) F128 := {}
  PC : HashMap (Nat × Nat) F128 := {}
  S : HashMap (Nat × Nat) F128 := {}
  C : HashMap Nat F128 := {}
  U : HashMap (Nat × Nat × Nat) F128 := {}
  work : Nat := 0
  deriving Inhabited

/-- Add child table `ch` re-keyed by `rm` (row chunks) and `cm` (column chunks), each a caller chunk and its factor,
its summed parts scaled by `ls` (own entries) and `lc` (the constant's). -/
def Tab.addChild (o ch : Tab) (rm cm : Nat → Nat × F128) (ls lc : F128) : Tab :=
  let P := ch.P.fold (fun m (cx, cy, s) v =>
    let (px, fx) := rm cx
    let (py, fy) := cm cy
    bump m (px, py, s) (v * fx * fy)) o.P
  let PC := ch.PC.fold (fun m (cx, s) v => let (px, fx) := rm cx; bump m (px, s) (v * fx)) o.PC
  let S := ch.S.fold (fun m (cx, cy) v =>
    let (px, fx) := rm cx
    let (py, fy) := cm cy
    bump m (px, py) (ls * v * fx * fy)) o.S
  let C := ch.C.fold (fun m cx v => let (px, fx) := rm cx; bump m px (lc * v * fx)) o.C
  { o with P, PC, S, C, work := o.work + ch.work + ch.P.size + ch.PC.size + ch.S.size + ch.C.size }

structure SplitSt where
  tabs : HashMap (Nat × Nat × Array Nat) Tab := {}
  halves : HashMap (Bool × Nat × Nat) Half := {}
  tables : Nat := 0
  tableWork : Nat := 0

abbrev SV := StateT SplitSt V

def half (pt : FoldPt) (rho : Bool) (lo n : Nat) : SV Half := do
  match (← get).halves.get? (rho, lo, n) with
  | some T => return T
  | none =>
    let T := eqHalf (if rho then pt.rho else pt.r) lo n
    modify fun s => { s with halves := s.halves.insert (rho, lo, n) T }
    return T

def ident (c : Nat) : Nat × F128 := (c, 1)

/-- **Walk Definition `d`'s class** `(β, as)` at local offset `off` in chunks of `2^al` (`fuel` bounds the depth;
`multi`: reached on more than one path). A call's class is its table re-keyed if its Definition is reached on more than
one path and placed at a multiple of its `palign`, else walked directly in `d`'s chunks. -/
def walk (t : Tmpl) (lays : Array Lay) (multi : Array Bool) (pt : FoldPt) :
    Nat → Nat → Nat → Array Nat → Nat → Nat → SV Tab
  | 0, _, _, _, _, _ => throw "split fold: calls nest deeper than the definitions"
  | fuel + 1, d, β, as, off, al => do
    let D := t.defs.getD d default
    let L := lays.getD d default
    let summed := 6 ≤ β
    if !summed && β + al < 6 then throw s!"split fold: definition {d}: chunks below 64 slots"
    let B := 2 ^ β
    let A := 2 ^ al
    let nk := if summed then 1 else as.size
    let pos := fun (y : Nat) => off + posIn t lays (t.defs.size + 1) d y
    let lo := if summed then β - 6 else 0
    let n := if summed then al else β + al - 6
    let eL ← half pt true lo n
    let gL ← half pt false lo n
    let rowLow := fun (k P : Nat) =>
      if summed then eL.at (P % A)
      else
        let q := as.getD k 0 + B * (P % A)
        pt.lw.getD (q % 64) 0 * eL.at (q / 64)
    -- caller reference `v` with weight `w` for a row of chunk `cx`, low part `k`
    let ref := fun (o : Tab) (k cx : Nat) (w : F128) (v : Nat) =>
      if v < D.cols then
        let P := pos v
        if summed then { o with S := bump o.S (cx, P / A) (w * gL.at (P % A)) }
        else
          let q := as.getD k 0 + B * (P % A)
          { o with P := bump o.P (cx, P / A, q % 64) (w * gL.at (q / 64)) }
      else if v < D.cols + D.ins then { o with U := bump o.U (v - D.cols, k, cx) w }
      else if summed then { o with C := bump o.C cx w }
      else { o with PC := bump o.PC (cx, pt.qc % 64) (w * pt.g pt.qc) }
    -- callee `e`'s class `(β', as')` at child-unit offset `oc`, `d`'s chunks `2^al'` in child units
    let child := fun (e β' : Nat) (as' : Array Nat) (oc al' : Nat) =>
      (do
        let alc := Nat.log2 (lays.getD e default).palign
        if multi.getD e false && oc % 2 ^ alc == 0 && alc ≤ al' && 6 ≤ β' + alc then
          let T ← match (← get).tabs.get? (e, β', as') with
            | some T => pure T
            | none => do
              let T ← walk t lays multi pt fuel e β' as' 0 alc
              modify fun s => { s with tabs := s.tabs.insert (e, β', as') T, tables := s.tables + 1,
                                       tableWork := s.tableWork + T.work }
              pure T
          let mρ ← half pt true (β' + alc - 6) (al' - alc)
          let mr ← half pt false (β' + alc - 6) (al' - alc)
          let key := fun (M : Half) (c : Nat) =>
            let P := oc + c * 2 ^ alc
            (P / 2 ^ al', M.at (P % 2 ^ al' / 2 ^ alc))
          return ({ T with work := 0 }, key mρ, key mr)
        else
          let T ← walk t lays multi pt fuel e β' as' oc al'
          return (T, ident, ident) : SV (Tab × (Nat → Nat × F128) × (Nat → Nat × F128)))
    let mut o : Tab := {}
    for (x, ra, rb) in D.rows do
      let P := pos x
      for k in [0:nk] do
        let ex := rowLow k P
        for v in ra do o := ref o k (P / A) (pt.alpha * ex) v
        for v in rb do o := ref o k (P / A) ex v
      o := { o with work := o.work + nk * (ra.size + rb.size) }
    for i in [0:D.calls.size] do
      let c := D.calls.getD i default
      let pl := L.places.getD i default
      -- copy `j`'s bindings of `ch`'s open inputs: `wk kc j` the low parts of `d` and weights an entry at `kc` opens to
      let bind := fun (o : Tab) (j : Nat) (ch : Tab) (rm : Nat → Nat × F128) (wk : Nat → Nat → List (Nat × F128)) =>
        let bd := (c.bound id j).toArray
        ch.U.fold (fun o (t', kc, cx) v =>
          let (px, fx) := rm cx
          (wk kc j).foldl (fun o (k, f) => ref o k px (f * v * fx) (bd.getD t' 0)) o) { o with work := o.work + ch.U.size }
      if pl.ms == 1 then
        if summed && c.count != 1 then throw "split fold: copies side by side below the root"
        for j in [0:c.count] do
          let (ch, rm, cm) ← child c.def_ β as (off + pl.nb + j * pl.cs) al
          o := o.addChild ch rm cm 1 1
          o := bind o j ch rm fun kc _ => [(kc, 1)]
      else
        let M := pl.ms
        let m := Nat.log2 M
        if (off + pl.nb) % M != 0 || al < m then throw "split fold: a run of lanes is not aligned to its lanes"
        let oc := (off + pl.nb) / M
        if summed then
          let (ch, rm, cm) ← child c.def_ (β + m) #[] oc (al - m)
          let rb := fun (b : Nat) => pt.rho.getD (β - 6 + b) 0
          let tb := fun (b : Nat) => pt.r.getD (β - 6 + b) 0
          let lrr := laneSum (fun b bit => if bit then rb b * tb b else (rb b + 1) * (tb b + 1)) m c.count
          let lr := laneSum (fun b bit => if bit then rb b else rb b + 1) m c.count
          o := o.addChild ch rm cm lrr lr
          for j in [0:c.count] do
            let ej := eqAt pt.rho (β - 6) m j
            o := bind o j ch rm fun _ _ => [(0, ej)]
        else if β + m < 6 then
          let as' := (Array.range (nk * c.count)).map fun i => as.getD (i / c.count) 0 + B * (i % c.count)
          let (ch, rm, cm) ← child c.def_ (β + m) as' oc (al - m)
          o := o.addChild ch rm cm 0 0
          for j in [0:c.count] do
            o := bind o j ch rm fun kc j => if kc % c.count == j then [(kc / c.count, 1)] else []
        else
          let β' := β + m
          let (ch, rm, cm) ← child c.def_ β' #[] oc (al - m)
          let eLow := fun (a : Nat) => pt.lw.getD (a % 64) 0 * eqAt pt.rho 0 (β' - 6) (a / 64)
          let gLow := fun (a : Nat) => eqAt pt.r 0 (β' - 6) (a / 64)
          let mut W := Array.replicate 64 (0 : F128)
          let mut EL : F128 := 0
          for k in [0:nk] do
            for j in [0:c.count] do
              let a := as.getD k 0 + B * j
              W := addAt W a (eLow a * gLow a)
              EL := EL + eLow a
          let slots := (List.range 64).filter fun s => !(W.getD s 0).isZero
          let P := ch.S.fold (fun mp (cx, cy) v =>
            let (px, fx) := rm cx
            let (py, fy) := cm cy
            slots.foldl (fun mp s => bump mp (px, py, s) ((W.getD s 0) * v * fx * fy)) mp) o.P
          let PC := ch.C.fold (fun mp cx v =>
            let (px, fx) := rm cx
            bump mp (px, pt.qc % 64) (EL * v * fx * pt.g pt.qc)) o.PC
          o := { o with P, PC, work := o.work + ch.work + ch.S.size * slots.length + ch.C.size + nk * c.count }
          for j in [0:c.count] do
            o := bind o j ch rm fun _ j => (List.range nk).map fun k => (k, eLow (as.getD k 0 + B * j))
    return o

/-- **The unit's fold, split**: `Tmpl.fold`'s, with the tables; its work (the walks' entries and bindings, and each
placement's re-keyed entries), the tables, and the work spent building them. -/
def foldSplit (t : Tmpl) (lays : Array Lay) (alpha : F128) (lw rho r : Array F128) :
    V (Array F128 × Nat × Nat × Nat) := do
  let qc := posOf t lays ((t.defs.getD t.root default).cols - 1)
  let pt : FoldPt := { alpha, lw, rho, r, qc }
  let multi := t.paths.map fun n => decide (1 < n)
  let al := log2ceil (lays.getD t.root default).size
  let (o, st) ← (walk t lays multi pt (t.defs.size + 1) t.root 0 #[0] 0 al).run {}
  let lo := al - 6
  let p := o.P.fold (fun p (cx, cy, s) v =>
    p.modify s (· + v * eqTail pt.rhoT lo cx * eqTail pt.rT lo cy)) (Array.replicate 64 (0 : F128))
  let p := o.PC.fold (fun p (cx, s) v => p.modify s (· + v * eqTail pt.rhoT lo cx)) p
  return (addAt p qc ((alpha + 1) * pt.e qc * pt.g qc), o.work + st.tableWork, st.tables, st.tableWork)

end Flock.Tmpl
