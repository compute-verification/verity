/-
The lincheck fold of a templated unit, split: `Tmpl.fold`'s sum, with each Definition that the call tree
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

/-- A row's low factor at local position `P`, class copy `k`: `eq(ρ[β−6 ..], P mod 2^al)` summed, else
`lw[q mod 64] · eq(ρ, ⌊q/64⌋)` at the low part `q = as[k] + 2^β (P mod 2^al)`. -/
def Tab.rowLow (pt : FoldPt) (summed : Bool) (A B : Nat) (as : Array Nat) (eL : Half) (k P : Nat) : F128 :=
  if summed then eL.at (P % A)
  else
    let q := as.getD k 0 + B * (P % A)
    pt.lw.getD (q % 64) 0 * eL.at (q / 64)

/-- Caller reference `v` with weight `w` for a row of chunk `cx`, low part `k`: an own column at local position `pos v`
adds at its column chunk (and slot, while explicit), an input opens, the constant adds at the row chunk. -/
def Tab.ref (D : Def) (pt : FoldPt) (summed : Bool) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (o : Tab) (k cx : Nat) (w : F128) (v : Nat) : Tab :=
  if v < D.cols then
    let P := pos v
    if summed then { o with S := bump o.S (cx, P / A) (w * gL.at (P % A)) }
    else
      let q := as.getD k 0 + B * (P % A)
      { o with P := bump o.P (cx, P / A, q % 64) (w * gL.at (q / 64)) }
  else if v < D.cols + D.ins then { o with U := bump o.U (v - D.cols, k, cx) w }
  else if summed then { o with C := bump o.C cx w }
  else { o with PC := bump o.PC (cx, pt.qc % 64) (w * pt.g pt.qc) }

/-- Copy `j`'s bindings of child table `ch`'s open inputs through `ref`: `wk kc j` the low parts and weights an entry at
`kc` opens to, its row chunk re-keyed by `rm`. -/
def Tab.bind (ref : Tab → Nat → Nat → F128 → Nat → Tab) (c : Call) (o : Tab) (j : Nat) (ch : Tab)
    (rm : Nat → Nat × F128) (wk : Nat → Nat → List (Nat × F128)) : Tab :=
  let bd := (c.bound id j).toArray
  ch.U.fold (fun o (t', kc, cx) v =>
    let (px, fx) := rm cx
    (wk kc j).foldl (fun o (k, f) => ref o k px (f * v * fx) (bd.getD t' 0)) o) { o with work := o.work + ch.U.size }

/-- Copies `0 .. count`' bindings through `bind`. -/
def Tab.binds (bind : Tab → Nat → Tab → (Nat → Nat × F128) → (Nat → Nat → List (Nat × F128)) → Tab) (o : Tab)
    (count : Nat) (ch : Tab) (rm : Nat → Nat × F128) (wk : Nat → Nat → List (Nat × F128)) : Tab :=
  (List.range count).foldl (fun o j => bind o j ch rm wk) o

/-- Definition `d`'s own rows: copy `k`'s references through `ref`, an `A` reference weighted `α`. -/
def Tab.rows (ref : Tab → Nat → Nat → F128 → Nat → Tab) (rowLow : Nat → Nat → F128) (alpha : F128) (nk A : Nat)
    (pos : Nat → Nat) (rows : Array (Nat × Array Nat × Array Nat)) (o : Tab) : Tab :=
  rows.foldl (fun o (x, ra, rb) =>
    let P := pos x
    let o := (List.range nk).foldl (fun o k =>
      let ex := rowLow k P
      let o := ra.foldl (fun o v => ref o k (P / A) (alpha * ex) v) o
      rb.foldl (fun o v => ref o k (P / A) ex v) o) o
    { o with work := o.work + nk * (ra.size + rb.size) }) o

/-- A run of lanes from an explicit class into a summed one: the child's summed entries spread over the 64 slots by `W`
(`W[s]`: `Σ E(a) · eq(r, ⌊a/64⌋)` over the lanes' low parts `a ≡ s`), the constant's by `EL` (`Σ E(a)`); `n` lanes. -/
def Tab.slots (pt : FoldPt) (W : Array F128) (EL : F128) (n : Nat) (o ch : Tab) (rm cm : Nat → Nat × F128) : Tab :=
  let slots := (List.range 64).filter fun s => !(W.getD s 0).isZero
  let P := ch.S.fold (fun mp (cx, cy) v =>
    let (px, fx) := rm cx
    let (py, fy) := cm cy
    slots.foldl (fun mp s => bump mp (px, py, s) ((W.getD s 0) * v * fx * fy)) mp) o.P
  let PC := ch.C.fold (fun mp cx v =>
    let (px, fx) := rm cx
    bump mp (px, pt.qc % 64) (EL * v * fx * pt.g pt.qc)) o.PC
  { o with P, PC, work := o.work + ch.work + ch.S.size * slots.length + ch.C.size + n }

/-- Callee `e`'s class `(β', as')` table at offset 0 in chunks of `2^alc`: cached, else walked by `w` and cached. -/
def cachedTab (w : Nat → Nat → Array Nat → Nat → Nat → SV Tab) (e β' : Nat) (as' : Array Nat) (alc : Nat) : SV Tab := do
  match (← get).tabs.get? (e, β', as') with
  | some T => pure T
  | none =>
    let T ← w e β' as' 0 alc
    modify fun s => { s with tabs := s.tabs.insert (e, β', as') T, tables := s.tables + 1,
                             tableWork := s.tableWork + T.work }
    pure T

/-- A child's chunk `c` (of `2^alc`) at child-unit offset `oc`: its caller's chunk (of `2^al'`), and `M`'s factor for
the coordinates between. -/
def rekeyAt (oc alc al' : Nat) (M : Half) (c : Nat) : Nat × F128 :=
  let P := oc + c * 2 ^ alc
  (P / 2 ^ al', M.at (P % 2 ^ al' / 2 ^ alc))

/-- Callee `e`'s class `(β', as')` at child-unit offset `oc`, its caller's chunks `2^al'` in child units: its table
(`cachedTab`) re-keyed if `e` is reached on more than one path and placed at a multiple of its `palign`, else walked by
`w` directly in the caller's chunks. -/
def walkChild (lays : Array Lay) (multi : Array Bool) (pt : FoldPt) (w : Nat → Nat → Array Nat → Nat → Nat → SV Tab)
    (e β' : Nat) (as' : Array Nat) (oc al' : Nat) : SV (Tab × (Nat → Nat × F128) × (Nat → Nat × F128)) := do
  let alc := Nat.log2 (lays.getD e default).palign
  if multi.getD e false && oc % 2 ^ alc == 0 && alc ≤ al' && 6 ≤ β' + alc then
    let T ← cachedTab w e β' as' alc
    let mρ ← half pt true (β' + alc - 6) (al' - alc)
    let mr ← half pt false (β' + alc - 6) (al' - alc)
    return ({ T with work := 0 }, rekeyAt oc alc al' mρ, rekeyAt oc alc al' mr)
  else
    let T ← w e β' as' oc al'
    return (T, ident, ident)

/-- Call `i` of Definition `d`'s class `(β, as)` at local offset `off` in chunks of `2^al`, its column halves `gL`:
its callee's tables (`child`) added to `o`, and their open inputs bound. -/
def walkCall (t : Tmpl) (lays : Array Lay) (pt : FoldPt)
    (child : Nat → Nat → Array Nat → Nat → Nat → SV (Tab × (Nat → Nat × F128) × (Nat → Nat × F128)))
    (d β : Nat) (as : Array Nat) (off al : Nat) (gL : Half) (o : Tab) (i : Nat) : SV Tab := do
  let D := t.defs.getD d default
  let summed := 6 ≤ β
  let B := 2 ^ β
  let nk := if summed then 1 else as.size
  let c := D.calls.getD i default
  let pl := (lays.getD d default).places.getD i default
  let bind := Tab.bind (Tab.ref D pt summed (2 ^ al) B as gL fun y => off + posIn t lays (t.defs.size + 1) d y) c
  if pl.ms == 1 then
    if summed && c.count != 1 then throw "split fold: copies side by side below the root"
    (List.range c.count).foldlM (fun o j => do
      let (ch, rm, cm) ← child c.def_ β as (off + pl.nb + j * pl.cs) al
      pure (bind (o.addChild ch rm cm 1 1) j ch rm fun kc _ => [(kc, 1)])) o
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
      pure (Tab.binds bind (o.addChild ch rm cm lrr lr) c.count ch rm fun _ j => [(0, eqAt pt.rho (β - 6) m j)])
    else if β + m < 6 then
      let as' := (Array.range (nk * c.count)).map fun i => as.getD (i / c.count) 0 + B * (i % c.count)
      let (ch, rm, cm) ← child c.def_ (β + m) as' oc (al - m)
      pure (Tab.binds bind (o.addChild ch rm cm 0 0) c.count ch rm
        fun kc j => if kc % c.count == j then [(kc / c.count, 1)] else [])
    else
      let β' := β + m
      let (ch, rm, cm) ← child c.def_ β' #[] oc (al - m)
      let eLow := fun (a : Nat) => pt.lw.getD (a % 64) 0 * eqAt pt.rho 0 (β' - 6) (a / 64)
      let lows := (List.range nk).flatMap fun k => (List.range c.count).map fun j => as.getD k 0 + B * j
      let W := lows.foldl (fun W a => addAt W a (eLow a * eqAt pt.r 0 (β' - 6) (a / 64))) (Array.replicate 64 0)
      let EL := lows.foldl (fun EL a => EL + eLow a) 0
      pure (Tab.binds bind (Tab.slots pt W EL (nk * c.count) o ch rm cm) c.count ch rm fun _ j =>
        (List.range nk).map fun k => (k, eLow (as.getD k 0 + B * j)))

/-- **Walk Definition `d`'s class** `(β, as)` at local offset `off` in chunks of `2^al` (`fuel` bounds the depth;
`multi`: reached on more than one path): its own rows, then each call's (`walkCall`), a call's class `walkChild`'s. -/
def walk (t : Tmpl) (lays : Array Lay) (multi : Array Bool) (pt : FoldPt) :
    Nat → Nat → Nat → Array Nat → Nat → Nat → SV Tab
  | 0, _, _, _, _, _ => throw "split fold: calls nest deeper than the definitions"
  | fuel + 1, d, β, as, off, al => do
    let D := t.defs.getD d default
    let summed := 6 ≤ β
    if !summed && β + al < 6 then throw s!"split fold: definition {d}: chunks below 64 slots"
    let lo := if summed then β - 6 else 0
    let n := if summed then al else β + al - 6
    let eL ← half pt true lo n
    let gL ← half pt false lo n
    let pos := fun (y : Nat) => off + posIn t lays (t.defs.size + 1) d y
    let o := Tab.rows (Tab.ref D pt summed (2 ^ al) (2 ^ β) as gL pos) (Tab.rowLow pt summed (2 ^ al) (2 ^ β) as eL)
      pt.alpha (if summed then 1 else as.size) (2 ^ al) pos D.rows {}
    (List.range D.calls.size).foldlM
      (walkCall t lays pt (walkChild lays multi pt (walk t lays multi pt fuel)) d β as off al gL) o

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

/-- **A templated unit slot's fold** at a slot of `2^sl` bits: `foldSplit`, refused unless the point is the slot's
(`|lw| = 64`, `|ρ| + 6 = sl = |r| + 6`, `sl ≤ 32`) and the root is one chunk of at least 64 and at most `2^sl` columns,
the bounds under which it is `project (baseOf α (lw ⊗ eq(ρ, ·)) A B none sl) (eqTable r)` on `unrollAt`'s net. -/
def foldUnit (t : Tmpl) (lays : Array Lay) (sl : Nat) (alpha : F128) (lw rho r : Array F128) : V (Array F128) := do
  if lw.size != 64 || rho.size + 6 != sl || r.size != rho.size || 32 < sl then
    throw "template fold: the point is not the slot's"
  let al := log2ceil (lays.getD t.root default).size
  if al < 6 || sl < al then throw "template fold: the root is not one chunk of 64 to 2^sl columns"
  let (p, _) ← foldSplit t lays alpha lw rho r
  return p

end Flock.Tmpl
