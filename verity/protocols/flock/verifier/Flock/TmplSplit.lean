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
module

public import Flock.TmplFold
public import Std.Data.HashMap.Basic

@[expose] public section

namespace Flock.Tmpl

open Std

/-- `eq(v[lo .. lo + n], ·)` in two halves of `n / 2` and `n − n / 2` bits. -/
abbrev Half := Nat × Array F128 × Array F128

def eqHalf (v : Array F128) (lo n : Nat) : Half :=
  let h := n / 2
  (h, (Array.range (2 ^ h)).map (eqAt v lo h), (Array.range (2 ^ (n - h))).map (eqAt v (lo + h) (n - h)))

def Half.at (T : Half) (u : Nat) : F128 := T.2.1.getD (u % 2 ^ T.1) 0 * T.2.2.getD (u / 2 ^ T.1) 0

/-- `Half.at` by shifts: `Nat.pow` builds a GMP number on every call. -/
def Half.atFast (T : Half) (u : Nat) : F128 := T.2.1.getD (u &&& ((1 <<< T.1) - 1)) 0 * T.2.2.getD (u >>> T.1) 0

@[csimp] theorem Half.at_eq_atFast : @Half.at = @Half.atFast := by
  funext T u
  simp only [Half.at, Half.atFast, Nat.one_shiftLeft, Nat.and_two_pow_sub_one_eq_mod, Nat.shiftRight_eq_div_pow]

def bump {κ : Type} [BEq κ] [Hashable κ] (m : HashMap κ F128) (k : κ) (v : F128) : HashMap κ F128 :=
  m.insert k (m.getD k 0 + v)

/-- `f y`, read from `a` below its size: `tabled ((Array.range n).map f) f` is `f` with its values on `0 .. n` computed
once. Its caller builds the table: a definition that built it and returned the lookup would be compiled taking `y` too,
rebuilding the table on every call. -/
@[specialize]
def tabled {α : Type} [Inhabited α] (a : Array α) (f : Nat → α) (y : Nat) : α :=
  if y < a.size then a.getD y default else f y

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

/-- A row's low factor at local position `P`, class copy `k`, from its half's values `eAt`: `eq(ρ[β−6 ..], P mod 2^al)`
summed, else `lw[q mod 64] · eq(ρ, ⌊q/64⌋)` at the low part `q = as[k] + 2^β (P mod 2^al)`. -/
@[specialize]
def Tab.rowLow (pt : FoldPt) (summed : Bool) (A B : Nat) (as : Array Nat) (eAt : Nat → F128) (k P : Nat) : F128 :=
  if summed then eAt (P % A)
  else
    let q := as.getD k 0 + B * (P % A)
    pt.lw.getD (q % 64) 0 * eAt (q / 64)

/-- Caller reference `v` with weight `w` for a row of chunk `cx`, low part `k`: an own column at local position `pos v`
adds at its column chunk (and slot, while explicit), an input opens, the constant adds at the row chunk. -/
@[specialize]
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

/-- `Tab.bind` given only the child's open inputs `chU`. -/
def Tab.bindU (ref : Tab → Nat → Nat → F128 → Nat → Tab) (c : Call) (o : Tab) (j : Nat)
    (chU : HashMap (Nat × Nat × Nat) F128) (rm : Nat → Nat × F128) (wk : Nat → Nat → List (Nat × F128)) : Tab :=
  let bd := (c.bound id j).toArray
  chU.fold (fun o (t', kc, cx) v =>
    let (px, fx) := rm cx
    (wk kc j).foldl (fun o (k, f) => ref o k px (f * v * fx) (bd.getD t' 0)) o) { o with work := o.work + chU.size }

/-- `Tab.bind` through `Tab.bindU`: handed the whole child, the compiled code builds the result in the child's cell and
borrows `o`, so the first insert into `o`'s tables copies them whole. -/
def Tab.bindFast (ref : Tab → Nat → Nat → F128 → Nat → Tab) (c : Call) (o : Tab) (j : Nat) (ch : Tab)
    (rm : Nat → Nat × F128) (wk : Nat → Nat → List (Nat × F128)) : Tab :=
  Tab.bindU ref c o j ch.U rm wk

@[csimp] theorem Tab.bind_eq_bindFast : @Tab.bind = @Tab.bindFast := rfl

/-- Copies `0 .. count`' bindings through `bind`. -/
def Tab.binds (bind : Tab → Nat → Tab → (Nat → Nat × F128) → (Nat → Nat → List (Nat × F128)) → Tab) (o : Tab)
    (count : Nat) (ch : Tab) (rm : Nat → Nat × F128) (wk : Nat → Nat → List (Nat × F128)) : Tab :=
  (List.range count).foldl (fun o j => bind o j ch rm wk) o

/-- Add a run's sums `acc` for a row of chunk `cx` and column chunk `cy`, weight `w`: `w · acc[s]` at slot `s < 64`
while explicit, `w · acc[0]` summed, each nonzero sum once. -/
def Tab.flush (summed : Bool) (o : Tab) (cx cy : Nat) (w : F128) (acc : Array F128) : Tab :=
  if summed then
    let x := acc.getD 0 0
    if x.isZero then o else { o with S := bump o.S (cx, cy) (w * x) }
  else
    { o with P := (List.range 64).foldl (fun m s =>
        let x := acc.getD s 0
        if x.isZero then m else bump m (cx, cy, s) (w * x)) o.P }

/-- One reference of `Tab.refs` on its table, run chunk and sums: an own column's factor summed at its slot (`0`
summed), a new run once its column chunk changes, any other reference through `Tab.ref`. -/
@[specialize]
def Tab.refStep (D : Def) (pt : FoldPt) (summed : Bool) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (k cx : Nat) (w : F128) (st : Tab × Nat × Array F128) (v : Nat) : Tab × Nat × Array F128 :=
  if v < D.cols then
    let P := pos v
    let q := as.getD k 0 + B * (P % A)
    let i := if summed then 0 else q % 64
    let g := if summed then gL.at (P % A) else gL.at (q / 64)
    if P / A == st.2.1 then (st.1, st.2.1, st.2.2.modify i (· + g))
    else (Tab.flush summed st.1 cx st.2.1 w st.2.2, P / A, (Array.replicate 64 0).modify i (· + g))
  else (Tab.ref D pt summed A B as gL pos st.1 k cx w v, st.2.1, st.2.2)

/-- Copy `k`'s references `vs` of a row of chunk `cx`, weight `w`, as `Tab.ref` adds them one at a time. A row of 64
or more sums an own column's factors along each run of references in one column chunk (`Tab.refStep`) and adds each
run's sums once (`Tab.flush`): a table read's row of thousands of references costs a sum per reference and at most 64
table entries. -/
@[specialize]
def Tab.refs (D : Def) (pt : FoldPt) (summed : Bool) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (o : Tab) (k cx : Nat) (w : F128) (vs : Array Nat) : Tab :=
  if vs.size < 64 then vs.foldl (fun o v => Tab.ref D pt summed A B as gL pos o k cx w v) o
  else
    let st := vs.foldl (Tab.refStep D pt summed A B as gL pos k cx w) (o, 0, Array.replicate 64 0)
    Tab.flush summed st.1 cx st.2.1 w st.2.2

/-- Add a run's sums `acc` for a row of chunk `cx` and column chunk `cy` at each copy `k < nk`, weight `ws[k]`:
`ws[k] · acc[t]` at slot `as[k] + t`, each nonzero sum. -/
def Tab.flushK (as : Array Nat) (nk : Nat) (o : Tab) (cx cy : Nat) (ws acc : Array F128) : Tab :=
  { o with P := (List.range 64).foldl (fun m t =>
      let x := acc.getD t 0
      if x.isZero then m
      else (List.range nk).foldl (fun m k => bump m (cx, cy, as.getD k 0 + t) (ws.getD k 0 * x)) m) o.P }

/-- One reference of `Tab.refsK`: an own column's factor `fac v`, which is `gL.at (B · (pos v mod A) / 64)`, summed at
its slot past the copies' low parts, a new run once its column chunk changes, any other reference through `Tab.ref` at
each copy. -/
@[specialize]
def Tab.refStepK (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (nk cx : Nat) (ws : Array F128) (st : Tab × Nat × Array F128) (v : Nat) : Tab × Nat × Array F128 :=
  if v < D.cols then
    let P := pos v
    let u := B * (P % A)
    if P / A == st.2.1 then (st.1, st.2.1, st.2.2.modify (u % 64) (· + fac v))
    else (Tab.flushK as nk st.1 cx st.2.1 ws st.2.2, P / A, (Array.replicate 64 0).modify (u % 64) (· + fac v))
  else ((List.range nk).foldl (fun o k => Tab.ref D pt false A B as gL pos o k cx (ws.getD k 0) v) st.1, st.2.1, st.2.2)

/-- Add pending sums `pd` of a row chunk `cx` and column chunk `cy`: `pd[s]` at slot `s < 64`, each nonzero sum once. -/
def Tab.flushP (o : Tab) (cx cy : Nat) (pd : Array F128) : Tab :=
  { o with P := (List.range 64).foldl (fun m s =>
      let x := pd.getD s 0
      if x.isZero then m else bump m (cx, cy, s) x) o.P }

/-- `addK as ws t f n k pd`: `ws[j] · f` added at slot `as[j] + t` of `pd` for the `n` copies `j` from `k` on. -/
def addK (as : Array Nat) (ws : Array F128) (t : Nat) (f : F128) : Nat → Nat → Array F128 → Array F128
  | 0, _, pd => pd
  | n + 1, k, pd => addK as ws t f n (k + 1) (pd.modify (as.getD k 0 + t) (· + ws.getD k 0 * f))

/-- Slot `t` of `addRun`. -/
@[inline]
def addRunStep (as : Array Nat) (nk : Nat) (ws acc : Array F128) (pd : Array F128) (t : Nat) : Array F128 :=
  let x := acc.getD t 0
  if x.isZero then pd else addK as ws t x nk 0 pd

/-- A run's sums `acc` at every copy `k < nk`, weight `ws[k]`, added to pending sums `pd`: `ws[k] · acc[t]` at slot
`as[k] + t` (`addK`), each nonzero sum. -/
def addRun (as : Array Nat) (nk : Nat) (ws acc : Array F128) (pd : Array F128) : Array F128 :=
  (List.range 64).foldl (addRunStep as nk ws acc) pd

/-- Add a run's sums `acc` of a row chunk `cx` and column chunk `cy` to pending sums `(o, cx', cy', pd)` (`addRun`)
once they are at the run's chunks, those of other chunks first added (`Tab.flushP`). -/
def Tab.flushKP (as : Array Nat) (nk : Nat) (st : Tab × Nat × Nat × Array F128) (cx cy : Nat) (ws acc : Array F128) :
    Tab × Nat × Nat × Array F128 :=
  let (o, px, py, pd) := st
  if px == cx && py == cy then (o, px, py, addRun as nk ws acc pd)
  else (Tab.flushP o px py pd, cx, cy, addRun as nk ws acc (Array.replicate 64 0))

/-- An explicit row's references `vs` at every copy `k < nk` at once, weight `ws[k]`, while each copy's low part is
below `B` and `B` divides 64, on pending sums: an own column's slot is then its copy's low part plus a slot common to the
copies, and its factor is common, so a run's sums are formed once (`Tab.refStepK`) and added at each copy, a run before
the row's last to the table (`Tab.flushK`) and the last to the pending sums (`Tab.flushKP`), which the rows after it at
the same chunks add to. -/
@[specialize]
def Tab.refsK (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (nk cx : Nat) (ws : Array F128) (st : Tab × Nat × Nat × Array F128) (vs : Array Nat) :
    Tab × Nat × Nat × Array F128 :=
  let (o, px, py, pd) := st
  let r := vs.foldl (Tab.refStepK D pt A B as gL pos fac nk cx ws) (o, py, Array.replicate 64 0)
  Tab.flushKP as nk (r.1, px, py, pd) cx r.2.1 ws r.2.2

/-- One reference of a row of chunk `cx` at every copy, on pending sums `(o, cx', cy', pd)` carried across rows: an own
column's factor `fac v` times each copy's weight `ws[k]`, added at its slot (`addK`) once the pending sums are at its
chunks, those of other chunks first added (`Tab.flushP`); any other reference through `Tab.ref` at each copy. -/
@[specialize]
def Tab.refDK (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (nk cx : Nat) (ws : Array F128) (st : Tab × Nat × Nat × Array F128) (v : Nat) : Tab × Nat × Nat × Array F128 :=
  let (o, px, py, pd) := st
  if v < D.cols then
    let P := pos v
    let t := B * (P % A) % 64
    if px == cx && py == P / A then (o, px, py, addK as ws t (fac v) nk 0 pd)
    else (Tab.flushP o px py pd, cx, P / A, addK as ws t (fac v) nk 0 (Array.replicate 64 0))
  else ((List.range nk).foldl (fun o k => Tab.ref D pt false A B as gL pos o k cx (ws.getD k 0) v) o, px, py, pd)

/-- A row's references `vs` of chunk `cx` at every copy, weight `ws[k]`, on pending sums: one at a time (`Tab.refDK`)
while under 64 at a unit low part (`B = 1`), whose references' slots are their positions' and rarely shared; else summed
in runs (`Tab.refsK`), whose products serve every reference of a slot: at `B = 64` a row's references share one. -/
@[specialize]
def Tab.halfK (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (nk cx : Nat) (ws : Array F128) (st : Tab × Nat × Nat × Array F128) (vs : Array Nat) :
    Tab × Nat × Nat × Array F128 :=
  if vs.size < 64 && B == 1 then vs.foldl (Tab.refDK D pt A B as gL pos fac nk cx ws) st
  else Tab.refsK D pt A B as gL pos fac nk cx ws st vs

/-- One explicit row `(x, ra, rb)` at every copy on pending sums: its weights formed once, an `A` reference's times `α`,
its references through `Tab.halfK`. -/
@[specialize]
def Tab.rowK (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (rowLow : Nat → Nat → F128) (alpha : F128) (nk : Nat) (st : Tab × Nat × Nat × Array F128)
    (r : Nat × Array Nat × Array Nat) : Tab × Nat × Nat × Array F128 :=
  let (x, ra, rb) := r
  let P := pos x
  let ex := (Array.range nk).map fun k => rowLow k P
  let st := Tab.halfK D pt A B as gL pos fac nk (P / A) (ex.map (alpha * ·)) st ra
  let st := Tab.halfK D pt A B as gL pos fac nk (P / A) ex st rb
  ({ st.1 with work := st.1.work + nk * (ra.size + rb.size) }, st.2)

/-- Explicit rows at every copy at once (each copy's low part below `B`, `B` dividing 64), row by row (`Tab.rowK`) on
pending sums carried from row to row and added at the end: a row of a few references costs a product and a sum per
reference at each copy and no table entry. -/
@[specialize]
def Tab.rowsK (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (rowLow : Nat → Nat → F128) (alpha : F128) (nk : Nat) (rows : Array (Nat × Array Nat × Array Nat)) (o : Tab) :
    Tab :=
  let st := rows.foldl (Tab.rowK D pt A B as gL pos fac rowLow alpha nk) (o, 0, 0, Array.replicate 64 0)
  Tab.flushP st.1 st.2.1 st.2.2.1 st.2.2.2

/-- One reference of `Tab.rowsT`'s rows, weight `w`: an own column's weight summed into its column's `W[v]`, any
other reference through `Tab.ref` at copy `0`. The sum is `w + W[v]`, so the compiled code reuses `W[v]`'s cell, which
the array alone holds, where `W[v] + w` would allocate one for each reference. -/
@[specialize]
def Tab.colT (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (cx : Nat) (w : F128)
    (st : Tab × Array F128) (v : Nat) : Tab × Array F128 :=
  let (o, W) := st
  if v < D.cols then (o, W.modify v (w + ·)) else (Tab.ref D pt false A B as gL pos o 0 cx w v, W)

/-- One row `(x, ra, rb)` of `Tab.rowsT`: its weight formed once, an `A` reference's times `α`, its references through
`Tab.colT`. -/
@[specialize]
def Tab.rowT (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (rowLow : Nat → Nat → F128) (alpha : F128) (cx : Nat) (st : Tab × Array F128) (r : Nat × Array Nat × Array Nat) :
    Tab × Array F128 :=
  let (x, ra, rb) := r
  let ex := rowLow 0 (pos x)
  let st := ra.foldl (Tab.colT D pt A B as gL pos cx (alpha * ex)) st
  let st := rb.foldl (Tab.colT D pt A B as gL pos cx ex) st
  ({ st.1 with work := st.1.work + (ra.size + rb.size) }, st.2)

/-- Own column `v`'s sum `x` at copy `0`, on pending sums: `x · fac v` added at its slot once the pending sums are at its
chunks, as `Tab.refDK` adds it with weights `#[x]`. The sum is `x · fac v + pd[t]`, so the compiled code reuses
`pd[t]`'s cell. -/
@[specialize]
def Tab.colD (A B : Nat) (as : Array Nat) (pos : Nat → Nat) (fac : Nat → F128) (cx : Nat) (x : F128)
    (st : Tab × Nat × Nat × Array F128) (v : Nat) : Tab × Nat × Nat × Array F128 :=
  let (o, px, py, pd) := st
  let P := pos v
  let t := as.getD 0 0 + B * (P % A) % 64
  if px == cx && py == P / A then (o, px, py, pd.modify t (x * fac v + ·))
  else (Tab.flushP o px py pd, cx, P / A, (Array.replicate 64 0).modify t (x * fac v + ·))

/-- `Tab.rowsT`'s columns `v, v + 1, …`, `n` of them: each nonzero sum `W[v]` times its factor at its slot, on pending
sums (`Tab.colD`). -/
@[specialize]
def Tab.colsP (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (cx : Nat) (W : Array F128) : Nat → Nat → Tab × Nat × Nat × Array F128 → Tab × Nat × Nat × Array F128
  | 0, _, st => st
  | n + 1, v, st =>
    let x := W.getD v 0
    Tab.colsP D pt A B as gL pos fac cx W n (v + 1) (if x.isZero then st else Tab.colD A B as pos fac cx x st v)

/-- `Tab.colsP` taking each column's sum out of `W` as it reads it, so the compiled code frees the sum's cell while it is
in cache rather than in one pass over `W` when `W` is dropped. -/
@[specialize]
def Tab.colsPFast (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (fac : Nat → F128) (cx : Nat) : Array F128 → Nat → Nat → Tab × Nat × Nat × Array F128 → Tab × Nat × Nat × Array F128
  | _, 0, _, st => st
  | W, n + 1, v, st =>
    let x := W.getD v 0
    Tab.colsPFast D pt A B as gL pos fac cx (W.set! v 0) n (v + 1)
      (if x.isZero then st else Tab.colD A B as pos fac cx x st v)

theorem Tab.colsPFast_eq (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (fac : Nat → F128) (cx : Nat) (W : Array F128) :
    ∀ (n v : Nat) (st : Tab × Nat × Nat × Array F128) (W' : Array F128), (∀ j, v ≤ j → W'.getD j 0 = W.getD j 0) →
      Tab.colsPFast D pt A B as gL pos fac cx W' n v st = Tab.colsP D pt A B as gL pos fac cx W n v st
  | 0, _, _, _, _ => rfl
  | n + 1, v, st, W', h => by
    simp only [Tab.colsPFast, Tab.colsP, h v (Nat.le_refl v)]
    exact Tab.colsPFast_eq D pt A B as gL pos fac cx W n (v + 1) _ _ fun j hj => by
      rw [← h j (by omega)]
      simp [Array.getD_eq_getD_getElem?, show v ≠ j by omega]

@[csimp] theorem Tab.colsP_eq_colsPFast : @Tab.colsP = @Tab.colsPFast := by
  funext D pt A B as gL pos fac cx W n v st
  exact (Tab.colsPFast_eq D pt A B as gL pos fac cx W n v st W fun _ _ => rfl).symm

/-- Explicit rows of one row chunk `cx` at one copy, column by column, as the flat fold's `Sparse.foldT` folds: each
row's weight summed per own column (`Tab.rowT`), then each column's sum times its factor (`Tab.colsP`), so a
reference costs a sum and a column a product. -/
@[specialize]
def Tab.rowsT (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (rowLow : Nat → Nat → F128) (alpha : F128) (cx : Nat) (rows : Array (Nat × Array Nat × Array Nat)) (o : Tab) :
    Tab :=
  let st := rows.foldl (Tab.rowT D pt A B as gL pos rowLow alpha cx) (o, Array.replicate D.cols 0)
  let st := Tab.colsP D pt A B as gL pos fac cx st.2 D.cols 0 (st.1, 0, 0, Array.replicate 64 0)
  Tab.flushP st.1 st.2.1 st.2.2.1 st.2.2.2

/-- Explicit rows at every copy at once (each copy's low part below `B`, `B` dividing 64): column by column
(`Tab.rowsT`) at one copy whose rows are of one row chunk and make at least one reference per column, else row by row
(`Tab.rowsK`). -/
@[specialize]
def Tab.rowsC (D : Def) (pt : FoldPt) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat) (fac : Nat → F128)
    (rowLow : Nat → Nat → F128) (alpha : F128) (nk : Nat) (rows : Array (Nat × Array Nat × Array Nat)) (o : Tab) :
    Tab :=
  let cx := pos (rows.getD 0 (0, #[], #[])).1 / A
  if nk == 1 && D.cols ≤ rows.foldl (fun n r => n + (r.2.1.size + r.2.2.size)) 0 &&
      rows.all (fun r => pos r.1 / A == cx) then
    Tab.rowsT D pt A B as gL pos fac rowLow alpha cx rows o
  else Tab.rowsK D pt A B as gL pos fac rowLow alpha nk rows o

/-- Definition `d`'s own rows: copy `k`'s references through `Tab.refs`, an `A` reference weighted `α`; all copies at
once (`Tab.rowsC`, an own column's factor `fac`) while explicit with every copy's low part below `B` and `B` dividing
64. Specialized at `walk`'s `pos`, `fac` and `rowLow`, as are `Tab.ref`, `Tab.refs`, `Tab.refStep`, `Tab.rowsC`,
`Tab.rowsT`, `Tab.rowT`, `Tab.colT`, `Tab.colsP`, `Tab.colD`, `Tab.rowsK`, `Tab.rowK`, `Tab.halfK`, `Tab.refDK`,
`Tab.refsK`, `Tab.refStepK`, `Tab.rowLow` and `tabled`, so a reference reads its position and factor from their tables
without a closure call. -/
@[specialize]
def Tab.rows (D : Def) (pt : FoldPt) (summed : Bool) (A B : Nat) (as : Array Nat) (gL : Half) (pos : Nat → Nat)
    (fac : Nat → F128) (rowLow : Nat → Nat → F128) (alpha : F128) (nk : Nat)
    (rows : Array (Nat × Array Nat × Array Nat)) (o : Tab) : Tab :=
  if !summed && 64 % B == 0 && (List.range nk).all (fun k => as.getD k 0 < B) then
    Tab.rowsC D pt A B as gL pos fac rowLow alpha nk rows o
  else
    rows.foldl (fun o (x, ra, rb) =>
      let P := pos x
      let o := (List.range nk).foldl (fun o k =>
        let ex := rowLow k P
        let o := Tab.refs D pt summed A B as gL pos o k (P / A) (alpha * ex) ra
        Tab.refs D pt summed A B as gL pos o k (P / A) ex rb) o
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

/-- `rekeyAt` by shifts. -/
def rekeyAtFast (oc alc al' : Nat) (M : Half) (c : Nat) : Nat × F128 :=
  let P := oc + (c <<< alc)
  (P >>> al', M.at ((P &&& ((1 <<< al') - 1)) >>> alc))

@[csimp] theorem rekeyAt_eq_fast : @rekeyAt = @rekeyAtFast := by
  funext oc alc al' M c
  simp only [rekeyAt, rekeyAtFast, Nat.shiftLeft_eq, Nat.one_mul, Nat.and_two_pow_sub_one_eq_mod,
    Nat.shiftRight_eq_div_pow]

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

/-- Definition `d`'s own columns' positions at local offset `off`, `m` of them (none, or all its columns): `fillMap`
over its copy, which writes each column's position once along its layout's runs and its calls' copies, where `posIn`
would search for each. -/
def posTab (t : Tmpl) (lays : Array Lay) (d off m : Nat) : Array Nat :=
  if m = 0 then #[] else fillMap t lays (t.defs.size + 1) d 0 off 1 (Array.replicate m 0)

/-- **Walk Definition `d`'s class** `(β, as)` at local offset `off` in chunks of `2^al` (`fuel` bounds the depth;
`multi`: reached on more than one path): its own rows, then each call's (`walkCall`), a call's class `walkChild`'s. Its
columns' positions (`posTab`), and its halves' values at the low parts of its columns' factors and its rows', are tabled
once (at most one entry per column, and per row for the rows'), so a column or row reads them without a search or a
product. -/
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
    let A := 2 ^ al
    let B := 2 ^ β
    let m := if D.cols ≤ D.rows.foldl (fun r (_, ra, rb) => r + ra.size + rb.size) 0 then D.cols else 0
    let posD := fun (y : Nat) => off + posIn t lays (t.defs.size + 1) d y
    let pa := posTab t lays d off m
    let pos := tabled pa posD
    let nh := if summed then 0 else B * m / 64 + 1
    let gA := (Array.range nh).map gL.at
    let fac := fun (y : Nat) => tabled gA gL.at (B * (pos y % A) / 64)
    let eA := (Array.range (min nh D.rows.size)).map eL.at
    let o := Tab.rows D pt summed A B as gL pos fac (Tab.rowLow pt summed A B as (tabled eA eL.at))
      pt.alpha (if summed then 1 else as.size) D.rows {}
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
