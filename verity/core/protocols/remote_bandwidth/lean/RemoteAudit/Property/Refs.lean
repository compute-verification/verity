import Std.Data.HashMap
import Flock.Program
import RemoteAudit.Property.Table

/-!
# Leaf intervals and widths of a Program as the descriptor states it

`verity.ml.circuits.intervals` and the parts of `refs` and `types` property/v1 reads, over `Flock.Program`'s decoded
descriptor: a reference sequence's parts (a concatenation flat, an alias spliced in), their slices (`Refs.slice`), the
leaf intervals they name (`ref_intervals`, `strided_intervals` with its broadcast and transpose cases) and a type's
leaf widths over a run of leaves (`stacking._width`). An interval set is sorted, disjoint and non-empty (`norm`).
-/

namespace RemoteAudit.Property

open Flock.Program

abbrev Ivs := Array (Nat × Nat)

/-- `norm`: empty pairs dropped, overlapping and adjacent pairs merged. -/
def norm (xs : Array (Nat × Nat)) : Ivs := Id.run do
  let ys := (xs.filter fun (a, b) => b > a).qsort fun x y => x.1 < y.1 || (x.1 == y.1 && x.2 < y.2)
  let mut out : Ivs := #[]
  for (lo, hi) in ys do
    match out.back? with
    | some (a, b) =>
      if lo ≤ b then
        if hi > b then out := out.pop.push (a, hi)
      else out := out.push (lo, hi)
    | none => out := out.push (lo, hi)
  return out

def total (iv : Ivs) : Nat := iv.foldl (fun n (a, b) => n + (b - a)) 0

def full (n : Nat) : Ivs := if n > 0 then #[(0, n)] else #[]

def isect (a b : Ivs) : Ivs := Id.run do
  let mut out : Ivs := #[]
  let mut i := 0
  let mut j := 0
  for _ in [0:a.size + b.size] do
    if i ≥ a.size || j ≥ b.size then break
    let lo := max a[i]!.1 b[j]!.1
    let hi := min a[i]!.2 b[j]!.2
    if lo < hi then out := out.push (lo, hi)
    if a[i]!.2 < b[j]!.2 then i := i + 1 else j := j + 1
  return out

/-- `a` without the leaves of `b`. -/
def minus (a b : Ivs) : Ivs := Id.run do
  let mut out : Ivs := #[]
  for (lo, hi) in a do
    let mut cur := lo
    for (x, y) in b do
      if y ≤ cur || x ≥ hi then continue
      if x > cur then out := out.push (cur, x)
      cur := max cur y
    if cur < hi then out := out.push (cur, hi)
  return out

/-! ### Parts -/

def STRIDED_ENUM_LIMIT : Nat := 2 ^ 20

def partSpace? : Part → Option (Bool × Nat)
  | .affine p i _ c => if c > 0 then some (p, i) else none
  | .strided p i _ _ _ _ c => if c > 0 then some (p, i) else none
  | .explicit _ => none

private def stridedLeaf (base : Int) (inner : Nat) (os is_ : Int) (k : Nat) : Int :=
  base + ((k / inner : Nat) : Int) * os + ((k % inner : Nat) : Int) * is_

/-- `Refs.slice(lo, hi)` of one part, as `refs` cuts it: an affine run, a strided run cut at whole inner runs, one inner
run (affine, or a strided run of inner 1), else the leaves one by one. -/
def partSlice (q : Part) (lo hi : Nat) : Part :=
  match q with
  | .affine p i b c =>
    let hi := min hi c
    .affine p i (b + lo) (hi - lo)
  | .explicit it => .explicit (it.extract lo hi)
  | .strided p i b inner os is_ c =>
    let hi := min hi c
    if hi ≤ lo then .explicit #[]
    else if lo % inner == 0 && (hi % inner == 0 || hi == c) then
      .strided p i (b + ((lo / inner : Nat) : Int) * os) inner os is_ (hi - lo)
    else if hi - lo ≤ inner - lo % inner then
      let first := (stridedLeaf b inner os is_ lo).toNat
      if is_ == 1 then .affine p i first (hi - lo) else .strided p i first 1 is_ 0 (hi - lo)
    else .explicit ((Array.range (hi - lo)).map fun k => q.get (lo + k))

/-- `strided_intervals`: intervals whose union is the leaves the strided run names, through its broadcast, affine and
transpose cases (a broadcast's whole inner run even when the run is shorter, as `intervals` has it), else leaf by
leaf up to `STRIDED_ENUM_LIMIT`. -/
def stridedIvs (base : Int) (inner : Nat) (os is_ : Int) (n : Nat) : Except String (Array (Int × Int)) := do
  if n == 0 then return #[]
  let outer := (n + inner - 1) / inner
  let last := n - (outer - 1) * inner
  let innerI : Int := inner
  if os == 0 && is_ == 0 then return #[(base, base + 1)]
  if is_ == 1 then
    if os == innerI && last == inner then return #[(base, base + n)]
    if os == 0 then return #[(base, base + inner)]
    return (Array.range outer).map fun (q : Nat) =>
      let s := base + (q : Int) * os
      (s, s + (if q < outer - 1 then inner else last))
  if os == 1 && is_ ≥ (outer : Int) then
    if is_ == (outer : Int) && last == inner then return #[(base, base + n)]
    return (Array.range inner).filterMap fun (rr : Nat) =>
      let rows := if rr < last then outer else outer - 1
      if rows > 0 then some (base + (rr : Int) * is_, base + (rr : Int) * is_ + rows) else none
  if is_ == 0 then return (Array.range outer).map fun (q : Nat) => (base + (q : Int) * os, base + (q : Int) * os + 1)
  if n ≤ STRIDED_ENUM_LIMIT then
    return (Array.range n).map fun (j : Nat) => let l := stridedLeaf base inner os is_ j; (l, l + 1)
  throw s!"a strided reference of {n} leaves has no interval structure"

private def addIv (m : Array ((Bool × Nat) × Array (Nat × Nat))) (key : Bool × Nat) (iv : Nat × Nat) :
    Array ((Bool × Nat) × Array (Nat × Nat)) :=
  match m.findIdx? (·.1 == key) with
  | some k => m.modify k fun (c, xs) => (c, xs.push iv)
  | none => m.push (key, #[iv])

/-- `ref_intervals` of parts: the leaf intervals they name, by collection in first-named order, each normalized. -/
def partsIvs (ps : Array Part) : Except String (Array ((Bool × Nat) × Ivs)) := do
  let mut m : Array ((Bool × Nat) × Array (Nat × Nat)) := #[]
  for q in ps do
    match q with
    | .affine p i b c => if c > 0 then m := addIv m (p, i) (b, b + c)
    | .explicit it => for r in it do m := addIv m (r.p, r.i) (r.j, r.j + 1)
    | .strided p i b inner os is_ c =>
      for (lo, hi) in ← stridedIvs b inner os is_ c do
        m := addIv m (p, i) (lo.toNat, hi.toNat)
  return m.map fun (k, xs) => (k, norm xs)

/-- The collections parts read (`_colls`). -/
def partsColls (ps : Array Part) : Array (Bool × Nat) := Id.run do
  let mut out : Array (Bool × Nat) := #[]
  for q in ps do
    match q with
    | .explicit it => for r in it do if !out.contains (r.p, r.i) then out := out.push (r.p, r.i)
    | q => if let some k := partSpace? q then if !out.contains k then out := out.push k
  return out

/-- `Concat.slice(lo, hi)` of parts: each part's slice over the positions it holds. -/
def sliceParts (ps : Array Part) (lo hi : Nat) : Array Part := Id.run do
  let mut out : Array Part := #[]
  let mut st := 0
  for q in ps do
    let c := q.count
    let a := max lo st
    let b := min hi (st + c)
    if b > a then out := out.push (partSlice q (a - st) (b - st))
    st := st + c
  return out

def partsCount (ps : Array Part) : Nat := ps.foldl (· + ·.count) 0

def partsGet (ps : Array Part) (k : Nat) : Option Ref := Id.run do
  let mut st := 0
  for q in ps do
    if k < st + q.count then return some (q.get (k - st))
    st := st + q.count
  return none

/-! ### Widths -/

private theorem sizeOf_snd_lt_of_mem {fs : Array (String × Ty)} {f : String × Ty} (h : f ∈ fs) :
    sizeOf f.2 < 1 + sizeOf fs := by
  have := Array.sizeOf_lt_of_mem h
  obtain ⟨k, t⟩ := f
  simp only [Prod.mk.sizeOf_spec] at this ⊢
  omega

def widthSet : Ty → List Nat
  | .value w => [w]
  | .array n e => if n == 0 then [] else widthSet e
  | .tuple fs => fs.foldl (fun acc f => acc ++ (widthSet f).filter (!acc.contains ·)) []
  | .record fs => fs.foldl (fun acc f => acc ++ (widthSet f.2).filter (!acc.contains ·)) []
decreasing_by all_goals first | decreasing_tactic | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

def homogeneous (t : Ty) : Option Nat := match (widthSet t).eraseDups with
  | [w] => some w
  | _ => none

/-- `_bits`: the widths of `t`'s leaves, summed. -/
def bitsAll : Ty → Nat
  | .value w => w
  | .array n e => n * bitsAll e
  | .tuple fs => fs.foldl (· + bitsAll ·) 0
  | .record fs => fs.foldl (· + bitsAll ·.2) 0
decreasing_by all_goals first | decreasing_tactic | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

private def one (xs : List (Option Nat)) : Option Nat := match xs.eraseDups with
  | [w] => w
  | _ => none

/-- The widths of the fields (each its leaf count and its `widthOf`) that leaves `[lo, hi)` meet, each over its own. -/
private def fieldWidths (fs : Array (Nat × (Nat → Nat → Option Nat))) (lo hi : Nat) : Option Nat := Id.run do
  let mut ws : List (Option Nat) := []
  let mut pos := 0
  for (n, w) in fs do
    if pos < hi && pos + n > lo then ws := ws ++ [w (max lo pos - pos) (min hi (pos + n) - pos)]
    pos := pos + n
  return one ws

/-- `stacking._width`: the one width of `t`'s leaves `[lo, hi)`, or none when they have two (structurally). -/
def widthOf (t : Ty) (lo hi : Nat) : Option Nat :=
  match homogeneous t with
  | some w => some w
  | none =>
    match t with
    | .array _ e =>
      let L := e.leaves
      if L == 0 then none else
      let a := lo / L
      let b := (hi - 1) / L
      if a == b then widthOf e (lo - a * L) (hi - a * L)
      else one ([widthOf e (lo - a * L) L, widthOf e 0 (hi - b * L)] ++ (if b - a > 1 then [none] else []))
    | .tuple fs => fieldWidths (fs.attach.map fun ⟨f, _⟩ => (f.leaves, widthOf f)) lo hi
    | .record fs => fieldWidths (fs.attach.map fun ⟨f, _⟩ => (f.2.leaves, widthOf f.2)) lo hi
    | .value w => some w
termination_by t
decreasing_by
  all_goals first
    | decreasing_tactic
    | (simp only [Ty.tuple.sizeOf_spec]; have := Array.sizeOf_lt_of_mem ‹_›; omega)
    | (simp only [Ty.record.sizeOf_spec]; exact sizeOf_snd_lt_of_mem ‹_›)

end RemoteAudit.Property
