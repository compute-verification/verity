/-
Layout (iv) of a templated unit: the committed position of each of its columns, given per Definition, so the
lincheck fold can read a templated unit from its description. No verifier path reads it yet.

Each Definition is laid out once, from its callees' layouts. Its own columns keep their order, in runs; a call's copies
sit together. A run of `count ≥ 2` copies takes `M = nextPow2 count` lanes, copy `j`'s column at callee position `p` at
`nb + p · M + j`, so the copy index is the position's low bits. `nb` is a multiple of `M` times the callee's alignment
(lenient), of `M` times its size rounded up to a power of two (strict), or (hybrid `T`) of `M` times that power of two over
`2^T` for a callee reached on more than one call path (its `palign`). A single copy sits at `nb + p`, `nb` aligned the
same way without `M`. In the root, the input rows keep their columns (a call there is to a Definition laid out as it
is), and each output group starts at a multiple of its width and keeps its columns in order, as `flock-ir-unit/v2` has
them; the constant, the root's last column, is the layout's last.

A placed copy's positions are affine in its Definition's: a column at local position `p` of a copy placed under `A + B·q`
is at `A + B·p`, with `B` a product of lane counts, a power of two. `posMap` is the reference, every column's position
(O(unrolled)); `posOf` one column's, through the call tree.
-/
import Flock.Tmpl

namespace Flock.Tmpl

/-- Where one call's copies sit in its caller's layout: copy `j`'s column at callee position `p` is at
`nb + j · cs + p · ms`. A run of lanes has `cs = 1` and `ms = M`; a single copy, or copies in the root's input rows or an
output group, `cs` the callee's columns and `ms = 1`. -/
structure Place where
  nb : Nat
  cs : Nat
  ms : Nat
  deriving Inhabited, Repr, BEq

/-- A Definition's layout: its size and alignment, its own columns' runs `(x0, x1, p0)` (own column `x ∈ [x0, x1)` at
`p0 + x − x0`), and each call's place, in call order. `ident`: every column at its own index. `palign`: the alignment its
copies are placed at (times their lanes), a power of two at least `align`. -/
structure Lay where
  size : Nat
  align : Nat
  runs : Array (Nat × Nat × Nat)
  places : Array Place
  ident : Bool
  palign : Nat := align
  deriving Inhabited, Repr

/-- Where a Definition's copies are placed: at its own alignment (`lenient`), at its size rounded up to a power of two
(`strict`), or (`hybrid T`) a Definition the call tree reaches on more than one path (a run counted once) at that power of
two over `2^T` (never below its own alignment), the rest lenient. -/
inductive Mode where
  | lenient
  | strict
  | hybrid (T : Nat)
  deriving Inhabited, Repr, BEq

/-- Per Definition, the call paths from the root that reach it, each run of copies counted once. -/
def Tmpl.paths (t : Tmpl) : Array Nat := Id.run do
  let n := t.defs.size
  let mut p := (Array.replicate n 0).set! t.root 1
  for i in [0:n] do
    let d := n - 1 - i
    for c in (t.defs.getD d default).calls do p := p.modify c.def_ (· + p.getD d 0)
  return p

def up (x a : Nat) : Nat := (x + a - 1) / a * a

/-- Lay out Definition `d` of `t` from its callees' layouts `lays`, under `mode` (`paths`: `Tmpl.paths`). Refuses copies
of a run that are not side by side, calls out of order of their columns, and a call in the root's input rows or an output
group to a Definition not laid out as it is. -/
def layDefM (t : Tmpl) (lays : Array Lay) (mode : Mode) (paths : Array Nat) (d : Nat) : V Lay := do
  let D := t.defs.getD d default
  let root := d == t.root
  let outs := if root then (t.outGroups.toList.map fun g => (g.col * WORD, g.words * WORD)).mergeSort (·.1 ≤ ·.1)
    else []
  let mut pos := 0
  let mut x := 0
  let mut al := 1
  let mut runs : Array (Nat × Nat × Nat) := #[]
  let mut places : Array Place := #[]
  let mut done : List Nat := []
  let mut ident := true
  for c in D.calls do
    let Le := (t.defs.getD c.def_ default).cols
    let E := lays.getD c.def_ default
    if c.base < x then throw s!"layout: definition {d}: its calls are not in order of their columns"
    if c.count > 1 && c.stride != Le then throw s!"layout: definition {d}: a call's copies are not side by side"
    -- own columns `[x, c.base)`, each output group's start aligned to its width
    let mut x0 := x
    for (p, w) in outs do
      if x0 ≤ p && p ≤ c.base && !done.contains p then
        if x0 < p then runs := runs.push (x0, p, pos)
        pos := up (pos + (p - x0)) w
        x0 := p
        done := p :: done
    if x0 < c.base then runs := runs.push (x0, c.base, pos)
    pos := pos + (c.base - x0)
    let inRows := root && c.base < t.inCols
    if inRows || (root && outs.any fun (p, w) => p ≤ c.base && c.base < p + w) then
      if E.size != Le || (inRows && !E.ident) then
        throw s!"layout: definition {d}: a call in the input rows or an output group to a definition laid out apart"
      places := places.push ⟨pos, Le, 1⟩
      pos := pos + c.count * Le
    else if c.count == 1 then
      let a := E.palign
      let nb := up pos a
      al := max al a
      places := places.push ⟨nb, Le, 1⟩
      pos := nb + E.size
    else
      let M := nextPow2 c.count
      let a := M * E.palign
      let nb := up pos a
      al := max al a
      places := places.push ⟨nb, 1, M⟩
      pos := nb + E.size * M
    ident := ident && E.ident && places.back!.nb == c.base && (c.count == 1 || places.back!.ms == 1)
    x := c.base + c.count * Le
  let mut x0 := x
  for (p, w) in outs do
    if x0 ≤ p && p ≤ D.cols && !done.contains p then
      if x0 < p then runs := runs.push (x0, p, pos)
      pos := up (pos + (p - x0)) w
      x0 := p
      done := p :: done
  if x0 < D.cols then runs := runs.push (x0, D.cols, pos)
  pos := pos + (D.cols - x0)
  ident := ident && pos == D.cols && runs.all fun (a, _, p) => a == p
  let palign := match mode with
    | .lenient => al
    | .strict => max al (nextPow2 pos)
    | .hybrid T => if 1 < paths.getD d 0 then max al (nextPow2 pos / 2 ^ T) else al
  return { size := pos, align := al, runs, places, ident, palign }

/-- Every Definition's layout under `mode`, callees first. -/
def layoutsM (t : Tmpl) (mode : Mode) : V (Array Lay) :=
  let paths := t.paths
  (List.range t.defs.size).foldlM (fun lays d => return lays.push (← layDefM t lays mode paths d)) #[]

def layDef (t : Tmpl) (lays : Array Lay) (strict : Bool) (d : Nat) : V Lay :=
  layDefM t lays (if strict then .strict else .lenient) t.paths d

/-- Every Definition's layout, lenient or `strict`, callees first. -/
def layouts (t : Tmpl) (strict : Bool) : V (Array Lay) := layoutsM t (if strict then .strict else .lenient)

/-- The first `i ∈ [lo, hi)` with `key i > x`, else `hi`, for `key` ascending (`fuel` halvings). -/
def firstGt (key : Nat → Nat) (x : Nat) : Nat → Nat → Nat → Nat
  | 0, lo, _ => lo
  | fuel + 1, lo, hi =>
    if hi ≤ lo then lo
    else if key ((lo + hi) / 2) ≤ x then firstGt key x fuel ((lo + hi) / 2 + 1) hi
    else firstGt key x fuel lo ((lo + hi) / 2)

/-- Where Definition `d`'s column `x` is: in copy `j` of call `i`, at its column `x'` there, or its own, at position
`p`. Calls ascend by base and runs by their first column, so each is found by halving. -/
inductive Loc where
  | call (i j x' : Nat)
  | own (p : Nat)
  | none

def locate (t : Tmpl) (lays : Array Lay) (d x : Nat) : Loc :=
  let D := t.defs.getD d default
  let L := lays.getD d default
  let i := firstGt (fun i => (D.calls.getD i default).base) x 64 0 D.calls.size
  let c := D.calls.getD (i - 1) default
  let Le := (t.defs.getD c.def_ default).cols
  if 0 < i && x < c.base + c.count * Le then .call (i - 1) ((x - c.base) / Le) ((x - c.base) % Le)
  else
    let r := firstGt (fun r => (L.runs.getD r default).1) x 64 0 L.runs.size
    let (a, b, p) := L.runs.getD (r - 1) default
    if 0 < r && x < b then .own (p + (x - a)) else .none

/-- Column `x` of Definition `d`'s copy, at its position in `d`'s layout (`fuel` bounds the call depth). -/
def posIn (t : Tmpl) (lays : Array Lay) : Nat → Nat → Nat → Nat
  | 0, _, x => x
  | fuel + 1, d, x =>
    match locate t lays d x with
    | .call i j x' =>
      let c := (t.defs.getD d default).calls.getD i default
      let pl := (lays.getD d default).places.getD i default
      pl.nb + j * pl.cs + pl.ms * posIn t lays fuel c.def_ x'
    | .own p => p
    | .none => x

/-- The root's column `x`, at its committed position. -/
def posOf (t : Tmpl) (lays : Array Lay) (x : Nat) : Nat := posIn t lays (t.defs.size + 1) t.root x

/-- Write Definition `d`'s copy at logical offset `o`, its local position `p` at `A + B · p` (`fuel` bounds the depth). -/
def fillMap (t : Tmpl) (lays : Array Lay) : Nat → Nat → Nat → Nat → Nat → Array Nat → Array Nat
  | 0, _, _, _, _, m => m
  | fuel + 1, d, o, A, B, m =>
    let D := t.defs.getD d default
    let L := lays.getD d default
    let m := L.runs.foldl (fun m (a, b, p) =>
      (List.range (b - a)).foldl (fun m i => m.set! (o + a + i) (A + B * (p + i))) m) m
    (List.range D.calls.size).foldl (fun m i =>
      let c := D.calls.getD i default
      let pl := L.places.getD i default
      (List.range c.count).foldl (fun m j =>
        fillMap t lays fuel c.def_ (o + c.base + j * c.stride) (A + B * (pl.nb + j * pl.cs)) (B * pl.ms) m) m) m

/-- **Every column's committed position**, the reference (O(unrolled)). -/
def posMap (t : Tmpl) (lays : Array Lay) : Array Nat :=
  fillMap t lays (t.defs.size + 1) t.root 0 0 1 (Array.replicate (t.defs.getD t.root default).cols 0)

/-- **Layout (iv)** under `mode`, with what the unit's format needs of it: the input rows at their columns, each output
group at a multiple of its width with its columns in order, and the constant last. -/
def layoutM (t : Tmpl) (mode : Mode) : V (Array Lay) := do
  let lays ← layoutsM t mode
  let L := lays.getD t.root default
  let cols := (t.defs.getD t.root default).cols
  let inFixed := L.runs.all (fun (a, _, p) => t.inCols ≤ a || a == p) &&
    (List.range L.places.size).all fun i =>
      let c := (t.defs.getD t.root default).calls.getD i default
      t.inCols ≤ c.base || ((L.places.getD i default).nb == c.base && (lays.getD c.def_ default).ident)
  if !inFixed then throw "layout: the input rows do not keep their columns"
  for g in t.outGroups do
    let p := posOf t lays (g.col * WORD)
    if p % (g.words * WORD) != 0 || (List.range (g.words * WORD)).any fun i => posOf t lays (g.col * WORD + i) != p + i then
      throw s!"layout: output group at word {g.col} does not keep its columns at a multiple of its width"
  if posOf t lays (cols - 1) + 1 != L.size then throw "layout: the constant is not the layout's last column"
  return lays

/-- **Layout (iv)**, lenient or `strict`. -/
def layout (t : Tmpl) (strict : Bool) : V (Array Lay) := layoutM t (if strict then .strict else .lenient)

end Flock.Tmpl
