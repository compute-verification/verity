/-
A templated unit, `flock-ir-tmpl/v2`: a `flock-ir-unit/v2` unit's rows given once per Definition, with placed
copies, so its text is O(description) where the flat form is O(unrolled). No verifier path reads it: `HmRow.parse` takes a
flat `NET` block only, and the guarantees read the flat circuit `Desc.unroll` stands for. `unroll` is the reference, the
flat net it stands for, built by `Net.ofRows` with every check `Net.parse` runs; `Flock.Test.Tmpl` holds it to `Net.parse`
on the flat text, and `benchmarks/verifier_cost/flock_tmpl` to the H1 tile's flat unit byte for byte.

The text, one item a line:

* `flock-ir-tmpl/v2 NAME ROOT <in groups> <out groups>`: the unit's name, its root Definition, and its port groups as
  `flock-ir-unit/v2`'s header writes them;
* per Definition `d = 0, 1, …`: `DEF COLS IN ROWS CALLS`, then `ROWS` rows `col na a… nb b…`, then `CALLS` lines
  `CALL e count base stride segs (len at step reps rstride)…`;
* optionally, last, the flat unit's `LEAVES <json>` line, carried as it is (`Net.parse` reads it as bound by the pin).

Definition `d` has `COLS` columns of its own and `IN` inputs. A reference `v` in its rows and calls is its own column `v`
for `v < COLS`, its input `v − COLS` for `COLS ≤ v < COLS + IN`, and the unit's constant for `v = COLS + IN`. A row
`col na a… nb b…` gives its own column `col`'s A and B. A call places `count` copies of Definition `e` (`e < d`) on its
own columns `base + j · stride …`, `stride` at least `e`'s `COLS`, inside `d`'s. A segment binds `reps · len` inputs:
copy `j`'s `(r · len + t')`-th input of the segment is the reference `at + r · rstride + j · step + t'` of `d` (`r < reps`,
`t' < len`; segments in order, their `reps · len` summing to `e`'s `IN`; `step = 0` binds every copy to the same
references, `rstride = 0` repeats them).

Unrolled, a copy at global offset `o` with its inputs bound to global columns writes each of its rows at `o + col`, its
own columns `x ↦ o + x`, its inputs to their columns and the constant to the unit's: substituted, so a copy adds no rows of
its own. Each row is then canonical as `flock-ir-unit/v2` writes it, a sum over GF(2): its columns ascending, a column
named an even number of times dropped. The root has no inputs; the unit's columns are its, and the constant its last,
`[const]·[const]`. Every column is written at most once, and one never written is empty.
-/
import Flock.Net

namespace Flock.Tmpl

/-- A segment of a call's input binding: `reps` runs of `len` inputs, run `r` bound for copy `j` to caller references
`at + r · rstride + j · step + t'`. -/
structure Seg where
  len : Nat
  at_ : Nat
  step : Nat
  reps : Nat
  rstride : Nat
  deriving Inhabited, Repr, BEq

/-- A call: `count` copies of Definition `def_` at own columns `base + j · stride`, inputs bound by `segs`. -/
structure Call where
  def_ : Nat
  count : Nat
  base : Nat
  stride : Nat
  segs : Array Seg
  deriving Inhabited, Repr, BEq

/-- A Definition: its own columns, its inputs, its own rows `(col, A, B)` in references, and its calls. -/
structure Def where
  cols : Nat
  ins : Nat
  rows : Array (Nat × Array Nat × Array Nat)
  calls : Array Call
  deriving Inhabited, Repr, BEq

structure Tmpl where
  name : String
  root : Nat
  inGroups : Array Group
  outGroups : Array Group
  defs : Array Def
  leaves : Option String := none
  deriving Inhabited

def FORMAT : String := "flock-ir-tmpl/v2"

/-! ## Row order, checked on the description

`Net.checkOrder` walks the unrolled rows; `ordered` checks the same order per Definition, in O(description), and
`Flock.TmplOrder.rows_ordered` proves it gives the order `checkOrder` checks. A Definition's row reads only its inputs, the
constant and its own earlier columns. A segment binds, for every copy, only the caller's inputs and constant, or only its
own columns, all before that copy's first column: a segment that straddles the two is refused. A reference is linear in
the copy index and grows with the run index, so the last run of copy 0 and of the last copy bound every reference. In the root, a row or copy in the input rows is not checked
(`checkOrder` skips those rows), and an input column read elsewhere is a port bit, so not padding. -/

/-- The unit's input rows end here. -/
def Tmpl.inCols (t : Tmpl) : Nat := ((t.inGroups.back?.map fun g => g.col + g.words).getD 0) * WORD

/-- Columns `[lo, hi)` are port bits of one input group. -/
def Tmpl.portSpan (t : Tmpl) (lo hi : Nat) : Bool :=
  t.inGroups.toList.any fun g => g.col * WORD ≤ lo && hi ≤ g.col * WORD + g.bits.foldl (· + ·) 0

/-- Past every reference segment `s` binds over `count` copies. -/
def Seg.hi (s : Seg) (count : Nat) : Nat := s.at_ + (s.reps - 1) * s.rstride + (count - 1) * s.step + s.len

/-- Segment `s` of a call (`count` copies at `base + j · stride`) in a Definition of `cols` columns binds, for every copy,
only references past its own columns, or only its own columns, each before that copy's first. -/
def Seg.ordered (cols base stride count : Nat) (s : Seg) : Bool :=
  s.len == 0 || cols ≤ s.at_ ||
    (s.hi count ≤ cols && s.at_ + (s.reps - 1) * s.rstride + s.len ≤ base && s.hi count ≤ base + (count - 1) * stride)

/-- Definition `df`'s rows and copies sit inside its columns. -/
def Def.fits (defs : Array Def) (df : Def) : Bool :=
  df.rows.toList.all (fun r => r.1 < df.cols) &&
    df.calls.toList.all fun c => c.base + (c.count - 1) * c.stride + (defs.getD c.def_ default).cols ≤ df.cols

/-- Definition `df` placed anywhere past the input rows writes rows that read earlier columns (`F`: the Definitions that
are). -/
def Def.ordered (F : Array Bool) (df : Def) : Bool :=
  df.rows.toList.all (fun r => (r.2.1.toList ++ r.2.2.toList).all fun v => df.cols ≤ v || v < r.1) &&
    df.calls.toList.all fun c => F.getD c.def_ false && c.segs.toList.all (Seg.ordered df.cols c.base c.stride c.count)

/-- Which Definitions are ordered, each from the earlier ones. -/
def flags (defs : Array Def) : Array Bool := defs.foldl (fun F df => F.push (df.ordered F)) #[]

/-- The root's rows past the input rows read earlier columns, port bits if input columns; each call is in the input rows,
or past them, of an ordered Definition, bound to the constant (its last column) or ordered segments. -/
def Tmpl.rootOrdered (t : Tmpl) (F : Array Bool) : Bool :=
  let df := t.defs.getD t.root default
  df.rows.toList.all (fun r => decide (r.1 < t.inCols) || (r.2.1.toList ++ r.2.2.toList).all fun v =>
      df.cols ≤ v + 1 || (v < r.1 && (t.inCols ≤ v || t.portSpan v (v + 1)))) &&
    df.calls.toList.all fun c =>
      decide (c.base + (c.count - 1) * c.stride + (t.defs.getD c.def_ default).cols ≤ t.inCols) ||
        (t.inCols ≤ c.base && F.getD c.def_ false && c.segs.toList.all fun s =>
          (s.len == 1 && s.at_ + 1 == df.cols && s.hi c.count ≤ df.cols) ||
            (Seg.ordered df.cols c.base c.stride c.count s && (t.inCols ≤ s.at_ || t.portSpan s.at_ (s.hi c.count))))

/-- **The unrolled rows are in order**, checked on the description (`Flock.TmplOrder.rows_ordered`). -/
def Tmpl.ordered (t : Tmpl) : Bool :=
  let F := flags t.defs
  (List.range t.defs.size).all (fun d =>
      (t.defs.getD d default).fits t.defs && (!F.getD d false || (t.defs.getD d default).ordered F)) &&
    t.rootOrdered F

/-- Definition `d`'s rows at its columns, its calls to earlier Definitions with their copies side by side inside its
columns and its segments binding their callee's inputs, its columns and calls fewer than `2^64` (`locate` halves
64 times). -/
def Def.wellFormed (defs : Array Def) (d : Nat) (df : Def) : Bool :=
  df.cols < 2 ^ 64 && df.calls.size < 2 ^ 64 && df.rows.toList.all (fun r => r.1 < df.cols) &&
    df.calls.toList.all fun c => c.def_ < d && c.base + c.count * (defs.getD c.def_ default).cols ≤ df.cols &&
      c.segs.foldl (fun n s => n + s.reps * s.len) 0 == (defs.getD c.def_ default).ins

/-- **What the template fold's proof reads of a template** (`FlockSoundness.TmplFold.fold_eq_foldRef_wf`), checked on
the description: every Definition well formed, and the root a Definition with no inputs and a constant. -/
def Tmpl.wellFormed (t : Tmpl) : Bool :=
  t.root < t.defs.size && (t.defs.getD t.root default).ins == 0 && 0 < (t.defs.getD t.root default).cols &&
    (List.range t.defs.size).all fun d => (t.defs.getD d default).wellFormed t.defs d

private def num (h : Array String) (i : Nat) (what : String) : V Nat :=
  match h[i]?.bind String.toNat? with
  | some n => pure n
  | none => throw s!"{FORMAT}: {what}: field {i} is not a number"

private def groups (h : Array String) (at0 : Nat) : V (Array Group × Nat) := do
  let count ← num h at0 "header"
  let mut at_ := at0 + 1
  let mut out := #[]
  for _ in [0:count] do
    let col ← num h at_ "header"
    let words ← num h (at_ + 1) "header"
    let np ← num h (at_ + 2) "header"
    let mut bits := #[]
    for i in [0:np] do bits := bits.push (← num h (at_ + 3 + i) "header")
    out := out.push ⟨col, words, bits⟩
    at_ := at_ + 3 + np
  return (out, at_)

/-- Parse a templated unit's lines. Checks the form only: each call's callee is an earlier Definition, its copies fit
side by side inside the caller's columns, and its segments cover its callee's inputs with the caller's references. -/
def parse (lines : Array String) : V Tmpl := do
  let h := tokens (lines.getD 0 "")
  if h[0]? != some FORMAT then throw s!"not a {FORMAT} unit"
  let name := h.getD 1 ""
  let root ← num h 2 "header"
  let (inGroups, at1) ← groups h 3
  let (outGroups, at2) ← groups h at1
  if at2 != h.size then throw s!"{FORMAT}: trailing header fields"
  let mut defs : Array Def := #[]
  let mut leaves := none
  let mut at_ := 1
  while at_ < lines.size do
    if at_ + 1 == lines.size && lines[at_]!.startsWith "LEAVES " then
      leaves := some lines[at_]!
      break
    let dh := tokens lines[at_]!
    if dh.size != 5 || dh[0]! != "DEF" then throw s!"{FORMAT}: line {at_} is not a DEF line"
    let cols ← num dh 1 "DEF"
    let ins ← num dh 2 "DEF"
    let nrows ← num dh 3 "DEF"
    let ncalls ← num dh 4 "DEF"
    let refs := cols + ins + 1
    let mut rows := #[]
    for i in [0:nrows] do
      let v ← (tokens (lines.getD (at_ + 1 + i) "")).mapM fun t => match t.toNat? with
        | some n => pure n
        | none => throw s!"{FORMAT}: definition {defs.size}: row {i} is not numbers"
      let na := v.getD 1 0
      let nb := v.getD (2 + na) 0
      if v.size < 2 || v.size != 3 + na + nb then throw s!"{FORMAT}: definition {defs.size}: row {i} is malformed"
      let (col, a, b) := (v[0]!, v.extract 2 (2 + na), v.extract (3 + na) (3 + na + nb))
      if col ≥ cols then throw s!"{FORMAT}: definition {defs.size}: row {i} is not at one of its columns"
      if (a ++ b).any (· ≥ refs) then throw s!"{FORMAT}: definition {defs.size}: row {i} reads past its references"
      rows := rows.push (col, a, b)
    let mut calls := #[]
    for i in [0:ncalls] do
      let ct := tokens (lines.getD (at_ + 1 + nrows + i) "")
      if ct[0]? != some "CALL" then throw s!"{FORMAT}: definition {defs.size}: call {i} is not a CALL line"
      let e ← num ct 1 "CALL"
      let count ← num ct 2 "CALL"
      let base ← num ct 3 "CALL"
      let stride ← num ct 4 "CALL"
      let nseg ← num ct 5 "CALL"
      if ct.size != 6 + 5 * nseg then throw s!"{FORMAT}: definition {defs.size}: call {i} is malformed"
      let mut segs := #[]
      for s in [0:nseg] do
        segs := segs.push ⟨← num ct (6 + 5 * s) "CALL", ← num ct (7 + 5 * s) "CALL", ← num ct (8 + 5 * s) "CALL",
          ← num ct (9 + 5 * s) "CALL", ← num ct (10 + 5 * s) "CALL"⟩
      let some callee := defs[e]? | throw s!"{FORMAT}: definition {defs.size}: call {i} names no earlier definition"
      if count == 0 || (count > 1 && stride < callee.cols) || base + (count - 1) * stride + callee.cols > cols then
        throw s!"{FORMAT}: definition {defs.size}: call {i}'s copies overlap, or overrun its columns"
      if segs.foldl (fun n s => n + s.reps * s.len) 0 != callee.ins then
        throw s!"{FORMAT}: definition {defs.size}: call {i}'s segments do not bind its callee's inputs"
      if segs.any (fun s => s.len > 0 && s.reps > 0 && s.hi count > refs) then
        throw s!"{FORMAT}: definition {defs.size}: call {i} binds past its references"
      calls := calls.push { def_ := e, count, base, stride, segs }
    defs := defs.push { cols, ins, rows, calls }
    at_ := at_ + 1 + nrows + ncalls
  if root ≥ defs.size then throw s!"{FORMAT}: the root names no definition"
  if (defs.getD root default).ins != 0 || (defs.getD root default).cols == 0 then
    throw s!"{FORMAT}: the root has inputs, or no constant"
  let t : Tmpl := { name, root, inGroups, outGroups, defs, leaves }
  if !t.ordered then
    throw s!"{FORMAT}: a row or a binding reads a later column, input padding, or both its caller's own columns and its inputs"
  if !t.wellFormed then throw s!"{FORMAT}: a definition has 2^64 columns or calls, or a call the fold's proof does not cover"
  return t

/-- A sorted list with each run of equal columns cut to its length mod 2. -/
def dropPairs : List Nat → List Nat
  | [] => []
  | [a] => [a]
  | a :: b :: rest => if a == b then dropPairs rest else a :: dropPairs (b :: rest)

/-- A row as `flock-ir-unit/v2` writes it: a sum over GF(2), its columns ascending, each named an odd number of times. -/
def canon (xs : Array Nat) : Array Nat := (dropPairs xs.toList.mergeSort).toArray

/-- The unrolled rows so far: each column's A and B, and whether it was written. -/
structure Acc where
  ra : Array (Array Nat)
  rb : Array (Array Nat)
  set : Array Bool

/-- No column written yet. -/
def Acc.empty (cols : Nat) : Acc :=
  { ra := Array.replicate cols #[], rb := Array.replicate cols #[], set := Array.replicate cols false }

def write (acc : Acc) (c : Nat) (a b : Array Nat) : V Acc :=
  if acc.set.getD c true then throw s!"{FORMAT}: column {c} is written twice, or past the unit"
  else pure { ra := acc.ra.set! c a, rb := acc.rb.set! c b, set := acc.set.set! c true }

/-- Definition `df`'s reference `x` in its copy at global offset `o`, its inputs bound to `inp`, the unit's constant `k`. -/
def Def.ref (df : Def) (o : Nat) (inp : List Nat) (k x : Nat) : Nat :=
  if x < df.cols then o + x else if x < df.cols + df.ins then inp.getD (x - df.cols) k else k

/-- Copy `j`'s inputs: each segment's references `at + r · rstride + j · step + t`, run by run, through the caller's
references `g`. -/
def Call.bound (c : Call) (g : Nat → Nat) (j : Nat) : List Nat :=
  c.segs.toList.flatMap fun s => (List.range s.reps).flatMap fun r =>
    (List.range s.len).map fun t => g (s.at_ + r * s.rstride + j * s.step + t)

/-- Write Definition `d`'s copy at global offset `o`, its inputs bound to the global columns `inp`, the unit's constant
`k`. `fuel` bounds the call depth; a callee is an earlier definition, so `defs.size` is enough. -/
def place (defs : Array Def) (k : Nat) : Nat → Nat → Nat → List Nat → Acc → V Acc
  | 0, _, _, _, _ => throw s!"{FORMAT}: calls nest deeper than the definitions"
  | fuel + 1, d, o, inp, acc => do
    let some df := defs[d]? | throw s!"{FORMAT}: no definition {d}"
    let acc ← df.rows.toList.foldlM (fun acc r =>
      write acc (o + r.1) (canon (r.2.1.map (df.ref o inp k))) (canon (r.2.2.map (df.ref o inp k)))) acc
    df.calls.toList.foldlM (fun acc c => (List.range c.count).foldlM (fun acc j =>
      place defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound (df.ref o inp k) j) acc) acc) acc

/-! ## In linear time

`place` reads a copy's inputs from a `List`, so each reference to an input walks it: the H1 tile's unit at K = 448 has
Definitions of a million inputs. `placeFast` reads them from an `Array` made once per copy (`Def.refA`), folds the rows
and calls as arrays, and writes the accumulator's arrays in place (`writeFast`). `@[csimp]` compiles it in `place`'s
place, so the verifier runs it while every theorem reads `place`. -/

/-- `Def.ref` with the inputs in an `Array`. -/
def Def.refA (df : Def) (o : Nat) (inp : Array Nat) (k x : Nat) : Nat :=
  if x < df.cols then o + x else if x < df.cols + df.ins then inp.getD (x - df.cols) k else k

theorem Def.refA_toArray (df : Def) (o : Nat) (inp : List Nat) (k : Nat) : df.refA o inp.toArray k = df.ref o inp k := by
  funext x
  simp only [Def.refA, Def.ref, Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getD_eq_getElem?_getD]

/-- `write`, its accumulator taken apart first so that each array is updated in place. -/
def writeFast : Acc → Nat → Array Nat → Array Nat → V Acc
  | ⟨ra, rb, set⟩, c, a, b =>
    if set.getD c true then throw s!"{FORMAT}: column {c} is written twice, or past the unit"
    else pure ⟨ra.set! c a, rb.set! c b, set.set! c true⟩

theorem writeFast_eq : @writeFast = @write := by
  funext acc c a b
  cases acc
  rfl

/-- `place`, through `Def.refA` and `writeFast`. -/
def placeFast (defs : Array Def) (k : Nat) : Nat → Nat → Nat → List Nat → Acc → V Acc
  | 0, _, _, _, _ => throw s!"{FORMAT}: calls nest deeper than the definitions"
  | fuel + 1, d, o, inp, acc => do
    let some df := defs[d]? | throw s!"{FORMAT}: no definition {d}"
    let ia := inp.toArray
    let g := df.refA o ia k
    let acc ← df.rows.foldlM (fun acc r => writeFast acc (o + r.1) (canon (r.2.1.map g)) (canon (r.2.2.map g))) acc
    df.calls.foldlM (fun acc c => (List.range c.count).foldlM (fun acc j =>
      placeFast defs k fuel c.def_ (o + c.base + j * c.stride) (c.bound g j) acc) acc) acc

theorem placeFast_eq (defs : Array Def) (k : Nat) :
    ∀ fuel d o inp acc, placeFast defs k fuel d o inp acc = place defs k fuel d o inp acc
  | 0, _, _, _, _ => rfl
  | fuel + 1, d, o, inp, acc => by
    simp only [placeFast, place, Def.refA_toArray, writeFast_eq, ← Array.foldlM_toList, placeFast_eq defs k fuel]

@[csimp] theorem place_eq_placeFast : @place = @placeFast := by
  funext defs k fuel d o inp acc
  exact (placeFast_eq defs k fuel d o inp acc).symm

/-- The unit's last column, its constant. -/
def Tmpl.constPos (t : Tmpl) : Nat := (t.defs.getD t.root default).cols - 1

/-- **The unrolled rows**: the root's copy at offset 0, then the constant's row `[const]·[const]`. -/
def Tmpl.rows (t : Tmpl) : V Acc := do
  let acc ← place t.defs t.constPos (t.defs.size + 1) t.root 0 [] (Acc.empty (t.defs.getD t.root default).cols)
  write acc t.constPos #[t.constPos] #[t.constPos]

/-- **The flat net a templated unit stands for**: its rows (`rows`), then `Net.ofRows` with the checks `Net.parse` runs on
a flat unit. -/
def unroll (t : Tmpl) : V Net := do
  let acc ← t.rows
  Net.ofRows t.name (t.defs.getD t.root default).cols t.constPos t.inGroups t.outGroups acc.ra acc.rb

/-- The unrolled net's `flock-ir-unit/v2` header line. -/
def headerLine (n : Net) : String :=
  let grp := fun (gs : Array Group) => s!"{gs.size}" ++ String.join (gs.toList.map fun g =>
    s!" {g.col} {g.words} {g.bits.size}" ++ String.join (g.bits.toList.map (s!" {·}")))
  s!"flock-ir-unit/v2 {n.name} {n.useful} {n.constPos} {grp n.inGroups} {grp n.outGroups}"

/-- The unrolled net's row `i`, as `flock-ir-unit/v2` writes it. -/
def rowLine (n : Net) (i : Nat) : String :=
  let row := fun (m : Sparse) =>
    let cs := (List.range (m.len.getD i 0)).map (m.col i)
    s!"{cs.length}" ++ String.join (cs.map (s!" {·}"))
  s!"{row n.a} {row n.b}"

/-- The unrolled net's `flock-ir-unit/v2` lines, as `Net.parse` reads them, with the carried `LEAVES` line. -/
def flatLines (t : Tmpl) : V (Array String) := do
  let n ← unroll t
  let mut out := #[headerLine n]
  for i in [0:n.useful] do out := out.push (rowLine n i)
  return match t.leaves with
    | some l => out.push l
    | none => out

end Flock.Tmpl
