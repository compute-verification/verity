/-
A templated unit, `flock-ir-tmpl/v1` (PRF-133): a `flock-ir-unit/v2` unit's rows given once per Definition, with placed
copies, so its text is O(description) where the flat form is O(unrolled). No verifier path reads it: `HmRow.parse` takes a
flat `NET` block only, and the guarantees read the flat circuit `Desc.unroll` stands for. `unroll` is the reference, the
flat net it stands for, built by `Net.ofRows` with every check `Net.parse` runs; `Flock.Test.Tmpl` holds it to `Net.parse`
on the flat text, and `benchmarks/verifier_cost/flock_tmpl` to the H1 tile's flat unit byte for byte.

The text, one item a line:

* `flock-ir-tmpl/v1 NAME ROOT <in groups> <out groups>`: the unit's name, its root Definition, and its port groups as
  `flock-ir-unit/v2`'s header writes them;
* per Definition `d = 0, 1, …`: `DEF COLS IN ROWS CALLS`, then `ROWS` rows `col na a… nb b…`, then `CALLS` lines
  `CALL e count base stride segs (len at step)…`;
* optionally, last, the flat unit's `LEAVES <json>` line, carried as it is (`Net.parse` reads it as bound by the pin).

Definition `d` has `COLS` columns of its own and `IN` inputs. A reference `v` in its rows and calls is its own column `v`
for `v < COLS`, its input `v − COLS` for `COLS ≤ v < COLS + IN`, and the unit's constant for `v = COLS + IN`. A row
`col na a… nb b…` gives its own column `col`'s A and B. A call places `count` copies of Definition `e` (`e < d`) on its
own columns `base + j · stride …`, `stride` at least `e`'s `COLS`, inside `d`'s. Copy `j`'s input `t` is the reference
`at + j · step + t'` of `d`, `t` the `t'`-th input of a segment of `len` (segments in order, their lengths summing to `e`'s
`IN`; `step = 0` binds every copy to the same references).

Unrolled, a copy at global offset `o` with its inputs bound to global columns writes each of its rows at `o + col`, its
own columns `x ↦ o + x`, its inputs to their columns and the constant to the unit's: substituted, so a copy adds no rows of
its own. Each row is then canonical as `flock-ir-unit/v2` writes it, a sum over GF(2): its columns ascending, a column
named an even number of times dropped. The root has no inputs; the unit's columns are its, and the constant its last,
`[const]·[const]`. Every column is written at most once, and one never written is empty.
-/
import Flock.Net

namespace Flock.Tmpl

/-- A segment of a call's input binding: `len` inputs, bound for copy `j` to caller references `at + j · step + t'`. -/
structure Seg where
  len : Nat
  at_ : Nat
  step : Nat
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

def FORMAT : String := "flock-ir-tmpl/v1"

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
      if ct.size != 6 + 3 * nseg then throw s!"{FORMAT}: definition {defs.size}: call {i} is malformed"
      let mut segs := #[]
      for s in [0:nseg] do
        segs := segs.push ⟨← num ct (6 + 3 * s) "CALL", ← num ct (7 + 3 * s) "CALL", ← num ct (8 + 3 * s) "CALL"⟩
      let some callee := defs[e]? | throw s!"{FORMAT}: definition {defs.size}: call {i} names no earlier definition"
      if count == 0 || (count > 1 && stride < callee.cols) || base + (count - 1) * stride + callee.cols > cols then
        throw s!"{FORMAT}: definition {defs.size}: call {i}'s copies overlap, or overrun its columns"
      if segs.foldl (· + ·.len) 0 != callee.ins then
        throw s!"{FORMAT}: definition {defs.size}: call {i}'s segments do not bind its callee's inputs"
      if segs.any (fun s => s.len > 0 && s.at_ + (count - 1) * s.step + s.len > refs) then
        throw s!"{FORMAT}: definition {defs.size}: call {i} binds past its references"
      calls := calls.push { def_ := e, count, base, stride, segs }
    defs := defs.push { cols, ins, rows, calls }
    at_ := at_ + 1 + nrows + ncalls
  if root ≥ defs.size then throw s!"{FORMAT}: the root names no definition"
  if (defs.getD root default).ins != 0 || (defs.getD root default).cols == 0 then
    throw s!"{FORMAT}: the root has inputs, or no constant"
  return { name, root, inGroups, outGroups, defs, leaves }

/-- A row as `flock-ir-unit/v2` writes it: a sum over GF(2), its columns ascending, each named an odd number of times. -/
def canon (xs : Array Nat) : Array Nat := Id.run do
  let s := xs.qsort (· < ·)
  let mut out := #[]
  let mut i := 0
  while i < s.size do
    let c := s[i]!
    let mut j := i + 1
    while j < s.size && s[j]! == c do j := j + 1
    if (j - i) % 2 == 1 then out := out.push c
    i := j
  return out

/-- The unrolled rows so far: each column's A and B, and whether it was written. -/
structure Acc where
  ra : Array (Array Nat)
  rb : Array (Array Nat)
  set : Array Bool

private def write (acc : Acc) (c : Nat) (a b : Array Nat) : V Acc := do
  if acc.set.getD c true then throw s!"{FORMAT}: column {c} is written twice, or past the unit"
  return { ra := acc.ra.set! c a, rb := acc.rb.set! c b, set := acc.set.set! c true }

/-- Write Definition `d`'s copy at global offset `o`, its inputs bound to the global columns `inp`, the unit's constant
`k`. `fuel` bounds the call depth; a callee is an earlier definition, so `defs.size` is enough. -/
def place (defs : Array Def) (k : Nat) : Nat → Nat → Nat → Array Nat → Acc → V Acc
  | 0, _, _, _, _ => throw s!"{FORMAT}: calls nest deeper than the definitions"
  | fuel + 1, d, o, inp, acc0 => do
    let some df := defs[d]? | throw s!"{FORMAT}: no definition {d}"
    let g := fun x => if x < df.cols then o + x else if x < df.cols + df.ins then inp.getD (x - df.cols) k else k
    let mut acc := acc0
    for (col, a, b) in df.rows do
      acc ← write acc (o + col) (canon (a.map g)) (canon (b.map g))
    for c in df.calls do
      for j in [0:c.count] do
        let mut bound := #[]
        for s in c.segs do
          for t in [0:s.len] do
            bound := bound.push (g (s.at_ + j * s.step + t))
        acc ← place defs k fuel c.def_ (o + c.base + j * c.stride) bound acc
    return acc

/-- **The flat net a templated unit stands for**: the root's copy at offset 0, its last column the unit's constant, then
`Net.ofRows` with the checks `Net.parse` runs on a flat unit. -/
def unroll (t : Tmpl) : V Net := do
  let cols := (t.defs.getD t.root default).cols
  let k := cols - 1
  let acc0 : Acc := { ra := Array.replicate cols #[], rb := Array.replicate cols #[], set := Array.replicate cols false }
  let acc ← place t.defs k (t.defs.size + 1) t.root 0 #[] acc0
  let acc ← write acc k #[k] #[k]
  Net.ofRows t.name cols k t.inGroups t.outGroups acc.ra acc.rb

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
