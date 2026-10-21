/-
A templated unit, `flock-ir-tmpl/v1` (PRF-133): a `flock-ir-unit/v2` unit's rows given once per Definition, with placed
copies, so its text is O(description) where the flat form is O(unrolled). No verifier path reads it: `HmRow.parse` takes a
flat `NET` block only, and the guarantees read the flat circuit `Desc.unroll` stands for. `unroll` is the reference, the
flat net it stands for, built by `Net.ofRows` with every check `Net.parse` runs; `Flock.Test.Tmpl` holds it to `Net.parse`
on the flat text.

The text, one item a line:

* `flock-ir-tmpl/v1 NAME ROOT <in groups> <out groups>`: the unit's name, its root Definition, and its port groups as
  `flock-ir-unit/v2`'s header writes them;
* per Definition `d = 0, 1, …`: `DEF COLS IN ROWS CALLS`, then `ROWS` rows `col na a… nb b…`, then `CALLS` lines
  `CALL e count base stride segs (len at step)…`.

Definition `d`'s local columns are `0 … COLS − 1`; the last, `COLS − 1`, is its constant, and `0 … IN − 1` its inputs. A
row `col na a… nb b…` gives local column `col`'s A and B, in local columns. A call places `count` copies of Definition `e`
(`e < d`) at local offsets `base + j · stride`, `stride` a power of two at least `e`'s `COLS`, `base` a multiple of it. Copy
`j`'s input `t` is bound to local column `at + j · step + t'` of the caller, `t` the `t'`-th input of a segment of `len`
(segments in order, their lengths summing to `e`'s `IN`; `step = 0` binds every copy to the same columns).

Unrolled, a copy at global offset `o` writes, at column `o + t` for each input `t`, the binding row `[bound column]·[const]`,
and each of its rows at `o + col`, its columns `x ↦ o + x` and its constant to the unit's. The unit's constant is the
root's, `COLS − 1`, with `[const]·[const]`; the root has no inputs (its rows give the input rows). Every column is written
at most once, and one never written is empty.
-/
import Flock.Net

namespace Flock.Tmpl

/-- A segment of a call's input binding: `len` inputs, bound for copy `j` to caller columns `at + j · step + t'`. -/
structure Seg where
  len : Nat
  at_ : Nat
  step : Nat
  deriving Inhabited, Repr, BEq

/-- A call: `count` copies of Definition `def_` at offsets `base + j · stride`, inputs bound by `segs`. -/
structure Call where
  def_ : Nat
  count : Nat
  base : Nat
  stride : Nat
  segs : Array Seg
  deriving Inhabited, Repr, BEq

/-- A Definition: its column count (its constant the last), its inputs, its own rows `(col, A, B)` and its calls. -/
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

/-- Parse a templated unit's lines. Checks the form only: each call's callee is an earlier Definition, its stride an
aligned power of two its callee fits in, and its segments cover its callee's inputs. -/
def parse (lines : Array String) : V Tmpl := do
  let h := tokens (lines.getD 0 "")
  if h[0]? != some FORMAT then throw s!"not a {FORMAT} unit"
  let name := h.getD 1 ""
  let root ← num h 2 "header"
  let (inGroups, at1) ← groups h 3
  let (outGroups, at2) ← groups h at1
  if at2 != h.size then throw s!"{FORMAT}: trailing header fields"
  let mut defs : Array Def := #[]
  let mut at_ := 1
  while at_ < lines.size do
    let dh := tokens lines[at_]!
    if dh.size != 5 || dh[0]! != "DEF" then throw s!"{FORMAT}: line {at_} is not a DEF line"
    let cols ← num dh 1 "DEF"
    let ins ← num dh 2 "DEF"
    let nrows ← num dh 3 "DEF"
    let ncalls ← num dh 4 "DEF"
    if cols == 0 || ins ≥ cols then throw s!"{FORMAT}: definition {defs.size}: its inputs leave no constant"
    let mut rows := #[]
    for i in [0:nrows] do
      let v ← (tokens (lines.getD (at_ + 1 + i) "")).mapM fun t => match t.toNat? with
        | some n => pure n
        | none => throw s!"{FORMAT}: definition {defs.size}: row {i} is not numbers"
      let na := v.getD 1 0
      let nb := v.getD (2 + na) 0
      if v.size < 2 || v.size != 3 + na + nb then throw s!"{FORMAT}: definition {defs.size}: row {i} is malformed"
      rows := rows.push (v[0]!, v.extract 2 (2 + na), v.extract (3 + na) (3 + na + nb))
    let mut calls := #[]
    for i in [0:ncalls] do
      let ln := lines.getD (at_ + 1 + nrows + i) ""
      let ct := tokens ln
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
      if !(isPow2 stride && stride ≥ callee.cols && base % stride == 0) then
        throw s!"{FORMAT}: definition {defs.size}: call {i}'s copies are not at aligned power-of-two slots its callee fits"
      if segs.foldl (· + ·.len) 0 != callee.ins then
        throw s!"{FORMAT}: definition {defs.size}: call {i}'s segments do not bind its callee's inputs"
      calls := calls.push { def_ := e, count, base, stride, segs }
    defs := defs.push { cols, ins, rows, calls }
    at_ := at_ + 1 + nrows + ncalls
  if root ≥ defs.size then throw s!"{FORMAT}: the root names no definition"
  if (defs.getD root default).ins != 0 then throw s!"{FORMAT}: the root has inputs; its rows give the input rows"
  return { name, root, inGroups, outGroups, defs }

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
    let g := fun x => if x + 1 == df.cols then k else o + x
    let mut acc := acc0
    for t in [0:df.ins] do
      acc ← write acc (o + t) #[inp.getD t 0] #[k]
    for (col, a, b) in df.rows do
      if col < df.ins || col + 1 ≥ df.cols then
        throw s!"{FORMAT}: definition {d}: a row at column {col}, an input or not below its constant"
      if (a ++ b).any (· ≥ df.cols) then throw s!"{FORMAT}: definition {d}: row {col} reads past its columns"
      acc ← write acc (o + col) (a.map g) (b.map g)
    for c in df.calls do
      let cols := (defs.getD c.def_ default).cols
      for j in [0:c.count] do
        let slot := c.base + j * c.stride
        if slot + cols > df.cols then throw s!"{FORMAT}: definition {d}: a call's copy {j} overruns its constant"
        let mut bound := #[]
        for s in c.segs do
          for t in [0:s.len] do
            let x := s.at_ + j * s.step + t
            if x ≥ df.cols then throw s!"{FORMAT}: definition {d}: a call's copy {j} binds past its columns"
            bound := bound.push (g x)
        acc ← place defs k fuel c.def_ (o + slot) bound acc
    return acc

/-- **The flat net a templated unit stands for**: the root's copy at offset 0, its constant the unit's, then
`Net.ofRows` with the checks `Net.parse` runs on a flat unit. -/
def unroll (t : Tmpl) : V Net := do
  let cols := (t.defs.getD t.root default).cols
  let k := cols - 1
  let acc0 : Acc := { ra := Array.replicate cols #[], rb := Array.replicate cols #[], set := Array.replicate cols false }
  let acc ← place t.defs k (t.defs.size + 1) t.root 0 #[] acc0
  let acc ← write acc k #[k] #[k]
  Net.ofRows t.name cols k t.inGroups t.outGroups acc.ra acc.rb

/-- The unrolled net's `flock-ir-unit/v2` lines, as `Net.parse` reads them. -/
def flatLines (t : Tmpl) : V (Array String) := do
  let n ← unroll t
  let grp := fun (gs : Array Group) => s!"{gs.size}" ++ String.join (gs.toList.map fun g =>
    s!" {g.col} {g.words} {g.bits.size}" ++ String.join (g.bits.toList.map (s!" {·}")))
  let row := fun (m : Sparse) (i : Nat) =>
    let cs := (List.range (m.len.getD i 0)).map (m.col i)
    s!"{cs.length}" ++ String.join (cs.map (s!" {·}"))
  let mut out := #[s!"flock-ir-unit/v2 {n.name} {n.useful} {n.constPos} {grp n.inGroups} {grp n.outGroups}"]
  for i in [0:n.useful] do out := out.push s!"{row n.a i} {row n.b i}"
  return out

end Flock.Tmpl
