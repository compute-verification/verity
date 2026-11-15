module

public import Flock.Circuit
public import Flock.HmRow
public import Flock.KnownRead
public import Flock.Program
public import Std.Data.HashMap
meta import Flock.Circuit
meta import Flock.HmRow
meta import Flock.KnownRead
meta import Flock.Program
meta import Std.Data.HashMap

/-!
# A sampled statement's bits

The one layout of a sampled statement's gates, as code: the Boolean-profile program a registered Flock circuit file
describes, where each of its gates sits, and each unit's gates and inputs. The sampled leg's bridge,
`Properties.NetworkScan.Flattens` and NetworkCheck read these definitions and hold no second copy of them.

* `Src`, `Index`, `srcAt`, `srcKey`, `keyIdx`: Δ's source keys (`Aliased.srcKey`), the one definition of them in the
  code: the column Δ copies into each unit input, read through the circuit's `Index` of row ports, leaf cuts and wires,
  and each input's key, the first input of its source.
* `keyProg c n`: the circuit's program on Δ's source keys, for `n` units. Its input gates are the constant (gate 0), the
  zero (gate 1) and `B` source gates per unit, unit `u`'s input column `x` on its key's (`keyIdx`), or on the zero's where
  it has none. Then come the units, each the unit net's rows (`Rows.ofNet`) stacked `upv` times (`Rows.stack`), a row
  reading earlier gates. Its outputs are the gates on the returned outputs' bits.
* `gateSize`, `off`, `posOf`, `lower`, `prog`: that program lowered to the Boolean profile (`LGate`), gate by gate in
  order. A row `(⊕ a) ∧ (⊕ b)` is three gates, the two XORs then their AND, and an input is one, `.const b` where the
  constants `κ` fix it to `b`. Gate `g` sits at `posOf`, its last lowered gate.
* `unitOf`, `unitGates`, `owner`, `insOf`, `blockOf`: unit `u`'s lowered gates, the input gates it owns (the unfixed
  inputs whose first reader it is) and its proof block, the first then the second's positions.
* `owners`, `insOfAt`: every gate's `owner` from one pass a unit, and `insOf` reading it, equal to them (`getD_owners`).
* `offs`, `offAt`, `posAt`, `lowerAt`, `unitGatesAt`, `blockOfAt`, `progFast`, `blockOfFast`: every gate's `off` from
  one pass and `prog` and `blockOf` reading it (`blockOf` the owners too), equal to them (`offAt_offs`) and compiled in
  their place (`@[csimp]`).
* `inBit`, `outBit`: the claims position of a bit of an instance's input port or returned output leaf, the instance
  `(u, k)` being unit `u`'s template instance `k < upv`.
* `RowPos`, `rowBit`, `outRowBit`: a commitment position (a registered row, its leaf and a bit), where a source gate
  reads and where an output bit is committed.
* `consts`: the key program's constants, its constant set, and its zero and every input gate no unit reads clear;
  `constsAt` is its answer reading the owners (`constsAt_owners`), which a caller asking at many gates builds once.
* `Members`, `base`, `wiresOf`: an audit's members, each a statement on a circuit file over some of the partition's VUs,
  side by side, and each source bit's wire to the first claims position at its commitment position, compared at the
  partition's VUs: the writer's output bit, else the first copy.
* `defWidth d`, `bitsOf`, `fieldsOf`, `wordsOf`, `retWords`: the width of a primitive's leaf, a type's bits, a return's
  fields, and the 16-bit words a parameter and a return are packed on, a return each field on words of its own.
* `runsOf`, `paramsWhy`: a file's input rows run by parameter, and whether its unit's input ports read each parameter's
  words off them.
* `check c P tmpl`: why Bits doesn't lay `c` out as template `tmpl` of `P`, or none.
* `unitReads`, `readsWhy`: the unit's input columns its rows read, and whether they read input padding or leave a bit of
  a named parameter unread; a check of the unit net's rows, which `check` never reads.

The circuit file holds each unit's Boolean circuit as a whole, not each primitive's: no row is a primitive's own, so a
packed primitive's bits are placed through its unit's ports, never through its Definition.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.Bits

/-- **A Boolean-profile gate**, constructor for constructor `Claims.Sampling.BGate`: an input, a constant, the negation
of a gate, or the AND or XOR of a list of gates. -/
inductive LGate where
  | input
  | const (b : Bool)
  | not (i : Nat)
  | and (ws : List Nat)
  | xor (ws : List Nat)
  deriving Inhabited, BEq, Repr

/-- The gates an `LGate` reads. -/
def LGate.reads : LGate → List Nat
  | .input | .const _ => []
  | .not i => [i]
  | .and ws | .xor ws => ws

/-! ## A template's rows -/

/-- **A template's rows**: columns below `nIn` are its inputs, column `nIn + i` its computed row `i`, and column
`nIn + nComp` its constant. Row `i` is `(⊕ a[i]) ∧ (⊕ b[i])` over columns. -/
structure Rows where
  nIn : Nat
  nComp : Nat
  a : Array (List Nat)
  b : Array (List Nat)
  deriving Inhabited

/-- **A net's rows** (`Rows.ofNet`): its input columns are the first `min (inWords · WORD) constPos`, its computed rows
the rest up to the constant, each read off the net's A and B. -/
def Rows.ofNet (net : Flock.Net) : Rows :=
  let nIn := min (net.inWords * Flock.WORD) net.constPos
  let nComp := net.constPos - nIn
  { nIn, nComp
    a := (Array.range nComp).map fun i => (net.a.row (nIn + i)).toList
    b := (Array.range nComp).map fun i => (net.b.row (nIn + i)).toList }

/-- **Copy `u`'s column `col`** among `k` stacked copies (`Rows.stackCol`): input `col` is input `u·nIn + col`, computed
row `col − nIn` follows the copy's constant row, and the constant is the copy's constant row. -/
def Rows.stackCol (R : Rows) (k u col : Nat) : Nat :=
  if col < R.nIn then u * R.nIn + col
  else if col < R.nIn + R.nComp then k * R.nIn + (u * (R.nComp + 1) + (1 + (col - R.nIn)))
  else k * R.nIn + (u * (R.nComp + 1) + 0)

/-- **Row `i` of `k` stacked copies** of the rows `f` (`Rows.stackRow`): a copy's constant row reads the shared constant,
and its computed row `j` reads its own columns. -/
def Rows.stackRow (R : Rows) (k : Nat) (f : Array (List Nat)) (i : Nat) : List Nat :=
  if i % (R.nComp + 1) = 0 then [k * R.nIn + k * (R.nComp + 1)]
  else (f.getD (i % (R.nComp + 1) - 1) []).map (R.stackCol k (i / (R.nComp + 1)))

/-- **`k` copies of the rows `R`, stacked** (`Rows.stack`): every copy's inputs; then, copy by copy, the copy's constant
row and its computed rows; then the shared constant. -/
def Rows.stack (R : Rows) (k : Nat) : Rows :=
  { nIn := k * R.nIn, nComp := k * (R.nComp + 1)
    a := (Array.range (k * (R.nComp + 1))).map (R.stackRow k R.a)
    b := (Array.range (k * (R.nComp + 1))).map (R.stackRow k R.b) }

/-! ## Δ's source keys of the unit's inputs -/

/-- **The column Δ copies into a unit input, without the template instance** (`Aliased.Src`): a forced-zero row,
message bit `bit` of byte `byte` of row port `p`, output column `col` of net `net`'s slot `slot`, literal column `col`,
or input padding row `off` of copy `k`. -/
inductive Src
  | zero
  | msg (p byte bit : Nat)
  | wire (net slot col : Nat)
  | lit (col : Nat)
  | pad (k off : Nat)
  deriving DecidableEq, Hashable, Inhabited, Repr

/-- The last index of the ascending `s` whose entry is at most `x`, or 0
(`((List.range s.size).reverse.find? (s[·]! ≤ x)).getD 0`). -/
def lastLe (s : Array Nat) (x : Nat) : Nat := Id.run do
  let mut lo := 0
  let mut hi := s.size
  while lo < hi do
    let mid := (lo + hi) / 2
    if s.getD mid 0 ≤ x then lo := mid + 1 else hi := mid
  return lo - 1

/-- What `srcAt` looks up once a circuit: the row ports' first words, each `(copy, port)`'s first leaf cut and its first
wire into the unit net. -/
structure Index where
  starts : Array Nat
  cuts : Std.HashMap (Nat × Nat) Nat
  wires : Std.HashMap (Nat × Nat) (Nat × Nat × Nat)

def Index.of (c : Flock.Circuit) : Index :=
  { starts := Flock.KnownRead.starts c
    cuts := c.leafCuts.foldl (fun m (k, p, w) => if m.contains (k, p) then m else m.insert (k, p) w) {}
    wires := c.wires.foldl (fun m w =>
      if w.dst.1 == c.unitNet && !m.contains (w.dst.2.1, w.dst.2.2) then m.insert (w.dst.2.1, w.dst.2.2) w.src else m) {} }

/-- Leaf word `x`'s bit `t` as row port `q`'s message bit, where `q` is the port holding word `x`: within its bits, or
the zero. -/
def msgAt (c : Flock.Circuit) (ix : Index) (x t : Nat) (wide : Bool) : Src :=
  let q := lastLe ix.starts x
  let s := ix.starts.getD q 0
  if (wide || t < 16) && 16 * (x - s) + t < (c.ports.getD q default).bits then .msg q (2 * (x - s) + t / 8) (t % 8)
  else .zero

/-- Copy `k`'s input port `p`, bit `t` (`Aliased.srcAt`): its leaf's message bit for a leaf port, else its first leaf
cut's, else its first wire's output column, else `.lit 0`; `check` makes the last two unreachable by refusing a wire
into the unit and an input port past the leaf ports that no one leaf cut wires. -/
def srcAt (c : Flock.Circuit) (ix : Index) (k p t : Nat) : Src :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    let l := (c.leavesIn.getD k #[]).getD p 0
    if l < 0 then .zero else msgAt c ix l.toNat t true
  else match ix.cuts[(k, p)]? with
    | some w => msgAt c ix w t false
    | none => match ix.wires[(k, p)]? with
      | some (net, slot, j) =>
        .wire net slot (((Flock.portCols (c.nets.getD net default).2.outGroups).getD j default).1 + t)
      | none => .lit 0

/-- **Stacked input `x`'s key** (`Aliased.srcKey`): copy `x / W`'s input at slot offset `x % W` (`W = inWords · WORD`),
`srcAt` at the port bit it is, or the padding row it is. -/
def srcKey (c : Flock.Circuit) (ix : Index) (x : Nat) : Src :=
  let W := c.unit.inWords * Flock.WORD
  let pc := Flock.portCols c.unit.inGroups
  let off := x % W
  let col := fun p => pc.getD p (0, 0)
  match (List.range pc.size).find? fun p => decide ((col p).1 ≤ off ∧ off < (col p).1 + (col p).2) with
  | some p => srcAt c ix (x / W) p (off - (col p).1)
  | none => .pad (x / W) off

/-- **Equal sources are equal keys** (`Aliased.keyIdx`): of `n` inputs, input `x`'s key is `none` at the zero, else the
first input with the same source. -/
def keyIdx (c : Flock.Circuit) (n x : Nat) : Option Nat :=
  let ix := Index.of c
  if srcKey c ix x = .zero then none else (List.range n).find? fun i => srcKey c ix i = srcKey c ix x

/-! ## The circuit's program -/

/-- **A program of units on source-keyed input gates**: `inputs` input gates, then `units` units of `perUnit` rows each,
gate `inputs + i` the row `rows[i]` (its two lists of gates), and `outs` the gates on returned outputs' bits, ascending. -/
structure KeyProg where
  inputs : Nat
  units : Nat
  perUnit : Nat
  rows : Array (List Nat × List Nat)
  outs : List Nat
  deriving Inhabited

/-- The program's gates. -/
def KeyProg.size (kp : KeyProg) : Nat := kp.inputs + kp.rows.size

/-- Gate `g`'s row, or none for an input. -/
def KeyProg.row? (kp : KeyProg) (g : Nat) : Option (List Nat × List Nat) :=
  if g < kp.inputs then none else kp.rows[g - kp.inputs]?

/-- Column `16 j + t` of the unit's returned-output group, in the unit net's rows (`Layout.outCol`). -/
def outCol (c : Flock.Circuit) (j t : Nat) : Nat := (c.unit.outGroups.getD 0 default).col * Flock.WORD + (16 * j + t)

/-- **Instance `k`'s row on the unit net's computed column `col`, in unit `u`**, of `inputs` input gates and `perUnit`
rows a unit: the instance's constant row, then its computed rows. -/
def rowGate (R0 : Rows) (inputs perUnit u k col : Nat) : Nat :=
  inputs + u * perUnit + (k * (R0.nComp + 1) + (1 + (col - R0.nIn)))

/-- **The key program's gate on bit `t` of returned output leaf `j` of unit `u`'s instance `k`**, for `n` units: its
copy row, none where the leaf's column is not a computed row. -/
def outGate (c : Flock.Circuit) (n u k j t : Nat) : Option Nat :=
  let R0 := Rows.ofNet c.unit
  let R := R0.stack c.upv
  let oc := outCol c j t
  if u < n && k < c.upv && j < (c.leavesOut.getD k #[]).size && t < 16 && R0.nIn ≤ oc && oc < R0.nIn + R0.nComp then
    some (rowGate R0 (2 + n * R.nIn) R.nComp u k oc)
  else none

/-- **The circuit's program on Δ's source keys**, for `n` units (`Placed.keyProg`, its outputs `Layout.outsOf`): the
constant, the zero and `B` source gates per unit, `B` the stacked rows' inputs; then unit by unit the unit net's rows
stacked `upv` times, an input column on its key's source gate `2 + u·B + b` (the zero's where it has no key), a computed
column on the unit's own gate, the constant on gate 0. The outputs are the gates on each instance's returned output
leaves' bits `t < 16`. -/
def keyProg (c : Flock.Circuit) (n : Nat) : KeyProg :=
  let R0 := Rows.ofNet c.unit
  let R := R0.stack c.upv
  let B := R.nIn
  let keys := (Array.range B).map (keyIdx c B)
  let inputs := 2 + n * B
  let wire := fun (u col : Nat) =>
    if col < R.nIn then (match keys.getD col none with | some b => 2 + u * B + b | none => 1)
    else if col < R.nIn + R.nComp then inputs + u * R.nComp + (col - R.nIn)
    else 0
  let rows := ((List.range n).flatMap fun u => (List.range R.nComp).map fun i =>
    ((R.a.getD i []).map (wire u), (R.b.getD i []).map (wire u))).toArray
  let outs := (List.range n).flatMap fun u => (List.range c.upv).flatMap fun k =>
    (List.range (c.leavesOut.getD k #[]).size).flatMap fun j => (List.range 16).filterMap (outGate c n u k j)
  { inputs, units := n, perUnit := R.nComp, rows, outs := (outs.mergeSort fun x y => decide (x ≤ y)).eraseDups }

/-! ## The program lowered to the Boolean profile -/

/-- **The Boolean-profile gates gate `g` lowers to**: three for a row, one for an input. -/
def gateSize (kp : KeyProg) (g : Nat) : Nat := if (kp.row? g).isSome then 3 else 1

/-- **Where gate `g`'s lowered gates start**: the lowered sizes of the gates before it. -/
def off (kp : KeyProg) (g : Nat) : Nat :=
  (((List.range kp.size).filter (· < g)).map (gateSize kp)).sum

/-- **Gate `g`'s position** in the lowered program: its last lowered gate. -/
def posOf (kp : KeyProg) (g : Nat) : Nat := off kp g + gateSize kp g - 1

/-- **Gate `g` lowered**: a row to the XOR of its first list, the XOR of its second and their AND; an input the constants
`κ` fix to `b` to `.const b`, any other input to `.input`. -/
def lower (kp : KeyProg) (κ : Nat → Option Bool) (g : Nat) : List LGate :=
  match kp.row? g with
  | some (a, b) => [.xor (a.map (posOf kp)), .xor (b.map (posOf kp)), .and [off kp g, off kp g + 1]]
  | none => match κ g with
    | some b => [.const b]
    | none => [.input]

/-- **The program as Boolean-profile gates** at constants `κ`: its gates lowered in order, and its outputs' positions. -/
def prog (kp : KeyProg) (κ : Nat → Option Bool) : List LGate × List Nat :=
  ((List.range kp.size).flatMap (lower kp κ), kp.outs.map (posOf kp))

/-! ## Units and proof blocks -/

/-- **Gate `g`'s unit**: none for an input, else the unit whose rows hold it. -/
def unitOf (kp : KeyProg) (g : Nat) : Option Nat :=
  if decide (g < kp.inputs) || decide (kp.size ≤ g) || kp.perUnit == 0 then none
  else some ((g - kp.inputs) / kp.perUnit)

/-- **Unit `u`'s lowered gates**. -/
def unitGates (kp : KeyProg) (u : Nat) : List Nat :=
  (List.range kp.size).flatMap fun g =>
    if unitOf kp g == some u then (List.range (gateSize kp g)).map (off kp g + ·) else []

/-- The gates gate `g` reads. -/
def args (kp : KeyProg) (g : Nat) : List Nat :=
  match kp.row? g with
  | some (a, b) => a ++ b
  | none => []

/-- **Gate `g`'s owner**: the first unit that reads it, none when no unit does. -/
def owner (kp : KeyProg) (g : Nat) : Option Nat :=
  (List.range kp.units).find? fun u =>
    (List.range kp.size).any fun h => unitOf kp h == some u && (args kp h).contains g

/-- **Unit `u`'s inputs**: the input gates the constants `κ` don't fix whose owner is `u`, in gate order. -/
def insOf (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) : List Nat :=
  (List.range kp.size).filter fun g => (kp.row? g).isNone && (κ g).isNone && owner kp g == some u

/-- **Unit `u`'s proof block**: its lowered gates, then the position of each input it owns. -/
def blockOf (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) : List Nat :=
  unitGates kp u ++ (insOf kp κ u).map (posOf kp)

/-! ## The owners in one pass

`owner kp g` scans every gate of each unit up to `g`'s owner, so `insOf` as stated, which asks it at every gate, takes
time quadratic in the program's gates for each unit. `owners kp` holds every gate's `owner` from one pass a unit
(`getD_owners`): unit `u`'s pass marks each gate its gates read `some u`, and the passes run from the last unit to the
first, so a gate keeps the first unit that reads it. `insOfAt` is `insOf` reading it. -/

/-- Each gate of `L` marked `some u` in `t`. -/
def markArgs (u : Nat) (L : List Nat) (t : Array (Option Nat)) : Array (Option Nat) :=
  L.foldl (fun t g => t.setIfInBounds g (some u)) t

/-- One step of `markUnit`: the gates gate `h` reads marked `some u` when `h` is one of unit `u`'s gates. -/
def markStep (kp : KeyProg) (u : Nat) (t : Array (Option Nat)) (h : Nat) : Array (Option Nat) :=
  if unitOf kp h == some u then markArgs u (args kp h) t else t

/-- **Unit `u`'s pass**: each gate one of unit `u`'s gates reads marked `some u`, the others as `t` has them. -/
def markUnit (kp : KeyProg) (u : Nat) (t : Array (Option Nat)) : Array (Option Nat) :=
  (List.range kp.size).foldl (markStep kp u) t

/-- **Every gate's `owner`, in one pass a unit**: entry `g` is `owner kp g` for each `g < kp.size` (`getD_owners`). -/
def owners (kp : KeyProg) : Array (Option Nat) :=
  (List.range kp.units).foldr (markUnit kp) (Array.replicate kp.size none)

/-- `insOf kp κ u` read from the owners `ow`. -/
def insOfAt (ow : Array (Option Nat)) (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) : List Nat :=
  (List.range kp.size).filter fun g => (kp.row? g).isNone && (κ g).isNone && ow.getD g none == some u

theorem getD_setIfInBounds (t : Array (Option Nat)) (i j : Nat) (v : Option Nat) (hj : j < t.size) :
    (t.setIfInBounds i v).getD j none = if i = j then v else t.getD j none := by
  rw [Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_setIfInBounds]
  by_cases h : i = j
  · subst h
    simp [hj]
  · simp [h]

theorem size_markArgs (u : Nat) (L : List Nat) (t : Array (Option Nat)) : (markArgs u L t).size = t.size := by
  induction L generalizing t with
  | nil => rfl
  | cons g L ih =>
    show (markArgs u L (t.setIfInBounds g (some u))).size = t.size
    rw [ih]
    simp

theorem getD_markArgs (u j : Nat) (L : List Nat) (t : Array (Option Nat)) (hj : j < t.size) :
    (markArgs u L t).getD j none = if L.contains j then some u else t.getD j none := by
  induction L generalizing t with
  | nil => simp [markArgs]
  | cons g L ih =>
    have hj' : j < (t.setIfInBounds g (some u)).size := by simpa using hj
    show (markArgs u L (t.setIfInBounds g (some u))).getD j none = _
    rw [ih _ hj', getD_setIfInBounds _ _ _ _ hj]
    by_cases hg : g = j
    · subst hg
      simp
    · have hg' : ¬ j = g := fun h => hg h.symm
      simp [hg, hg']

theorem size_markStep (kp : KeyProg) (u : Nat) (t : Array (Option Nat)) (h : Nat) :
    (markStep kp u t h).size = t.size := by
  unfold markStep
  split
  · exact size_markArgs _ _ _
  · rfl

theorem getD_foldl_markStep (kp : KeyProg) (u j : Nat) (hs : List Nat) (t : Array (Option Nat)) (hj : j < t.size) :
    (hs.foldl (markStep kp u) t).getD j none =
      bif hs.any (fun h => unitOf kp h == some u && (args kp h).contains j) then some u else t.getD j none := by
  induction hs generalizing t with
  | nil => rfl
  | cons h hs ih =>
    rw [List.foldl_cons, ih (markStep kp u t h) (by rw [size_markStep]; exact hj)]
    show (bif _ then some u else (markStep kp u t h).getD j none) =
      bif (unitOf kp h == some u && (args kp h).contains j) ||
          hs.any (fun x => unitOf kp x == some u && (args kp x).contains j)
        then some u else t.getD j none
    unfold markStep
    cases unitOf kp h == some u with
    | false => first | rfl | simp
    | true =>
      show (bif _ then some u else (markArgs u (args kp h) t).getD j none) = _
      rw [getD_markArgs _ _ _ _ hj]
      cases (args kp h).contains j <;>
        cases hs.any (fun x => unitOf kp x == some u && (args kp x).contains j) <;> first | rfl | simp

theorem size_foldl_markStep (kp : KeyProg) (u : Nat) (hs : List Nat) (t : Array (Option Nat)) :
    (hs.foldl (markStep kp u) t).size = t.size := by
  induction hs generalizing t with
  | nil => rfl
  | cons h hs ih => rw [List.foldl_cons, ih, size_markStep]

theorem size_foldr_markUnit (kp : KeyProg) (us : List Nat) (t : Array (Option Nat)) :
    (us.foldr (markUnit kp) t).size = t.size := by
  induction us with
  | nil => rfl
  | cons u us ih => rw [List.foldr_cons, markUnit, size_foldl_markStep, ih]

theorem getD_foldr_markUnit (kp : KeyProg) (j : Nat) (us : List Nat) (hj : j < kp.size) :
    (us.foldr (markUnit kp) (Array.replicate kp.size none)).getD j none =
      us.find? fun u => (List.range kp.size).any fun h => unitOf kp h == some u && (args kp h).contains j := by
  induction us with
  | nil =>
    simp only [List.foldr_nil, List.find?_nil]
    first
      | simp [Array.getD_eq_getD_getElem?, hj]
      | simp [Array.getD, hj]
  | cons u us ih =>
    have hj' : j < (us.foldr (markUnit kp) (Array.replicate kp.size none)).size := by
      rw [size_foldr_markUnit, Array.size_replicate]
      exact hj
    rw [List.foldr_cons, markUnit, getD_foldl_markStep _ _ _ _ _ hj', ih]
    first
      | rw [List.find?_cons]
        cases (List.range kp.size).any (fun h => unitOf kp h == some u && (args kp h).contains j) <;> rfl
      | cases hr : (List.range kp.size).any (fun h => unitOf kp h == some u && (args kp h).contains j) <;>
          simp [List.find?, hr]

/-- **The table is `owner`**: entry `g` of `owners kp` is `owner kp g`, at every gate `g` of the program. -/
theorem getD_owners (kp : KeyProg) {g : Nat} (hg : g < kp.size) : (owners kp).getD g none = owner kp g :=
  getD_foldr_markUnit kp g (List.range kp.units) hg

theorem insOfAt_owners (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) :
    insOfAt (owners kp) kp κ u = insOf kp κ u := by
  unfold insOfAt insOf
  refine List.filter_congr fun g hg => ?_
  simp only [getD_owners kp (List.mem_range.1 hg)]

/-! ## The offsets in one pass

`off kp g` sums the lowered sizes of the gates before `g` at every call, so `prog` and `blockOf` as stated take time
quadratic in the program's gates. `offs kp` holds every gate's `off` from one pass (`offAt_offs`); `progFast` and
`blockOfFast` are `prog` and `blockOf` reading it, `blockOf` the owners too, built once. `@[csimp]` compiles them in
`prog`'s and `blockOf`'s place, so evaluation runs them while every theorem reads `prog` and `blockOf`, and `off` and
`owner` stay their specification. -/

/-- One step of `offs`: entry `g + 1`, where gate `g + 1`'s lowered gates start, is entry `g` plus gate `g`'s size. -/
def offsStep (kp : KeyProg) (t : Array Nat) (g : Nat) : Array Nat := t.push (t.getD g 0 + gateSize kp g)

/-- **Every gate's `off`, in one pass**: entry `g` is `off kp g` for each `g ≤ kp.size` (`offAt_offs`). -/
def offs (kp : KeyProg) : Array Nat := (List.range kp.size).foldl (offsStep kp) #[0]

/-- **`off kp g` read from the table `t`**: its entry `g`, its last past the program's gates. -/
def offAt (t : Array Nat) (kp : KeyProg) (g : Nat) : Nat := t.getD (min g kp.size) 0

/-- `posOf kp g` read from the table `t`. -/
def posAt (t : Array Nat) (kp : KeyProg) (g : Nat) : Nat := offAt t kp g + gateSize kp g - 1

/-- `lower kp κ g` read from the table `t`: a row's gates at its positions there, an input as `lower` lowers it. -/
def lowerAt (t : Array Nat) (kp : KeyProg) (κ : Nat → Option Bool) (g : Nat) : List LGate :=
  match kp.row? g with
  | some (a, b) => [.xor (a.map (posAt t kp)), .xor (b.map (posAt t kp)), .and [offAt t kp g, offAt t kp g + 1]]
  | none => lower kp κ g

/-- `unitGates kp u` read from the table `t`. -/
def unitGatesAt (t : Array Nat) (kp : KeyProg) (u : Nat) : List Nat :=
  (List.range kp.size).flatMap fun g =>
    if unitOf kp g == some u then (List.range (gateSize kp g)).map (offAt t kp g + ·) else []

/-- `blockOf kp κ u` read from the offsets `t` and the owners `ow`, which a caller builds once for all of `kp`'s
units. -/
def blockOfAt (t : Array Nat) (ow : Array (Option Nat)) (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) :
    List Nat :=
  unitGatesAt t kp u ++ (insOfAt ow kp κ u).map (posAt t kp)

/-- `prog`, its offsets built once (`prog_eq_progFast`). -/
def progFast (kp : KeyProg) (κ : Nat → Option Bool) : List LGate × List Nat :=
  let t := offs kp
  ((List.range kp.size).flatMap (lowerAt t kp κ), kp.outs.map (posAt t kp))

/-- `blockOf`, its offsets and owners built once (`blockOf_eq_blockOfFast`). -/
def blockOfFast (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) : List Nat :=
  blockOfAt (offs kp) (owners kp) kp κ u

theorem filter_range_lt (n g : Nat) : (List.range n).filter (· < g) = List.range (min g n) := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [List.range_succ, List.filter_append, ih]
    by_cases h : n < g
    · rw [Nat.min_eq_right (show n ≤ g by omega), Nat.min_eq_right (show n + 1 ≤ g by omega), List.range_succ]
      simp [h]
    · rw [Nat.min_eq_left (show g ≤ n by omega), Nat.min_eq_left (show g ≤ n + 1 by omega)]
      simp [h]

/-- `off kp g` is the lowered sizes of the gates below `g`, the program's when `g` is past it. -/
theorem off_eq (kp : KeyProg) (g : Nat) : off kp g = ((List.range (min g kp.size)).map (gateSize kp)).sum := by
  rw [off, filter_range_lt]

theorem offs_fold (kp : KeyProg) (n : Nat) :
    ((List.range n).foldl (offsStep kp) #[0]).size = n + 1 ∧
      ∀ i ≤ n, ((List.range n).foldl (offsStep kp) #[0]).getD i 0 = ((List.range i).map (gateSize kp)).sum := by
  induction n with
  | zero =>
    refine ⟨rfl, fun i hi => ?_⟩
    obtain rfl : i = 0 := by omega
    rfl
  | succ n ih =>
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
    obtain ⟨hs, hg⟩ := ih
    refine ⟨by rw [offsStep, Array.size_push, hs], fun i hi => ?_⟩
    rw [offsStep, Array.getD_eq_getD_getElem?, Array.getElem?_push, hs]
    by_cases h : i = n + 1
    · subst h
      rw [ite_eq_left rfl, Option.getD_some, hg n (Nat.le_refl n), List.range_succ, List.map_append, List.sum_append]
      simp
    · rw [ite_eq_right h, ← Array.getD_eq_getD_getElem?]
      exact hg i (by omega)

/-- **The table is `off`**: entry `g` of `offs kp` is `off kp g`, at every `g`. -/
theorem offAt_offs (kp : KeyProg) (g : Nat) : offAt (offs kp) kp g = off kp g := by
  rw [offAt, off_eq]
  exact (offs_fold kp kp.size).2 _ (Nat.min_le_right _ _)

theorem posAt_offs (kp : KeyProg) : posAt (offs kp) kp = posOf kp := by
  funext g
  rw [posAt, posOf, offAt_offs]

theorem lowerAt_offs (kp : KeyProg) (κ : Nat → Option Bool) : lowerAt (offs kp) kp κ = lower kp κ := by
  funext g
  unfold lowerAt
  rcases h : kp.row? g with _ | ⟨a, b⟩
  · rfl
  · simp only [lower, h, posAt_offs, offAt_offs]

theorem unitGatesAt_offs (kp : KeyProg) (u : Nat) : unitGatesAt (offs kp) kp u = unitGates kp u := by
  simp only [unitGatesAt, unitGates, offAt_offs]

theorem blockOfAt_offs (kp : KeyProg) (κ : Nat → Option Bool) (u : Nat) :
    blockOfAt (offs kp) (owners kp) kp κ u = blockOf kp κ u := by
  rw [blockOfAt, blockOf, unitGatesAt_offs, posAt_offs, insOfAt_owners]

@[csimp] theorem prog_eq_progFast : @prog = @progFast := by
  funext kp κ
  simp only [prog, progFast, lowerAt_offs, posAt_offs]

@[csimp] theorem blockOf_eq_blockOfFast : @blockOf = @blockOfFast := by
  funext kp κ u
  exact (blockOfAt_offs kp κ u).symm

/-! ## Leaf bits -/

/-- **Bit `t` of instance `k`'s input port `p`** among the stacked rows' inputs: the instance's columns, then the port's. -/
def inCol (c : Flock.Circuit) (k p t : Nat) : Nat :=
  k * (Rows.ofNet c.unit).nIn + (((Flock.portCols c.unit.inGroups).getD p default).1 + t)

/-- **The key program's gate on bit `t` of input port `p` of unit `u`'s instance `k`**, for `n` units: the source gate of
its key, none at the zero or past the port. -/
def inGate (c : Flock.Circuit) (n u k p t : Nat) : Option Nat :=
  let B := ((Rows.ofNet c.unit).stack c.upv).nIn
  let pc := Flock.portCols c.unit.inGroups
  if u < n && k < c.upv && p < pc.size && t < (pc.getD p default).2 then
    (keyIdx c B (inCol c k p t)).map fun b => 2 + u * B + b
  else none

/-- **The claims position of bit `t` (least significant first) of input port `p` of unit `u`'s instance `k`**, for `n`
units: its source gate's. -/
def inBit (c : Flock.Circuit) (n u k p t : Nat) : Option Nat := (inGate c n u k p t).map (posOf (keyProg c n))

/-- **The claims position of bit `t` of returned output leaf `j` of unit `u`'s instance `k`**, for `n` units: its copy
row's AND, never an input or a constant. -/
def outBit (c : Flock.Circuit) (n u k j t : Nat) : Option Nat := (outGate c n u k j t).map (posOf (keyProg c n))

/-- **A commitment position**: bit `bit` of leaf `leaf` of the registered row named `row`. A circuit file's port `q` is
the row whose frame-v3 root's domain is its name (`PortRoots.domOf`), and unit `u`'s row of it is that root's leaf `u`
(`PortRoots.leafOf`). -/
structure RowPos where
  row : String
  leaf : Nat
  bit : Nat
  deriving DecidableEq, Inhabited, Repr

/-- **Where unit `u` reads source `s`**: message bit `bit` of byte `byte` of port `p` is bit `8 byte + bit` of the unit's
leaf of that row; a forced zero, a wire, a literal and padding are no commitment position. -/
def srcPos (c : Flock.Circuit) (u : Nat) : Src → Option RowPos
  | .msg p byte bit => some { row := (c.ports.getD p default).name, leaf := u, bit := 8 * byte + bit }
  | _ => none

/-- **Source gate `g`'s commitment position** (the row map), for `n` units: gate `2 + u·B + b`, where `b` is its own key,
is unit `u`'s read of `srcKey c b`. None for the constant, the zero, a row, a source gate whose key is another input's
(`keyIdx`), and a source no row port holds. An input gate's claims position is the gate itself. -/
def rowBit (c : Flock.Circuit) (n g : Nat) : Option RowPos :=
  let B := ((Rows.ofNet c.unit).stack c.upv).nIn
  if g < 2 || B == 0 || 2 + n * B ≤ g || keyIdx c B ((g - 2) % B) != some ((g - 2) % B) then none
  else srcPos c ((g - 2) / B) (srcKey c (Index.of c) ((g - 2) % B))

/-- **The key program's constants**, for `n` units: gate 0, the constant, set; gate 1, the zero, and every other input
gate no unit reads (`owner`), clear; no other gate fixed. Compiled, it takes the gate as a third argument, so each
answer at another input gate builds the key program and scans it for the gate's owner; `constsAt (owners kp) kp` gives
the same answers with both built once (`constsAt_owners`). -/
def consts (c : Flock.Circuit) (n : Nat) : Nat → Option Bool :=
  let kp := keyProg c n
  fun g => if g == 0 then some true else if g == 1 || (g < kp.inputs && owner kp g == none) then some false else none

/-- `consts`' answer at gate `g` for the key program `kp`, its owners read from `ow`. -/
def constsAt (ow : Array (Option Nat)) (kp : KeyProg) (g : Nat) : Option Bool :=
  if g == 0 then some true else if g == 1 || (g < kp.inputs && ow.getD g none == none) then some false else none

/-- **`consts` from the owners**: `constsAt (owners kp) kp` is `consts c n` at `kp`, its key program. -/
theorem constsAt_owners (c : Flock.Circuit) (n : Nat) :
    constsAt (owners (keyProg c n)) (keyProg c n) = consts c n := by
  funext g
  simp only [constsAt, consts]
  by_cases hg : g < (keyProg c n).inputs
  · rw [getD_owners (keyProg c n) (by unfold KeyProg.size; omega)]
  · simp [hg]

/-- **Where unit `u` commits output word `x`'s bit `t`**: bit `16 (x − at) + t` of its leaf of the output row whose run
of output ports holds the word (`HmOut.outGroup`, `at` the run's first word). -/
def outPos (c : Flock.Circuit) (u x t : Nat) : Option RowPos :=
  let lo o := (Flock.HmOut.outGroup c o).2.2.1
  let wd o := (Flock.HmOut.outGroup c o).2.2.2
  ((List.range c.outRowPorts.size).find? fun o => decide (lo o ≤ x) && decide (x < lo o + wd o)).map fun o =>
    { row := (c.ports.getD (Flock.HmOut.nIn c + o) default).name, leaf := u, bit := 16 * (x - lo o) + t }

/-- **The commitment position of bit `t` of returned output leaf `j` of unit `u`'s instance `k`**: its output word's
(`leavesOut`). -/
def outRowBit (c : Flock.Circuit) (u k j t : Nat) : Option RowPos :=
  ((c.leavesOut.getD k #[])[j]?).bind fun x => outPos c u x t

/-! ## An audit's members -/

/-- **An audit's members**, in its order: each a statement, its circuit file and its units' global indices (the
instance file's `indices`, the partition's VUs it proves), unit `u` of the statement being VU `idx[u]`. Their programs sit
side by side, member `m`'s positions from `base ms m`. -/
abbrev Members := List (Flock.Circuit × Array Nat)

/-- A statement's commitment position at the partition's VUs: its leaf `u` is VU `idx[u]`. -/
def RowPos.globalize (idx : Array Nat) (r : RowPos) : RowPos := { r with leaf := idx.getD r.leaf r.leaf }

/-- **Member `m`'s first claims position**: the lowered gates of the members before it. -/
def base (ms : Members) (m : Nat) : Nat :=
  ((ms.take m).map fun ci => off (keyProg ci.1 ci.2.size) (keyProg ci.1 ci.2.size).size).sum

/-- **Every source bit**: each source gate the constants leave free (`consts`, so a unit reads it) that has a
commitment position, its claims position and that position at the partition's VUs, member by member. -/
def sources (ms : Members) : List (Nat × RowPos) :=
  (List.range ms.length).flatMap fun m =>
    let ci := ms.getD m default
    let b := base ms m
    let κ := consts ci.1 ci.2.size
    (List.range (keyProg ci.1 ci.2.size).inputs).filterMap fun g =>
      if (κ g).isSome then none else (rowBit ci.1 ci.2.size g).map fun r => (b + g, r.globalize ci.2)

/-- **Every written bit**: each returned output leaf bit's claims position (`outBit`) and commitment position at the
partition's VUs, member by member. -/
def writers (ms : Members) : List (Nat × RowPos) :=
  (List.range ms.length).flatMap fun m =>
    let ci := ms.getD m default
    let b := base ms m
    let kp := keyProg ci.1 ci.2.size
    (List.range ci.2.size).flatMap fun u => (List.range ci.1.upv).flatMap fun k =>
      (List.range (ci.1.leavesOut.getD k #[]).size).flatMap fun j => (List.range 16).filterMap fun t =>
        match (outGate ci.1 ci.2.size u k j t).map (posOf kp), outRowBit ci.1 u k j t with
        | some x, some r => some (b + x, r.globalize ci.2)
        | _, _ => none

/-- **Source bit `x`'s wire**: the first claims position at its commitment position, compared at the partition's VUs.
That is the first writer's output bit where a unit writes it, else the first source bit there (the lowest reading
unit's), and none for that first source bit itself or for `x` no source bit. -/
def wiresOf (ms : Members) (x : Nat) : Option Nat :=
  let ss := sources ms
  match ss.find? (·.1 == x) with
  | none => none
  | some (_, r) => match (writers ms).find? (·.2 == r) with
    | some (w, _) => some w
    | none => ((ss.find? (·.2 == r)).map (·.1)).filter (· != x)

/-! ## A Program's leaf widths -/

/-- **The width of a primitive's leaf**: its `Value<w>` return's `w`, and 0 for any other return. An input primitive's
id names the same width, which `Flock.Program.decode` holds it to. -/
def defWidth (d : Flock.Program.Defn) : Nat :=
  match d.ret with
  | .value w => w
  | _ => 0

/-- **A type's bits**: its leaves' widths summed. -/
def bitsOf (t : Flock.Program.Ty) : Nat := t.widths.foldl (· + ·) 0

/-- **A return's fields**: a tuple's or a record's, else the return itself. -/
def fieldsOf : Flock.Program.Ty → List Flock.Program.Ty
  | .tuple fs => fs.toList
  | .record fs => fs.toList.map (·.2)
  | t => [t]

/-- **A value's 16-bit words**: its bits packed sixteen to a word, least significant first, the last word's high bits
padding. A parameter is one value. -/
def wordsOf (t : Flock.Program.Ty) : Nat := (bitsOf t + 15) / 16

/-- **A return's words**: each field packed on words of its own (`wordsOf`). -/
def retWords (t : Flock.Program.Ty) : Nat := ((fieldsOf t).map wordsOf).sum

/-! ## The file against its Program -/

/-- **A row port's parameter**: its name up to its first `#`, so a parameter's rows are named `π` or `π#0`, `π#1`, …. -/
def paramOf (name : String) : String := (name.splitOn "#").headD name

/-- **The input rows' runs**: the input row ports (`HmOut.nIn`) in order, consecutive ones of one parameter (`paramOf`)
merged, each run its parameter, its words and its first word among the rows' (`KnownRead.starts`). -/
def runsOf (c : Flock.Circuit) : List (String × Nat × Nat) :=
  let st := Flock.KnownRead.starts c
  ((List.range (Flock.HmOut.nIn c)).foldl (fun (acc : List (String × Nat × Nat)) q =>
    let p := c.ports.getD q default
    match acc with
    | (nm, w, s) :: rest =>
      if nm == paramOf p.name then (nm, w + p.words, s) :: rest else (paramOf p.name, p.words, st.getD q 0) :: acc
    | [] => [(paramOf p.name, p.words, st.getD q 0)]) []).reverse

/-- **Bit `β` of field `f`'s rows, on the unit**, of fields `words` (each its name and its words a template instance, in
order: a template's parameters, as `Template.rows` gives them, or its return's fields, `Template.outs`): the rows hold
`c.upv` instances' `w` words each, so their word `j = β / 16` is instance `k = j / w`'s input port, or output leaf,
`P + j % w`, `P` the words of the fields before `f`, at bit `β % 16`: `(k, P + j % w, β % 16)`. None for no such field
or a bit past its rows. `paramsWhy` holds a file's input ports to it. -/
def fieldBit (c : Flock.Circuit) (words : List (String × Nat)) (f : String) (β : Nat) : Option (Nat × Nat × Nat) :=
  (words.findIdx? (·.1 == f)).bind fun i =>
    let w := (words.getD i default).2
    if w == 0 || c.upv * w ≤ β / 16 then none
    else some (β / 16 / w, ((words.take i).map (·.2)).sum + β / 16 % w, β % 16)

/-- **Why the unit's input ports don't read the Definition `d`'s parameters off their rows**, or none, for a file whose
input rows are `d`'s parameters in order (`runsOf`), each its packed words (`wordsOf`) an instance: at each bit of
parameter `π`'s rows, its port (`fieldBit`) reads the row word it is, `s + β / 16` with `s` the rows' first word (its
leaf map's word at a leaf port, else its leaf cut's), and each of `π`'s bits there, not the padding past them in an
instance's last word, is a message bit of its row (`srcAt`), not the zero past the row's bits. -/
def paramsWhy (c : Flock.Circuit) (d : Flock.Program.Defn) : Option String := Id.run do
  let ix := Index.of c
  let runs := (runsOf c).toArray
  let cut := (c.unit.inGroups.getD 0 default).bits.size
  let ws := d.params.toList.map fun (nm, t) => (nm, wordsOf t)
  for j in [0:d.params.size] do
    let (nm, t) := d.params[j]!
    let w := wordsOf t
    let s := (runs.getD j default).2.2
    for β in [0:16 * (c.upv * w)] do
      let some (k, p, b) := fieldBit c ws nm β | return some s!"bit {β} of parameter {nm}'s rows is on no input port"
      if b == 0 then
        let x : Int :=
          if p < cut then (c.leavesIn.getD k #[]).getD p (-1) else ((ix.cuts[(k, p)]?).map Int.ofNat).getD (-1)
        if x != ((s + β / 16 : Nat) : Int) then
          return some s!"template instance {k}'s input port {p} reads row word {x}, not word {β / 16} of its parameter \
            {nm}'s rows"
      if 16 * (β / 16 % w) + b < bitsOf t then
        unless (srcAt c ix k p b matches .msg _ _ _) do
          return some s!"bit {16 * (β / 16 % w) + b} of parameter {nm} of template instance {k} is no message bit of \
            its rows"
  return none

/-- **Why Bits doesn't lay out the circuit file `c` as template `tmpl` of the Program `P`**, or none: the template
(its descriptor id, as a circuit record names it) is no Definition of `P`, no template instance makes a unit, a port of
the unit's first input or output group is not one 16-bit word (`srcAt` and `outCol` place a port's bits in one), the
leaf maps are not one per template instance over the unit's ports (a typed template's block, whose wiring is derived,
has none), a leaf map names a word past the input rows (`HmOut.inWords`, so no unit reads a row it writes), a wire
runs into the unit (a `.wire` source, which `wiresOf` doesn't place), a leaf cut is outside the unit's cut inputs or the
input rows, an instance's input port past its leaf ports is not wired exactly once by a leaf cut (the three are
`HmRow.check`'s wiring rule with no wire into the unit, so `srcAt` reaches neither `.wire` nor `.lit 0`), two
instances return one output word (`writers` would hold two claims positions at one commitment position), the unit net's
constant sits among its input words (`Rows.ofNet` and `srcKey` split the inputs at different columns), a net other than
the unit returns the outputs (`keyProg`'s outputs are the unit net's), or a returned output bit sits on no computed row,
so that every writer is a copy row and none an input or a constant. Then it holds the file to the template's Definition
`d`: it refuses two row ports of one name (so a commitment position names one row), input rows that are not `d`'s
parameters in order, each `upv` times its packed words (`runsOf`, `wordsOf`), output rows that are not `upv` times its
return's (`retWords`, each field on words of its own), an instance's output leaves other than its return's words, unit
input ports other than its parameters' words, and an input port that doesn't read its parameter's word or a parameter
bit that is no message bit (`paramsWhy`). Last, it refuses more than one template instance a VU (`upv`): the layouts
the proofs read place one. It reads only the unit net's header (`Rows.ofNet`'s `nIn` and `nComp`), never its rows. -/
def check (c : Flock.Circuit) (P : Flock.Program.Prog) (tmpl : String) : Option String :=
  let nIn := min (c.unit.inWords * Flock.WORD) c.unit.constPos
  let nComp := c.unit.constPos - nIn
  let gi := (c.unit.inGroups.getD 0 default).bits
  let go := (c.unit.outGroups.getD 0 default).bits
  let outs := c.leavesOut.toList.flatMap (·.toList)
  let wired : Std.HashMap (Nat × Nat) Nat :=
    c.leafCuts.foldl (fun m (k, p, _) => m.insert (k, p) (m.getD (k, p) 0 + 1)) {}
  let cut := gi.size
  let pc := Flock.portCols c.unit.inGroups
  let unwired := ((List.range c.upv).flatMap fun k =>
    (List.range (pc.size - cut)).map fun i => (k, cut + i)).find?
    fun kp => wired.getD kp 0 != 1
  if !P.defs.any (·.key == tmpl) then some s!"the circuit file's template {tmpl} is no Definition of the Program"
  else if c.upv == 0 then some "the circuit file's unit holds no template instance"
  else if gi.any (· != 16) || go.any (· != 16) then
    some "a port of the circuit file's unit is not one 16-bit word"
  else if c.leavesIn.size != c.upv || c.leavesOut.size != c.upv || (List.range c.upv).any fun k =>
      (c.leavesIn.getD k #[]).size != gi.size || (c.leavesOut.getD k #[]).size != go.size then
    some "the circuit file's leaf maps are not one per template instance over its unit's ports"
  else if c.leavesIn.any (·.any ((Flock.HmOut.inWords c : Int) ≤ ·)) then
    some "a leaf map names a word past the circuit file's input rows"
  else if c.wires.any (·.dst.1 == c.unitNet) then some "a wire into the circuit file's unit"
  else if c.leafCuts.any fun (k, p, w) =>
      c.upv ≤ k || p < cut || pc.size ≤ p || (pc.getD p (0, 0)).2 < 16 || Flock.HmOut.inWords c ≤ w then
    some "a leaf cut outside the unit's cut inputs or the input rows"
  else if let some (k, p) := unwired then
    some s!"input port {p} of the circuit file's unit, template instance {k}, is not wired exactly once by a leaf cut"
  else if !decide outs.Nodup then
    some "the circuit file's unit returns an output word twice"
  else if nIn != c.unit.inWords * Flock.WORD then
    some "the circuit file's unit net has its constant among its input words"
  else if c.outNet != c.unitNet then some "the circuit file returns its outputs from a net other than its unit"
  else if (List.range c.upv).any fun k => (List.range (c.leavesOut.getD k #[]).size).any fun j =>
      (List.range 16).any fun t => !(decide (nIn ≤ outCol c j t) && decide (outCol c j t < nIn + nComp)) then
    some "a returned output bit of the circuit file's unit sits on no computed row (an input or the constant)"
  else (match P.defs.find? (·.key == tmpl) with
    | none => none
    | some d =>
      let rw := retWords d.ret
      if !decide (c.ports.toList.map (·.name)).Nodup then some "two row ports of the circuit file have one name"
      else if (runsOf c).map (fun r => (r.1, r.2.1)) != d.params.toList.map fun (nm, t) => (nm, c.upv * wordsOf t) then
        some s!"the circuit file's input rows are not its template's parameters in order, each {c.upv} times its \
          packed words"
      else if ((c.ports.toList.drop (Flock.HmOut.nIn c)).map (·.words)).sum != c.upv * rw then
        some s!"the circuit file's output rows are not {c.upv} times its template's return's packed words"
      else if c.leavesOut.any (·.size != rw) then
        some "a template instance's output leaves are not its template's return's packed words"
      else if pc.size != (d.params.toList.map (wordsOf ·.2)).sum then
        some "the circuit file's unit's input ports are not its template's parameters' packed words"
      else paramsWhy c d) <|>
    if c.upv != 1 then some s!"the circuit file has {c.upv} template instances a VU, not one" else none

/-! ## The unit's rows -/

/-- **The unit's read input columns**: of the unit net's input columns (`Rows.ofNet`'s `nIn`), those a computed row
reads in its A or its B. -/
def unitReads (c : Flock.Circuit) : Array Bool :=
  let R := Rows.ofNet c.unit
  (R.a ++ R.b).foldl (fun m r => r.foldl (fun m x => if x < R.nIn then m.set! x true else m) m)
    (Array.replicate R.nIn false)

/-- **Why the unit's rows don't read the Definition `d`'s inputs as Bits counts them**, or none, `reads` the unit's read
input columns (`unitReads`): there is not one per unit input column, a computed row reads an input column no port holds
(input padding, which Δ fills from no row, so its value is the prover's, a source with no commitment position), or no
row reads a bit of a parameter in `names` (`consts` fixes it, so it is no source). -/
def readsWhy (c : Flock.Circuit) (d : Flock.Program.Defn) (reads : Array Bool) (names : List String) :
    Option String := Id.run do
  let pc := Flock.portCols c.unit.inGroups
  let nIn := min (c.unit.inWords * Flock.WORD) c.unit.constPos
  if reads.size != nIn then return some "the circuit file's unit's read columns are not one per input column"
  let mut held := Array.replicate nIn false
  for (a0, w) in pc do
    for t in [0:w] do held := held.set! (a0 + t) true
  for x in [0:nIn] do
    if reads[x]! && !held[x]! then
      return some s!"a row of the circuit file's unit reads its input column {x}, which no port holds"
  let mut P := 0
  for (nm, t) in d.params do
    if names.contains nm then
      for β in [0:bitsOf t] do
        unless reads.getD ((pc.getD (P + β / 16) (0, 0)).1 + β % 16) false do
          return some s!"no row of the circuit file's unit reads bit {β} of its parameter {nm}"
    P := P + wordsOf t
  return none

/-! ## Checks -/

/-- Two units of two rows on the constant, the zero and two source gates: unit 0 reads source 2, unit 1 source 3, and
each unit's second row returns. -/
private def toy : KeyProg :=
  { inputs := 4, units := 2, perUnit := 2
    rows := #[([2], [0]), ([4, 2], [0]), ([3], [0]), ([6], [3])]
    outs := [5, 7] }

private def toyκ (g : Nat) : Option Bool := if g == 0 then some true else if g == 1 then some false else none

#guard (List.range toy.size).map (posOf toy) == [0, 1, 2, 3, 6, 9, 12, 15]
#guard offs toy == #[0, 1, 2, 3, 4, 7, 10, 13, 16]
#guard (List.range (toy.size + 2)).all fun g => offAt (offs toy) toy g == off toy g
#guard (prog toy toyκ).1.length == (List.range toy.size).foldl (fun s g => s + gateSize toy g) 0
#guard lower toy toyκ 0 == [.const true]
#guard lower toy toyκ 2 == [.input]
#guard lower toy toyκ 5 == [.xor [6, 2], .xor [0], .and [7, 8]]
#guard (prog toy toyκ).2 == [9, 15]
#guard let gs := (prog toy toyκ).1; (List.range gs.length).all fun i => (gs[i]!).reads.all (· < i)
#guard blockOf toy toyκ 0 == [4, 5, 6, 7, 8, 9, 2]
#guard blockOf toy toyκ 1 == [10, 11, 12, 13, 14, 15, 3]
#guard owner toy 0 == some 0 && owner toy 1 == none && owner toy 3 == some 1
#guard owners toy == #[some 0, none, some 0, some 1, some 0, none, some 1, none]
#guard (List.range toy.size).all fun g => (owners toy).getD g none == owner toy g

/-- Two 16-bit inputs and three 1-bit inputs into one 32-bit primitive: leaves of widths 16, 16, 1, 1, 1 and 32. -/
private def mixed : String :=
  "{\"annotations\":{},\"callees\":[\"Input16_v1\",\"Input1_v1\",\"TMix_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Input1_v1\":{\"form\":\"primitive\",\"id\":\"Input1_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"Root[TMix_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":0},{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":3,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2]},\"type\":0},{\"refs\":{\"affine\":[\"n\",1,0,3]},\"type\":1}],\"fn\":2,\"form\":\"primitive\",\"out\":2}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",32]},\"form\":\"composite\",\"id\":\"Root[TMix_v1]_v1\"},\"TMix_v1\":{\"form\":\"primitive\",\"id\":\"TMix_v1\",\"params\":[[\"a\",[\"Array\",2,[\"Value\",16]]],[\"b\",[\"Array\",3,[\"Value\",1]]]],\"ret\":[\"Value\",32]}},\"program_fn\":\"TMix_v1\",\"root\":\"Root[TMix_v1]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Array\",2,[\"Value\",16]],[\"Array\",3,[\"Value\",1]],[\"Value\",32]]}"

#guard match Flock.Program.decode mixed.toUTF8 with
  | .ok P => P.defs.all fun d => !d.prim ||
      defWidth d == (if d.key == "Input16_v1" then 16 else if d.key == "Input1_v1" then 1 else 32)
  | .error _ => false

/-- A file of one template instance on a unit of no ports. -/
private def bare : Flock.Circuit := { (default : Flock.Circuit) with upv := 1, leavesIn := #[#[]], leavesOut := #[#[]] }

private def refuses (why : String) : Option String → Bool := Option.any (·.startsWith why)

/-- A primitive template of `Value` parameters `ps` (name, width) returning `Value ret`. -/
private def toyDef (key : String) (ps : List (String × Nat)) (ret : Nat) : Flock.Program.Defn :=
  { key, id := key, prim := true, params := (ps.map fun (p, w) => (p, Flock.Program.Ty.value w)).toArray,
    ret := .value ret, gates := 1, paramLeaves := ps.length, outLeaves := 1 }

/-- Templates on 16-bit parameters `x` and `y` returning 16 bits or 32, on three returning 16, and on a 32-bit `x` and a
16-bit `y`. -/
private def toyP : Flock.Program.Prog :=
  { defs := #[toyDef "TPair_v1" [("x", 16), ("y", 16)] 16, toyDef "TWide_v1" [("x", 16), ("y", 16)] 32,
      toyDef "TTri_v1" [("x", 16), ("y", 16), ("z", 16)] 16, toyDef "TBig_v1" [("x", 32), ("y", 16)] 16]
    index := {}, root := 0, programFn := "", sha512 := .empty }

#guard bitsOf (.array 3 (.value 1)) == 3 && wordsOf (.array 3 (.value 1)) == 1 &&
  wordsOf (.tuple #[.value 16, .value 17]) == 3 && retWords (.tuple #[.value 16, .value 17]) == 3 &&
  retWords (.tuple #[.value 1, .value 1]) == 2 &&
  retWords (.record #[("a", .value 8), ("b", .array 2 (.value 8))]) == 2 &&
  paramOf "x#3" == "x" && paramOf "x" == "x"

#guard refuses "the circuit file's template" (check bare toyP "TOther_v1") &&
  refuses "the circuit file's unit holds no" (check default toyP "TPair_v1") &&
  refuses "the circuit file returns" (check { bare with outNet := 1 } toyP "TPair_v1") &&
  refuses "the circuit file's leaf maps" (check { bare with leavesOut := #[#[0]] } toyP "TPair_v1") &&
  refuses "the circuit file's leaf maps" (check { bare with leavesIn := #[] } toyP "TPair_v1") &&
  refuses "the circuit file's input rows" (check bare toyP "TPair_v1")

#guard wiresOf [] 0 == none

/-- A unit net of one input word holding two 16-bit ports, and 16 computed rows, row `i` the XOR of the input columns
`r i` times the constant, returned from column 128 as one 16-bit leaf. -/
private def netOf (r : Nat → Array Nat) : Flock.Net :=
  let a := Array.replicate Flock.WORD #[] ++ (Array.range 16).map r
  let b := Array.replicate Flock.WORD #[] ++ Array.replicate 16 #[Flock.WORD + 16]
  match Flock.Sparse.ofRows a (Flock.WORD + 17), Flock.Sparse.ofRows b (Flock.WORD + 17) with
  | .ok a, .ok b =>
    { (default : Flock.Net) with
      constPos := Flock.WORD + 16, inWords := 1, aT := .pure a, bT := .pure b,
      inGroups := #[⟨0, 1, #[16, 16]⟩], outGroups := #[⟨1, 1, #[16]⟩] }
  | _, _ => default

/-- `netOf` with row `i` reading both ports' bit `i`. -/
private def toyNet : Flock.Net := netOf fun i => #[i, 16 + i]

private def toyPort (name : String) : Flock.Port := { name, words := 1, chunks := 1, nb := 1, v2 := true }

/-- Member A: its unit reads rows `x` and `y` and writes the output row `o`. -/
private def toyA : Flock.Circuit :=
  { (default : Flock.Circuit) with
    nets := #[("unit", toyNet)], upv := 1, ports := #[toyPort "x", toyPort "y", toyPort "o"], outRowPorts := #[1],
    outPorts := #[("o", 1)], leavesIn := #[#[0, 1]], leavesOut := #[#[0]] }

/-- Member B: its unit reads A's output row `o` and A's input row `x`, and returns nothing. -/
private def toyB : Flock.Circuit :=
  { (default : Flock.Circuit) with
    nets := #[("unit", toyNet)], upv := 1, ports := #[toyPort "o", toyPort "x"], leavesIn := #[#[0, 1]] }

private def toyMs : Members := [(toyA, #[0, 1]), (toyB, #[0, 1])]

#guard (keyProg toyA 2).inputs == 258 && (keyProg toyA 2).perUnit == 17 && base toyMs 1 == 360
#guard inBit toyA 2 1 0 1 3 == some 149 && inBit toyA 2 0 0 2 0 == none && inBit toyA 2 2 0 0 0 == none
#guard rowBit toyA 2 149 == some (RowPos.mk "y" 1 3) && rowBit toyA 2 40 == none && rowBit toyA 2 1 == none
#guard outBit toyA 2 0 0 0 0 == some 263 && outBit toyA 2 1 0 0 5 == some 329 && outBit toyB 2 0 0 0 0 == none
#guard outRowBit toyA 1 0 0 4 == some (RowPos.mk "o" 1 4)
#guard let gs := (prog (keyProg toyA 2) (consts toyA 2)).1
  (writers [(toyA, #[0, 1])]).all fun (x, _) => match gs[x]? with | some (.and _) => true | _ => false
#guard consts toyA 2 0 == some true && consts toyA 2 1 == some false && consts toyA 2 149 == none &&
  consts toyA 2 40 == some false && consts toyA 2 258 == none
#guard let kp := keyProg toyA 2; (List.range (kp.size + 2)).all fun g => constsAt (owners kp) kp g == consts toyA 2 g

/-- Member A on a unit whose rows read only row `x`'s port. -/
private def toyX : Flock.Circuit := { toyA with nets := #[("unit", netOf fun i => #[i])] }

/-- Member B reading row `y` in place of row `x`. -/
private def toyY : Flock.Circuit := { toyB with ports := #[toyPort "o", toyPort "y"] }

#guard rowBit toyX 2 21 == some (RowPos.mk "y" 0 3) && consts toyX 2 21 == some false && consts toyX 2 5 == none &&
  (sources [(toyX, #[0, 1])]).all (·.1 != 21) && (sources [(toyX, #[0, 1])]).any (·.1 == 5)
#guard wiresOf [(toyA, #[0, 1]), (toyY, #[0, 1])] (360 + 2 + 19) == some 21 &&
  wiresOf [(toyX, #[0, 1]), (toyY, #[0, 1])] (360 + 2 + 19) == none
#guard lastLe #[0, 1, 2] 0 == 0 && lastLe #[0, 1, 2] 5 == 2 && lastLe #[0, 4] 3 == 0
#guard srcAt toyA (Index.of toyA) 0 1 3 == .msg 1 0 3 && srcAt toyA (Index.of toyA) 0 1 9 == .msg 1 1 1
#guard srcKey toyA (Index.of toyA) 19 == .msg 1 0 3 && srcKey toyA (Index.of toyA) 40 == .pad 0 40
#guard keyIdx toyA 128 19 == some 19 && keyIdx toyA 128 40 == some 40 &&
  keyIdx { toyA with leavesIn := #[#[1, 1]] } 128 19 == some 3

/-- `toyNet` with a second input group of one cut port, so its constant past two input words. -/
private def cutNet : Flock.Net :=
  { toyNet with
    inWords := 2, constPos := 2 * Flock.WORD + 16, inGroups := #[⟨0, 1, #[16, 16]⟩, ⟨1, 1, #[16]⟩]
    outGroups := #[⟨2, 1, #[16]⟩] }

/-- Member A with `cutNet`'s unit and a row `z`, its cut port 2 the row word 2 (row `z`'s). -/
private def toyCut : Flock.Circuit :=
  { toyA with
    nets := #[("unit", cutNet)], ports := #[toyPort "x", toyPort "y", toyPort "z", toyPort "o"]
    leafCuts := #[(0, 2, 2)] }

#guard srcAt toyCut (Index.of toyCut) 0 2 3 == .msg 2 0 3 && srcAt toyCut (Index.of toyCut) 0 2 16 == .zero
#guard srcAt { toyCut with leafCuts := #[] } (Index.of { toyCut with leafCuts := #[] }) 0 2 3 == .lit 0
#guard wiresOf toyMs (360 + 2 + 128 + 5) == some 329
#guard wiresOf toyMs (360 + 2 + 16 + 5) == some 7
#guard wiresOf toyMs 7 == none && wiresOf toyMs 263 == none
#guard wiresOf [(toyA, #[0, 1]), (toyB, #[2, 3])] (360 + 2 + 128 + 5) == none
#guard wiresOf [(toyA, #[0, 1]), (toyB, #[1, 3])] (360 + 2 + 5) == some 329
/-- Member A at two template instances a VU: row `x` as the rows `x#0` and `x#1`, a word each, and rows `y` and `o` two
words each. -/
private def toyA2 : Flock.Circuit :=
  { toyA with
    upv := 2, outPorts := #[("o", 2)], leavesIn := #[#[0, 2], #[1, 3]], leavesOut := #[#[0], #[1]]
    ports := #[toyPort "x#0", toyPort "x#1", { toyPort "y" with words := 2, bits := 32 },
      { toyPort "o" with words := 2, bits := 32 }] }

#guard fieldBit toyA2 [("x", 1), ("y", 2)] "y" 0 == some (0, 1, 0) &&
  fieldBit toyA2 [("x", 1), ("y", 2)] "y" 16 == some (0, 2, 0) &&
  fieldBit toyA2 [("x", 1), ("y", 2)] "y" 37 == some (1, 1, 5) &&
  fieldBit toyA2 [("x", 1), ("y", 2)] "x" 17 == some (1, 0, 1) &&
  fieldBit toyA2 [("x", 1), ("y", 2)] "y" 64 == none && fieldBit toyA2 [("x", 1), ("y", 2)] "z" 0 == none

#guard check toyA toyP "TPair_v1" == none && paramsWhy toyA2 toyP.defs[0]! == none &&
  refuses "the circuit file has 2 template instances a VU, not one" (check toyA2 toyP "TPair_v1") &&
  check toyCut toyP "TTri_v1" == none &&
  refuses "a port of the circuit file's unit"
    (check { toyA with nets := #[("unit", { toyNet with inGroups := #[⟨0, 1, #[16, 8]⟩] })] } toyP "TPair_v1") &&
  refuses "the circuit file's leaf maps" (check { toyA with leavesIn := #[#[]] } toyP "TPair_v1") &&
  refuses "the circuit file's unit returns an output word twice"
    (check { toyA2 with leavesOut := #[#[0], #[0]] } toyP "TPair_v1") &&
  refuses "a returned output bit"
    (check { toyA with nets := #[("unit", { toyNet with outGroups := #[⟨0, 1, #[16]⟩] })] } toyP "TPair_v1") &&
  refuses "input port 2 of the circuit file's unit, template instance 0, is not wired exactly once"
    (check { toyCut with leafCuts := #[] } toyP "TTri_v1") &&
  refuses "input port 2 of the circuit file's unit, template instance 0"
    (check { toyCut with leafCuts := #[(0, 2, 2), (0, 2, 1)] } toyP "TTri_v1") &&
  [(0, 0, 1), (0, 0, 2)].all (fun d =>
    refuses "a wire into the circuit file's unit" (check { toyCut with wires := #[⟨d, (0, 0, 0)⟩] } toyP "TTri_v1")) &&
  check { toyCut with wires := #[⟨(1, 0, 0), (0, 0, 0)⟩] } toyP "TTri_v1" == none &&
  [(0, 1, 0), (1, 2, 0), (0, 3, 0), (0, 2, 3)].all (fun e =>
    refuses "a leaf cut outside the unit's cut inputs or the input rows"
      (check { toyCut with leafCuts := #[e] } toyP "TTri_v1")) &&
  refuses "a leaf cut outside the unit's cut inputs or the input rows"
    (check { toyCut with nets := #[("unit", { cutNet with inGroups := #[⟨0, 1, #[16, 16]⟩, ⟨1, 1, #[8]⟩] })] }
      toyP "TTri_v1") &&
  refuses "a leaf map names a word past the circuit file's input rows"
    (check { toyA with leavesIn := #[#[0, 2]] } toyP "TPair_v1")

#guard refuses "two row ports of the circuit file have one name"
    (check { toyA with ports := #[toyPort "x", toyPort "x", toyPort "o"] } toyP "TPair_v1") &&
  refuses "the circuit file's input rows are not its template's parameters in order, each 1 times"
    (check { toyA with ports := #[toyPort "y", toyPort "x", toyPort "o"] } toyP "TPair_v1") &&
  refuses "the circuit file's input rows" (check toyA toyP "TBig_v1") &&
  refuses "the circuit file's input rows"
    (check { toyA2 with upv := 1, leavesIn := #[#[0, 2]], leavesOut := #[#[0]] } toyP "TPair_v1") &&
  refuses "the circuit file's output rows are not 1 times" (check toyA toyP "TWide_v1") &&
  refuses "a template instance's output leaves"
    (check { toyA with ports := #[toyPort "x", toyPort "y", { toyPort "o" with words := 2, bits := 32 }] }
      toyP "TWide_v1") &&
  refuses "the circuit file's unit's input ports"
    (check { toyA with ports := #[{ toyPort "x" with words := 2, bits := 32 }, toyPort "y", toyPort "o"] }
      toyP "TBig_v1") &&
  refuses "template instance 0's input port 0 reads row word 1, not word 0 of its parameter x's rows"
    (check { toyA with leavesIn := #[#[1, 0]] } toyP "TPair_v1") &&
  refuses "template instance 0's input port 0 reads row word -1"
    (check { toyA with leavesIn := #[#[-1, 1]] } toyP "TPair_v1") &&
  refuses "template instance 1's input port 1 reads row word 2, not word 1 of its parameter y's rows"
    (check { toyA2 with leavesIn := #[#[0, 2], #[1, 2]] } toyP "TPair_v1") &&
  refuses "template instance 0's input port 2 reads row word 0"
    (check { toyCut with leafCuts := #[(0, 2, 0)] } toyP "TTri_v1") &&
  refuses "bit 8 of parameter x of template instance 0 is no message bit of its rows"
    (check { toyA with ports := #[{ toyPort "x" with bits := 8 }, toyPort "y", toyPort "o"] } toyP "TPair_v1")

/-- `toyA` with a row reading input column 40, which neither port holds. -/
private def toyPad : Flock.Circuit := { toyA with nets := #[("unit", netOf fun i => #[i, 40])] }

#guard unitReads toyA == (Array.range 128).map (fun x => decide (x < 32)) &&
  readsWhy toyA toyP.defs[0]! (unitReads toyA) ["x", "y"] == none &&
  readsWhy toyX toyP.defs[0]! (unitReads toyX) ["x"] == none &&
  refuses "no row of the circuit file's unit reads bit 0 of its parameter y"
    (readsWhy toyX toyP.defs[0]! (unitReads toyX) ["y"]) &&
  refuses "a row of the circuit file's unit reads its input column 40"
    (readsWhy toyPad toyP.defs[0]! (unitReads toyPad) []) &&
  refuses "the circuit file's unit's read columns" (readsWhy toyA toyP.defs[0]! #[] [])

end Spot.Bits
