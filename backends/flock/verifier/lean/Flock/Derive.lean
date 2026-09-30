/-
`derive` (Project doc `verified-lowering-design.md` §1.2): a circuit type's rows, computed by the verifier itself, by the
v2 rule that `ir_lower.layout` uses. So no statement needs to carry rows, and `compose_sound` (the soundness package)
covers every unit. `flock-rows` (`FlockRows.lean`) writes them as `flock-ir-unit/v2`, and `check` holds the Python layout
to it byte for byte.

The v2 rule, for a flat type (AND items only):
* the input port groups, each group's bits consecutive in an aligned power-of-two run of 128-bit words, its unused rows
  forced zero: an input bit's row is its self row `z_c · z_c`;
* one row `(form_a) · (form_b)` per AND item, in item order;
* the output port groups as copy rows `(form) · 1`, laid out as the inputs are;
* the constant last, `1 · 1`.

A wire's column is its input bit's place or its AND item's row. A form's entries are its wires' columns, then the
constant's column if its constant is 1. A forced-zero row has none. Placed calls, exported inputs and reads come later.
-/
import Flock.CircuitType

namespace Flock.Derive

open CircuitType

def WORD : Nat := 128

/-- The smallest power of two at least `n` (1 for `n ≤ 1`), as `fp.ceil_pow2`. -/
def ceilPow2 (n : Nat) : Nat := if n ≤ 1 then 1 else 2 ^ (Nat.log2 (n - 1) + 1)

/-- A port group, as the header lists it: its first word, its words, and its ports' bits. -/
structure Group where
  word : Nat
  words : Nat
  bits : List Nat
  deriving Repr, BEq, Inhabited

/-- The ports' bits, cut into groups at the port indices `split` (each where a group starts), as `ir_lower._groups`. -/
def groupsOf (bits : List Nat) (split : List Nat) : List (List Nat) :=
  let cuts := 0 :: split ++ [bits.length]
  (cuts.zip cuts.tail).filterMap fun (a, b) => if a < b then some ((bits.drop a).take (b - a)) else none

/-- The groups laid out from row `r`: each opens at the next multiple of its size, so it's aligned, and ends at its last
word. Also returns the row after the last. -/
def place : Nat → List (List Nat) → List Group × Nat
  | r, [] => ([], r)
  | r, g :: gs =>
    let size := WORD * ceilPow2 (max 1 ((g.sum + WORD - 1) / WORD))
    let start := (r + size - 1) / size * size
    let (rest, e) := place (start + size) gs
    (⟨start / WORD, size / WORD, g⟩ :: rest, e)

/-- The layout's geometry: the port groups, where the ANDs start, and the constant's column. -/
structure Geo where
  ins : List Group
  andBase : Nat
  nAnd : Nat
  outs : List Group
  const : Nat
  deriving Repr, Inhabited

def portBits (ps : Array (Nat × Nat)) : List Nat := ps.toList.map fun p => p.1 * p.2

def geo (t : CircuitType) (inSplit outSplit : List Nat) : Geo :=
  let (ins, a) := place 0 (groupsOf (portBits t.ins) inSplit)
  let (outs, c) := place (a + t.items.size) (groupsOf (portBits t.outs) outSplit)
  ⟨ins, a, t.items.size, outs, c⟩

/-- The column of port bit `i`, the groups laid out as `gs`. -/
def colOf : List Group → Nat → Option Nat
  | [], _ => none
  | g :: gs, i => if i < g.bits.sum then some (g.word * WORD + i) else colOf gs (i - g.bits.sum)

/-- The port bit at column `r`, the groups laid out as `gs`. -/
def bitAt : List Group → Nat → Option Nat
  | [], _ => none
  | g :: gs, r =>
    if g.word * WORD ≤ r ∧ r < g.word * WORD + g.bits.sum then some (r - g.word * WORD)
    else (bitAt gs r).map (· + g.bits.sum)

namespace Geo

variable (G : Geo)

/-- A wire's column: its input bit's place (the first `nIn` wires), or its AND item's row. -/
def col (nIn w : Nat) : Nat := if w < nIn then (colOf G.ins w).getD 0 else G.andBase + (w - nIn)

/-- A form's row entries: its wires' columns, then the constant's column if its constant is 1. -/
def formCols (nIn : Nat) (f : Form) : List Nat := f.ws.toList.map (G.col nIn) ++ if f.c = 1 then [G.const] else []

/-- Row `r` of the flat type `t`. -/
def rowAt (t : CircuitType) (r : Nat) : List Nat × List Nat :=
  if r = G.const then ([G.const], [G.const])
  else if r < G.andBase then
    match bitAt G.ins r with
    | some _ => ([r], [r])
    | none => ([], [])
  else if r < G.andBase + G.nAnd then
    match t.items[r - G.andBase]? with
    | some (.and a b) => (G.formCols t.inBits a, G.formCols t.inBits b)
    | _ => ([], [])
  else
    match bitAt G.outs r with
    | some o => (G.formCols t.inBits (t.outputs[o]?.getD default), [G.const])
    | none => ([], [])

end Geo

/-- What `derive` relies on: AND items only, each form over earlier wires (ascending, so distinct) with its constant 0
or 1, and one output form per output bit, over all the wires. -/
def flatOk (t : CircuitType) : Bool :=
  (List.range t.items.size).all (fun i => match t.items[i]? with
    | some (.and a b) => formOk (t.inBits + i) a && formOk (t.inBits + i) b
    | _ => false) &&
  t.outputs.size == t.outBits && t.outputs.all (formOk (t.inBits + t.items.size))

/-- A split: port indices where a group starts, ascending, none past the ports. -/
def splitOk (n : Nat) (split : List Nat) : Bool :=
  split.all (· ≤ n) && (List.range (split.length - 1)).all fun i => split[i]! ≤ split[i + 1]!

/-- The geometry, checked: each port bit has a column in its region that names it back, no other column of a region names
a bit, and the output region lies between the ANDs and the constant. `place` always lays groups out so; `derive` checks
it anyway, in time linear in the port regions, so its theorems read these facts off the check. -/
def geoOk (t : CircuitType) (G : Geo) : Bool :=
  decide (G.andBase + G.nAnd ≤ G.const) &&
  (List.range t.inBits).all (fun i => match colOf G.ins i with
    | some c => decide (c < G.andBase) && bitAt G.ins c == some i
    | none => false) &&
  (List.range G.andBase).all (fun r => match bitAt G.ins r with
    | some i => decide (i < t.inBits) && colOf G.ins i == some r
    | none => true) &&
  (List.range t.outBits).all (fun o => match colOf G.outs o with
    | some c => decide (G.andBase + G.nAnd ≤ c ∧ c < G.const) && bitAt G.outs c == some o
    | none => false) &&
  (List.range (G.const - (G.andBase + G.nAnd))).all (fun j => match bitAt G.outs (G.andBase + G.nAnd + j) with
    | some o => decide (o < t.outBits) && colOf G.outs o == some (G.andBase + G.nAnd + j)
    | none => true)

structure Derived where
  geo : Geo
  /-- The type's input and output bits. -/
  nIn : Nat
  nOut : Nat
  rows : List (List Nat × List Nat)

/-- Input bit `i`'s column. -/
def Derived.inCol (d : Derived) (i : Nat) : Nat := (colOf d.geo.ins i).getD 0

/-- Output bit `o`'s column. -/
def Derived.outCol (d : Derived) (o : Nat) : Nat := (colOf d.geo.outs o).getD 0

/-- **The columns in logical order:** the input bits, then the unit's segment: its constant first, one row per AND item
in item order, and the output copy rows in output order. The block's `one`, which the constant copies through Δ, comes
after them all. Each row reads only earlier columns here, or `one` (as settled with audit-lean's 1d plan and the constant
rollout). -/
def Derived.order (d : Derived) : List Nat :=
  (List.range d.nIn).map d.inCol ++ [d.geo.const] ++ (List.range d.geo.nAnd).map (d.geo.andBase + ·) ++
    (List.range d.nOut).map d.outCol

/-- **The rows in logical order**, each with its physical column: the dense rows at `order`'s columns. The audit's
composed rows (`Rows.compose`) are defined from this list, so the two can't drift. -/
def Derived.logical (d : Derived) : List (Nat × (List Nat × List Nat)) :=
  d.order.map fun c => (c, d.rows.getD c ([], []))

/-- **The rows of a flat type** by the v2 rule, with its port groups cut at `inSplit` and `outSplit`. -/
def derive (t : CircuitType) (inSplit outSplit : List Nat) : Except String Derived :=
  if !flatOk t then .error "derive: not a flat type of AND items over earlier wires"
  else if !(splitOk t.ins.size inSplit && splitOk t.outs.size outSplit) then .error "derive: a split out of order"
  else
    let G := geo t inSplit outSplit
    if !geoOk t G then .error "derive: the port groups do not lay out"
    else .ok ⟨G, t.inBits, t.outBits, (List.range (G.const + 1)).map (G.rowAt t)⟩

/-- The logical list as the prover's mirror writes it (`verity_flock.derive.logical_text`): per column in `order`,
`column |A| A.. |B| B..`, the constant's row as the block holds it (a copy of `one`, through Δ), then `one` itself. -/
def logicalText (d : Derived) : String :=
  let line := fun (c : String) (a b : List String) =>
    " ".intercalate ([c, toString a.length] ++ a ++ [toString b.length] ++ b)
  let rows := d.order.map fun c =>
    if c = d.geo.const then line (toString c) ["one"] ["one"]
    else
      let p := d.rows.getD c ([], [])
      line (toString c) (p.1.map toString) (p.2.map toString)
  "\n".intercalate (rows ++ [line "one" ["one"] ["one"]]) ++ "\n"

/-! ## `flock-ir-unit/v2` -/

def FORMAT : String := "flock-ir-unit/v2"

def rowLine (p : List Nat × List Nat) : String :=
  " ".intercalate (((p.1.length :: p.1) ++ (p.2.length :: p.2)).map toString)

/-- The header: `FORMAT name rows const`, then each side's `n_groups (word words n_ports bits..)..`. -/
def header (name : String) (d : Derived) : String :=
  let side := fun (gs : List Group) =>
    toString gs.length :: gs.flatMap fun g => ([g.word, g.words, g.bits.length] ++ g.bits).map toString
  " ".intercalate ([FORMAT, name, toString d.rows.length, toString d.geo.const] ++ side d.geo.ins ++ side d.geo.outs)

/-- The rows as `ir_lower.netlist` writes them, with the lowering's `LEAVES` and `CUT` lines (canonical JSON) passed
through. -/
def netlist (name : String) (d : Derived) (leaves cut : Option String) : String :=
  let lines := header name d :: d.rows.map rowLine ++
    (leaves.map (fun l => ["LEAVES " ++ l]) |>.getD []) ++ (cut.map (fun c => ["CUT " ++ c]) |>.getD [])
  "\n".intercalate lines ++ "\n"

end Flock.Derive
