import Flock.Circuit
import Flock.HmRow
import Flock.Program

/-!
# A sampled statement's bits

The one layout of a sampled statement's gates, as code: the Boolean-profile program a registered C-Flock circuit file
describes, where each of its gates sits, and each unit's gates and inputs. The sampled leg's bridge,
`Properties.NetworkScan.Flattens` and NetworkCheck read these definitions and hold no second copy of them.

* `keyProg c n`: the circuit's program on Δ's source keys, for `n` units. Its input gates are the constant (gate 0), the
  zero (gate 1) and `B` source gates per unit, unit `u`'s input column `x` on its key's (`keyIdx`), or on the zero's where
  it has none. Then come the units, each the unit net's rows (`Rows.ofNet`) stacked `upv` times (`Rows.stack`), a row
  reading earlier gates. Its outputs are the gates on the returned outputs' bits.
* `gateSize`, `off`, `posOf`, `lower`, `prog`: that program lowered to the Boolean profile (`LGate`), gate by gate in
  order. A row `(⊕ a) ∧ (⊕ b)` is three gates, the two XORs then their AND, and an input is one, `.const b` where the
  constants `κ` fix it to `b`. Gate `g` sits at `posOf`, its last lowered gate.
* `unitOf`, `unitGates`, `owner`, `insOf`, `blockOf`: unit `u`'s lowered gates, the input gates it owns (the unfixed
  inputs whose first reader it is) and its proof block, the first then the second's positions.
* `inBit`, `outBit`: the claims position of a bit of an instance's input port or returned output leaf, the instance
  `(u, k)` being unit `u`'s template instance `k < upv`.
* `RowPos`, `rowBit`, `outRowBit`: a commitment position (a registered row, its leaf and a bit), where a source gate
  reads and where an output bit is committed.
* `Members`, `base`, `wiresOf`: an audit's members, each a statement on a circuit file over some of the partition's VUs,
  side by side, and each source bit's wire to the first claims position at its commitment position, compared at the
  partition's VUs: the writer's output bit, else the first copy.
* `width P g`: the width of the leaf a Program's layout gate `g` produces.
* `check c P tmpl`: why Bits doesn't lay `c` out as template `tmpl` of `P`, or none.

The circuit file holds each unit's Boolean circuit as a whole, not each primitive's: no row is a primitive's own, so a
packed primitive's bits are placed through its unit's ports, never through its Definition.
-/

namespace SampledProofs.Bits

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

/-- Each row word's first message leaf (`HmRow.delta`'s `starts`). -/
def starts (c : Flock.Circuit) : Array Nat :=
  (c.ports.foldl (fun (acc : Array Nat × Nat) p => (acc.1.push acc.2, acc.2 + p.words)) (#[], 0)).1

/-- A net's input port bits, port by port: each port's columns. -/
def portBits (gs : Array Flock.Group) : List Nat :=
  (Flock.portCols gs).toList.flatMap fun q => List.range' q.1 q.2

/-- The port a column of the input rows is a bit of: the first whose columns hold it. -/
def portOf (pc : Array (Nat × Nat)) (off : Nat) : Nat :=
  ((List.range pc.size).find? fun p => decide (pc[p]!.1 ≤ off ∧ off < pc[p]!.1 + pc[p]!.2)).getD 0

/-- **The column Δ copies into a unit input**, without the template instance: a forced-zero row, message bit `bit` of
byte `byte` of row port `p`, output column `col` of net `net`'s slot `slot`, the fixed column `col`, or input padding row
`off` of copy `k`. -/
inductive Src where
  | zero
  | msg (p byte bit : Nat)
  | wire (net slot col : Nat)
  | lit (col : Nat)
  | pad (k off : Nat)
  deriving DecidableEq, Inhabited, Repr

/-- Copy `k`'s input port `p`, bit `t`: its source, without the template instance. -/
def srcAt (c : Flock.Circuit) (k p t : Nat) : Src :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    if (c.leavesIn.getD k #[])[p]! < 0 then .zero else
      let x := ((c.leavesIn.getD k #[])[p]!).toNat
      let q := ((List.range (starts c).size).reverse.find? (fun q => (starts c)[q]! ≤ x)).getD 0
      if 16 * (x - (starts c)[q]!) + t < c.ports[q]!.bits then
        .msg q (2 * (x - (starts c)[q]!) + t / 8) (t % 8)
      else .zero
  else
    match c.leafCuts.toList.filter (fun e => decide (e.1 = k ∧ e.2.1 = p)) with
    | e :: _ =>
      let q := ((List.range (starts c).size).reverse.find? (fun q => (starts c)[q]! ≤ e.2.2)).getD 0
      if t < 16 && 16 * (e.2.2 - (starts c)[q]!) + t < c.ports[q]!.bits then
        .msg q (2 * (e.2.2 - (starts c)[q]!) + t / 8) (t % 8)
      else .zero
    | [] => match c.wires.toList.filter (fun w => decide (w.dst = (c.unitNet, k, p))) with
      | w :: _ => .wire w.src.1 w.src.2.1
          (((Flock.portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).1 + t)
      | [] => .lit 0

/-- **Stacked input `x`'s key**: copy `x / W`'s input at slot offset `x % W` (`W = inWords · WORD`), `srcAt` at the port
bit it is, or the padding row it is. -/
def srcKey (c : Flock.Circuit) (x : Nat) : Src :=
  let W := c.unit.inWords * Flock.WORD
  let pc := Flock.portCols c.unit.inGroups
  if x % W ∈ portBits c.unit.inGroups then
    srcAt c (x / W) (portOf pc (x % W)) (x % W - pc[portOf pc (x % W)]!.1)
  else .pad (x / W) (x % W)

/-- **Equal sources are equal keys**: of `n` inputs, input `x`'s key is `none` at the zero, else the first input with the
same source. -/
def keyIdx (c : Flock.Circuit) (n x : Nat) : Option Nat :=
  if srcKey c x = .zero then none else (List.range n).find? fun i => srcKey c i = srcKey c x

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
leaf of that row; a forced zero, a wire, a fixed column and padding are no commitment position. -/
def srcPos (c : Flock.Circuit) (u : Nat) : Src → Option RowPos
  | .msg p byte bit => some { row := (c.ports.getD p default).name, leaf := u, bit := 8 * byte + bit }
  | _ => none

/-- **Source gate `g`'s commitment position** (the row map), for `n` units: gate `2 + u·B + b`, where `b` is its own key,
is unit `u`'s read of `srcKey c b`. None for the constant, the zero, a row, an unread source gate, and a key no row holds.
An input gate's claims position is the gate itself. -/
def rowBit (c : Flock.Circuit) (n g : Nat) : Option RowPos :=
  let B := ((Rows.ofNet c.unit).stack c.upv).nIn
  if g < 2 || B == 0 || 2 + n * B ≤ g || keyIdx c B ((g - 2) % B) != some ((g - 2) % B) then none
  else srcPos c ((g - 2) / B) (srcKey c ((g - 2) % B))

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

/-- **Every source bit**: each read source gate's claims position and commitment position at the partition's VUs,
member by member. -/
def sources (ms : Members) : List (Nat × RowPos) :=
  (List.range ms.length).flatMap fun m =>
    let ci := ms.getD m default
    let b := base ms m
    (List.range (keyProg ci.1 ci.2.size).inputs).filterMap fun g =>
      (rowBit ci.1 ci.2.size g).map fun r => (b + g, r.globalize ci.2)

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

/-- **The width of the leaf a Program's layout gate `g` produces**: its primitive's `Value<w>`, and 0 where the layout has
no gate `g`. -/
def width (P : Flock.Program.Prog) (g : Nat) : Nat :=
  match P.gate g with
  | .ok (p, _, _) => match (P.defs[p]!).ret with
    | .value w => w
    | _ => 0
  | .error _ => 0

/-! ## The file against its Program -/

/-- **Why Bits doesn't lay out the circuit file `c` as template `tmpl` of the Program `P`**, or none: the template is
no Definition of `P`, no template instance makes a unit, the unit net's constant sits among its input words (`Rows.ofNet`
and `srcKey` split the inputs at different columns), a net other than the unit returns the outputs (`keyProg`'s
outputs are the unit net's), or a returned output bit sits on no computed row, so that every writer is a copy row and
none an input or a constant. -/
def check (c : Flock.Circuit) (P : Flock.Program.Prog) (tmpl : String) : Option String :=
  let R0 := Rows.ofNet c.unit
  if !P.defs.any (·.id == tmpl) then some s!"the circuit file's template {tmpl} is no Definition of the Program"
  else if c.upv == 0 then some "the circuit file's unit holds no template instance"
  else if R0.nIn != c.unit.inWords * Flock.WORD then
    some "the circuit file's unit net has its constant among its input words"
  else if c.outNet != c.unitNet then some "the circuit file returns its outputs from a net other than its unit"
  else if (List.range c.upv).any fun k => (List.range (c.leavesOut.getD k #[]).size).any fun j =>
      (List.range 16).any fun t => !(decide (R0.nIn ≤ outCol c j t) && decide (outCol c j t < R0.nIn + R0.nComp)) then
    some "a returned output bit of the circuit file's unit sits on no computed row (an input or the constant)"
  else none

/-! ## Checks -/

/-- Two units of two rows on the constant, the zero and two source gates: unit 0 reads source 2, unit 1 source 3, and
each unit's second row returns. -/
private def toy : KeyProg :=
  { inputs := 4, units := 2, perUnit := 2
    rows := #[([2], [0]), ([4, 2], [0]), ([3], [0]), ([6], [3])]
    outs := [5, 7] }

private def toyκ (g : Nat) : Option Bool := if g == 0 then some true else if g == 1 then some false else none

#guard (List.range toy.size).map (posOf toy) == [0, 1, 2, 3, 6, 9, 12, 15]
#guard (prog toy toyκ).1.length == (List.range toy.size).foldl (fun s g => s + gateSize toy g) 0
#guard lower toy toyκ 0 == [.const true]
#guard lower toy toyκ 2 == [.input]
#guard lower toy toyκ 5 == [.xor [6, 2], .xor [0], .and [7, 8]]
#guard (prog toy toyκ).2 == [9, 15]
#guard let gs := (prog toy toyκ).1; (List.range gs.length).all fun i => (gs[i]!).reads.all (· < i)
#guard blockOf toy toyκ 0 == [4, 5, 6, 7, 8, 9, 2]
#guard blockOf toy toyκ 1 == [10, 11, 12, 13, 14, 15, 3]
#guard owner toy 0 == some 0 && owner toy 1 == none && owner toy 3 == some 1

/-- Two 16-bit inputs and three 1-bit inputs into one 32-bit primitive: leaves of widths 16, 16, 1, 1, 1 and 32. -/
private def mixed : String :=
  "{\"annotations\":{},\"callees\":[\"Input16_v1\",\"Input1_v1\",\"TMix_v1\"],\"definitions\":{\"Input16_v1\":{\"form\":\"primitive\",\"id\":\"Input16_v1\",\"params\":[],\"ret\":[\"Value\",16]},\"Input1_v1\":{\"form\":\"primitive\",\"id\":\"Input1_v1\",\"params\":[],\"ret\":[\"Value\",1]},\"Root[TMix_v1]_v1\":{\"bindings\":{},\"body\":{\"nodes\":[{\"args\":[],\"axes\":[],\"fn\":0,\"form\":\"batch\",\"n\":2,\"out\":0},{\"args\":[],\"axes\":[],\"fn\":1,\"form\":\"batch\",\"n\":3,\"out\":1},{\"args\":[{\"refs\":{\"affine\":[\"n\",0,0,2]},\"type\":0},{\"refs\":{\"affine\":[\"n\",1,0,3]},\"type\":1}],\"fn\":2,\"form\":\"primitive\",\"out\":2}],\"params\":[],\"ret\":{\"affine\":[\"n\",2,0,1]},\"ret_type\":[\"Value\",32]},\"form\":\"composite\",\"id\":\"Root[TMix_v1]_v1\"},\"TMix_v1\":{\"form\":\"primitive\",\"id\":\"TMix_v1\",\"params\":[[\"a\",[\"Array\",2,[\"Value\",16]]],[\"b\",[\"Array\",3,[\"Value\",1]]]],\"ret\":[\"Value\",32]}},\"program_fn\":\"TMix_v1\",\"root\":\"Root[TMix_v1]_v1\",\"schema\":\"verity-ir/descriptor/v1\",\"types\":[[\"Array\",2,[\"Value\",16]],[\"Array\",3,[\"Value\",1]],[\"Value\",32]]}"

#guard match Flock.Program.decode mixed.toUTF8 with
  | .ok P => (List.range 7).map (width P) == [16, 16, 1, 1, 1, 32, 0]
  | .error _ => false

#guard match Flock.Program.decode mixed.toUTF8 with
  | .ok P => check { (default : Flock.Circuit) with upv := 1 } P "TMix_v1" == none &&
      (check { (default : Flock.Circuit) with upv := 1 } P "TOther_v1").isSome &&
      (check default P "TMix_v1").isSome &&
      (check { (default : Flock.Circuit) with upv := 1, outNet := 1 } P "TMix_v1").isSome &&
      (check { (default : Flock.Circuit) with upv := 1, leavesOut := #[#[0]] } P "TMix_v1").isSome
  | .error _ => false

#guard wiresOf [] 0 == none

/-- A unit net of one input word holding two 16-bit ports, and 16 computed rows, row `i` the XOR of the ports' bit `i`
times the constant, returned from column 128 as one 16-bit leaf. -/
private def toyNet : Flock.Net :=
  let a := Array.replicate Flock.WORD #[] ++ (Array.range 16).map fun i => #[i, 16 + i]
  let b := Array.replicate Flock.WORD #[] ++ Array.replicate 16 #[Flock.WORD + 16]
  match Flock.Sparse.ofRows a (Flock.WORD + 17), Flock.Sparse.ofRows b (Flock.WORD + 17) with
  | .ok a, .ok b =>
    { (default : Flock.Net) with
      constPos := Flock.WORD + 16, inWords := 1, aT := .pure a, bT := .pure b,
      inGroups := #[⟨0, 1, #[16, 16]⟩], outGroups := #[⟨1, 1, #[16]⟩] }
  | _, _ => default

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
#guard let gs := (prog (keyProg toyA 2) toyκ).1
  (writers [(toyA, #[0, 1])]).all fun (x, _) => match gs[x]? with | some (.and _) => true | _ => false
#guard wiresOf toyMs (360 + 2 + 128 + 5) == some 329
#guard wiresOf toyMs (360 + 2 + 16 + 5) == some 7
#guard wiresOf toyMs 7 == none && wiresOf toyMs 263 == none
#guard wiresOf [(toyA, #[0, 1]), (toyB, #[2, 3])] (360 + 2 + 128 + 5) == none
#guard wiresOf [(toyA, #[0, 1]), (toyB, #[1, 3])] (360 + 2 + 5) == some 329
#guard match Flock.Program.decode mixed.toUTF8 with
  | .ok P => check toyA P "TMix_v1" == none
  | .error _ => false

end SampledProofs.Bits
