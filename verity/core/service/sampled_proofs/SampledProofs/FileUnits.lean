/-
A circuit file's units in the claims' program (`Flock.Bridge.progOf` of `Placed.keyProg`), summarized once: what
`Slots.layout` expands into proof blocks. It stands in for `FilePart.ofCircuit`, whose summary it is field for field
(`n`, `B`, `m`, `srcs`), until that lands; it reads what `keyProg` reads:

* **Inputs.** Gate 0 is the constant, gate 1 the zero, then `B = upv · W` sources a unit (`W` the unit's input columns,
  `min (inWords · WORD) constPos`): unit `u`'s source `b` is gate `2 + u·B + b`. An input gate lowers to one position, so
  gate `g` is position `g`.
* **Rows.** Unit `u`'s computed gates follow every input gate, `m = upv · (constPos − W + 1)` of them (the stack of `upv`
  copies, each with its constant row: `Rows.stack`), each a row, lowered to three positions: positions
  `[inGates + 3·m·u, inGates + 3·m·(u+1))`.
* **Owned sources.** Copy `k`'s input column `x` reads the source of its key (`srcKey`, which mirrors
  `Aliased.srcKey`): the first stacked input with that key (`keyIdx`), or the zero where Δ copies the zero. Only unit
  `u` reads its sources, so it owns the ones a computed row reads (`RecResiduals.readIns`): `srcs`, the same for every
  unit. The constant and the zero are constants, owned by none; a source no row reads is in no proof block.
-/
import Std.Data.HashMap
import Std.Data.HashSet
import Flock.KnownRead
import Flock.ProvedScope
import Flock.RecResiduals

namespace SampledProofs

open Flock (Circuit WORD)

namespace FileUnits

/-- The column Δ copies into a unit input, without the template instance (`Aliased.Src`): a forced zero, message bit
`bit` of byte `byte` of row port `p`, output column `col` of net `net`'s slot `slot`, the fixed column `col`, or input
padding row `off` of copy `k`. -/
inductive Src
  | zero
  | msg (p byte bit : Nat)
  | wire (net slot col : Nat)
  | lit (col : Nat)
  | pad (k off : Nat)
  deriving BEq, Hashable, Inhabited, Repr

/-- The row port whose words hold message word `x`, among the ports' first words `st`: the last that starts at or
before it. -/
def portAt (st : Array Nat) (x : Nat) : Nat :=
  ((List.range st.size).reverse.find? (fun q => st.getD q 0 ≤ x)).getD 0

/-- Copy `k`'s input port `p`, bit `t` (`Aliased.srcAt`), the ports' first words `st` (`KnownRead.starts`). -/
def srcAt (c : Circuit) (st : Array Nat) (k p t : Nat) : Src :=
  if p < (c.unit.inGroups.getD 0 default).bits.size then
    let l := (c.leavesIn.getD k #[]).getD p 0
    if l < 0 then .zero else
      let x := l.toNat
      let q := portAt st x
      if 16 * (x - st.getD q 0) + t < (c.ports.getD q default).bits then
        .msg q (2 * (x - st.getD q 0) + t / 8) (t % 8)
      else .zero
  else
    match c.leafCuts.find? (fun e => e.1 == k && e.2.1 == p) with
    | some e =>
      let q := portAt st e.2.2
      if t < 16 && 16 * (e.2.2 - st.getD q 0) + t < (c.ports.getD q default).bits then
        .msg q (2 * (e.2.2 - st.getD q 0) + t / 8) (t % 8)
      else .zero
    | none => match c.wires.find? (fun w => w.dst == (c.unitNet, k, p)) with
      | some w => .wire w.src.1 w.src.2.1
          (((Flock.portCols (c.nets.getD w.src.1 default).2.outGroups).getD w.src.2.2 default).1 + t)
      | none => .lit 0

/-- **Stacked input `x`'s key** (`Aliased.srcKey`): copy `x / W`'s input at offset `x % W`, `srcAt` at the port bit it
is (`mask`, the unit's port bits; `pc`, its `portCols`), or the padding row it is. -/
def srcKey (c : Circuit) (st : Array Nat) (mask : Array Bool) (pc : Array (Nat × Nat)) (W x : Nat) : Src :=
  if mask.getD (x % W) false then
    let p := Flock.KnownRead.portOf pc (x % W)
    srcAt c st (x / W) p (x % W - (pc.getD p default).1)
  else .pad (x / W) (x % W)

end FileUnits

/-- A circuit file's key program at `n` units: each unit's `B` sources and `m` computed rows, and the sources a unit
owns, ascending (`FilePart`'s fields). -/
structure FileUnits where
  n : Nat
  B : Nat
  m : Nat
  srcs : Array Nat
  deriving BEq, Inhabited, Repr

namespace FileUnits

/-- The input gates: the constant, the zero, and `B` sources a unit. -/
def inGates (f : FileUnits) : Nat := 2 + f.n * f.B

/-- The lowered program's positions: the input gates, then three per computed row. -/
def size (f : FileUnits) : Nat := f.inGates + 3 * f.n * f.m

/-- **Unit `u`'s proof block**, its positions as intervals `[lo, hi)`: its rows' lowered gates, then each source it
owns. -/
def block (f : FileUnits) (u : Nat) : Array (Nat × Nat) :=
  (if f.m = 0 then #[] else #[(f.inGates + 3 * f.m * u, f.inGates + 3 * f.m * (u + 1))]) ++
    f.srcs.map fun b => (2 + u * f.B + b, 2 + u * f.B + b + 1)

/-- **The circuit `c`'s key program at `n` units**, summarized as the module's header reads it. -/
def ofCircuit (c : Circuit) (n : Nat) : FileUnits :=
  let g := c.unit
  let W := g.inWords * WORD
  let nIn := min W g.constPos
  let B := c.upv * nIn
  let st := Flock.KnownRead.starts c
  let mask := Flock.ProvedScope.portMask g.inGroups W
  let pc := Flock.portCols g.inGroups
  let keys := (Array.range B).map (srcKey c st mask pc W)
  let first : Std.HashMap Src Nat := keys.zipIdx.foldl (fun m (k, i) => if m.contains k then m else m.insert k i) {}
  let rd := Flock.RecResiduals.readIns g
  let owned := (List.range B).foldl (fun s x =>
      match rd.getD (x % nIn) false, keys[x]? with
      | true, some key => if key == .zero then s else s.insert (first.getD key 0)
      | _, _ => s) ({} : Std.HashSet Nat)
  { n, B, m := c.upv * (g.constPos - nIn + 1), srcs := owned.toArray.qsort (· < ·) }

/-! Two copies of a unit of one input word whose leaf port `x` is message word 0's low bit and padding past it, whose
computed rows read column 0 and the constant: copy 1's column 0 is the same message bit as copy 0's, so both read source
0, and its padding is its own source, which no row reads. -/

private def testNet : Flock.Net :=
  let self := (Array.range WORD).map fun i => #[i]
  let ra := self ++ #[#[0], #[WORD], #[WORD + 1]]
  match Flock.Sparse.ofRows ra (2 ^ 8), Flock.Sparse.ofRows ra (2 ^ 8) with
  | .ok a, .ok b =>
    { name := "unit", useful := WORD + 3, constPos := WORD + 2, inGroups := #[⟨0, 1, #[1]⟩], outGroups := #[⟨1, 1, #[1]⟩],
      inWords := 1, aT := .pure a, bT := .pure b, unitLog := 8 }
  | _, _ => default

private def testCircuit : Circuit :=
  { name := "t", sha := .empty, nets := #[("unit", testNet)], ranges := #[], ports := #[{ name := "x", words := 1, chunks := 1, nb := 0 }],
    outPorts := #[], g := 1, kLog := 0, upv := 2, unitNet := 0, leavesIn := #[#[0], #[0]], leavesOut := #[],
    key := .empty, dummyDigests := #[], dummyOutputs := #[], leafScheme := "", perVu := #[2], wires := #[] }

#guard ofCircuit testCircuit 3 == { n := 3, B := 2 * WORD, m := 2 * 3, srcs := #[0] }
#guard (let f := ofCircuit testCircuit 3; (f.inGates, f.size, f.block 1)) ==
  (2 + 3 * 256, 2 + 3 * 256 + 3 * 3 * 6, #[(2 + 3 * 256 + 18, 2 + 3 * 256 + 36), (2 + 256, 2 + 256 + 1)])

end FileUnits

end SampledProofs
