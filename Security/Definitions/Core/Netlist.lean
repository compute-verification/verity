import Mathlib.Algebra.Group.Nat.Defs

/-!
# A private circuit's netlist and the universal unit's size classes

A private circuit's gates, as the universal unit (`verity.ml.private_circuits.universal`) takes them: SSA wires, the
`nIn` inputs and then one per gate, each gate one of XOR, AND, XNOR or NAND of two wires (`Netlist`), run gate by gate
(`Netlist.eval`). A size class `(G, N_IN, N_OUT)` (`SizeClass`) fixes the netlists one universal unit runs
(`Netlist.Fits`), and `Netlist.args` is a netlist as that unit's arguments. The sampling claims' `Universal` states that
the universal unit of each class computes every netlist of it.

The import is no narrower than `Mathlib.Algebra.Group.Nat.Defs` because there `2 ^ k` on `ℕ` elaborates through
`Monoid.npow`, the term the claims' records hold for `Netlist.Fits`; with less, it elaborates through `Nat.pow`.
-/

namespace Verity

/-- A size class `(G, N_IN, N_OUT)`: `G` gates over `N_IN` inputs, and `N_OUT` output bits. -/
structure SizeClass where
  G : ℕ
  nIn : ℕ
  nOut : ℕ

/-- `K`: the bits of a wire index, enough for the `W = N_IN + G` wires, at least one. -/
def SizeClass.indexBits (c : SizeClass) : ℕ := Nat.log2 (c.nIn + c.G - 1) + 1

/-- **A private circuit's gates**, as the universal unit takes them (`verity.ml.private_circuits.universal`): SSA wires,
the `nIn` inputs and then one per gate. Gate `(op, a, b)` reads wires `a` and `b` and computes XOR, AND, XNOR or NAND
(`op`'s bit 0 picks AND, bit 1 negates); a wire at or past the reading gate's own reads `0`, as does an output past the
last wire. -/
structure Netlist where
  nIn : ℕ
  gates : List (Fin 4 × ℕ × ℕ)
  outs : List ℕ

/-- The netlist run gate by gate on `x` (`interpret`). -/
def Netlist.eval (n : Netlist) (x : List Bool) : List Bool :=
  let ws := n.gates.foldl (fun ws g =>
    let a := ws.getD g.2.1 false
    let b := ws.getD g.2.2 false
    ws ++ [xor (if g.1.val % 2 = 1 then a && b else xor a b) (decide (g.1.val / 2 = 1))]) x
  n.outs.map fun o => ws.getD o false

/-- The netlist is of class `c`: `G` gates over `N_IN` inputs, `N_OUT` outputs, every index in `K` bits. -/
def Netlist.Fits (c : SizeClass) (n : Netlist) : Prop :=
  n.nIn = c.nIn ∧ n.gates.length = c.G ∧ n.outs.length = c.nOut ∧
    (∀ g ∈ n.gates, g.2.1 < 2 ^ c.indexBits ∧ g.2.2 < 2 ^ c.indexBits) ∧ ∀ o ∈ n.outs, o < 2 ^ c.indexBits

/-- `v` in `k` bits, low bit first. -/
def bitsLE (k v : ℕ) : List Bool := (List.range k).map v.testBit

/-- The netlist as the universal unit's `(op, src, out)` arguments, in leaf order (`arguments`). -/
def Netlist.args (c : SizeClass) (n : Netlist) : List Bool :=
  n.gates.flatMap (fun g => bitsLE 2 g.1.val) ++
    n.gates.flatMap (fun g => bitsLE c.indexBits g.2.1 ++ bitsLE c.indexBits g.2.2) ++
    n.outs.flatMap (bitsLE c.indexBits)

end Verity
