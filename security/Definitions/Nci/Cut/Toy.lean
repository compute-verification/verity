module

public import Definitions.Nci.Cut

/-!
# A toy for the min-cut statements

Five one-bit gates: three unregistered inputs, `w` (gate 0), `x₀` and `x₁` (gates 1 and 2), and two computed gates,
`y₀ = w x₀` (gate 3) and `y₁ = w x₁` (gate 4), with nothing registered (`θ = ∅`). One matmul `y = w x` with
`i = l = 0` and `j ∈ {0, 1}` (`M`): `w_00 = {w}`, `x_0j = {x_j}`, and `y_0j`'s one wire is its MAC `g_0j0 = {y_j}`. The
work `W` is that matmul alone: `N = 2` MACs and `O = 2` output wires, and `|C| = 2` computed gates. Each word carries
its own value (`val`), and the arrays are `W`'s `w`-words and its `x`-words (`arrays`), which the matmul reads as the
edge `(0, 1)` (`edge`). `P1` puts both computed gates in one block, which reads the three inputs; `P2` gives each its
own, which reads two.
-/

@[expose] public section

namespace Nci.Cut.Toy

open Verity Finset

/-- The gates each gate reads. -/
def reads : Fin 5 → Finset (Fin 5) := ![∅, ∅, ∅, {0, 1}, {0, 2}]

/-- The toy circuit, one bit a gate. -/
abbrev C : Circuit where
  N := 5
  w := fun _ => 1
  isInput := fun g => decide (g.val < 3)
  reads := reads
  reads_lt := by decide
  input_reads := by decide
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := {3, 4}

/-- The toy, nothing registered. -/
abbrev d : RegisteredCircuit := .ofCircuit C ∅ (by decide)

/-- `x_0j`'s gate, by `j`. -/
def xGate : Fin 2 → Fin 5 := ![1, 2]

/-- `y_0j`'s gate, which is also the MAC `g_0j0`. -/
def yGate : Fin 2 → Fin 5 := ![3, 4]

/-- The matmul `y = w x`: `w_00 = {0}`, `x_0j = {1 + j}`, `y_0j = {3 + j}`, and the MAC `g_0j0 = {3 + j}`. -/
def M : Matmul C where
  ni := 1
  nj := 2
  nl := 1
  w := fun _ => {0}
  x := fun p => {xGate p.2}
  y := fun p => {yGate p.2}
  w_inj := fun a b _ => Subsingleton.elim a b
  x_inj := by decide
  mac := fun t => {yGate t.2.1}
  mac_computed := by decide
  mac_connected := fun _ => ⟨singleton_nonempty _, fun _ hT hne hT' =>
    absurd ((subset_singleton_iff.1 hT).resolve_left hne.ne_empty) hT'⟩
  reads_w := by decide
  reads_x := by decide
  feeds_y := fun _ => ⟨_, mem_singleton_self _, _, mem_singleton_self _, .refl _⟩

/-- The charged work: `M` alone. -/
def W : Work C Unit where
  M := fun _ => M
  mac_disjoint := by decide

/-- The values: each word carries its own. -/
abbrev val : Finset (Fin C.N) → Finset (Fin C.N) := id

/-- The arrays: `W`'s `w`-words and its `x`-words. -/
abbrev arrays : Fin 2 → Finset (Finset (Fin C.N)) := ![W.allW, W.allX]

/-- The one matmul reads the two arrays, the edge `(0, 1)` of the complete graph on them. -/
abbrev edge : Unit → Fin 2 × Fin 2 := fun _ => (0, 1)

/-- One block holding both computed gates. -/
def P1 : Partition C 1 where
  unit := fun g => if g.val < 3 then none else some 0
  input_iff := by decide

/-- A block for each computed gate. -/
def P2 : Partition C 2 where
  unit := fun g => if g.val < 3 then none else if g.val = 3 then some 0 else some 1
  input_iff := by decide

end Nci.Cut.Toy
