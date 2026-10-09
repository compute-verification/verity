import Models.Nci.Assumptions

/-!
# NCI's two incompressibility assumptions are false at some declarations

`Nci.Assumptions.ActivationsIncompressible` and `PartialSumsIncompressible` are each a `Prop` about one declaration and
one chain it runs, and nothing in either restricts where a product's weight operand comes from. `Circuit` requires only
input gates to read nothing, so a computed gate may hold a constant; and `Chain` names the product gates
`W_ℓ[i, j] x_t^{(ℓ)}[j]` without saying that a weight entry is read. A declaration that holds its weights in constant
gates refutes both.

The declaration (`decl`, chain `chain`): one multiplication (`L = 1`) at one position (`N = 1`), with inner dimension
`d = 1` and output dimension `e = 2`, all five gates 4 bits wide, and empty baseline `θ`:

* gate 0, an input gate: the activation entry `x^{(0)}[0]`, novel;
* gates 1 and 2, computed and reading nothing: the weight entries `W_0[0, 0]` and `W_0[1, 0]`;
* gates 3 and 4: the products `W_0[i, 0] x^{(0)}[0]`. Gate 3 reads `{0, 1}` and gate 4 reads `{0, 2}`; each is also the
  complete output entry `y^{(0)}[i]`, its only product.

Take `b_w = b_a = 4`, the FP4 widths, and `Γ = 4`, `K = 4` (the guarantees' `K = αΓ` at `α = 1`). The IU `R` of all four
computed gates has size 4 and reads only gate 0, so `novelReads R = 4 ≤ K`. But `R` holds products of 2 weight entries
and of 1 input entry, and that input entry, at the first multiplication, is not linked to anything:

* `ActivationsIncompressible` charges `R` `b_w · 2 + b_a · 1 = 12 > 4` bits (`not_activationsIncompressible`);
* `PartialSumsIncompressible` at the division of `D` into the one IU `R` charges it at least `b_w · 2 = 8 > 4` bits
  whatever partial sums `s` it is charged, since `s ≥ 0` (`not_partialSumsIncompressible`).

`R` reads fewer bits than the weight entries it uses because it computes them itself from nothing. The assumptions'
module docstring excludes such chains in prose (the chain's weights and first inputs must depend on more than `Γ` work
of training since the commitment, unlike weights computed from a committed checkpoint, or from layers before the first
multiplication committed in `θ`), but neither `Prop` states the exclusion, so each is false here as stated.
-/

namespace Nci.Counterexample

open Verity Finset

/-- The circuit: an input `x` (gate 0), two constant weights (gates 1 and 2) and their products with `x` (gates 3 and
4), each 4 bits wide. The values are irrelevant to the assumptions, which read only the wires and widths. -/
def circuit : Circuit where
  N := 5
  w _ := 4
  isInput g := g == 0
  reads g := if g = 3 then {0, 1} else if g = 4 then {0, 2} else ∅
  reads_lt := by decide
  input_reads := by decide
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3, 4}

/-- The declaration: `circuit` with nothing committed, so the input `x` is novel. -/
def decl : Declaration where
  D := circuit
  θ := ∅
  θ_input := by simp

/-- One multiplication `y = W x` at one position, `W` of 2 rows and 1 column: product `(i, 0)` is gate `3 + i`, which
also holds the complete entry `y[i]`. With one multiplication there are no links. -/
def chain : Chain decl.D where
  L := 1
  N := 1
  d _ := 1
  e _ := 2
  prod _ _ i _ := ⟨3 + i.val, Nat.add_lt_add_left i.isLt 3⟩
  prod_inj := by decide +kernel
  prod_computed := by decide +kernel
  y _ _ i := ⟨3 + i.val, Nat.add_lt_add_left i.isLt 3⟩
  sums _ _ _ _ := .refl _
  π ℓ h := absurd h (by have := ℓ.isLt; omega)
  π_inj ℓ h := absurd h (by have := ℓ.isLt; omega)
  links ℓ h := absurd h (by have := ℓ.isLt; omega)

/-- The IU: every computed gate. -/
def R : Finset (Fin decl.D.N) := ({1, 2, 3, 4} : Finset (Fin 5))

theorem R_computed : R ⊆ decl.D.computed := by decide +kernel

theorem R_size : decl.D.size R = 4 := rfl

theorem novelReads_R : novelReads decl R = 4 := by decide +kernel

theorem weightsUsed_R : (chain.weightsUsed R).card = 2 := by decide +kernel

theorem inputsUsed_R : (chain.inputsUsed R).card = 1 := by decide +kernel

open Classical in
/-- **`ActivationsIncompressible` is false** at `decl` and `chain` with `Γ = 4`, `K = 4` and `b_w = b_a = 4`: the IU of
every computed gate reads 4 bits and is charged 12. -/
theorem not_activationsIncompressible : ¬ Assumptions.ActivationsIncompressible decl chain 4 4 4 4 := by
  intro h
  have hR := h R R_computed R_size.le (by rw [novelReads_R]; norm_num)
  rw [Finset.filter_true_of_mem fun x _ => ?_, weightsUsed_R, inputsUsed_R, novelReads_R] at hR
  · norm_num at hR
  obtain ⟨⟨ℓ, hℓ⟩, t, j⟩ := x
  obtain rfl : ℓ = 0 := by change ℓ < 1 at hℓ; omega
  exact id

/-- The division of `decl` into one IU, of every computed gate. -/
def P : Partition decl.D 1 where
  unit g := if decl.D.isInput g then none else some 0
  input_iff g := by cases decl.D.isInput g <;> simp

theorem P_gates (u : Fin 1) : P.gates u = R := by
  obtain rfl : u = 0 := Subsingleton.elim _ _
  decide +kernel

/-- **`PartialSumsIncompressible` is false** at `decl` and `chain` with `Γ = 4`, `K = 4` and `b_w = b_a = 4`: in the
division into the one IU of every computed gate, that IU reads 4 bits and is charged at least 8, whatever partial sums
are charged to it. -/
theorem not_partialSumsIncompressible : ¬ Assumptions.PartialSumsIncompressible decl chain 4 4 4 4 := by
  intro h
  obtain ⟨s, -, hs⟩ := h 1 P fun u => by
    rw [P_gates, R_size, novelReads_R]
    norm_num
  have h0 := hs 0
  rw [P_gates, weightsUsed_R, novelReads_R] at h0
  have : (0 : ℝ) ≤ s 0 := Nat.cast_nonneg _
  push_cast at h0
  linarith

end Nci.Counterexample
