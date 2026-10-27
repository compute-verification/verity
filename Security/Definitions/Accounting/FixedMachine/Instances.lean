import Definitions.Accounting.FixedMachine

/-!
# Fixed machine instances: an honest machine and a hard-coding circuit

The instances `Accounting.Guarantees.FixedMachine.BlockDescription`'s witnesses are at, each two one-bit gates
(`twoGates f`): gate `0` an input gate and gate `1` computing `f` of it.

* **The honest machine** `H₀ = twoGates not`: one state bit, gate `0`, and its negation, gate `1`, the next state
  (`S₀`). Run `T₀ = 2` steps it is `C₀`: gate `0` the initial state, gates `1` and `2` the two steps. `R₀` registers
  `C₀` with no registered gate and one proof block, its two operation gates; `M₀` outputs a one-bit program as it is,
  and the empty evaluator extracts the final state, gate `2` (`y₀`). `A₀` places `C₀` on one leaf, gate `g` at tick
  `g`, its initial information the initial state, `D₀ = 1` bit.
* **The hard-coding circuit** `Hard = twoGates (fun _ => 1)`, placed on one leaf: gate `0` a registered input gate and
  gate `1` the constant `1`, the string `[true]` (`hardString`) built into the circuit rather than computed by a fixed
  machine. It reads nothing outside its location, and its initial budget is `0` bits.
-/

namespace Accounting.FixedMachine

open Verity Finset

/-- **Two one-bit gates**: gate `0` an input gate, and gate `1` computing `f` of it, the output gate. -/
def twoGates (f : BitVec 1 → BitVec 1) : Circuit where
  N := 2
  w _ := 1
  isInput g := decide ((g : ℕ) = 0)
  reads g := if (g : ℕ) = 1 then {0} else ∅
  reads_lt := by decide
  input_reads := by decide
  op g v := if (g : ℕ) = 1 then f (v 0) else 0
  op_congr g v v' hv := by
    by_cases h : (g : ℕ) = 1
    · have h0 : (0 : Fin 2) ∈ (if (g : ℕ) = 1 then ({0} : Finset (Fin 2)) else ∅) := by simp [h]
      simp only [h, ite_true]
      rw [hv 0 h0]
    · simp only [h, ite_false]
  outputs := {1}

/-- The honest machine's step: one state bit and its negation, the next state. -/
def H₀ : Circuit := twoGates fun b => ~~~b

/-- `H₀`'s state wire: gate `0`, written by gate `1`. -/
def S₀ : H₀.StateWires where
  k := 1
  k_le := by decide
  input_lt g hg := decide_eq_true (by omega)
  out _ := ⟨1, by decide⟩
  le_out _ := le_refl 1
  w_out _ := rfl

/-- The honest machine's number of steps. -/
def T₀ : ℕ := 2

/-- The honest machine run `T₀` steps: gate `0` the initial state, gates `1` and `2` the two steps. -/
abbrev C₀ : Circuit := H₀.unroll S₀ T₀

/-- The honest registration: `C₀` with no registered gate and one proof block, its operation gates. -/
def R₀ : Registration where
  d := { D := C₀, θ := ∅, θ_input := by simp }
  val _ := 0
  B := 1
  P := { unit := fun g => if C₀.isInput g then none else some 0,
         input_iff := fun g => by cases C₀.isInput g <;> simp }

/-- The honest decoder: it outputs a one-bit program as it is, and nothing else. -/
def M₀ : Decoder Unit where
  run _ p _ := if p.length = 1 then some p else none
  mono _ _ _ _ _ _ h := h
  prefixFree _ _ p p' _ _ h h' hp := by
    have h1 : p.length = 1 := by
      by_contra hc
      simp [hc] at h
    have h2 : p'.length = 1 := by
      by_contra hc
      simp [hc] at h'
    exact hp.eq_of_length (h1.trans h2.symm)

/-- The honest extraction: the final state, gate `2`'s bit. -/
def y₀ : Fin R₀.B → R₀.d.D.Transcript → List Bool := fun _ X => [(X ⟨2, by decide⟩).getLsbD 0]

/-- The honest actual circuit: `C₀` on one leaf, gate `g` at tick `g`. -/
def A₀ : Placed Unit where
  toCircuit := C₀
  x _ := ()
  t g := g
  cell _ := 0
  cell_inj _ _ hgh := Fin.ext (congrArg (fun p => p.2.2) hgh)

/-- `A₀` is `C₀` with its gates in place. -/
def φ₀ : Circuit.Iso R₀.d.D A₀.toCircuit := Circuit.Iso.refl C₀

/-- The honest proof block's location: the one leaf. -/
def X₀ : Fin R₀.B → Finset Unit := fun _ => univ

/-- The honest proof block's initial information: the initial state, gate `0`. -/
def a₀ : Fin R₀.B → Finset (Fin A₀.N) := fun _ => {⟨0, by decide⟩}

/-- The honest proof block's initial budget: one bit. -/
def D₀ : Fin R₀.B → ℕ := fun _ => 1

/-- The hard-coding circuit: gate `0` an input gate and gate `1` the constant `1`, on one leaf, gate `g` at tick `g`. -/
def Hard : Placed Unit where
  toCircuit := twoGates fun _ => 1
  x _ := ()
  t g := g
  cell _ := 0
  cell_inj _ _ hgh := Fin.ext (congrArg (fun p => p.2.2) hgh)

/-- The string the hard-coding circuit outputs, gate `1`'s bit on any inputs: `[true]`. -/
def hardString : List Bool := [(Hard.eval fun _ => 0) ⟨1, by decide⟩ |>.getLsbD 0]

end Accounting.FixedMachine
