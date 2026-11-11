module

public import Definitions.Accounting.Gates.Model
public import Definitions.Core.Circuit

/-!
# The circuit model as a core circuit (trusted)

A circuit of the model (`IntDag`) is a core circuit (`Verity.Circuit`, `IntDag.toCircuit`): gate `i` is gate `i`, of
its declared bits (`IntGate.bits`), and holds its value as its offset from its declared `lo` (`IntGate.enc`), which
reads back (`IntGate.dec`) while the value lies in its declared interval. An input gate is an input. Every other gate
reads the gates its `args` name below it and applies its operation to their values (`IntOp.compute`), an operand at or
past it reading `0` as `values` reads one (`IntDag.readBack`). The committed gates are the outputs. An input `x` is the
words `IntDag.inputWords x` at the input gates.

On an input in its window, the core circuit's evaluation on its words reads back as the model's values, gate by gate
(`Accounting.SecurityProofs.Gates.toCircuit_eval`).
-/

@[expose] public section

namespace Accounting.Gates

variable {ι : Type}

/-- An input. -/
def IntOp.isInput : IntOp ι → Bool
  | .input .. => true
  | _ => false

/-- A computed gate's value on operands `v`: its exact value, then an s8-family step's clamp or wrap (`IntOp.out`). It
reads no input slot (an input's is `0` here: its word is given, never computed). -/
def IntOp.compute (o : IntOp ι) (v : List ℤ) : ℤ := o.out (o.raw (fun _ => 0) v)

namespace IntGate

variable (g : IntGate ι)

/-- A value's word: its offset from the declared `lo`, in the gate's declared bits. -/
def enc (z : ℤ) : BitVec g.bits := BitVec.ofNat g.bits (z - g.lo).toNat

/-- A word's value: the declared `lo` plus the word. -/
def dec (b : BitVec g.bits) : ℤ := g.lo + b.toNat

end IntGate

namespace IntDag

variable (C : IntDag ι)

/-- Operand `j` of gate `g`, from words `v` for the gates: gate `j`'s word read back, or `0` at or past `g`, as `values`
reads an operand. -/
def readBack (v : (a : Fin C.length) → BitVec C[a].bits) (g : Fin C.length) (j : ℕ) : ℤ :=
  if h : j < g.val then (C[j]'(h.trans g.isLt)).dec (v ⟨j, h.trans g.isLt⟩) else 0

/-- The model's circuit as a core circuit: gate `g` of `C[g].bits` bits, an input when `C[g]` is one, and otherwise
reading the gates its `args` name below it. -/
@[reducible] def toCircuit : Verity.Circuit where
  N := C.length
  w g := C[g].bits
  isInput g := C[g].op.isInput
  reads g := Finset.univ.filter fun a => C[g].op.isInput = false ∧ a < g ∧ a.val ∈ C[g].args
  reads_lt g a ha := (Finset.mem_filter.1 ha).2.2.1
  input_reads g h := by
    have h' : C[g.val].op.isInput = true := h
    ext a
    simp [h']
  op g v := C[g].enc (C[g].op.compute (C[g].args.map (C.readBack v g)))
  op_congr g v v' h := by
    show C[g].enc _ = C[g].enc _
    cases hi : C[g].op.isInput
    · have : C[g].args.map (C.readBack v g) = C[g].args.map (C.readBack v' g) := by
        apply List.map_congr_left
        intro j hj
        unfold readBack
        by_cases hlt : j < g.val
        · simp only [hlt, ↓reduceDIte]
          rw [h ⟨j, hlt.trans g.isLt⟩ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hi, hlt, hj⟩)]
        · simp only [hlt, ↓reduceDIte]
      rw [this]
    · have : ∀ o : IntOp ι, o.isInput = true → ∀ u u' : List ℤ, o.compute u = o.compute u' := by
        intro o ho u u'
        cases o <;> simp_all [IntOp.isInput, IntOp.compute, IntOp.raw]
      exact congrArg C[g].enc (this _ hi _ _)
  outputs := Finset.univ.filter fun g => C[g].commit = true

/-- Input `x`'s words: each gate's operation on `x` with no operands, which at an input gate is its slot's value
(`eval` reads no other gate's). -/
def inputWords (x : ι → ℤ) : C.toCircuit.Trace := fun (g : Fin C.length) => C[g].enc (C[g].op.raw x [])

end IntDag

end Accounting.Gates
