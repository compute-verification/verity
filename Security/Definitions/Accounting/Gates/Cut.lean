import Definitions.Accounting.Gates.Model

/-!
# A cut of a circuit: the gates on its write side, with what crosses into them as inputs (trusted)

The min-cut bound (`Models/Accounting/Guarantees/Cut.lean`) prices the words a circuit `C` of the model computes by
the gates on one side `D` of a cut and the values that cross into `D`, not by all of `C`. A cut is a set `D` of `C`'s
gate positions, the side that holds every write; the other side is every other gate.

* **Its boundary** (`boundary`): the gates outside `D` that a gate in `D` reads. What `D` needs of the other side is
  exactly their values.
* **The cut circuit** (`cut`): `C` with each gate at its own position, so its operand indices stand unchanged. A gate
  in `D` is itself (its input slots `ι` carried into `ι ⊕ ℕ` as `inl`). A boundary gate `j` becomes the input in slot
  `inr j`, declared at the interval the interval analysis derives for `j` in `C` (`intervals`), and bound at run time
  (`run`) when `j`'s value depends on a run or coin input (`dynamic`), at registration when it doesn't. Every other gate
  becomes the constant `0`, which nothing in `D` reads. Its inputs are `cutInput`: `C`'s inputs in `inl`, and in
  `inr j` the value gate `j` takes in `C`.
* **What it charges.** A dynamic boundary gate is a run input of the cut circuit, charged its derived interval's bits
  (`Iv.bits`); `boundaryBits` sums them, the bits that cross the cut. A static one (the weights, and whatever is
  computed from them and the payload alone) is registered and charged nothing: it takes one value on every draw.
  `runBitsIn` is the run inputs' bits in `D`. The cut circuit's units (`budgeted`) are `D`'s units in `C`, but that a
  gather in `D` may pay index bits again that a gather outside `D` already paid.
-/

namespace Accounting.Gates

variable {ι κ : Type}

/-- An operation with its input slots renamed by `e`. -/
def Op.map (e : ι → κ) : Op ι → Op κ
  | .input b i => .input b (e i)
  | .const c => .const c
  | .affine c₀ cs => .affine c₀ cs
  | .mma a b k s => .mma a b k s
  | .gather low d => .gather low d
  | .budgeted cls f => .budgeted cls f

/-- A gate with its input slots renamed by `e`. -/
def Gate.map (e : ι → κ) (g : Gate ι) : Gate κ := { g with op := g.op.map e }

/-- **The boundary of `D`**: the gates of `C` outside `D` that a gate in `D` reads. -/
def boundary (C : Dag ι) (D : Finset ℕ) : Finset ℕ :=
  (Finset.range C.length).filter fun j => j ∉ D ∧ ∃ i ∈ D, (C[i]?.any fun g => decide (j ∈ g.args)) = true

/-- Gate `i` of the cut circuit: `g = C[i]` itself in `D`; for a boundary gate, the input in slot `inr i` at the
interval `C`'s analysis derives for `i`, bound at run time when `i` is dynamic and at registration when it isn't; and
the constant `0` elsewhere. -/
def cutGate (C : Dag ι) (D : Finset ℕ) (i : ℕ) (g : Gate ι) : Gate (ι ⊕ ℕ) :=
  if i ∈ D then g.map Sum.inl
  else if i ∈ boundary C D then
    ⟨.input (if (dynamic C).getD i false then .run else .registered) (.inr i), [], ((intervals C).getD i (0, 0)).1,
      ((intervals C).getD i (0, 0)).2, g.step, g.prim, .other, false⟩
  else ⟨.const 0, [], 0, 0, g.step, g.prim, .other, false⟩

/-- **The cut circuit** of `C` at `D`: `D`'s gates in place, its boundary as inputs, and constants elsewhere. -/
def cut (C : Dag ι) (D : Finset ℕ) : Dag (ι ⊕ ℕ) := C.mapIdx (cutGate C D)

/-- The cut circuit's inputs on `C`'s input `x`: `x` in `inl`, and gate `j`'s value in `C` in `inr j`. -/
def cutInput (C : Dag ι) (x : ι → ℤ) : ι ⊕ ℕ → ℤ
  | .inl i => x i
  | .inr j => val C x j

/-- An interval's bits: `⌈log₂⌉` of the number of values it holds, as `Gate.bits` reads a declared one. -/
def Iv.bits (a : Iv) : ℕ := Nat.clog 2 (a.2 + 1 - a.1).toNat

/-- **The bits that cross the cut**: each dynamic boundary gate's derived interval's bits. -/
def boundaryBits (C : Dag ι) (D : Finset ℕ) : ℕ :=
  ∑ j ∈ boundary C D, if (dynamic C).getD j false then ((intervals C).getD j (0, 0)).bits else 0

/-- The run inputs' declared bits in `D`. -/
def runBitsIn (C : Dag ι) (D : Finset ℕ) : ℕ :=
  ∑ i ∈ (Finset.range C.length).filter (· ∈ D), (C[i]?.map (Gate.inputBits .run)).getD 0

end Accounting.Gates
