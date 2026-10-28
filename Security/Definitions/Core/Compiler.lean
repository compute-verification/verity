import Mathlib.Data.Fin.Embedding
import Mathlib.Data.Fin.Tuple.Basic
import Definitions.Core.Partition

/-!
# The registered compiler and its execution blocks

Setup fixes the public parameters, among them the advice cap `ℓa`, and the registered compiler `B` (registration's
builder entry names it by digest). In execution phase `j` the hardware runs and the developer commits every value it
computes: the committed trace. At the start of verification phase `j`, before any of its coins, the developer reports
its advice `a_j`, at most `ℓa` bits that Verity never interprets, committed, and `B` runs on it. Its output is that
phase's **execution block** `E_j`, the gates it appends to the registered circuit so far, partitioned into inference
blocks, each partitioned into proof blocks, with each committed gate's position in the committed trace. The registered
circuit after `n` phases is `B.init ‖ E_1 ‖ … ‖ E_n`, and its partitions are the blocks' side by side (`Compiler.run`).
`B.init` is `B`'s output at setup, before any advice: the gates of the registered information `θ`, whose values were
committed beforehand.

* **An execution block over `C`** (`Circuit.Extension`): `n` gates after `C`'s, gate `i` of the block being gate
  `C.N + i` of the appended circuit, each reading earlier gates of `C` or of the block (the KV cache a later phase reads,
  say). `C.append E` (`Circuit.append`) is the circuit with the block appended; `C`'s gates keep their indices.
* **What the compiler outputs** (`Compiler.Output`): a circuit file, its registered input gates `θ`, its partition `𝓘`
  into inference blocks, which the inference-only check reads, its partition `P` into proof blocks, which sampled proofs
  prove, nested in `𝓘` (`Partition.Refines`), and each gate's position in the committed trace (`pos`).
* **One phase's output** (`Compiler.Phase o`): the execution block over `o`'s circuit, its proof blocks, each proof
  block's inference block, and its gates' positions, numbered after `o`'s (`Compiler.Output.append`).
* **The compiler** (`Compiler`): its advice cap, its output at setup, and its output on each phase's advice over the
  circuit so far.

The compiler's output is the one description of the registered circuit: `Verity.Registered` reads `D`, `θ`, `𝓘` and
`P` from it, and nothing restates them. That the output explains the committed trace is `Verity.Committed.Explains`.
-/

namespace Verity

open Finset

/-- **An execution block over `C`**: `n` gates after `C`'s, gate `i` holding one value of `w i` bits and reading gates of
`C` or earlier gates of the block, gate `i` being gate `C.N + i` of `C.append E`. -/
structure Circuit.Extension (C : Circuit) where
  n : ℕ
  w : Fin n → ℕ
  isInput : Fin n → Bool
  reads : Fin n → Finset (Fin (C.N + n))
  reads_lt : ∀ i, ∀ a ∈ reads i, (a : ℕ) < C.N + i
  input_reads : ∀ i, isInput i = true → reads i = ∅
  op : (i : Fin n) → ((a : Fin (C.N + n)) → BitVec (Fin.append C.w w a)) → BitVec (w i)
  op_congr : ∀ i (v v' : (a : Fin (C.N + n)) → BitVec (Fin.append C.w w a)), (∀ a ∈ reads i, v a = v' a) →
    op i v = op i v'
  outputs : Finset (Fin (C.N + n))

namespace Circuit

variable (C : Circuit) (E : C.Extension)

/-- The block of no gates. -/
def Extension.empty : C.Extension where
  n := 0
  w := Fin.elim0
  isInput := Fin.elim0
  reads := Fin.elim0
  reads_lt i := i.elim0
  input_reads i := i.elim0
  op i := i.elim0
  op_congr i := i.elim0
  outputs := ∅

/-- The gates each gate of `C.append E` reads: `C`'s reads for `C`'s gates, the block's for its own. -/
def appendReads : Fin (C.N + E.n) → Finset (Fin (C.N + E.n)) :=
  Fin.append (fun g => (C.reads g).map (Fin.castAddEmb E.n)) E.reads

/-- What each gate of `C.append E` computes: `C`'s gates their operations on the values of `C`'s gates, the block's
gates theirs. -/
def appendOp (g : Fin (C.N + E.n)) (v : (a : Fin (C.N + E.n)) → BitVec (Fin.append C.w E.w a)) :
    BitVec (Fin.append C.w E.w g) :=
  Fin.addCases (motive := fun g => BitVec (Fin.append C.w E.w g))
    (fun g => (C.op g fun a => (v (Fin.castAdd E.n a)).cast (Fin.append_left C.w E.w a)).cast
      (Fin.append_left C.w E.w g).symm)
    (fun i => (E.op i v).cast (Fin.append_right C.w E.w i).symm) g

/-- **`C ‖ E`**: the circuit `C` with the execution block `E` appended after its gates. -/
def append : Circuit where
  N := C.N + E.n
  w := Fin.append C.w E.w
  isInput := Fin.append C.isInput E.isInput
  reads := C.appendReads E
  reads_lt g := by
    refine Fin.addCases (motive := fun g => ∀ a ∈ C.appendReads E g, a < g) (fun g a ha => ?_) (fun i a ha => ?_) g
    · simp only [appendReads, Fin.append_left, Finset.mem_map] at ha
      obtain ⟨b, hb, rfl⟩ := ha
      exact Fin.lt_def.2 (Fin.lt_def.1 (C.reads_lt g b hb))
    · simp only [appendReads, Fin.append_right] at ha
      exact Fin.lt_def.2 (E.reads_lt i a ha)
  input_reads g := by
    refine Fin.addCases
      (motive := fun g => Fin.append C.isInput E.isInput g = true → C.appendReads E g = ∅) (fun g h => ?_)
      (fun i h => ?_) g
    · simp only [Fin.append_left] at h
      simp [appendReads, C.input_reads g h]
    · simp only [Fin.append_right] at h
      simp [appendReads, E.input_reads i h]
  op := C.appendOp E
  op_congr g v v' := by
    refine Fin.addCases
      (motive := fun g => (∀ a ∈ C.appendReads E g, v a = v' a) → C.appendOp E g v = C.appendOp E g v')
      (fun g h => ?_) (fun i h => ?_) g
    · simp only [appendOp, Fin.addCases_left]
      refine congrArg (BitVec.cast _) (C.op_congr g _ _ fun a ha => ?_)
      rw [h (Fin.castAdd E.n a) (by simpa [appendReads] using Finset.mem_map_of_mem (Fin.castAddEmb E.n) ha)]
    · simp only [appendOp, Fin.addCases_right]
      exact congrArg (BitVec.cast _) (E.op_congr i v v' fun a ha => h a (by simpa [appendReads] using ha))
  outputs := C.outputs.map (Fin.castAddEmb E.n) ∪ E.outputs

end Circuit

/-- **What the registered compiler outputs**: a circuit file `D`, its registered input gates `θ` (the gates of the
registered information, whose values were committed beforehand), its partition `𝓘` into `n` inference blocks, each
partitioned into proof blocks (`P`, `m` of them, refining `𝓘`), and each gate's position in the committed trace
(`pos`, `none` for a gate whose value isn't committed there). -/
structure Compiler.Output where
  D : Circuit
  θ : Finset (Fin D.N)
  θ_input : ∀ g ∈ θ, D.isInput g = true
  n : ℕ
  𝓘 : Partition D n
  m : ℕ
  P : Partition D m
  refines : P.Refines 𝓘
  pos : Fin D.N → Option ℕ

/-- **One execution phase's output** over the compiler's output `o` so far: the execution block `E` over `o`'s circuit,
its partition into `m` proof blocks, which hold the block's computed gates, each proof block's inference block among
`n` (`parent`), and each gate's position in the committed trace. -/
structure Compiler.Phase (o : Compiler.Output) where
  E : o.D.Extension
  n : ℕ
  m : ℕ
  unit : Fin E.n → Option (Fin m)
  input_iff : ∀ i, unit i = none ↔ E.isInput i = true
  parent : Fin m → Fin n
  pos : Fin E.n → Option ℕ

namespace Compiler

/-- The phase that appends nothing. -/
def Phase.empty (o : Output) : Phase o where
  E := .empty o.D
  n := 0
  m := 0
  unit := Fin.elim0
  input_iff i := i.elim0
  parent := Fin.elim0
  pos := Fin.elim0

/-- **`o ‖ b`**: the compiler's output `o` with the phase `b` appended: the circuit `o.D ‖ b.E`, the same registered
gates, the inference blocks and proof blocks of `o` followed by `b`'s, and the positions of `o`'s gates followed by
`b`'s. -/
def Output.append (o : Output) (b : Phase o) : Output where
  D := o.D.append b.E
  θ := o.θ.map (Fin.castAddEmb b.E.n)
  θ_input g hg := by
    obtain ⟨g, hθ, rfl⟩ := Finset.mem_map.1 hg
    show Fin.append o.D.isInput b.E.isInput (Fin.castAdd b.E.n g) = true
    rw [Fin.append_left]
    exact o.θ_input g hθ
  n := o.n + b.n
  𝓘 :=
    { unit := Fin.append (fun g => (o.𝓘.unit g).map (Fin.castAdd b.n))
        fun i => ((b.unit i).map b.parent).map (Fin.natAdd o.n)
      input_iff := fun g => by
        refine Fin.addCases (motive := fun g =>
          Fin.append (fun g => (o.𝓘.unit g).map (Fin.castAdd b.n))
              (fun i => ((b.unit i).map b.parent).map (Fin.natAdd o.n)) g =
            none ↔ (o.D.append b.E).isInput g = true) (fun g => ?_) (fun i => ?_) g
        · simp [Circuit.append, o.𝓘.input_iff g]
        · simp [Circuit.append, b.input_iff i] }
  m := o.m + b.m
  P :=
    { unit := Fin.append (fun g => (o.P.unit g).map (Fin.castAdd b.m)) fun i => (b.unit i).map (Fin.natAdd o.m)
      input_iff := fun g => by
        refine Fin.addCases (motive := fun g =>
          Fin.append (fun g => (o.P.unit g).map (Fin.castAdd b.m)) (fun i => (b.unit i).map (Fin.natAdd o.m)) g =
            none ↔ (o.D.append b.E).isInput g = true) (fun g => ?_) (fun i => ?_) g
        · simp [Circuit.append, o.P.input_iff g]
        · simp [Circuit.append, b.input_iff i] }
  refines :=
    { parent := Fin.append (fun u => Fin.castAdd b.n (o.refines.parent u)) fun u => Fin.natAdd o.n (b.parent u)
      sub := fun g => by
        refine Fin.addCases (motive := fun g =>
          Fin.append (fun g => (o.𝓘.unit g).map (Fin.castAdd b.n))
              (fun i => ((b.unit i).map b.parent).map (Fin.natAdd o.n)) g =
            (Fin.append (fun g => (o.P.unit g).map (Fin.castAdd b.m))
              (fun i => (b.unit i).map (Fin.natAdd o.m)) g).map
              (Fin.append (fun u => Fin.castAdd b.n (o.refines.parent u)) fun u => Fin.natAdd o.n (b.parent u)))
          (fun g => ?_) (fun i => ?_) g
        · simp [o.refines.sub g, Option.map_map, Function.comp_def]
        · simp [Option.map_map, Function.comp_def] }
  pos := Fin.append o.pos b.pos

end Compiler

/-- **The registered compiler** `B`, fixed at setup: the most bits of advice it reads in one phase (`ℓa`, a public
parameter), its output `init` before any advice, the gates of `θ`, and on each execution phase's advice the phase it
appends to its output so far. -/
structure Compiler where
  ℓa : ℕ
  init : Compiler.Output
  step : (o : Compiler.Output) → List Bool → Compiler.Phase o

namespace Compiler

/-- **The compiler's output on the advice `a_1, …, a_n`**: `init ‖ B(a_1) ‖ … ‖ B(a_n)`, each phase compiled over the
output before it. -/
@[reducible] def runFrom (B : Compiler) : List (List Bool) → Output → Output
  | [], o => o
  | x :: a, o => B.runFrom a (o.append (B.step o x))

/-- The compiler's output on the advice list, from its output at setup. -/
@[reducible] def run (B : Compiler) (a : List (List Bool)) : Output := B.runFrom a B.init

/-- **The one-block compiler of `o`**: `o` at setup, no advice, and nothing appended in any phase. -/
@[reducible] def const (o : Output) : Compiler := ⟨0, o, fun o _ => .empty o⟩

end Compiler

/-- The partition of a circuit's computed gates into one block each. -/
def Partition.singletons (C : Circuit) : Partition C C.N where
  unit g := if C.isInput g then none else some g
  input_iff g := by cases C.isInput g <;> simp

end Verity
