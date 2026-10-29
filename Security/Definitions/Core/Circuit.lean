import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Data.Fintype.Card

/-!
# Circuits

A circuit has gates `0, …, N - 1` in topological order: a gate reads only earlier gates. Some gates are input gates
and some are output gates. Gate `g` holds one value of `w g` bits (`verity.ir`'s `Value<w>`). An input gate's value is
given; every other gate computes its value from the gates it reads, the wires into it. The Boolean profile, which
C-Flock proves, is `w = 1` (`Boolean.lean`).

For a set `S` of gates, in the notation of the NCI draft:
* `size S`: the number of gates in `S`. The size of a circuit is `size computed`, its number of non-input gates;
* `width S`: the total width of their values;
* `into S`: in(S), the gates outside `S` with a wire into `S`;
* `exits S`: out(S), the gates of `S` that a gate outside `S` reads, and the output gates in `S`.

A trace is one run's data, a value for every gate. `eval a` is the honest trace on the inputs `a`, and
`run fix` evaluates with the gates that `fix` names held at the values it gives (how a unit is recomputed,
`Partition.localEval`).

`D` implements `C` (`Implements C D`, the draft's `C ≼ D`) when every input gate of `C` is an input gate of `D` and, on
every assignment to `D`'s inputs, `D` produces the values of `C`'s output gates among its own outputs. `D` contains `C`
(`Contains C D`) when, besides, `D` computes every non-input gate of `C` from the same wires.
-/

namespace Verity

open Finset

/-- A circuit: gates `0, …, N - 1` in topological order, gate `g` holding one value of `w g` bits. -/
structure Circuit where
  N : ℕ
  /-- The width of each gate's value, in bits. -/
  w : Fin N → ℕ
  isInput : Fin N → Bool
  /-- The gates `g` reads: the wires into `g`. -/
  reads : Fin N → Finset (Fin N)
  reads_lt : ∀ g, ∀ a ∈ reads g, a < g
  input_reads : ∀ g, isInput g = true → reads g = ∅
  /-- What a computed gate does: its value, from the values of the gates it reads. -/
  op : (g : Fin N) → ((a : Fin N) → BitVec (w a)) → BitVec (w g)
  op_congr : ∀ g (v v' : (a : Fin N) → BitVec (w a)), (∀ a ∈ reads g, v a = v' a) → op g v = op g v'
  outputs : Finset (Fin N)

namespace Circuit

variable (C : Circuit)

/-- One run's data: a value for every gate. -/
abbrev Transcript := (g : Fin C.N) → BitVec (C.w g)

/-- The non-input gates. -/
def computed : Finset (Fin C.N) := univ.filter fun g => C.isInput g = false

/-- `size(S)`: the number of gates in `S`. -/
def size (S : Finset (Fin C.N)) : ℕ := S.card

/-- `width(S)`: the total width of the values of the gates in `S`, in bits. -/
def width (S : Finset (Fin C.N)) : ℕ := ∑ g ∈ S, C.w g

/-- `in(S)`: the gates outside `S` with a wire into `S`. -/
def into (S : Finset (Fin C.N)) : Finset (Fin C.N) :=
  univ.filter fun a => a ∉ S ∧ ∃ g ∈ S, a ∈ C.reads g

/-- `out(S)`: the gates of `S` that a gate outside `S` reads, and the output gates in `S`. -/
def exits (S : Finset (Fin C.N)) : Finset (Fin C.N) :=
  S.filter fun g => g ∈ C.outputs ∨ ∃ h, h ∉ S ∧ g ∈ C.reads h

/-- Evaluation with some gates fixed: a fixed gate takes its given value, and every other gate applies its operation
to the values of the gates it reads. -/
def run (fix : (g : Fin C.N) → Option (BitVec (C.w g))) (g : Fin C.N) : BitVec (C.w g) :=
  match fix g with
  | some b => b
  | none => C.op g fun x => if _h : x < g then run fix x else 0
termination_by g.val
decreasing_by all_goals assumption

/-- The circuit's evaluation on the inputs `a`: the honest trace. Only `a`'s values at input gates are read. -/
def eval (a : C.Transcript) : C.Transcript :=
  C.run fun g => if C.isInput g then some (a g) else none

/-- The committed inputs of `X` are the values `a` gives the input gates. -/
def InputsAgree (X a : C.Transcript) : Prop :=
  ∀ g, C.isInput g = true → X g = a g

/-- `g` feeds `t`: `g = t`, or `g` is read by a gate that feeds `t`. -/
inductive Feeds : Fin C.N → Fin C.N → Prop
  | refl (t : Fin C.N) : Feeds t t
  | step {g h t : Fin C.N} : g ∈ C.reads h → Feeds h t → Feeds g t

end Circuit

/-- A witness of `C ≼ D`: every input gate `g` of `C` is an input gate `inp g` of `D` of the same width, distinct
gates going to distinct gates, and every output gate `o` of `C` is an output gate `out o` of `D` of the same width, at
which `D` produces, on every assignment `a` to its inputs, the value `C` produces on the inputs `a` gives it. -/
structure Implementation (C D : Circuit) where
  inp : Fin C.N → Fin D.N
  inp_input : ∀ g, C.isInput g = true → D.isInput (inp g) = true
  inp_inj : ∀ g h, C.isInput g = true → C.isInput h = true → inp g = inp h → g = h
  inp_w : ∀ g, C.isInput g = true → D.w (inp g) = C.w g
  out : Fin C.N → Fin D.N
  out_mem : ∀ o ∈ C.outputs, out o ∈ D.outputs
  out_w : ∀ o ∈ C.outputs, D.w (out o) = C.w o
  correct : ∀ (a : D.Transcript) (o : Fin C.N) (ho : o ∈ C.outputs),
    (D.eval a (out o)).cast (out_w o ho) =
      C.eval (fun g => if h : C.isInput g = true then (a (inp g)).cast (inp_w g h) else 0) o

/-- `C ≼ D`: `D` implements `C`. It may compute anything else, and need not compute `C`'s outputs the way `C` does. -/
def Implements (C D : Circuit) : Prop := Nonempty (Implementation C D)

/-- A witness that `D` contains `C`: each gate `g` of `C` is a gate `emb g` of `D` of the same width, input gates going
to input gates and the others to non-input gates that read exactly the images of the gates `g` reads and apply `g`'s
operation to them, and output gates going to output gates. -/
structure Containment (C D : Circuit) where
  emb : Fin C.N → Fin D.N
  emb_inj : Function.Injective emb
  emb_w : ∀ g, D.w (emb g) = C.w g
  isInput_emb : ∀ g, D.isInput (emb g) = C.isInput g
  reads_emb : ∀ g, C.isInput g = false → D.reads (emb g) = (C.reads g).map ⟨emb, emb_inj⟩
  op_emb : ∀ g, C.isInput g = false → ∀ v : D.Transcript,
    (D.op (emb g) v).cast (emb_w g) = C.op g fun a => (v (emb a)).cast (emb_w a)
  outputs_emb : ∀ o ∈ C.outputs, emb o ∈ D.outputs

/-- `D` contains `C`: `C ≼ D`, and `D` computes every non-input gate of `C` from the same wires. -/
def Contains (C D : Circuit) : Prop := Nonempty (Containment C D)

end Verity
