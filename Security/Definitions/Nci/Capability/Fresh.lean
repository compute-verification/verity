import Definitions.Core.Formalism

/-!
# Local bounds: what else a block reads fresh

An inference block's effective fresh input is `k_I + k_{F,I} + k_{a,I}` (`Security/README.md`, part 3): its `k`
unregistered input bits, the bits faulty proof blocks inject into it, and the advice bits its structure depends on. Two
objects carry the last two into a `(c, k)`-partition, so that the bounds stated for one apply unchanged:

* **The advice.** The builder takes a declared advice `a`, padded to `ℓa` bits, to a registration (`build a`), so the
  blocks' structure is a function of `a`. `EvaluatesAdvice` is an evaluator fixed before the advice that reads it after
  its own bits.
* **What faulty proof blocks inject.** `injected Q X` is the exits of the faulty proof blocks of the partition `Q` at
  the trace `X` (`Partition.wrong`): committed values that need not be what their gates compute. The cut registration
  (`cutReg R S`) gives each computed gate of `S` a fresh input of its own and makes the gate copy it, so a trace whose
  values at `S` are injected is an honest run of the cut, and the injected bits are unregistered inputs it counts.
-/

namespace Nci.Capability

open Verity Finset

/-- **An evaluator that reads the advice**: `e`, fixed before the advice, followed by any advice `a` of `ℓa` bits, is an
evaluator (`Evaluates`) of `y a` for the registration the builder makes of `a`. -/
def EvaluatesAdvice {ρ : Type} (M : Decoder ρ) (r : ρ) (t : ℕ) (build : List Bool → Registration) (ℓa : ℕ)
    (e : List Bool) (y : (a : List Bool) → Fin (build a).B → (build a).d.D.Transcript → List Bool) : Prop :=
  ∀ a, a.length = ℓa → Evaluates M r t (build a) (e ++ a) (y a)

/-- **What faulty proof blocks inject** at the trace `X`: the exits of `Q`'s faulty proof blocks, the gates whose
committed values need not be what their gates compute. -/
noncomputable def injected {C : Circuit} {n : ℕ} (Q : Partition C n) (X : C.Transcript) : Finset (Fin C.N) :=
  (Q.wrong X).biUnion fun v => C.exits (Q.gates v)

/-- **The cut circuit** of a Boolean circuit `C` at `S`: two copies of `C`'s gates, each one bit. Gate `g` of the first
copy is a fresh input. The second copy is `C`, except that each computed gate of `S` reads gate `g` of the first copy and
copies it. -/
@[reducible] def cutCircuit (C : Circuit) (S : Finset (Fin C.N)) : Circuit where
  N := C.N + C.N
  w := fun _ => 1
  isInput := Fin.addCases (fun _ => true) C.isInput
  reads := Fin.addCases (fun _ => ∅) fun g =>
    if g ∈ S ∧ C.isInput g = false then {Fin.castAdd C.N g} else (C.reads g).map (Fin.natAddEmb C.N)
  reads_lt := by
    intro j
    refine Fin.addCases (fun i => ?_) (fun g => ?_) j
    · simp
    · intro x hx
      simp only [Fin.addCases_right] at hx
      split_ifs at hx with h
      · rw [mem_singleton.1 hx, Fin.lt_def, Fin.val_castAdd, Fin.val_natAdd]
        omega
      · obtain ⟨a, ha, rfl⟩ := mem_map.1 hx
        rw [Fin.lt_def]
        simpa only [Fin.natAddEmb_apply, Fin.val_natAdd, Nat.add_lt_add_iff_left] using
          Fin.lt_def.1 (C.reads_lt g a ha)
  input_reads := by
    intro j
    refine Fin.addCases (fun i => ?_) (fun g => ?_) j
    · simp
    · intro h
      simp only [Fin.addCases_right] at h ⊢
      simp [h, C.input_reads g h]
  op := Fin.addCases (motive := fun _ => ((a : Fin (C.N + C.N)) → BitVec 1) → BitVec 1) (fun _ _ => 0) fun g v =>
    if g ∈ S ∧ C.isInput g = false then v (Fin.castAdd C.N g)
    else (C.op g fun a => (v (Fin.natAdd C.N a)).setWidth (C.w a)).setWidth 1
  op_congr := by
    intro j v v' h
    revert h
    refine Fin.addCases (fun i => ?_) (fun g => ?_) j
    · intro _
      simp
    · intro h
      simp only [Fin.addCases_right] at h ⊢
      split_ifs with hg
      · exact h _ (by simp [hg])
      · rw [C.op_congr g _ _ fun a ha => by rw [h _ (by simp [hg, ha])]]
  outputs := C.outputs.map (Fin.natAddEmb C.N)

/-- The partition `P` on the cut circuit's second copy; the fresh inputs are in no block. -/
def cutPartition {C : Circuit} {n : ℕ} (P : Partition C n) (S : Finset (Fin C.N)) :
    Partition (cutCircuit C S) n where
  unit := Fin.addCases (fun _ => none) P.unit
  input_iff := by
    intro j
    refine Fin.addCases (fun i => ?_) (fun g => ?_) j
    · simp only [Fin.addCases_left]
    · simp only [Fin.addCases_right]
      exact P.input_iff g

/-- The declaration `d` on the cut circuit: `θ` on its second copy, the fresh inputs unregistered. -/
@[reducible] def cutDecl (d : Declaration) (S : Finset (Fin d.D.N)) : Declaration where
  D := cutCircuit d.D S
  θ := d.θ.map (Fin.natAddEmb d.D.N)
  θ_input := by
    intro j hj
    obtain ⟨g, hg, rfl⟩ := mem_map.1 hj
    rw [Fin.natAddEmb_apply]
    show Fin.addCases (fun _ => true) d.D.isInput (Fin.natAdd d.D.N g) = true
    rw [Fin.addCases_right]
    exact d.θ_input g hg

/-- The cut circuit's transcript carrying `X` on both copies. -/
def cutInputs {C : Circuit} (S : Finset (Fin C.N)) (X : C.Transcript) : (cutCircuit C S).Transcript :=
  Fin.addCases (motive := fun _ => BitVec 1) (fun g => (X g).setWidth 1) (fun g => (X g).setWidth 1)

/-- **The cut registration** of `R` at `S`: its circuit cut at `S`, with its partition and its registered values on the
second copy. -/
@[reducible] def cutReg (R : Registration) (S : Finset (Fin R.d.D.N)) : Registration where
  d := cutDecl R.d S
  val := cutInputs S R.val
  B := R.B
  P := cutPartition R.P S

end Nci.Capability
