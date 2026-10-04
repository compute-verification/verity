import FlockSoundness.Audit.Circuit
import Verity.Protocol.Partition
import Verity.SecurityProofs.Boolean

/-!
# C-Flock's circuits as core's

C-Flock's Boolean circuit (`FlockSoundness.Audit.Circuit`) is core's Boolean profile (`Verity.Boolean.Circuit`) field by
field, and so a core circuit of one-bit gates (`Circuit.toCore`). A C-Flock transcript, a bit at every gate, is a core
transcript of one-bit values (`Transcript.toCore`, inverse to `Transcript.ofCore`), and a C-Flock partition is a core
partition with the same units (`Partition.toCore`).

Everything the audit reads agrees across the map: the evaluation (`eval_toCore`), a unit's local evaluation
(`localEval_toCore`), the committed wires, a unit's committed inputs and outputs and the cone of a set of wires
(`committed_toCore`, `io_toCore`, `cone_toCore`), a unit's correctness and so the wrong units (`correct_toCore`,
`wrong_toCore`), and agreement with the anchors (`inputsAgree_toCore`).
-/

namespace FlockBridge

open FlockSoundness.Audit (Op Circuit Partition)

namespace Circuit

variable {N : ℕ}

/-- A C-Flock gate as core's: the same gate. -/
def Op.toCore : Op N → Verity.Boolean.Op N
  | .input => .input
  | .and a b => .and a b
  | .xor a b => .xor a b
  | .not a => .not a
  | .row a b => .row a b

theorem Op.args_toCore (o : Op N) : (Op.toCore o).args = o.args := by cases o <;> rfl

theorem Op.isInput_toCore (o : Op N) : (Op.toCore o).isInput = o.isInput := by cases o <;> rfl

theorem Op.apply_toCore (o : Op N) (v : Fin N → Bool) : (Op.toCore o).apply v = o.apply v := by cases o <;> rfl

/-- A C-Flock circuit as a circuit of core's Boolean profile. -/
def toBoolean (C : Circuit) : Verity.Boolean.Circuit where
  N := C.N
  op g := Op.toCore (C.op g)
  topo g a ha := C.topo g a (by rwa [Op.args_toCore] at ha)
  outputs := C.outputs

/-- A C-Flock circuit as a core circuit: one-bit gates, each applying its C-Flock operation to the bits it reads. It is
the core circuit of `toBoolean C` (`toCore_eq`); written out, so that its gates are `C`'s gates up to reducible
unfolding. -/
@[reducible] def toCore (C : Circuit) : Verity.Circuit where
  N := C.N
  w _ := 1
  isInput g := (C.op g).isInput
  reads g := (C.op g).args.toFinset
  reads_lt g a ha := C.topo g a (List.mem_toFinset.1 ha)
  input_reads g h := by
    revert h
    cases C.op g with
    | input => exact fun _ => List.toFinset_nil
    | and _ _ | xor _ _ | not _ | row _ _ => exact fun h => absurd h Bool.false_ne_true
  op g v := BitVec.ofBool ((C.op g).apply fun a => (v a).getLsbD 0)
  op_congr g v v' h := by
    rw [Op.apply_congr (C.op g) fun x hx => by rw [h x (List.mem_toFinset.2 hx)]]
  outputs := C.outputs

/-- **C-Flock's circuits are core's Boolean profile**: `toCore C` is the core circuit of the Boolean circuit
`toBoolean C`. -/
theorem toCore_eq (C : Circuit) : toCore C = (toBoolean C).toCircuit := by
  simp only [toCore, Verity.Boolean.Circuit.toCircuit, toBoolean, Op.isInput_toCore, Op.args_toCore,
    Op.apply_toCore]
  rfl

variable {C : Circuit}

theorem mem_reads_toCore {g h : Fin C.N} : g ∈ (toCore C).reads h ↔ g ∈ (C.op h).args := List.mem_toFinset

/-- A C-Flock transcript as a core transcript: each bit as a one-bit value. -/
def Transcript.toCore (X : Fin C.N → Bool) : (FlockBridge.Circuit.toCore C).Transcript :=
  fun g => BitVec.ofBool (X g)

/-- A core transcript of the circuit as a C-Flock transcript: each value's bit. -/
def Transcript.ofCore (X : (FlockBridge.Circuit.toCore C).Transcript) : Fin C.N → Bool := fun g => (X g).getLsbD 0

theorem Transcript.ofCore_toCore (X : Fin C.N → Bool) : Transcript.ofCore (Transcript.toCore X) = X := by
  funext g
  simp [Transcript.ofCore, Transcript.toCore]

theorem Transcript.toCore_ofCore (X : (FlockBridge.Circuit.toCore C).Transcript) :
    Transcript.toCore (Transcript.ofCore X) = X := by
  funext g
  refine BitVec.eq_of_getLsbD_eq fun i hi => ?_
  obtain rfl : i = 0 := by
    have : i < 1 := hi
    omega
  simp [Transcript.ofCore, Transcript.toCore]

theorem Transcript.toCore_inj {X Y : Fin C.N → Bool} (g : Fin C.N) :
    Transcript.toCore X g = Transcript.toCore Y g ↔ X g = Y g :=
  BitVec.ofBool_eq_iff_eq

/-- **Evaluation with fixed gates agrees**: fixing each gate at the one-bit value of C-Flock's fixed bit, core's run is
C-Flock's, bit for bit. -/
theorem run_toCore (fix : Fin C.N → Option Bool) (fix' : (g : Fin C.N) → Option (BitVec 1))
    (hfix : ∀ g, fix' g = (fix g).map BitVec.ofBool) (g : Fin C.N) :
    (toCore C).run fix' g = BitVec.ofBool (C.run fix g) := by
  induction g using WellFoundedLT.induction with
  | _ g ih =>
  rw [Verity.Circuit.run_eq (toCore C) fix' g, C.run_eq fix g, hfix g]
  cases fix g with
  | some b => rfl
  | none =>
      show BitVec.ofBool ((C.op g).apply fun a => ((toCore C).run fix' a).getLsbD 0) = _
      congr 1
      refine Op.apply_congr _ fun x hx => ?_
      rw [ih x (C.topo g x hx)]
      simp

theorem eval_toCore (a : Fin C.N → Bool) (g : Fin C.N) :
    (toCore C).eval (Transcript.toCore a) g = BitVec.ofBool (C.eval a g) :=
  run_toCore _ _ (fun h => by
    show (if (C.op h).isInput then some (BitVec.ofBool (a h)) else none) = _
    split <;> rfl) g

theorem inputsAgree_toCore {X a : Fin C.N → Bool} :
    (toCore C).InputsAgree (Transcript.toCore X) (Transcript.toCore a) ↔ C.InputsAgree X a :=
  forall_congr' fun g => imp_congr_right fun _ => Transcript.toCore_inj g

theorem feeds_of_toCore : ∀ {g t : Fin C.N}, (toCore C).Feeds g t → C.Feeds g t
  | _, _, .refl t => .refl t
  | _, _, .step hg h => .step (mem_reads_toCore.1 hg) (feeds_of_toCore h)

theorem feeds_toCore_of : ∀ {g t : Fin C.N}, C.Feeds g t → (toCore C).Feeds g t
  | _, _, .refl t => Verity.Circuit.Feeds.refl (C := toCore C) t
  | _, _, .step hg h => .step (mem_reads_toCore.2 hg) (feeds_toCore_of h)

theorem feeds_toCore {g t : Fin C.N} : (toCore C).Feeds g t ↔ C.Feeds g t := ⟨feeds_of_toCore, feeds_toCore_of⟩

end Circuit

namespace Partition

open Circuit

variable {C : FlockSoundness.Audit.Circuit} {n : ℕ}

/-- A C-Flock partition as a core partition of the same circuit: the same units. -/
def toCore (P : Partition C n) : Verity.Partition (Circuit.toCore C) n where
  unit := P.unit
  input_iff g := P.input_iff g

variable (P : Partition C n)

theorem unit_toCore (g : Fin C.N) : (toCore P).unit g = P.unit g := rfl

theorem committed_toCore : (toCore P).committed = P.committed := by
  classical
  ext g
  simp only [Verity.Partition.committed, FlockSoundness.Audit.Partition.committed, Finset.mem_filter,
    Finset.mem_univ, true_and, List.mem_toFinset, unit_toCore]

theorem io_toCore (u : Fin n) : (toCore P).io u = P.io u := by
  classical
  ext g
  simp only [Verity.Partition.io, FlockSoundness.Audit.Partition.io, Finset.mem_filter, committed_toCore,
    List.mem_toFinset, unit_toCore]

theorem cone_toCore (T : Finset (Fin C.N)) : (toCore P).cone T = P.cone T := by
  classical
  ext u
  simp only [Verity.Partition.cone, FlockSoundness.Audit.Partition.cone, Finset.mem_filter, Finset.mem_univ,
    true_and, Circuit.feeds_toCore, unit_toCore]

theorem localEval_toCore (X : Fin C.N → Bool) (u : Fin n) (g : Fin C.N) :
    (toCore P).localEval (Transcript.toCore X) u g = BitVec.ofBool (P.localEval X u g) :=
  run_toCore _ _ (fun h => by
    show (if P.unit h = some u then none else some (BitVec.ofBool (X h))) = _
    split <;> rfl) g

theorem correct_toCore (X : Fin C.N → Bool) (u : Fin n) :
    (toCore P).Correct (Transcript.toCore X) u ↔ P.Correct X u := by
  simp only [Verity.Partition.Correct, FlockSoundness.Audit.Partition.Correct, committed_toCore,
    localEval_toCore]
  exact forall_congr' fun g => imp_congr_right fun _ => imp_congr_right fun _ => BitVec.ofBool_eq_iff_eq

theorem wrong_toCore (X : Fin C.N → Bool) : (toCore P).wrong (Transcript.toCore X) = P.wrong X := by
  classical
  ext u
  rw [FlockSoundness.Audit.Partition.mem_wrong, Verity.Partition.wrong, Finset.mem_filter]
  simp only [Finset.mem_univ, true_and]
  exact not_congr (correct_toCore P X u)

end Partition

end FlockBridge
