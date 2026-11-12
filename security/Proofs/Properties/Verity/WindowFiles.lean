module

public import Proofs.Core.Circuit
public import Properties.Verity.WindowFiles
public import Definitions.Accounting.JointLive

/-!
# Circuits side by side: a unit of the whole is a unit of its own circuit

`Properties.Verity.WindowFiles` reads a window's committed files as its members' circuits side by side
(`Circuit.beside`, `Partition.beside`). Here a unit of the two side by side is correct, and bound, exactly when it is
in its own circuit at that circuit's part of the trace, so the units a trace doesn't bind are each circuit's own:

* **A trace of the whole at either's gates** (`Circuit.traceLeft X`, `Circuit.traceRight X`): its values at `C`'s
  gates and at `D`'s.
* **The whole at a gate of either** (`Circuit.beside_isInput_left`, `beside_reads_left`, `beside_op_left`,
  `mem_beside_outputs_left`, `Partition.beside_unit_left` and their `right` forms): its own circuit's input flag,
  wires, operation, outputs and unit.
* **Evaluation** (`Circuit.run_beside_left`, `run_beside_right`): evaluating the two side by side with some gates
  fixed is, at a gate of either, evaluating that circuit with its own fixed gates.
* **A unit's recomputation, committed gates and correctness** (`Partition.localEval_beside_left`,
  `mem_committed_beside_left`, `mem_io_beside_left`, `correct_beside_left` and their `right` forms).
* **The incorrect and the unbound units** (`Partition.wrong_beside`, `unbound_beside`, `card_unbound_beside`): those
  of the two side by side are `C`'s at `C`'s part of the trace, then `D`'s at `D`'s, so their counts add.
* **A window's members** (`FileRead.unitsWith_eq_sum`, `FileRead.unboundWith`, `FileRead.card_unbound_partsWith`):
  the window's units are the members' summed, and the units a trace of the window's circuit doesn't bind are counted
  a member at a time, each member at its own part of the trace and the openings at its own gates.
-/

@[expose] public section

namespace Verity

theorem natAdd_ne_castAdd {n m : ℕ} (j : Fin m) (i : Fin n) : Fin.natAdd n j ≠ Fin.castAdd m i := fun e => by
  have h := congrArg Fin.val e
  simp only [Fin.val_natAdd, Fin.val_castAdd] at h
  omega

/-- A value cast to another width and back is itself. The two circuits side by side have `Circuit.append`'s gate count
and widths, which a gate of either meets only once `append` is unfolded, so a cast there and back is closed by this
lemma applied as a term, never by rewriting. -/
theorem cast_cast_self {a b : ℕ} (h : a = b) (h' : b = a) (x : BitVec a) : (x.cast h).cast h' = x := by
  subst h
  simp only [BitVec.cast_eq]

namespace Circuit

variable {C D : Circuit}

/-- **A trace of two circuits side by side, at `C`'s gates.** -/
def traceLeft (X : (C.beside D).Trace) : C.Trace := fun i => (X (Fin.castAdd D.N i)).cast (Fin.append_left C.w D.w i)

/-- **A trace of two circuits side by side, at `D`'s gates.** -/
def traceRight (X : (C.beside D).Trace) : D.Trace := fun j => (X (Fin.natAdd C.N j)).cast (Fin.append_right C.w D.w j)

theorem beside_isInput_left (i : Fin C.N) : (C.beside D).isInput (Fin.castAdd D.N i) = C.isInput i :=
  Fin.append_left C.isInput D.isInput i

theorem beside_isInput_right (j : Fin D.N) : (C.beside D).isInput (Fin.natAdd C.N j) = D.isInput j :=
  Fin.append_right C.isInput D.isInput j

theorem beside_reads_left (i : Fin C.N) :
    (C.beside D).reads (Fin.castAdd D.N i) = (C.reads i).map (Fin.castAddEmb D.N) :=
  Fin.append_left (fun g => (C.reads g).map (Fin.castAddEmb D.N)) (fun j => (D.reads j).map (Fin.natAddEmb C.N)) i

theorem beside_reads_right (j : Fin D.N) :
    (C.beside D).reads (Fin.natAdd C.N j) = (D.reads j).map (Fin.natAddEmb C.N) :=
  Fin.append_right (fun g => (C.reads g).map (Fin.castAddEmb D.N)) (fun j => (D.reads j).map (Fin.natAddEmb C.N)) j

theorem mem_beside_reads_left (g : Fin (C.beside D).N) (i : Fin C.N) :
    g ∈ (C.beside D).reads (Fin.castAdd D.N i) ↔ g ∈ (C.reads i).map (Fin.castAddEmb D.N) :=
  iff_of_eq (congrArg (fun s : Finset (Fin (C.beside D).N) => g ∈ s) (beside_reads_left i))

theorem mem_beside_reads_right (g : Fin (C.beside D).N) (j : Fin D.N) :
    g ∈ (C.beside D).reads (Fin.natAdd C.N j) ↔ g ∈ (D.reads j).map (Fin.natAddEmb C.N) :=
  iff_of_eq (congrArg (fun s : Finset (Fin (C.beside D).N) => g ∈ s) (beside_reads_right j))

/-- **The operation of two circuits side by side, at a gate of `C`**: `C`'s, on the values at `C`'s gates. -/
theorem beside_op_left (i : Fin C.N) (v : (a : Fin (C.beside D).N) → BitVec ((C.beside D).w a)) :
    (C.beside D).op (Fin.castAdd D.N i) v =
      (C.op i fun a => (v (Fin.castAdd D.N a)).cast (Fin.append_left C.w D.w a)).cast
        (Fin.append_left C.w D.w i).symm := by
  show C.appendOp (.ofCircuit C D) (Fin.castAdd D.N i) v = _
  exact Fin.addCases_left (motive := fun g => BitVec (Fin.append C.w D.w g)) i

/-- **The operation of two circuits side by side, at a gate of `D`**: `D`'s, on the values at `D`'s gates. -/
theorem beside_op_right (j : Fin D.N) (v : (a : Fin (C.beside D).N) → BitVec ((C.beside D).w a)) :
    (C.beside D).op (Fin.natAdd C.N j) v =
      (D.op j fun a => (v (Fin.natAdd C.N a)).cast (Fin.append_right C.w D.w a)).cast
        (Fin.append_right C.w D.w j).symm := by
  show C.appendOp (.ofCircuit C D) (Fin.natAdd C.N j) v = _
  exact Fin.addCases_right (motive := fun g => BitVec (Fin.append C.w D.w g)) j

theorem mem_beside_outputs_left (i : Fin C.N) : Fin.castAdd D.N i ∈ (C.beside D).outputs ↔ i ∈ C.outputs := by
  show Fin.castAdd D.N i ∈ C.outputs.map (Fin.castAddEmb D.N) ∪ D.outputs.map (Fin.natAddEmb C.N) ↔ _
  rw [Finset.mem_union]
  constructor
  · rintro (h | h)
    · exact (Finset.mem_map' (Fin.castAddEmb D.N)).1 h
    · obtain ⟨b, -, hb⟩ := Finset.mem_map.1 h
      exact absurd hb (natAdd_ne_castAdd b i)
  · exact fun h => Or.inl (Finset.mem_map_of_mem _ h)

theorem mem_beside_outputs_right (j : Fin D.N) : Fin.natAdd C.N j ∈ (C.beside D).outputs ↔ j ∈ D.outputs := by
  show Fin.natAdd C.N j ∈ C.outputs.map (Fin.castAddEmb D.N) ∪ D.outputs.map (Fin.natAddEmb C.N) ↔ _
  rw [Finset.mem_union]
  constructor
  · rintro (h | h)
    · obtain ⟨b, -, hb⟩ := Finset.mem_map.1 h
      exact absurd hb.symm (natAdd_ne_castAdd j b)
    · exact (Finset.mem_map' (Fin.natAddEmb C.N)).1 h
  · exact fun h => Or.inr (Finset.mem_map_of_mem _ h)

/-- **Evaluating two circuits side by side, at a gate of `C`**: `C`'s evaluation with `C`'s gates fixed as the whole's
are. -/
theorem run_beside_left (fix : (g : Fin (C.beside D).N) → Option (BitVec ((C.beside D).w g))) (i : Fin C.N) :
    (C.beside D).run fix (Fin.castAdd D.N i) =
      (C.run (fun a => (fix (Fin.castAdd D.N a)).map fun b => b.cast (Fin.append_left C.w D.w a)) i).cast
        (Fin.append_left C.w D.w i).symm := by
  induction i using WellFoundedLT.induction with
  | _ i ih => ?_
  cases h : fix (Fin.castAdd D.N i) with
  | some b =>
    rw [(C.beside D).run_fixed h, C.run_fixed (b := b.cast (Fin.append_left C.w D.w i)) (by rw [h]; rfl)]
    exact (cast_cast_self _ _ b).symm
  | none =>
    rw [(C.beside D).run_free h, C.run_free (by rw [h]; rfl)]
    rw [beside_op_left]
    congr 1
    exact C.op_congr i _ _ fun a ha => by rw [ih a (C.reads_lt i a ha)]; exact cast_cast_self _ _ _

/-- **Evaluating two circuits side by side, at a gate of `D`**: `D`'s evaluation with `D`'s gates fixed as the whole's
are. -/
theorem run_beside_right (fix : (g : Fin (C.beside D).N) → Option (BitVec ((C.beside D).w g))) (j : Fin D.N) :
    (C.beside D).run fix (Fin.natAdd C.N j) =
      (D.run (fun a => (fix (Fin.natAdd C.N a)).map fun b => b.cast (Fin.append_right C.w D.w a)) j).cast
        (Fin.append_right C.w D.w j).symm := by
  induction j using WellFoundedLT.induction with
  | _ j ih => ?_
  cases h : fix (Fin.natAdd C.N j) with
  | some b =>
    rw [(C.beside D).run_fixed h, D.run_fixed (b := b.cast (Fin.append_right C.w D.w j)) (by rw [h]; rfl)]
    exact (cast_cast_self _ _ b).symm
  | none =>
    rw [(C.beside D).run_free h, D.run_free (by rw [h]; rfl)]
    rw [beside_op_right]
    congr 1
    exact D.op_congr j _ _ fun a ha => by rw [ih a (D.reads_lt j a ha)]; exact cast_cast_self _ _ _

end Circuit

namespace Partition

theorem mem_committed_iff {C : Circuit} {n : ℕ} (P : Partition C n) (g : Fin C.N) :
    g ∈ P.committed ↔ C.isInput g = true ∨ g ∈ C.outputs ∨ ∃ h, g ∈ C.reads h ∧ P.unit h ≠ P.unit g := by
  simp only [committed, Finset.mem_filter, Finset.mem_univ, true_and]

theorem mem_io_iff {C : Circuit} {n : ℕ} (P : Partition C n) (u : Fin n) (g : Fin C.N) :
    g ∈ P.io u ↔ g ∈ P.committed ∧ (P.unit g = some u ∨ ∃ h, P.unit h = some u ∧ g ∈ C.reads h) := by
  classical
  simp only [io, Finset.mem_filter]

variable {C D : Circuit} {n m : ℕ} (P : Partition C n) (Q : Partition D m)

theorem beside_unit_left (i : Fin C.N) :
    (P.beside Q).unit (Fin.castAdd D.N i) = (P.unit i).map (Fin.castAdd m) := by
  show Fin.addCases (m := C.N) (n := D.N) (motive := fun _ => Option (Fin (n + m)))
    (fun g => (P.unit g).map (Fin.castAdd m)) (fun g => (Q.unit g).map (Fin.natAdd n)) (Fin.castAdd D.N i) = _
  rw [Fin.addCases_left]

theorem beside_unit_right (j : Fin D.N) :
    (P.beside Q).unit (Fin.natAdd C.N j) = (Q.unit j).map (Fin.natAdd n) := by
  show Fin.addCases (m := C.N) (n := D.N) (motive := fun _ => Option (Fin (n + m)))
    (fun g => (P.unit g).map (Fin.castAdd m)) (fun g => (Q.unit g).map (Fin.natAdd n)) (Fin.natAdd C.N j) = _
  rw [Fin.addCases_right]

theorem beside_unit_left_iff (i : Fin C.N) (u : Fin n) :
    (P.beside Q).unit (Fin.castAdd D.N i) = some (Fin.castAdd m u) ↔ P.unit i = some u := by
  rw [beside_unit_left]
  constructor
  · intro h
    obtain ⟨v, hv, e⟩ := Option.map_eq_some_iff.1 h
    rw [hv, Fin.castAdd_injective n m e]
  · intro h
    rw [h]
    rfl

theorem beside_unit_right_iff (j : Fin D.N) (u : Fin m) :
    (P.beside Q).unit (Fin.natAdd C.N j) = some (Fin.natAdd n u) ↔ Q.unit j = some u := by
  rw [beside_unit_right]
  constructor
  · intro h
    obtain ⟨v, hv, e⟩ := Option.map_eq_some_iff.1 h
    rw [hv, Fin.natAdd_injective m n e]
  · intro h
    rw [h]
    rfl

theorem beside_unit_left_ne (i : Fin C.N) (u : Fin m) :
    (P.beside Q).unit (Fin.castAdd D.N i) ≠ some (Fin.natAdd n u) := by
  rw [beside_unit_left]
  intro h
  obtain ⟨v, -, e⟩ := Option.map_eq_some_iff.1 h
  exact natAdd_ne_castAdd u v e.symm

theorem beside_unit_right_ne (j : Fin D.N) (u : Fin n) :
    (P.beside Q).unit (Fin.natAdd C.N j) ≠ some (Fin.castAdd m u) := by
  rw [beside_unit_right]
  intro h
  obtain ⟨v, -, e⟩ := Option.map_eq_some_iff.1 h
  exact natAdd_ne_castAdd v u e

/-- Two gates of `C` are in one unit of the two side by side exactly when they are in one unit of `C`'s. -/
theorem beside_unit_left_eq_iff (i h : Fin C.N) :
    (P.beside Q).unit (Fin.castAdd D.N h) = (P.beside Q).unit (Fin.castAdd D.N i) ↔ P.unit h = P.unit i := by
  rw [beside_unit_left, beside_unit_left]
  exact (Option.map_injective (Fin.castAdd_injective n m)).eq_iff

theorem beside_unit_right_eq_iff (j h : Fin D.N) :
    (P.beside Q).unit (Fin.natAdd C.N h) = (P.beside Q).unit (Fin.natAdd C.N j) ↔ Q.unit h = Q.unit j := by
  rw [beside_unit_right, beside_unit_right]
  exact (Option.map_injective (Fin.natAdd_injective m n)).eq_iff

/-- **A gate of `C` is committed in the two side by side exactly when it is in `C`.** -/
theorem mem_committed_beside_left (i : Fin C.N) :
    Fin.castAdd D.N i ∈ (P.beside Q).committed ↔ i ∈ P.committed := by
  refine (mem_committed_iff (P.beside Q) (Fin.castAdd D.N i)).trans (Iff.trans ?_ (mem_committed_iff P i).symm)
  refine or_congr (iff_of_eq (congrArg (· = true) (Circuit.beside_isInput_left i)))
    (or_congr (Circuit.mem_beside_outputs_left i) ⟨?_, ?_⟩)
  · rintro ⟨h, hr, hne⟩
    revert hr hne
    refine Fin.addCases (motive := fun h => Fin.castAdd D.N i ∈ (C.beside D).reads h →
      (P.beside Q).unit h ≠ (P.beside Q).unit (Fin.castAdd D.N i) → _) (fun h' hr hne => ?_) (fun j hr _ => ?_) h
    · replace hr := (Circuit.mem_beside_reads_left _ h').1 hr
      exact ⟨h', (Finset.mem_map' (Fin.castAddEmb D.N)).1 hr, fun e => hne ((P.beside_unit_left_eq_iff Q i h').2 e)⟩
    · replace hr := (Circuit.mem_beside_reads_right _ j).1 hr
      obtain ⟨b, -, hb⟩ := Finset.mem_map.1 hr
      exact absurd hb (natAdd_ne_castAdd b i)
  · rintro ⟨h', hr, hne⟩
    exact ⟨Fin.castAdd D.N h', (Circuit.mem_beside_reads_left _ h').2 (Finset.mem_map_of_mem (Fin.castAddEmb D.N) hr),
      fun e => hne ((P.beside_unit_left_eq_iff Q i h').1 e)⟩

/-- **A gate of `D` is committed in the two side by side exactly when it is in `D`.** -/
theorem mem_committed_beside_right (j : Fin D.N) :
    Fin.natAdd C.N j ∈ (P.beside Q).committed ↔ j ∈ Q.committed := by
  refine (mem_committed_iff (P.beside Q) (Fin.natAdd C.N j)).trans (Iff.trans ?_ (mem_committed_iff Q j).symm)
  refine or_congr (iff_of_eq (congrArg (· = true) (Circuit.beside_isInput_right j)))
    (or_congr (Circuit.mem_beside_outputs_right j) ⟨?_, ?_⟩)
  · rintro ⟨h, hr, hne⟩
    revert hr hne
    refine Fin.addCases (motive := fun h => Fin.natAdd C.N j ∈ (C.beside D).reads h →
      (P.beside Q).unit h ≠ (P.beside Q).unit (Fin.natAdd C.N j) → _) (fun i hr _ => ?_) (fun h' hr hne => ?_) h
    · replace hr := (Circuit.mem_beside_reads_left _ i).1 hr
      obtain ⟨b, -, hb⟩ := Finset.mem_map.1 hr
      exact absurd hb.symm (natAdd_ne_castAdd j b)
    · replace hr := (Circuit.mem_beside_reads_right _ h').1 hr
      exact ⟨h', (Finset.mem_map' (Fin.natAddEmb C.N)).1 hr, fun e => hne ((P.beside_unit_right_eq_iff Q j h').2 e)⟩
  · rintro ⟨h', hr, hne⟩
    exact ⟨Fin.natAdd C.N h', (Circuit.mem_beside_reads_right _ h').2 (Finset.mem_map_of_mem (Fin.natAddEmb C.N) hr),
      fun e => hne ((P.beside_unit_right_eq_iff Q j h').1 e)⟩

/-- **A unit of `C`'s recomputed in the two side by side, at a gate of `C`**: its recomputation in `C` at `C`'s part
of the trace. -/
theorem localEval_beside_left (X : (C.beside D).Trace) (u : Fin n) (i : Fin C.N) :
    (P.beside Q).localEval X (Fin.castAdd m u) (Fin.castAdd D.N i) =
      (P.localEval (Circuit.traceLeft X) u i).cast (Fin.append_left C.w D.w i).symm := by
  unfold localEval
  rw [Circuit.run_beside_left]
  congr 2
  funext a
  by_cases h : P.unit a = some u
  · rw [if_pos h, if_pos ((P.beside_unit_left_iff Q a u).2 h)]
    rfl
  · rw [if_neg h, if_neg (mt (P.beside_unit_left_iff Q a u).1 h)]
    rfl

/-- **A unit of `D`'s recomputed in the two side by side, at a gate of `D`**: its recomputation in `D` at `D`'s part
of the trace. -/
theorem localEval_beside_right (X : (C.beside D).Trace) (u : Fin m) (j : Fin D.N) :
    (P.beside Q).localEval X (Fin.natAdd n u) (Fin.natAdd C.N j) =
      (Q.localEval (Circuit.traceRight X) u j).cast (Fin.append_right C.w D.w j).symm := by
  unfold localEval
  rw [Circuit.run_beside_right]
  congr 2
  funext a
  by_cases h : Q.unit a = some u
  · rw [if_pos h, if_pos ((P.beside_unit_right_iff Q a u).2 h)]
    rfl
  · rw [if_neg h, if_neg (mt (P.beside_unit_right_iff Q a u).1 h)]
    rfl

/-- **A unit of `C`'s is correct in the two side by side exactly when it is in `C`**, at `C`'s part of the trace. -/
theorem correct_beside_left (X : (C.beside D).Trace) (u : Fin n) :
    (P.beside Q).Correct X (Fin.castAdd m u) ↔ P.Correct (Circuit.traceLeft X) u := by
  constructor
  · intro h i hi hu
    have hc := h (Fin.castAdd D.N i) ((P.mem_committed_beside_left Q i).2 hi) ((P.beside_unit_left_iff Q i u).2 hu)
    rw [P.localEval_beside_left Q] at hc
    show (X (Fin.castAdd D.N i)).cast _ = _
    rw [hc]
    exact cast_cast_self _ _ _
  · intro h g
    refine Fin.addCases (motive := fun g => g ∈ (P.beside Q).committed →
      (P.beside Q).unit g = some (Fin.castAdd m u) → X g = (P.beside Q).localEval X (Fin.castAdd m u) g)
      (fun i hg hu => ?_) (fun j _ hu => ?_) g
    · rw [P.localEval_beside_left Q,
        ← h i ((P.mem_committed_beside_left Q i).1 hg) ((P.beside_unit_left_iff Q i u).1 hu)]
      show _ = ((X (Fin.castAdd D.N i)).cast _).cast _
      exact (cast_cast_self _ _ _).symm
    · exact absurd hu (P.beside_unit_right_ne Q j u)

/-- **A unit of `D`'s is correct in the two side by side exactly when it is in `D`**, at `D`'s part of the trace. -/
theorem correct_beside_right (X : (C.beside D).Trace) (u : Fin m) :
    (P.beside Q).Correct X (Fin.natAdd n u) ↔ Q.Correct (Circuit.traceRight X) u := by
  constructor
  · intro h j hj hu
    have hc := h (Fin.natAdd C.N j) ((P.mem_committed_beside_right Q j).2 hj) ((P.beside_unit_right_iff Q j u).2 hu)
    rw [P.localEval_beside_right Q] at hc
    show (X (Fin.natAdd C.N j)).cast _ = _
    rw [hc]
    exact cast_cast_self _ _ _
  · intro h g
    refine Fin.addCases (motive := fun g => g ∈ (P.beside Q).committed →
      (P.beside Q).unit g = some (Fin.natAdd n u) → X g = (P.beside Q).localEval X (Fin.natAdd n u) g)
      (fun i _ hu => ?_) (fun j hg hu => ?_) g
    · exact absurd hu (P.beside_unit_left_ne Q i u)
    · rw [P.localEval_beside_right Q,
        ← h j ((P.mem_committed_beside_right Q j).1 hg) ((P.beside_unit_right_iff Q j u).1 hu)]
      show _ = ((X (Fin.natAdd C.N j)).cast _).cast _
      exact (cast_cast_self _ _ _).symm

open Classical in
/-- **The incorrect units of two partitions side by side**: `P`'s at `C`'s part of the trace, then `Q`'s at `D`'s. -/
theorem wrong_beside (X : (C.beside D).Trace) :
    (P.beside Q).wrong X = (P.wrong (Circuit.traceLeft X)).map (Fin.castAddEmb m) ∪
      (Q.wrong (Circuit.traceRight X)).map (Fin.natAddEmb n) := by
  ext w
  rw [mem_incorrect, Finset.mem_union]
  refine Fin.addCases (motive := fun w => ¬ (P.beside Q).Correct X w ↔
    w ∈ (P.wrong (Circuit.traceLeft X)).map (Fin.castAddEmb m) ∨
      w ∈ (Q.wrong (Circuit.traceRight X)).map (Fin.natAddEmb n)) (fun u => ?_) (fun u => ?_) w
  · rw [P.correct_beside_left Q, ← P.mem_incorrect]
    constructor
    · exact fun h => Or.inl (Finset.mem_map_of_mem _ h)
    · rintro (h | h)
      · exact (Finset.mem_map' (Fin.castAddEmb m)).1 h
      · obtain ⟨b, -, hb⟩ := Finset.mem_map.1 h
        exact absurd hb (natAdd_ne_castAdd b u)
  · rw [P.correct_beside_right Q, ← Q.mem_incorrect]
    constructor
    · exact fun h => Or.inr (Finset.mem_map_of_mem _ h)
    · rintro (h | h)
      · obtain ⟨b, -, hb⟩ := Finset.mem_map.1 h
        exact absurd hb.symm (natAdd_ne_castAdd u b)
      · exact (Finset.mem_map' (Fin.natAddEmb n)).1 h

/-- **A gate of `C` is among a unit of `C`'s committed inputs and outputs in the two side by side exactly when it is in
`C`.** -/
theorem mem_io_beside_left (u : Fin n) (i : Fin C.N) :
    Fin.castAdd D.N i ∈ (P.beside Q).io (Fin.castAdd m u) ↔ i ∈ P.io u := by
  refine (mem_io_iff (P.beside Q) (Fin.castAdd m u) (Fin.castAdd D.N i)).trans (Iff.trans ?_ (mem_io_iff P u i).symm)
  refine and_congr (P.mem_committed_beside_left Q i) (or_congr (P.beside_unit_left_iff Q i u) ⟨?_, ?_⟩)
  · rintro ⟨h, hu, hr⟩
    revert hu hr
    refine Fin.addCases (motive := fun h => (P.beside Q).unit h = some (Fin.castAdd m u) →
      Fin.castAdd D.N i ∈ (C.beside D).reads h → _) (fun h' hu hr => ?_) (fun j hu _ => ?_) h
    · replace hr := (Circuit.mem_beside_reads_left _ h').1 hr
      exact ⟨h', (P.beside_unit_left_iff Q h' u).1 hu, (Finset.mem_map' (Fin.castAddEmb D.N)).1 hr⟩
    · exact absurd hu (P.beside_unit_right_ne Q j u)
  · rintro ⟨h', hu, hr⟩
    exact ⟨Fin.castAdd D.N h', (P.beside_unit_left_iff Q h' u).2 hu,
      (Circuit.mem_beside_reads_left _ h').2 (Finset.mem_map_of_mem (Fin.castAddEmb D.N) hr)⟩

/-- **A gate of `D` is among a unit of `D`'s committed inputs and outputs in the two side by side exactly when it is in
`D`.** -/
theorem mem_io_beside_right (u : Fin m) (j : Fin D.N) :
    Fin.natAdd C.N j ∈ (P.beside Q).io (Fin.natAdd n u) ↔ j ∈ Q.io u := by
  refine (mem_io_iff (P.beside Q) (Fin.natAdd n u) (Fin.natAdd C.N j)).trans (Iff.trans ?_ (mem_io_iff Q u j).symm)
  refine and_congr (P.mem_committed_beside_right Q j) (or_congr (P.beside_unit_right_iff Q j u) ⟨?_, ?_⟩)
  · rintro ⟨h, hu, hr⟩
    revert hu hr
    refine Fin.addCases (motive := fun h => (P.beside Q).unit h = some (Fin.natAdd n u) →
      Fin.natAdd C.N j ∈ (C.beside D).reads h → _) (fun i hu _ => ?_) (fun h' hu hr => ?_) h
    · exact absurd hu (P.beside_unit_left_ne Q i u)
    · replace hr := (Circuit.mem_beside_reads_right _ h').1 hr
      exact ⟨h', (P.beside_unit_right_iff Q h' u).1 hu, (Finset.mem_map' (Fin.natAddEmb C.N)).1 hr⟩
  · rintro ⟨h', hu, hr⟩
    exact ⟨Fin.natAdd C.N h', (P.beside_unit_right_iff Q h' u).2 hu,
      (Circuit.mem_beside_reads_right _ h').2 (Finset.mem_map_of_mem (Fin.natAddEmb C.N) hr)⟩

/-- No gate of `D` is among a unit of `C`'s committed inputs and outputs in the two side by side. -/
theorem not_mem_io_beside_left (u : Fin n) (j : Fin D.N) : Fin.natAdd C.N j ∉ (P.beside Q).io (Fin.castAdd m u) := by
  intro hmem
  obtain ⟨-, hu | ⟨h, hu, hr⟩⟩ := (mem_io_iff (P.beside Q) (Fin.castAdd m u) (Fin.natAdd C.N j)).1 hmem
  · exact P.beside_unit_right_ne Q j u hu
  · revert hu hr
    refine Fin.addCases (motive := fun h => (P.beside Q).unit h = some (Fin.castAdd m u) →
      Fin.natAdd C.N j ∈ (C.beside D).reads h → False) (fun h' _ hr => ?_) (fun j' hu _ => ?_) h
    · replace hr := (Circuit.mem_beside_reads_left _ h').1 hr
      obtain ⟨b, -, hb⟩ := Finset.mem_map.1 hr
      exact natAdd_ne_castAdd j b hb.symm
    · exact P.beside_unit_right_ne Q j' u hu

/-- No gate of `C` is among a unit of `D`'s committed inputs and outputs in the two side by side. -/
theorem not_mem_io_beside_right (u : Fin m) (i : Fin C.N) : Fin.castAdd D.N i ∉ (P.beside Q).io (Fin.natAdd n u) := by
  intro hmem
  obtain ⟨-, hu | ⟨h, hu, hr⟩⟩ := (mem_io_iff (P.beside Q) (Fin.natAdd n u) (Fin.castAdd D.N i)).1 hmem
  · exact P.beside_unit_left_ne Q i u hu
  · revert hu hr
    refine Fin.addCases (motive := fun h => (P.beside Q).unit h = some (Fin.natAdd n u) →
      Fin.castAdd D.N i ∈ (C.beside D).reads h → False) (fun i' hu _ => ?_) (fun h' _ hr => ?_) h
    · exact P.beside_unit_left_ne Q i' u hu
    · replace hr := (Circuit.mem_beside_reads_right _ h').1 hr
      obtain ⟨b, -, hb⟩ := Finset.mem_map.1 hr
      exact natAdd_ne_castAdd b i hb

end Partition

open Classical in
/-- **The units a trace of two circuits side by side doesn't bind**: `P`'s that `C`'s part of the trace doesn't bind
under the openings at `C`'s gates, then `Q`'s that `D`'s part doesn't under the openings at `D`'s. -/
theorem unbound_beside {C D : Circuit} {n m : ℕ} (P : Partition C n) (Q : Partition D m)
    (opens : (g : Fin (C.beside D).N) → BitVec ((C.beside D).w g) → Prop) (X : (C.beside D).Trace) :
    Accounting.unbound (P.beside Q) opens X =
      (Accounting.unbound P (fun i v => opens (Fin.castAdd D.N i) (v.cast (Fin.append_left C.w D.w i).symm))
        (Circuit.traceLeft X)).map (Fin.castAddEmb m) ∪
      (Accounting.unbound Q (fun j v => opens (Fin.natAdd C.N j) (v.cast (Fin.append_right C.w D.w j).symm))
        (Circuit.traceRight X)).map (Fin.natAddEmb n) := by
  have hL : ∀ i : Fin C.N, ((Circuit.traceLeft X) i).cast (Fin.append_left C.w D.w i).symm =
      X (Fin.castAdd D.N i) := fun i => by
    show ((X (Fin.castAdd D.N i)).cast _).cast _ = _
    exact cast_cast_self _ _ _
  have hR : ∀ j : Fin D.N, ((Circuit.traceRight X) j).cast (Fin.append_right C.w D.w j).symm =
      X (Fin.natAdd C.N j) := fun j => by
    show ((X (Fin.natAdd C.N j)).cast _).cast _ = _
    exact cast_cast_self _ _ _
  ext w
  simp only [Accounting.unbound, Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
  refine Fin.addCases (motive := fun w =>
    (¬ (P.beside Q).Correct X w ∨ ∃ g ∈ (P.beside Q).io w, ¬ opens g (X g)) ↔
      w ∈ (Finset.univ.filter fun u => ¬ P.Correct (Circuit.traceLeft X) u ∨ ∃ g ∈ P.io u,
          ¬ opens (Fin.castAdd D.N g) (((Circuit.traceLeft X) g).cast (Fin.append_left C.w D.w g).symm)).map
            (Fin.castAddEmb m) ∨
        w ∈ (Finset.univ.filter fun u => ¬ Q.Correct (Circuit.traceRight X) u ∨ ∃ g ∈ Q.io u,
          ¬ opens (Fin.natAdd C.N g) (((Circuit.traceRight X) g).cast (Fin.append_right C.w D.w g).symm)).map
            (Fin.natAddEmb n)) (fun u => ?_) (fun u => ?_) w
  · have hmem : (¬ P.Correct (Circuit.traceLeft X) u ∨ ∃ g ∈ P.io u,
        ¬ opens (Fin.castAdd D.N g) (((Circuit.traceLeft X) g).cast (Fin.append_left C.w D.w g).symm)) ↔
        (¬ (P.beside Q).Correct X (Fin.castAdd m u) ∨
          ∃ g ∈ (P.beside Q).io (Fin.castAdd m u), ¬ opens g (X g)) := by
      rw [P.correct_beside_left Q]
      refine or_congr Iff.rfl ⟨?_, ?_⟩
      · rintro ⟨i, hi, ho⟩
        exact ⟨Fin.castAdd D.N i, (P.mem_io_beside_left Q u i).2 hi,
          fun h => ho ((congrArg (opens (Fin.castAdd D.N i)) (hL i)).mpr h)⟩
      · rintro ⟨g, hg, ho⟩
        revert hg ho
        refine Fin.addCases (motive := fun g => g ∈ (P.beside Q).io (Fin.castAdd m u) → ¬ opens g (X g) → _)
          (fun i hg ho => ?_) (fun j hg _ => ?_) g
        · exact ⟨i, (P.mem_io_beside_left Q u i).1 hg,
            fun h => ho ((congrArg (opens (Fin.castAdd D.N i)) (hL i)).mp h)⟩
        · exact absurd hg (P.not_mem_io_beside_left Q u j)
    rw [← hmem]
    constructor
    · exact fun h => Or.inl (Finset.mem_map_of_mem _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    · rintro (h | h)
      · exact (Finset.mem_filter.1 ((Finset.mem_map' (Fin.castAddEmb m)).1 h)).2
      · obtain ⟨b, -, hb⟩ := Finset.mem_map.1 h
        exact absurd hb (natAdd_ne_castAdd b u)
  · have hmem : (¬ Q.Correct (Circuit.traceRight X) u ∨ ∃ g ∈ Q.io u,
        ¬ opens (Fin.natAdd C.N g) (((Circuit.traceRight X) g).cast (Fin.append_right C.w D.w g).symm)) ↔
        (¬ (P.beside Q).Correct X (Fin.natAdd n u) ∨
          ∃ g ∈ (P.beside Q).io (Fin.natAdd n u), ¬ opens g (X g)) := by
      rw [P.correct_beside_right Q]
      refine or_congr Iff.rfl ⟨?_, ?_⟩
      · rintro ⟨j, hj, ho⟩
        exact ⟨Fin.natAdd C.N j, (P.mem_io_beside_right Q u j).2 hj,
          fun h => ho ((congrArg (opens (Fin.natAdd C.N j)) (hR j)).mpr h)⟩
      · rintro ⟨g, hg, ho⟩
        revert hg ho
        refine Fin.addCases (motive := fun g => g ∈ (P.beside Q).io (Fin.natAdd n u) → ¬ opens g (X g) → _)
          (fun i hg _ => ?_) (fun j hg ho => ?_) g
        · exact absurd hg (P.not_mem_io_beside_right Q u i)
        · exact ⟨j, (P.mem_io_beside_right Q u j).1 hg,
            fun h => ho ((congrArg (opens (Fin.natAdd C.N j)) (hR j)).mp h)⟩
    rw [← hmem]
    constructor
    · exact fun h => Or.inr (Finset.mem_map_of_mem _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, h⟩))
    · rintro (h | h)
      · obtain ⟨b, -, hb⟩ := Finset.mem_map.1 h
        exact absurd hb.symm (natAdd_ne_castAdd u b)
      · exact (Finset.mem_filter.1 ((Finset.mem_map' (Fin.natAddEmb n)).1 h)).2

open Classical in
/-- **The unbound units of two circuits side by side are counted a circuit at a time.** -/
theorem card_unbound_beside {C D : Circuit} {n m : ℕ} (P : Partition C n) (Q : Partition D m)
    (opens : (g : Fin (C.beside D).N) → BitVec ((C.beside D).w g) → Prop) (X : (C.beside D).Trace) :
    (Accounting.unbound (P.beside Q) opens X).card =
      (Accounting.unbound P (fun i v => opens (Fin.castAdd D.N i) (v.cast (Fin.append_left C.w D.w i).symm))
        (Circuit.traceLeft X)).card +
      (Accounting.unbound Q (fun j v => opens (Fin.natAdd C.N j) (v.cast (Fin.append_right C.w D.w j).symm))
        (Circuit.traceRight X)).card := by
  rw [unbound_beside, Finset.card_union_of_disjoint, Finset.card_map, Finset.card_map]
  refine Finset.disjoint_left.2 fun w hl hr => ?_
  obtain ⟨a, -, ha⟩ := Finset.mem_map.1 hl
  obtain ⟨b, -, hb⟩ := Finset.mem_map.1 hr
  exact natAdd_ne_castAdd b a (hb.trans ha.symm)

namespace FileRead

theorem unitsWith_eq_sum (r : FileRead) (rs : List FileRead) :
    unitsWith r rs = ((r :: rs).map FileRead.units).sum := by
  induction rs generalizing r with
  | nil => simp [unitsWith]
  | cons s rs ih =>
    rw [unitsWith, ih s]
    simp only [List.map_cons, List.sum_cons]

open Classical in
/-- **The units of a first member and the rest that a trace doesn't bind, counted a member at a time**: the first
member's that its part of the trace doesn't bind under the openings at its gates, then the rest's at theirs. -/
noncomputable def unboundWith : (r : FileRead) → (rs : List FileRead) →
    ((g : Fin (circuitWith r rs).N) → BitVec ((circuitWith r rs).w g) → Prop) → (circuitWith r rs).Trace → ℕ
  | r, [], opens, X => (Accounting.unbound r.parts opens X).card
  | r, s :: rs, opens, X =>
    (Accounting.unbound r.parts
        (fun i v => opens (Fin.castAdd (circuitWith s rs).N i)
          (v.cast (Fin.append_left r.circuit.w (circuitWith s rs).w i).symm))
        (Circuit.traceLeft X)).card +
      unboundWith s rs
        (fun j v => opens (Fin.natAdd r.circuit.N j)
          (v.cast (Fin.append_right r.circuit.w (circuitWith s rs).w j).symm))
        (Circuit.traceRight X)

open Classical in
/-- **The units a trace of the members' circuits side by side doesn't bind are counted a member at a time.** -/
theorem card_unbound_partsWith (r : FileRead) (rs : List FileRead)
    (opens : (g : Fin (circuitWith r rs).N) → BitVec ((circuitWith r rs).w g) → Prop) (X : (circuitWith r rs).Trace) :
    (Accounting.unbound (partsWith r rs) opens X).card = unboundWith r rs opens X := by
  induction rs generalizing r with
  | nil => rfl
  | cons s rs ih =>
    exact (card_unbound_beside r.parts (partsWith s rs) opens X).trans (congrArg (_ + ·) (ih s _ _))

end FileRead

end Verity
