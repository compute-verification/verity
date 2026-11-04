import Proofs.Flock.Bridge.Lowers

/-!
# The lowered program is in the sampling scope's program part

`progOf C κ` and `partOf P κ ns` meet the readable scope's program part (`Claims.Sampling.InScope`'s first two
conjuncts, what `Claims.Sampling.ProgramScope` asks the scope check to decide) at every circuit, constants and
partition:

* `progOf_topological`: the lowered program reads only earlier gates (`reads_lt`, from the circuit's own order
  `Circuit.topo`) and returns its outputs' positions, each one of its gates (`Prog.Topological`).
* `partOf_partitions`: at strata that sum to `n`, `partOf P κ ns` partitions `progOf C κ` (`Partition.Partitions`) when
  every unit holds a gate, no output is an input, and every proof block commits at most `unitClassMax` bits. A lowered
  gate is in one block at most (`eq_of_mem_blockOf`): a unit's gate in its unit's, an input the constants don't fix in
  its owner's. An input no unit reads is unread in the lowered program (`Prog.Unread`), and a constant is in no block.
* `unitClass_of_le`: a proof unit that commits at most `unitClassMax` bits has a public size class.
-/

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

/-- **A proof unit within the widest class has a class**: the least public class that holds its committed values. -/
theorem unitClass_of_le {Cp : Prog} {u : List ℕ} (h : committed Cp u ≤ unitClassMax) :
    ∃ w, unitClass Cp u = some w ∧ committed Cp u ≤ w := by
  have hmem : unitClassMax ∈ unitClasses := by
    unfold unitClasses
    refine List.mem_map.2 ⟨14, List.mem_range.2 ?_, rfl⟩
    rw [show unitClassMax / 16 = 2 ^ 14 from rfl, Nat.log2_two_pow]
    omega
  have hsome : (unitClasses.find? fun w => decide (committed Cp u ≤ w)).isSome :=
    List.find?_isSome.2 ⟨unitClassMax, hmem, decide_eq_true h⟩
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hsome
  exact ⟨w, hw, of_decide_eq_true (List.find?_some hw)⟩

section Lowered

variable {C : FlockSoundness.Audit.Circuit} {κ : Fin C.N → Option Bool}

/-- Wire `g`'s lowered gates are gates of the lowered program. -/
theorem off_add_lt_length (g : Fin C.N) {j : ℕ} (hj : j < gateSize (C.op g)) :
    off C g.val + j < (progOf C κ).gates.length := by
  have h := gates_off (κ := κ) g hj
  have hl : j < (lower C κ g).length := by rw [length_lower]; exact hj
  rw [List.getElem?_eq_getElem hl] at h
  exact (List.getElem?_eq_some_iff.1 h).1

theorem posOf_lt_length (g : Fin C.N) : posOf C g < (progOf C κ).gates.length := by
  have := one_le_gateSize (C.op g)
  rw [posOf_eq]
  exact off_add_lt_length g (by omega)

/-- **Where a block's gates sit**: each is a lowered gate of a wire its unit holds, or the position of an input its unit
owns. -/
theorem wire_of_mem_blockOf {n : ℕ} {P : Partition C n} {u : Fin n} {i : ℕ} (hi : i ∈ blockOf P κ u) :
    ∃ g : Fin C.N, ∃ j < gateSize (C.op g), i = off C g.val + j ∧
      (P.unit g = some u ∨ (C.op g = .input ∧ owner P g = some u)) := by
  rcases mem_blockOf.1 hi with h | ⟨a, ha, rfl⟩
  · obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 h
    exact ⟨g, j, hj, rfl, Or.inl hu⟩
  · obtain ⟨hin, -, ho⟩ := mem_insOf.1 ha
    have := one_le_gateSize (C.op a)
    exact ⟨a, gateSize (C.op a) - 1, by omega, posOf_eq a, Or.inr ⟨hin, ho⟩⟩

theorem lt_length_of_mem_blockOf {n : ℕ} {P : Partition C n} {u : Fin n} {i : ℕ} (hi : i ∈ blockOf P κ u) :
    i < (progOf C κ).gates.length := by
  obtain ⟨g, j, hj, rfl, -⟩ := wire_of_mem_blockOf hi
  exact off_add_lt_length g hj

/-- **A gate is in one block at most**: the wire it lowers names its unit, or, at an input, its owner. -/
theorem eq_of_mem_blockOf {n : ℕ} {P : Partition C n} {u v : Fin n} {i : ℕ} (hu : i ∈ blockOf P κ u)
    (hv : i ∈ blockOf P κ v) : u = v := by
  obtain ⟨g, j, hj, rfl, hg⟩ := wire_of_mem_blockOf hu
  obtain ⟨h, k, hk, he, hh⟩ := wire_of_mem_blockOf hv
  obtain ⟨rfl, -⟩ := block_inj hj hk he
  rcases hg with hg | ⟨hin, hg⟩ <;> rcases hh with hh | ⟨hin', hh⟩
  · exact Option.some.inj (hg.symm.trans hh)
  · exact absurd (isInput_of_unit hg) (by simp [hin', Op.isInput])
  · exact absurd (isInput_of_unit hh) (by simp [hin, Op.isInput])
  · exact Option.some.inj (hg.symm.trans hh)

theorem nodup_unitGates {n : ℕ} (P : Partition C n) (u : Fin n) : (unitGates P u).Nodup := by
  unfold unitGates
  refine List.nodup_flatMap.2 ⟨fun g _ => ?_, (List.nodup_finRange C.N).pairwise_of_forall_ne fun g _ h _ hgh => ?_⟩
  · split
    · exact List.nodup_range.map fun _ _ => Nat.add_left_cancel
    · exact List.nodup_nil
  · simp only [Function.onFun]
    intro i hg hh
    split at hg
    · split at hh
      · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hg
        obtain ⟨k, hk, he⟩ := List.mem_map.1 hh
        exact hgh (block_inj (List.mem_range.1 hj) (List.mem_range.1 hk) he.symm).1
      · simp at hh
    · simp at hg

theorem nodup_blockOf {n : ℕ} (P : Partition C n) (u : Fin n) : (blockOf P κ u).Nodup := by
  refine List.Nodup.append (nodup_unitGates P u)
    (((List.nodup_finRange C.N).filter _).map posOf_injective) fun i hi hi' => ?_
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hi
  obtain ⟨a, ha, he⟩ := List.mem_map.1 hi'
  have hin := (mem_insOf.1 ha).1
  have := one_le_gateSize (C.op a)
  rw [posOf_eq] at he
  obtain ⟨rfl, -⟩ := block_inj (by omega) hj he
  exact absurd (isInput_of_unit hu) (by simp [hin, Op.isInput])

/-- **No gate is in two proof blocks**, nor twice in one. -/
theorem nodup_gates_partOf {n : ℕ} (P : Partition C n) {ns : List ℕ} (hns : ns.sum = n) :
    (partOf P κ ns).gates.Nodup := by
  rw [Claims.Sampling.Partition.gates, blocks_partOf P hns]
  refine List.nodup_flatten.2 ⟨fun l hl => ?_, List.pairwise_map.2 ((List.nodup_finRange n).pairwise_of_forall_ne
    fun u _ v _ huv i hu hv => huv (eq_of_mem_blockOf hu hv))⟩
  obtain ⟨u, -, rfl⟩ := List.mem_map.1 hl
  exact nodup_blockOf P u

/-- **Every lowered gate but a constant and an unread input is in a proof block**: a unit's gate in its unit's, an
input the constants don't fix in its owner's. An input no unit reads is read by no lowered gate, and, no output being
an input, not returned. -/
theorem mem_gates_partOf {n : ℕ} (P : Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    (hout : ∀ g ∈ C.outputs, (C.op g).isInput = false) {i : ℕ} (hi : i < (progOf C κ).gates.length)
    (hc : ∀ b, (progOf C κ).gates[i]? ≠ some (.const b)) (hU : ¬ (progOf C κ).Unread i) :
    i ∈ (partOf P κ ns).gates := by
  rw [Claims.Sampling.Partition.gates, blocks_partOf P hns, List.mem_flatten]
  obtain ⟨g, j, hj, rfl, -⟩ := exists_block (List.getElem?_eq_getElem hi)
  suffices h : ∃ u, off C g.val + j ∈ blockOf P κ u by
    obtain ⟨u, hu⟩ := h
    exact ⟨_, List.mem_map.2 ⟨u, List.mem_finRange u, rfl⟩, hu⟩
  cases hpu : P.unit g with
  | some u => exact ⟨u, mem_blockOf.2 (Or.inl (mem_unitGates.2 ⟨g, hpu, j, hj, rfl⟩))⟩
  | none =>
    have hin := input_of_isInput ((P.input_iff g).1 hpu)
    have hj0 : j = 0 := by rw [hin] at hj; simp [gateSize] at hj; omega
    subst hj0
    have hpos : off C g.val + 0 = posOf C g := by rw [posOf_eq, hin]; rfl
    rw [hpos] at hc hU ⊢
    cases hk : κ g with
    | some b => exact absurd (gates_posOf_const hin hk) (hc b)
    | none =>
      cases ho : owner P g with
      | some u => exact ⟨u, mem_blockOf.2 (Or.inr ⟨g, mem_insOf.2 ⟨hin, hk, ho⟩, rfl⟩)⟩
      | none =>
        refine absurd ⟨gates_posOf_input hin hk, fun hmem => ?_, fun x hx hr => ?_⟩ hU
        · simp only [progOf, List.mem_map, List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq] at hmem
          obtain ⟨o, hoo, he⟩ := hmem
          rw [posOf_injective he] at hoo
          have := hout g hoo
          simp [hin, Op.isInput] at this
        · obtain ⟨p, hp⟩ := List.mem_iff_getElem?.1 hx
          obtain ⟨h, k, hk', rfl, hxk⟩ := exists_block hp
          rcases reads_lower_args hxk hr with ⟨a, ha, he⟩ | ⟨rfl, he | he⟩
          · rw [← posOf_injective he] at ha
            cases hph : P.unit h with
            | none =>
              rw [Op.args_of_isInput ((P.input_iff h).1 hph)] at ha
              simp at ha
            | some v =>
              rw [owner, List.find?_eq_none] at ho
              exact ho v (List.mem_finRange v) (decide_eq_true ⟨h, hph, ha⟩)
          · exact off_ne_posOf (g := h) (a := g) (j := 0) (by omega) (by rw [Nat.add_zero]; exact he.symm)
          · exact off_ne_posOf (g := h) (a := g) (j := 1) (by omega) he.symm

end Lowered

/-- **The lowered program is topologically ordered**: its gates read only earlier gates (`reads_lt`), and it returns
its outputs' positions, each one of its gates. -/
theorem progOf_topological (C : FlockSoundness.Audit.Circuit) (κ : Fin C.N → Option Bool) :
    (progOf C κ).Topological := by
  refine ⟨fun _ _ hx _ hi => reads_lt hx hi, fun o ho => ?_⟩
  simp only [progOf, List.mem_map] at ho
  obtain ⟨g, -, rfl⟩ := ho
  exact posOf_lt_length g

/-- **`partOf P κ ns` partitions `progOf C κ`**, at strata that sum to `n`, when every unit holds a gate, no output is
an input, and every proof block commits at most `unitClassMax` bits: no gate is in two blocks, every block's gates are
the program's, every gate but a constant and an unread input is in a block, every block has a gate, and every block has
a public size class. -/
theorem partOf_partitions {C : FlockSoundness.Audit.Circuit} {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool)
    {ns : List ℕ} (hns : ns.sum = n) (hne : ∀ u, ∃ g, P.unit g = some u)
    (hout : ∀ g ∈ C.outputs, (C.op g).isInput = false)
    (hcls : ∀ u, committed (progOf C κ) (blockOf P κ u) ≤ unitClassMax) :
    (partOf P κ ns).Partitions (progOf C κ) := by
  have hb := blocks_partOf (κ := κ) P hns
  refine ⟨nodup_gates_partOf P hns, fun i hi => ?_, fun i hi hc hU => mem_gates_partOf P hns hout hi hc hU,
    fun l hl => ?_, fun l hl => ?_⟩
  · rw [Claims.Sampling.Partition.gates, hb, List.mem_flatten] at hi
    obtain ⟨l, hl, hil⟩ := hi
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hl
    exact lt_length_of_mem_blockOf hil
  · rw [hb] at hl
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hl
    obtain ⟨g, hg⟩ := hne u
    have := one_le_gateSize (C.op g)
    exact List.ne_nil_of_mem (mem_blockOf.2 (Or.inl (mem_unitGates.2 ⟨g, hg, 0, by omega, rfl⟩)))
  · rw [hb] at hl
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hl
    exact unitClass_of_le (hcls u)

end Flock.Bridge
