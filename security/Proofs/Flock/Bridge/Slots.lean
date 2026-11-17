module

public import Proofs.Flock.Bridge.Lowers
public import Proofs.Flock.Claims.Sampling.InputSlot

/-! The population's slots, then the input class, partition the lowered program (`partitions_slots`). -/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

variable {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool}

theorem sizeClass_of_le_max {Cp : Prog} {u : List ℕ} (h : committed Cp u ≤ sizeClassMax) :
    ∃ w, sizeClass Cp u = some w ∧ committed Cp u ≤ w := by
  have hlog : Nat.log2 (sizeClassMax / 16) = 14 := by
    rw [show sizeClassMax / 16 = 2 ^ 14 from rfl, Nat.log2_two_pow]
  have hmem : sizeClassMax ∈ sizeClasses := by
    rw [sizeClasses, hlog]
    exact List.mem_map.2 ⟨14, by simp, rfl⟩
  have hs : (sizeClasses.find? fun w => decide (committed Cp u ≤ w)).isSome :=
    List.find?_isSome.2 ⟨sizeClassMax, hmem, decide_eq_true h⟩
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hs
  have hd := List.find?_some hw
  exact ⟨w, hw, of_decide_eq_true hd⟩

theorem lower_ne_input {g : Fin C.N} (hni : (C.op g).isInput = false) : ∀ x ∈ lower C κ g, x ≠ .input := by
  unfold lower
  cases h : C.op g <;> simp_all [Verity.Boolean.Op.isInput]

theorem progOf_ne_input_of_unit {n : ℕ} {P : Partition C n} {u : Fin n} {p : ℕ} (hp : p ∈ unitGates P u) :
    (progOf C κ).gates[p]? ≠ some .input := by
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hp
  rw [gates_off g hj]
  intro h
  exact lower_ne_input (isInput_of_unit hu) _ (List.mem_of_getElem? h) rfl

theorem lt_length_of_unit {n : ℕ} {P : Partition C n} {u : Fin n} {p : ℕ} (hp : p ∈ unitGates P u) :
    p < (progOf C κ).gates.length := by
  obtain ⟨g, -, j, hj, rfl⟩ := mem_unitGates.1 hp
  have h := gates_off (κ := κ) g hj
  have hl : j < (lower C κ g).length := by rw [length_lower]; exact hj
  rw [List.getElem?_eq_getElem hl] at h
  exact (List.getElem?_eq_some_iff.1 h).1

theorem nodup_unitGates {n : ℕ} (P : Partition C n) (u : Fin n) : (unitGates P u).Nodup := by
  unfold unitGates
  refine List.nodup_flatMap.2 ⟨fun g _ => ?_, (List.nodup_finRange C.N).pairwise_of_forall_ne fun g _ h _ hgh => ?_⟩
  · split
    · exact List.nodup_range.map (add_right_injective _)
    · exact List.nodup_nil
  · intro i hg hh
    beta_reduce at hg hh
    split at hg
    · split at hh
      · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hg
        obtain ⟨k, hk, he⟩ := List.mem_map.1 hh
        exact hgh (block_inj (List.mem_range.1 hj) (List.mem_range.1 hk) he.symm).1
      · simp at hh
    · simp at hg

theorem disjoint_unitGates {n : ℕ} (P : Partition C n) {u v : Fin n} (huv : u ≠ v) :
    (unitGates P u).Disjoint (unitGates P v) := by
  intro i hu hv
  obtain ⟨g, hg, j, hj, rfl⟩ := mem_unitGates.1 hu
  obtain ⟨h, hh, k, hk, he⟩ := mem_unitGates.1 hv
  obtain ⟨rfl, -⟩ := block_inj hj hk he
  exact huv (Option.some.inj (hg.symm.trans hh))

theorem exists_unitGates {n : ℕ} (P : Partition C n) {i : ℕ} (hi : i < (progOf C κ).gates.length)
    (hc : ∀ b, (progOf C κ).gates[i]? ≠ some (.const b)) (hin : (progOf C κ).gates[i]? ≠ some .input) :
    ∃ u, i ∈ unitGates P u := by
  obtain ⟨g, j, hj, rfl, hx⟩ := exists_block (List.getElem?_eq_getElem hi)
  cases hu : P.unit g with
  | some u => exact ⟨u, mem_unitGates.2 ⟨g, hu, j, hj, rfl⟩⟩
  | none =>
    exfalso
    have hop := input_of_isInput ((P.input_iff g).1 hu)
    have hj0 : j = 0 := by rw [hop] at hj; simp [gateSize] at hj; omega
    subst hj0
    rw [List.getElem?_eq_getElem hi] at hc hin
    unfold lower at hx
    rw [hop] at hx
    cases hk : κ g <;> simp [hk] at hx
    · exact hin (congrArg some hx.symm)
    · exact hc _ (congrArg some hx.symm)

theorem blocks_slots {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) {ns : List ℕ} (hns : n ≤ ns.sum) :
    (InputSlot.withInputs ⟨[ns.splitLengths ((List.finRange n).map (unitGates P))]⟩ (progOf C κ)).blocks =
      (List.finRange n).map (unitGates P) ++ (InputSlot.inputsAt (progOf C κ)).map fun g => [g] := by
  rw [InputSlot.blocks_withInputs]
  simp [Partition.blocks, List.flatten_splitLengths ((List.finRange n).map (unitGates P)) ns (by simpa using hns)]

theorem partitions_slots {n : ℕ} (P : Partition C n) (κ : Fin C.N → Option Bool) {ns : List ℕ} (hns : n ≤ ns.sum)
    (hne : ∀ u, unitGates P u ≠ []) (hw : ∀ u, committed (progOf C κ) (unitGates P u) ≤ sizeClassMax) :
    (InputSlot.withInputs ⟨[ns.splitLengths ((List.finRange n).map (unitGates P))]⟩ (progOf C κ)).Partitions
      (progOf C κ) := by
  have hb := blocks_slots P κ hns
  have hg : (InputSlot.withInputs ⟨[ns.splitLengths ((List.finRange n).map (unitGates P))]⟩ (progOf C κ)).gates =
      (List.finRange n).flatMap (unitGates P) ++ InputSlot.inputsAt (progOf C κ) := by
    have hs : ∀ l : List ℕ, (l.map fun g => [g]).flatten = l := fun l => by induction l <;> simp_all
    simp [Partition.gates, hb, List.flatMap, hs]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [hg]
    refine List.nodup_append.2 ⟨?_, List.nodup_range.filter _, ?_⟩
    · exact List.nodup_flatMap.2 ⟨fun u _ => nodup_unitGates P u,
        (List.nodup_finRange n).pairwise_of_forall_ne fun _ _ _ _ huv => disjoint_unitGates P huv⟩
    · intro a ha b hb he
      obtain ⟨u, -, hu⟩ := List.mem_flatMap.1 ha
      subst he
      exact progOf_ne_input_of_unit hu (InputSlot.mem_inputsAt.1 hb)
  · rw [hg]
    intro p hp
    rcases List.mem_append.1 hp with h | h
    · obtain ⟨u, -, hu⟩ := List.mem_flatMap.1 h
      exact lt_length_of_unit hu
    · exact (List.getElem?_eq_some_iff.1 (InputSlot.mem_inputsAt.1 h)).1
  · intro i hi hc _
    rw [hg]
    by_cases hin : (progOf C κ).gates[i]? = some .input
    · exact List.mem_append_right _ (InputSlot.mem_inputsAt.2 hin)
    · obtain ⟨u, hu⟩ := exists_unitGates P hi hc hin
      exact List.mem_append_left _ (List.mem_flatMap.2 ⟨u, List.mem_finRange u, hu⟩)
  · rw [hb]
    intro b hb'
    rcases List.mem_append.1 hb' with h | h
    · obtain ⟨u, -, rfl⟩ := List.mem_map.1 h
      exact hne u
    · obtain ⟨g, -, rfl⟩ := List.mem_map.1 h
      exact List.cons_ne_nil _ _
  · rw [hb]
    intro b hb'
    apply sizeClass_of_le_max
    rcases List.mem_append.1 hb' with h | h
    · obtain ⟨u, -, rfl⟩ := List.mem_map.1 h
      exact hw u
    · obtain ⟨g, -, rfl⟩ := List.mem_map.1 h
      exact (List.length_filter_le _ _).trans (by simp [sizeClassMax])

end Flock.Bridge
