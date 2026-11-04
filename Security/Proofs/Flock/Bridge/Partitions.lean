import Proofs.Flock.Bridge.Lowers

/-!
# The bridge's partition partitions its program

`partOf_partitions`: at strata that sum to `n`, `partOf P κ ns` partitions `progOf C κ` (`Partition.Partitions`) when
no output of `C` is an input, every unit holds a gate, and every proof block outputs at most `unitClassMax` bits.

* No gate is in two blocks: a block's gates are lowered gates of the wires its unit holds and the positions of the
  inputs it owns (`wire_of_mem_blockOf`), and a gate is one wire's (`block_inj`), which one unit holds or owns
  (`eq_of_mem_blockOf`).
* Every gate but a constant and an unread input is in a block (`mem_blockOf_of`): a held wire's gates are in its unit's
  block, an input the constants don't fix in its owner's, and an input nobody owns that `C` doesn't return is
  `Prog.Unread` (`unread_posOf`).
* A block that outputs at most `unitClassMax` bits has a class that holds them (`unitClass_of_le`).
-/

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

theorem unitClassMax_mem : unitClassMax ∈ unitClasses := by
  have h : unitClassMax / 16 = 2 ^ 14 := by norm_num [unitClassMax]
  unfold unitClasses
  rw [h, Nat.log2_two_pow]
  exact List.mem_map.2 ⟨14, List.mem_range.2 (by norm_num), by norm_num [unitClassMax]⟩

/-- **A block that outputs at most `unitClassMax` bits has a size class** that holds them. -/
theorem unitClass_of_le {Cp : Claims.Sampling.Prog} {u : List ℕ} (h : outputs Cp u ≤ unitClassMax) :
    ∃ w, unitClass Cp u = some w ∧ outputs Cp u ≤ w := by
  unfold unitClass
  cases hf : unitClasses.find? (fun w => decide (outputs Cp u ≤ w)) with
  | none => exact absurd (decide_eq_true h) (List.find?_eq_none.1 hf _ unitClassMax_mem)
  | some w => exact ⟨w, rfl, by simpa using List.find?_some hf⟩

variable {C : FlockSoundness.Audit.Circuit} {κ : Fin C.N → Option Bool}

theorem off_add_lt_length (g : Fin C.N) {j : ℕ} (hj : j < gateSize (C.op g)) :
    off C g.val + j < (progOf C κ).gates.length := by
  have h := gates_off (κ := κ) g hj
  have hl : j < (lower C κ g).length := by rw [length_lower]; exact hj
  rw [List.getElem?_eq_getElem hl] at h
  exact (List.getElem?_eq_some_iff.1 h).1

theorem posOf_input {a : Fin C.N} (h : C.op a = .input) : posOf C a = off C a.val + 0 := by
  rw [posOf_eq, h]
  rfl

/-- **A block's gate is a wire's**: a lowered gate of a wire its unit holds, or the position of an input it owns. -/
theorem wire_of_mem_blockOf {n : ℕ} {P : Partition C n} {u : Fin n} {i : ℕ} (h : i ∈ blockOf P κ u) :
    ∃ g : Fin C.N, ∃ j < gateSize (C.op g), i = off C g.val + j ∧
      (P.unit g = some u ∨ C.op g = .input ∧ owner P g = some u) := by
  rcases mem_blockOf.1 h with h | ⟨g, hg, rfl⟩
  · obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 h
    exact ⟨g, j, hj, rfl, Or.inl hu⟩
  · obtain ⟨hin, -, ho⟩ := mem_insOf.1 hg
    exact ⟨g, 0, by rw [hin]; simp [gateSize], posOf_input hin, Or.inr ⟨hin, ho⟩⟩

/-- **A gate is in one block**: its wire is held or owned by one unit. -/
theorem eq_of_mem_blockOf {n : ℕ} {P : Partition C n} {u v : Fin n} {i : ℕ} (hu : i ∈ blockOf P κ u)
    (hv : i ∈ blockOf P κ v) : u = v := by
  obtain ⟨g, j, hj, rfl, hg⟩ := wire_of_mem_blockOf hu
  obtain ⟨g', j', hj', he, hg'⟩ := wire_of_mem_blockOf hv
  obtain ⟨rfl, -⟩ := block_inj hj hj' he
  rcases hg with hg | ⟨hin, ho⟩ <;> rcases hg' with hg' | ⟨hin', ho'⟩
  · exact Option.some.inj (hg.symm.trans hg')
  · exact absurd (isInput_of_unit hg) (by simp [hin', Op.isInput])
  · exact absurd (isInput_of_unit hg') (by simp [hin, Op.isInput])
  · exact Option.some.inj (ho.symm.trans ho')

theorem nodup_unitGates {n : ℕ} (P : Partition C n) (u : Fin n) : (unitGates P u).Nodup := by
  have hf : ∀ (g : Fin C.N) {i : ℕ},
      i ∈ (if P.unit g = some u then (List.range (gateSize (C.op g))).map (off C g.val + ·) else []) →
        ∃ j < gateSize (C.op g), i = off C g.val + j := by
    intro g i hi
    split_ifs at hi
    · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hi
      exact ⟨j, List.mem_range.1 hj, rfl⟩
    · simp at hi
  unfold unitGates
  refine List.nodup_flatMap.2 ⟨fun g _ => ?_, (List.nodup_finRange C.N).pairwise_of_forall_ne fun g _ g' _ hne => ?_⟩
  · split_ifs
    · exact List.nodup_range.map fun a b h => by simpa using h
    · exact List.nodup_nil
  · dsimp only [Function.onFun]
    refine List.disjoint_left.2 fun i hi hi' => hne ?_
    obtain ⟨j, hj, rfl⟩ := hf g hi
    obtain ⟨j', hj', he⟩ := hf g' hi'
    exact (block_inj hj hj' he).1

theorem nodup_blockOf {n : ℕ} (P : Partition C n) (u : Fin n) : (blockOf P κ u).Nodup := by
  have hins : (insOf P κ u).Nodup := (List.nodup_finRange C.N).filter _
  refine List.nodup_append.2 ⟨nodup_unitGates P u, hins.map posOf_injective, ?_⟩
  rintro i hi b hb rfl
  obtain ⟨a, ha, rfl⟩ := List.mem_map.1 hb
  exact absurd (isInput_of_unit (posOf_mem_unitGates.1 hi)) (by simp [(mem_insOf.1 ha).1, Op.isInput])

/-- **An input nobody owns is unread**, when the constants don't fix it and `C` doesn't return it: a gate that reads
it reads it as an argument of its wire, whose unit would own it. -/
theorem unread_posOf {n : ℕ} {P : Partition C n} {a : Fin C.N} (hin : C.op a = .input) (hk : κ a = none)
    (ho : owner P a = none) (hout : a ∉ C.outputs) : (progOf C κ).Unread (posOf C a) := by
  refine ⟨gates_posOf_input hin hk, fun hm => hout ?_, fun x hx hr => ?_⟩
  · obtain ⟨g, hg, he⟩ := List.mem_map.1 hm
    obtain rfl := posOf_injective he
    simpa using hg
  · obtain ⟨p, hp⟩ := List.getElem?_of_mem hx
    obtain ⟨g, j, hj, rfl, hxj⟩ := exists_block hp
    have hpa := posOf_input hin
    rcases reads_lower_args hxj hr with ⟨a', ha', he⟩ | ⟨rfl, he⟩
    · rw [← posOf_injective he] at ha'
      have hni : (C.op g).isInput = false := by
        cases h : (C.op g).isInput
        · rfl
        · rw [Op.args_of_isInput h] at ha'
          simp at ha'
      cases hu : P.unit g with
      | none => exact absurd ((P.input_iff g).1 hu) (by simp [hni])
      | some v =>
        rw [owner, List.find?_eq_none] at ho
        exact ho v (List.mem_finRange v) (decide_eq_true ⟨g, hu, ha'⟩)
    · have h0 : 0 < gateSize (C.op a) := by rw [hin]; simp [gateSize]
      rcases he with he | he
      · obtain ⟨rfl, -⟩ := block_inj (g := a) (h := g) (j := 0) (k := 0) h0 (by omega) (by omega)
        rw [hin] at hj
        simp [gateSize] at hj
      · obtain ⟨rfl, -⟩ := block_inj (g := a) (h := g) (j := 0) (k := 1) h0 (by omega) (by omega)
        rw [hin] at hj
        simp [gateSize] at hj

/-- **Every gate but a constant and an unread input is in a block**, when no output of `C` is an input. -/
theorem mem_blockOf_of {n : ℕ} {P : Partition C n} (hout : ∀ g ∈ C.outputs, C.op g ≠ .input) {i : ℕ}
    (hi : i < (progOf C κ).gates.length) (hc : ∀ b, (progOf C κ).gates[i]? ≠ some (.const b))
    (hu : ¬ (progOf C κ).Unread i) : ∃ u, i ∈ blockOf P κ u := by
  obtain ⟨g, j, hj, rfl, -⟩ := exists_block (List.getElem?_eq_getElem hi)
  cases hpu : P.unit g with
  | some u => exact ⟨u, mem_blockOf.2 (Or.inl (mem_unitGates.2 ⟨g, hpu, j, hj, rfl⟩))⟩
  | none =>
    have hin := input_of_isInput ((P.input_iff g).1 hpu)
    have hj0 : j = 0 := by
      rw [hin] at hj
      simp [gateSize] at hj
      omega
    subst hj0
    have hp : posOf C g = off C g.val + 0 := posOf_input hin
    cases hk : κ g with
    | some b => exact absurd (by rw [← hp]; exact gates_posOf_const hin hk) (hc b)
    | none =>
      cases ho : owner P g with
      | some u => exact ⟨u, mem_blockOf.2 (Or.inr ⟨g, mem_insOf.2 ⟨hin, hk, ho⟩, hp⟩)⟩
      | none => exact absurd (by rw [← hp]; exact unread_posOf hin hk ho fun hg => hout g hg hin) hu

/-- A block with a gate has a unit gate: its own, or the reader of an input it owns. -/
theorem exists_unit_of_mem_blockOf {n : ℕ} {P : Partition C n} {u : Fin n} {i : ℕ} (h : i ∈ blockOf P κ u) :
    ∃ g, P.unit g = some u := by
  obtain ⟨g, -, -, -, hg | ⟨-, ho⟩⟩ := wire_of_mem_blockOf h
  · exact ⟨g, hg⟩
  · obtain ⟨g', hg', -⟩ := reads_of_owner ho
    exact ⟨g', hg'⟩

theorem blockOf_ne_nil {n : ℕ} {P : Partition C n} {u : Fin n} (h : ∃ g, P.unit g = some u) :
    blockOf P κ u ≠ [] := by
  obtain ⟨g, hg⟩ := h
  exact List.ne_nil_of_mem (mem_blockOf.2 (Or.inl (mem_unitGates.2 ⟨g, hg, 0, one_le_gateSize _, rfl⟩)))

/-- **`partOf P κ ns` partitions `progOf C κ`**, at strata that sum to `n`, when no output of `C` is an input, every
unit holds a gate, and every proof block outputs at most `unitClassMax` bits. -/
theorem partOf_partitions {n : ℕ} (P : Partition C n) {ns : List ℕ} (hns : ns.sum = n)
    (hout : ∀ g ∈ C.outputs, C.op g ≠ .input) (hne : ∀ u, ∃ g, P.unit g = some u)
    (hcls : ∀ u, outputs (progOf C κ) (blockOf P κ u) ≤ unitClassMax) :
    (partOf P κ ns).Partitions (progOf C κ) := by
  have hb := blocks_partOf (κ := κ) P hns
  have hmem : ∀ {i}, i ∈ (partOf P κ ns).gates ↔ ∃ u, i ∈ blockOf P κ u := by
    intro i
    simp [Claims.Sampling.Partition.gates, hb]
  unfold Claims.Sampling.Partition.Partitions
  refine ⟨?_, fun i hi => ?_, fun i hi hc hu => hmem.2 (mem_blockOf_of hout hi hc hu), fun b hb' => ?_,
    fun b hb' => ?_⟩
  · rw [Claims.Sampling.Partition.gates, hb, List.nodup_flatten]
    refine ⟨fun b hb' => ?_,
      List.pairwise_map.2 ((List.nodup_finRange n).pairwise_of_forall_ne fun u _ v _ huv => ?_)⟩
    · obtain ⟨u, -, rfl⟩ := List.mem_map.1 hb'
      exact nodup_blockOf P u
    · exact List.disjoint_left.2 fun i hu hv => huv (eq_of_mem_blockOf hu hv)
  · obtain ⟨u, hu⟩ := hmem.1 hi
    obtain ⟨g, j, hj, rfl, -⟩ := wire_of_mem_blockOf hu
    exact off_add_lt_length g hj
  · rw [hb] at hb'
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hb'
    exact blockOf_ne_nil (hne u)
  · rw [hb] at hb'
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hb'
    exact unitClass_of_le (hcls u)

/-! ## What a block outputs -/

theorem lower_ne_input {g : Fin C.N} (hni : (C.op g).isInput = false) : ∀ x ∈ lower C κ g, x ≠ .input := by
  unfold lower
  cases h : C.op g <;> simp_all [Op.isInput]

/-- **A unit's gate is committed only at a returned wire's position**, when no unit reads another unit's gate: a row's
XORs are interior (`boundary_inner`), the wire's last gate is no input, and a gate outside the unit that reads it reads
it as an argument of its own wire, which is then in the unit, or an input's, which reads nothing. -/
theorem output_of_boundary {n : ℕ} {P : Partition C n} {u : Fin n}
    (hread : ∀ (h g : Fin C.N) (v v' : Fin n), P.unit h = some v → g ∈ (C.op h).args → P.unit g = some v' → v' = v)
    {i : ℕ} (hi : i ∈ unitGates P u) (hB : (progOf C κ).Boundary (unitGates P u) i = true) :
    ∃ g : Fin C.N, P.unit g = some u ∧ g ∈ C.outputs ∧ posOf C g = i := by
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hi
  by_cases hjl : j + 1 < gateSize (C.op g)
  · rw [boundary_inner hu hjl] at hB
    exact absurd hB (by simp)
  have hpos : posOf C g = off C g.val + j := by rw [posOf_eq]; omega
  refine ⟨g, hu, ?_, hpos⟩
  rw [← hpos] at hB
  have hni := isInput_of_unit hu
  have h1 := one_le_gateSize (C.op g)
  simp only [Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq, List.any_eq_true,
    List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true', decide_eq_false_iff_not] at hB
  rcases hB with (hout | hin) | ⟨g', -, hg'u, hrd⟩
  · simp only [progOf, List.mem_map] at hout
    obtain ⟨a, ha, he⟩ := hout
    obtain rfl := posOf_injective he
    simpa using ha
  · exfalso
    rw [posOf_eq, gates_off g (j := gateSize (C.op g) - 1) (by omega)] at hin
    exact lower_ne_input hni _ (List.mem_of_getElem? hin) rfl
  · exfalso
    cases hx : (progOf C κ).gates[g']? with
    | none => simp [hx] at hrd
    | some x =>
      simp only [hx, Option.map_some, Option.getD_some] at hrd
      obtain ⟨h, k, hk, rfl, hxk⟩ := exists_block hx
      rcases reads_lower_args hxk hrd with ⟨a, ha, he⟩ | ⟨rfl, he⟩
      · obtain rfl := posOf_injective he
        cases hv : P.unit h with
        | none =>
          rw [Op.args_of_isInput ((P.input_iff h).1 hv)] at ha
          simp at ha
        | some v =>
          obtain rfl := hread h _ v u hv ha hu
          exact hg'u (mem_unitGates.2 ⟨h, hv, k, hk, rfl⟩)
      · rw [posOf_eq] at he
        rcases he with he | he
        · obtain ⟨rfl, hj0⟩ := block_inj (g := g) (h := h) (j := gateSize (C.op g) - 1) (k := 0) (by omega)
            (by omega) (by simpa using he)
          omega
        · obtain ⟨rfl, hj0⟩ := block_inj (g := g) (h := h) (j := gateSize (C.op g) - 1) (k := 1) (by omega)
            (by omega) he
          omega

/-- **A block outputs at most its unit's returned wires**, when no unit reads another unit's gate
(`output_of_boundary`): the inputs it owns are input gates, which `outputs` leaves out. -/
theorem outputs_blockOf_le {n : ℕ} {P : Partition C n} (u : Fin n)
    (hread : ∀ (h g : Fin C.N) (v v' : Fin n), P.unit h = some v → g ∈ (C.op h).args → P.unit g = some v' → v' = v) :
    outputs (progOf C κ) (blockOf P κ u) ≤
      (Finset.univ.filter fun g => P.unit g = some u ∧ g ∈ C.outputs).card := by
  have hf : (blockOf P κ u).filter (fun i => (progOf C κ).Boundary (blockOf P κ u) i &&
        decide ((progOf C κ).gates[i]? ≠ some .input)) =
      (blockOf P κ u).filter (fun i => (progOf C κ).Boundary (unitGates P u) i &&
        decide ((progOf C κ).gates[i]? ≠ some .input)) :=
    List.filter_congr fun i _ => by rw [blockOf, boundary_append_inputs inputs_blockOf]
  have hins : ((insOf P κ u).map (posOf C)).filter (fun i => (progOf C κ).Boundary (unitGates P u) i &&
        decide ((progOf C κ).gates[i]? ≠ some .input)) = [] :=
    List.filter_eq_nil_iff.2 fun i hi => by simp [inputs_blockOf i hi]
  unfold outputs
  rw [hf, blockOf, List.filter_append, hins, List.append_nil]
  have hsub : (unitGates P u).filter (fun i => (progOf C κ).Boundary (unitGates P u) i &&
        decide ((progOf C κ).gates[i]? ≠ some .input)) ⊆
      (Finset.univ.filter fun g => P.unit g = some u ∧ g ∈ C.outputs).toList.map (posOf C) := by
    intro i hi
    rw [List.mem_filter, Bool.and_eq_true] at hi
    obtain ⟨g, hu, ho, rfl⟩ := output_of_boundary hread hi.1 hi.2.1
    exact List.mem_map.2 ⟨g, by simp [hu, ho], rfl⟩
  exact (((nodup_unitGates P u).filter _).length_le_of_subset hsub).trans (by simp)

end Flock.Bridge
