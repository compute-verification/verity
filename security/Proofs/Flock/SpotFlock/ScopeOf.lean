module

public import Proofs.Flock.SpotFlock.Facts
public import Proofs.Flock.Bridge.Members
public import Proofs.Flock.Bridge.Slots

/-!
# The two-stage scope at the members

`scopeOk` decides the two-stage scope (`scopeOk_sound`); `scopeOk_complete` is the converse. Here the scope of the
members' programs side by side (`bitsProgs`), their proof blocks (`membersPart`) and their replay partition
(`filesReplay`) is reduced to conditions on each member and on the replay layout:

* `partitions_blockOf`: a circuit's units' proof blocks (`blockOf`: a unit's lowered gates, then the inputs it owns)
  partition its lowering (`Partition.Partitions`), where every unit has a lowered gate, no output is an input and each
  block commits at most `unitClassMax` values; `partitions_keyProg` is a member's, its key program's units' blocks
  (`Spot.Bits.blockOf`) partitioning its program (`memberProg`).
* `SitsAt.unread`, `SitsAt.committed_map`, `SitsAt.unitClass_map`: a program sitting in another keeps its unread inputs,
  and its blocks commit as many values and have the same size class, `b` later.
* `topological_bitsProgs`, `partitions_membersPart`: the members side by side are topologically ordered and partitioned
  by the members' proof blocks, when each member's program is.
* `nodup_gates_filesReplay`, `isSome_replayOf_filesReplay`: the replay partition's gates are distinct where the layout
  fits the members' files (`FilesFit`), and every proof block replays in a request where each member's units are its
  requests' (`FilesFill`); slot `p < n_t` of request `ρ` in member `t` is proof block `filesBlockOf ms ns ρ t p`
  (`replayOf_filesReplay_iff_filesBlockOf`), distinct for distinct `p` (`injective_filesBlockOf`).
* `scopeOk_members`, `two_eq_filesReplay`, `isSome_two`: `scopeOk` accepts the members, and `two` returns their replay
  design, for members that fit (`MemberFits`), a layout that fits and fills their files, a well-formed design, and
  templates whose reference statements read as members' (`templateOk`).
-/

@[expose] public section

namespace Claims.Sampling

/-- **A replay partition of unions of proof blocks has distinct gates**: block `ρ` is the proof blocks of `Pt` at the
indices `I ρ` together, `Pt`'s gates distinct, each `I ρ` without repeats and the index sets disjoint. -/
theorem Partition.nodup_gates_union {Pt Pρ : Partition} {I : ℕ → List ℕ} (hnd : Pt.gates.Nodup)
    (hρ : ∀ ρ, ρ < Pρ.blocks.length → Pρ.blocks[ρ]? = some ((I ρ).flatMap fun w => Pt.blocks.getD w []))
    (hI : ∀ ρ, ρ < Pρ.blocks.length → (I ρ).Nodup)
    (hdisj : ∀ ρ ρ', ρ < Pρ.blocks.length → ρ' < Pρ.blocks.length → ρ ≠ ρ' → ∀ w ∈ I ρ, w ∉ I ρ') :
    Pρ.gates.Nodup := by
  have hblk : ∀ w, (Pt.blocks.getD w []).Nodup := fun w => by
    rw [List.getD_eq_getElem?_getD]
    rcases Nat.lt_or_ge w Pt.blocks.length with hw | hw
    · rw [List.getElem?_eq_getElem hw, Option.getD_some]
      exact (List.nodup_flatten.1 (show Pt.blocks.flatten.Nodup from hnd)).1 _ (List.getElem_mem hw)
    · rw [List.getElem?_eq_none hw, Option.getD_none]
      exact List.nodup_nil
  have hdis : ∀ w w', w ≠ w' → (Pt.blocks.getD w []).Disjoint (Pt.blocks.getD w' []) := fun w w' hww' => by
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD]
    rcases Nat.lt_or_ge w Pt.blocks.length with hw | hw
    · rcases Nat.lt_or_ge w' Pt.blocks.length with hw' | hw'
      · rw [List.getElem?_eq_getElem hw, List.getElem?_eq_getElem hw', Option.getD_some, Option.getD_some]
        exact Partition.disjoint_blocks hnd hww' (List.getElem?_eq_getElem hw) (List.getElem?_eq_getElem hw')
      · rw [List.getElem?_eq_none hw', Option.getD_none]
        intro a _ ha
        simp at ha
    · rw [List.getElem?_eq_none hw, Option.getD_none]
      intro a ha
      simp at ha
  have hU : ∀ ρ (hρl : ρ < Pρ.blocks.length), Pρ.blocks[ρ] = (I ρ).flatMap fun w => Pt.blocks.getD w [] :=
    fun ρ hρl => Option.some.inj ((List.getElem?_eq_getElem hρl).symm.trans (hρ ρ hρl))
  refine List.nodup_flatten.2 ⟨fun x hx => ?_, List.pairwise_iff_getElem.2 fun ρ ρ' hρl hρl' hlt => ?_⟩
  · obtain ⟨ρ, hρl, rfl⟩ := List.getElem_of_mem hx
    rw [hU ρ hρl]
    exact List.nodup_flatMap.2 ⟨fun w _ => hblk w,
      (hI ρ hρl).pairwise_of_forall_ne fun w _ w' _ hww' => hdis w w' hww'⟩
  · rw [hU ρ hρl, hU ρ' hρl']
    intro g hg hg'
    obtain ⟨w, hw, hgw⟩ := List.mem_flatMap.1 hg
    obtain ⟨w', hw', hgw'⟩ := List.mem_flatMap.1 hg'
    exact hdis w w' (fun h => hdisj ρ ρ' hρl hρl' (Nat.ne_of_lt hlt) w hw (h ▸ hw')) hgw hgw'

end Claims.Sampling

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

/-! ## A circuit's units' proof blocks partition its lowering -/

section blocks

variable {C : Verity.Boolean.Circuit} {κ : Fin C.N → Option Bool} {n : ℕ}

theorem nodup_blockOf (P : FlockSoundness.Audit.Partition C n) (u : Fin n) : (blockOf P κ u).Nodup := by
  refine List.nodup_append.2 ⟨nodup_unitGates P u, ((List.nodup_finRange C.N).filter _).map posOf_injective, ?_⟩
  intro a ha b hb he
  obtain ⟨g, hg, rfl⟩ := List.mem_map.1 hb
  subst he
  exact progOf_ne_input_of_unit (κ := κ) ha (gates_posOf_input (mem_insOf.1 hg).1 (mem_insOf.1 hg).2.1)

theorem disjoint_blockOf (P : FlockSoundness.Audit.Partition C n) {u v : Fin n} (huv : u ≠ v) :
    (blockOf P κ u).Disjoint (blockOf P κ v) := by
  intro i hi hj
  rcases mem_blockOf.1 hi with hi | ⟨g, hg, rfl⟩ <;> rcases mem_blockOf.1 hj with hj | ⟨g', hg', hj⟩
  · exact disjoint_unitGates P huv hi hj
  · subst hj
    exact progOf_ne_input_of_unit (κ := κ) hi (gates_posOf_input (mem_insOf.1 hg').1 (mem_insOf.1 hg').2.1)
  · exact progOf_ne_input_of_unit (κ := κ) hj (gates_posOf_input (mem_insOf.1 hg).1 (mem_insOf.1 hg).2.1)
  · obtain rfl := posOf_injective hj
    exact huv (Option.some.inj ((mem_insOf.1 hg).2.2.symm.trans (mem_insOf.1 hg').2.2))

theorem lt_length_blockOf {P : FlockSoundness.Audit.Partition C n} {u : Fin n} {i : ℕ} (hi : i ∈ blockOf P κ u) :
    i < (progOf C κ).gates.length := by
  rcases mem_blockOf.1 hi with hi | ⟨g, -, rfl⟩
  · exact lt_length_of_unit hi
  · exact Members.posOf_lt_length g

theorem length_blockOf_le (P : FlockSoundness.Audit.Partition C n) (u : Fin n) :
    (blockOf P κ u).length ≤ (progOf C κ).gates.length := by
  rw [← List.toFinset_card_of_nodup (nodup_blockOf P u), ← Finset.card_range (progOf C κ).gates.length]
  exact Finset.card_le_card fun i hi => Finset.mem_range.2 (lt_length_blockOf (List.mem_toFinset.1 hi))

/-- An input gate of the lowering is an input wire's position, a wire the constants leave free. -/
theorem input_pos {i : ℕ} (h : (progOf C κ).gates[i]? = some .input) :
    ∃ a : Fin C.N, i = posOf C a ∧ C.op a = .input ∧ κ a = none := by
  obtain ⟨g, j, hj, rfl, hx⟩ := exists_block h
  by_cases hin : C.op g = .input
  · have hj0 : j = 0 := by rw [hin] at hj; simp [gateSize] at hj; omega
    subst hj0
    refine ⟨g, by rw [posOf_eq, hin]; rfl, hin, ?_⟩
    unfold lower at hx
    rw [hin] at hx
    cases hk : κ g <;> simp [hk] at hx ⊢
  · exfalso
    have hni : (C.op g).isInput = false := by cases h : C.op g <;> simp_all [Verity.Boolean.Op.isInput]
    exact lower_ne_input hni _ (List.mem_of_getElem? hx) rfl

/-- A wire some gate reads has an owner. -/
theorem isSome_owner (P : FlockSoundness.Audit.Partition C n) {a g : Fin C.N} (ha : a ∈ (C.op g).args) :
    (owner P a).isSome := by
  have hg : (C.op g).isInput = false := by
    cases h : (C.op g).isInput
    · rfl
    · rw [Verity.Boolean.Op.args_of_isInput h] at ha
      simp at ha
  obtain ⟨u, hu⟩ := Option.ne_none_iff_exists'.1 fun h => by
    rw [(P.input_iff g).1 h] at hg
    simp at hg
  unfold owner
  exact List.find?_isSome.2 ⟨u, List.mem_finRange u, decide_eq_true ⟨g, hu, ha⟩⟩

/-- **Every gate of the lowering but a constant or an unread input is in some unit's block**, where no output is an
input: an input gate some gate reads is its owner's, every other gate its unit's. -/
theorem exists_blockOf (P : FlockSoundness.Audit.Partition C n) (hout : ∀ o ∈ C.outputs, (C.op o).isInput = false)
    {i : ℕ} (hi : i < (progOf C κ).gates.length) (hc : ∀ b, (progOf C κ).gates[i]? ≠ some (.const b))
    (hun : ¬ (progOf C κ).Unread i) : ∃ u, i ∈ blockOf P κ u := by
  by_cases hin : (progOf C κ).gates[i]? = some .input
  · obtain ⟨a, rfl, ha, hk⟩ := input_pos hin
    have hown : ∃ u, owner P a = some u := by
      by_cases ho : posOf C a ∈ (progOf C κ).outs
      · exfalso
        simp only [progOf, List.mem_map, List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq] at ho
        obtain ⟨o, ho, he⟩ := ho
        obtain rfl := posOf_injective he
        have := hout _ ho
        simp [ha, Verity.Boolean.Op.isInput] at this
      · by_contra hno
        apply hun
        unfold Prog.Unread
        refine ⟨hin, ho, fun x hx hr => ?_⟩
        obtain ⟨q, hq⟩ := List.mem_iff_getElem?.1 hx
        obtain ⟨g, j, hj, rfl, hgx⟩ := exists_block hq
        rcases reads_lower_args hgx hr with ⟨a', ha', he⟩ | ⟨rfl, he | he⟩
        · obtain rfl := posOf_injective he
          exact hno (Option.isSome_iff_exists.1 (isSome_owner P ha'))
        · exact off_ne_posOf (g := g) (a := a) (j := 0) (by omega) (by rw [Nat.add_zero]; exact he.symm)
        · exact off_ne_posOf (g := g) (a := a) (j := 1) (by omega) he.symm
    obtain ⟨u, hu⟩ := hown
    exact ⟨u, mem_blockOf.2 (Or.inr ⟨a, mem_insOf.2 ⟨ha, hk, hu⟩, rfl⟩)⟩
  · obtain ⟨u, hu⟩ := exists_unitGates P hi hc hin
    exact ⟨u, mem_blockOf.2 (Or.inl hu)⟩

/-- **A circuit's units' proof blocks partition its lowering**, where every unit has a lowered gate, no output is an
input, and each block commits at most `unitClassMax` values. -/
theorem partitions_blockOf (P : FlockSoundness.Audit.Partition C n) (κ : Fin C.N → Option Bool)
    (hne : ∀ u, unitGates P u ≠ []) (hout : ∀ o ∈ C.outputs, (C.op o).isInput = false)
    (hw : ∀ u, committed (progOf C κ) (blockOf P κ u) ≤ unitClassMax) :
    (⟨[[(List.finRange n).map (blockOf P κ)]]⟩ : Claims.Sampling.Partition).Partitions (progOf C κ) := by
  have hb : (⟨[[(List.finRange n).map (blockOf P κ)]]⟩ : Claims.Sampling.Partition).blocks =
      (List.finRange n).map (blockOf P κ) := by
    simp [Partition.blocks]
  have hg : (⟨[[(List.finRange n).map (blockOf P κ)]]⟩ : Claims.Sampling.Partition).gates =
      (List.finRange n).flatMap (blockOf P κ) := by
    simp [Partition.gates, hb, List.flatMap]
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [hg]
    exact List.nodup_flatMap.2 ⟨fun u _ => nodup_blockOf P u,
      (List.nodup_finRange n).pairwise_of_forall_ne fun _ _ _ _ huv => disjoint_blockOf P huv⟩
  · rw [hg]
    intro i hi
    obtain ⟨u, -, hu⟩ := List.mem_flatMap.1 hi
    exact lt_length_blockOf hu
  · intro i hi hc hun
    rw [hg]
    obtain ⟨u, hu⟩ := exists_blockOf P hout hi hc hun
    exact List.mem_flatMap.2 ⟨u, List.mem_finRange u, hu⟩
  · rw [hb]
    intro x hx
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hx
    intro h
    exact hne u (List.append_eq_nil_iff.1 h).1
  · rw [hb]
    intro x hx
    obtain ⟨u, -, rfl⟩ := List.mem_map.1 hx
    exact unitClass_of_le (hw u)

end blocks

/-! ## A member's key program -/

section keyProg

open FlockSoundness FlockSoundness.Discharge

theorem isSome_row? {kp : Spot.Bits.KeyProg} {g : ℕ} (h1 : kp.inputs ≤ g) (h2 : g < kp.size) :
    (kp.row? g).isSome := by
  have h3 : g - kp.inputs < kp.rows.size := by unfold Spot.Bits.KeyProg.size at h2; omega
  rw [Spot.Bits.KeyProg.row?, ite_eq_right (show ¬ g < kp.inputs by omega), Array.getElem?_eq_getElem h3]
  rfl

/-- **A key program's output is a row**: each is a copy row of a unit's instance. -/
theorem isSome_row?_of_mem_outs {c : _root_.Flock.Circuit} {n g : ℕ} (hg : g ∈ (Spot.Bits.keyProg c n).outs) :
    ((Spot.Bits.keyProg c n).row? g).isSome := by
  rw [keyProg_outs, List.mem_eraseDups, List.mem_mergeSort] at hg
  simp only [List.mem_flatMap, List.mem_range, List.mem_filterMap] at hg
  obtain ⟨u, -, k, -, j, -, t, -, h⟩ := hg
  simp only [Spot.Bits.outGate, Spot.Bits.rowGate] at h
  split at h
  · rename_i hc
    simp only [Bool.and_eq_true, decide_eq_true_eq, and_assoc] at hc
    obtain ⟨hun, hk, -, -, h1, h2⟩ := hc
    obtain rfl := Option.some.inj h
    have bM : (bitsRows c).nComp = c.upv * ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) := rfl
    have e : (Spot.Bits.Rows.ofNet c.unit).stack c.upv = bitsRows c := rfl
    have hk' := Nat.mul_le_mul_right ((Spot.Bits.Rows.ofNet c.unit).nComp + 1) (show k + 1 ≤ c.upv from hk)
    have hu' := Nat.mul_le_mul_right (bitsRows c).nComp (show u + 1 ≤ n from hun)
    rw [Nat.add_one_mul] at hk' hu'
    rw [e]
    refine isSome_row? ?_ ?_
    · rw [keyProg_inputs]
      omega
    · rw [keyProg_size]
      omega
  · simp at h

theorem isInput_outputs_keyProg {c : _root_.Flock.Circuit} {n : ℕ} {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32) :
    ∀ o ∈ ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)).outputs,
      (((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)).op o).isInput = false := by
  intro o ho
  have hL := laysOut_keyProg (n := n) (hU := hU) hu
  have hmem : o.val ∈ (Spot.Bits.keyProg c n).outs := by
    rw [hL.outs]
    exact List.mem_map.2 ⟨o, List.mem_filter.2 ⟨List.mem_finRange o, decide_eq_true ho⟩, rfl⟩
  obtain ⟨⟨a, b⟩, hab⟩ := Option.isSome_iff_exists.1 (isSome_row?_of_mem_outs hmem)
  obtain ⟨A, B, hop, -, -⟩ := hL.row o a b hab
  rw [hop]
  rfl

theorem unitGates_keyProg_ne_nil {c : _root_.Flock.Circuit} {n : ℕ} {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32)
    (hc : 0 < (bitsRows c).nComp) (u : Fin n) :
    unitGates ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) u ≠ [] := by
  have hR : RowsAgree (bitsRows c) (Placed.circuitRows c hU) := rowsAgree_circuitRows c hU hu
  have hc' : 0 < (Placed.circuitRows c hU).nComp := by rw [← hR.2.1]; exact hc
  have hlt : 2 + n * (Placed.circuitRows c hU).nIn + u.val * (Placed.circuitRows c hU).nComp + 0 <
      Layout.keyN c hU n := by
    have := Nat.mul_le_mul_right (bitsRows c).nComp (show u.val + 1 ≤ n from u.isLt)
    rw [Nat.add_one_mul] at this
    rw [keyN_eq hR, ← hR.1, ← hR.2.1]
    omega
  have hg := keyProg_unit_row (hU := hU) u.isLt ⟨0, hc'⟩ ⟨_, hlt⟩ rfl
  intro h
  have hmem := (posOf_mem_unitGates (P := (Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (u := u)
    (a := ⟨_, hlt⟩)).2 hg
  rw [h] at hmem
  simp at hmem

/-- **A member's key program's units' blocks partition its program**, where its rows are built (`Built`), it has a
computed row and its program has at most `unitClassMax` gates. -/
theorem partitions_keyProg (c : _root_.Flock.Circuit) (n : ℕ) {hU : Built c.unit} (hu : c.unit.unitLog ≤ 32)
    (hc : 0 < (bitsRows c).nComp) (hw : (Properties.Spot.memberProg c n).gates.length ≤ unitClassMax) :
    (⟨[[(List.range n).map fun u => Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u]]⟩ :
      Claims.Sampling.Partition).Partitions (Properties.Spot.memberProg c n) := by
  have hlen : (progOf ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) (constsOf c hU n)).gates.length ≤
      unitClassMax := by
    rw [progOf_keyProg hu]
    exact hw
  have hP := partitions_blockOf ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (constsOf c hU n)
    (unitGates_keyProg_ne_nil hu hc) (isInput_outputs_keyProg hu)
    fun u => (List.length_filter_le _ _).trans ((length_blockOf_le _ u).trans hlen)
  have e : (List.finRange n).map (blockOf ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n))
      (constsOf c hU n)) = (List.range n).map fun u =>
        Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u := by
    refine List.ext_getElem (by simp) fun i h1 h2 => ?_
    rw [List.getElem_map, List.getElem_map, blockOf_bitsKeyProg hu]
    simp
  rw [e, progOf_keyProg hu] at hP
  exact hP

end keyProg

end Flock.Bridge

namespace Flock.Bridge.Members

open Claims.Sampling Properties.Spot

/-! ## A program sitting in another -/

theorem blocks_single (l : List (List ℕ)) : (⟨[[l]]⟩ : Partition).blocks = l := by
  simp [Partition.blocks]

theorem gates_single (l : List (List ℕ)) : (⟨[[l]]⟩ : Partition).gates = l.flatten := by
  rw [Partition.gates, blocks_single]

/-- An unread input of `P` is an unread input of `Q`, `b` later. -/
theorem SitsAt.unread {Q P : Prog} {b : ℕ} (hA : SitsAt Q b P) {p : ℕ} (hp : p < P.gates.length)
    (h : P.Unread p) : Q.Unread (b + p) := by
  unfold Prog.Unread at h ⊢
  obtain ⟨hin, hout, hrd⟩ := h
  refine ⟨by simp [hA.gates p hp, hin, shiftGate], fun ho => hout ((hA.outs p hp).1 ho), fun x hx hr => ?_⟩
  obtain ⟨q, hq⟩ := List.mem_iff_getElem?.1 hx
  by_cases hlo : b ≤ q ∧ q < b + P.gates.length
  · obtain ⟨q', rfl⟩ : ∃ q', q = b + q' := ⟨q - b, by omega⟩
    have hq' : q' < P.gates.length := by omega
    rw [hA.gates_eq hq', Option.some.injEq] at hq
    subst hq
    rw [reads_shiftGate, mem_map_add] at hr
    exact hrd _ (List.getElem_mem hq') hr
  · have := hA.closed q x hq (by omega) (b + p) hr
    omega

/-- `Q` commits as many of gates `u` `b` later as `P` commits of `u`. -/
theorem SitsAt.committed_map {Q P : Prog} {b : ℕ} (hA : SitsAt Q b P) {u : List ℕ}
    (hu : ∀ g ∈ u, g < P.gates.length) : committed Q (u.map (b + ·)) = committed P u := by
  unfold committed
  rw [List.filter_map, List.length_map]
  congr 1
  exact List.filter_congr fun g hg => hA.boundary u (hu g hg)

theorem SitsAt.unitClass_map {Q P : Prog} {b : ℕ} (hA : SitsAt Q b P) {u : List ℕ}
    (hu : ∀ g ∈ u, g < P.gates.length) : unitClass Q (u.map (b + ·)) = unitClass P u := by
  simp only [unitClass, hA.committed_map hu]

/-! ## The members side by side -/

theorem topological_sideBySide : ∀ ps : List Prog, (∀ p ∈ ps, p.Topological) → (sideBySide ps).Topological
  | [], _ => ⟨fun g x hx => by simp [sideBySide] at hx, fun o ho => by simp [sideBySide] at ho⟩
  | p :: ps, htop => by
    have hp := htop p List.mem_cons_self
    have ih := topological_sideBySide ps fun q hq => htop q (List.mem_cons_of_mem _ hq)
    refine ⟨fun g x hx i hi => ?_, fun o ho => ?_⟩
    · rw [sideBySide_gates] at hx
      by_cases hg : g < p.gates.length
      · rw [List.getElem?_append_left hg] at hx
        exact hp.1 g x hx i hi
      · rw [List.getElem?_append_right (Nat.le_of_not_lt hg), List.getElem?_map] at hx
        obtain ⟨y, hy, rfl⟩ := Option.map_eq_some_iff.1 hx
        rw [reads_shiftGate] at hi
        obtain ⟨j, hj, rfl⟩ := List.mem_map.1 hi
        have := ih.1 _ y hy j hj
        omega
    · rw [sideBySide_gates, List.length_append, List.length_map]
      rw [sideBySide_outs, List.mem_append] at ho
      rcases ho with ho | ho
      · have := hp.2 o ho
        omega
      · obtain ⟨j, hj, rfl⟩ := List.mem_map.1 ho
        have := ih.2 j hj
        omega

/-- Below `f N` and from `f 0`, `v` lies in some step `[f t, f (t + 1))`, `t < N`. -/
theorem exists_step (f : ℕ → ℕ) {v : ℕ} (h0 : f 0 ≤ v) : ∀ N, v < f N → ∃ t < N, f t ≤ v ∧ v < f (t + 1)
  | 0, h => absurd h (Nat.not_lt.2 h0)
  | N + 1, h => by
    by_cases hN : f N ≤ v
    · exact ⟨N, Nat.lt_add_one N, hN, h⟩
    · obtain ⟨t, ht, h1, h2⟩ := exists_step f h0 N (Nat.lt_of_not_le hN)
      exact ⟨t, by omega, h1, h2⟩

theorem length_sideBySide : ∀ ps : List Prog, (sideBySide ps).gates.length = (ps.map (·.gates.length)).sum
  | [] => rfl
  | p :: ps => by
    rw [sideBySide_gates, List.length_append, List.length_map, length_sideBySide ps, List.map_cons, List.sum_cons]

theorem offsetOf_length (ps : List Prog) : offsetOf ps ps.length = (ps.map (·.gates.length)).sum := by
  simp [offsetOf, List.take_of_length_le]

variable (ms : Spot.Bits.Members)

theorem topological_bitsProgs (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological) :
    (bitsProgs ms).Topological :=
  topological_sideBySide _ fun p hp => by
    obtain ⟨ci, hci, rfl⟩ := List.mem_map.1 hp
    exact htop ci hci

theorem length_bitsProgs : (bitsProgs ms).gates.length = Spot.Bits.base ms ms.length := by
  rw [← offsetOf_bitsProgs, bitsProgs, length_sideBySide]
  have := offsetOf_length (ms.map fun ci => memberProg ci.1 ci.2.size)
  rw [List.length_map] at this
  exact this.symm

theorem memberBlocks_eq {m : ℕ} (hm : m < ms.length) :
    memberBlocks ms m = ((List.range ms[m].2.size).map fun u => Spot.Bits.blockOf
      (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size) u).map
        (·.map (Spot.Bits.base ms m + ·)) := by
  simp [memberBlocks, List.getElem?_eq_getElem hm, List.map_map, Function.comp_def]

/-- **The members' proof blocks partition the members side by side**, when each member's key program's units' blocks
partition its program. -/
theorem partitions_membersPart (htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological)
    (hpart : ∀ ci ∈ ms, (⟨[[(List.range ci.2.size).map fun u =>
      Spot.Bits.blockOf (Spot.Bits.keyProg ci.1 ci.2.size) (Spot.Bits.consts ci.1 ci.2.size) u]]⟩ :
        Partition).Partitions (memberProg ci.1 ci.2.size)) :
    (membersPart ms).Partitions (bitsProgs ms) := by
  have hA := sitsAt_bitsProgs ms htop
  have hlt : ∀ m (hm : m < ms.length), ∀ y ∈ (List.range ms[m].2.size).map (fun u => Spot.Bits.blockOf
      (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size) u),
      ∀ p ∈ y, p < (memberProg ms[m].1 ms[m].2.size).gates.length := fun m hm y hy p hp => by
    obtain ⟨-, h, -, -, -⟩ := hpart ms[m] (List.getElem_mem hm)
    rw [gates_single] at h
    exact h p (List.mem_flatten.2 ⟨y, hy, hp⟩)
  have hmemb : ∀ {x}, x ∈ (membersPart ms).blocks ↔ ∃ m, ∃ hm : m < ms.length, ∃ y ∈ (List.range ms[m].2.size).map
      (fun u => Spot.Bits.blockOf (Spot.Bits.keyProg ms[m].1 ms[m].2.size) (Spot.Bits.consts ms[m].1 ms[m].2.size) u),
      x = y.map (Spot.Bits.base ms m + ·) := by
    intro x
    rw [blocks_membersPart, List.mem_flatMap]
    constructor
    · rintro ⟨m, hm, hx⟩
      have hm' := List.mem_range.1 hm
      rw [memberBlocks_eq ms hm'] at hx
      obtain ⟨y, hy, rfl⟩ := List.mem_map.1 hx
      exact ⟨m, hm', y, hy, rfl⟩
    · rintro ⟨m, hm, y, hy, rfl⟩
      exact ⟨m, List.mem_range.2 hm, by rw [memberBlocks_eq ms hm]; exact List.mem_map.2 ⟨y, hy, rfl⟩⟩
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · refine List.nodup_flatten.2 ⟨fun x hx => ?_, ?_⟩
    · obtain ⟨m, hm, y, hy, rfl⟩ := hmemb.1 hx
      obtain ⟨hnd, -, -, -, -⟩ := hpart ms[m] (List.getElem_mem hm)
      rw [gates_single] at hnd
      exact ((List.nodup_flatten.1 hnd).1 y hy).map (add_right_injective _)
    · rw [blocks_membersPart, List.pairwise_flatMap]
      refine ⟨fun m hm => ?_, List.nodup_range.pairwise_of_forall_ne fun m hm m' hm' hne => ?_⟩
      · have hm1 := List.mem_range.1 hm
        obtain ⟨hnd, -, -, -, -⟩ := hpart ms[m] (List.getElem_mem hm1)
        rw [gates_single] at hnd
        rw [memberBlocks_eq ms hm1]
        refine List.pairwise_map.2 ((List.nodup_flatten.1 hnd).2.imp fun {y y'} hyy' => ?_)
        intro z hz hz'
        obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hz
        obtain ⟨p', hp', he⟩ := List.mem_map.1 hz'
        obtain rfl : p' = p := by omega
        exact hyy' hp hp'
      · intro x hx y hy z hzx hzy
        have hm1 := List.mem_range.1 hm
        have hm1' := List.mem_range.1 hm'
        rw [memberBlocks_eq ms hm1] at hx
        rw [memberBlocks_eq ms hm1'] at hy
        obtain ⟨x', hx', rfl⟩ := List.mem_map.1 hx
        obtain ⟨y', hy', rfl⟩ := List.mem_map.1 hy
        obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hzx
        obtain ⟨p', hp', he⟩ := List.mem_map.1 hzy
        have h1 := hlt m hm1 x' hx' p hp
        have h2 := hlt m' hm1' y' hy' p' hp'
        rcases Nat.lt_or_gt_of_ne hne with h | h
        · have := base_mono ms (show m + 1 ≤ m' by omega)
          rw [base_succ ms hm1] at this
          omega
        · have := base_mono ms (show m' + 1 ≤ m by omega)
          rw [base_succ ms hm1'] at this
          omega
  · intro g hg
    obtain ⟨x, hx, hgx⟩ := List.mem_flatten.1 hg
    obtain ⟨m, hm, y, hy, rfl⟩ := hmemb.1 hx
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hgx
    exact (hA m hm).lt_length (hlt m hm y hy p hp)
  · intro g hg hc hun
    rw [length_bitsProgs] at hg
    obtain ⟨m, hm, h1, h2⟩ := exists_step (Spot.Bits.base ms) (by simp [Spot.Bits.base]) ms.length hg
    rw [base_succ ms hm] at h2
    obtain ⟨p, rfl⟩ : ∃ p, g = Spot.Bits.base ms m + p := ⟨g - Spot.Bits.base ms m, by omega⟩
    have hp : p < (memberProg ms[m].1 ms[m].2.size).gates.length := by omega
    obtain ⟨-, -, hcov, -, -⟩ := hpart ms[m] (List.getElem_mem hm)
    rw [gates_single] at hcov
    have hcp : ∀ b, (memberProg ms[m].1 ms[m].2.size).gates[p]? ≠ some (.const b) := fun b h =>
      hc b (by simp [(hA m hm).gates p hp, h, shiftGate])
    obtain ⟨y, hy, hpy⟩ := List.mem_flatten.1 (hcov p hp hcp fun h => hun ((hA m hm).unread hp h))
    exact List.mem_flatten.2 ⟨y.map (Spot.Bits.base ms m + ·), hmemb.2 ⟨m, hm, y, hy, rfl⟩,
      List.mem_map.2 ⟨p, hpy, rfl⟩⟩
  · intro x hx
    obtain ⟨m, hm, y, hy, rfl⟩ := hmemb.1 hx
    obtain ⟨-, -, -, hne, -⟩ := hpart ms[m] (List.getElem_mem hm)
    rw [blocks_single] at hne
    simpa using hne y hy
  · intro x hx
    obtain ⟨m, hm, y, hy, rfl⟩ := hmemb.1 hx
    obtain ⟨-, -, -, -, hcl⟩ := hpart ms[m] (List.getElem_mem hm)
    rw [blocks_single] at hcl
    obtain ⟨w, hw1, hw2⟩ := hcl y hy
    have hy' := hlt m hm y hy
    exact ⟨w, by rw [(hA m hm).unitClass_map hy']; exact hw1, by rw [(hA m hm).committed_map hy']; exact hw2⟩

/-- **A member the scope covers**: its unit net built (`Built`) with `unitLog ≤ 32`, a computed row, and at most
`unitClassMax` gates in its program. -/
def MemberFits (ci : _root_.Flock.Circuit × Array ℕ) : Prop :=
  FlockSoundness.Built ci.1.unit ∧ ci.1.unit.unitLog ≤ 32 ∧ 0 < (bitsRows ci.1).nComp ∧
    (memberProg ci.1 ci.2.size).gates.length ≤ unitClassMax

/-- **The replay layout fills the members' files**: each unit `x` of each member `t` is in some request slot `ρ`'s run,
`ρ·n_t` to `(ρ + 1)·n_t`. -/
def FilesFill (ns : Array ℕ) (l : Spot.Slots.SlotOrder) : Prop :=
  ∀ t (ht : t < ms.length), ∀ x < ms[t].2.size, ∃ ρ b, l.blocks[ρ]? = some b ∧ b.unit = some ρ ∧
    ρ * ns.getD t 0 ≤ x ∧ x < (ρ + 1) * ns.getD t 0

/-! ## The members' replay partition -/

section files

variable {ns : Array ℕ} {l : Spot.Slots.SlotOrder}

theorem nodup_slotFiles (hfit : FilesFit ms ns l) (ρ : ℕ) : (slotFiles ms ns l ρ).Nodup := by
  unfold slotFiles
  rcases hb : l.blocks[ρ]? with _ | b
  · simp
  · rcases hu : b.unit with _ | u
    · simp [hu]
    · simp only [hu, Option.bind_some]
      obtain ⟨hρu, hroom⟩ := hfit ρ b hb u hu
      rw [hρu]
      unfold filesUnits
      refine List.nodup_flatMap.2 ⟨fun t _ => List.nodup_range.map (injective_filesBlockOf ms ns ρ t),
        List.nodup_range.pairwise_of_forall_ne fun t ht t' ht' htt' => ?_⟩
      intro w hw hw'
      obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
      obtain ⟨p', hp', he⟩ := List.mem_map.1 hw'
      have ht1 := List.mem_range.1 ht
      have ht1' := List.mem_range.1 ht'
      have hx : ρ * ns.getD t 0 + p < ms[t].2.size :=
        lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ) (List.mem_range.1 hp)) (hroom t ht1)
      have hx' : ρ * ns.getD t' 0 + p' < ms[t'].2.size :=
        lt_of_lt_of_le (rank_mul_add_lt (Nat.lt_add_one ρ) (List.mem_range.1 hp')) (hroom t' ht1')
      have e := stratumOf_membersPart ms ht1' hx'
      simp only [filesBlockOf] at he
      rw [he, stratumOf_membersPart ms ht1 hx] at e
      exact htt' e

/-- **The members' replay partition has distinct gates** where the layout fits the members' files. -/
theorem nodup_gates_filesReplay (hnd : (membersPart ms).gates.Nodup) (hfit : FilesFit ms ns l) :
    (filesReplay ms ns l).gates.Nodup :=
  Partition.nodup_gates_union (I := slotFiles ms ns l) hnd (fun _ h => getElem?_blocks_filesReplay ms ns l h)
    (fun ρ _ => nodup_slotFiles ms hfit ρ) (fun _ _ _ _ h12 _ hw => not_mem_slotFiles ms hfit h12 hw)

theorem exists_unitBase {v : ℕ} (hv : v < (membersPart ms).blocks.length) :
    ∃ t, ∃ ht : t < ms.length, ∃ x < ms[t].2.size, v = unitBase ms t + x := by
  rw [length_blocks_membersPart] at hv
  obtain ⟨t, ht, h1, h2⟩ := exists_step (unitBase ms) (by simp [unitBase]) ms.length hv
  rw [unitBase_succ ms ht] at h2
  exact ⟨t, ht, v - unitBase ms t, by omega, by omega⟩

/-- **Every proof block replays in some request** where the layout fits and fills the members' files. -/
theorem isSome_replayOf_filesReplay (hnd : (membersPart ms).gates.Nodup)
    (hne : ∀ u ∈ (membersPart ms).blocks, u ≠ []) (hfit : FilesFit ms ns l) (hfill : FilesFill ms ns l) {v : ℕ}
    (hv : v < (membersPart ms).blocks.length) : (replayOf (membersPart ms) (filesReplay ms ns l) v).isSome := by
  obtain ⟨t, ht, x, hx, rfl⟩ := exists_unitBase ms hv
  obtain ⟨ρ, b, hb, hbu, h1, h2⟩ := hfill t ht x hx
  have e := Nat.add_one_mul ρ (ns.getD t 0)
  have hr := (replayOf_filesReplay_iff_filesBlockOf ms hnd hne hfit hb hbu hv).2
    ⟨t, ht, x - ρ * ns.getD t 0, by omega, by simp only [filesBlockOf]; omega⟩
  simp [hr]

end files

end Flock.Bridge.Members

namespace SpotFlock

open Claims Claims.Sampling Properties.Spot

/-! ## The scope check's converse -/

theorem topoOk_complete {C : Prog} (hC : C.Topological) : topoOk C = true := by
  unfold topoOk
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq]
  refine ⟨fun g _ => ?_, hC.2⟩
  cases hx : C.gates[g]? with
  | none => simp
  | some x => simpa using hC.1 g x hx

theorem partitionsOk_complete {P : Partition} {C : Prog} (hP : P.Partitions C) : partitionsOk P C = true := by
  obtain ⟨hnd, hlt, hcov, hne, hw⟩ := hP
  unfold partitionsOk
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq]
  refine ⟨⟨⟨⟨hnd, hlt⟩, fun g hg => ?_⟩, fun u hu => ?_⟩, fun u hu => ?_⟩
  · by_cases hc : ∃ b, C.gates[g]? = some (.const b)
    · obtain ⟨b, hb⟩ := hc
      simp [constAt, hb]
    · by_cases hu : C.Unread g
      · simp [hu]
      · simp [hcov g hg (fun b hb => hc ⟨b, hb⟩) hu]
  · simpa [List.isEmpty_iff] using hne u hu
  · obtain ⟨w, hw₁, hw₂⟩ := hw u hu
    simp [hw₁, hw₂]

theorem oneStageOk_complete {ns : List ℕ} {o : OneStage} (h : o.WellFormed ns) : oneStageOk ns o = true := by
  cases o <;> simp_all [oneStageOk, OneStage.WellFormed]

/-- **`scopeOk` accepts what is in the two-stage scope** (`InScopeTwo`'s conjuncts), the converse of `scopeOk_sound`. -/
theorem scopeOk_complete {C : Prog} {Pt : Partition} {d : Design.Two} (htopo : C.Topological)
    (hpart : Pt.Partitions C) (hnd : d.part.gates.Nodup) (hrep : ∀ v < Pt.blocks.length, (replayOf Pt d.part v).isSome)
    (hwf : d.WellFormed Pt) : scopeOk C Pt d = true := by
  obtain ⟨⟨hlen, hrules⟩, hklen, hk⟩ := hwf
  unfold scopeOk
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, beq_iff_eq, decide_eq_true_eq]
  exact ⟨⟨⟨⟨⟨⟨⟨topoOk_complete htopo, partitionsOk_complete hpart⟩, hnd⟩, hrep⟩, hlen⟩,
    fun p hp => oneStageOk_complete (hrules p hp)⟩, hklen⟩, hk⟩

/-! ## The scope at the members -/

/-- **`scopeOk` accepts the members**, their proof blocks and their replay partition, for members that fit
(`MemberFits`), a replay layout that fits and fills their files (`FilesFit`, `FilesFill`), a well-formed first-stage
design and a second-stage count of at least `1` for each member. -/
theorem scopeOk_members {ms : Spot.Bits.Members} {ns : Array ℕ} {l : Spot.Slots.SlotOrder} {d : Design} {k : List ℕ}
    (hm : ∀ ci ∈ ms, Flock.Bridge.Members.MemberFits ci) (hfit : Flock.Bridge.Members.FilesFit ms ns l)
    (hfill : Flock.Bridge.Members.FilesFill ms ns l) (hd : d.WellFormed (filesReplay ms ns l))
    (hk : k.length = ms.length) (hk1 : ∀ x ∈ k, 1 ≤ x) :
    scopeOk (bitsProgs ms) (membersPart ms) ⟨filesReplay ms ns l, d, k⟩ = true := by
  have htop : ∀ ci ∈ ms, (memberProg ci.1 ci.2.size).Topological := fun ci hci =>
    Flock.Bridge.Members.topological_memberProg ci.1 (hU := (hm ci hci).1) (hm ci hci).2.1 ci.2.size
  have hpart := Flock.Bridge.Members.partitions_membersPart ms htop fun ci hci =>
    Flock.Bridge.partitions_keyProg ci.1 ci.2.size (hU := (hm ci hci).1) (hm ci hci).2.1 (hm ci hci).2.2.1
      (hm ci hci).2.2.2
  refine scopeOk_complete (Flock.Bridge.Members.topological_bitsProgs ms htop) hpart
    (Flock.Bridge.Members.nodup_gates_filesReplay ms hpart.1 hfit)
    (fun v hv => Flock.Bridge.Members.isSome_replayOf_filesReplay ms hpart.1 hpart.2.2.2.1 hfit hfill hv)
    ⟨hd, ?_, hk1⟩
  rw [Flock.Bridge.Members.strata_membersPart, List.length_map, List.length_range]
  exact hk

/-- **`two` returns the members' replay design** for members that fit, a replay layout that fits (`filesOk`) and fills
their files, a well-formed first-stage design, second-stage counts of at least `1`, one for each member, and templates
whose reference statements each read as a member's (`templateOk`). -/
theorem two_eq_filesReplay (h : Held) (s : AuditStatement) {l : Spot.Slots.SlotOrder} {d : Design}
    (hl : (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption = some l) (hd : replayDesign h = some d)
    (hm : ∀ ci ∈ h.ms, Flock.Bridge.Members.MemberFits ci) (hfill : Flock.Bridge.Members.FilesFill h.ms h.reg.ns l)
    (hwf : d.WellFormed (filesReplay h.ms h.reg.ns l)) (hk : h.k.length = h.ms.length) (hk1 : ∀ x ∈ h.k, 1 ≤ x)
    (hf : filesOk h.reg h.ms l h.k = true) (ht : ∀ t < h.ms.length, templateOk h t = true) :
    two h s = some ⟨filesReplay h.ms h.reg.ns l, d, h.k⟩ := by
  have hs := scopeOk_members hm (filesOk_sound hf).2.2.1 hfill hwf hk hk1
  unfold two
  rw [hl, hd]
  simpa [hs, hf] using ht

theorem isSome_two (h : Held) (s : AuditStatement) {l : Spot.Slots.SlotOrder} {d : Design}
    (hl : (Spot.Slots.layout h.reg.first.program h.reg.first.query).toOption = some l) (hd : replayDesign h = some d)
    (hm : ∀ ci ∈ h.ms, Flock.Bridge.Members.MemberFits ci) (hfill : Flock.Bridge.Members.FilesFill h.ms h.reg.ns l)
    (hwf : d.WellFormed (filesReplay h.ms h.reg.ns l)) (hk : h.k.length = h.ms.length) (hk1 : ∀ x ∈ h.k, 1 ≤ x)
    (hf : filesOk h.reg h.ms l h.k = true) (ht : ∀ t < h.ms.length, templateOk h t = true) : (two h s).isSome := by
  simp [two_eq_filesReplay h s hl hd hm hfill hwf hk hk1 hf ht]

end SpotFlock
