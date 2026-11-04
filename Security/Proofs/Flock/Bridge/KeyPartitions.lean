import Proofs.Flock.Bridge.Partitions
import Proofs.Flock.Bridge.FilePart
import Proofs.Flock.Soundness.Discharge.Layout.Outs

/-!
# The key program's partition partitions its program

`keyPart_partitions`: at a circuit file `c` with `upv > 0`, whose units output at most `unitClassMax` bits each
(`SampledProofs.FilePart.outBits`, what the code's class check bounds), `partOf` of the key program's partition at the
returned outputs' bits partitions its lowered program, at any strata that sum to `n`. It is `partOf_partitions` with its
three conditions discharged:

* No returned output is an input (`keyProg_outputs_not_input`): it is a computed column of the instance that puts it on
  a returned output's column, which that instance's unit holds (`unit_of_outCol`).
* Every unit holds a gate (`exists_unit_keyProg`): its rows' lowered gates are `3 · nComp` consecutive gates
  (`unitGates_keyProg`), and `nComp > 0` at `upv > 0`.
* Every block outputs at most `outBits c` (`outputs_keyProg_le`): no unit reads another unit's gate
  (`keyProg_hread`, from `Layout.keyProg_reads`), so a block outputs at most its returned outputs' bits
  (`outputs_blockOf_le`), and those are at most `outBits c` (`card_outs_le`: a unit's returned outputs are distinct
  computed columns of its instance on returned outputs' columns, at most `outBitsAt`'s count of them,
  `card_outCols_le`).
-/

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge

theorem foldl_add_map (f L : ℕ → ℕ) : ∀ m a : ℕ,
    ((List.range m).map L).foldl (fun a k => a + f k) a = a + ∑ u ∈ Finset.range m, f (L u)
  | 0, a => by simp
  | m + 1, a => by
    rw [List.range_succ, List.map_append, List.foldl_append, foldl_add_map f L m a, Finset.sum_range_succ]
    simp [Nat.add_assoc]

variable (c : Flock.Circuit) (hU : Built c.unit)

/-- `outBits` as a sum over the unit's copies. -/
theorem outBits_eq : SampledProofs.FilePart.outBits c = ∑ u' ∈ Finset.range c.upv,
    (min ((c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * (c.leavesOut.getD u' #[]).size) c.unit.constPos -
      max ((c.unit.outGroups.getD 0 default).col * Flock.WORD)
        (min (c.unit.inWords * Flock.WORD) c.unit.constPos)) := by
  simp only [SampledProofs.FilePart.outBits, SampledProofs.FilePart.outBitsAt]
  exact (foldl_add_map _ _ c.upv 0).trans (Nat.zero_add _)

/-- **Columns on returned outputs' columns are at most `outBits c`**: copy `u'` of the template puts its output `j`'s
bit `t` on one stacked column, and those of copy `u'` that are computed rows are at most the interval `outBitsAt`
counts. -/
theorem card_outCols_le {w : ℕ} (cols : Finset (Fin w))
    (hX : ∀ x ∈ cols, ∃ u' j t, Layout.OutColAt c hU x.val u' j t) :
    cols.card ≤ SampledProofs.FilePart.outBits c := by
  have hcp : (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp = c.unit.constPos :=
    FlockSoundness.Rows.ofNet_one hU
  have hnI : (FlockSoundness.Rows.ofNet hU).nIn = min (c.unit.inWords * Flock.WORD) c.unit.constPos :=
    FlockSoundness.Rows.ofNet_nIn hU
  let A : ℕ → Finset (ℕ × ℕ) := fun u' =>
    ((Finset.range (c.leavesOut.getD u' #[]).size) ×ˢ Finset.range 16).filter fun jt =>
      (FlockSoundness.Rows.ofNet hU).nIn ≤ Layout.outCol c jt.1 jt.2 ∧
        Layout.outCol c jt.1 jt.2 < (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp
  let F : ℕ → ℕ × ℕ → ℕ := fun u' jt => c.upv * (FlockSoundness.Rows.ofNet hU).nIn +
    (u' * ((FlockSoundness.Rows.ofNet hU).nComp + 1) +
      (1 + (Layout.outCol c jt.1 jt.2 - (FlockSoundness.Rows.ofNet hU).nIn)))
  have hsub : cols.map Fin.valEmbedding ⊆ (Finset.range c.upv).biUnion fun u' => (A u').image (F u') := by
    intro y hy
    obtain ⟨x, hx, rfl⟩ := Finset.mem_map.1 hy
    obtain ⟨u', j, t, hu', hj, ht, h1, h2, hxv⟩ := hX x hx
    exact Finset.mem_biUnion.2 ⟨u', Finset.mem_range.2 hu', Finset.mem_image.2 ⟨(j, t),
      Finset.mem_filter.2 ⟨Finset.mem_product.2 ⟨Finset.mem_range.2 hj, Finset.mem_range.2 ht⟩, h1, h2⟩, hxv.symm⟩⟩
  have hA : ∀ u', (A u').card ≤
      min ((c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * (c.leavesOut.getD u' #[]).size) c.unit.constPos -
        max ((c.unit.outGroups.getD 0 default).col * Flock.WORD)
          (min (c.unit.inWords * Flock.WORD) c.unit.constPos) := by
    intro u'
    refine le_trans (Finset.card_le_card_of_injOn (s := A u') (t := Finset.Ico
      (max ((c.unit.outGroups.getD 0 default).col * Flock.WORD) (min (c.unit.inWords * Flock.WORD) c.unit.constPos))
      (min ((c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * (c.leavesOut.getD u' #[]).size)
        c.unit.constPos)) (fun jt => Layout.outCol c jt.1 jt.2) (fun jt hjt => ?_) (fun a ha b hb hab => ?_))
      (Nat.card_Ico _ _).le
    · obtain ⟨hp, h1, h2⟩ := Finset.mem_filter.1 (Finset.mem_coe.1 hjt)
      obtain ⟨hj, ht⟩ := Finset.mem_product.1 hp
      rw [Finset.mem_range] at hj ht
      refine Finset.mem_coe.2 (Finset.mem_Ico.2 ⟨?_, ?_⟩)
      · simp only [Layout.outCol] at h1 h2 ⊢
        omega
      · simp only [Layout.outCol] at h1 h2 ⊢
        omega
    · have hta := Finset.mem_range.1 (Finset.mem_product.1 (Finset.mem_filter.1 (Finset.mem_coe.1 ha)).1).2
      have htb := Finset.mem_range.1 (Finset.mem_product.1 (Finset.mem_filter.1 (Finset.mem_coe.1 hb)).1).2
      simp only [Layout.outCol] at hab
      exact Prod.ext (by omega) (by omega)
  rw [outBits_eq]
  calc cols.card = (cols.map Fin.valEmbedding).card := (Finset.card_map _).symm
    _ ≤ _ := Finset.card_le_card hsub
    _ ≤ _ := Finset.card_biUnion_le
    _ ≤ _ := Finset.sum_le_sum fun u' _ => Finset.card_image_le.trans (hA u')

variable (n : ℕ)

/-- **A returned output's bit is held by the unit whose instance puts it there**: `OutColAt` is a computed column. -/
theorem unit_of_outCol {u₁ : Fin n} {x₁ : Fin ((Placed.keyProg c hU n).rows u₁).w} {u' j t : ℕ}
    (hoc : Layout.OutColAt c hU x₁.val u' j t) :
    (Placed.keyProg c hU n).unit ((Const.Prog.rowsWireC (Placed.keyProg c hU n) u₁).wire x₁) = some u₁ := by
  have eR := Placed.rows_circuitProg c hU n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)) u₁
  have eN : ((Placed.keyProg c hU n).rows u₁).nIn = c.upv * (FlockSoundness.Rows.ofNet hU).nIn := by
    rw [congrArg FlockSoundness.Rows.nIn eR]; rfl
  have eC : ((Placed.keyProg c hU n).rows u₁).nComp = c.upv * ((FlockSoundness.Rows.ofNet hU).nComp + 1) := by
    rw [congrArg FlockSoundness.Rows.nComp eR]; rfl
  obtain ⟨hu', hj, ht, h1, h2, hxv⟩ := hoc
  have hlt := FlockSoundness.Rows.mul_add_lt hu'
    (show 1 + (Layout.outCol c j t - (FlockSoundness.Rows.ofNet hU).nIn) < (FlockSoundness.Rows.ofNet hU).nComp + 1 by
      omega)
  have hi : x₁.val - ((Placed.keyProg c hU n).rows u₁).nIn < ((Placed.keyProg c hU n).rows u₁).nComp := by
    rw [eN, eC]; omega
  have hx₁ : x₁ = ((Placed.keyProg c hU n).rows u₁).comp ⟨_, hi⟩ :=
    Fin.ext (by simp only [FlockSoundness.Rows.comp]; rw [eN] at hi ⊢; omega)
  have hm := (Const.Prog.instC (Placed.keyProg c hU n) (Layout.outsOf c hU n) u₁).mem ⟨_, hi⟩
  rw [← hx₁] at hm
  exact hm

theorem unit_of_mem_outsOf {g : Fin (Layout.keyN c hU n)} (hg : g ∈ Layout.outsOf c hU n) :
    ∃ u, (Placed.keyProg c hU n).unit g = some u := by
  obtain ⟨-, u₁, x₁, hw₁, u', j, t, hoc⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hg
  rw [← hw₁]
  exact ⟨u₁, unit_of_outCol c hU n hoc⟩

/-- A unit's returned output is a column of its own instance on a returned output's column. -/
theorem wire_of_mem_outsOf {u : Fin n} {g : Fin (Layout.keyN c hU n)} (hg : g ∈ Layout.outsOf c hU n)
    (hu : (Placed.keyProg c hU n).unit g = some u) :
    ∃ x : Fin ((Placed.keyProg c hU n).rows u).w, (Const.Prog.rowsWireC (Placed.keyProg c hU n) u).wire x = g ∧
      ∃ u' j t, Layout.OutColAt c hU x.val u' j t := by
  obtain ⟨-, u₁, x₁, hw₁, u', j, t, hoc⟩ := (@Finset.mem_filter _ _ (Classical.decPred _) _ _).1 hg
  have h := unit_of_outCol c hU n hoc
  rw [hw₁, hu] at h
  obtain rfl := Option.some.inj h
  exact ⟨x₁, hw₁, u', j, t, hoc⟩

/-- **No returned output is an input.** -/
theorem keyProg_outputs_not_input :
    ∀ g ∈ ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)).outputs,
      ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)).op g ≠ .input := by
  intro g hg hin
  obtain ⟨u, hu⟩ := unit_of_mem_outsOf c hU n hg
  have h := (((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)).input_iff g).2
    (by simp [hin, FlockSoundness.Audit.Op.isInput])
  exact absurd (h.symm.trans hu) (by simp)

/-- **No unit of the key program reads another unit's gate**: a row reads its own unit's computed columns and input
gates (`Layout.keyProg_reads`). -/
theorem keyProg_hread (outs : Finset (Fin (Layout.keyN c hU n)))
    (h g : Fin ((Placed.keyProg c hU n).circuit outs).N) (v v' : Fin n)
    (hv : ((Placed.keyProg c hU n).partition outs).unit h = some v)
    (hg : g ∈ (((Placed.keyProg c hU n).circuit outs).op h).args)
    (hv' : ((Placed.keyProg c hU n).partition outs).unit g = some v') : v' = v := by
  obtain ⟨i, rfl⟩ := (Const.Prog.instC (Placed.keyProg c hU n) outs v).onto h hv
  rw [(Const.Prog.instC (Placed.keyProg c hU n) outs v).op i] at hg
  simp only [FlockSoundness.Audit.Op.args, List.mem_append, List.mem_map] at hg
  obtain ⟨y, -, rfl⟩ : ∃ y, y ∈ ((Placed.keyProg c hU n).rows v).a i ++ ((Placed.keyProg c hU n).rows v).b i ∧
      (Const.Prog.instC (Placed.keyProg c hU n) outs v).wire y = g := by
    rcases hg with ⟨y, hy, e⟩ | ⟨y, hy, e⟩
    · exact ⟨y, List.mem_append_left _ hy, e⟩
    · exact ⟨y, List.mem_append_right _ hy, e⟩
  by_cases hcomp : ∃ j, y = ((Placed.keyProg c hU n).rows v).comp j
  · obtain ⟨j, rfl⟩ := hcomp
    exact Option.some.inj (hv'.symm.trans ((Const.Prog.instC (Placed.keyProg c hU n) outs v).mem j))
  · rw [Layout.keyProg_reads c hU n outs v y fun j e => hcomp ⟨j, e⟩] at hv'
    cases hv'

/-- **A unit's returned outputs are at most `outBits c`.** -/
theorem card_outs_le (u : Fin n) {p : Fin (Layout.keyN c hU n) → Prop} {inst : DecidablePred p}
    (hp : ∀ g, p g → (Placed.keyProg c hU n).unit g = some u ∧ g ∈ Layout.outsOf c hU n) :
    (@Finset.filter _ p inst Finset.univ).card ≤ SampledProofs.FilePart.outBits c := by
  classical
  have hsub : @Finset.filter _ p inst Finset.univ ⊆
      (Finset.univ.filter fun x : Fin ((Placed.keyProg c hU n).rows u).w =>
        ∃ u' j t, Layout.OutColAt c hU x.val u' j t).image
          (Const.Prog.rowsWireC (Placed.keyProg c hU n) u).wire := by
    intro g hg
    obtain ⟨hu, ho⟩ := hp g ((@Finset.mem_filter _ p inst _ g).1 hg).2
    obtain ⟨x, hx, hoc⟩ := wire_of_mem_outsOf c hU n ho hu
    exact Finset.mem_image.2 ⟨x, Finset.mem_filter.2 ⟨Finset.mem_univ _, hoc⟩, hx⟩
  exact (Finset.card_le_card hsub).trans (Finset.card_image_le.trans
    (card_outCols_le c hU _ fun x hx => (Finset.mem_filter.1 hx).2))

/-- **A unit's lowered rows are `3 · nComp` consecutive gates.** -/
theorem unitGates_keyProg (outs : Finset (Fin (Layout.keyN c hU n))) (u : Fin n) :
    unitGates ((Placed.keyProg c hU n).partition outs) u =
      (List.range (3 * (Placed.circuitRows c hU).nComp)).map
        (Placed.inGates n (Layout.keyB c hU) + 3 * (Placed.circuitRows c hU).nComp * u.val + ·) :=
  FilePart.unitGates_of ((Placed.keyProg c hU n).partition outs)
    (FilePart.gates_eq (Placed.inGates n (Layout.keyB c hU)) (Placed.circuitRows c hU) n)
    (FilePart.size_uniform (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
      (one := Placed.oneGate n (Layout.keyB c hU))
      (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) outs)
    (FilePart.unit_uniform_val (k := Placed.inGates n (Layout.keyB c hU)) (R := Placed.circuitRows c hU)
      (ins := Placed.insOf n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU)))
      (one := Placed.oneGate n (Layout.keyB c hU))
      (hone := Placed.insOf_ne_one n (Layout.keyB c hU) (Aliased.keyIdx c (Layout.keyB c hU))) n) u

/-- **Every unit holds a gate**, at `upv > 0`. -/
theorem exists_unit_keyProg (outs : Finset (Fin (Layout.keyN c hU n))) (hupv : 0 < c.upv) (u : Fin n) :
    ∃ g, ((Placed.keyProg c hU n).partition outs).unit g = some u := by
  have e : (Placed.circuitRows c hU).nComp = c.upv * ((FlockSoundness.Rows.ofNet hU).nComp + 1) := rfl
  have hm : 0 < (Placed.circuitRows c hU).nComp := by
    rw [e]
    exact Nat.mul_pos hupv (Nat.succ_pos _)
  have hmem : Placed.inGates n (Layout.keyB c hU) + 3 * (Placed.circuitRows c hU).nComp * u.val + 0 ∈
      unitGates ((Placed.keyProg c hU n).partition outs) u := by
    rw [unitGates_keyProg c hU n outs u]
    exact List.mem_map.2 ⟨0, List.mem_range.2 (by omega), rfl⟩
  obtain ⟨g, hg, -⟩ := mem_unitGates.1 hmem
  exact ⟨g, hg⟩

/-- **A block outputs at most `outBits c`.** -/
theorem outputs_keyProg_le (u : Fin n) :
    Claims.Sampling.outputs
        (progOf ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) (fun g => constsOf c hU n g))
        (blockOf ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (fun g => constsOf c hU n g) u) ≤
      SampledProofs.FilePart.outBits c :=
  (outputs_blockOf_le u (keyProg_hread c hU n _)).trans (card_outs_le c hU n u fun _ hg => hg)

/-- **The key program's partition partitions its program**, at a circuit file with `upv > 0` whose units output at
most `unitClassMax` bits, at strata that sum to `n`. -/
theorem keyPart_partitions (hupv : 0 < c.upv)
    (hcls : SampledProofs.FilePart.outBits c ≤ Claims.Sampling.unitClassMax) {ns : List ℕ} (hns : ns.sum = n) :
    (partOf ((Placed.keyProg c hU n).partition (Layout.outsOf c hU n)) (fun g => constsOf c hU n g) ns).Partitions
      (progOf ((Placed.keyProg c hU n).circuit (Layout.outsOf c hU n)) (fun g => constsOf c hU n g)) :=
  partOf_partitions _ hns (keyProg_outputs_not_input c hU n) (exists_unit_keyProg c hU n _ hupv)
    fun u => (outputs_keyProg_le c hU n u).trans hcls

end Flock.Bridge
