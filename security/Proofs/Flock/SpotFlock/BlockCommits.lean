module

public import Proofs.Flock.SpotFlock.BlockSize

/-!
# What a unit's proof block commits, from its circuit file's descriptor

A proof block commits its gates that `Prog.Boundary` marks: those returned, the inputs, and those a gate outside it
reads. Where each block reads only its own gates and constants, the blocks are disjoint and every gate but a constant
or an input is in one, no gate outside a block reads one of its gates, so it commits only its returned gates and its
inputs (`committed_le_ends`). A member's units' blocks are such (`member_closed`), so a unit's block commits at most
its stacked rows' inputs and the sixteen bits of each returned output leaf of each of its instances,
`(bitsRows c).nIn + outBits c` (`length_ends_le`): a bound on one proof block that reads neither the member's unit count
nor its rows.

* `committed_le_ends`: a closed block among disjoint blocks that cover the program commits at most its `ends`.
* `unit_outGate`: a returned gate of unit `u` is one of the copy rows `outGate c n u k j t`.
* `length_ends_le`: a unit's block's returned gates and inputs.
-/

@[expose] public section

namespace Flock.Bridge.BlockCommits

open Claims.Sampling Properties.Spot Flock.Bridge.NeededBitsOf Flock.Bridge.BlockSize

/-- Whether gate `g` of `C` is returned or an input. -/
def IsEnd (C : Prog) (g : ℕ) : Bool := C.outs.contains g || decide (C.gates[g]? = some .input)

/-- **The gates of `u` that are returned or are inputs.** -/
def ends (C : Prog) (u : List ℕ) : List ℕ := u.filter (IsEnd C)

/-- **A closed proof block commits only its returned gates and its inputs**: where every block of `bs` reads only its
own gates and constants and holds no constant, distinct blocks share no gate, and every gate but a constant or an input
is in some block, no gate outside a block reads one of its gates. -/
theorem committed_le_ends {C : Prog} {bs : List (List ℕ)} {bu : List ℕ} (hbu : bu ∈ bs)
    (hdis : ∀ bv ∈ bs, ∀ g ∈ bu, g ∈ bv → bv = bu)
    (hcov : ∀ i < C.gates.length, (∀ b, C.gates[i]? ≠ some (.const b)) → C.gates[i]? ≠ some .input →
      ∃ bv ∈ bs, i ∈ bv)
    (hcl : ∀ bv ∈ bs, ∀ i ∈ bv, ∀ x, C.gates[i]? = some x →
      (∀ b, x ≠ .const b) ∧ ∀ w ∈ x.reads, w ∈ bv ∨ ∃ b, C.gates[w]? = some (.const b)) :
    committed C bu ≤ (ends C bu).length := by
  unfold committed ends
  rw [← List.countP_eq_length_filter, ← List.countP_eq_length_filter]
  refine List.countP_mono_left fun g hg hb => ?_
  beta_reduce at hb
  unfold Prog.Boundary at hb
  unfold IsEnd
  rcases Bool.or_eq_true_iff.1 hb with hb | hb
  · exact hb
  · exfalso
    obtain ⟨g', hg', hr⟩ := List.any_eq_true.1 hb
    rw [Bool.and_eq_true, decide_eq_true_eq] at hr
    obtain ⟨hnot, hrd⟩ := hr
    have hl := List.mem_range.1 hg'
    have hx : C.gates[g']? = some C.gates[g'] := List.getElem?_eq_getElem hl
    rw [hx] at hrd
    have hrd' : g ∈ C.gates[g'].reads := by simpa using hrd
    have hnc : ∀ b, C.gates[g']? ≠ some (.const b) := fun b h => by
      rw [hx] at h
      rw [Option.some.inj h] at hrd'
      simp [BGate.reads] at hrd'
    have hni : C.gates[g']? ≠ some .input := fun h => by
      rw [hx] at h
      rw [Option.some.inj h] at hrd'
      simp [BGate.reads] at hrd'
    obtain ⟨bv, hbv, hg'v⟩ := hcov g' hl hnc hni
    rcases (hcl bv hbv g' hg'v _ hx).2 g hrd' with hgv | ⟨b, hcb⟩
    · exact hnot (hdis bv hbv g hg hgv ▸ hg'v)
    · exact (hcl bu hbu g hg _ hcb).1 b rfl

/-- A list's length, from the lengths of a `flatMap`'s pieces. -/
theorem length_flatMap_le_sum {α β : Type} (f : α → List β) (w : α → ℕ) :
    ∀ l : List α, (∀ a ∈ l, (f a).length ≤ w a) → (l.flatMap f).length ≤ (l.map w).sum
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.length_append, List.map_cons, List.sum_cons]
    have ha := h a (List.mem_cons.2 (Or.inl rfl))
    have ih := length_flatMap_le_sum f w l fun b hb => h b (List.mem_cons.2 (Or.inr hb))
    omega

theorem sum_const_range (m w : ℕ) : ((List.range m).map fun _ => w).sum = w * m := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [List.range_succ, List.map_append, List.sum_append, ih]
    simp [Nat.mul_succ]

/-- A list without repeats whose entries are all in `L` is no longer than `L`. -/
theorem length_le_of_subset {l L : List ℕ} (hnd : l.Nodup) (h : ∀ x ∈ l, x ∈ L) : l.length ≤ L.length := by
  have hc := Finset.card_le_card (show l.toFinset ⊆ L.toFinset from
    fun x hx => List.mem_toFinset.2 (h x (List.mem_toFinset.1 hx)))
  rw [List.toFinset_card_of_nodup hnd] at hc
  exact hc.trans (List.toFinset_card_le L)

variable (c : Flock.Circuit) (n : ℕ)

/-- **The returned output bits of one unit**: sixteen for each returned output leaf of each of its `upv` instances. -/
def outBits (c : Flock.Circuit) : ℕ := ((List.range c.upv).map fun k => 16 * (c.leavesOut.getD k #[]).size).sum

/-- **Unit `u`'s returned gates**, as copy rows: instance `k`'s returned output leaf `j`, bit `t`. -/
def unitOuts (u : ℕ) : List ℕ :=
  (List.range c.upv).flatMap fun k => (List.range (c.leavesOut.getD k #[]).size).flatMap fun j =>
    (List.range 16).filterMap (Spot.Bits.outGate c n u k j)

theorem length_unitOuts_le (u : ℕ) : (unitOuts c n u).length ≤ outBits c := by
  unfold unitOuts outBits
  refine length_flatMap_le_sum _ _ _ fun k _ => ?_
  refine (length_flatMap_le_sum _ (fun _ => 16) _ fun j _ => ?_).trans (le_of_eq (sum_const_range _ _))
  exact (List.length_filterMap_le _ _).trans (by simp)

/-- **A returned gate of unit `u` is one of its copy rows.** -/
theorem unit_outGate {u o : ℕ} (ho : o ∈ (Spot.Bits.keyProg c n).outs)
    (hu : Spot.Bits.unitOf (Spot.Bits.keyProg c n) o = some u) : o ∈ unitOuts c n u := by
  obtain ⟨u', k, j, t, hk⟩ := of_mem_outs c n ho
  obtain ⟨hu', hkl, hjl, htl, i, hi, ho'⟩ := outGate_some c n hk
  have hwu := unitOf_row c n hu' hi
  rw [← ho', hu] at hwu
  obtain rfl := Option.some.inj hwu
  unfold unitOuts
  simp only [List.mem_flatMap, List.mem_range, List.mem_filterMap]
  exact ⟨k, hkl, j, hjl, t, htl, hk⟩

/-- **A unit's lowered gates' returned ones** are the positions of its returned copy rows, one a row. -/
theorem length_ends_unitGates_le (u : ℕ) :
    ((Spot.Bits.unitGates (Spot.Bits.keyProg c n) u).filter (IsEnd (memberProg c n))).length ≤
      outBits c := by
  unfold Spot.Bits.unitGates
  rw [List.filter_flatMap]
  refine (length_flatMap_le _ (fun g => Spot.Bits.unitOf (Spot.Bits.keyProg c n) g == some u &&
      (Spot.Bits.keyProg c n).outs.contains g) 1 _ fun g hg => ?_).trans ?_
  · beta_reduce
    have hgs := List.mem_range.1 hg
    by_cases hgu : Spot.Bits.unitOf (Spot.Bits.keyProg c n) g = some u
    · obtain ⟨a, b, hab⟩ := row?_of_unitOf hgu
      have hpc : ∀ q ∈ ((List.range (Spot.Bits.gateSize (Spot.Bits.keyProg c n) g)).map
          (Spot.Bits.off (Spot.Bits.keyProg c n) g + ·)).filter (IsEnd (memberProg c n)),
          (Spot.Bits.keyProg c n).outs.contains g = true ∧ q = Spot.Bits.off (Spot.Bits.keyProg c n) g + 2 := by
        intro q hq
        obtain ⟨hq, hqe⟩ := List.mem_filter.1 hq
        obtain ⟨jj, hjj, rfl⟩ := List.mem_map.1 hq
        have hjj' := List.mem_range.1 hjj
        unfold IsEnd at hqe
        rcases Bool.or_eq_true_iff.1 hqe with hqe | hqe
        · rw [memberProg_outs, List.contains_iff_mem, List.mem_map] at hqe
          obtain ⟨o, ho, he⟩ := hqe
          obtain ⟨u', k, j, t, hk⟩ := of_mem_outs c n ho
          obtain ⟨hu', -, -, -, i, hi, ho'⟩ := outGate_some c n hk
          have hou := unitOf_row c n hu' hi
          rw [← ho'] at hou
          obtain ⟨a', b', hab'⟩ := row?_of_unitOf hou
          rw [posOf_of_some hab'] at he
          have hos := (unitOf_eq_some hou).2.1
          have hgo := eq_of_off hgs hos hjj' (by rw [gateSize_of_some hab']; omega) he.symm
          subst hgo
          have hj2 : jj = 2 := by omega
          exact ⟨List.contains_iff_mem.2 ho, by rw [hj2]⟩
        · exact (row_gate_not_input (Spot.Bits.consts c n) hgs hab hjj' (of_decide_eq_true hqe)).elim
      have hpu : (Spot.Bits.unitOf (Spot.Bits.keyProg c n) g == some u) = true := by simp [hgu]
      rw [if_pos hpu]
      by_cases hgo : (Spot.Bits.keyProg c n).outs.contains g = true
      · have hp1 : (Spot.Bits.unitOf (Spot.Bits.keyProg c n) g == some u &&
            (Spot.Bits.keyProg c n).outs.contains g) = true := by
          rw [Bool.and_eq_true]
          exact ⟨hpu, hgo⟩
        rw [if_pos hp1]
        exact length_le_of_window (a := Spot.Bits.off (Spot.Bits.keyProg c n) g + 2)
          ((List.nodup_range.map (fun _ _ h => Nat.add_left_cancel h)).filter _)
          fun q hq => by rw [(hpc q hq).2]; omega
      · have hp0 : ¬ (Spot.Bits.unitOf (Spot.Bits.keyProg c n) g == some u &&
            (Spot.Bits.keyProg c n).outs.contains g) = true := by
          rw [Bool.and_eq_true]
          exact fun hp => hgo hp.2
        rw [if_neg hp0, Nat.le_zero, List.length_eq_zero_iff, List.eq_nil_iff_forall_not_mem]
        exact fun q hq => hgo (hpc q hq).1
    · have hpu : ¬ (Spot.Bits.unitOf (Spot.Bits.keyProg c n) g == some u) = true := by simp [hgu]
      rw [if_neg hpu]
      simp
  · rw [Nat.one_mul]
    refine (length_le_of_subset (List.nodup_range.filter _) fun o ho => ?_).trans (length_unitOuts_le c n u)
    obtain ⟨-, hp⟩ := List.mem_filter.1 ho
    rw [Bool.and_eq_true, beq_iff_eq, List.contains_iff_mem] at hp
    exact unit_outGate c n hp.2 hp.1

/-- **A unit's proof block's returned gates and inputs**: at most its stacked rows' inputs and its returned output
bits, whatever the member's unit count and its rows. -/
theorem length_ends_le (u : ℕ) :
    (ends (memberProg c n) (Spot.Bits.blockOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u)).length ≤
      (bitsRows c).nIn + outBits c := by
  unfold ends Spot.Bits.blockOf
  rw [List.filter_append, List.length_append]
  have h1 := length_ends_unitGates_le c n u
  have h2 : (((Spot.Bits.insOf (Spot.Bits.keyProg c n) (Spot.Bits.consts c n) u).map
      (Spot.Bits.posOf (Spot.Bits.keyProg c n))).filter (IsEnd (memberProg c n))).length ≤ (bitsRows c).nIn :=
    (List.length_filter_le _ _).trans ((List.length_map _).le.trans (length_insOf_le c n u))
  omega

end Flock.Bridge.BlockCommits
