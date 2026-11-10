module

public import Proofs.Flock.Bridge.Slots
public import Proofs.Flock.Soundness.Discharge.Placed.KeyedP
public import Proofs.Flock.Soundness.Discharge.Layout.OutsP

/-!
# The key program's units have a lowered gate and commit their returned bits

`partitions_slots`' two hypotheses at `keyProgP c pi hU n`'s partition, from v2's template check
(`OneStageZk.templateOk`):

* `unitGates_keyProgP_ne_nil`: at `units_per_vu ≥ 1`, every unit has a lowered gate. A template instance stacks
  `units_per_vu` copies of the unit net's rows, each with its own constant, a computed row (`Rows.stack`), so the
  instance has `upv · (nComp + 1)` computed rows; `uniform` gives unit `u` the gates `gates k R u + m` for each of them
  (`uniform_unit`).
* `committed_keyProgP_le`: at the returned outputs' bits `outsOfP`, a unit commits at most `16` values per output leaf
  of its `units_per_vu` copies. A unit's lowered gate on its boundary is a committed wire's position
  (`boundary_unitGates`), a committed wire a unit computes is a returned output's bit (`outsOutP_outsOfP`), and each
  is on a column `OutColAt` names by its copy, leaf and bit.
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

variable {C : Verity.Boolean.Circuit}

/-- **A unit's lowered gate on its boundary is a committed wire's position**: a row's two XORs are interior
(`boundary_inner`), and a gate outside the unit reading a wire's position reads the wire. -/
theorem boundary_unitGates {κ : Fin C.N → Option Bool} {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {u : Fin n}
    {p : ℕ} (hp : p ∈ unitGates P u) (hb : (progOf C κ).Boundary (unitGates P u) p = true) :
    ∃ g, P.unit g = some u ∧ g ∈ P.committed ∧ p = posOf C g := by
  classical
  obtain ⟨g, hu, j, hj, rfl⟩ := mem_unitGates.1 hp
  have hjg : j = gateSize (C.op g) - 1 := by
    by_contra hne
    rw [boundary_inner hu (by omega)] at hb
    simp at hb
  subst hjg
  rw [← posOf_eq g] at hb ⊢
  refine ⟨g, hu, ?_, rfl⟩
  simp only [FlockSoundness.Audit.Partition.committed, Finset.mem_filter, Finset.mem_univ, true_and]
  simp only [Claims.Sampling.Prog.Boundary, Bool.or_eq_true, List.contains_iff_mem, decide_eq_true_eq,
    List.any_eq_true, List.mem_range, Bool.and_eq_true, decide_not, Bool.not_eq_true',
    decide_eq_false_iff_not] at hb
  rcases hb with (hout | hin) | ⟨g', -, hg'u, hrd⟩
  · simp only [progOf, List.mem_map, List.mem_filter, List.mem_finRange, true_and, decide_eq_true_eq] at hout
    obtain ⟨a, ha, he⟩ := hout
    obtain rfl := posOf_injective he
    exact Or.inr (Or.inl ha)
  · exact absurd hin (progOf_ne_input_of_unit (posOf_mem_unitGates.2 hu))
  · cases hx : (progOf C κ).gates[g']? with
    | none => simp [hx] at hrd
    | some x =>
      simp only [hx, Option.map_some, Option.getD_some] at hrd
      obtain ⟨h, k, hk, rfl, hxk⟩ := exists_block hx
      rcases reads_lower_args hxk hrd with ⟨a, ha, he⟩ | ⟨rfl, hi⟩
      · obtain rfl := posOf_injective he
        exact Or.inr (Or.inr ⟨h, ha, fun hh => hg'u (mem_unitGates.2 ⟨h, hh.trans hu, k, hk, rfl⟩)⟩)
      · exfalso
        have h1 := one_le_gateSize (C.op g)
        rw [posOf_eq] at hi
        have hgh : g = h := by
          rcases hi with hi | hi
          · exact (block_inj (j := gateSize (C.op g) - 1) (k := 0) (by omega) (by omega)
              (by rw [Nat.add_zero]; exact hi)).1
          · exact (block_inj (j := gateSize (C.op g) - 1) (k := 1) (by omega) (by omega) hi).1
        exact hg'u (mem_unitGates.2 ⟨h, hgh ▸ hu, 2, hk, rfl⟩)

/-- **A unit commits at most its committed wires**, one lowered gate each (`boundary_unitGates`). -/
theorem committed_le_card (κ : Fin C.N → Option Bool) {n : ℕ} (P : FlockSoundness.Audit.Partition C n) (u : Fin n) :
    committed (progOf C κ) (unitGates P u) ≤
      (Finset.univ.filter fun g => P.unit g = some u ∧ g ∈ P.committed).card := by
  classical
  have hsub : ((unitGates P u).filter fun p => (progOf C κ).Boundary (unitGates P u) p).toFinset ⊆
      (Finset.univ.filter fun g => P.unit g = some u ∧ g ∈ P.committed).image (posOf C) := by
    intro p hp
    rw [List.mem_toFinset, List.mem_filter] at hp
    obtain ⟨g, hu, hc, rfl⟩ := boundary_unitGates hp.1 hp.2
    exact Finset.mem_image.2 ⟨g, Finset.mem_filter.2 ⟨Finset.mem_univ _, hu, hc⟩, rfl⟩
  calc committed (progOf C κ) (unitGates P u)
      = ((unitGates P u).filter fun p => (progOf C κ).Boundary (unitGates P u) p).toFinset.card :=
        (List.toFinset_card_of_nodup ((nodup_unitGates P u).filter _)).symm
    _ ≤ _ := Finset.card_le_card hsub
    _ ≤ _ := Finset.card_image_le

end Flock.Bridge

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Placed

open Model Audit

theorem gates_mono (k : ℕ) (R : Rows) {u n : ℕ} (h : u ≤ n) : gates k R u ≤ gates k R n := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => exact ih.trans (Nat.le_add_right _ _)

theorem gates_add_lt (k : ℕ) (R : Rows) {u n m : ℕ} (hu : u < n) (hm : m < R.nComp) :
    gates k R u + m < gates k R n :=
  calc gates k R u + m < gates k R (u + 1) := by show _ < gates k R u + R.nComp; omega
    _ ≤ gates k R n := gates_mono k R hu

/-- **Unit `u` of `uniform` holds its computed rows' gates**, `gates k R u + m` for `m < R.nComp`. -/
theorem uniform_unit (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (hone : ∀ u c, ins u c ≠ one) :
    ∀ (n : ℕ) (u : Fin n) {m : ℕ} (hm : m < R.nComp),
      (uniform k R ins one hone n).unit ⟨gates k R u.val + m, gates_add_lt k R u.isLt hm⟩ = some u
  | 0, u, _, _ => u.elim0
  | n + 1, u, m, hm => by
    simp only [uniform, Prog.unit]
    split
    · rename_i h
      have hu : u.val < n := by
        by_contra hu
        have : gates k R n ≤ gates k R u.val := gates_mono k R (by omega)
        omega
      rw [uniform_unit k R ins one hone n ⟨u.val, hu⟩ hm]
      exact congrArg some (Fin.ext rfl)
    · rename_i h
      have hu : u.val = n := by
        by_contra hu
        exact h (gates_add_lt k R (show u.val < n by omega) hm)
      exact congrArg some (Fin.ext hu.symm)

/-- **Every unit of the key program has a lowered gate** where the circuit has a unit per VU. -/
theorem unitGates_keyProgP_ne_nil (c : Flock.Circuit) (pi : HmIn.Ports) (hU : Built c.unit) {n : ℕ}
    (outs : Finset (Fin (gates (inGates n (circuitRows c hU).nIn) (circuitRows c hU) n))) (hupv : c.upv ≠ 0)
    (u : Fin n) : Flock.Bridge.unitGates ((keyProgP c pi hU n).partition outs) u ≠ [] := by
  have hcomp : 0 < (circuitRows c hU).nComp :=
    show 0 < c.upv * ((Rows.ofNet hU).nComp + 1) from Nat.mul_pos (Nat.pos_of_ne_zero hupv) (Nat.succ_pos _)
  have hu := uniform_unit (inGates n (circuitRows c hU).nIn) (circuitRows c hU)
    (insOf n (circuitRows c hU).nIn (Aliased.keyIdxP c pi (circuitRows c hU).nIn)) (oneGate n (circuitRows c hU).nIn)
    (insOf_ne_one n (circuitRows c hU).nIn (Aliased.keyIdxP c pi (circuitRows c hU).nIn)) n u hcomp
  intro h
  have hm := Flock.Bridge.mem_unitGates (P := (keyProgP c pi hU n).partition outs) (u := u) |>.2
    ⟨⟨_, gates_add_lt (inGates n (circuitRows c hU).nIn) (circuitRows c hU) u.isLt hcomp⟩, hu, 0,
      Flock.Bridge.one_le_gateSize _, rfl⟩
  rw [h] at hm
  simp at hm

end FlockSoundness.Discharge.Placed

namespace FlockSoundness.Discharge.Layout

open Audit Placed

theorem sum_range_mul_sixteen (f : ℕ → ℕ) : ∀ m, ∑ i ∈ Finset.range m, f i * 16 = 16 * ((List.range m).map f).sum
  | 0 => rfl
  | m + 1 => by
    rw [Finset.sum_range_succ, sum_range_mul_sixteen f m, List.range_succ, List.map_append, List.sum_append]
    simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
    ring

/-- **A unit of the key program commits at most `16` values per output leaf of its `units_per_vu` copies**, at the
returned outputs' bits `outsOfP`. -/
theorem committed_keyProgP_le (c : Flock.Circuit) (pi : HmIn.Ports) (hU : Built c.unit) (n : ℕ)
    (κ : Fin ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)).N → Option Bool) (u : Fin n) :
    Claims.Sampling.committed (Flock.Bridge.progOf ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)) κ)
        (Flock.Bridge.unitGates ((keyProgP c pi hU n).partition (outsOfP c hU n pi)) u) ≤
      16 * ((List.range c.upv).map fun v => (c.leavesOut.getD v #[]).size).sum := by
  classical
  let I := Const.Prog.instC (keyProgP c pi hU n) (outsOfP c hU n pi) u
  let F : Finset (Fin ((keyProgP c pi hU n).rows u).w) :=
    Finset.univ.filter fun x => ∃ u' j t, OutColAt c hU x.val u' j t
  let T : Finset (Σ _ : ℕ, ℕ × ℕ) :=
    (Finset.range c.upv).sigma fun v => Finset.range (c.leavesOut.getD v #[]).size ×ˢ Finset.range 16
  let xOf : (Σ _ : ℕ, ℕ × ℕ) → ℕ := fun v => c.upv * (Rows.ofNet hU).nIn +
    (v.1 * ((Rows.ofNet hU).nComp + 1) + (1 + (outCol c v.2.1 v.2.2 - (Rows.ofNet hU).nIn)))
  have h1 : (Finset.univ.filter fun g => ((keyProgP c pi hU n).partition (outsOfP c hU n pi)).unit g = some u ∧
      g ∈ ((keyProgP c pi hU n).partition (outsOfP c hU n pi)).committed) ⊆ F.image I.wire := by
    intro g hg
    obtain ⟨-, hgu, hgc⟩ := Finset.mem_filter.1 hg
    obtain ⟨i, hi⟩ := I.onto g hgu
    have hio : g ∈ ((keyProgP c pi hU n).partition (outsOfP c hU n pi)).io u := by
      simp only [Audit.Partition.io, Finset.mem_filter]
      exact ⟨hgc, Or.inl hgu⟩
    obtain ⟨u', j, t, ho⟩ := outsOutP_outsOfP c hU n pi u g hio hgu _ hi
    exact Finset.mem_image.2 ⟨_, Finset.mem_filter.2 ⟨Finset.mem_univ _, u', j, t, ho⟩, hi⟩
  have h2 : F.image Fin.val ⊆ T.image xOf := by
    intro x hx
    obtain ⟨y, hy, rfl⟩ := Finset.mem_image.1 hx
    obtain ⟨-, u', j, t, hu', hj, ht, -, -, hxv⟩ := Finset.mem_filter.1 hy
    exact Finset.mem_image.2 ⟨⟨u', (j, t)⟩, Finset.mem_sigma.2 ⟨Finset.mem_range.2 hu',
      Finset.mem_product.2 ⟨Finset.mem_range.2 hj, Finset.mem_range.2 ht⟩⟩, hxv.symm⟩
  calc _ ≤ _ := Flock.Bridge.committed_le_card κ _ u
    _ ≤ (F.image I.wire).card := Finset.card_le_card h1
    _ ≤ F.card := Finset.card_image_le
    _ = (F.image Fin.val).card := (Finset.card_image_of_injective _ Fin.val_injective).symm
    _ ≤ (T.image xOf).card := Finset.card_le_card h2
    _ ≤ T.card := Finset.card_image_le
    _ = ∑ v ∈ Finset.range c.upv, (c.leavesOut.getD v #[]).size * 16 := by
      simp only [T, Finset.card_sigma, Finset.card_product, Finset.card_range]
    _ = _ := sum_range_mul_sixteen _ c.upv

end FlockSoundness.Discharge.Layout
