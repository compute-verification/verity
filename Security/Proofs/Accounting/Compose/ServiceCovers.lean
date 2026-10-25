import Definitions.Accounting.Guarantees.Compose
import Proofs.Accounting.TS.Drawn

/-!
# `ServiceCovers`: the service's share covers every tile that isn't good

`Accounting.SecurityProofs.TS.DrawnLayer.covers_of_cone` at the tiles' words reduces `Covers` to the unpaid tiles'
layer cones against the game's deficiency, `Σ_{t unpaid} tileCone t`. A layer late edge into tile `v` at slot `k`
(`DrawnLayer.Late`) has its position written by `u` (`Declared.Writes`, `u` its unit) and in `v`'s slots: either one
of `v`'s own gates, so `u = v`, or an operand of a gate `j` of `v`'s, whose round is `v`'s (`Declared.Wf`). Under
`LayerCommitted` the position is committed by its IR cut, so the layer's cut is the IR's and `j`'s round is before it,
`Declared.Late` and a `Declared.UnitLate` edge; or it is a computed value that only gates of its own unit read, so `j`'s
unit `v` is `u`, and the edge adds nothing to the cone. Along `ReflTransGen` the layer's cone lies in
`Declared.Reaches`, and a tile's cone weight is at most `N_w` times
its `tileCone` (`0` when `N_w = 0`: a tile's words are among the circuit's).
-/

namespace Accounting.SecurityProofs.Compose.CoversProof

open Accounting.TS Accounting.Compose Pouw.RelativeSecurity

/-- A tile's words are among the circuit's. -/
theorem tileWords_le_wordsOf {ι K : Type} (Sc : Scheme K) (Cd : Declared ι) (Np : ℕ) (t : Fin Np) :
    tileWords Sc Cd Np t ≤ wordsOf Sc Cd := by
  unfold tileWords wordsOf unitWords creditedWords
  apply List.Sublist.length_le
  apply List.Sublist.filter
  apply List.filter_sublist

theorem mem_unitGates_iff {ι : Type} (Cd : Declared ι) (u j : ℕ) :
    j ∈ unitGates Cd u ↔ j < Cd.dag.length ∧ Cd.unit j = u := by
  simp [unitGates]

/-- Under `Declared.Wf` a gate's unit's round is the gate's, and one of the circuit's. -/
theorem roundOf_unit {ι : Type} (Cd : Declared ι) (hw : Cd.Wf) {j : ℕ} (hj : j < Cd.dag.length) :
    Cd.roundOf (Cd.unit j) = Cd.round (Cd.call j) ∧ Cd.round (Cd.call j) < Cd.rounds := by
  refine ⟨?_, hw.2.1 _ (hw.1 j hj).1⟩
  unfold Declared.roundOf
  rcases hl : unitGates Cd (Cd.unit j) with _ | ⟨i, l⟩
  · have hm : j ∈ unitGates Cd (Cd.unit j) := (mem_unitGates_iff Cd _ j).mpr ⟨hj, rfl⟩
    rw [hl] at hm
    cases hm
  · have hi : i ∈ unitGates Cd (Cd.unit j) := by rw [hl]; exact List.mem_cons_self
    obtain ⟨hi1, hi2⟩ := (mem_unitGates_iff Cd _ i).mp hi
    exact hw.2.2.1 i j hi1 hj hi2

section Layer

variable {ι C Ξ V : Type} (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
  (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)

theorem src_eq (ℓ : ℕ) : (Cd.layer hr hK pr).src ℓ =
    if h : ℓ < Cd.dag.length ∧ Cd.binding ℓ ≠ some .coin ∧ Cd.unit ℓ < Np then some (0, ⟨Cd.unit ℓ, h.2.2⟩)
    else none := rfl

theorem pos_mem (w : Fin 1) (t : Fin Np) (k : ℕ) :
    (Cd.layer hr hK pr).pos w t k ∈ Cd.slots t.val ∨
      (Cd.slots t.val = [] ∧ (Cd.layer hr hK pr).pos w t k = Cd.dag.length) := by
  show (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) ∈ Cd.slots t.val ∨
    (Cd.slots t.val = [] ∧
      (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) = Cd.dag.length)
  cases hl : Cd.slots t.val with
  | nil => exact Or.inr ⟨rfl, rfl⟩
  | cons a l =>
    refine Or.inl ?_
    cases hk : (a :: l)[k]? with
    | none => exact List.mem_cons_self
    | some b => exact List.mem_iff_getElem?.mpr ⟨k, hk⟩

/-- A position of tile `v`'s slots written by `u` after `v`'s coin is an IR late write from `u`'s unit to `v`'s, or
`u` is `v`. -/
theorem reaches_of_slot (hw : Cd.Wf) (hc : LayerCommitted Cd) (u v : Fin 1 × Fin Np) (ℓ : ℕ)
    (hmem : ℓ ∈ Cd.slots v.2.val ∨ (Cd.slots v.2.val = [] ∧ ℓ = Cd.dag.length))
    (hsrc : (Cd.layer hr hK pr).src ℓ = some u)
    (hlt : min (Cd.roundOf v.2.val) (Cd.rounds - 1) < min (max (Cd.cutOf ℓ) (Cd.commitRound ℓ)) Cd.rounds) :
    Cd.Reaches u.2.val v.2.val := by
  rw [src_eq] at hsrc
  split at hsrc
  · rename_i hs
    obtain rfl := Option.some.inj hsrc
    show Cd.Reaches (Cd.unit ℓ) v.2.val
    have hW : Cd.Writes ℓ := ⟨hs.1, hs.2.1⟩
    rcases hmem with hm | ⟨-, hl⟩
    · rcases List.mem_append.mp hm with hg | ha
      · have hu := ((mem_unitGates_iff Cd _ _).mp hg).2
        exact hu ▸ Relation.ReflTransGen.refl
      · obtain ⟨j, hj, hlj⟩ := List.mem_flatMap.mp ha
        obtain ⟨hjl, hju⟩ := (mem_unitGates_iff Cd _ _).mp hj
        have hread : Cd.Reads j ℓ := by
          unfold Declared.Reads
          rcases hd : Cd.dag[j]? with _ | g
          · rw [hd] at hlj
            simp at hlj
          · rw [hd] at hlj
            exact List.contains_iff_mem.mpr hlj
        rcases hc ℓ hW with hcm | ⟨-, -, hown⟩
        · obtain ⟨hR, hR2⟩ := roundOf_unit Cd hw hjl
          rw [hju] at hR
          refine Relation.ReflTransGen.single ⟨ℓ, j, hW, rfl, hju, hread, ?_⟩
          omega
        · have hv : Cd.unit ℓ = v.2.val := (hown j hread).symm.trans hju
          exact hv ▸ Relation.ReflTransGen.refl
    · exact absurd hs.1 (by rw [hl]; exact lt_irrefl _)
  · cases hsrc

/-- **A layer late edge is in the IR's late cone**: under `Declared.Wf` and `LayerCommitted`. -/
theorem reaches_of_late (hw : Cd.Wf) (hc : LayerCommitted Cd) (u v : Fin 1 × Fin Np)
    (h : (Cd.layer hr hK pr).Late u v) : Cd.Reaches u.2.val v.2.val := by
  obtain ⟨k, hsrc, hcut⟩ := h
  refine reaches_of_slot Cd hr hK pr hw hc u v _ (pos_mem Cd hr hK pr v.1 v.2 k) hsrc ?_
  rcases hcut with h0 | ⟨-, h1⟩
  · have h0' := Fin.lt_def.mp h0
    have := ((Cd.layer hr hK pr).cut ((Cd.layer hr hK pr).pos v.1 v.2 k)).1.isLt
    omega
  · exact Fin.lt_def.mp h1

theorem reaches_of_rtg (hw : Cd.Wf) (hc : LayerCommitted Cd) {u v : Fin 1 × Fin Np}
    (h : Relation.ReflTransGen (Cd.layer hr hK pr).Late u v) : Cd.Reaches u.2.val v.2.val := by
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact Relation.ReflTransGen.trans ih (reaches_of_late Cd hr hK pr hw hc _ _ hbc)

/-- **A tile's layer cone weighs at most `N_w` times its `tileCone`.** -/
theorem coneWt_le {K : Type} (Sc : Scheme K) (hw : Cd.Wf) (hc : LayerCommitted Cd) (t : Fin Np) :
    (Cd.layer hr hK pr).coneWt (fun _ t => (tileWords Sc Cd Np t : ℝ)) (0, t) ≤
      tileCone Sc Cd Np t * (wordsOf Sc Cd : ℝ) := by
  classical
  simp only [DrawnLayer.coneWt, tileCone, Fintype.sum_prod_type, Fin.sum_univ_one, Finset.sum_mul]
  refine Finset.sum_le_sum fun t' _ => ?_
  by_cases hR : Relation.ReflTransGen (Cd.layer hr hK pr).Late (0, t) (0, t')
  · rw [ite_eq_left hR, ite_eq_left (reaches_of_rtg Cd hr hK pr hw hc hR)]
    unfold tileWeight
    rcases Nat.eq_zero_or_pos (wordsOf Sc Cd) with h0 | hpos
    · have h00 : tileWords Sc Cd Np t' = 0 := Nat.le_zero.mp (h0 ▸ tileWords_le_wordsOf Sc Cd Np t')
      simp [h0, h00]
    · exact (div_mul_cancel₀ _ (Nat.cast_pos.mpr hpos).ne').ge
  · rw [ite_eq_right hR]
    refine mul_nonneg ?_ (Nat.cast_nonneg _)
    split_ifs
    · exact div_nonneg (Nat.cast_nonneg _) (Nat.cast_nonneg _)
    · exact le_rfl

end Layer

/-- `ServiceCovers`: `covers_of_cone` at the tiles' words, each unpaid tile's layer cone within `N_w` times its
`tileCone` (`coneWt_le`), and the game's deficiency the unpaid tiles' `tileCone`. -/
theorem serviceCovers : Accounting.Guarantees.Compose.ServiceCovers := by
  intro ι K T L U C Ξ V _ _ _ _ _ _ Sc Cd Np Kd hr hK R Pl pr hw hc hG
  refine Accounting.SecurityProofs.TS.DrawnLayer.covers_of_cone (Cd.layer hr hK pr) Pl (jointMode R)
    (fun _ t => (tileWords Sc Cd Np t : ℝ)) (wordsOf Sc Cd : ℝ) (fun _ _ => Nat.cast_nonneg _) fun x _ => ?_
  rw [hG]
  simp only [jointGame, RoundGame.deficiency, JointRun.share, one_mul, add_zero, Fintype.sum_prod_type,
    Fin.sum_univ_one]
  rw [← Finset.sum_add_distrib, Finset.sum_mul]
  refine Finset.sum_le_sum fun t _ => ?_
  have hm : jointMode R x (0, t) = R.mode x.1.1 t := rfl
  rw [hm]
  rcases hmode : R.mode x.1.1 t with _ | _ | _
  · simp
  · simpa using coneWt_le Cd hr hK pr Sc hw hc t
  · simpa using coneWt_le Cd hr hK pr Sc hw hc t

end Accounting.SecurityProofs.Compose.CoversProof
