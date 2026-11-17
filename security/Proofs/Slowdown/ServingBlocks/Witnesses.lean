module

public import Proofs.Slowdown.ServingBlocks
public import Definitions.Slowdown.Guarantees.ServingBlocks.Witnesses

/-! The serving-block bound's honest and nontrivial lemmas, at the instances of
`Definitions/Slowdown/ServingBlocks/Instances.lean`. -/

@[expose] public section

namespace ResourceAccounting.SecurityProofs.ServingBlocks

open Verity Finset ResourceAccounting.ServingBlocks

namespace Counted

variable {C : Circuit} (J : Counted C)

theorem inside_le_card (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (I : Finset (Fin C.N)) : J.inside I ≤ I.card := by
  unfold Counted.inside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  refine Finset.card_le_card fun v h => ?_
  rw [Finset.mem_filter] at h
  exact h.2 (J.rep_mem v h.1)

theorem outside_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) :
    J.outside R B = ((J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i).card : ℝ) := by
  unfold Counted.outside
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

theorem split_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) :
    J.split R B = ((J.V.filter fun v => v ∈ R ∧ (∃ i, v ∈ B i) ∧ ∀ i, ¬ J.body v ⊆ B i).card : ℝ) := by
  unfold Counted.split
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

theorem inside_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (I : Finset (Fin C.N)) :
    J.inside I = ((J.V.filter fun v => J.body v ⊆ I).card : ℝ) := by
  unfold Counted.inside
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

theorem total_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) : J.total = J.V.card := by
  unfold Counted.total
  rw [Finset.sum_congr rfl h1, Finset.sum_const, nsmul_one]

theorem inside_eq_zero {c : ℕ} (hbig : ∀ v ∈ J.V, c < (J.body v).card) {I : Finset (Fin C.N)} (hI : I.card ≤ c) :
    J.inside I = 0 := by
  unfold Counted.inside
  refine Finset.sum_eq_zero fun v hv => ?_
  obtain ⟨hv, hsub⟩ := Finset.mem_filter.1 hv
  exact absurd ((Finset.card_le_card hsub).trans hI) (not_le.2 (hbig v hv))

end Counted

namespace Honest

open ResourceAccounting.ServingBlocks.Honest

theorem ω_one : ∀ v ∈ J.V, J.ω v = 1 := fun _ _ => rfl

theorem ω_le : ∀ v ∈ J.V, J.ω v ≤ 1 := fun _ _ => le_rfl

theorem cap (c k : ℕ) : PerBlockCap d J q c k := fun I _ hc _ =>
  (Counted.inside_le_card J ω_le I).trans (by exact_mod_cast hc)

theorem blocks : ∀ u ∈ S, (P.gates u).card ≤ 1 ∧ (d.unregistered (P.gates u)).card ≤ 1 := by decide

theorem isCK : P.IsCK d 1 1 := ⟨⟨by decide, by decide⟩, by decide, by decide⟩

theorem uncertified : ∀ u ∈ S, UncertifiedCap J (P.gates u) (fun i : {i // r.parent i = u} => P.gates i) 1 := by
  intro u hu
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  norm_cast
  revert u
  decide

theorem dark : UncertifiedCap J ((univ \ S).biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin d.D.N)) 1 := by
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  exact (Nat.cast_le.2 (show _ ≤ 1 by decide)).trans_eq Nat.cast_one

end Honest

namespace Counter

open ResourceAccounting.ServingBlocks.Counter

theorem ω_one : ∀ v ∈ J.V, J.ω v = 1 := fun _ _ => rfl

theorem total_eq : J.total = 1 := by rw [Counted.total_eq J ω_one]; simp

theorem cap : PerBlockCap d J q 1 1 := fun I _ hc _ =>
  (Counted.inside_le_card J (fun _ _ => le_rfl) I).trans (by
    simp only [q, one_ne_zero, ↓reduceIte, Nat.cast_one]
    exact_mod_cast hc)

theorem not_cap_zero : ¬ PerBlockCap d J q 1 0 := fun h => by
  have := h {1} (by decide) (by decide) (by decide)
  rw [Counted.inside_eq J ω_one] at this
  have h1 : (J.V.filter fun v => J.body v ⊆ {1}).card = 1 := by decide
  rw [h1] at this
  norm_num [q] at this

theorem blocks : ∀ u ∈ S, (P.gates u).card ≤ 1 ∧ (d.unregistered (P.gates u)).card ≤ 0 := by decide

theorem isCK : P.IsCK d 1 0 := ⟨⟨by decide, by decide⟩, by decide, by decide⟩

theorem uncertified : ∀ u ∈ S, UncertifiedCap J (P.gates u) (fun i : {i // r.parent i = u} => P.gates i) 0 := by
  intro u hu
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  norm_cast
  revert u
  decide

theorem dark : UncertifiedCap J ((univ \ S).biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin d.D.N)) 0 := by
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  exact (Nat.cast_le.2 (show _ ≤ 0 by decide)).trans_eq Nat.cast_zero

end Counter

namespace Straddle

open ResourceAccounting.ServingBlocks.Straddle

theorem ω_one : ∀ v ∈ J.V, J.ω v = 1 := fun _ _ => rfl

theorem total_eq : J.total = 1 := by rw [Counted.total_eq J ω_one]; simp

theorem cap (k : ℕ) : PerBlockCap d J q 1 k := fun _ _ hc _ =>
  (Counted.inside_eq_zero J (by decide) hc).le

theorem blocks : ∀ u ∈ S, (P.gates u).card ≤ 1 ∧ (d.unregistered (P.gates u)).card ≤ 1 := by decide

theorem isCK : P.IsCK d 1 1 := ⟨⟨by decide, by decide⟩, by decide, by decide⟩

theorem uncertified : ∀ u ∈ S, UncertifiedCap J (P.gates u) (fun i : {i // r.parent i = u} => P.gates i) 1 := by
  intro u hu
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  norm_cast
  revert u
  decide

theorem dark : UncertifiedCap J ((univ \ S).biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin d.D.N)) 0 := by
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  exact (Nat.cast_le.2 (show _ ≤ 0 by decide)).trans_eq Nat.cast_zero

end Straddle

namespace Detached

open ResourceAccounting.ServingBlocks.Detached

theorem ω_one : ∀ v ∈ J.V, J.ω v = 1 := fun _ _ => rfl

theorem total_eq : J.total = 1 := by rw [Counted.total_eq J ω_one]; simp

theorem cap (k : ℕ) : PerBlockCap d J q 1 k := fun _ _ hc _ =>
  (Counted.inside_eq_zero J (by decide) hc).le

theorem blocks : ∀ u ∈ S, (P.gates u).card ≤ 1 ∧ (d.unregistered (P.gates u)).card ≤ 0 := by decide

theorem isCK : P.IsCK d 1 0 := ⟨⟨by decide, by decide⟩, by decide, by decide⟩

theorem uncertified : ∀ u ∈ S, UncertifiedCap J (P.gates u) (fun i : {i // r.parent i = u} => P.gates i) 1 := by
  intro u hu
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  norm_cast
  revert u
  decide

theorem dark : UncertifiedCap J ((univ \ S).biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin d.D.N)) 0 := by
  rw [UncertifiedCap, Counted.outside_eq J ω_one]
  exact (Nat.cast_le.2 (show _ ≤ 0 by decide)).trans_eq Nat.cast_zero

end Detached

theorem ServingBlockBound_honest : Guarantees.ServingBlocks.ServingBlockBound_honest := by
  refine ⟨Honest.blocks, fun _ _ => Honest.cap 1 1, Honest.isCK, Honest.cap 1 1, ?_, ?_, zero_le_one, Honest.ω_le,
    Honest.uncertified, Honest.dark⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide

theorem ServingBlockBound_nontrivial : Guarantees.ServingBlocks.ServingBlockBound_nontrivial := by
  refine ⟨Counter.blocks, fun h => Counter.not_cap_zero (h 0 (Finset.mem_singleton_self 0)),
    ⟨Counter.isCK.boolean, Counter.isCK.card, fun u => (Counter.isCK.reads u).trans (Nat.zero_le 1)⟩, Counter.cap, ?_,
    ?_, zero_le_one, fun _ _ => le_rfl, Counter.uncertified, Counter.dark, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · refine not_le.2 ?_
    rw [Counter.total_eq]
    norm_num [Counter.q]

/-- At `Detached`, whose operation takes in gate 1 though no gate of it reads gate 1, every other premise holds and
the bound fails, so it needs operation bodies. -/
theorem ServingBlockBound_nontrivial_bodies :
    ServingSizes Detached.d Detached.P Detached.S (fun _ => 1) (fun _ => 0) ∧
    (∀ u ∈ Detached.S, PerBlockCap Detached.d Detached.J Detached.q 1 0) ∧ Detached.P.IsCK Detached.d 1 0 ∧
    PerBlockCap Detached.d Detached.J Detached.q 1 0 ∧ ¬ OperationBodies Detached.d Detached.J ∧
    Overlap Detached.J 1 ∧ (0 : ℝ) ≤ 1 ∧ WeightsAtMost Detached.J 1 ∧
    (∀ u ∈ Detached.S, UncertifiedCap Detached.J (Detached.P.gates u)
      (fun i : {i // Detached.r.parent i = u} => Detached.P.gates i) 1) ∧
    UncertifiedCap Detached.J ((univ \ Detached.S).biUnion Detached.P.gates)
      (Fin.elim0 : Fin 0 → Finset (Fin Detached.d.D.N)) 0 ∧
    ¬ Detached.J.total ≤ ∑ u ∈ Detached.S, min (Detached.q 1 0 + 1 * (1 : ℕ) * (0 : ℕ))
      (((univ.filter fun i => Detached.r.parent i = u).card : ℝ) * Detached.q 1 0 + 1) + 0 := by
  refine ⟨Detached.blocks, fun _ _ => Detached.cap 0, Detached.isCK, Detached.cap 0, ?_, ?_, zero_le_one,
    fun _ _ => le_rfl, Detached.uncertified, Detached.dark, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · refine not_le.2 ?_
    rw [Detached.total_eq]
    norm_num [Detached.q]

/-- At `Straddle`, whose operation is split across the wipe, every other premise holds at `μ = 0` and the bound
fails, so it needs overlap and its boundary term. -/
theorem ServingBlockBound_nontrivial_overlap :
    ServingSizes Straddle.d Straddle.P Straddle.S (fun _ => 1) (fun _ => 1) ∧
    (∀ u ∈ Straddle.S, PerBlockCap Straddle.d Straddle.J Straddle.q 1 1) ∧ Straddle.P.IsCK Straddle.d 1 1 ∧
    PerBlockCap Straddle.d Straddle.J Straddle.q 1 1 ∧ OperationBodies Straddle.d Straddle.J ∧
    Overlap Straddle.J 1 ∧ ¬ Overlap Straddle.J 0 ∧ (0 : ℝ) ≤ 1 ∧ WeightsAtMost Straddle.J 1 ∧
    (∀ u ∈ Straddle.S, UncertifiedCap Straddle.J (Straddle.P.gates u)
      (fun i : {i // Straddle.r.parent i = u} => Straddle.P.gates i) 1) ∧
    UncertifiedCap Straddle.J ((univ \ Straddle.S).biUnion Straddle.P.gates)
      (Fin.elim0 : Fin 0 → Finset (Fin Straddle.d.D.N)) 0 ∧
    ¬ Straddle.J.total ≤ ∑ u ∈ Straddle.S, min (Straddle.q 1 1 + 1 * (0 : ℕ) * (1 : ℕ))
      (((univ.filter fun i => Straddle.r.parent i = u).card : ℝ) * Straddle.q 1 1 + 1) + 0 := by
  refine ⟨Straddle.blocks, fun _ _ => Straddle.cap 1, Straddle.isCK, Straddle.cap 1, ?_, ?_, ?_, zero_le_one,
    fun _ _ => le_rfl, Straddle.uncertified, Straddle.dark, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold Overlap; decide
  · refine not_le.2 ?_
    rw [Straddle.total_eq]
    norm_num [Straddle.q]

theorem ServingBlockBoundRemote_honest : Guarantees.ServingBlocks.ServingBlockBoundRemote_honest :=
  ⟨Honest.isCK, Honest.cap 1 1, Honest.uncertified, Honest.dark⟩

theorem ServingBlockBoundRemote_nontrivial : Guarantees.ServingBlocks.ServingBlockBoundRemote_nontrivial := by
  refine ⟨Counter.isCK, Counter.not_cap_zero, Counter.uncertified, Counter.dark, not_le.2 ?_⟩
  rw [Counter.total_eq]
  norm_num [Counter.q]

theorem UncertifiedElementary_honest : Guarantees.ServingBlocks.UncertifiedElementary_honest := by
  refine ⟨?_, ?_⟩
  · rw [Counted.outside_eq Honest.J Honest.ω_one, Counted.split_eq Honest.J Honest.ω_one]
    norm_cast
  · rw [Counted.split_eq Honest.J Honest.ω_one]
    norm_cast

theorem UncertifiedElementary_nontrivial : Guarantees.ServingBlocks.UncertifiedElementary_nontrivial := by
  refine ⟨?_, fun h => ?_⟩
  · rw [Counted.outside_eq Straddle.J Straddle.ω_one, Counted.split_eq Straddle.J Straddle.ω_one]
    norm_cast
  · rw [UncertifiedCap, Counted.outside_eq Straddle.J Straddle.ω_one] at h
    norm_cast at h

example : Honest.J.total ≤ ∑ u ∈ Honest.S, min (Honest.q 1 1 + 1 * (1 : ℕ) * (1 : ℕ))
    (((univ.filter fun i => Honest.r.parent i = u).card : ℝ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := ServingBlockBound_honest
  exact ServingBlockBound Honest.d Honest.J Honest.q 1 1 3 3 Honest.P Honest.S (fun _ => 1) (fun _ => 1) Honest.P
    Honest.r 1 1 (fun _ => 1) 1 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10

example (h2 : ∀ u ∈ Counter.S, PerBlockCap Counter.d Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, h7, h8, h9, h10, hlt⟩ := ServingBlockBound_nontrivial
  exact hlt (ServingBlockBound Counter.d Counter.J Counter.q 1 1 1 1 Counter.P Counter.S (fun _ => 1)
    (fun _ => 0) Counter.P Counter.r 1 1 (fun _ => 0) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10)

example (h5 : OperationBodies Detached.d Detached.J) : False := by
  obtain ⟨h1, h2, h3, h4, -, h6, h7, h8, h9, h10, hlt⟩ := ServingBlockBound_nontrivial_bodies
  exact hlt (ServingBlockBound Detached.d Detached.J Detached.q 1 0 2 2 Detached.P Detached.S (fun _ => 1)
    (fun _ => 0) Detached.P Detached.r 1 1 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10)

example (h6 : Overlap Straddle.J 0) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, -, h7, h8, h9, h10, hlt⟩ := ServingBlockBound_nontrivial_overlap
  exact hlt (ServingBlockBound Straddle.d Straddle.J Straddle.q 1 1 2 2 Straddle.P Straddle.S (fun _ => 1)
    (fun _ => 1) Straddle.P Straddle.r 1 0 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10)

example : Honest.J.total ≤
    ∑ u ∈ Honest.S, (((univ.filter fun i => Honest.r.parent i = u).card : ℝ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4⟩ := ServingBlockBoundRemote_honest
  exact ServingBlockBoundRemote Honest.d Honest.J Honest.q 1 1 3 3 Honest.P Honest.S Honest.P Honest.r (fun _ => 1) 1
    h1 h2 h3 h4

example (h2 : PerBlockCap Counter.d Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, hlt⟩ := ServingBlockBoundRemote_nontrivial
  exact hlt (ServingBlockBoundRemote Counter.d Counter.J Counter.q 1 0 1 1 Counter.P Counter.S Counter.P
    Counter.r (fun _ => 0) 0 h1 h2 h3 h4)

example : UncertifiedCap Honest.J (Honest.P.gates 1) (fun i : {i // Honest.r.parent i = 1} => Honest.P.gates i) 1 := by
  obtain ⟨-, h1⟩ := UncertifiedElementary_honest
  rw [← h1]
  exact UncertifiedElementary _ Honest.J _ _ Honest.P Honest.P Honest.r 1

example (q : ℝ)
    (hq : q < Straddle.J.split (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i)) :
    UncertifiedCap Straddle.J (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i)
      (Straddle.J.split (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i)) ∧
    ¬ UncertifiedCap Straddle.J (Straddle.P.gates 1) (fun i : {i // Straddle.r.parent i = 1} => Straddle.P.gates i)
      q := by
  obtain ⟨heq, -⟩ := UncertifiedElementary_nontrivial
  refine ⟨UncertifiedElementary _ Straddle.J _ _ Straddle.P Straddle.P Straddle.r 1, fun h => ?_⟩
  rw [UncertifiedCap, heq] at h
  exact not_le.2 hq h

end ResourceAccounting.SecurityProofs.ServingBlocks
