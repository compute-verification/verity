import Proofs.Accounting.ServingBlocks
import Definitions.Accounting.Guarantees.ServingBlocks.Witnesses

/-!
# The serving-block bound: honest and nontrivial lemmas

The proofs of `Security/Definitions/Accounting/Guarantees/ServingBlocks/Witnesses.lean`, at the instances of
`Security/Definitions/Accounting/ServingBlocks/Instances.lean`. Every instance weighs each counted operation 1, so each
count is a number of operations, which `decide` computes. The `example`s at the end apply each guarantee to its lemmas,
so the lemmas are the premises as the guarantee states them.
-/

namespace Accounting.SecurityProofs.ServingBlocks

open Verity Finset Accounting.ServingBlocks

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- With every weight at most 1, a set of gates computes at most its number of gates wholly inside itself: each
operation's body holds its representative. -/
theorem inside_le_card (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (I : Finset (Fin C.N)) : J.inside I ≤ I.card := by
  unfold Counted.inside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  refine Finset.card_le_card fun v h => ?_
  rw [Finset.mem_filter] at h
  exact h.2 (J.rep_mem v h.1)

/-- With every weight 1, what the blocks `B` leave of `R` is the number of such operations. -/
theorem outside_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) :
    J.outside R B = ((J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i).card : ℝ) := by
  unfold Counted.outside
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

/-- With every weight 1, the operations split between a block `B i` and gates outside it are as many as their number. -/
theorem split_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) :
    J.split R B = ((J.V.filter fun v => v ∈ R ∧ (∃ i, v ∈ B i) ∧ ∀ i, ¬ J.body v ⊆ B i).card : ℝ) := by
  unfold Counted.split
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

/-- With every weight 1, a set of gates computes as much wholly inside itself as the number of such operations. -/
theorem inside_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) (I : Finset (Fin C.N)) :
    J.inside I = ((J.V.filter fun v => J.body v ⊆ I).card : ℝ) := by
  unfold Counted.inside
  rw [Finset.sum_congr rfl fun v hv => h1 v (Finset.mem_of_mem_filter v hv), Finset.sum_const, nsmul_one]

/-- With every weight 1, the counted weight is the number of counted operations. -/
theorem total_eq (h1 : ∀ v ∈ J.V, J.ω v = 1) : J.total = J.V.card := by
  unfold Counted.total
  rw [Finset.sum_congr rfl h1, Finset.sum_const, nsmul_one]

/-- No block of at most `c` gates computes wholly an operation whose body has more than `c` gates. -/
theorem inside_eq_zero {c : ℕ} (hbig : ∀ v ∈ J.V, c < (J.body v).card) {I : Finset (Fin C.N)} (hI : I.card ≤ c) :
    J.inside I = 0 := by
  unfold Counted.inside
  refine Finset.sum_eq_zero fun v hv => ?_
  obtain ⟨hv, hsub⟩ := Finset.mem_filter.1 hv
  exact absurd ((Finset.card_le_card hsub).trans hI) (not_le.2 (hbig v hv))

end Counted

namespace Honest

open Accounting.ServingBlocks.Honest

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

open Accounting.ServingBlocks.Counter

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

open Accounting.ServingBlocks.Straddle

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

open Accounting.ServingBlocks.Detached

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
  · rw [Counter.total_eq]
    norm_num [Counter.q]

theorem ServingBlockBound_nontrivial_bodies : Guarantees.ServingBlocks.ServingBlockBound_nontrivial_bodies := by
  refine ⟨Detached.blocks, fun _ _ => Detached.cap 0, Detached.isCK, Detached.cap 0, ?_, ?_, zero_le_one,
    fun _ _ => le_rfl, Detached.uncertified, Detached.dark, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · rw [Detached.total_eq]
    norm_num [Detached.q]

theorem ServingBlockBound_nontrivial_overlap : Guarantees.ServingBlocks.ServingBlockBound_nontrivial_overlap := by
  refine ⟨Straddle.blocks, fun _ _ => Straddle.cap 1, Straddle.isCK, Straddle.cap 1, ?_, ?_, ?_, zero_le_one,
    fun _ _ => le_rfl, Straddle.uncertified, Straddle.dark, ?_⟩
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold Overlap; decide
  · rw [Straddle.total_eq]
    norm_num [Straddle.q]

theorem ServingBlockBoundRemote_honest : Guarantees.ServingBlocks.ServingBlockBoundRemote_honest :=
  ⟨Honest.isCK, Honest.cap 1 1, Honest.uncertified, Honest.dark⟩

theorem ServingBlockBoundRemote_nontrivial : Guarantees.ServingBlocks.ServingBlockBoundRemote_nontrivial := by
  refine ⟨Counter.isCK, Counter.not_cap_zero, Counter.uncertified, Counter.dark, ?_⟩
  rw [Counter.total_eq]
  norm_num [Counter.q]

theorem UncertifiedElementary_honest : Guarantees.ServingBlocks.UncertifiedElementary_honest := ⟨Honest.r⟩

theorem UncertifiedElementary_nontrivial : Guarantees.ServingBlocks.UncertifiedElementary_nontrivial := by
  refine ⟨by decide, fun h => ?_⟩
  rw [UncertifiedCap, Counted.outside_eq Counter.J Counter.ω_one, Counted.split_eq Counter.J Counter.ω_one] at h
  exact absurd (Nat.cast_le.1 h) (by decide)

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at a counter-example the dropped premise would give the bound the lemma's last conjunct refutes. -/

example : Honest.J.total ≤ ∑ u ∈ Honest.S, min (Honest.q 1 1 + 1 * (1 : ℕ) * (1 : ℕ))
    (((univ.filter fun i => Honest.r.parent i = u).card : ℝ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10⟩ := ServingBlockBound_honest
  exact ServingBlockBound Honest.d Honest.J Honest.q 1 1 3 3 Honest.P Honest.S (fun _ => 1) (fun _ => 1) Honest.P
    Honest.r 1 1 (fun _ => 1) 1 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10

example (h2 : ∀ u ∈ Counter.S, PerBlockCap Counter.d Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, h7, h8, h9, h10, hlt⟩ := ServingBlockBound_nontrivial
  exact not_le.2 hlt (ServingBlockBound Counter.d Counter.J Counter.q 1 1 1 1 Counter.P Counter.S (fun _ => 1)
    (fun _ => 0) Counter.P Counter.r 1 1 (fun _ => 0) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10)

example (h5 : OperationBodies Detached.d Detached.J) : False := by
  obtain ⟨h1, h2, h3, h4, -, h6, h7, h8, h9, h10, hlt⟩ := ServingBlockBound_nontrivial_bodies
  exact not_le.2 hlt (ServingBlockBound Detached.d Detached.J Detached.q 1 0 2 2 Detached.P Detached.S (fun _ => 1)
    (fun _ => 0) Detached.P Detached.r 1 1 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10)

example (h6 : Overlap Straddle.J 0) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, -, h7, h8, h9, h10, hlt⟩ := ServingBlockBound_nontrivial_overlap
  exact not_le.2 hlt (ServingBlockBound Straddle.d Straddle.J Straddle.q 1 1 2 2 Straddle.P Straddle.S (fun _ => 1)
    (fun _ => 1) Straddle.P Straddle.r 1 0 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10)

example : Honest.J.total ≤
    ∑ u ∈ Honest.S, (((univ.filter fun i => Honest.r.parent i = u).card : ℝ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4⟩ := ServingBlockBoundRemote_honest
  exact ServingBlockBoundRemote Honest.d Honest.J Honest.q 1 1 3 3 Honest.P Honest.S Honest.P Honest.r (fun _ => 1) 1
    h1 h2 h3 h4

example (h2 : PerBlockCap Counter.d Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, hlt⟩ := ServingBlockBoundRemote_nontrivial
  exact not_le.2 hlt (ServingBlockBoundRemote Counter.d Counter.J Counter.q 1 0 1 1 Counter.P Counter.S Counter.P
    Counter.r (fun _ => 0) 0 h1 h2 h3 h4)

example : ∃ r : Honest.P.Refines Honest.P, ∀ u, UncertifiedCap Honest.J (Honest.P.gates u)
    (fun i : {i // r.parent i = u} => Honest.P.gates i)
    (Honest.J.split (Honest.P.gates u) (fun i : {i // r.parent i = u} => Honest.P.gates i)) := by
  obtain ⟨r⟩ := UncertifiedElementary_honest
  exact ⟨r, UncertifiedElementary _ Honest.J _ _ _ _ r⟩

example (h : ∀ g, Counter.P₂.unit g = (Counter.P.unit g).map Counter.away) : False := by
  obtain ⟨-, hcap⟩ := UncertifiedElementary_nontrivial
  exact hcap (UncertifiedElementary _ Counter.J _ _ Counter.P Counter.P₂ ⟨Counter.away, h⟩ 0)

end Accounting.SecurityProofs.ServingBlocks
