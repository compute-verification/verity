import Proofs.Accounting.ServingBlocks
import Definitions.Accounting.Guarantees.ServingBlocks.Witnesses

/-!
# The serving-block bound: honest and nontrivial lemmas

The proofs of `Security/Definitions/Accounting/Guarantees/ServingBlocks/Witnesses.lean`, at the instances of
`Security/Definitions/Accounting/ServingBlocks/Instances.lean`. The `example`s at the end apply each guarantee to its
lemmas, so the lemmas are the premises as the guarantee states them.
-/

namespace Accounting.SecurityProofs.ServingBlocks

open Verity Finset Accounting.ServingBlocks

variable {L σ : Type*}

theorem mem_block {A : Placed L} {X : σ → Finset L} {t : ℕ} {κ : σ → Cuts t} {S : Block κ} {g : Fin A.N} :
    g ∈ block A X κ S ↔
      A.isInput g = false ∧ A.x g ∈ X S.1 ∧ (κ S.1).τ S.2.castSucc ≤ A.t g ∧ A.t g < (κ S.1).τ S.2.succ := by
  simp only [block, SecurityProofs.Actual.mem_region, Set.mem_Ico]

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- With every operation's body holding its representative and every weight at most 1, a set of gates computes at most
its number of gates wholly inside itself. -/
theorem inside_le_card (hv : ∀ v ∈ J.V, v ∈ J.body v) (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (I : Finset (Fin C.N)) :
    J.inside I ≤ I.card := by
  unfold Counted.inside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  refine Finset.card_le_card fun v h => ?_
  rw [Finset.mem_filter] at h
  exact h.2 (hv v h.1)

/-- With every weight at most 1, what the blocks `B` leave of `R` is at most `R`'s number of gates. -/
theorem outside_le_card (hω : ∀ v ∈ J.V, J.ω v ≤ 1) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) : J.outside R B ≤ R.card := by
  unfold Counted.outside
  refine (Finset.sum_le_card_nsmul _ _ 1 fun v h => hω v (Finset.mem_of_mem_filter v h)).trans ?_
  rw [nsmul_one, Nat.cast_le]
  refine Finset.card_le_card fun v h => ?_
  rw [Finset.mem_filter] at h
  exact h.2.1

end Counted

namespace Honest

open Accounting.ServingBlocks.Honest

theorem block_zero : block A X κ ⟨(), 0⟩ = {1} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem block_one : block A X κ ⟨(), 1⟩ = {2} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = {3} := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp only [Set.mem_univ, and_true]
  revert g
  decide

theorem forall_block {P : Block κ → Prop} (h0 : P ⟨(), 0⟩) (h1 : P ⟨(), 1⟩) : ∀ S, P S := by
  rintro ⟨⟨⟩, j⟩
  fin_cases j
  · exact h0
  · exact h1

theorem body_self : ∀ v ∈ J.V, v ∈ J.body v := by decide

theorem ω_le : ∀ v ∈ J.V, J.ω v ≤ 1 := fun _ _ => le_rfl

theorem cap (c k : ℕ) : PerBlockCap A.toCircuit θ J q c k := fun I _ hc _ =>
  (Counted.inside_le_card J body_self ω_le I).trans (by exact_mod_cast hc)

theorem blocks : ∀ S, A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 1 :=
  forall_block (by rw [block_zero]; decide) (by rw [block_one]; decide)

theorem card_block : ∀ S, (block A X κ S).card = 1 :=
  forall_block (by rw [block_zero]; rfl) (by rw [block_one]; rfl)

/-- Gate 2's operation is assigned to the serving block after the wipe and computed wholly inside none of its blocks. -/
theorem straddles : J.outside (block A X κ ⟨(), 1⟩) (inf ⟨(), 1⟩) = 1 := by
  simp only [Honest.inf, block_one, Counted.outside, Finset.sum_const, nsmul_eq_mul, mul_one, Nat.cast_eq_one]
  decide

end Honest

namespace Counter

open Accounting.ServingBlocks.Counter

theorem block_eq (S : Block κ) : block A X κ S = {0} := by
  obtain ⟨⟨⟩, j⟩ := S
  fin_cases j
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = ∅ := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem cap_one (c : ℕ) (hc : 1 ≤ c) : PerBlockCap A.toCircuit θ J q c 1 := fun I _ hI _ =>
  (Counted.inside_le_card J (by decide) (fun _ _ => le_rfl) I).trans (by
    have : (I.card : ℝ) ≤ 1 := by exact_mod_cast (Finset.card_le_univ I).trans (by simp)
    simp only [q, one_ne_zero, ↓reduceIte]
    exact this.trans (by exact_mod_cast hc))

theorem not_cap_zero : ¬ PerBlockCap A.toCircuit θ J q 1 0 := fun h => by
  have := h {0} (by decide) (by decide) (by decide)
  simp [Counted.inside, q] at this

theorem blocks (S : Block κ) : A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 0 := by
  rw [block_eq]
  decide

theorem inside_block (S : Block κ) : J.outside (block A X κ S) (inf S) = 0 := by
  simp only [Counter.inf, block_eq, Counted.outside, Finset.sum_const, nsmul_eq_mul, mul_one, Nat.cast_eq_zero]
  decide

theorem outside_dark : J.outside (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) = 0 := by
  simp only [dark_eq, Counted.outside]
  simp

end Counter

namespace Straddle

open Accounting.ServingBlocks.Straddle

theorem block_zero : block A X κ ⟨(), 0⟩ = {0} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem block_one : block A X κ ⟨(), 1⟩ = {1} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = ∅ := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp

theorem forall_block {P : Block κ → Prop} (h0 : P ⟨(), 0⟩) (h1 : P ⟨(), 1⟩) : ∀ S, P S := by
  rintro ⟨⟨⟩, j⟩
  fin_cases j
  · exact h0
  · exact h1

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem cap (k : ℕ) : PerBlockCap A.toCircuit θ J q 1 k := fun I _ hc _ => by
  unfold Counted.inside
  refine le_of_eq (Finset.sum_eq_zero fun v hv => ?_)
  obtain ⟨hv, hsub⟩ := Finset.mem_filter.1 hv
  obtain rfl := Finset.mem_singleton.1 hv
  exact absurd ((Finset.card_le_card hsub).trans hc) (by decide)

theorem blocks : ∀ S, A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 1 :=
  forall_block (by rw [block_zero]; decide) (by rw [block_one]; decide)

theorem outside_le : ∀ S, J.outside (block A X κ S) (inf S) ≤ 1 := fun S =>
  (Counted.outside_le_card _ (fun _ _ => le_rfl) _ _).trans (by exact_mod_cast (blocks S).1)

theorem outside_dark : J.outside (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) = 0 := by
  simp only [dark_eq, Counted.outside]
  simp

end Straddle

namespace Detached

open Accounting.ServingBlocks.Detached

theorem block_zero : block A X κ ⟨(), 0⟩ = {0} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem block_one : block A X κ ⟨(), 1⟩ = {1} := by
  ext g
  rw [mem_block]
  revert g
  decide

theorem dark_eq : A.region dark Set.univ = ∅ := by
  ext g
  rw [SecurityProofs.Actual.mem_region]
  simp

theorem forall_block {P : Block κ → Prop} (h0 : P ⟨(), 0⟩) (h1 : P ⟨(), 1⟩) : ∀ S, P S := by
  rintro ⟨⟨⟩, j⟩
  fin_cases j
  · exact h0
  · exact h1

theorem total_eq : J.total = 1 := by simp [Counted.total]

theorem cap (k : ℕ) : PerBlockCap A.toCircuit θ J q 1 k := fun I _ hc _ => by
  unfold Counted.inside
  refine le_of_eq (Finset.sum_eq_zero fun v hv => ?_)
  obtain ⟨hv, hsub⟩ := Finset.mem_filter.1 hv
  obtain rfl := Finset.mem_singleton.1 hv
  exact absurd ((Finset.card_le_card hsub).trans hc) (by decide)

theorem blocks : ∀ S, A.size (block A X κ S) ≤ 1 ∧ A.width (A.into (block A X κ S) \ θ) ≤ 0 :=
  forall_block (by rw [block_zero]; decide) (by rw [block_one]; decide)

theorem outside_le : ∀ S, J.outside (block A X κ S) (inf S) ≤ 1 := fun S =>
  (Counted.outside_le_card _ (fun _ _ => le_rfl) _ _).trans (by exact_mod_cast (blocks S).1)

theorem outside_dark : J.outside (A.region dark Set.univ) (Fin.elim0 : Fin 0 → Finset (Fin A.N)) = 0 := by
  simp only [dark_eq, Counted.outside]
  simp

end Detached

theorem ServingBlockBound_honest : Guarantees.ServingBlocks.ServingBlockBound_honest := by
  refine ⟨⟨?_, Honest.blocks⟩, fun _ => Honest.cap 1 1, Honest.cap 1 1,
    fun S _ => ⟨block_subset _ _ _ S, Honest.blocks S⟩, ?_, ?_, fun _ => le_rfl, zero_le_one, Honest.ω_le,
    fun S => ?_, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · exact (Counted.outside_le_card _ Honest.ω_le _ _).trans (by rw [Honest.card_block]; norm_num)
  · exact (Counted.outside_le_card _ Honest.ω_le _ _).trans (by rw [Honest.dark_eq]; norm_num)

theorem ServingBlockBound_nontrivial : Guarantees.ServingBlocks.ServingBlockBound_nontrivial := by
  refine ⟨⟨?_, Counter.blocks⟩, fun h => Counter.not_cap_zero (h ⟨(), 0⟩), Counter.cap_one 1 le_rfl,
    fun S _ => ⟨block_subset _ _ _ S, (Counter.blocks S).1, (Counter.blocks S).2.trans (Nat.zero_le 1)⟩, ?_, ?_,
    fun _ => le_rfl, zero_le_one, fun _ _ => le_rfl, fun S => (Counter.inside_block S).le, Counter.outside_dark.le, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · rw [Counter.total_eq]
    simp [Counter.q]

theorem ServingBlockBound_nontrivial_bodies : Guarantees.ServingBlocks.ServingBlockBound_nontrivial_bodies := by
  refine ⟨⟨?_, Detached.blocks⟩, fun _ => Detached.cap 0, Detached.cap 0,
    fun S _ => ⟨block_subset _ _ _ S, Detached.blocks S⟩, ?_, ?_, fun _ => le_rfl, zero_le_one, fun _ _ => le_rfl,
    Detached.outside_le, Detached.outside_dark.le, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · rw [Detached.total_eq]
    simp [Detached.q]

theorem ServingBlockBound_nontrivial_overlap : Guarantees.ServingBlocks.ServingBlockBound_nontrivial_overlap := by
  refine ⟨⟨?_, Straddle.blocks⟩, fun _ => Straddle.cap 1, Straddle.cap 1,
    fun S _ => ⟨block_subset _ _ _ S, Straddle.blocks S⟩, ?_, ?_, ?_, fun _ => le_rfl, zero_le_one,
    fun _ _ => le_rfl, Straddle.outside_le, Straddle.outside_dark.le, ?_⟩
  · unfold ServingBlocksCover; decide
  · unfold OperationBodies; decide
  · unfold Overlap; decide
  · unfold Overlap; decide
  · rw [Straddle.total_eq]
    simp [Straddle.q]

theorem ServingBlockBoundRemote_honest : Guarantees.ServingBlocks.ServingBlockBoundRemote_honest := by
  obtain ⟨⟨hcov, -⟩, -, hq, hinf, -, -, -, -, -, hU, hD⟩ := ServingBlockBound_honest
  exact ⟨hcov, hq, hinf, hU, hD⟩

theorem ServingBlockBoundRemote_nontrivial : Guarantees.ServingBlocks.ServingBlockBoundRemote_nontrivial := by
  obtain ⟨⟨hcov, -⟩, -, -, -, -, -, -, -, -, hU, hD, -⟩ := ServingBlockBound_nontrivial
  refine ⟨hcov, Counter.not_cap_zero, fun S _ => ⟨block_subset _ _ _ S, Counter.blocks S⟩, hU, hD, ?_⟩
  rw [Counter.total_eq]
  simp [Counter.q]

theorem UncertifiedElementary_honest : Guarantees.ServingBlocks.UncertifiedElementary_honest :=
  ⟨zero_le_one, Honest.ω_le, fun _ _ hg _ => ⟨0, hg⟩, fun _ _ h => absurd h (Finset.notMem_empty _)⟩

theorem UncertifiedElementary_nontrivial : Guarantees.ServingBlocks.UncertifiedElementary_nontrivial := by
  refine ⟨zero_le_one, fun _ _ => le_rfl, fun h => ?_, fun h => ?_⟩
  · obtain ⟨i, _⟩ := h 0 (Finset.mem_singleton_self 0) (Finset.mem_singleton_self 0)
    exact i.elim0
  · simp [UncertifiedCap, Counted.outside, Counted.split] at h

/-! Each lemma is its guarantee's premises as the guarantee states them: the guarantee applies at the honest instance,
and at a counter-example the dropped premise would give the bound the lemma's last conjunct refutes. -/

example : Honest.J.total ≤
    ∑ _S : Block Honest.κ, min (Honest.q 1 1 + 1 * (1 : ℕ) * (1 : ℕ)) ((1 : ℕ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11⟩ := ServingBlockBound_honest
  exact ServingBlockBound Honest.A Honest.θ Honest.J Honest.q 1 1 Honest.dark Honest.X 2 Honest.κ (fun _ => 1)
    (fun _ => 1) (fun _ => 1) Honest.inf 1 1 (fun _ => 1) 1 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11

example (h2 : ∀ _ : Block Counter.κ, PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, h5, h6, h7, h8, h9, h10, h11, hlt⟩ := ServingBlockBound_nontrivial
  exact not_le.2 hlt (ServingBlockBound Counter.A Counter.θ Counter.J Counter.q 1 1 Counter.dark Counter.X 1 Counter.κ
    (fun _ => 1) (fun _ => 0) (fun _ => 1) Counter.inf 1 1 (fun _ => 0) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11)

example (h5 : OperationBodies Detached.A.toCircuit Detached.θ Detached.J) : False := by
  obtain ⟨h1, h2, h3, h4, -, h6, h7, h8, h9, h10, h11, hlt⟩ := ServingBlockBound_nontrivial_bodies
  exact not_le.2 hlt (ServingBlockBound Detached.A Detached.θ Detached.J Detached.q 1 0 Detached.dark Detached.X 2
    Detached.κ (fun _ => 1) (fun _ => 0) (fun _ => 1) Detached.inf 1 1 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11)

example (h6 : Overlap Straddle.J 0) : False := by
  obtain ⟨h1, h2, h3, h4, h5, -, -, h7, h8, h9, h10, h11, hlt⟩ := ServingBlockBound_nontrivial_overlap
  exact not_le.2 hlt (ServingBlockBound Straddle.A Straddle.θ Straddle.J Straddle.q 1 1 Straddle.dark Straddle.X 2
    Straddle.κ (fun _ => 1) (fun _ => 1) (fun _ => 1) Straddle.inf 1 0 (fun _ => 1) 0 h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
    h11)

example : Honest.J.total ≤ ∑ _S : Block Honest.κ, ((1 : ℕ) * Honest.q 1 1 + 1) + 1 := by
  obtain ⟨h1, h2, h3, h4, h5⟩ := ServingBlockBoundRemote_honest
  exact ServingBlockBoundRemote Honest.A Honest.θ Honest.J Honest.q 1 1 Honest.dark Honest.X 2 Honest.κ (fun _ => 1)
    Honest.inf (fun _ => 1) 1 h1 h2 h3 h4 h5

example (h2 : PerBlockCap Counter.A.toCircuit Counter.θ Counter.J Counter.q 1 0) : False := by
  obtain ⟨h1, -, h3, h4, h5, hlt⟩ := ServingBlockBoundRemote_nontrivial
  exact not_le.2 hlt (ServingBlockBoundRemote Counter.A Counter.θ Counter.J Counter.q 1 0 Counter.dark Counter.X 1
    Counter.κ (fun _ => 1) Counter.inf (fun _ => 0) 0 h1 h2 h3 h4 h5)

example (S : Block Honest.κ) :
    UncertifiedCap Honest.J (block Honest.A Honest.X Honest.κ S) (Honest.inf S)
      (1 * (block Honest.A Honest.X Honest.κ S \ block Honest.A Honest.X Honest.κ S).card +
        Honest.J.split (block Honest.A Honest.X Honest.κ S) (Honest.inf S)) := by
  obtain ⟨h1, h2, h3, -⟩ := UncertifiedElementary_honest
  exact UncertifiedElementary _ Honest.J _ _ _ 1 h1 h2 (h3 S)

example (h3 : Alignment ({0} : Finset (Fin Counter.A.N)) (Fin.elim0 : Fin 0 → Finset (Fin Counter.A.N)) {0}) :
    False := by
  obtain ⟨h1, h2, -, hcap⟩ := UncertifiedElementary_nontrivial
  exact hcap (UncertifiedElementary _ Counter.J _ _ _ 1 h1 h2 h3)

end Accounting.SecurityProofs.ServingBlocks
