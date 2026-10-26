import Definitions.Accounting.Guarantees.ServingBlocks
import Proofs.Accounting.Actual.Windows
import Mathlib.Tactic.Linarith

/-!
# The serving-block bound: proofs

`assigned_le` is Theorem 3 on one family of blocks inside a set `R` of gates, with what the family leaves: split `R`'s
operations by whether one of the blocks computes them wholly; those it does weigh at most the blocks' caps summed (the
union bound, `sum_le_sum_sum_filter`, as weights are nonnegative), and the rest is `J.outside R B`. The serving cap is
the family `{S}`, the inference cap the inference blocks counted for `S`, and the dark leaves the empty family.
`total_le` assigns each counted operation to the dark leaves or to the serving block holding its representative: the
server's windows cover `[0, t)` (`cuts_exists_mem`).
-/

namespace Accounting.SecurityProofs.ServingBlocks

open Verity Finset Accounting.ServingBlocks

/-- The union bound: a nonnegative sum over `V` is at most the sum of its parts `P i`, when every `v` lies in one. -/
theorem sum_le_sum_sum_filter {α ι : Type*} (V : Finset α) (s : Finset ι) (P : ι → α → Prop)
    [∀ i, DecidablePred (P i)] (f : α → ℝ) (hf : ∀ v ∈ V, 0 ≤ f v) (hcov : ∀ v ∈ V, ∃ i ∈ s, P i v) :
    ∑ v ∈ V, f v ≤ ∑ i ∈ s, ∑ v ∈ V with P i v, f v := by
  calc ∑ v ∈ V, f v ≤ ∑ v ∈ V, ∑ i ∈ s, if P i v then f v else 0 := by
        refine Finset.sum_le_sum fun v hv => ?_
        obtain ⟨i, hi, hP⟩ := hcov v hv
        calc f v = if P i v then f v else 0 := by simp [hP]
          _ ≤ ∑ j ∈ s, if P j v then f v else 0 :=
            Finset.single_le_sum (f := fun j => if P j v then f v else 0)
              (fun j _ => by split_ifs <;> simp [hf v hv]) hi
    _ = ∑ i ∈ s, ∑ v ∈ V, if P i v then f v else 0 := Finset.sum_comm
    _ = ∑ i ∈ s, ∑ v ∈ V with P i v, f v := Finset.sum_congr rfl fun i _ => (Finset.sum_filter _ _).symm

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- **Theorem 3 on a family of blocks, with what it leaves**: the weight assigned to `R` is at most the number of
blocks times a cap on what each computes wholly inside itself, plus the weight of `R`'s operations computed wholly
inside none of them. -/
theorem assigned_le (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) (Q : ℝ)
    (hB : ∀ i, J.inside (B i) ≤ Q) : J.assigned R ≤ Fintype.card ι * Q + J.outside R B := by
  classical
  have hsplit := Finset.sum_filter_add_sum_filter_not (J.V.filter (· ∈ R)) (fun v => ∃ i, J.body v ⊆ B i) J.ω
  have hout : ∑ v ∈ (J.V.filter (· ∈ R)).filter (fun v => ¬ ∃ i, J.body v ⊆ B i), J.ω v = J.outside R B := by
    unfold Counted.outside
    rw [Finset.filter_filter]
    exact Finset.sum_congr (Finset.filter_congr fun v _ => by simp only [not_exists]) fun _ _ => rfl
  have hin : ∑ v ∈ (J.V.filter (· ∈ R)).filter (fun v => ∃ i, J.body v ⊆ B i), J.ω v ≤
      Fintype.card ι * Q := by
    calc _ ≤ ∑ i, ∑ v ∈ (J.V.filter (· ∈ R)).filter (fun v => ∃ i, J.body v ⊆ B i) with J.body v ⊆ B i,
            J.ω v :=
          sum_le_sum_sum_filter _ univ (fun i v => J.body v ⊆ B i) J.ω
            (fun v hv => J.ω_nonneg v (Finset.mem_of_mem_filter v (Finset.mem_of_mem_filter v hv)))
            (fun v hv => by
              obtain ⟨i, hi⟩ := (Finset.mem_filter.1 hv).2
              exact ⟨i, Finset.mem_univ _, hi⟩)
      _ ≤ ∑ i : ι, J.inside (B i) := by
          refine Finset.sum_le_sum fun i _ => ?_
          unfold Counted.inside
          refine Finset.sum_le_sum_of_subset_of_nonneg ?_ fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)
          intro v hv
          simp only [Finset.mem_filter] at hv ⊢
          exact ⟨hv.1.1.1, hv.2⟩
      _ ≤ ∑ _i : ι, Q := Finset.sum_le_sum fun i _ => hB i
      _ = Fintype.card ι * Q := by simp
  unfold Counted.assigned
  linarith

end Counted

variable {L σ : Type*}

theorem block_subset (A : Placed L) (X : σ → Finset L) {t : ℕ} (κ : σ → Cuts t) (S : Block κ) :
    block A X κ S ⊆ A.computed := fun g hg => by
  simp only [Circuit.computed, Finset.mem_filter, Finset.mem_univ, true_and]
  exact ((SecurityProofs.Actual.mem_region A).1 hg).1

/-- Every counted operation not at a dark leaf lies in a serving block, so the counted weight is at most what the dark
leaves are assigned plus what each serving block is assigned. -/
theorem total_le (A : Placed L) (J : Counted A.toCircuit) (dark : Finset L) (X : σ → Finset L) {t : ℕ}
    [Fintype σ] (κ : σ → Cuts t) (hcov : ServingBlocksCover A J dark X t) :
    J.total ≤ J.assigned (A.region dark Set.univ) + ∑ S : Block κ, J.assigned (block A X κ S) := by
  classical
  set D := A.region dark Set.univ
  have hrest : ∑ v ∈ J.V with v ∉ D, J.ω v ≤ ∑ S : Block κ, J.assigned (block A X κ S) := by
    calc _ ≤ ∑ S : Block κ, ∑ v ∈ J.V.filter (· ∉ D) with v ∈ block A X κ S, J.ω v :=
          sum_le_sum_sum_filter _ univ (fun S v => v ∈ block A X κ S) J.ω
            (fun v hv => J.ω_nonneg v (Finset.mem_of_mem_filter v hv))
            (fun v hv => by
              obtain ⟨hvV, hvD⟩ := Finset.mem_filter.1 hv
              rcases hcov v hvV with hx | ⟨s, hx, ht⟩
              · exact absurd ((SecurityProofs.Actual.mem_region A).2 ⟨J.V_computed v hvV, hx, Set.mem_univ _⟩) hvD
              · obtain ⟨j, hj⟩ := SecurityProofs.Actual.cuts_exists_mem (κ s) ht
                exact ⟨⟨s, j⟩, Finset.mem_univ _, (SecurityProofs.Actual.mem_region A).2 ⟨J.V_computed v hvV, hx, hj⟩⟩)
      _ ≤ _ := by
          refine Finset.sum_le_sum fun S _ => ?_
          unfold Counted.assigned
          rw [Finset.filter_filter]
          refine Finset.sum_le_sum_of_subset_of_nonneg (fun v hv => ?_)
            fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)
          simp only [Finset.mem_filter] at hv ⊢
          exact ⟨hv.1, hv.2.2⟩
  have htot : J.total = J.assigned D + ∑ v ∈ J.V with v ∉ D, J.ω v :=
    (Finset.sum_filter_add_sum_filter_not J.V (· ∈ D) J.ω).symm
  rw [htot]
  linarith

/-- What the dark leaves are assigned is what the empty family of blocks leaves. -/
theorem assigned_dark_le {C : Circuit} (J : Counted C) (D : Finset (Fin C.N)) {qD : ℝ}
    (hD : UncertifiedCap J D (Fin.elim0 : Fin 0 → Finset (Fin C.N)) qD) : J.assigned D ≤ qD := by
  have h := Counted.assigned_le J D (Fin.elim0 : Fin 0 → Finset (Fin C.N)) 0 (fun i => i.elim0)
  simp only [Fintype.card_fin, Nat.cast_zero, zero_mul, zero_add] at h
  exact h.trans hD

/-- **The inference cap** of a set of gates with `m` inference blocks: `Ω_R ≤ m q(c, k) + q_U(R)`. -/
theorem assigned_le_inference {C : Circuit} (θ : Finset (Fin C.N)) (J : Counted C) (q : ℕ → ℕ → ℝ) (c k : ℕ)
    (R : Finset (Fin C.N)) (m : ℕ) (I : Fin m → Finset (Fin C.N)) (qU : ℝ) (hq : PerBlockCap C θ J q c k)
    (hI : ∀ i, I i ⊆ C.computed ∧ C.size (I i) ≤ c ∧ C.width (C.into (I i) \ θ) ≤ k)
    (hU : UncertifiedCap J R I qU) : J.assigned R ≤ (m : ℝ) * q c k + qU := by
  have h := Counted.assigned_le J R I (q c k) fun i => hq (I i) (hI i).1 (hI i).2.1 (hI i).2.2
  rw [Fintype.card_fin] at h
  exact h.trans (add_le_add le_rfl hU)

/-- **The serving cap** of a block `R` of at most `c` gates reading at most `k` unregistered input bits:
`Ω_R ≤ q(c, k) + q_B(R)`, Theorem 3 on its one-block partition. -/
theorem assigned_le_serving {C : Circuit} (θ : Finset (Fin C.N)) (J : Counted C) (q : ℕ → ℕ → ℝ) (c k : ℕ)
    (R : Finset (Fin C.N)) (qB : ℝ) (hq : PerBlockCap C θ J q c k) (hR : R ⊆ C.computed) (hc : C.size R ≤ c)
    (hk : C.width (C.into R \ θ) ≤ k) (hB : UncertifiedCap J R (fun _ : Unit => R) qB) :
    J.assigned R ≤ q c k + qB := by
  have h := Counted.assigned_le J R (fun _ : Unit => R) (q c k) fun _ => hq R hR hc hk
  simp only [Fintype.card_unit, Nat.cast_one, one_mul] at h
  exact h.trans (add_le_add le_rfl hB)

theorem ServingBlockBound : Guarantees.ServingBlocks.ServingBlockBound := by
  intro L σ _ A θ J q c k dark X t κ cS kS n inf qB qU qD hP hqS hq hinf hB hU hD
  have hS := Finset.sum_le_sum fun S (_ : S ∈ (univ : Finset (Block κ))) =>
    le_min (assigned_le_serving θ J q (cS S) (kS S) (block A X κ S) (qB S) (hqS S) (block_subset A X κ S)
        (hP.2 S).1 (hP.2 S).2 (hB S))
      (assigned_le_inference θ J q c k (block A X κ S) (n S) (inf S) (qU S) hq (hinf S) (hU S))
  linarith [total_le A J dark X κ hP.1, assigned_dark_le J _ hD]

theorem ServingBlockBoundRemote : Guarantees.ServingBlocks.ServingBlockBoundRemote := by
  intro L σ _ A θ J q c k dark X t κ n inf qU qD hcov hq hinf hU hD
  have hS := Finset.sum_le_sum fun S (_ : S ∈ (univ : Finset (Block κ))) =>
    assigned_le_inference θ J q c k (block A X κ S) (n S) (inf S) (qU S) hq (hinf S) (hU S)
  linarith [total_le A J dark X κ hcov, assigned_dark_le J _ hD]

theorem UncertifiedElementary : Guarantees.ServingBlocks.UncertifiedElementary := by
  intro C J R ι _ B W ωmax hω0 hω hA
  classical
  unfold UncertifiedCap Counted.outside Counted.split
  rw [← Finset.sum_filter_add_sum_filter_not (J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i)
    (fun v => ∃ i, v ∈ B i) J.ω]
  have hsplit : ∑ v ∈ (J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i).filter (fun v => ∃ i, v ∈ B i), J.ω v =
      ∑ v ∈ J.V with v ∈ R ∧ (∃ i, v ∈ B i) ∧ ∀ i, ¬ J.body v ⊆ B i, J.ω v := by
    rw [Finset.filter_filter]
    exact Finset.sum_congr (Finset.filter_congr fun v _ => by tauto) fun _ _ => rfl
  have hunc : ∑ v ∈ (J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i).filter (fun v => ¬ ∃ i, v ∈ B i), J.ω v ≤
      ωmax * (R \ W).card := by
    set F := (J.V.filter fun v => v ∈ R ∧ ∀ i, ¬ J.body v ⊆ B i).filter (fun v => ¬ ∃ i, v ∈ B i)
    have hF : F ⊆ R \ W := fun v hv => by
      simp only [F, Finset.mem_filter] at hv
      exact Finset.mem_sdiff.2 ⟨hv.1.2.1, fun hW => hv.2 (hA v hv.1.2.1 hW)⟩
    calc ∑ v ∈ F, J.ω v ≤ ∑ _v ∈ F, ωmax :=
          Finset.sum_le_sum fun v hv => hω v (Finset.mem_of_mem_filter v (Finset.mem_of_mem_filter v hv))
      _ = F.card * ωmax := by rw [Finset.sum_const, nsmul_eq_mul]
      _ ≤ (R \ W).card * ωmax := mul_le_mul_of_nonneg_right (by exact_mod_cast Finset.card_le_card hF) hω0
      _ = ωmax * (R \ W).card := mul_comm _ _
  rw [hsplit]
  linarith

end Accounting.SecurityProofs.ServingBlocks
