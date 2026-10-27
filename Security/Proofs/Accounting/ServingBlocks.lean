import Definitions.Accounting.Guarantees.ServingBlocks
import Proofs.Core.Padding
import Mathlib.Tactic.Linarith

/-!
# The serving-block bound: proofs

Each serving block's two caps are summed by `Verity.SecurityProofs.ServingBlockSum`, beside the dark remainder's.
* The serving cap (`assigned_le_serving`) is Theorem 3 on the one block `u`: what `u` computes wholly inside itself is at
  most `q(c_S, k_S)`, and the rest, the operations assigned to `u` with a gate outside it, `boundary_le` charges to
  `in(u) \ θ`, at most `μ` operations per gate.
* The inference cap (`assigned_le_inference`) is `Verity.SecurityProofs.InferenceCap` with `U` the operations computed
  wholly inside no inference block: an operation outside `U` is wholly inside the inference block holding its
  representative, which the per-block cap at `(c, k)` bounds.
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

theorem V_subset : J.V ⊆ C.computed := fun v hv => by
  simpa [Circuit.computed] using J.V_computed v hv

/-- A partition's sum over `J.V ∩ R` is `R`'s local count. -/
theorem sum_inter (R : Finset (Fin C.N)) : ∑ v ∈ J.V ∩ R, J.ω v = J.assigned R := by
  rw [Counted.assigned, Finset.filter_mem_eq_inter]

/-- **Theorem 3 on one block, with what it leaves**: `R`'s local count is what `R` computes wholly inside itself plus
the weight of its operations not computed wholly inside it. -/
theorem assigned_le (R : Finset (Fin C.N)) : J.assigned R ≤ J.inside R + J.outside R (fun _ : Unit => R) := by
  classical
  unfold Counted.assigned Counted.inside Counted.outside
  rw [← Finset.sum_filter_add_sum_filter_not (J.V.filter (· ∈ R)) (fun v => J.body v ⊆ R) J.ω, Finset.filter_filter,
    Finset.filter_filter]
  refine add_le_add (Finset.sum_le_sum_of_subset_of_nonneg (fun v hv => ?_)
    fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)) (le_of_eq (Finset.sum_congr
      (Finset.filter_congr fun v _ => by simp) fun _ _ => rfl))
  simp only [Finset.mem_filter] at hv ⊢
  exact ⟨hv.1, hv.2.2⟩

end Counted

/-- An operation assigned to `R` but not computed wholly inside it has a gate in `in(R) \ θ`: of its gates outside `R`
take the last; it is not the representative, so another gate of the operation reads it, a later one, which is in `R`. -/
theorem exists_crossing {d : Declaration} {J : Counted d.D} (hB : OperationBodies d J) {R : Finset (Fin d.D.N)}
    {v : Fin d.D.N} (hv : v ∈ J.V) (hvR : v ∈ R) (hout : ¬ J.body v ⊆ R) :
    ∃ a ∈ d.unregistered R, a ∈ J.body v := by
  classical
  set T := (J.body v).filter (· ∉ R)
  have hT : T.Nonempty := by
    obtain ⟨u, hu, huR⟩ := Finset.not_subset.1 hout
    exact ⟨u, Finset.mem_filter.2 ⟨hu, huR⟩⟩
  obtain ⟨hub, huR⟩ := Finset.mem_filter.1 (T.max'_mem hT)
  obtain ⟨hθ, hfeed⟩ := hB v hv _ hub
  obtain ⟨g, hg, hug⟩ := hfeed fun h => huR (h ▸ hvR)
  have hgR : g ∈ R := by
    by_contra hgR
    exact not_le.2 (d.D.reads_lt g _ hug) (T.le_max' g (Finset.mem_filter.2 ⟨hg, hgR⟩))
  refine ⟨_, Finset.mem_sdiff.2 ⟨?_, hθ⟩, hub⟩
  simp only [Circuit.into, Finset.mem_filter, Finset.mem_univ, true_and]
  exact ⟨huR, g, hgR, hug⟩

/-- **The boundary charge**: the weight assigned to `R` that is not computed wholly inside `R` is at most
`ω_max μ |in(R) \ θ|`, each such operation charged to one of its gates in `in(R) \ θ`, each in at most `μ` of them. -/
theorem boundary_le (d : Declaration) (J : Counted d.D) (R : Finset (Fin d.D.N)) (ωmax : ℝ) (μ : ℕ)
    (hB : OperationBodies d J) (hμ : Overlap J μ) (hω0 : 0 ≤ ωmax) (hω : ∀ v ∈ J.V, J.ω v ≤ ωmax) :
    J.outside R (fun _ : Unit => R) ≤ ωmax * μ * (d.unregistered R).card := by
  classical
  unfold Counted.outside
  calc _ ≤ ∑ a ∈ d.unregistered R,
          ∑ v ∈ J.V.filter (fun v => v ∈ R ∧ ∀ _ : Unit, ¬ J.body v ⊆ R) with a ∈ J.body v, J.ω v :=
        sum_le_sum_sum_filter _ _ (fun a v => a ∈ J.body v) J.ω
          (fun v hv => J.ω_nonneg v (Finset.mem_of_mem_filter v hv))
          (fun v hv => by
            obtain ⟨hvV, hvR, hout⟩ := Finset.mem_filter.1 hv
            exact exists_crossing hB hvV hvR (hout ()))
    _ ≤ ∑ _a ∈ d.unregistered R, ωmax * μ := by
        refine Finset.sum_le_sum fun a _ => ?_
        calc _ ≤ ∑ v ∈ J.V with a ∈ J.body v, J.ω v := by
              rw [Finset.filter_filter]
              refine Finset.sum_le_sum_of_subset_of_nonneg (fun v hv => ?_)
                fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)
              simp only [Finset.mem_filter] at hv ⊢
              exact ⟨hv.1, hv.2.2⟩
          _ ≤ ∑ v ∈ J.V with a ∈ J.body v, ωmax :=
              Finset.sum_le_sum fun v hv => hω v (Finset.mem_of_mem_filter v hv)
          _ = (J.V.filter fun v => a ∈ J.body v).card * ωmax := by rw [Finset.sum_const, nsmul_eq_mul]
          _ ≤ μ * ωmax := mul_le_mul_of_nonneg_right (by exact_mod_cast hμ a) hω0
          _ = ωmax * μ := mul_comm _ _
    _ = ωmax * μ * (d.unregistered R).card := by rw [Finset.sum_const, nsmul_eq_mul, mul_comm]

/-- **The serving cap** of a block `R` of at most `c` gates reading at most `k` unregistered input bits:
`Ω_R ≤ q(c, k) + ω_max μ k`, Theorem 3 on the one block with the boundary charge. -/
theorem assigned_le_serving (d : Declaration) (J : Counted d.D) (q : ℕ → ℕ → ℝ) (c k : ℕ) (R : Finset (Fin d.D.N))
    (ωmax : ℝ) (μ : ℕ) (hq : PerBlockCap d J q c k) (hR : R ⊆ d.D.computed) (hc : R.card ≤ c)
    (hk : (d.unregistered R).card ≤ k) (hB : OperationBodies d J) (hμ : Overlap J μ) (hω0 : 0 ≤ ωmax)
    (hω : ∀ v ∈ J.V, J.ω v ≤ ωmax) :
    J.assigned R ≤ q c k + ωmax * μ * k := by
  have hcard : ((d.unregistered R).card : ℝ) ≤ k := by exact_mod_cast hk
  have := mul_le_mul_of_nonneg_left hcard (mul_nonneg hω0 (Nat.cast_nonneg μ))
  linarith [Counted.assigned_le J R, hq R hR hc hk, boundary_le d J R ωmax μ hB hμ hω0 hω]

/-- **The inference cap** of a unit `u` of `P` with the `(c, k)`-partition `Pf` refining `P`:
`Ω_u ≤ n_u q(c, k) + (what the inference blocks inside u leave of u)`, `Verity.SecurityProofs.InferenceCap` with `U` the
operations computed wholly inside no inference block. -/
theorem assigned_le_inference (d : Declaration) (J : Counted d.D) (q : ℕ → ℕ → ℝ) (c k : ℕ) {B nf : ℕ}
    (P : Partition d.D B) (Pf : Partition d.D nf) (r : Pf.Refines P) (hCK : Pf.IsCK d c k)
    (hq : PerBlockCap d J q c k) (u : Fin B) :
    ∑ v ∈ J.V ∩ P.gates u, J.ω v ≤ ((univ.filter fun i => r.parent i = u).card : ℝ) * q c k +
      J.outside (P.gates u) (fun i : {i // r.parent i = u} => Pf.gates i) := by
  classical
  set U := J.V.filter fun v => ∀ i, ¬ J.body v ⊆ Pf.gates i
  have hcap : ∀ i, ∑ v ∈ (J.V \ U) ∩ Pf.gates i, J.ω v ≤ q c k := fun i => by
    refine le_trans ?_ (hq (Pf.gates i) (Pf.gates_subset_computed i) (hCK.card i).le (hCK.reads i))
    unfold Counted.inside
    refine Finset.sum_le_sum_of_subset_of_nonneg (fun v hv => ?_)
      fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)
    obtain ⟨hvV, hvU⟩ := Finset.mem_sdiff.1 (Finset.mem_inter.1 hv).1
    have hvi := (Finset.mem_inter.1 hv).2
    obtain ⟨j, hj⟩ : ∃ j, J.body v ⊆ Pf.gates j := by
      by_contra h
      exact hvU (Finset.mem_filter.2 ⟨hvV, fun j hj => h ⟨j, hj⟩⟩)
    have hvj := hj (J.rep_mem v hvV)
    rw [Partition.mem_gates] at hvi hvj
    obtain rfl : i = j := Option.some_injective _ (hvi.symm.trans hvj)
    exact Finset.mem_filter.2 ⟨hvV, hj⟩
  refine (Verity.SecurityProofs.InferenceCap d.D nf B Pf P r J.V U J.ω (q c k) (Counted.V_subset J) hcap u).trans
    (add_le_add le_rfl ?_)
  unfold Counted.outside
  refine Finset.sum_le_sum_of_subset_of_nonneg (fun v hv => ?_)
    fun v hv _ => J.ω_nonneg v (Finset.mem_of_mem_filter v hv)
  obtain ⟨hvVU, hvu⟩ := Finset.mem_inter.1 hv
  obtain ⟨hvV, hvU⟩ := Finset.mem_inter.1 hvVU
  exact Finset.mem_filter.2 ⟨hvV, hvu, fun i => (Finset.mem_filter.1 hvU).2 i⟩

/-- What the dark remainder's units are assigned is what the empty family of blocks leaves of their gates. -/
theorem dark_le {C : Circuit} (J : Counted C) {B : ℕ} (P : Partition C B) (T : Finset (Fin B)) {qD : ℝ}
    (hD : UncertifiedCap J (T.biUnion P.gates) (Fin.elim0 : Fin 0 → Finset (Fin C.N)) qD) :
    ∑ u ∈ T, ∑ v ∈ J.V ∩ P.gates u, J.ω v ≤ qD := by
  classical
  have hdisj : Set.PairwiseDisjoint (T : Set (Fin B)) fun u => J.V ∩ P.gates u := by
    intro u _ w _ huw
    refine Finset.disjoint_left.2 fun v hvu hvw => huw ?_
    rw [Finset.mem_inter, Partition.mem_gates] at hvu hvw
    exact Option.some_injective _ (hvu.2.symm.trans hvw.2)
  rw [← Finset.sum_biUnion hdisj]
  refine le_of_eq_of_le (Finset.sum_congr ?_ fun _ _ => rfl) hD
  ext v
  simp only [Finset.mem_biUnion, Finset.mem_inter, Finset.mem_filter, IsEmpty.forall_iff, and_true]
  exact ⟨fun ⟨a, ha, hv, hg⟩ => ⟨hv, a, ha, hg⟩, fun ⟨hv, a, ha, hg⟩ => ⟨a, ha, hv, hg⟩⟩

theorem ServingBlockBound : Guarantees.ServingBlocks.ServingBlockBound := by
  intro d J q c k B nf P S cS kS Pf r ωmax μ qU qD hP hqS hCK hq hB hμ hω0 hω hU hD
  exact Verity.SecurityProofs.ServingBlockSum d.D B P S J.V J.ω _ _ qD (Counted.V_subset J)
    (fun u hu => by
      rw [Counted.sum_inter J]
      exact assigned_le_serving d J q (cS u) (kS u) (P.gates u) ωmax μ (hqS u hu) (P.gates_subset_computed u)
        (hP u hu).1 (hP u hu).2 hB hμ hω0 hω)
    (fun u hu => (assigned_le_inference d J q c k P Pf r hCK hq u).trans (add_le_add le_rfl (hU u hu)))
    (dark_le J P _ hD)

theorem ServingBlockBoundRemote : Guarantees.ServingBlocks.ServingBlockBoundRemote := by
  intro d J q c k B nf P S Pf r qU qD hCK hq hU hD
  have hI : ∀ u ∈ S, ∑ v ∈ J.V ∩ P.gates u, J.ω v ≤
      ((univ.filter fun i => r.parent i = u).card : ℝ) * q c k + qU u := fun u hu =>
    (assigned_le_inference d J q c k P Pf r hCK hq u).trans (add_le_add le_rfl (hU u hu))
  simpa only [min_self, Counted.total] using
    Verity.SecurityProofs.ServingBlockSum d.D B P S J.V J.ω _ _ qD (Counted.V_subset J) hI hI (dark_le J P _ hD)

theorem UncertifiedElementary : Guarantees.ServingBlocks.UncertifiedElementary := by
  intro C J nf nc Pf Pc r s
  classical
  unfold UncertifiedCap Counted.outside Counted.split
  refine le_of_eq (Finset.sum_congr (Finset.filter_congr fun v hv => ?_) fun _ _ => rfl)
  refine ⟨fun ⟨hvs, hout⟩ => ⟨hvs, ?_, hout⟩, fun ⟨hvs, _, hout⟩ => ⟨hvs, hout⟩⟩
  obtain ⟨i, hi⟩ : ∃ i, Pf.unit v = some i :=
    Option.ne_none_iff_exists'.1 fun h => Bool.noConfusion (((Pf.input_iff v).1 h).symm.trans (J.V_computed v hv))
  have hsub := r.sub v
  rw [(Pc.mem_gates).1 hvs, hi, Option.map_some] at hsub
  exact ⟨⟨i, (Option.some_injective _ hsub).symm⟩, Pf.mem_gates.2 hi⟩

end Accounting.SecurityProofs.ServingBlocks
