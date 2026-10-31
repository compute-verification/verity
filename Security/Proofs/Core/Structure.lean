import Definitions.Core.Structure

/-!
# Proofs about the structural vocabulary

Short facts about `Verity.Structure`'s definitions: a coarser equivalence realizes and attains more
(`Realizes.mono`, `profile_mono`); fresh input splits into external and produced reads
(`fresh_eq_external_add_produced`); a partition's blocks make up the computed gates (`computed_eq_biUnion`), so exact
blocks give `|C| = c · n` (`size_computed_of_exact`); the blocks before a period form a prefix
(`TimeOrdered.upto_isPrefix`), and distinct periods make a time-ordered partition ordered
(`TimeOrdered.ordered_of_injective`); an attained vector bounds `K*` and `n*` (`leastFresh_le`, `fewestBlocks_le`), and
each evaluation its method's worst case (`le_worst`).
-/

namespace Verity.Structure

open Finset

/-- A coarser equivalence realizes more. -/
theorem Realizes.mono {E E' : Setoid Evaluation} (h : ∀ e e', E.r e e' → E'.r e e') {e : Evaluation}
    {X : Execution} (hr : Realizes E e X) : Realizes E' e X := by
  obtain ⟨e', he, rest⟩ := hr
  exact ⟨e', h e e' he, rest⟩

/-- All the computed gates form a prefix. -/
theorem isPrefix_computed (C : Circuit) : IsPrefix C C.computed := by
  refine ⟨Finset.Subset.refl _, fun a ha => ?_⟩
  have h := (Finset.mem_filter.1 ha).2.1
  cases hi : C.isInput a with
  | true => rfl
  | false => exact absurd (show a ∈ C.computed from Finset.mem_filter.2 ⟨Finset.mem_univ _, hi⟩) h

/-- Fresh input is external plus produced reads, in wire bits. -/
theorem fresh_eq_external_add_produced (d : RegisteredCircuit) (B : Finset (Fin d.D.N)) :
    d.D.width (d.unregistered B) = d.D.width (externalIn d B) + d.D.width (producedIn d B) :=
  (Finset.sum_filter_add_sum_filter_not (d.unregistered B) (fun a => d.D.isInput a = true) d.D.w).symm

/-- A partition's blocks make up the computed gates. -/
theorem computed_eq_biUnion {C : Circuit} {n : ℕ} (P : Partition C n) : C.computed = univ.biUnion P.gates := by
  ext g
  rw [Finset.mem_biUnion]
  constructor
  · intro hg
    cases hu : P.unit g with
    | none => simp [(P.input_iff g).1 hu, Circuit.computed] at hg
    | some u => exact ⟨u, Finset.mem_univ _, Finset.mem_filter.2 ⟨Finset.mem_univ _, hu⟩⟩
  · rintro ⟨u, -, hg⟩
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩
    cases hi : C.isInput g with
    | false => rfl
    | true =>
      have hu : P.unit g = some u := (Finset.mem_filter.1 hg).2
      simp [(P.input_iff g).2 hi] at hu

/-- **`|C| = c · n`** when every block has exactly `c` gates. -/
theorem size_computed_of_exact {C : Circuit} {n c : ℕ} (P : Partition C n) (h : ExactScale P c) :
    C.size C.computed = n * c := by
  have hdisj : ((univ : Finset (Fin n)) : Set (Fin n)).PairwiseDisjoint P.gates := by
    intro u _ v _ huv
    show Disjoint (P.gates u) (P.gates v)
    rw [Finset.disjoint_left]
    intro g hu hv
    exact huv (Option.some_injective _ ((Finset.mem_filter.1 hu).2.symm.trans (Finset.mem_filter.1 hv).2))
  rw [computed_eq_biUnion P, Circuit.size, Finset.card_biUnion hdisj]
  calc ∑ u ∈ univ, (P.gates u).card = ∑ _u ∈ (univ : Finset (Fin n)), c := Finset.sum_congr rfl fun u _ => h u
    _ = n * c := by simp

/-- The blocks before each period form a prefix. -/
theorem TimeOrdered.upto_isPrefix {C : Circuit} {n m : ℕ} {Loc : Type} (F : TimeOrdered C n Loc m) (j : ℕ) :
    IsPrefix C (F.upto j) := by
  refine ⟨fun g hg => ?_, fun a ha => ?_⟩
  · obtain ⟨u, hu, -⟩ := (Finset.mem_filter.1 hg).2
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩
    cases hi : C.isInput g with
    | false => rfl
    | true => simp [(F.input_iff g).2 hi] at hu
  · obtain ⟨-, hnot, g, hg, hag⟩ := Finset.mem_filter.1 ha
    obtain ⟨u, hu, hj⟩ := (Finset.mem_filter.1 hg).2
    by_contra hin
    cases hv : F.unit a with
    | none => exact hin ((F.input_iff a).1 hv)
    | some v =>
      exact hnot (Finset.mem_filter.2 ⟨Finset.mem_univ _, v, hv,
        lt_of_le_of_lt (Fin.le_def.1 (F.ordered g a hag u v hu hv)) hj⟩)

/-- Distinct periods make a time-ordered partition ordered. -/
theorem TimeOrdered.ordered_of_injective {C : Circuit} {n m : ℕ} {Loc : Type} (F : TimeOrdered C n Loc m)
    (hinj : Function.Injective F.period) : Ordered F.toPartition := by
  have step : ∀ v u, Dep F.toPartition v u → F.period v < F.period u := by
    rintro v u ⟨hne, g, a, ha, hu, hv⟩
    exact lt_of_le_of_ne (F.ordered g a ha u v hu hv) fun h => hne (hinj h)
  have chain : ∀ v w, Relation.TransGen (Dep F.toPartition) v w → F.period v < F.period w := by
    intro v w hvw
    induction hvw with
    | single h => exact step _ _ h
    | tail _ h ih => exact ih.trans (step _ _ h)
  intro u hu
  exact lt_irrefl _ (chain u u hu)

/-- A coarser equivalence attains more. -/
theorem profile_mono {Loc : Type} {E E' : Setoid Evaluation} (h : ∀ e e', E.r e e' → E'.r e e')
    (𝔉 : Family Loc) (e : Evaluation) : profile E 𝔉 e ⊆ profile E' 𝔉 e := by
  rintro r ⟨X, hX, rest⟩
  exact ⟨X, hX.mono h, rest⟩

/-- An attained vector bounds `K*`. -/
theorem leastFresh_le {𝓡 : Set Resources} {r : Resources} (h : r ∈ 𝓡) : leastFresh 𝓡 ≤ r.fresh :=
  iInf₂_le_of_le r h le_rfl

/-- An attained vector within `k` bounds `n*`. -/
theorem fewestBlocks_le {𝓡 : Set Resources} {r : Resources} {k : ℕ} (h : r ∈ 𝓡) (hk : r.peakFresh ≤ k) :
    fewestBlocks 𝓡 k ≤ r.blocks :=
  iInf₂_le_of_le r h (iInf_le _ hk)

/-- Each evaluation is at most its method's worst case. -/
theorem le_worst (q : Evaluation → ℕ∞) (M : Method) (x ω : M.d.D.Transcript) : q ⟨M, x, ω⟩ ≤ worst q M :=
  le_iSup₂ (f := fun (x ω : M.d.D.Transcript) => q ⟨M, x, ω⟩) x ω

end Verity.Structure
