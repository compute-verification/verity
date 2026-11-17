module

public import Definitions.Spot.Oblivious
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Proofs.Spot.Pointer
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Proofs.Spot.Oblivious
import Proofs.Spot.Work

/-! The damage theorems need no obliviousness: locality and leakage over the honest runs' cones. -/

@[expose] public section

namespace Population.DynPopulation

open Finset

variable {U S V : Type} (D : DynPopulation U S V)

theorem mem_downOver [Fintype U] {C : Set (S → V)} {B : Finset U} {v : U} :
    v ∈ D.downOver C B ↔ ∃ h ∈ C, v ∈ D.downAt h B := by
  classical
  simp [downOver]

theorem downAt_mono {σ : S → V} {B B' : Finset U} (hB : B ⊆ B') {v : U} (hv : v ∈ D.downAt σ B) :
    v ∈ D.downAt σ B' := by
  obtain ⟨u, hu, huv⟩ := hv
  exact ⟨u, hB hu, huv⟩

theorem eq_honest_off_downOver [Fintype U] (hsa : D.SingleAssignment) (hac : D.AddressClosed) {step : U → ℕ}
    (ht : D.TimeOrdered step) {C : Set (S → V)} {h σ : S → V} (hhC : h ∈ C) (hh : ∀ u, D.Correct h u)
    (hin : ∀ s ∈ D.input, σ s = h s) {B : Finset U} (hB : D.incorrect σ ⊆ B) {s : S}
    (hs : s ∉ D.writesOf (D.downOver C B)) : σ s = h s := by
  classical
  by_cases hsi : s ∈ D.input
  · exact hin s hsi
  · obtain ⟨v, hv⟩ := ((hsa s).2 hsi).exists
    refine D.eq_honest_off_downAt hac ht hh hin v (fun hvd => hs ?_) s hv
    simp only [writesOf, mem_biUnion]
    exact ⟨v, D.mem_downOver.2 ⟨h, hhC, D.downAt_mono hB hvd⟩, hv⟩

theorem mem_coneEgressOver [Fintype U] {C : Set (S → V)} {O : Finset S} {B : Finset U} {s : S} :
    s ∈ D.coneEgressOver C O B ↔ s ∈ O ∧ s ∈ D.writesOf (D.downOver C B) := by
  classical
  simp [coneEgressOver]

theorem card_coneEgressOver_le [Fintype U] (C : Set (S → V)) (O : Finset S) (B : Finset U) :
    (D.coneEgressOver C O B).card ≤ ∑ u ∈ B, (D.coneEgressOver C O {u}).card := by
  classical
  refine (card_le_card fun s hs => ?_).trans card_biUnion_le
  obtain ⟨hsO, hsw⟩ := D.mem_coneEgressOver.1 hs
  simp only [writesOf, mem_biUnion] at hsw
  obtain ⟨v, hv, hsv⟩ := hsw
  obtain ⟨h, hhC, u, hu, huv⟩ := D.mem_downOver.1 hv
  refine mem_biUnion.2 ⟨u, hu, D.mem_coneEgressOver.2 ⟨hsO, ?_⟩⟩
  simp only [writesOf, mem_biUnion]
  exact ⟨v, D.mem_downOver.2 ⟨h, hhC, u, mem_singleton_self u, huv⟩, hsv⟩

theorem card_leak_le_coneOver [Fintype U] [Fintype V] [DecidableEq V] (hsa : D.SingleAssignment)
    (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step) (C : Set (S → V)) (O : Finset S)
    (𝓑 : Finset (Finset U)) {ι : Type} (Cl : Finset ι) (σ honest : ι → S → V) (hhC : ∀ i ∈ Cl, honest i ∈ C)
    (hin : ∀ i ∈ Cl, ∀ t ∈ D.input, σ i t = honest i t) (hh : ∀ i ∈ Cl, ∀ u, D.Correct (honest i) u)
    (hB : ∀ i ∈ Cl, ∃ B ∈ 𝓑, D.incorrect (σ i) ⊆ B) :
    (Cl.image fun i (o : O) => σ i o).card ≤
      (Cl.image fun i (o : O) => honest i o).card *
        ∑ B ∈ 𝓑, Fintype.card V ^ (D.coneEgressOver C O B).card := by
  classical
  let H : Finset (O → V) := Cl.image fun i (o : O) => honest i o
  let φ : (B : Finset U) → (O → V) × (D.coneEgressOver C O B → V) → O → V := fun B p o =>
    if ho : o.1 ∈ D.coneEgressOver C O B then p.2 ⟨o.1, ho⟩ else p.1 o
  have hsub : Cl.image (fun i (o : O) => σ i o) ⊆ 𝓑.biUnion fun B => (H ×ˢ univ).image (φ B) := by
    intro e he
    obtain ⟨i, hi, rfl⟩ := mem_image.1 he
    obtain ⟨B, hB𝓑, hwB⟩ := hB i hi
    refine mem_biUnion.2 ⟨B, hB𝓑, mem_image.2 ⟨((fun o => honest i o), fun t => σ i t),
      mem_product.2 ⟨mem_image_of_mem _ hi, mem_univ _⟩, funext fun o => ?_⟩⟩
    by_cases ho : o.1 ∈ D.coneEgressOver C O B
    · simp only [φ, ho, ↓reduceDIte]
    · simp only [φ, ho, ↓reduceDIte]
      have hoc : o.1 ∉ D.writesOf (D.downOver C B) := fun h => ho (D.mem_coneEgressOver.2 ⟨o.2, h⟩)
      exact (D.eq_honest_off_downOver hsa hac ht (hhC i hi) (hh i hi) (hin i hi) hwB hoc).symm
  calc _ ≤ _ := card_le_card hsub
    _ ≤ ∑ B ∈ 𝓑, ((H ×ˢ univ).image (φ B)).card := card_biUnion_le
    _ ≤ ∑ B ∈ 𝓑, H.card * Fintype.card V ^ (D.coneEgressOver C O B).card := by
        refine sum_le_sum fun B _ => card_image_le.trans (le_of_eq ?_)
        rw [card_product, card_univ, Fintype.card_fun, Fintype.card_coe]
    _ = _ := by rw [mul_sum]

theorem card_leak_le_coneEgressOver [Fintype U] [Fintype V] [DecidableEq V] [Nonempty V]
    (hsa : D.SingleAssignment) (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step)
    (C : Set (S → V)) (O : Finset S) {m E : ℕ} (hmU : m ≤ Fintype.card U)
    (hE : ∀ u, (D.coneEgressOver C O {u}).card ≤ E) {ι : Type} (Cl : Finset ι) (σ honest : ι → S → V)
    (hhC : ∀ i ∈ Cl, honest i ∈ C) (hin : ∀ i ∈ Cl, ∀ t ∈ D.input, σ i t = honest i t)
    (hh : ∀ i ∈ Cl, ∀ u, D.Correct (honest i) u) (hm : ∀ i ∈ Cl, (D.incorrect (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => σ i o).card ≤
      (Cl.image fun i (o : O) => honest i o).card * (Fintype.card U).choose m * Fintype.card V ^ (m * E) := by
  classical
  have hB : ∀ i ∈ Cl, ∃ B ∈ powersetCard m (univ : Finset U), D.incorrect (σ i) ⊆ B := by
    intro i hi
    obtain ⟨B, hsub, hcard⟩ := Finset.exists_superset_card_eq (hm i hi) hmU
    exact ⟨B, mem_powersetCard.2 ⟨subset_univ _, hcard⟩, hsub⟩
  refine (D.card_leak_le_coneOver hsa hac ht C O _ Cl σ honest hhC hin hh hB).trans ?_
  rw [Nat.mul_assoc]
  refine Nat.mul_le_mul_left _ ?_
  calc ∑ B ∈ powersetCard m (univ : Finset U), Fintype.card V ^ (D.coneEgressOver C O B).card
      ≤ ∑ _B ∈ powersetCard m (univ : Finset U), Fintype.card V ^ (m * E) := by
        refine sum_le_sum fun B hB => Nat.pow_le_pow_right Fintype.card_pos ?_
        obtain ⟨-, hBm⟩ := mem_powersetCard.1 hB
        calc (D.coneEgressOver C O B).card ≤ ∑ u ∈ B, (D.coneEgressOver C O {u}).card :=
              D.card_coneEgressOver_le C O B
          _ ≤ ∑ _u ∈ B, E := sum_le_sum fun u _ => hE u
          _ = m * E := by simp [hBm]
    _ = (Fintype.card U).choose m * Fintype.card V ^ (m * E) := by
        rw [sum_const, card_powersetCard, card_univ]
        simp

theorem skipped_work_le_wt_over [Fintype U] [DecidableEq U] (hac : D.AddressClosed) {step : U → ℕ}
    (ht : D.TimeOrdered step) {C : Set (S → V)} {h σ : S → V} (hhC : h ∈ C) (hh : ∀ u, D.Correct h u)
    (hin : ∀ s ∈ D.input, σ s = h s) {K L : Finset U} (hl : D.SkipsLucky h σ K L) (wt : U → ℕ) (F : ℕ)
    (hF : ∀ u, ∑ v ∈ D.downOver C {u}, wt v ≤ F * wt u) :
    ∑ u ∈ univ \ K, wt u ≤ F * ∑ u ∈ D.incorrect σ, wt u + ∑ u ∈ L, wt u := by
  have hsub : univ \ K ⊆ ((D.incorrect σ).biUnion fun u => D.downOver C {u}) ∪ L := by
    intro v hv
    rcases D.skipped_mem_downAt hac ht hh hin hl v (mem_sdiff.1 hv).2 with ⟨u, hu, huv⟩ | hvL
    · exact mem_union_left _ (mem_biUnion.2 ⟨u, hu, D.mem_downOver.2 ⟨h, hhC, u, mem_singleton_self u, huv⟩⟩)
    · exact mem_union_right _ hvL
  calc ∑ u ∈ univ \ K, wt u ≤ ∑ u ∈ ((D.incorrect σ).biUnion fun u => D.downOver C {u}) ∪ L, wt u :=
        sum_le_sum_of_subset hsub
    _ ≤ ∑ u ∈ (D.incorrect σ).biUnion (fun u => D.downOver C {u}), wt u + ∑ u ∈ L, wt u := Dataflow.sum_union_le _ _ _
    _ ≤ F * ∑ u ∈ D.incorrect σ, wt u + ∑ u ∈ L, wt u := by
        refine Nat.add_le_add_right ?_ _
        calc ∑ u ∈ (D.incorrect σ).biUnion (fun u => D.downOver C {u}), wt u
            ≤ ∑ u ∈ D.incorrect σ, ∑ v ∈ D.downOver C {u}, wt v := Dataflow.sum_biUnion_le _ _ _
          _ ≤ ∑ u ∈ D.incorrect σ, F * wt u := sum_le_sum fun u _ => hF u
          _ = F * ∑ u ∈ D.incorrect σ, wt u := (mul_sum _ _ _).symm

end Population.DynPopulation

namespace Population

open Finset

theorem vm_coneEgressOver_ptr {N : ℕ} {C : Set (Slot N → Bool)} {h : Slot N → Bool} (hhC : h ∈ C) :
    (vm N).coneEgressOver C (univ.filter fun s => s.isOut) {none} = univ.filter fun s => s.isOut := by
  ext s
  rw [DynPopulation.mem_coneEgressOver]
  refine ⟨fun hs => hs.1, fun hs => ⟨hs, ?_⟩⟩
  cases s with
  | ptr => simp at hs
  | zero i => simp at hs
  | secret i => simp at hs
  | out i =>
    simp only [DynPopulation.writesOf, mem_biUnion]
    exact ⟨some i, (vm N).mem_downOver.2 ⟨h, hhC, none, mem_singleton_self _,
      .single ⟨.ptr, by simp [vm], by simp [vm]⟩⟩, by simp [vm]⟩

end Population
