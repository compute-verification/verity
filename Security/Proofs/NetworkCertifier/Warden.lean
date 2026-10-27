import Definitions.NetworkCertifier.Guarantees.Warden
import Proofs.NetworkCertifier.AdviceString
import Mathlib.Data.Finset.Option
import Mathlib.Data.Fintype.BigOperators

/-!
# The warden's channel bound

`Guarantees.Warden`. The warden's code on a lane emits, cell by cell, what the client view gives for the lane's first
late cell (`wardenLane_eq`): its data cells and padding before it, the fail cell, padding after. So the device's output
is the client view of the schedule, the data and the fail cells (`trusted-host`). A lane of `n` cells has its
fail cell among `none, 0, …, n − 1`, so the views are an image of the schedules, the fail choices and the data; with
the data from the advice strings within the caps and the content, their count is the advice string's count times `2^k`.
-/

namespace NetTiming

open Finset

/-- A cell before the fail cell is the stream's. -/
theorem cellView_of_lt {Cell : Type} (stream : List Cell) {o : Option ℕ} {j : ℕ} (h : ∀ f, o = some f → j < f) :
    cellView stream o j = streamCell stream j := by
  cases o with
  | none => rfl
  | some f => simp [cellView, h f rfl]

/-- **The warden's lane is the client view of its first late cell.** -/
theorem wardenLane_eq {Cell : Type} (a : Arrivals Cell) (t : ℕ → ℕ) (j k : ℕ) :
    wardenLane a t j k = (List.range' j k).map fun i =>
      (t i, cellView a.stream ((List.range' j k).find? fun i => decide (i ≤ a.stream.length ∧ t i < a.ready i)) i) := by
  induction k generalizing j with
  | zero => simp [wardenLane]
  | succ k ih =>
    have hafter : ∀ f, (List.range' (j + 1) k).find? (fun i => decide (i ≤ a.stream.length ∧ t i < a.ready i)) =
        some f → j < f := by
      intro f hf
      have := List.mem_of_find?_eq_some hf
      rw [List.mem_range'_1] at this
      omega
    rw [show List.range' j (k + 1) = j :: List.range' (j + 1) k from rfl, List.map_cons]
    by_cases hend : a.stream.length < j
    · have hp : ¬ (j ≤ a.stream.length ∧ t j < a.ready j) := fun h => by omega
      rw [List.find?_cons_of_neg (by simpa using hp)]
      simp only [wardenLane, hend, ↓reduceIte, ih]
      congr 1
      rw [cellView_of_lt _ hafter]
      simp [streamCell, show ¬ j < a.stream.length by omega]
    · by_cases hlate : t j < a.ready j
      · have hp : j ≤ a.stream.length ∧ t j < a.ready j := ⟨by omega, hlate⟩
        rw [List.find?_cons_of_pos (by simpa using hp)]
        simp only [wardenLane, hend, hlate, ↓reduceIte]
        refine List.cons_eq_cons.mpr ⟨by simp [cellView], ?_⟩
        apply List.map_congr_left
        intro i hi
        rw [List.mem_range'_1] at hi
        simp [cellView, show ¬ i < j by omega, show i ≠ j by omega]
      · have hp : ¬ (j ≤ a.stream.length ∧ t j < a.ready j) := fun h => hlate h.2
        rw [List.find?_cons_of_neg (by simpa using hp)]
        simp only [wardenLane, hend, hlate, ↓reduceIte, ih]
        rw [cellView_of_lt _ hafter]

theorem wardenCode_eq {κ Cell : Type} (s : WardenSchedule κ) (arr : κ → Arrivals Cell) :
    wardenCode s arr = clientView s (fun c => (arr c).stream) (failCell s arr) := by
  funext c
  simp only [wardenCode, clientView, failCell, wardenLane_eq, List.range_eq_range']

theorem wardenView_eq_clientView {κ Cell Srv Dev : Type} (arr : Srv → κ → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule κ × Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (s : WardenSchedule κ) (σ : Srv) :
    run w (s, σ) = clientView s (fun c => (arr σ c).stream) (failCell s (arr σ)) := by
  rw [htrust (s, σ)]
  exact wardenCode_eq s (arr σ)

/-- A lane of at most `cap` cells fails at none of them or at one of the first `cap`. -/
theorem failCell_mem {κ Cell : Type} (s : WardenSchedule κ) (arr : κ → Arrivals Cell) (c : κ) {cap : ℕ}
    (h : s.cells c ≤ cap) : failCell s arr c ∈ (range cap).insertNone := by
  rw [Finset.mem_insertNone]
  intro f hf
  have := List.mem_of_find?_eq_some (Option.mem_def.mp hf)
  rw [List.mem_range] at this
  exact Finset.mem_range.mpr (by omega)

theorem encard_wardenViews_le {κ Cell Srv Dev : Type} [Fintype κ] (arr : Srv → κ → Arrivals Cell) (w : Dev)
    (run : Dev → WardenSchedule κ × Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (cap : κ → ℕ) (Ss : Set (WardenSchedule κ)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (Srvs : Set Srv) (Ds : Set (κ → List Cell)) (hD : ∀ σ ∈ Srvs, (fun c => (arr σ c).stream) ∈ Ds) :
    {v | ∃ s ∈ Ss, ∃ σ ∈ Srvs, run w (s, σ) = v}.encard ≤
      Ss.encard * ((∏ c, (cap c + 1) : ℕ) : ℕ∞) * Ds.encard := by
  classical
  let F : Finset (κ → Option ℕ) := Fintype.piFinset fun c => (range (cap c)).insertNone
  let g : WardenSchedule κ × (κ → Option ℕ) × (κ → List Cell) → κ → List (ℕ × CellView Cell) :=
    fun p => clientView p.1 p.2.2 p.2.1
  have hsub : {v | ∃ s ∈ Ss, ∃ σ ∈ Srvs, run w (s, σ) = v} ⊆ g '' (Ss ×ˢ ((F : Set (κ → Option ℕ)) ×ˢ Ds)) := by
    rintro _ ⟨s, hs, σ, hσ, rfl⟩
    refine ⟨(s, failCell s (arr σ), fun c => (arr σ c).stream), ⟨hs, ?_, hD σ hσ⟩, ?_⟩
    · rw [Finset.mem_coe, Fintype.mem_piFinset]
      exact fun c => failCell_mem s (arr σ) c (hcap s hs c)
    · exact (wardenView_eq_clientView arr w run htrust s σ).symm
  calc _ ≤ (g '' (Ss ×ˢ ((F : Set (κ → Option ℕ)) ×ˢ Ds))).encard := Set.encard_le_encard hsub
    _ ≤ (Ss ×ˢ ((F : Set (κ → Option ℕ)) ×ˢ Ds)).encard := Set.encard_image_le _ _
    _ = Ss.encard * ((F.card : ℕ∞) * Ds.encard) := by
        rw [Set.encard_prod, Set.encard_prod, Set.encard_coe_eq_coe_finsetCard]
    _ = _ := by
        rw [← mul_assoc]
        congr 2
        simp [F, Fintype.card_piFinset, Finset.card_insertNone]

theorem encard_wardenDecodable_le {κ Cell Srv Dev Z Adv U : Type} {W : ℕ} [Fintype κ]
    (arr : Srv → κ → Arrivals Cell) (w : Dev) (run : Dev → WardenSchedule κ × Srv → κ → List (ℕ × CellView Cell))
    (htrust : Verity.Assumptions.TrustedHost w run fun p => wardenCode p.1 (arr p.2))
    (cap : κ → ℕ) (Ss : Set (WardenSchedule κ)) (hcap : ∀ s ∈ Ss, ∀ c, s.cells c ≤ cap c)
    (bits : Adv → Fin W → ℕ) (hbits : BitsCount bits) (b : Fin W → ℕ)
    (dataOf : Adv → U → κ → List Cell) (Us : Set U) (k : ℕ) (hUs : Us.encard ≤ ((2 ^ k : ℕ) : ℕ∞))
    (Zs : Set Z) (schedOf : Z → WardenSchedule κ) (srvOf : Z → Srv) (hs : ∀ z ∈ Zs, schedOf z ∈ Ss)
    (hdata : ∀ z ∈ Zs, ∃ a, WithinCaps bits b a ∧ ∃ u ∈ Us, (fun c => (arr (srvOf z) c).stream) = dataOf a u)
    (dec : (κ → List (ℕ × CellView Cell)) → Z) (hdec : ∀ z ∈ Zs, dec (run w (schedOf z, srvOf z)) = z) :
    Zs.encard ≤ Ss.encard * (((∏ c, (cap c + 1)) * 2 ^ (∑ i, b i) * 2 ^ k : ℕ) : ℕ∞) := by
  classical
  have hinj : Set.InjOn (fun z => run w (schedOf z, srvOf z)) Zs := by
    intro z hz z' hz' h
    rw [← hdec z hz, ← hdec z' hz']
    exact congrArg dec h
  let A : Finset Adv := (finite_charged bits hbits (∑ i, b i)).toFinset
  let Ds : Set (κ → List Cell) := (fun p : Adv × U => dataOf p.1 p.2) '' ((A : Set Adv) ×ˢ Us)
  have hDs : Ds.encard ≤ ((2 ^ (∑ i, b i) * 2 ^ k : ℕ) : ℕ∞) := by
    calc Ds.encard ≤ ((A : Set Adv) ×ˢ Us).encard := Set.encard_image_le _ _
      _ = (A.card : ℕ∞) * Us.encard := by rw [Set.encard_prod, Set.encard_coe_eq_coe_finsetCard]
      _ ≤ ((2 ^ (∑ i, b i) : ℕ) : ℕ∞) * ((2 ^ k : ℕ) : ℕ∞) :=
          mul_le_mul' (by exact_mod_cast card_charged_le bits hbits _) hUs
      _ = _ := by push_cast; rfl
  have hD : ∀ σ ∈ srvOf '' Zs, (fun c => (arr σ c).stream) ∈ Ds := by
    rintro _ ⟨z, hz, rfl⟩
    obtain ⟨a, ha, u, hu, he⟩ := hdata z hz
    refine ⟨(a, u), ⟨?_, hu⟩, he.symm⟩
    simp only [A, Finset.mem_coe, Set.Finite.mem_toFinset, Set.mem_ofPred_eq]
    exact sum_le_of_withinCaps ha
  calc Zs.encard = ((fun z => run w (schedOf z, srvOf z)) '' Zs).encard := hinj.encard_image.symm
    _ ≤ {v | ∃ s ∈ Ss, ∃ σ ∈ srvOf '' Zs, run w (s, σ) = v}.encard := by
        apply Set.encard_le_encard
        rintro _ ⟨z, hz, rfl⟩
        exact ⟨schedOf z, hs z hz, srvOf z, ⟨z, hz, rfl⟩, rfl⟩
    _ ≤ Ss.encard * ((∏ c, (cap c + 1) : ℕ) : ℕ∞) * Ds.encard :=
        encard_wardenViews_le arr w run htrust cap Ss hcap (srvOf '' Zs) Ds hD
    _ ≤ Ss.encard * ((∏ c, (cap c + 1) : ℕ) : ℕ∞) * ((2 ^ (∑ i, b i) * 2 ^ k : ℕ) : ℕ∞) :=
        mul_le_mul_right hDs _
    _ = _ := by
        rw [mul_assoc]
        push_cast
        simp only [mul_assoc]

end NetTiming
