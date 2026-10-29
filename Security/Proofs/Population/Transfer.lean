import Proofs.Population.Oblivious
import Proofs.Population.Leakage

/-!
# The population theorems for a program, with its padding's public cones

`Oblivious` pads a time-ordered program over a set `C` of assignments (`padOver C`) to a checked population with the
program's faulty proof blocks on `C`. This file carries every population theorem across, so that each holds for the
program with the cones and upstreams of the padding, which are public when `C` is (the assignments holding the public
values, `{σ | ∀ t ∈ Pub, σ t = a t}`; or every assignment, the static superset). For an oblivious program those cones
are its own dataflow's (`population_of_addressedBy`).

* `egress_eq_pad`: on `C`, the program and its padding show the same egress.
* `eq_honest_off_padDown` (T1), `card_egress_le_pad` (T2), `card_leak_le_coneEgress_pad` and
  `card_leak_le_of_not_up_pad` (T3), `skipped_work_le_wt_pad` (T4).
* `downAt_sub_padDown`: the padding's cone holds the cone of every honest run in `C`. It can hold more than their
  union, a path joining reads of different runs; `HonestCone` states the damage theorems at the union.
-/

namespace Population.DynPopulation

open Finset

variable {U S V : Type} (D : DynPopulation U S V)

theorem wrong_pad_of_mem [Fintype U] [Fintype S] [Inhabited V] {C : Set (S → V)} {σ : S → V} (hσ : σ ∈ C) :
    (D.padOver C).wrong (D.padFns C) σ = D.wrong σ :=
  D.wrong_padOver fun u => D.reads_subset_padOver hσ u

theorem holds_mem_pub {Pub X : Finset S} (hPub : Pub ⊆ D.input) (hPX : ∀ t ∈ Pub, t ∉ X) {a σ : S → V}
    (h : D.Holds a X σ) : σ ∈ {σ : S → V | ∀ t ∈ Pub, σ t = a t} :=
  show ∀ t ∈ Pub, σ t = a t from fun t ht => h t (hPub ht) (hPX t ht)

/-- **The program and its padding show the same egress** when `C` holds every assignment the egress ranges over. -/
theorem egress_eq_pad [Fintype U] [Fintype S] [Fintype V] [Inhabited V] {C : Set (S → V)} {a : S → V}
    {X : Finset S} (hC : ∀ σ, D.Holds a X σ → σ ∈ C) (O : Finset S) (B : Finset U) :
    D.egress a X O B = (D.padOver C).egress (D.padFns C) a X O B := by
  classical
  ext e
  simp only [DynPopulation.egress, Dataflow.egress, mem_image, mem_filter, mem_univ, true_and]
  constructor
  · rintro ⟨σ, ⟨h1, h2⟩, rfl⟩
    refine ⟨σ, ⟨h1, ?_⟩, rfl⟩
    rw [D.wrong_pad_of_mem (hC σ h1)]
    exact h2
  · rintro ⟨σ, ⟨h1, h2⟩, rfl⟩
    refine ⟨σ, ⟨h1, ?_⟩, rfl⟩
    rw [← D.wrong_pad_of_mem (hC σ h1)]
    exact h2

/-- **T1 for a program**: a run in `C` equals the honest run (in `C`) on every unit outside the padding's cone of its
faulty proof blocks. -/
theorem eq_honest_off_padDown [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment) {step : U → ℕ}
    (ht : D.TimeOrdered step) {C : Set (S → V)} (a : S → V) {honest σ : S → V} (hhC : honest ∈ C) (hσC : σ ∈ C)
    (hh : ∀ s ∈ D.input, honest s = a s) (hc : ∀ u, D.Correct honest u) (hσ : ∀ s ∈ D.input, σ s = a s) :
    ∀ v, v ∉ (D.padOver C).down (D.wrong σ) → ∀ s ∈ D.writes v, σ s = honest s := by
  rw [← D.wrong_pad_of_mem hσC]
  exact (D.padOver C).eq_honest_off_down (D.padOver_checked C hsa ht) (D.padFns C) a hh
    (fun u => (D.padOver_correct_iff (D.reads_subset_padOver hhC u)).2 (hc u)) hσ

/-- **T2 for a program**: with the secrets outside the padding's upstream of the egress, at most `|V| ^ |writes B|`
egress values at the wrong set `B`. -/
theorem card_egress_le_pad [Fintype U] [Fintype S] [Fintype V] [Inhabited V] (hsa : D.SingleAssignment)
    {step : U → ℕ} (ht : D.TimeOrdered step) {C : Set (S → V)} {a : S → V} {X O : Finset S} {B : Finset U}
    (hC : ∀ σ, D.Holds a X σ → σ ∈ C) (hX : ∀ s ∈ X, s ∉ (D.padOver C).up O) :
    (D.egress a X O B).card ≤ Fintype.card V ^ ((D.padOver C).writesOf B).card := by
  rw [D.egress_eq_pad hC O B]
  have hch := D.padOver_checked C hsa ht
  exact (D.padOver C).card_egress_le ((D.padOver C).covered_of_singleAssignment hch.1) hch.2 _ a hX

/-- **T3 for a program, the uniform form**: at most `m` faulty proof blocks and every unit's padded cone egress at most
`E` give `#honest · C(|U|, m) · |V| ^ (m E)` egress values. -/
theorem card_leak_le_coneEgress_pad [Fintype U] [Fintype S] [Fintype V] [DecidableEq V] [Inhabited V]
    (hsa : D.SingleAssignment) {step : U → ℕ} (ht : D.TimeOrdered step) {C : Set (S → V)} (O : Finset S)
    {m E : ℕ} (hmU : m ≤ Fintype.card U) (hE : ∀ u, ((D.padOver C).coneEgress O {u}).card ≤ E) {ι : Type}
    (Cl : Finset ι) (σ honest : ι → S → V) (hσC : ∀ i ∈ Cl, σ i ∈ C) (hhC : ∀ i ∈ Cl, honest i ∈ C)
    (hin : ∀ i ∈ Cl, ∀ t ∈ D.input, σ i t = honest i t) (hh : ∀ i ∈ Cl, ∀ u, D.Correct (honest i) u)
    (hm : ∀ i ∈ Cl, (D.wrong (σ i)).card ≤ m) :
    (Cl.image fun i (o : O) => σ i o).card ≤
      (Cl.image fun i (o : O) => honest i o).card * (Fintype.card U).choose m * Fintype.card V ^ (m * E) := by
  have hch := D.padOver_checked C hsa ht
  exact (D.padOver C).card_leak_le_coneEgress ((D.padOver C).covered_of_singleAssignment hch.1) hch.2
    (D.padFns C) O hmU hE Cl σ honest hin
    (fun i hi u => (D.padOver_correct_iff (D.reads_subset_padOver (hhC i hi) u)).2 (hh i hi u))
    (fun i hi => by rw [D.wrong_pad_of_mem (hσC i hi)]; exact hm i hi)

/-- **T3 for a program, `α = 1`**: when no secret is upstream of the egress in the padding, the leakage is at most the
control. -/
theorem card_leak_le_of_not_up_pad [Fintype U] [Fintype S] [DecidableEq S] [Fintype V] [DecidableEq V]
    [Inhabited V] (hsa : D.SingleAssignment) {step : U → ℕ} (ht : D.TimeOrdered step) {C : Set (S → V)}
    {X O : Finset S} (a : S → V) (hX : ∀ s ∈ X, s ∉ (D.padOver C).up O) (Cl : Finset (S → V)) (m w : ℕ)
    (hw : ∀ u, (D.writes u).card ≤ w) (σ : (S → V) → S → V) (hσC : ∀ s, σ s ∈ C)
    (hσ : ∀ s, ∀ t ∈ D.input, σ s t = if t ∈ X then s t else a t) (hm : ∀ s, (D.wrong (σ s)).card ≤ m) :
    (Cl.image fun s (o : O) => σ s o).card ≤
      (∑ j ∈ range (m + 1), (Fintype.card U).choose j) * Fintype.card V ^ (m * w) := by
  have hch := D.padOver_checked C hsa ht
  exact (D.padOver C).card_leak_le_of_not_up ((D.padOver C).covered_of_singleAssignment hch.1) hch.2
    (D.padFns C) a hX Cl m w hw σ hσ (fun s => by rw [D.wrong_pad_of_mem (hσC s)]; exact hm s)

/-- **T4 for a program**: every padded cone at most `F` times its root's weight gives skipped work at most
`F · wt(wrong)` plus the lucky work. -/
theorem skipped_work_le_wt_pad [Fintype U] [DecidableEq U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment)
    {step : U → ℕ} (ht : D.TimeOrdered step) {C : Set (S → V)} (a : S → V) {honest σ : S → V} (hhC : honest ∈ C)
    (hσC : σ ∈ C) (hh : ∀ s ∈ D.input, honest s = a s) (hc : ∀ u, D.Correct honest u)
    (hσ : ∀ s ∈ D.input, σ s = a s) {K L : Finset U}
    (hl : (D.padOver C).SkipsLucky (D.padFns C) honest σ K L) (wt : U → ℕ) (F : ℕ)
    (hF : ∀ u, ∑ v ∈ (D.padOver C).down {u}, wt v ≤ F * wt u) :
    ∑ u ∈ univ \ K, wt u ≤ F * ∑ u ∈ D.wrong σ, wt u + ∑ u ∈ L, wt u := by
  rw [← D.wrong_pad_of_mem hσC]
  exact (D.padOver C).skipped_work_le_wt (D.padOver_checked C hsa ht) (D.padFns C) a hh
    (fun u => (D.padOver_correct_iff (D.reads_subset_padOver hhC u)).2 (hc u)) hσ hl wt F hF

/-- **The padding's cone holds every honest run's**: for an honest run in `C`, the cone of `B` in its own dataflow lies
inside the padding's. -/
theorem downAt_sub_padDown [Fintype U] [Fintype S] {C : Set (S → V)} {h : S → V} (hhC : h ∈ C) (B : Finset U)
    {v : U} (hv : v ∈ D.downAt h B) : v ∈ (D.padOver C).down B := by
  obtain ⟨u, hu, huv⟩ := hv
  refine (D.padOver C).mem_down.2 ⟨u, hu, ?_⟩
  induction huv with
  | refl => exact .refl
  | tail _ hbc ih =>
    obtain ⟨s, hs, hr⟩ := hbc
    exact ih.tail ⟨s, hs, D.reads_subset_padOver hhC _ hr⟩

end Population.DynPopulation
