import Definitions.Population.Oblivious
import Proofs.Population.Determinacy
import Proofs.Population.Pointer

/-!
# Which programs are as good as circuits

A program's trace is a `DynPopulation`. Three facts decide what the population theorems give it.

* `padOver_checked`, `wrong_padOver`, **every program pads to a population**: for a time-ordered, single-assignment
  trace and any set `C` of assignments, the padding over `C` is a checked population, and in every assignment whose
  read sets it covers, its wrong units are the program's. With `C = Set.univ` it covers every assignment
  (`wrong_pad`): the padded static circuit, with T1–T4 at its public cones.
* `population_of_addressedBy`, **an oblivious program is that population with nothing added** (Theorem O1): if the read
  sets depend only on the values of the slots `A`, then at each value `a` on `A`, the padding over the assignments
  holding `a` is checked, has exactly the program's read sets in each of them, and the same wrong units. With `A` the
  public inputs it is a circuit indexed by them; with `A = ∅` (`Oblivious`), one circuit.
* `eq_honest_off_downAt`, **locality needs no obliviousness**: in a time-ordered, address-closed trace, every run
  agrees with the honest one off the wrong units' cone in the honest run's own dataflow, and `skipped_mem_downAt`
  puts every skipped unit in that cone or among the lucky. What a program lacks is that this cone be public, and
  isolation: `vm` is address-closed and honestly oblivious (`vm_honestlyOblivious`), so the
  classical notion holds, and still one wrong one-bit unit copies `N` secret words that no honest run reads
  (`vm_leak`, `vm_honest_up`). Its padding is `vmStatic` (`vm_pad`), whose wiring shows the secrets upstream.
* `wrong_freeze`, `wrong_freeze_of_off_downAt`, **addressing proved, data sampled** (the hybrid): when the slots the
  read sets depend on hold their honest values, which holds when none is written inside the wrong units' cone, the run
  is an assignment of the honest run's own frozen dataflow, with the same wrong units.
-/

namespace Population

open Finset

namespace DynPopulation

variable {U S V : Type} (D : DynPopulation U S V)

theorem mem_padOver_reads [Fintype S] {C : Set (S → V)} {u : U} {s : S} :
    s ∈ (D.padOver C).reads u ↔ ∃ σ ∈ C, s ∈ D.reads u σ := by
  classical
  simp [padOver]

theorem reads_subset_padOver [Fintype S] {C : Set (S → V)} {σ : S → V} (hσ : σ ∈ C) (u : U) :
    D.reads u σ ⊆ (D.padOver C).reads u :=
  fun _ hs => (D.mem_padOver_reads).2 ⟨σ, hσ, hs⟩

/-- **The padding computes the program's step**: in an assignment whose read set for `u` the padding covers, `u` is
correct in the padding exactly when it is correct in the program. -/
theorem padOver_correct_iff [Fintype S] [Inhabited V] {C : Set (S → V)} {σ : S → V} {u : U}
    (hσ : D.reads u σ ⊆ (D.padOver C).reads u) :
    (D.padOver C).Correct (D.padFns C) σ u ↔ D.Correct σ u := by
  classical
  have key : ∀ s ∈ D.writes u,
      D.f u (fun t => if h : t ∈ (D.padOver C).reads u then σ t else default) s = D.f u σ s := by
    intro s hs
    refine (D.reads_only u σ _ (fun t ht => ?_) s hs).symm
    simp [hσ ht]
  constructor
  · intro h s hs
    rw [h ⟨s, hs⟩]
    exact key s hs
  · intro h s
    rw [h s.1 s.2]
    exact (key s.1 s.2).symm

/-- **The padding's wrong units are the program's**, in every assignment whose read sets it covers. -/
theorem wrong_padOver [Fintype U] [Fintype S] [Inhabited V] {C : Set (S → V)} {σ : S → V}
    (hσ : ∀ u, D.reads u σ ⊆ (D.padOver C).reads u) :
    (D.padOver C).wrong (D.padFns C) σ = D.wrong σ := by
  classical
  ext u
  simp only [Dataflow.wrong, DynPopulation.wrong, mem_filter, mem_univ, true_and]
  exact not_congr (D.padOver_correct_iff (hσ u))

/-- **The static superset covers every assignment**: the padded static circuit's wrong units are the program's. -/
theorem wrong_pad [Fintype U] [Fintype S] [Inhabited V] (σ : S → V) :
    (D.padOver Set.univ).wrong (D.padFns Set.univ) σ = D.wrong σ :=
  D.wrong_padOver fun u => D.reads_subset_padOver (Set.mem_univ σ) u

/-- **Every padding of a time-ordered, single-assignment trace is a checked population**: a padded read still comes
from an earlier step. -/
theorem padOver_checked [Fintype S] (C : Set (S → V)) (hsa : D.SingleAssignment) {step : U → ℕ}
    (ht : D.TimeOrdered step) : (D.padOver C).Checked := by
  have hlt : ∀ u v, (D.padOver C).Dep u v → step u < step v := by
    rintro u v ⟨s, hs, hsr⟩
    obtain ⟨σ, -, hσ⟩ := (D.mem_padOver_reads).1 hsr
    have hsi : s ∉ D.input := fun hin => (hsa s).1 hin u hs
    rcases ht v σ s hσ with hin | ⟨w, hw, hwv⟩
    · exact absurd hin hsi
    · have : w = u := ((hsa s).2 hsi).unique hw hs
      exact this ▸ hwv
  refine ⟨hsa, fun u h => ?_⟩
  have : ∀ a b, Relation.TransGen (D.padOver C).Dep a b → step a < step b := by
    intro a b hab
    induction hab with
    | single h => exact hlt _ _ h
    | tail _ h ih => exact ih.trans (hlt _ _ h)
  exact lt_irrefl _ (this u u h)

/-- **Padding over assignments that all read alike adds nothing**: if every assignment of `C` gives `u` the read set
it has in `σ ∈ C`, the padding reads exactly that. -/
theorem padOver_reads_eq [Fintype S] {C : Set (S → V)} {σ : S → V} (hσ : σ ∈ C) {u : U}
    (hC : ∀ τ ∈ C, D.reads u τ = D.reads u σ) : (D.padOver C).reads u = D.reads u σ := by
  ext s
  rw [D.mem_padOver_reads]
  exact ⟨fun ⟨τ, hτ, hs⟩ => hC τ hτ ▸ hs, fun hs => ⟨σ, hσ, hs⟩⟩

/-- **Theorem O1: an oblivious program is a population.** If a time-ordered, single-assignment trace's read sets
depend only on the values of the slots `A`, then at each value `a` on `A` the padding over the assignments holding `a`
is a checked population whose read sets are the program's in each such assignment, and whose wrong units are the
program's. T1–T4 apply to it as they stand. -/
theorem population_of_addressedBy [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment) {step : U → ℕ}
    (ht : D.TimeOrdered step) {A : Finset S} (hA : D.AddressedBy A) (a : S → V) :
    (D.padOver {σ | ∀ s ∈ A, σ s = a s}).Checked ∧
      (∀ σ ∈ {σ : S → V | ∀ s ∈ A, σ s = a s}, ∀ u, (D.padOver {σ | ∀ s ∈ A, σ s = a s}).reads u = D.reads u σ) ∧
      ∀ σ ∈ {σ : S → V | ∀ s ∈ A, σ s = a s},
        (D.padOver {σ | ∀ s ∈ A, σ s = a s}).wrong (D.padFns {σ | ∀ s ∈ A, σ s = a s}) σ = D.wrong σ := by
  have hreads : ∀ σ ∈ {σ : S → V | ∀ s ∈ A, σ s = a s}, ∀ u,
      (D.padOver {σ | ∀ s ∈ A, σ s = a s}).reads u = D.reads u σ :=
    fun σ hσ u => D.padOver_reads_eq hσ fun τ hτ => hA u τ σ fun s hs => (hτ s hs).trans (hσ s hs).symm
  exact ⟨D.padOver_checked _ hsa ht, hreads, fun σ hσ => D.wrong_padOver fun u => (hreads σ hσ u).symm ▸ le_rfl⟩

/-- **An oblivious program is one circuit**: its padding over every assignment reads what it reads in each. -/
theorem pad_reads_of_oblivious [Fintype S] (ho : D.Oblivious) (σ : S → V) (u : U) :
    (D.padOver Set.univ).reads u = D.reads u σ :=
  D.padOver_reads_eq (Set.mem_univ σ) fun τ _ => ho u τ σ

/-- **Locality needs no obliviousness**: in a time-ordered, address-closed trace, a run that holds the honest run's
inputs agrees with it on every slot written outside the wrong units' cone, the cone drawn in the honest run's own
dataflow. -/
theorem eq_honest_off_downAt [Fintype U] (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s) :
    ∀ v, v ∉ D.downAt h (D.wrong σ) → ∀ s ∈ D.writes v, σ s = h s := by
  intro v
  induction v using (measure step).wf.induction with
  | _ v ih =>
    intro hv s hs
    have cσ := D.correct_of_not_mem_wrong (σ := σ) fun hw => hv ⟨v, hw, .refl⟩
    have hr : ∀ r ∈ D.reads v h, σ r = h r := by
      intro r hr
      rcases ht v h r hr with hri | ⟨w, hw, hwv⟩
      · exact hin r hri
      · refine ih w hwv (fun hwd => hv ?_) r hw
        obtain ⟨u, hu, huw⟩ := hwd
        exact ⟨u, hu, huw.tail ⟨r, hw, hr⟩⟩
    have hrs : D.reads v h = D.reads v σ := hac v h σ fun r hr' => (hr r hr').symm
    rw [cσ s hs, hh v s hs]
    exact D.reads_only v σ h (fun r hr' => hr r (hrs ▸ hr')) s hs

/-- **Skipped work needs no obliviousness either**: in a time-ordered, address-closed trace, every skipped unit is in
the wrong units' cone in the honest run's dataflow, or lucky. -/
theorem skipped_mem_downAt [Fintype U] (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s) {K L : Finset U}
    (hl : D.SkipsLucky h σ K L) : ∀ u, u ∉ K → u ∈ D.downAt h (D.wrong σ) ∨ u ∈ L := by
  intro u huK
  by_cases hd : u ∈ D.downAt h (D.wrong σ)
  · exact Or.inl hd
  · refine Or.inr (hl u huK (fun r hr => ?_) (D.correct_of_not_mem_wrong fun hw => hd ⟨u, hw, .refl⟩))
    rcases ht u h r hr with hri | ⟨w, hw, -⟩
    · exact hin r hri
    · refine D.eq_honest_off_downAt hac ht hh hin w (fun hwd => hd ?_) r hw
      obtain ⟨v, hv, hvw⟩ := hwd
      exact ⟨v, hv, hvw.tail ⟨r, hw, hr⟩⟩

/-- **Addressing honest, data sampled**: if the read sets depend only on the slots `P` and those hold their honest
values, the run is an assignment of the honest run's frozen dataflow `padOver {h}`, with the program's wrong units. -/
theorem wrong_freeze [Fintype U] [Fintype S] [Inhabited V] {P : Finset S} (hP : D.AddressedBy P) {h σ : S → V}
    (hσ : ∀ p ∈ P, σ p = h p) : (D.padOver {h}).wrong (D.padFns {h}) σ = D.wrong σ :=
  D.wrong_padOver fun u => (hP u σ h hσ) ▸ D.reads_subset_padOver (Set.mem_singleton h) u

/-- **The hybrid**: in a time-ordered, address-closed trace whose read sets depend only on the slots `P`, if every
slot of `P` is an input or is written outside the wrong units' cone in the honest run, then the run is an assignment of
the honest run's frozen dataflow, which is checked, with the program's wrong units. -/
theorem wrong_freeze_of_off_downAt [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment)
    (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step) {P : Finset S} (hP : D.AddressedBy P)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s)
    (hoff : ∀ p ∈ P, p ∈ D.input ∨ ∃ v, v ∉ D.downAt h (D.wrong σ) ∧ p ∈ D.writes v) :
    (D.padOver {h}).Checked ∧ (D.padOver {h}).wrong (D.padFns {h}) σ = D.wrong σ := by
  refine ⟨D.padOver_checked _ hsa ht, D.wrong_freeze hP fun p hp => ?_⟩
  rcases hoff p hp with hpi | ⟨v, hv, hpv⟩
  · exact hin p hpi
  · exact D.eq_honest_off_downAt hac ht hh hin v hv p hpv

end DynPopulation

/-! ## The pointer machine: honestly oblivious, and still not a circuit -/

theorem vm_addressClosed (N : ℕ) : (vm N).AddressClosed := by
  intro u σ τ h
  cases u with
  | none => rfl
  | some i =>
    have hp : σ .ptr = τ .ptr := h .ptr (by simp [vm])
    show ({.ptr, src (σ .ptr) i} : Finset (Slot N)) = {.ptr, src (τ .ptr) i}
    rw [hp]

/-- **The pointer machine is oblivious in the classical sense**: every honest run, on any inputs, reads the pointer
and the zero region. -/
theorem vm_honestlyOblivious (N : ℕ) : (vm N).HonestlyOblivious := by
  intro u σ τ hσ hτ
  have hpσ : σ .ptr = false := hσ none .ptr (by simp [vm])
  have hpτ : τ .ptr = false := hτ none .ptr (by simp [vm])
  cases u with
  | none => rfl
  | some i =>
    show ({.ptr, src (σ .ptr) i} : Finset (Slot N)) = {.ptr, src (τ .ptr) i}
    rw [hpσ, hpτ]

theorem vm_not_oblivious {N : ℕ} (i : Fin N) : ¬ (vm N).Oblivious := by
  intro ho
  have := ho (some i) (fun _ => true) (fun _ => false)
  have hmem : Slot.secret i ∈ (vm N).reads (some i) (fun _ => true) := by simp [vm, src]
  rw [this] at hmem
  simp [vm, src] at hmem

/-- **The pointer machine's padding is its static declaration**: its padded read sets are `vmStatic`'s. -/
theorem vm_pad (N : ℕ) (u : Option (Fin N)) : ((vm N).padOver Set.univ).reads u = (vmStatic N).reads u := by
  ext s
  rw [DynPopulation.mem_padOver_reads]
  cases u with
  | none => simp [vm, vmStatic]
  | some i =>
    simp only [Set.mem_univ, true_and]
    constructor
    · rintro ⟨σ, hs⟩
      simp only [vm, mem_insert, mem_singleton] at hs
      simp only [vmStatic, mem_insert, mem_singleton]
      rcases hs with rfl | rfl
      · exact Or.inl rfl
      · unfold src
        split <;> simp
    · intro hs
      simp only [vmStatic, mem_insert, mem_singleton] at hs
      rcases hs with rfl | rfl | rfl
      · exact ⟨fun _ => false, by simp [vm]⟩
      · exact ⟨fun _ => false, by simp [vm, src]⟩
      · exact ⟨fun _ => true, by simp [vm, src]⟩

end Population
