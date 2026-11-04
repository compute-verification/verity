import Definitions.Population.Oblivious
import Proofs.Population.Determinacy
import Proofs.Population.Pointer

/-!
# Which programs are as good as circuits

A program's trace is a `DynPopulation`. Three facts decide what the population theorems give it.

* `padOver_checked`, `incorrect_padOver`, **every program pads to a population**: for a time-ordered, single-assignment
  trace and any set `C` of assignments, the padding over `C` is a checked population, and in every assignment whose
  read sets it covers, its incorrect units are the program's. With `C = Set.univ` it covers every assignment
  (`incorrect_pad`): the padded static circuit, with T1–T4 at its public cones.
* `population_of_addressedBy`, **an oblivious program is that population with nothing added** (Theorem O1): if the read
  sets depend only on the values of the slots `A`, then at each value `a` on `A`, the padding over the assignments
  holding `a` is checked, has exactly the program's read sets in each of them, and the same incorrect units. With `A` the
  public inputs it is a circuit indexed by them; with `A = ∅` (`Oblivious`), one circuit.
* `frozen_circuit` (Theorem O2), **a classically oblivious program compiled to its circuit**: over a set of honest runs
  that read alike, the padding is a checked population with their wiring, none of them incorrect, and no other run without
  an incorrect unit. It judges a unit by that wiring, not the program's: `vm_frozen_reads` is the pointer machine's.
* `eq_honest_off_downAt`, **locality needs no obliviousness**: in a time-ordered, address-closed trace, every run
  agrees with the honest one off the incorrect units' cone in the honest run's own dataflow, and `skipped_mem_downAt`
  puts every skipped unit in that cone or among the lucky. What a program lacks is that this cone be public, and
  isolation: `vm` is address-closed and honestly oblivious (`vm_honestlyOblivious`), so the
  classical notion holds, and still one incorrect one-bit unit copies `N` secret words that no honest run reads
  (`vm_leak`, `vm_honest_up`). Its padding is `vmStatic` (`vm_pad`), whose wiring shows the secrets upstream.
* `leak_of_moved_read`, **the converse**: if two assignments that differ only on one unit's writes give a step
  different read sets, one of them an input slot `t` the other doesn't read, then for some unit functions (`leaker`)
  every honest run's egress is the same whatever `t` holds, and with that one unit incorrect the egress copies `t`.
  `vm_moved_read` is the pointer machine meeting its hypotheses.
* `incorrect_freeze`, `incorrect_freeze_of_off_downAt`, `incorrect_freeze_of_off_padDown`, **addressing proved, data sampled**
  (the hybrid): when the slots the read sets depend on hold their honest values, which holds when none is written
  inside the incorrect units' cone (in the honest run, or in the public padding), the run is an assignment of the honest
  run's own frozen dataflow, with the same incorrect units. `incorrect_padOver_of_mem`: a run that a full check confines to a
  set `C` is an assignment of the padding over `C`.
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

/-- **The padding's incorrect units are the program's**, in every assignment whose read sets it covers. -/
theorem incorrect_padOver [Fintype U] [Fintype S] [Inhabited V] {C : Set (S → V)} {σ : S → V}
    (hσ : ∀ u, D.reads u σ ⊆ (D.padOver C).reads u) :
    (D.padOver C).incorrect (D.padFns C) σ = D.incorrect σ := by
  classical
  ext u
  simp only [Dataflow.incorrect, DynPopulation.incorrect, mem_filter, mem_univ, true_and]
  exact not_congr (D.padOver_correct_iff (hσ u))

/-- **A run the checks confine to `C` is an assignment of the padding over `C`**, with the program's incorrect units. -/
theorem incorrect_padOver_of_mem [Fintype U] [Fintype S] [Inhabited V] {C : Set (S → V)} {σ : S → V} (hσ : σ ∈ C) :
    (D.padOver C).incorrect (D.padFns C) σ = D.incorrect σ :=
  D.incorrect_padOver fun u => D.reads_subset_padOver hσ u

/-- **The static superset covers every assignment**: the padded static circuit's incorrect units are the program's. -/
theorem incorrect_pad [Fintype U] [Fintype S] [Inhabited V] (σ : S → V) :
    (D.padOver Set.univ).incorrect (D.padFns Set.univ) σ = D.incorrect σ :=
  D.incorrect_padOver fun u => D.reads_subset_padOver (Set.mem_univ σ) u

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
is a checked population whose read sets are the program's in each such assignment, and whose incorrect units are the
program's. T1–T4 apply to it as they stand. -/
theorem population_of_addressedBy [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment) {step : U → ℕ}
    (ht : D.TimeOrdered step) {A : Finset S} (hA : D.AddressedBy A) (a : S → V) :
    (D.padOver {σ | ∀ s ∈ A, σ s = a s}).Checked ∧
      (∀ σ ∈ {σ : S → V | ∀ s ∈ A, σ s = a s}, ∀ u, (D.padOver {σ | ∀ s ∈ A, σ s = a s}).reads u = D.reads u σ) ∧
      ∀ σ ∈ {σ : S → V | ∀ s ∈ A, σ s = a s},
        (D.padOver {σ | ∀ s ∈ A, σ s = a s}).incorrect (D.padFns {σ | ∀ s ∈ A, σ s = a s}) σ = D.incorrect σ := by
  have hreads : ∀ σ ∈ {σ : S → V | ∀ s ∈ A, σ s = a s}, ∀ u,
      (D.padOver {σ | ∀ s ∈ A, σ s = a s}).reads u = D.reads u σ :=
    fun σ hσ u => D.padOver_reads_eq hσ fun τ hτ => hA u τ σ fun s hs => (hτ s hs).trans (hσ s hs).symm
  exact ⟨D.padOver_checked _ hsa ht, hreads, fun σ hσ => D.incorrect_padOver fun u => (hreads σ hσ u).symm ▸ le_rfl⟩

/-- **An oblivious program is one circuit**: its padding over every assignment reads what it reads in each. -/
theorem pad_reads_of_oblivious [Fintype S] (ho : D.Oblivious) (σ : S → V) (u : U) :
    (D.padOver Set.univ).reads u = D.reads u σ :=
  D.padOver_reads_eq (Set.mem_univ σ) fun τ _ => ho u τ σ

/-- **Theorem O2, a classically oblivious program compiled to its circuit**: for a set `C` of honest runs that all read
alike (every honest run, for an honestly oblivious program; those holding the public values, relative to them), the
padding over `C` is a checked population that reads what each of them reads, has none of them incorrect, and has no other
run without an incorrect unit holding one's inputs. Its incorrect units are its own: a run that moves a read off the
honest wiring is incorrect in it, where in the program it need not be (`vm_leak`). -/
theorem frozen_circuit [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment) {step : U → ℕ}
    (ht : D.TimeOrdered step) {C : Set (S → V)} (hC : ∀ σ ∈ C, ∀ v, D.Correct σ v)
    (hread : ∀ u, ∀ σ ∈ C, ∀ τ ∈ C, D.reads u σ = D.reads u τ) :
    (D.padOver C).Checked ∧ (∀ σ ∈ C, ∀ u, (D.padOver C).reads u = D.reads u σ) ∧
      (∀ σ ∈ C, (D.padOver C).incorrect (D.padFns C) σ = ∅) ∧
      ∀ h ∈ C, ∀ σ : S → V, (D.padOver C).incorrect (D.padFns C) σ = ∅ → (∀ s ∈ D.input, σ s = h s) → σ = h := by
  classical
  have hch := D.padOver_checked C hsa ht
  have hw : ∀ σ ∈ C, (D.padOver C).incorrect (D.padFns C) σ = ∅ := fun σ hσ => by
    rw [D.incorrect_padOver_of_mem hσ]
    ext u
    simp [DynPopulation.incorrect, hC σ hσ u]
  refine ⟨hch, fun σ hσ u => D.padOver_reads_eq hσ fun τ hτ => hread u τ hτ σ hσ, hw, fun h hh σ hσ hin => ?_⟩
  have hc := (D.padOver C).covered_of_singleAssignment hch.1
  funext s
  by_cases hs : s ∈ (D.padOver C).input
  · exact hin s hs
  · obtain ⟨v, hv⟩ := hc s hs
    refine (D.padOver C).eq_off_down hc hch.2 (D.padFns C) (B := ∅) hin (by rw [hσ]) (by rw [hw h hh]) v
      (fun hvd => ?_) s hv
    obtain ⟨u, hu, -⟩ := (D.padOver C).mem_down.1 hvd
    simp at hu

/-- **Locality needs no obliviousness**: in a time-ordered, address-closed trace, a run that holds the honest run's
inputs agrees with it on every slot written outside the incorrect units' cone, the cone drawn in the honest run's own
dataflow. -/
theorem eq_honest_off_downAt [Fintype U] (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s) :
    ∀ v, v ∉ D.downAt h (D.incorrect σ) → ∀ s ∈ D.writes v, σ s = h s := by
  intro v
  induction v using (measure step).wf.induction with
  | _ v ih =>
    intro hv s hs
    have cσ := D.correct_of_not_mem_incorrect (σ := σ) fun hw => hv ⟨v, hw, .refl⟩
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
the incorrect units' cone in the honest run's dataflow, or lucky. -/
theorem skipped_mem_downAt [Fintype U] (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s) {K L : Finset U}
    (hl : D.SkipsLucky h σ K L) : ∀ u, u ∉ K → u ∈ D.downAt h (D.incorrect σ) ∨ u ∈ L := by
  intro u huK
  by_cases hd : u ∈ D.downAt h (D.incorrect σ)
  · exact Or.inl hd
  · refine Or.inr (hl u huK (fun r hr => ?_) (D.correct_of_not_mem_incorrect fun hw => hd ⟨u, hw, .refl⟩))
    rcases ht u h r hr with hri | ⟨w, hw, -⟩
    · exact hin r hri
    · refine D.eq_honest_off_downAt hac ht hh hin w (fun hwd => hd ?_) r hw
      obtain ⟨v, hv, hvw⟩ := hwd
      exact ⟨v, hv, hvw.tail ⟨r, hw, hr⟩⟩

/-- **The honest run's cone lies inside the padded one**: an honest read is a padded read. -/
theorem downAt_subset_pad_down [Fintype U] [Fintype S] (σ : S → V) (B : Finset U) :
    D.downAt σ B ⊆ ((D.padOver Set.univ).down B : Set U) := by
  rintro v ⟨u, hu, huv⟩
  refine Finset.mem_coe.2 (((D.padOver Set.univ).mem_down).2 ⟨u, hu, ?_⟩)
  induction huv with
  | refl => exact .refl
  | tail _ hbc ih =>
    obtain ⟨s, hs, hsr⟩ := hbc
    exact ih.tail ⟨s, hs, D.reads_subset_padOver (Set.mem_univ σ) _ hsr⟩

/-- **Addressing honest, data sampled**: if the read sets depend only on the slots `P` and those hold their honest
values, the run is an assignment of the honest run's frozen dataflow `padOver {h}`, with the program's incorrect units. -/
theorem incorrect_freeze [Fintype U] [Fintype S] [Inhabited V] {P : Finset S} (hP : D.AddressedBy P) {h σ : S → V}
    (hσ : ∀ p ∈ P, σ p = h p) : (D.padOver {h}).incorrect (D.padFns {h}) σ = D.incorrect σ :=
  D.incorrect_padOver fun u => (hP u σ h hσ) ▸ D.reads_subset_padOver (Set.mem_singleton h) u

/-- **The hybrid**: in a time-ordered, address-closed trace whose read sets depend only on the slots `P`, if every
slot of `P` is an input or is written outside the incorrect units' cone in the honest run, then the run is an assignment of
the honest run's frozen dataflow, which is checked, with the program's incorrect units. -/
theorem incorrect_freeze_of_off_downAt [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment)
    (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step) {P : Finset S} (hP : D.AddressedBy P)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s)
    (hoff : ∀ p ∈ P, p ∈ D.input ∨ ∃ v, v ∉ D.downAt h (D.incorrect σ) ∧ p ∈ D.writes v) :
    (D.padOver {h}).Checked ∧ (D.padOver {h}).incorrect (D.padFns {h}) σ = D.incorrect σ := by
  refine ⟨D.padOver_checked _ hsa ht, D.incorrect_freeze hP fun p hp => ?_⟩
  rcases hoff p hp with hpi | ⟨v, hv, hpv⟩
  · exact hin p hpi
  · exact D.eq_honest_off_downAt hac ht hh hin v hv p hpv

/-- **The hybrid, stated over the public padding**: if every slot the read sets depend on is an input or written
outside the padded cone of the corruptible units `Bc`, then every run whose incorrect units lie in `Bc` is an assignment
of the honest run's frozen dataflow, which is checked, with the program's incorrect units. -/
theorem incorrect_freeze_of_off_padDown [Fintype U] [Fintype S] [Inhabited V] (hsa : D.SingleAssignment)
    (hac : D.AddressClosed) {step : U → ℕ} (ht : D.TimeOrdered step) {P : Finset S} (hP : D.AddressedBy P)
    {Bc : Finset U} (hoff : ∀ p ∈ P, p ∈ D.input ∨ ∃ v, v ∉ (D.padOver Set.univ).down Bc ∧ p ∈ D.writes v)
    {h σ : S → V} (hh : ∀ u, D.Correct h u) (hin : ∀ s ∈ D.input, σ s = h s) (hσ : D.incorrect σ ⊆ Bc) :
    (D.padOver {h}).Checked ∧ (D.padOver {h}).incorrect (D.padFns {h}) σ = D.incorrect σ := by
  refine D.incorrect_freeze_of_off_downAt hsa hac ht hP hh hin fun p hp => ?_
  rcases hoff p hp with hpi | ⟨v, hv, hpv⟩
  · exact Or.inl hpi
  · refine Or.inr ⟨v, fun hvd => hv ?_, hpv⟩
    obtain ⟨u, hu, huv⟩ := hvd
    exact D.downAt_subset_pad_down h Bc ⟨u, hσ hu, huv⟩

open Classical in
/-- **The leaker**: `D`'s read sets and writes, every unit writing the constants `c`, except that unit `r` writes the
value of slot `t` to its slot `o` whenever it reads `t`. -/
noncomputable def leaker (hac : D.AddressClosed) (c : S → V) (r : U) (o t : S) : DynPopulation U S V where
  input := D.input
  writes := D.writes
  reads := D.reads
  f v σ s := if v = r ∧ s = o ∧ t ∈ D.reads r σ then σ t else c s
  reads_only := by
    intro v σ τ h s _
    by_cases hv : v = r
    · subst hv
      have hrs := hac v σ τ h
      by_cases ht : t ∈ D.reads v σ
      · have ht' : t ∈ D.reads v τ := hrs ▸ ht
        simp [ht, ht', h t ht]
      · have ht' : t ∉ D.reads v τ := hrs ▸ ht
        simp [ht, ht']
    · simp [hv]

/-- **The converse: a read set that a corruptible unit's writes move is not a circuit's.** In a time-ordered,
single-assignment, address-closed trace addressed by the slots `P`, suppose `σ'` differs from `σ` only on unit `u`'s
writes and moves unit `r` onto an input slot `t` that it does not read in `σ`, with neither `t` nor `r`'s slot `o`
among the address slots. Then, for the leaker's functions: every honest run holding `σ`'s inputs off `t` writes the
same value at `o`, whatever `t` holds; and for every value `x`, some run holding those inputs and `x` at `t`, whose
only incorrect unit is `u`, writes `x` at `o`. One incorrect unit, and the egress copies a slot no honest run reads. -/
theorem leak_of_moved_read [Fintype U] (hsa : D.SingleAssignment) (hac : D.AddressClosed) {step : U → ℕ}
    (ht : D.TimeOrdered step) {P : Finset S} (hP : D.AddressedBy P) {u r : U} {σ σ' : S → V}
    (hoff : ∀ s, s ∉ D.writes u → σ' s = σ s) {t o : S} (htin : t ∈ D.input) (ho : o ∈ D.writes r) (htP : t ∉ P)
    (hoP : o ∉ P) (hmoved : t ∈ D.reads r σ') (hnot : t ∉ D.reads r σ) :
    (∀ h : S → V, (∀ v, (D.leaker hac σ r o t).Correct h v) → (∀ s ∈ D.input, s ≠ t → h s = σ s) → h o = σ o) ∧
      ∀ x : V, ∃ g : S → V, (∀ s ∈ D.input, s ≠ t → g s = σ s) ∧ g t = x ∧
        (D.leaker hac σ r o t).incorrect g ⊆ {u} ∧ g o = x := by
  classical
  set L := D.leaker hac σ r o t with hL
  have hot : o ≠ t := fun e => (hsa t).1 htin r (e ▸ ho)
  have huniq : ∀ v s, s ∈ D.writes v → ∀ w, s ∈ D.writes w → v = w := fun v s hv w hw =>
    ((hsa s).2 fun hin => (hsa s).1 hin v hv).unique hv hw
  have hoR : ∀ v, o ∈ D.writes v → v = r := fun v hv => huniq v o hv r ho
  -- the honest run on secret `x`: `σ` with `x` at `t`
  have honest : ∀ x : V, ∀ v, L.Correct (Function.update σ t x) v := by
    intro x v s hs
    have hst : s ≠ t := fun e => (hsa t).1 htin v (e ▸ hs)
    have hrd : D.reads r (Function.update σ t x) = D.reads r σ :=
      (hac r σ _ fun q hq => by
        have : q ≠ t := fun e => hnot (e ▸ hq)
        simp [Function.update_of_ne this]).symm
    show Function.update σ t x s = if v = r ∧ s = o ∧ t ∈ D.reads r (Function.update σ t x) then
      Function.update σ t x t else σ s
    rw [hrd, Function.update_of_ne hst]
    simp [hnot]
  refine ⟨fun h hh hin => ?_, fun x => ?_⟩
  · have hcov : L.Covered := fun s hs => ((hsa s).2 hs).exists
    have heq := L.eq_of_timeOrdered hcov (step := step) ht (B := ∅) (σ := h) (τ := Function.update σ t (h t))
      (fun s hs => by
        by_cases hst : s = t
        · subst hst; simp
        · rw [Function.update_of_ne hst]; exact hin s hs hst)
      (fun v hv => by simp [DynPopulation.incorrect, hh v] at hv)
      (fun v hv => by simp [DynPopulation.incorrect, honest (h t) v] at hv)
      (fun v hv => by simp at hv)
    rw [heq, Function.update_of_ne hot]
  · let g : S → V := Function.update (Function.update σ' t x) o x
    have hgP : ∀ q ∈ P, g q = σ' q := fun q hq => by
      have h1 : q ≠ o := fun e => hoP (e ▸ hq)
      have h2 : q ≠ t := fun e => htP (e ▸ hq)
      simp [g, Function.update_of_ne h1, Function.update_of_ne h2]
    have hrg : D.reads r g = D.reads r σ' := hP r g σ' hgP
    refine ⟨g, fun s hs hst => ?_, by simp [g, Function.update_of_ne hot.symm], fun v hv => ?_, by simp [g]⟩
    · have hso : s ≠ o := fun e => (hsa s).1 hs r (e ▸ ho)
      simp only [g, Function.update_of_ne hso, Function.update_of_ne hst]
      exact hoff s fun hw => (hsa s).1 hs u hw
    · simp only [mem_singleton]
      by_contra hvu
      simp only [DynPopulation.incorrect, mem_filter, mem_univ, true_and] at hv
      refine hv fun s hs => ?_
      show g s = if v = r ∧ s = o ∧ t ∈ D.reads r g then g t else σ s
      have hst : s ≠ t := fun e => (hsa t).1 htin v (e ▸ hs)
      by_cases hso : s = o
      · subst hso
        have hvr : v = r := hoR v hs
        rw [hrg]
        simp [hvr, hmoved, g, Function.update_of_ne hot.symm]
      · have hsu : s ∉ D.writes u := fun hw => hvu (huniq v s hs u hw)
        have hgs : g s = σ s := by
          simp only [g, Function.update_of_ne hso, Function.update_of_ne hst]
          exact hoff s hsu
        rw [hgs]
        simp [hso]

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

theorem vm_addressedBy (N : ℕ) : (vm N).AddressedBy {.ptr} := by
  intro u σ τ h
  have hp : σ .ptr = τ .ptr := h .ptr (mem_singleton_self _)
  cases u with
  | none => rfl
  | some i =>
    show ({.ptr, src (σ .ptr) i} : Finset (Slot N)) = {.ptr, src (τ .ptr) i}
    rw [hp]

/-- **The converse is not vacuous**: the pointer machine meets its hypotheses, the pointer step moving copy step `i`
from zero word `i` onto secret word `i`. -/
theorem vm_moved_read {N : ℕ} (i : Fin N) :
    (∀ h : Slot N → Bool, (∀ v, ((vm N).leaker (vm_addressClosed N) (fun _ => false) (some i) (.out i)
        (.secret i)).Correct h v) → (∀ s ∈ (vm N).input, s ≠ .secret i → h s = false) → h (.out i) = false) ∧
      ∀ x : Bool, ∃ g : Slot N → Bool, (∀ s ∈ (vm N).input, s ≠ .secret i → g s = false) ∧ g (.secret i) = x ∧
        ((vm N).leaker (vm_addressClosed N) (fun _ => false) (some i) (.out i) (.secret i)).incorrect g ⊆ {none} ∧
        g (.out i) = x :=
  (vm N).leak_of_moved_read (vm_singleAssignment N) (vm_addressClosed N) (vm_timeOrdered N) (vm_addressedBy N)
    (u := none) (σ' := Function.update (fun _ => false) .ptr true)
    (fun s hs => by
      have : s ≠ .ptr := fun e => hs (by simp [vm, e])
      simp [Function.update_of_ne this])
    (by simp [vm]) (by simp [vm]) (by simp) (by simp) (by simp [vm, src]) (by simp [vm, src])

/-- **The pointer machine compiled to its circuit**: over its honest runs, copy step `i` reads the pointer and zero
word `i`, and no secret, so in that circuit (`frozen_circuit`) no secret is upstream of the egress. -/
theorem vm_frozen_reads (N : ℕ) (i : Fin N) :
    ((vm N).padOver {σ | ∀ v, (vm N).Correct σ v}).reads (some i) = {.ptr, .zero i} := by
  have h0 : (fun _ => false) ∈ {σ : Slot N → Bool | ∀ v, (vm N).Correct σ v} := by
    intro v s hs
    cases v <;> simp_all [vm, src]
  rw [(vm N).padOver_reads_eq h0 fun τ hτ => vm_honestlyOblivious N (some i) τ _ hτ h0]
  simp [vm, src]

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
