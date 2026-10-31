import Definitions.Population.Dataflow

/-!
# Integrity: robust determinacy from single assignment and acyclicity

* `wellFounded_dep`: on finitely many units, an acyclic population's dependency relation is well-founded.
* `eq_of_writes`, **robust determinacy**: in a covered acyclic population, two assignments that hold the same inputs,
  whose incorrect units lie in `B`, and that agree on the slots `B` writes, are equal. An assignment is determined by its
  inputs and its incorrect units' writes. With `B = ∅` it is determinacy.
* `eq_off_down`, **the cone**: two assignments that hold the same inputs and whose incorrect units lie in `B` agree on every
  slot written by a unit outside `B`'s downstream cone. `eq_honest_off_down` reads it against the honest evaluation.
* `exists_correct`: a checked population has, on every input, an assignment in which every unit is correct
  (`eval`, by well-founded recursion over the slots).
-/

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

theorem wellFounded_dep [Finite U] (h : P.Acyclic) : WellFounded P.Dep := by
  have : IsTrans U (Relation.TransGen P.Dep) := ⟨fun _ _ _ => Relation.TransGen.trans⟩
  have : Std.Irrefl (Relation.TransGen P.Dep) := ⟨h⟩
  exact Subrelation.wf (fun h => Relation.TransGen.single h)
    (Finite.wellFounded_of_trans_of_irrefl (Relation.TransGen P.Dep))

theorem covered_of_singleAssignment (h : P.SingleAssignment) : P.Covered :=
  fun s hs => ((h s).2 hs).exists

theorem correct_of_not_mem_incorrect [Fintype U] {V : Type} {f : P.Fns V} {σ : S → V} {u : U}
    (h : u ∉ P.incorrect f σ) : P.Correct f σ u := by
  classical
  by_contra hc
  exact h (by simp [incorrect, hc])

theorem incorrect_eq_empty [Fintype U] {V : Type} {f : P.Fns V} {σ : S → V} (h : ∀ u, P.Correct f σ u) :
    P.incorrect f σ = ∅ := by
  classical
  ext u
  simp [incorrect, h u]

/-- A correct unit's writes are its function of its reads; equal reads give equal writes. -/
theorem write_eq_of_reads {V : Type} {f : P.Fns V} {σ τ : S → V} {u : U} (hσ : P.Correct f σ u)
    (hτ : P.Correct f τ u) (hr : ∀ r ∈ P.reads u, σ r = τ r) : ∀ s ∈ P.writes u, σ s = τ s := by
  intro s hs
  calc σ s = f u (fun r => σ r) ⟨s, hs⟩ := hσ ⟨s, hs⟩
    _ = f u (fun r => τ r) ⟨s, hs⟩ := by
        congr 1
        funext r
        exact hr r.1 r.2
    _ = τ s := (hτ ⟨s, hs⟩).symm

/-- **Robust determinacy**: an assignment is determined by its inputs and the writes of a set containing its incorrect
units. -/
theorem eq_of_writes [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {B : Finset U}
    {σ τ : S → V} (hin : ∀ s ∈ P.input, σ s = τ s) (hσ : P.incorrect f σ ⊆ B) (hτ : P.incorrect f τ ⊆ B)
    (hB : ∀ u ∈ B, ∀ s ∈ P.writes u, σ s = τ s) : σ = τ := by
  have key : ∀ u, ∀ s ∈ P.writes u, σ s = τ s := by
    intro u
    induction u using (P.wellFounded_dep ha).induction with
    | _ u ih =>
      by_cases huB : u ∈ B
      · exact hB u huB
      · refine P.write_eq_of_reads (P.correct_of_not_mem_incorrect fun h => huB (hσ h))
          (P.correct_of_not_mem_incorrect fun h => huB (hτ h)) ?_
        intro r hr
        by_cases hri : r ∈ P.input
        · exact hin r hri
        · obtain ⟨v, hv⟩ := hc r hri
          exact ih v ⟨r, hv, hr⟩ r hv
  funext s
  by_cases hs : s ∈ P.input
  · exact hin s hs
  · obtain ⟨v, hv⟩ := hc s hs
    exact key v s hv

/-- **Determinacy**: with no incorrect unit, an assignment is determined by its inputs. -/
theorem eq_of_correct [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {σ τ : S → V}
    (hin : ∀ s ∈ P.input, σ s = τ s) (hσ : ∀ u, P.Correct f σ u) (hτ : ∀ u, P.Correct f τ u) : σ = τ :=
  P.eq_of_writes hc ha f (B := ∅) hin (by simp [P.incorrect_eq_empty hσ]) (by simp [P.incorrect_eq_empty hτ])
    (by simp)

theorem mem_down [Fintype U] {B : Finset U} {v : U} :
    v ∈ P.down B ↔ ∃ u ∈ B, Relation.ReflTransGen P.Dep u v := by
  classical
  simp [down]

/-- **The cone**: assignments that hold the same inputs and whose incorrect units lie in `B` agree on every slot that a
unit outside `B`'s downstream cone writes. -/
theorem eq_off_down [Fintype U] (hc : P.Covered) (ha : P.Acyclic) {V : Type} (f : P.Fns V) {B : Finset U}
    {σ τ : S → V} (hin : ∀ s ∈ P.input, σ s = τ s) (hσ : P.incorrect f σ ⊆ B) (hτ : P.incorrect f τ ⊆ B) :
    ∀ v, v ∉ P.down B → ∀ s ∈ P.writes v, σ s = τ s := by
  intro v
  induction v using (P.wellFounded_dep ha).induction with
  | _ v ih =>
    intro hv
    have hvB : v ∉ B := fun h => hv ((P.mem_down).2 ⟨v, h, Relation.ReflTransGen.refl⟩)
    refine P.write_eq_of_reads (P.correct_of_not_mem_incorrect fun h => hvB (hσ h))
      (P.correct_of_not_mem_incorrect fun h => hvB (hτ h)) ?_
    intro r hr
    by_cases hri : r ∈ P.input
    · exact hin r hri
    · obtain ⟨w, hw⟩ := hc r hri
      have hd : P.Dep w v := ⟨r, hw, hr⟩
      refine ih w hd ?_ r hw
      intro hwd
      obtain ⟨u, hu, huw⟩ := (P.mem_down).1 hwd
      exact hv ((P.mem_down).2 ⟨u, hu, huw.tail hd⟩)

/-! ## The honest evaluation -/

section Eval

variable {P}

/-- The writer of a non-input slot. -/
noncomputable def writer (hsa : P.SingleAssignment) (s : S) (h : s ∉ P.input) : U := ((hsa s).2 h).exists.choose

theorem writer_spec (hsa : P.SingleAssignment) (s : S) (h : s ∉ P.input) : s ∈ P.writes (writer hsa s h) :=
  ((hsa s).2 h).exists.choose_spec

theorem writer_eq (hsa : P.SingleAssignment) {s : S} (h : s ∉ P.input) {u : U} (hu : s ∈ P.writes u) :
    writer hsa s h = u :=
  ((hsa s).2 h).unique (writer_spec hsa s h) hu

/-- `r` comes before `s`: `s`'s writer reads `r`. -/
def Before (hsa : P.SingleAssignment) (r s : S) : Prop := ∃ h : s ∉ P.input, r ∈ P.reads (writer hsa s h)

theorem wellFounded_before [Finite U] (hsa : P.SingleAssignment) (ha : P.Acyclic) : WellFounded (Before hsa) := by
  have hin : ∀ s ∈ P.input, Acc (Before hsa) s := fun s hs => Acc.intro s fun _ ⟨h, _⟩ => absurd hs h
  have key : ∀ u, ∀ s (h : s ∉ P.input), writer hsa s h = u → Acc (Before hsa) s := by
    intro u
    induction u using (P.wellFounded_dep ha).induction with
    | _ u ih =>
      intro s h hw
      refine Acc.intro s fun r ⟨h', hr⟩ => ?_
      by_cases hri : r ∈ P.input
      · exact hin r hri
      · have hd : P.Dep (writer hsa r hri) u := ⟨r, writer_spec hsa r hri, by
          have : writer hsa s h' = u := by rw [← hw]
          rw [← this]; exact hr⟩
        exact ih _ hd r hri rfl
  classical
  exact ⟨fun s => if h : s ∈ P.input then hin s h else key _ s h rfl⟩

open Classical in
/-- **The honest evaluation**: an input slot holds its input, and every other slot its writer's function of the slots
that writer reads, evaluated first. -/
noncomputable def eval [Finite U] (hsa : P.SingleAssignment) (ha : P.Acyclic) {V : Type} (f : P.Fns V)
    (a : S → V) : S → V :=
  (wellFounded_before hsa ha).fix fun s rec =>
    if h : s ∈ P.input then a s
    else f (writer hsa s h) (fun r => rec r.1 ⟨h, r.2⟩) ⟨s, writer_spec hsa s h⟩

theorem eval_input [Finite U] (hsa : P.SingleAssignment) (ha : P.Acyclic) {V : Type} (f : P.Fns V) (a : S → V)
    {s : S} (h : s ∈ P.input) : eval hsa ha f a s = a s := by
  rw [eval, WellFounded.fix_eq]
  simp only [h, ↓reduceDIte]

theorem eval_correct [Finite U] (hsa : P.SingleAssignment) (ha : P.Acyclic) {V : Type} (f : P.Fns V) (a : S → V)
    (u : U) : P.Correct f (eval hsa ha f a) u := by
  intro ⟨s, hs⟩
  have hsi : s ∉ P.input := fun h => (hsa s).1 h u hs
  have hw : writer hsa s hsi = u := writer_eq hsa hsi hs
  show eval hsa ha f a s = _
  rw [eval, WellFounded.fix_eq]
  simp only [hsi, ↓reduceDIte]
  have : ∀ u' (e : u' = u) (h1 : s ∈ P.writes u'),
      f u' (fun r => eval hsa ha f a r.1) ⟨s, h1⟩ = f u (fun r => eval hsa ha f a r.1) ⟨s, hs⟩ := by
    intro u' e h1; subst e; rfl
  exact this _ hw _

end Eval

/-- A checked population has, on every input, an assignment in which every unit is correct. -/
theorem exists_correct [Finite U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) :
    ∃ σ : S → V, (∀ s ∈ P.input, σ s = a s) ∧ ∀ u, P.Correct f σ u :=
  ⟨eval h.1 h.2 f a, fun _ hs => eval_input h.1 h.2 f a hs, eval_correct h.1 h.2 f a⟩

/-- **Integrity against the honest run**: in a checked population, an assignment that holds the inputs agrees with
the honest one (every unit correct, the same inputs) on every slot written by a unit outside its incorrect units'
downstream cone. -/
theorem eq_honest_off_down [Fintype U] (h : P.Checked) {V : Type} (f : P.Fns V) (a : S → V) {honest σ : S → V}
    (hh : ∀ s ∈ P.input, honest s = a s) (hc : ∀ u, P.Correct f honest u) (hσ : ∀ s ∈ P.input, σ s = a s) :
    ∀ v, v ∉ P.down (P.incorrect f σ) → ∀ s ∈ P.writes v, σ s = honest s :=
  P.eq_off_down (P.covered_of_singleAssignment h.1) h.2 f (fun s hs => (hσ s hs).trans (hh s hs).symm) le_rfl
    (by simp [P.incorrect_eq_empty hc])

end Population.Dataflow
