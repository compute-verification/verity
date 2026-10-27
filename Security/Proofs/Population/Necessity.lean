import Proofs.Population.Determinacy

/-!
# Integrity: what the checker confirms is exactly determinacy for every choice of unit functions

`determinate_iff_checked`: a population is `Determinate` (every input has exactly one assignment making every unit
correct, whatever the units compute) if and only if it is `Checked` (single assignment and acyclic). A circuit is not
needed; the dataflow and these two checks are.

The counterexamples, each a choice of Boolean unit functions:
* a unit writing an input slot, and an input it disagrees with: no correct assignment;
* two units writing one slot, one writing `true` and one `false`: no correct assignment;
* a slot no unit writes: two correct assignments, differing there;
* a cycle: every unit of the cycle's strongly connected component computes the OR of what it reads, and every other
  unit `false`; then "all `false`" and "`true` on the component's writes" are both correct (`cycle_two_fixed_points`):
  a self-consistent wrong fixed point, with no faulty proof block.
-/

namespace Population.Dataflow

open Finset

variable {U S : Type} (P : Dataflow U S)

theorem determinate_of_checked [Finite U] (h : P.Checked) : P.Determinate := by
  have := Fintype.ofFinite U
  intro V f a
  obtain ⟨σ, hσa, hσc⟩ := P.exists_correct h f a
  refine ⟨σ, ⟨hσa, hσc⟩, ?_⟩
  rintro τ ⟨hτa, hτc⟩
  exact P.eq_of_correct (P.covered_of_singleAssignment h.1) h.2 f
    (fun s hs => (hτa s hs).trans (hσa s hs).symm) hτc hσc

theorem singleAssignment_of_determinate (h : P.Determinate) : P.SingleAssignment := by
  classical
  intro s
  constructor
  · intro hs u hu
    obtain ⟨σ, ⟨hσa, hσc⟩, -⟩ := h Bool (fun _ _ _ => true) (fun _ => false)
    have h1 : σ s = false := hσa s hs
    have h2 : σ s = true := hσc u ⟨s, hu⟩
    rw [h1] at h2
    exact Bool.false_ne_true h2
  · intro hs
    by_cases hw : ∃ u, s ∈ P.writes u
    · obtain ⟨u, hu⟩ := hw
      refine ⟨u, hu, fun u' hu' => ?_⟩
      by_contra hne
      obtain ⟨σ, ⟨-, hσc⟩, -⟩ := h Bool (fun v _ _ => decide (v = u)) (fun _ => false)
      have h1 : σ s = decide (u = u) := hσc u ⟨s, hu⟩
      have h2 : σ s = decide (u' = u) := hσc u' ⟨s, hu'⟩
      rw [h1] at h2
      simp [hne] at h2
    · exfalso
      push Not at hw
      obtain ⟨σ, ⟨hσa, hσc⟩, huniq⟩ := h Bool (fun _ _ _ => false) (fun _ => false)
      have heq := huniq (Function.update σ s (!σ s)) ⟨fun t ht => ?_, fun u t => ?_⟩
      · have := congrFun heq s
        simp at this
      · rw [Function.update_of_ne (by rintro rfl; exact hs ht)]
        exact hσa t ht
      · have hts : (t : S) ≠ s := by
          intro e
          exact hw u (e ▸ t.2)
        show Function.update σ s (!σ s) t = false
        rw [Function.update_of_ne hts]
        exact hσc u t

/-- In a single-assignment population with a cycle through `u₀`, two different assignments hold the all-`false`
inputs and make every unit correct, for the unit functions that OR their reads on the cycle's strongly connected
component and write `false` elsewhere. -/
theorem cycle_two_fixed_points (hsa : P.SingleAssignment) {u₀ : U} (hcyc : Relation.TransGen P.Dep u₀ u₀) :
    ∃ (f : P.Fns Bool) (σ τ : S → Bool), σ ≠ τ ∧ (∀ s ∈ P.input, σ s = false) ∧ (∀ s ∈ P.input, τ s = false) ∧
      (∀ u, P.Correct f σ u) ∧ ∀ u, P.Correct f τ u := by
  classical
  let C : U → Prop := fun v => Relation.TransGen P.Dep u₀ v ∧ Relation.TransGen P.Dep v u₀
  have pred : ∀ v, C v → ∃ v', C v' ∧ P.Dep v' v := by
    intro v ⟨h1, h2⟩
    obtain ⟨b, hb, hbv⟩ := Relation.TransGen.tail'_iff.1 h1
    refine ⟨b, ⟨?_, Relation.TransGen.head hbv h2⟩, hbv⟩
    rcases Relation.reflTransGen_iff_eq_or_transGen.1 hb with e | ht
    · exact e ▸ hcyc
    · exact ht
  let W : S → Prop := fun s => ∃ v, C v ∧ s ∈ P.writes v
  let sig : Bool → S → Bool := fun b s => b && decide (W s)
  let f : P.Fns Bool := fun v r _ => decide (C v ∧ ∃ x, r x = true)
  have hW_input : ∀ s ∈ P.input, ¬ W s := fun s hs ⟨v, _, hv⟩ => (hsa s).1 hs v hv
  have hcor : ∀ b v, P.Correct f (sig b) v := by
    intro b v ⟨s, hs⟩
    show (b && decide (W s)) = decide (C v ∧ ∃ x : P.reads v, (b && decide (W x)) = true)
    by_cases hv : C v
    · have hW : W s := ⟨v, hv, hs⟩
      cases b
      · simp
      · obtain ⟨v', hv', r, hr1, hr2⟩ := pred v hv
        have : ∃ x : P.reads v, W x := ⟨⟨r, hr2⟩, v', hv', hr1⟩
        simp [hW, hv, this]
    · have hW : ¬ W s := by
        rintro ⟨v', hv', hs'⟩
        have hsi : s ∉ P.input := fun h => (hsa s).1 h v hs
        have := ((hsa s).2 hsi).unique hs hs'
        exact hv (this ▸ hv')
      simp [hW, hv]
  obtain ⟨v', hv', r, hr1, -⟩ := pred u₀ ⟨hcyc, hcyc⟩
  refine ⟨f, sig true, sig false, fun e => ?_, fun s hs => by simp [sig, hW_input s hs],
    fun s hs => by simp [sig], hcor true, hcor false⟩
  have := congrFun e r
  have hr : W r := ⟨v', hv', hr1⟩
  simp [sig, hr] at this

theorem acyclic_of_determinate (h : P.Determinate) : P.Acyclic := by
  intro u₀ hcyc
  obtain ⟨f, σ, τ, hne, hσa, hτa, hσc, hτc⟩ :=
    P.cycle_two_fixed_points (P.singleAssignment_of_determinate h) hcyc
  obtain ⟨_, _, huniq⟩ := h Bool f (fun _ => false)
  exact hne ((huniq σ ⟨hσa, hσc⟩).trans (huniq τ ⟨hτa, hτc⟩).symm)

/-- **Integrity's necessary and sufficient condition**: determinacy for every choice of unit functions holds exactly
when the dataflow is single assignment and acyclic. -/
theorem determinate_iff_checked [Finite U] : P.Determinate ↔ P.Checked :=
  ⟨fun h => ⟨P.singleAssignment_of_determinate h, P.acyclic_of_determinate h⟩, P.determinate_of_checked⟩

end Population.Dataflow
