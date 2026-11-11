module

public import Definitions.Nci.Formalism
public import Proofs.Core.Circuit

/-!
# The training formalism v2: reconstruction, and the three kinds of a block's inputs

Two lemmas of the formalism (`Definitions.Nci.Formalism`), which its theorems rest on:

* `reconstruction` (§1): for a set of gates `S` and any `x` that holds the values of `in(S)`, the reconstruction of `S`
  from `x` evaluates `|S|` gates, one each, and holds every value `S` computes. It is `foldl_update_eval`'s case of all
  of `S`, sorted: evaluating gates of `S` in increasing order, each on what the earlier ones left, from values that are
  the circuit's on `in(S)` and on the gates of `S` not yet evaluated, leaves the circuit's value at every gate of `S`;
* `reads_split` (§2): `in(I)` is a block's registered values, its data and its computed values, and `in(I) \ θ`, what
  `k` bounds, is the last two. `mem_inputs_of_mem_θ`, a registered value is an input, makes it set algebra.
-/

@[expose] public section

namespace Nci.Formalism

open Finset
open Verity (Circuit RegisteredCircuit)

theorem foldl_update_eval {C : Circuit} (S : Finset (Fin C.N)) (hS : GateSet C S) (a : C.Trace)
    (l : List (Fin C.N)) (hl : l.Pairwise (· < ·)) (hlS : ∀ g ∈ l, g ∈ S) (y : C.Trace)
    (hy : ∀ v, v ∈ C.into S ∨ (v ∈ S ∧ v ∉ l) → y v = C.eval a v) :
    ∀ v ∈ S, l.foldl (fun y g => Function.update y g (C.op g y)) y v = C.eval a v := by
  induction l generalizing y with
  | nil => exact fun v hv => hy v (Or.inr ⟨hv, List.not_mem_nil⟩)
  | cons g l ih =>
    intro v hv
    rw [List.pairwise_cons] at hl
    have hgS : g ∈ S := hlS g List.mem_cons_self
    have hg : C.isInput g = false := by
      have := hS hgS
      simpa [Verity.Circuit.computed] using this
    have hstep : C.op g y = C.eval a g := by
      rw [C.eval_op a hg]
      apply C.op_congr
      intro r hr
      have hrg : r < g := C.reads_lt g r hr
      by_cases hrS : r ∈ S
      · refine hy r (Or.inr ⟨hrS, fun hmem => ?_⟩)
        rcases List.mem_cons.1 hmem with h | h
        · exact absurd h (ne_of_lt hrg)
        · exact absurd (hl.1 r h) (lt_asymm hrg)
      · refine hy r (Or.inl ?_)
        simp only [Verity.Circuit.into, Finset.mem_filter, Finset.mem_univ, true_and]
        exact ⟨hrS, g, hgS, hr⟩
    rw [List.foldl_cons]
    refine ih hl.2 (fun g' hg' => hlS g' (List.mem_cons_of_mem g hg')) _ (fun w hw => ?_) v hv
    by_cases hwg : w = g
    · subst hwg
      rw [Function.update_self]
      exact hstep
    · rw [Function.update_of_ne hwg]
      refine hy w ?_
      rcases hw with h | ⟨h1, h2⟩
      · exact Or.inl h
      · refine Or.inr ⟨h1, fun hmem => ?_⟩
        rcases List.mem_cons.1 hmem with h | h
        · exact hwg h
        · exact h2 h

/-- A registered value is an input. -/
theorem mem_inputs_of_mem_θ {d : RegisteredCircuit} {v : Fin d.D.N} (hθ : v ∈ d.θ) : v ∈ inputs d.D := by
  simp only [inputs, Finset.mem_filter, Finset.mem_univ, true_and]
  exact d.θ_input v hθ

/-- **Reconstruction**: for a set of gates `S` and any `x` that holds the values of `in(S)`, the reconstruction of `S`
from `x` evaluates `|S|` gates, one each, and holds every value `S` computes. -/
theorem reconstruction (C : Circuit) (S : Finset (Fin C.N)) (a x : C.Trace) (hS : GateSet C S)
    (hx : AgreesOn C x (C.eval a) (C.into S)) :
    (S.sort (· ≤ ·)).length = S.card ∧ AgreesOn C (reconstruct C S x) (C.eval a) S := by
  refine ⟨Finset.length_sort _, ?_⟩
  intro g hg
  exact foldl_update_eval S hS a _ (Finset.sortedLT_sort S).pairwise (fun g' hg' => (Finset.mem_sort _).1 hg') x
    (fun v hv => hv.elim (hx v) fun h => absurd ((Finset.mem_sort _).2 h.1) h.2) g hg

/-- **A block's reads split three ways**: for a set of gates `I`, `in(I)` is its registered values, its data and its
computed values, and `in(I) \ θ`, what `k` bounds, is the last two. -/
theorem reads_split (d : RegisteredCircuit) (I : Finset (Fin d.D.N)) :
    d.D.into I = registeredReads d I ∪ dataReads d I ∪ computedReads d I ∧
      d.unregistered I = dataReads d I ∪ computedReads d I := by
  constructor
  · ext v
    simp only [registeredReads, dataReads, computedReads, Finset.mem_union, Finset.mem_inter, Finset.mem_sdiff]
    tauto
  · ext v
    simp only [Verity.RegisteredCircuit.unregistered, dataReads, computedReads, Finset.mem_union, Finset.mem_inter,
      Finset.mem_sdiff]
    constructor
    · rintro ⟨hv, hθ⟩
      by_cases h : v ∈ inputs d.D
      · exact Or.inl ⟨hv, h, hθ⟩
      · exact Or.inr ⟨hv, h⟩
    · rintro (⟨hv, -, hθ⟩ | ⟨hv, h⟩)
      · exact ⟨hv, hθ⟩
      · exact ⟨hv, fun hθ => h (mem_inputs_of_mem_θ hθ)⟩

end Nci.Formalism
