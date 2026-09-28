import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# N10's two core lemmas, on the trusted definitions of `Pous.Invertible`

From `lean/submissions/pebbling/InvertibleExPostFactoDraft.lean`: the proofs of `transpose_mixing` and
`reverse_needs_all_children`, graded PASS against `Pous.Guarantees.TransposeMixing` and
`Pous.Guarantees.ReverseNeedsAllChildren`. The draft's copies of the trusted definitions and its N10 target (refuted,
`Pous.SecurityProofs.P3.Refutation`) are left out.

* `transpose_mixing`: with the `K_{n,n}` connector, recovering **one** lower-layer label needs **all** `n` top
  labels.
* `reverse_needs_all_children`: unfolding the trusted reverse rule, a reverse (decode) move on a lower node needs a
  pebble on every green child.
-/

namespace PousInvertible

open Pous Pous.Invertible

variable {n wc : ℕ}

/-- **Transpose mixing.** Fix a lower node `u`. Component `v` of `c¹_u` is a function of `c²_v` alone, and every `v`
contributes. So if the top labels are known only on a proper subset `S ⊊ univ`, there are two top configurations
agreeing on `S` whose `u`-th lower labels differ. -/
theorem transpose_mixing (P : Equiv.Perm (Lab n wc)) (k2 : Fin n → Lab n wc) (hwc : 0 < wc)
    (u : Fin n) (S : Finset (Fin n)) (hS : S ≠ Finset.univ) :
    ∃ c2 c2' : Fin n → Lab n wc, (∀ v ∈ S, c2 v = c2' v) ∧ decodeLower P c2 k2 u ≠ decodeLower P c2' k2 u := by
  obtain ⟨v₀, hv₀⟩ : ∃ v₀, v₀ ∉ S := by
    by_contra h
    push Not at h
    exact hS (Finset.eq_univ_of_forall h)
  set p0 : Fin wc := ⟨0, hwc⟩ with hp0
  -- the `u`-components of the two decoded inputs differ in bit `p0`
  let X0 : Bits wc := fun _ => false
  let X1 : Bits wc := Function.update X0 p0 true
  let base : Lab n wc := fun _ _ => false
  let a0 : Lab n wc := P (Function.update base u X0)
  let a1 : Lab n wc := P (Function.update base u X1)
  refine ⟨Function.update (fun _ => a0) v₀ a0, Function.update (fun _ => a0) v₀ a1, ?_, ?_⟩
  · intro v hv
    have hne : v ≠ v₀ := fun h => hv₀ (h ▸ hv)
    rw [Function.update_of_ne hne, Function.update_of_ne hne]
  · intro hcon
    have hval := congrArg (fun f => f v₀) hcon
    simp only [decodeLower, transpose, Function.update_self] at hval
    have h0 : P.symm a0 u = X0 := by simp [a0, Equiv.symm_apply_apply, Function.update_self]
    have h1 : P.symm a1 u = X1 := by simp [a1, Equiv.symm_apply_apply, Function.update_self]
    have hbit := congrArg (fun f => f p0) hval
    simp only [labXor, Labelling.bxor, h0, h1] at hbit
    rw [show X0 p0 = false from rfl, show X1 p0 = true from Function.update_self _ _ _] at hbit
    cases hk : k2 v₀ u p0 <;> simp [hk] at hbit

/-- **A reverse move needs all green children.** To place a pebble on a layer-0 node by a reverse (decode) move from
configuration `C`, every layer-1 node must already carry a pebble. -/
theorem reverse_needs_all_children (G : Labelling.TopoDAG n) (C : Finset (Fin 2 × Fin n)) (a : Fin n)
    (hrev : ¬ (∀ x, DRG.StackedEdge node2 (drg2 G) green2 x (node2 0 a) → x ∈ C))
    (havail : DRG.Available node2 (drg2 G) green2 C (node2 0 a)) :
    ∀ b : Fin n, node2 1 b ∈ C := by
  rcases havail with hfwd | ⟨_, hrev'⟩
  · exact absurd hfwd hrev
  · intro b
    exact (hrev' (node2 1 b) ⟨rfl, rfl⟩).1

end PousInvertible
