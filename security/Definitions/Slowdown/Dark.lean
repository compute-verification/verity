module

public import Definitions.InferenceOnly

/-! Partial containment, a reference training algorithm computed partly off the books; whether a dark value enters `d` in θ is the premise `InferenceOnly.Assumptions.ThetaRegistered`, not the containment's. -/

@[expose] public section

namespace Slowdown

open Verity

/-- **A partial containment** of the reference training algorithm `c` in a declaration `d`: `d` computes the gates `S` of `c` on
the books, and the rest of `c`'s computed gates are dark. Through `emb`:
1. `S` holds only computed gates of `c`;
2. distinct gates of `S` and of `in(S)`, the gates outside `S` that `S` reads, go to distinct gates of `d`, of the same
   width;
3. each gate of `S` goes to a computed gate of `d` that reads exactly the images of the gates it reads;
4. each gate of `in(S)` goes to an input gate of `d`: a dark value enters `d` only as an input.

Whether such an input is in `d`'s θ is not part of the containment (`InferenceOnly.Assumptions.ThetaRegistered`). -/
structure PartialContainment (c d : RegisteredCircuit) where
  S : Finset (Fin c.D.N)
  /-- The gate of `d` each gate of `S` and of `in(S)` goes to. -/
  emb : Fin c.D.N → Fin d.D.N
  S_computed : S ⊆ c.D.computed
  emb_inj : ∀ a ∈ S ∪ c.D.into S, ∀ b ∈ S ∪ c.D.into S, emb a = emb b → a = b
  emb_w : ∀ a ∈ S ∪ c.D.into S, d.D.w (emb a) = c.D.w a
  computed_emb : ∀ g ∈ S, d.D.isInput (emb g) = false
  reads_emb : ∀ g ∈ S, d.D.reads (emb g) = (c.D.reads g).image emb
  input_emb : ∀ a ∈ c.D.into S, d.D.isInput (emb a) = true

/-- The dark gates: the computed gates of `c` that `d` does not compute. -/
def PartialContainment.dark {c d : RegisteredCircuit} (M : PartialContainment c d) : Finset (Fin c.D.N) :=
  c.D.computed \ M.S

/-- **A total containment is a partial one** with no dark gates: `S` is every computed gate of `c`, and what `S` reads
from outside is `c`'s inputs. -/
def PartialContainment.ofContainment {c d : RegisteredCircuit} (M : Containment c.D d.D) : PartialContainment c d where
  S := c.D.computed
  emb := M.emb
  S_computed := Finset.Subset.refl _
  emb_inj _ _ _ _ h := M.emb_inj h
  emb_w a _ := M.emb_w a
  computed_emb g hg := by
    rw [M.isInput_emb]
    simpa [Circuit.computed] using hg
  reads_emb g hg := by
    rw [M.reads_emb g (by simpa [Circuit.computed] using hg)]
    exact Finset.map_eq_image _ _
  input_emb a ha := by
    rw [M.isInput_emb]
    have h := (Finset.mem_filter.1 ha).2.1
    by_contra hne
    exact h (Finset.mem_filter.2 ⟨Finset.mem_univ _, by simpa using hne⟩)

end Slowdown
