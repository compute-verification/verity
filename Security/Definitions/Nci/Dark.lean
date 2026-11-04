import Definitions.Nci

/-!
# Partial containment: a reference circuit computed partly off the books

`RegisteredCircuit.Contains` asks the declared circuit `d` to compute every gate of the reference circuit `c`. A prover can
instead compute some of `c`'s gates outside `d`, off the books, and feed their values into `d` as inputs. A partial
containment (`PartialContainment c d`) says what `d` still computes: a set `S` of `c`'s computed gates, each with its own
gate of `d` reading the images of the gates it reads. The rest of `c`'s computed gates are its dark gates (`dark`): `d`
holds none of them, and a dark value `S` reads enters `d` as an input gate. `d` need not compute any gate the way `c`
does, only from the same wires, so a total containment is the special case `S = computed`
(`PartialContainment.ofContainment`).

A partial containment says nothing about `d`'s θ: a dark value may enter `d` as an input in θ, read for free.
That it doesn't is a premise about the protocol, which decides what `θ` may hold, and not about the circuit:
`Nci.Assumptions.ThetaRegistered` (`Definitions/Nci/Assumptions/Baseline.lean`), which every guarantee through a
containment, total or partial, takes as a hypothesis.
-/

namespace Nci

open Verity

/-- **A partial containment** of the reference circuit `c` in a declaration `d`: `d` computes the gates `S` of `c` on
the books, and the rest of `c`'s computed gates are dark. Through `emb`:
1. `S` holds only computed gates of `c`;
2. distinct gates of `S` and of `in(S)`, the gates outside `S` that `S` reads, go to distinct gates of `d`, of the same
   width;
3. each gate of `S` goes to a computed gate of `d` that reads exactly the images of the gates it reads;
4. each gate of `in(S)` goes to an input gate of `d`: a dark value enters `d` only as an input.

Whether such an input is in `d`'s θ is not part of the containment (`Nci.Assumptions.ThetaRegistered`). -/
structure PartialContainment (c d : RegisteredCircuit) where
  /-- The gates of `c` that `d` computes. -/
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

end Nci
