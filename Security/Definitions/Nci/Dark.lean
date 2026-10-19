import Definitions.Nci

/-!
# Partial containment: a reference circuit computed partly off the books

`Declaration.Contains` asks the declared circuit `d` to compute every gate of the reference circuit `c`. A prover can
instead compute some of `c`'s gates outside `d`, off the books, and feed their values into `d` as inputs. A partial
containment (`PartialContainment c d`) says what `d` still computes: a set `S` of `c`'s computed gates, each with its own
gate of `d` reading the images of the gates it reads. The rest of `c`'s computed gates are its dark gates (`dark`): `d`
holds none of them, and a dark value `S` reads enters `d` as an input gate outside `d`'s baseline. `d` need not compute
any gate the way `c` does, only from the same wires, so a total containment is the special case `S = computed`.
-/

namespace Nci

open Verity

/-- **A partial containment** of the reference circuit `c` in a declaration `d`: `d` computes the gates `S` of `c` on
the books, and the rest of `c`'s computed gates are dark. Through `emb`:
1. `S` holds only computed gates of `c`;
2. distinct gates of `S` and of `in(S)`, the gates outside `S` that `S` reads, go to distinct gates of `d`, of the same
   width;
3. each gate of `S` goes to a computed gate of `d` that reads exactly the images of the gates it reads;
4. each gate of `in(S)` goes to an input gate of `d`: a dark value enters `d` only as an input;
5. a gate of `in(S)` goes into `d`'s baseline only when it is in `c`'s baseline: a dark value, or any novel input of
   `c`, is never registered as baseline. -/
structure PartialContainment (c d : Declaration) where
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
  θ_emb : ∀ a ∈ c.D.into S, emb a ∈ d.θ → a ∈ c.θ

/-- The dark gates: the computed gates of `c` that `d` does not compute. -/
def PartialContainment.dark {c d : Declaration} (M : PartialContainment c d) : Finset (Fin c.D.N) :=
  c.D.computed \ M.S

end Nci
