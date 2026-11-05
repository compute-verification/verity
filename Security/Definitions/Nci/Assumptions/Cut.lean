module

public import Definitions.Nci.Cut

/-!
# The min-cut bound's cut conjectures, (H) and (E)

`Nci.Cut.Guarantees.Bound` holds for every circuit; `HE` and `HEShared` bound its `Π(k)` and `α` under two premises
on cuts, each a `Prop` about the circuit, its charged work and the parameters. Neither is decided by a check of the
circuit: each is a cut conjecture, refuted by one set of gates that beats it.

* **(H), hardness** (`Hardness`): the values a set of words carries cost cut gates. Every set `S` of words with
  `κ(S) ≤ k` has `∑_r β_r |val(S ∩ A_r)| ≤ κ(S)`: each value of array `A_r` that `S`'s words carry takes `β_r` of the
  gates separating them from the unregistered inputs. It fails for values a few gates determine: a weight barely moved
  since `θ`, a copy of another value under a name `val` doesn't identify, an activation recomputed from a narrower cut.
* **(E), exports** (`Exports`): the outputs a set of computed gates finishes cost cut gates to send on. Every set `B` of
  computed gates has `κ(B → E(B)) ≥ b |E(B)|`, `E(B)` the outputs fed by a MAC inside `B` with no wire in `B`. At
  `b = b_y` (`Nci.Cut.Work.bY`) it holds by definition; at a given `b` it says `b_y ≥ b`, which no check computes, and
  it fails for outputs a few gates determine, such as a low-rank product's.

Both count gates of the circuit and its values, never a value's width.
-/

@[expose] public section

namespace Nci.Cut.Assumptions

open Verity Finset

/-- `training-cut-hardness` (conjectured; refuted by a set `S` of words with `κ(S) ≤ k` whose values cost more than
`κ(S)`). **(H), hardness** at arrays `A`, costs `β` a value and read cap `k`: every set `S` of words of `d` with
`κ(S) ≤ k` has `∑_r β_r |val(S ∩ A r)| ≤ κ(S)`. -/
def Hardness {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (A : ρ → Finset (Finset (Fin d.D.N)))
    (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (k : ℕ) : Prop :=
  ∀ S, kappaWords d S ≤ k → ∑ r, β r * ((S ∩ A r).image val).card ≤ kappaWords d S

/-- `training-cut-exports` (conjectured; refuted by a set `B` of computed gates with `κ(B → E(B)) < b |E(B)|`).
**(E), exports** at `b`: every set `B` of computed gates has `κ(B → E(B)) ≥ b |E(B)|`. -/
def Exports {C : Circuit} {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) (b : ℝ) : Prop :=
  ∀ B ⊆ C.computed, b * (W.clear B).card ≤ kappa C B ((W.clear B).biUnion W.y)

end Nci.Cut.Assumptions
