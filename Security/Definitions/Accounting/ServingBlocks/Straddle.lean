import Definitions.Accounting.ServingBlocks

/-!
# Operations across the certified boundary (trusted)

The uncertified cap `q_U(U) = ω_max |U| / s_×` counts the operations of a set `U` of uncertified gates at `s_×` of its
gates each. For a set of gates `R` and the blocks `B i` counted for it, `R`'s **uncertified gates** `U` are those in
none of the blocks (`uncertified R B`). An operation assigned to `R` and computed wholly inside none of the blocks
**straddles** when its body is not inside `U` (`Counted.straddle`): it has a gate in a block or outside `R`. That is what
the cap misses: such an operation may have fewer than `s_×` of its gates in `U`, the rest computed by a block (a
certified product an uncertified accumulation reads, say) or elsewhere.

The named premises:
* `BodySize s`: every counted operation's body has at least `s` gates (`s_×`, the fewest gates in any counted
  operation's subcircuit).
* `NoStraddle`: no operation assigned to `R` and computed wholly inside none of the blocks straddles. With `BodySize`
  and `Overlap` it gives the cap `ω_max μ |U| / s` (`Accounting.Guarantees.ServingBlocks.UncertifiedStructural`).
* `UncertifiedBodySize s`: every counted operation whose representative is in `U` has at least `s` of its gates in `U`.
  It allows straddling, and gives the same cap plus what the blocks' unregistered inputs bound (`UncertifiedLocal`).

`crossing d R B` counts the unregistered input bits of `U`, `|in(U) \ θ|`, and of each block, `Σ_i |in(B i) \ θ|`: the
bits that cross into either side of the certified boundary, or into `R` from outside it. They bound the straddling
operations with no premise on them (`StraddleBound`), as `|in(S) \ θ|` bounds the operations a serving block `S` is
assigned but doesn't compute wholly (`Accounting.SecurityProofs.ServingBlocks.boundary_le`).
-/

namespace Accounting.ServingBlocks

open Verity Finset

/-- `R`'s **uncertified gates**: the gates of `R` in none of the blocks `B i`. -/
def uncertified {C : Circuit} (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) :
    Finset (Fin C.N) :=
  R \ univ.biUnion B

namespace Counted

variable {C : Circuit} (J : Counted C)

/-- **The straddling weight**: the weight of the counted operations assigned to `R` and computed wholly inside none of the
blocks `B i` whose bodies are not inside `R`'s uncertified gates. -/
def straddle (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin C.N)) : ℝ :=
  ∑ v ∈ J.V with v ∈ R ∧ (∀ i, ¬ J.body v ⊆ B i) ∧ ¬ J.body v ⊆ uncertified R B, J.ω v

end Counted

/-- **Body size** `s`: every counted operation's body has at least `s` gates. -/
def BodySize {C : Circuit} (J : Counted C) (s : ℕ) : Prop :=
  ∀ v ∈ J.V, s ≤ (J.body v).card

/-- **No straddling** at `R` with the blocks `B`: every counted operation assigned to `R` and computed wholly inside none
of the blocks is computed wholly inside `R`'s uncertified gates. -/
def NoStraddle {C : Circuit} (J : Counted C) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) : Prop :=
  ∀ v ∈ J.V, v ∈ R → (∀ i, ¬ J.body v ⊆ B i) → J.body v ⊆ uncertified R B

/-- **Uncertified body size** `s` at `R` with the blocks `B`: every counted operation whose representative is one of
`R`'s uncertified gates has at least `s` of its gates among them. -/
def UncertifiedBodySize {C : Circuit} (J : Counted C) (R : Finset (Fin C.N)) {ι : Type*} [Fintype ι]
    (B : ι → Finset (Fin C.N)) (s : ℕ) : Prop :=
  ∀ v ∈ J.V, v ∈ uncertified R B → s ≤ (J.body v ∩ uncertified R B).card

/-- **The bits crossing the certified boundary** at `R` with the blocks `B`: the unregistered input bits of `R`'s
uncertified gates, `|in(U) \ θ|`, and of each block, `Σ_i |in(B i) \ θ|`. -/
def crossing (d : Declaration) (R : Finset (Fin d.D.N)) {ι : Type*} [Fintype ι] (B : ι → Finset (Fin d.D.N)) : ℕ :=
  (d.unregistered (uncertified R B)).card + ∑ i, (d.unregistered (B i)).card

end Accounting.ServingBlocks
