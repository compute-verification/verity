module

public import Definitions.Core.Levels

/-!
# The first level drawn by kind

One sampling design covers every block, and kinds of block are sampled at different rates: the first level's design
at the whole circuit is the product of one design per kind (`SamplingDesign.prod`), each kind drawn on coins of its
own (`Levels.FirstByKind`); weaker, the first level misses every set as that product does (`Levels.FirstMissesByKind`,
which `FirstByKind` gives: `Levels.FirstByKind.missesByKind`). Any levels may have their first level so drawn, keeping every level's blocks and every
later design (`Levels.exists_firstByKind`), and at one level the miss probability of a set is the product over kinds
of each kind's (`Levels.missProb_firstByKind_one`), both in `Proofs/Core/LevelsKinds.lean`.
-/

@[expose] public section

namespace Verity

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **The first level drawn by kind**: the first level's design at the whole circuit is the product of the designs
`D k`, one per kind `k`, each over its kind's units `V k`, the unit `u` of kind `k` read as the block `blockOf k u`. -/
def FirstByKind (L : Levels G) {K : Type} [Fintype K] [DecidableEq K] {V : K → Type}
    (blockOf : (k : K) → V k → Finset G) (D : (k : K) → SamplingDesign (V k)) : Prop :=
  ∃ l ls, L.levels = l :: ls ∧
    l.design Finset.univ = (SamplingDesign.prod D).map fun u : (k : K) × V k => blockOf u.1 u.2

/-- **The first level misses as if drawn by kind**: the first level's design at the whole circuit misses every set of
blocks with the probability the product of the designs `D k` does, its coins whatever they are. -/
def FirstMissesByKind (L : Levels G) {K : Type} [Fintype K] [DecidableEq K] {V : K → Type}
    (blockOf : (k : K) → V k → Finset G) (D : (k : K) → SamplingDesign (V k)) : Prop :=
  ∃ l ls, L.levels = l :: ls ∧ ∀ T : Finset (Finset G),
    (l.design Finset.univ).missProb T =
      ((SamplingDesign.prod D).map fun u : (k : K) × V k => blockOf u.1 u.2).missProb T

end Levels

end Verity
