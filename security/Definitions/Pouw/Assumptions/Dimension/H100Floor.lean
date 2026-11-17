module

public import Definitions.Pouw.Assumptions.Dimension.H100W
public import Definitions.Pouw.Assumptions.Dimension.H100LoopFree

/-! `MH100`'s floor table `h100Floor` and the distinct-writes bounds that cite the H100 (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

/-- **The H100 floor table** (draft semantics): a floor except at `tc`, which is one H100 SXM's measurement of legacy
`mma.sync`. -/
def h100Floor : Prices100 := { h100LoopFree with int32 := 32, other := 32 }

/-! ## Guarantees (draft semantics, pending review) -/

/-- **The distinct-writes bound at the loop-free prices** (draft semantics, pending review): the H100 citation for
programs of the measured forms. -/
def DistinctWritesLoopFree : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100LoopFree F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100LoopFree P

/-- **The distinct-writes bound at the floor table** (draft semantics, pending review): the H100 citation for every
scalar form and every `wgmma` shape, and for legacy tensor-core forms only as measured. -/
def DistinctWritesFloor : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100Floor F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100Floor P

end Pouw.Dimension.H100
