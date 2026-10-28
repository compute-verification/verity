import Definitions.Pouw.Assumptions.Dimension.H100Floor

/-!
# A citation with no measurement at `tc` (trusted statements; draft semantics)

**Draft semantics, pending review**.

`h100Floor`'s `tc = 32` is this board's measurement of legacy `mma.sync`, half the `wgmma` rate per MAC, and m8n8k4 and
sparse `mma.sp` were never run. So `DistinctWritesFloor` covers legacy tensor-core forms only as measured.

`h100PeakTc` is `h100Floor` with `tc = 8`: m8n8k4's price per register at the `wgmma` per-MAC rate, the least of the
legacy shapes at those rates. It is also the dispatch bound for a legacy MMA that writes 4 accumulator
registers per thread (128 per warp, at one warp instruction per SMSP per clock). The other entries are `h100Floor`'s.

`floorTcPeakTc` (`Pouw.Dimension.H100PeakTcProofs`) proves `FloorTc 32 8 h100PeakTc` by `decide`, and
`DistinctWritesPeakTc` is `DistinctWritesLegacy` there: 32 per target that no cheap tensor-core instruction can write
(outside `K`), and 8 per target that one can. Taking `K` to be every function gives 8 per target with no hypothesis
at all. A smaller `K` needs a statement of what legacy `mma.sync` can compute, such as the k8 lemma's for
multiplicands that meet `FragLanes`. The bound is additive-W1 work, not time (`H32`'s docstring).
-/

namespace Pouw.Dimension.H100

/-- **The H100 table with legacy tensor-core forms at the peak rate** (draft semantics):
`h100Floor` with `tc = 8`, m8n8k4's price per register at the `wgmma` per-MAC rate. -/
def h100PeakTc : Prices100 := { h100Floor with tc := 8 }

/-! ## Pinned statements (draft semantics, pending review) -/

/-- **The distinct-writes bound with no measurement at `tc`** (draft semantics, pending review): at
`h100PeakTc`, if every tensor-core instruction below 32 per register writes only functions in `K`, then for targets
`T₁` and `T₂`, pairwise distinct on `S` within and across, none equal on `S` to a free value, and no target of `T₁`
equal on `S` to a function of `K`, a well-formed `MH100w` program on free inputs matching them all on `S` costs at
least `32·|T₁| + 8·|T₂|`. -/
def DistinctWritesPeakTc : Prop :=
  ∀ (G : Type) (F K : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G))
      (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100PeakTc F) → (∀ o ∈ P, o.CheapIn 32 K) →
      DistinctOn S F T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ¬ ∀ g ∈ S, t g = t' g) →
      (∀ t ∈ T₁, ∀ f ∈ K, ¬ ∀ g ∈ S, t g = f g) →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      32 * T₁.card + 8 * T₂.card ≤ cost100 h100PeakTc P

end Pouw.Dimension.H100
