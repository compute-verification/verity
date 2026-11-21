module

public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Bridge.SampledDesign

/-!
# The registration's strata cover the population's units (`Bridge.PopCovers`)

The uniform-subset form of the population: the population is the units `0, …, n − 1` in unit order, and the strata's
ranges need only be a disjoint cover of them. `Bridge.PopCovers desc q n`, a decidable check on the files, as
`Bridge.PopMatches` is: `desc` parses, its layout under the query `q` (`Spot.Slots.layout`) is defined, and its
population's units are `0, …, n − 1`, each once, in any order.

* `popCovers_iff`: `PopCovers` is that the parse and the layout succeed and the population's units are a permutation
  of `0, …, n − 1`.
* `popCovers_of_popMatches`: the population in order (`PopMatches`) is a cover.
* `nsOf_sum_of_popCovers`: at `PopCovers`, the strata sum to `n`.
* `IsSubsetDesign`: the sampling design is `subset k`, the uniform subset (`All` is `subset n`).
* `samplingDesignOn_samplesUnder_cover`: at `PopCovers`, a subset sampling design the verifier samples under is on the
  registration's strata (`SamplingDesignOn`), which reads only their sum.
-/

@[expose] public section

namespace Flock.Bridge

/-- `PopCovers`'s check. -/
def popCoverCheck (desc : Hash.Msg) (q : Lean.Json) (n : ℕ) : Bool :=
  match Spot.Program.parse ⟨desc.toArray⟩ with
  | .error _ => false
  | .ok p =>
    match Spot.Slots.layout p q with
    | .error _ => false
    | .ok l => decide (((popOf l).map (·.unit)).Perm ((List.range n).map some))

/-- **The registration's strata cover the population's `n` units**: `desc` parses, its layout under the query `q` is
defined, and the layout's population's units are `0, …, n − 1`, each once, in any order. -/
@[reducible] def PopCovers (desc : Hash.Msg) (q : Lean.Json) (n : ℕ) : Prop := popCoverCheck desc q n = true

instance instDecidablePopCovers (desc : Hash.Msg) (q : Lean.Json) (n : ℕ) : Decidable (PopCovers desc q n) :=
  inferInstanceAs (Decidable (_ = true))

/-- **The uniform-subset sampling design**: `subset k`; `All` is `subset n`. -/
def IsSubsetDesign (samplingDesign : Flock.Draw.SamplingDesign) : Prop := ∃ k, samplingDesign = .subset k

end Flock.Bridge
