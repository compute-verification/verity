import Definitions.Claims.Sampling

/-!
# The strata a one-stage design draws (`OneStage.strata`, `StrataDrawn`)

Every design but Bernoulli's draws a uniform subset of each of its strata `(n_s, k_s)`: the stratified and work laws'
own, and one stratum of every slot for `all` (drawing all of it) and `subset`. `OneStage.Pa` of such a design is
`strataPa` of these strata (`OneStage.Pa_strata`, in `Proofs.Flock.Claims.Sampling.Escape`), and the sampled
verifier's count bound reads the same strata (`SampledProofs.Bound.countBound`). `StrataDrawn d P ss` says a design's
classes' strata, in order, are `ss`: the one description of a design's strata that the count bound's soundness
(`countBound_miss_design`) and the network check's audit bound (`Properties.NetworkPeriod.WrongBoundsDrawn`) read.

It sits beside `Definitions.Claims.Sampling`, which the sampling claims' frozen record reads whole, rather than in it.
-/

namespace Claims.Sampling

/-- The strata `(n_s, k_s)` a one-stage design draws uniform subsets of, all but Bernoulli's: the stratified and work
laws' own, and one stratum of every slot for `all` (drawing all of it) and `subset`. -/
def OneStage.strata (ns : List ℕ) : OneStage → Option (List (ℕ × ℕ))
  | .all => some [(ns.sum, ns.sum)]
  | .subset K => some [(ns.sum, K)]
  | .bernoulli _ _ => none
  | .stratified K => some (ns.map fun n => (n, max 1 ((2 * K * n + ns.sum) / (2 * ns.sum))))
  | .work K w f =>
    let W := ((ns.zip w).map fun p => p.1 * p.2).sum
    some ((ns.zip (w.zip f)).map fun p => (p.1, min p.1 (max p.2.2 ((K * p.2.1 * p.1 + W - 1) / W))))

/-- **`ss` is the design's strata draw on `P`**: `d` is well formed on `P`, and its classes' strata (`OneStage.strata`
of each rule at its class's sizes), flattened in order, are `ss`. A design that is one is a strata draw over `P`'s
proof blocks (`Design.Pa_eq_strataPa`), whose count bound the sampled verifier publishes at `ss`'s sizes and draws. -/
def StrataDrawn (d : Design) (P : Partition) (ss : List (ℕ × ℕ)) : Prop :=
  d.WellFormed P ∧ ∃ sss : List (List (ℕ × ℕ)), sss.length = P.classes.length ∧
    (∀ p ∈ (d.rules.zip P.classes).zip sss, p.1.1.strata (strataSizes p.1.2) = some p.2) ∧ sss.flatten = ss

end Claims.Sampling
